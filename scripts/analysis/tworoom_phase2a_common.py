import csv
import json
import math
import os
import subprocess
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf, open_dict

from tworoom_common import DEFAULT_CHECKPOINT, batch_to_device, compose_tworoom_cfg, load_model
from tworoom_failure_c_v1_common import (
    OUT as C_OUT,
    CLASS_LABELS as C_CLASS_LABELS,
    load_head as load_c_head,
    make_feature as make_c_feature,
)
from tworoom_v4_common import BEST_H_DATASET, table_md
from utils import get_column_normalizer, get_img_preprocessor


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs" / "tworoom_rule_residual_adapter_phase2a"
REPORT = REPO_ROOT / "reports" / "tworoom_rule_residual_adapter_phase2a"
TABLE = OUT / "tables"
MODEL_DIR = OUT / "models"
FIG = OUT / "figures"
PHASE1_DATASET = C_OUT / "aligned_failure_dataset.pt"
C_MODEL_PATH = C_OUT / "models" / "h_q_action__small_mlp_128__seed2.pt"
ADAPTER_DATASET = OUT / "adapter_dataset.pt"
PRED_CACHE = OUT / "official_pred_target_cache.pt"
V42_GROUP = [10, 13, 12, 1]
PRIMARY_LABEL = "official_window_mse_top10"
DIRECT_LABEL = "direct_mse_top10"

INPUT_GROUPS = [
    "h_action",
    "single_group_action",
    "full_q_action",
    "q_only_action",
    "shuffled_q_action",
    "z_action",
    "group_only_action",
]
MODEL_TYPES = ["linear_adapter", "small_mlp_64", "small_mlp_128"]
GATES = ["c_soft_gate", "c_sigmoid_gate", "always_on", "random_gate", "oracle_gate", "group_soft_gate", "c_times_group_gate"]


def ensure_dirs():
    for p in [OUT, REPORT, TABLE, MODEL_DIR, FIG]:
        p.mkdir(parents=True, exist_ok=True)


def git_text(args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_report(name, text):
    ensure_dirs()
    path = REPORT / name
    path.write_text(text)
    print(text[:8000])
    return path


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def load_phase1():
    return torch.load(PHASE1_DATASET, map_location="cpu")


def load_official_dataset_with_metadata(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    for key in ("pos_agent", "ep_idx", "step_idx"):
        if key not in dataset_cfg["keys_to_load"]:
            dataset_cfg["keys_to_load"].append(key)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR", "/data/lzt26/stable-wm"),
        **dataset_cfg,
    )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def compute_pred_target_cache(batch_size=256, device=None):
    ensure_dirs()
    if PRED_CACHE.exists():
        return torch.load(PRED_CACHE, map_location="cpu"), False
    phase1 = load_phase1()
    base_rows = phase1["source_base_idx"].long()
    base = torch.load(BEST_H_DATASET, map_location="cpu")
    row_to_local = {int(r): i for i, r in enumerate(base_rows.tolist())}
    key_to_local = {
        (int(base["episode_id"][r]), int(base["timestep"][r])): i for i, r in enumerate(base_rows.tolist())
    }
    cfg = compose_tworoom_cfg(batch_size=batch_size, num_workers=0)
    dataset = load_official_dataset_with_metadata(cfg)
    model, missing, unexpected = load_model(cfg, DEFAULT_CHECKPOINT)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    _, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    n = base_rows.numel()
    base_pred = torch.full((n, int(cfg.embed_dim)), float("nan"))
    target = torch.full((n, int(cfg.embed_dim)), float("nan"))
    window_mse = torch.full((n,), float("nan"))
    joined, unmatched = 0, 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch_gpu = batch_to_device(dict(batch), device)
            batch_gpu["action"] = torch.nan_to_num(batch_gpu["action"], 0.0)
            out = model.encode(batch_gpu)
            emb = out["emb"]
            act_emb = out["act_emb"]
            pred = model.predict(emb[:, : cfg.history_size], act_emb[:, : cfg.history_size])
            tgt = emb[:, cfg.num_preds :]
            ep = batch["ep_idx"][:, cfg.history_size - 1].long().cpu()
            ts = batch["step_idx"][:, cfg.history_size - 1].long().cpu()
            for j, (e, t) in enumerate(zip(ep.tolist(), ts.tolist())):
                loc = key_to_local.get((int(e), int(t)))
                if loc is None:
                    unmatched += 1
                    continue
                base_pred[loc] = pred[j, -1].detach().cpu()
                target[loc] = tgt[j, -1].detach().cpu()
                window_mse[loc] = (pred[j] - tgt[j]).pow(2).mean().detach().cpu()
                joined += 1
            if bi % 100 == 0:
                print(f"pred cache batch {bi}/{len(loader)} joined={joined} unmatched={unmatched}", flush=True)
    valid = torch.isfinite(base_pred).all(1) & torch.isfinite(target).all(1)
    cache = {
        "base_pred_z": base_pred,
        "target_z": target,
        "valid_pred_target": valid,
        "official_window_mse_recomputed": window_mse,
        "metadata": {
            "checkpoint": str(DEFAULT_CHECKPOINT),
            "joined": int(valid.sum().item()),
            "unmatched": int(unmatched),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "git_commit": git_text(["rev-parse", "HEAD"]),
        },
    }
    torch.save(cache, PRED_CACHE)
    return cache, True


def group_activation(q, mode="mean"):
    g = q[:, V42_GROUP].float()
    if mode == "sum":
        return g.sum(1, keepdim=True)
    if mode == "topk_avg":
        return torch.topk(g, k=2, dim=1).values.mean(1, keepdim=True)
    return g.mean(1, keepdim=True)


def normalize01_from_train(x, train_idx):
    lo = torch.quantile(x[train_idx].float().reshape(-1), 0.01)
    hi = torch.quantile(x[train_idx].float().reshape(-1), 0.99)
    return ((x.float() - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)


def c_risks_for_phase1(phase1):
    model, obj = load_c_head(C_MODEL_PATH)
    x, _ = make_c_feature(phase1, obj["feature_group"], seed=int(obj["seed"]))
    x_std = (x.float() - obj["x_mean"]) / obj["x_std"].clamp_min(1e-6)
    from tworoom_failure_c_v1_common import predict_model

    pred = predict_model(model, x_std)
    probs = torch.sigmoid(pred[:, : len(C_CLASS_LABELS)])
    return {
        "c_official_window_top10": probs[:, C_CLASS_LABELS.index("official_window_mse_top10")].float(),
        "c_direct_mse_top10": probs[:, C_CLASS_LABELS.index("direct_mse_top10")].float(),
    }


def make_adapter_dataset():
    ensure_dirs()
    if ADAPTER_DATASET.exists():
        return torch.load(ADAPTER_DATASET, map_location="cpu")
    phase1 = load_phase1()
    pred_cache, recomputed = compute_pred_target_cache()
    valid = pred_cache["valid_pred_target"]
    idx = torch.nonzero(valid).flatten()
    data = {}
    for key in ["h", "q", "action", "z", "position", "episode_id", "timestep", "direct_mse", "direct_cosine_error", "official_window_mse"]:
        data[key] = phase1[key][idx]
    data["source_base_idx"] = phase1["source_base_idx"][idx]
    data["base_pred_z"] = pred_cache["base_pred_z"][idx].float()
    data["target_z"] = pred_cache["target_z"][idx].float()
    data["target_delta_z"] = data["target_z"] - data["base_pred_z"]
    data["labels"] = {k: v[idx].float() for k, v in phase1["labels"].items() if torch.is_tensor(v)}
    data["thresholds"] = phase1["thresholds"]
    data["split"] = {}
    old_to_new = torch.full((phase1["h"].shape[0],), -1, dtype=torch.long)
    old_to_new[idx] = torch.arange(idx.numel())
    for split_key in ["head_train_idx", "head_val_idx", "head_test_idx"]:
        mapped = old_to_new[phase1["split"][split_key].long()]
        data["split"][split_key] = mapped[mapped >= 0]
    risks = c_risks_for_phase1(phase1)
    data.update({k: v[idx].float() for k, v in risks.items()})
    data["group_10_13_12_1"] = group_activation(data["q"], "mean").reshape(-1)
    data["metadata"] = {
        "phase1_dataset": str(PHASE1_DATASET),
        "c_model_path": str(C_MODEL_PATH),
        "pred_cache": str(PRED_CACHE),
        "official_predictor_recomputed": recomputed,
        "sigreg_used": False,
        "git_commit": git_text(["rev-parse", "HEAD"]),
    }
    torch.save(data, ADAPTER_DATASET)
    return data


def load_adapter_dataset():
    return torch.load(ADAPTER_DATASET, map_location="cpu")


def build_gates(data):
    tr, va = data["split"]["head_train_idx"], data["split"]["head_val_idx"]
    c = data["c_official_window_top10"].float()
    group = normalize01_from_train(data["group_10_13_12_1"].reshape(-1), tr)
    gates = {
        "always_on": torch.ones_like(c),
        "c_soft_gate": c.clamp(0, 1),
        "group_soft_gate": group,
        "c_times_group_gate": (c.clamp(0, 1) * group).clamp(0, 1),
        "oracle_gate": data["labels"][PRIMARY_LABEL].float(),
    }
    y = data["labels"][PRIMARY_LABEL][va].float()
    best = None
    for beta in [5.0, 10.0]:
        for tau in torch.quantile(c[tr], torch.tensor([0.50, 0.70, 0.80, 0.90])).tolist():
            g = torch.sigmoid(beta * (c - float(tau)))
            # Gate selection prefers higher gate on positives but penalizes firing on everything.
            score = g[va][y > 0.5].mean().item() - 0.3 * g[va][y < 0.5].mean().item()
            if best is None or score > best[0]:
                best = (score, beta, float(tau), g)
    gates["c_sigmoid_gate"] = best[3].float()
    gen = torch.Generator().manual_seed(4202)
    gates["random_gate"] = gates["c_soft_gate"][torch.randperm(c.numel(), generator=gen)]
    meta = {"c_sigmoid_beta": best[1], "c_sigmoid_tau": best[2]}
    return gates, meta


def standardize_train(x, train_idx):
    mean = x[train_idx].float().mean(0, keepdim=True)
    std = x[train_idx].float().std(0, keepdim=True).clamp_min(1e-6)
    return (x.float() - mean) / std, mean.squeeze(0), std.squeeze(0)


def adapter_input(data, input_group, seed=0, ablation=None):
    h, q, a = data["h"].float(), data["q"].float(), data["action"].float()
    z = data["z"].float()
    gen = torch.Generator().manual_seed(9100 + seed)
    q_use = q.clone()
    if input_group == "shuffled_q_action" or ablation == "shuffled_q":
        q_use = q_use[torch.randperm(q_use.shape[0], generator=gen)]
    if ablation == "mask_full_q":
        q_use = torch.zeros_like(q_use)
    if ablation == "mask_group":
        q_use = q_use.clone()
        q_use[:, V42_GROUP] = 0.0
    if ablation == "clamp_group_high":
        q_use = q_use.clone()
        q_use[:, V42_GROUP] = q_use[:, V42_GROUP].max()
    if ablation == "clamp_group_low":
        q_use = q_use.clone()
        q_use[:, V42_GROUP] = q_use[:, V42_GROUP].min()
    g = group_activation(q_use, "mean")
    if input_group == "h_action":
        return torch.cat([h, a], 1)
    if input_group == "single_group_action":
        return torch.cat([h, g, a], 1)
    if input_group == "full_q_action":
        return torch.cat([h, q_use, a], 1)
    if input_group == "q_only_action":
        return torch.cat([q_use, a], 1)
    if input_group == "shuffled_q_action":
        return torch.cat([h, q_use, a], 1)
    if input_group == "z_action":
        return torch.cat([z, a], 1)
    if input_group == "group_only_action":
        return torch.cat([g, a], 1)
    raise ValueError(input_group)


class ResidualAdapter(nn.Module):
    def __init__(self, in_dim, model_type="small_mlp_128", out_dim=192):
        super().__init__()
        if model_type == "linear_adapter":
            self.net = nn.Linear(in_dim, out_dim)
        elif model_type == "small_mlp_64":
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, out_dim))
        elif model_type == "small_mlp_128":
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, out_dim))
        else:
            raise ValueError(model_type)

    def forward(self, x):
        return self.net(x)


def train_adapter(data, input_group, model_type, gate_name, loss_mode="hard_weighted", seed=0, max_epochs=100, patience=10, batch_size=1024, lr=1e-3, device="cuda"):
    torch.manual_seed(seed)
    gates, gate_meta = build_gates(data)
    x = adapter_input(data, input_group, seed=seed)
    tr, va = data["split"]["head_train_idx"], data["split"]["head_val_idx"]
    x_std, mean, std = standardize_train(x, tr)
    gate = gates[gate_name].reshape(-1, 1).float()
    delta = data["target_delta_z"].float()
    label = data["labels"][PRIMARY_LABEL].float()
    weights = torch.ones_like(label)
    if loss_mode == "hard_weighted":
        weights = weights + 2.0 * label
    elif loss_mode == "risk_weighted":
        weights = weights + 2.0 * data["c_official_window_top10"].float()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ResidualAdapter(x_std.shape[1], model_type, out_dim=delta.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ds = TensorDataset(x_std[tr].float(), delta[tr].float(), gate[tr].float(), weights[tr].float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    best_state, best_val, bad = None, float("inf"), 0
    history = []
    hard_va = data["labels"][PRIMARY_LABEL][va] > 0.5
    val_eval_idx = va[hard_va] if hard_va.any() else va
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, db, gb, wb in loader:
            xb, db, gb, wb = xb.to(device), db.to(device), gb.to(device), wb.to(device)
            corr = model(xb)
            per = ((gb * corr - db) ** 2).mean(1)
            loss = (per * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred_corr = predict_adapter(model, x_std[val_eval_idx], device=device)
            final = data["base_pred_z"][val_eval_idx] + gate[val_eval_idx] * pred_corr
            val_mse = ((final - data["target_z"][val_eval_idx]) ** 2).mean().item()
        model.to(device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_hard_mse": val_mse})
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    obj = {
        "state_dict": model.cpu().state_dict(),
        "input_group": input_group,
        "model_type": model_type,
        "gate_name": gate_name,
        "loss_mode": loss_mode,
        "seed": seed,
        "x_mean": mean,
        "x_std": std,
        "history": history,
        "best_val_hard_mse": best_val,
        "gate_meta": gate_meta,
    }
    return obj


def load_adapter(path):
    obj = torch.load(path, map_location="cpu")
    model = ResidualAdapter(int(obj["x_mean"].numel()), obj["model_type"])
    model.load_state_dict(obj["state_dict"])
    return model, obj


@torch.no_grad()
def predict_adapter(model, x, batch_size=8192, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outs = []
    for i in range(0, x.shape[0], batch_size):
        outs.append(model(x[i : i + batch_size].float().to(device)).cpu())
    model.cpu()
    return torch.cat(outs, 0)


def evaluate_adapter_obj(data, model, obj, eval_gate_name=None, ablation=None):
    gates, _ = build_gates(data)
    gate_name = eval_gate_name or obj["gate_name"]
    gate = gates[gate_name].reshape(-1, 1).float()
    if ablation == "always_on_gate":
        gate = gates["always_on"].reshape(-1, 1).float()
    elif ablation == "random_gate":
        gate = gates["random_gate"].reshape(-1, 1).float()
    elif ablation == "oracle_gate":
        gate = gates["oracle_gate"].reshape(-1, 1).float()
    x = adapter_input(data, obj["input_group"], seed=int(obj["seed"]), ablation=ablation)
    x_std = (x.float() - obj["x_mean"]) / obj["x_std"].clamp_min(1e-6)
    corr = predict_adapter(model, x_std)
    final = data["base_pred_z"] + gate * corr
    return final, corr, gate.reshape(-1)


def subset_masks(data):
    test = data["split"]["head_test_idx"]
    c = data["c_official_window_top10"].float()
    group = data["group_10_13_12_1"].float()
    masks = {"all_test": torch.zeros(data["h"].shape[0], dtype=torch.bool)}
    masks["all_test"][test] = True
    for label in ["official_window_mse_top20", "official_window_mse_top10", "official_window_mse_top5", "direct_mse_top20", "direct_mse_top10", "direct_mse_top5", "direct_cosine_error_top10"]:
        masks[label] = masks["all_test"] & (data["labels"][label] > 0.5)
    for name, score, frac in [
        ("C_high_risk_top10", c, 0.9),
        ("C_high_risk_top20", c, 0.8),
        ("V4.2_group_active_top10", group, 0.9),
        ("V4.2_group_active_top20", group, 0.8),
    ]:
        th = torch.quantile(score[test], frac)
        masks[name] = masks["all_test"] & (score >= th)
    masks["rule_active_and_high_risk"] = masks["V4.2_group_active_top20"] & masks["C_high_risk_top20"]
    masks["rule_inactive_low_risk"] = masks["all_test"] & (group < torch.quantile(group[test], 0.5)) & (c < torch.quantile(c[test], 0.5))
    masks["false_positive_C_high_risk_but_label0"] = masks["C_high_risk_top10"] & (data["labels"][PRIMARY_LABEL] < 0.5)
    masks["false_negative_C_low_risk_but_label1"] = masks["all_test"] & (c < torch.quantile(c[test], 0.5)) & (data["labels"][PRIMARY_LABEL] > 0.5)
    return masks


def metrics_for_final(data, final, corr, gate, model_name, subset_name, mask):
    if mask.sum() == 0:
        return None
    base = data["base_pred_z"][mask]
    tgt = data["target_z"][mask]
    fin = final[mask]
    base_mse = ((base - tgt) ** 2).mean().item()
    final_mse = ((fin - tgt) ** 2).mean().item()
    base_cos = F.cosine_similarity(base, tgt, dim=1).mean().item()
    final_cos = F.cosine_similarity(fin, tgt, dim=1).mean().item()
    return {
        "model": model_name,
        "subset": subset_name,
        "n": int(mask.sum().item()),
        "base_mse": base_mse,
        "final_mse": final_mse,
        "mse_delta": final_mse - base_mse,
        "relative_improvement": (base_mse - final_mse) / max(base_mse, 1e-12),
        "base_cosine": base_cos,
        "final_cosine": final_cos,
        "cosine_improvement": final_cos - base_cos,
        "correction_norm": corr[mask].norm(dim=1).mean().item(),
        "gate_mean": gate[mask].mean().item(),
        "gate_active_rate": (gate[mask] > 0.5).float().mean().item(),
    }


def load_images(data, local_rows, image_size=64):
    import hdf5plugin  # noqa: F401
    import h5py

    local_rows = torch.as_tensor(local_rows, dtype=torch.long)
    with h5py.File(Path("/data/lzt26/stable-wm/datasets/tworoom.h5"), "r") as f:
        offsets = f["ep_offset"][:]
        ep = data["episode_id"][local_rows].numpy()
        ts = data["timestep"][local_rows].numpy()
        h5_idx = offsets[ep] + ts
        order = np.argsort(h5_idx)
        sorted_idx = h5_idx[order]
        restored = np.empty_like(order)
        restored[order] = np.arange(len(order))
        chunks = []
        for start in range(0, len(sorted_idx), 256):
            raw = f["pixels"][sorted_idx[start : start + 256]]
            x = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0
            small = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
            chunks.append(small.permute(0, 2, 3, 1).numpy())
        return np.concatenate(chunks, 0)[restored]


def save_image_grid(path, images, titles, ncols=4):
    if len(images) == 0:
        return
    ncols = min(ncols, len(images))
    nrows = int(math.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
