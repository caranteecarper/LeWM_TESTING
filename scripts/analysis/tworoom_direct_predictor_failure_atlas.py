import argparse
import csv
import math
import os
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from tworoom_common import DEFAULT_CHECKPOINT, batch_to_device, compose_tworoom_cfg, load_model, load_tworoom_dataset
from tworoom_preintegration_common import LATENTS_PATH
from tworoom_v4_common import BEST_H_DATASET, table_md
from tworoom_v42_common import compute_v4_q, compute_v41_q, compute_v42_q
from tworoom_v42_full_common import V42_GROUPS
from tworoom_v5_common import H5_PATH, REPO_ROOT, VISUAL_DATASET, git_text, load_visual
from utils import get_column_normalizer, get_img_preprocessor


OUT = REPO_ROOT / "outputs" / "tworoom_direct_predictor_failure_atlas"
REPORT = REPO_ROOT / "reports" / "tworoom_direct_predictor_failure_atlas"
TABLE = OUT / "tables"
FIG = OUT / "figures"

V42_GROUP0 = [10, 13, 12, 1]
V42_BEST_RULE = 10
V4_GROUP0 = [15, 2, 8, 11]


def ensure_dirs():
    for path in [OUT, REPORT, TABLE, FIG]:
        path.mkdir(parents=True, exist_ok=True)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(name, text):
    ensure_dirs()
    path = REPORT / name
    path.write_text(text)
    print(text[:8000])
    return path


def nanmean(x):
    x = x.float()
    m = torch.isfinite(x)
    if not m.any():
        return float("nan")
    return x[m].mean().item()


def finite_quantile(x, q):
    x = x.float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    return torch.quantile(x, q).item()


def build_latent_lookup(latents):
    ep = latents["episode_id"].long()
    ts = latents["timestep"].long()
    return {(int(e), int(t)): i for i, (e, t) in enumerate(zip(ep.tolist(), ts.tolist()))}


def action_normalization_stats(dataset):
    col_data = dataset.get_col_data("action")
    data = torch.from_numpy(np.array(col_data)).float()
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clamp_min(1e-6).clone()
    return mean, std


def normalize_action_like_training(raw_action, mean, std):
    raw_action = raw_action.float()
    action_dim = int(mean.numel())
    if raw_action.shape[-1] == action_dim:
        return (raw_action - mean) / std
    if raw_action.shape[-1] % action_dim != 0:
        raise ValueError(f"Cannot apply action normalizer of dim {action_dim} to action shape {tuple(raw_action.shape)}")
    flat_shape = raw_action.shape
    grouped = raw_action.reshape(*flat_shape[:-1], flat_shape[-1] // action_dim, action_dim)
    grouped = (grouped - mean.reshape(*([1] * (grouped.ndim - 1)), action_dim)) / std.reshape(
        *([1] * (grouped.ndim - 1)), action_dim
    )
    return grouped.reshape(flat_shape)


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
    # Match train.py exactly for the original training keys. Extra metadata keys are deliberately not normalized.
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def build_base_lookup(base):
    return {
        (int(e), int(t)): i
        for i, (e, t) in enumerate(zip(base["episode_id"].long().tolist(), base["timestep"].long().tolist()))
    }


def compute_direct_errors_from_official_val(checkpoint, batch_size=256, device=None):
    cfg = compose_tworoom_cfg(batch_size=batch_size, num_workers=0)
    dataset = load_official_dataset_with_metadata(cfg)
    model, missing, unexpected = load_model(cfg, checkpoint)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    _, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)
    loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    base = torch.load(BEST_H_DATASET, map_location="cpu")
    base_lookup = build_base_lookup(base)
    n = base["episode_id"].numel()
    last_mse = torch.full((n,), float("nan"))
    last_l2 = torch.full((n,), float("nan"))
    last_cosine = torch.full((n,), float("nan"))
    last_cosine_error = torch.full((n,), float("nan"))
    window_mse = torch.full((n,), float("nan"))
    pred_norm = torch.full((n,), float("nan"))
    target_norm = torch.full((n,), float("nan"))
    eval_base_rows = []
    unmatched = 0

    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            batch_gpu = batch_to_device(dict(batch), device)
            batch_gpu["action"] = torch.nan_to_num(batch_gpu["action"], 0.0)
            output = model.encode(batch_gpu)
            emb = output["emb"]
            act_emb = output["act_emb"]
            pred = model.predict(emb[:, : cfg.history_size], act_emb[:, : cfg.history_size])
            tgt = emb[:, cfg.num_preds :]
            last_pred = pred[:, -1]
            last_tgt = tgt[:, -1]
            diff = last_pred - last_tgt
            b_last_mse = diff.pow(2).mean(1).detach().cpu()
            b_last_l2 = diff.norm(dim=1).detach().cpu()
            b_cos = F.cosine_similarity(last_pred, last_tgt, dim=1).detach().cpu()
            b_window_mse = (pred - tgt).pow(2).mean(dim=(1, 2)).detach().cpu()
            b_pred_norm = last_pred.norm(dim=1).detach().cpu()
            b_target_norm = last_tgt.norm(dim=1).detach().cpu()

            ep = batch["ep_idx"][:, cfg.history_size - 1].long().cpu()
            ts = batch["step_idx"][:, cfg.history_size - 1].long().cpu()
            for j, (e, t) in enumerate(zip(ep.tolist(), ts.tolist())):
                row = base_lookup.get((int(e), int(t)))
                if row is None:
                    unmatched += 1
                    continue
                if torch.isfinite(last_mse[row]):
                    continue
                last_mse[row] = b_last_mse[j]
                last_l2[row] = b_last_l2[j]
                last_cosine[row] = b_cos[j]
                last_cosine_error[row] = 1.0 - b_cos[j]
                window_mse[row] = b_window_mse[j]
                pred_norm[row] = b_pred_norm[j]
                target_norm[row] = b_target_norm[j]
                eval_base_rows.append(row)
            if batch_i % 100 == 0:
                print(f"processed val batch {batch_i}/{len(loader)} joined={len(eval_base_rows)} unmatched={unmatched}", flush=True)

    eval_base_idx = torch.tensor(sorted(set(eval_base_rows)), dtype=torch.long)
    errors = {
        "last_step_mse": last_mse,
        "last_step_l2": last_l2,
        "last_step_cosine": last_cosine,
        "last_step_cosine_error": last_cosine_error,
        "official_window_mse": window_mse,
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "valid_context": torch.isfinite(last_mse),
        "eval_base_idx": eval_base_idx,
        "metadata": {
            "checkpoint": str(checkpoint),
            "best_h_dataset": str(BEST_H_DATASET),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "prediction_path": "official validation DataLoader batch -> model.encode(batch) -> official action_encoder + model.predict(ctx_emb, ctx_act_emb)",
            "last_step_definition": "for each official val sequence, compare final predictor token against emb[:, -1] target from train.py target window",
            "official_window_definition": "same as train.py pred_loss over shifted targets emb[:, num_preds:]",
            "joined_eval_samples": int(eval_base_idx.numel()),
            "unmatched_eval_samples": int(unmatched),
            "repo_commit": git_text(["rev-parse", "HEAD"]),
            "latent_dim": int(cfg.embed_dim),
            "action_dim_block": int(cfg.model.action_encoder.input_dim),
            "dataset_name": str(cfg.data.dataset.name),
            "dataset_action_dim": int(dataset.get_dim("action")),
            "batch_size": int(batch_size),
            "split": "official validation split from train.py random_split seed",
        },
    }
    torch.save(errors, OUT / "direct_predictor_errors.pt")
    return base, errors


def gather_context_indices(base, latents):
    lookup = build_latent_lookup(latents)
    ep = base["episode_id"].long()
    ts = base["timestep"].long()
    ctx = torch.full((ep.numel(), 3), -1, dtype=torch.long)
    target = torch.full((ep.numel(),), -1, dtype=torch.long)
    for i, (e, t) in enumerate(zip(ep.tolist(), ts.tolist())):
        ids = [lookup.get((int(e), int(t) - 2 + j), -1) for j in range(3)]
        tid = lookup.get((int(e), int(t) + 1), -1)
        if min(ids) >= 0 and tid >= 0:
            ctx[i] = torch.tensor(ids, dtype=torch.long)
            target[i] = int(tid)
    valid = (ctx >= 0).all(1) & (target >= 0)
    return ctx, target, valid


def compute_direct_errors(checkpoint, batch_size=4096, device=None):
    cfg = compose_tworoom_cfg(batch_size=128, num_workers=0)
    dataset = load_tworoom_dataset(cfg)
    model, missing, unexpected = load_model(cfg, checkpoint)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    latents = torch.load(LATENTS_PATH, map_location="cpu")
    base = torch.load(BEST_H_DATASET, map_location="cpu")
    ctx_idx, target_idx, valid = gather_context_indices(base, latents)

    z = latents["z"].float()
    action_mean, action_std = action_normalization_stats(dataset)
    action = torch.nan_to_num(normalize_action_like_training(latents["action"].float(), action_mean, action_std), 0.0)
    n = ctx_idx.shape[0]
    last_mse = torch.full((n,), float("nan"))
    last_l2 = torch.full((n,), float("nan"))
    last_cosine = torch.full((n,), float("nan"))
    last_cosine_error = torch.full((n,), float("nan"))
    window_mse = torch.full((n,), float("nan"))
    pred_norm = torch.full((n,), float("nan"))
    target_norm = torch.full((n,), float("nan"))

    valid_idx = torch.nonzero(valid).flatten()
    with torch.no_grad():
        for start in range(0, valid_idx.numel(), batch_size):
            rows = valid_idx[start : start + batch_size]
            ci = ctx_idx[rows]
            ti = target_idx[rows]
            ctx_z = z[ci].to(device)
            ctx_action = action[ci].to(device)
            target_z = z[ti].to(device)
            shifted_target = torch.cat([ctx_z[:, 1:], target_z[:, None]], 1)
            act_emb = model.action_encoder(ctx_action)
            pred = model.predict(ctx_z, act_emb)
            last_pred = pred[:, -1]
            diff = last_pred - target_z
            last_mse[rows] = diff.pow(2).mean(1).detach().cpu()
            last_l2[rows] = diff.norm(dim=1).detach().cpu()
            cos = F.cosine_similarity(last_pred, target_z, dim=1)
            last_cosine[rows] = cos.detach().cpu()
            last_cosine_error[rows] = (1.0 - cos).detach().cpu()
            window_mse[rows] = (pred - shifted_target).pow(2).mean(dim=(1, 2)).detach().cpu()
            pred_norm[rows] = last_pred.norm(dim=1).detach().cpu()
            target_norm[rows] = target_z.norm(dim=1).detach().cpu()

    errors = {
        "last_step_mse": last_mse,
        "last_step_l2": last_l2,
        "last_step_cosine": last_cosine,
        "last_step_cosine_error": last_cosine_error,
        "official_window_mse": window_mse,
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "valid_context": valid,
        "context_latent_indices": ctx_idx,
        "target_latent_index": target_idx,
        "metadata": {
            "checkpoint": str(checkpoint),
            "latent_cache": str(LATENTS_PATH),
            "best_h_dataset": str(BEST_H_DATASET),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "prediction_path": "official action_encoder + model.predict(ctx_emb, ctx_act_emb), where model.predict calls predictor and pred_proj",
            "action_normalization": "raw latent-cache action is reshaped as frameskip x raw_action_dim, z-scored with the same get_column_normalizer(dataset, 'action', 'action') statistics used by train.py, then flattened back to the action_encoder input dimension",
            "action_mean_first3": action_mean.flatten()[:3].tolist(),
            "action_std_first3": action_std.flatten()[:3].tolist(),
            "last_step_definition": "context [t-2,t-1,t], action [t-2,t-1,t], target z[t+1], using final predictor token",
            "official_window_definition": "same as train.py pred_loss over three shifted targets [t-1,t,t+1]",
            "repo_commit": git_text(["rev-parse", "HEAD"]),
            "latent_dim": int(cfg.embed_dim),
            "action_dim_block": int(cfg.model.action_encoder.input_dim),
            "dataset_name": str(cfg.data.dataset.name),
            "dataset_action_dim": int(dataset.get_dim("action")),
        },
    }
    torch.save(errors, OUT / "direct_predictor_errors.pt")
    return base, latents, errors


def split_mask(base, errors, split_name):
    if "eval_base_idx" in errors:
        idx = errors["eval_base_idx"].long()
        return idx[errors["valid_context"][idx]]
    idx = base[f"{split_name}_idx"].long()
    return idx[errors["valid_context"][idx]]


def failure_labels_from_train_thresholds(base, errors):
    tr = split_mask(base, errors, "train")
    score_defs = {
        "direct_mse": errors["last_step_mse"],
        "direct_cosine_error": errors["last_step_cosine_error"],
        "official_window_mse": errors["official_window_mse"],
    }
    labels = {}
    thresholds = {}
    for score_name, score in score_defs.items():
        for q, suffix in [(0.80, "top20"), (0.90, "top10"), (0.95, "top5")]:
            th = finite_quantile(score[tr], q)
            name = f"{score_name}_{suffix}"
            labels[name] = score >= th
            thresholds[name] = th
    return labels, thresholds


def summary_rows(base, errors, labels):
    te = split_mask(base, errors, "test")
    rows = []
    for name, label in labels.items():
        active = label[te] & errors["valid_context"][te]
        rows.append(
            {
                "failure_mode": name,
                "count": int(active.sum().item()),
                "rate": active.float().mean().item(),
                "last_step_mse_mean": nanmean(errors["last_step_mse"][te][active]),
                "last_step_mse_global_mean": nanmean(errors["last_step_mse"][te]),
                "last_step_cosine_mean": nanmean(errors["last_step_cosine"][te][active]),
                "official_window_mse_mean": nanmean(errors["official_window_mse"][te][active]),
                "official_window_mse_global_mean": nanmean(errors["official_window_mse"][te]),
                "x_mean": base["position_t"][te][active, 0].float().mean().item() if active.any() else float("nan"),
                "y_mean": base["position_t"][te][active, 1].float().mean().item() if active.any() else float("nan"),
            }
        )
    return rows


def rule_overlap_rows(base, errors, labels):
    te = split_mask(base, errors, "test")
    v4_key, q4 = compute_v4_q(base)
    v41_key, q41 = compute_v41_q(base)
    v42_key, q42 = compute_v42_q(base)
    scores = {
        "v42_rule_10": q42[:, V42_BEST_RULE].float(),
        "v42_group_10_13_12_1": q42[:, V42_GROUP0].float().mean(1),
        "v42_group_best_rules_10_13_12_6": q42[:, V42_GROUPS["best_rules"]].float().mean(1),
        "v4_group_15_2_8_11": q4[:, V4_GROUP0].float().mean(1),
        "v41_group_top4_mean": q41[:, :4].float().mean(1),
    }
    rows = []
    for sname, score in scores.items():
        score_te = score[te]
        active = score_te >= torch.quantile(score_te, 0.9)
        for label_name, label in labels.items():
            y = label[te]
            base_rate = y.float().mean().item()
            active_rate = y[active].float().mean().item() if active.any() else float("nan")
            inactive_rate = y[~active].float().mean().item() if (~active).any() else float("nan")
            rows.append(
                {
                    "rule_or_group": sname,
                    "failure_mode": label_name,
                    "base_rate": base_rate,
                    "active_rate": active_rate,
                    "inactive_rate": inactive_rate,
                    "enrichment": active_rate / max(base_rate, 1e-8),
                    "active_count": int(active.sum().item()),
                    "q_source": {"v4": v4_key, "v41": v41_key, "v42": v42_key}.get(sname.split("_")[0], ""),
                }
            )
    return rows, scores


def load_images_for_rows(base, rows, radius=0):
    import hdf5plugin  # noqa: F401
    import h5py

    rows = torch.as_tensor(rows, dtype=torch.long)
    requests = []
    for row in rows.tolist():
        ep = int(base["episode_id"][row])
        ts = int(base["timestep"][row])
        for dt in range(-radius, radius + 1):
            requests.append((row, ep, ts + dt, dt))
    with h5py.File(H5_PATH, "r") as f:
        offsets = f["ep_offset"][:]
        valid = []
        for row, ep, ts, dt in requests:
            if ts < 0 or ep < 0 or ep + 1 >= len(offsets):
                continue
            h5_idx = int(offsets[ep] + ts)
            if h5_idx < int(offsets[ep + 1]):
                valid.append((row, ep, ts, dt, h5_idx))
        order = np.argsort([x[-1] for x in valid])
        imgs = {}
        for start in range(0, len(order), 256):
            cur = [valid[i] for i in order[start : start + 256]]
            raw_idx = [x[-1] for x in cur]
            raw = f["pixels"][raw_idx]
            x = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0
            small = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
            small = small.permute(0, 2, 3, 1).numpy()
            for meta, img in zip(cur, small):
                imgs[(meta[0], meta[3])] = img
    return imgs


def save_image_grid(path, images, titles, ncols=4):
    if not images:
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


def scatter_map(path, pos, score, title, active=None):
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(pos[:, 0], pos[:, 1], c=score, s=5, cmap="magma", alpha=0.55)
    if active is not None and active.any():
        p = pos[active]
        plt.scatter(p[:, 0], p[:, 1], facecolors="none", edgecolors="cyan", s=26, linewidths=0.7, label="selected")
        plt.legend(fontsize=8)
    plt.colorbar(label="direct predictor error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_rule_error_overlay(path, pos, rule_score, error_score, title):
    top_rule = rule_score >= torch.quantile(rule_score, 0.9)
    top_error = error_score >= torch.quantile(error_score[torch.isfinite(error_score)], 0.9)
    both = top_rule & top_error
    plt.figure(figsize=(5.4, 5.0))
    plt.scatter(pos[:, 0], pos[:, 1], c="lightgray", s=5, alpha=0.25, label="test")
    plt.scatter(pos[top_error, 0], pos[top_error, 1], c="red", s=10, alpha=0.55, label="top error")
    plt.scatter(pos[top_rule, 0], pos[top_rule, 1], facecolors="none", edgecolors="blue", s=28, linewidths=0.7, label="top rule")
    plt.scatter(pos[both, 0], pos[both, 1], c="gold", s=16, alpha=0.75, label="both")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def make_figures(base, errors, labels, scores):
    te = split_mask(base, errors, "test")
    pos = base["position_t"][te].float()
    direct_mse = errors["last_step_mse"][te].float()
    window_mse = errors["official_window_mse"][te].float()
    figure_rows = []

    for name, score in [("direct_mse", direct_mse), ("official_window_mse", window_mse)]:
        active = score >= torch.quantile(score[torch.isfinite(score)], 0.9)
        path = FIG / f"{name}_spatial_map.png"
        scatter_map(path, pos, score, f"{name}: official predictor error over position", active=active)
        figure_rows.append({"figure": path.name, "type": "spatial_map", "source": name, "path": str(path)})

    for rule_name in ["v42_rule_10", "v42_group_10_13_12_1", "v4_group_15_2_8_11"]:
        path = FIG / f"{rule_name}_vs_direct_mse_overlay.png"
        save_rule_error_overlay(path, pos, scores[rule_name][te].float(), direct_mse, f"{rule_name} vs direct predictor top-error")
        figure_rows.append({"figure": path.name, "type": "rule_error_overlay", "source": rule_name, "path": str(path)})

    top_global = te[torch.argsort(errors["last_step_mse"][te], descending=True)[:12]]
    imgs = load_images_for_rows(base, top_global)
    images, titles = [], []
    for row in top_global.tolist():
        img = imgs.get((row, 0))
        if img is not None:
            images.append(img)
            titles.append(f"idx={row}\nt={int(base['timestep'][row])}\nmse={errors['last_step_mse'][row]:.4f}")
    path = FIG / "direct_mse_top_images.png"
    save_image_grid(path, images, titles)
    figure_rows.append({"figure": path.name, "type": "top_images", "source": "direct_mse_top12", "path": str(path)})

    story_rows = make_episode_stories(base, errors, scores["v42_group_10_13_12_1"], te)
    figure_rows.extend(story_rows)
    return figure_rows


def make_episode_stories(base, errors, rule_score, te, max_episodes=4):
    order = te[torch.argsort(errors["last_step_mse"][te], descending=True)]
    used = set()
    rows = []
    for row in order.tolist():
        ep = int(base["episode_id"][row])
        if ep in used:
            continue
        same = torch.nonzero(base["episode_id"].long() == ep).flatten()
        same = same[torch.argsort(base["timestep"][same])]
        center = torch.nonzero(same == row).flatten()
        if center.numel() == 0:
            continue
        c = int(center[0])
        seq = same[max(0, c - 4) : min(same.numel(), c + 5)]
        imgs = load_images_for_rows(base, seq)
        images, titles = [], []
        for s in seq.tolist():
            img = imgs.get((s, 0))
            if img is not None:
                images.append(img)
                titles.append(f"t={int(base['timestep'][s])}\nmse={errors['last_step_mse'][s]:.3f}\nqG={rule_score[s]:.2f}")
        prefix = f"episode_{ep}_around_t{int(base['timestep'][row])}"
        strip = FIG / f"{prefix}_direct_error_strip.png"
        save_image_grid(strip, images, titles, ncols=min(9, max(1, len(images))))

        tim = base["timestep"][same].float()
        mse = errors["last_step_mse"][same].float()
        qg = rule_score[same].float()
        plt.figure(figsize=(6.3, 3.2))
        plt.plot(tim.numpy(), mse.numpy(), label="direct predictor MSE", linewidth=1.5)
        qg_scaled = (qg - qg.min()) / max((qg.max() - qg.min()).item(), 1e-8)
        qg_scaled = qg_scaled * max(mse[torch.isfinite(mse)].max().item(), 1e-8)
        plt.plot(tim.numpy(), qg_scaled.numpy(), label="V4.2 group activation (rescaled)", linewidth=1.2)
        plt.axvline(float(base["timestep"][row]), color="red", linestyle="--", linewidth=1)
        plt.xlabel("timestep")
        plt.title(f"{prefix}: direct predictor error and rule activation")
        plt.legend(fontsize=8)
        plt.tight_layout()
        timeseries = FIG / f"{prefix}_direct_error_timeseries.png"
        plt.savefig(timeseries, dpi=170)
        plt.close()

        pos = base["position_t"][same].float()
        plt.figure(figsize=(5.2, 5.0))
        plt.scatter(pos[:, 0], pos[:, 1], c=mse, s=14, cmap="magma", alpha=0.75)
        plt.plot(pos[:, 0], pos[:, 1], color="gray", linewidth=0.7, alpha=0.45)
        p = base["position_t"][row].float()
        plt.scatter([p[0]], [p[1]], marker="x", s=80, c="cyan", linewidths=2, label="selected direct failure")
        plt.colorbar(label="direct predictor MSE")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{prefix}: trajectory by direct predictor error")
        plt.legend(fontsize=8)
        plt.tight_layout()
        traj = FIG / f"{prefix}_direct_error_trajectory.png"
        plt.savefig(traj, dpi=170)
        plt.close()

        rows.extend(
            [
                {"figure": strip.name, "type": "episode_strip", "source": prefix, "path": str(strip)},
                {"figure": timeseries.name, "type": "episode_timeseries", "source": prefix, "path": str(timeseries)},
                {"figure": traj.name, "type": "episode_trajectory", "source": prefix, "path": str(traj)},
            ]
        )
        used.add(ep)
        if len(used) >= max_episodes:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse-errors", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    checkpoint = Path(args.checkpoint)
    if args.reuse_errors and (OUT / "direct_predictor_errors.pt").exists():
        base = torch.load(BEST_H_DATASET, map_location="cpu")
        errors = torch.load(OUT / "direct_predictor_errors.pt", map_location="cpu")
    else:
        base, errors = compute_direct_errors_from_official_val(checkpoint, batch_size=args.batch_size, device=args.device)

    labels, thresholds = failure_labels_from_train_thresholds(base, errors)
    summary = summary_rows(base, errors, labels)
    overlap, scores = rule_overlap_rows(base, errors, labels)
    overlap_top = sorted(overlap, key=lambda r: r["enrichment"], reverse=True)[:18]
    figure_rows = make_figures(base, errors, labels, scores)

    csv_write(TABLE / "direct_prediction_error_summary.csv", summary)
    csv_write(TABLE / "rule_direct_failure_overlap.csv", overlap)
    csv_write(TABLE / "direct_failure_thresholds.csv", [{"failure_mode": k, "threshold": v} for k, v in thresholds.items()])
    csv_write(TABLE / "direct_failure_visual_index.csv", figure_rows)

    te = split_mask(base, errors, "test")
    meta = errors["metadata"]
    text = f"""# TwoRoom Direct Predictor Failure Atlas

## Purpose

This atlas uses the official trained TwoRoom LeWM baseline predictor directly. It is not a residual proxy atlas.

The computation path is:

```text
official validation batch pixels/action from train.py data path
official model.encode(batch) -> emb/action embeddings
official model.predict(ctx_emb, ctx_act_emb)
last predictor token -> pred_z[t+1]
compare with official encoded target emb[:, -1]
```

The `official_window_mse` column also mirrors `train.py` more closely by averaging the predictor outputs against shifted targets `[t-1, t, t+1]`.

## Metadata

- checkpoint: `{meta.get('checkpoint')}`
- eval source: `{meta.get('split')}`
- joined eval samples: `{meta.get('joined_eval_samples')}`
- unmatched eval samples: `{meta.get('unmatched_eval_samples')}`
- valid context samples: `{int(errors['valid_context'].sum().item())}`
- analyzed eval samples: `{int(te.numel())}`
- latent dim: `{meta.get('latent_dim')}`
- action block dim: `{meta.get('action_dim_block')}`
- git commit: `{meta.get('repo_commit')}`
- missing keys: `{meta.get('missing_keys')}`
- unexpected keys: `{meta.get('unexpected_keys')}`

## Direct Predictor Failure Summary

{table_md(summary, ["failure_mode", "count", "rate", "last_step_mse_mean", "last_step_mse_global_mean", "last_step_cosine_mean", "official_window_mse_mean", "official_window_mse_global_mean"])}

## Rule Alignment With Direct Predictor Failures

Top enrichments:

{table_md(overlap_top, ["rule_or_group", "failure_mode", "base_rate", "active_rate", "inactive_rate", "enrichment", "active_count"])}

## Figure Index

{table_md(figure_rows, ["figure", "type", "source", "path"])}

## Current Read

- `direct_mse_top10/top20` is the strongest one-step failure label because it comes from the official predictor output and official next latent target.
- If a V4/V4.2 rule group has high enrichment on `direct_mse_top10`, it means the rule is aligned with true official LeWM next-latent prediction failures, not only with previous physical residual proxies.
- The episode figures show concrete examples: current images, trajectory location, direct predictor error over time, and rescaled V4.2 group activation.

## Caveats

- This is one-step direct predictor failure analysis, aligned with the official training objective. It is not yet multi-step rollout failure.
- It directly re-encodes official validation pixels with the official checkpoint and official validation DataLoader transform, then runs the official predictor/action encoder.
- High enrichment is evidence of strong association, not causal proof. Causal proof still requires rule masking/intervention.
"""
    write_report("README.md", text)


if __name__ == "__main__":
    main()
