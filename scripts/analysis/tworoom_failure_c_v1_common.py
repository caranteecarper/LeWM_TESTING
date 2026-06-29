import csv
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v4_common import BEST_H_DATASET, table_md
from tworoom_v42_common import compute_v42_q


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs" / "tworoom_failure_confidence_head_v1"
REPORT = REPO_ROOT / "reports" / "tworoom_failure_confidence_head_v1"
TABLE = OUT / "tables"
MODEL_DIR = OUT / "models"
FIG = OUT / "figures"
DIRECT_ERROR_PATH = REPO_ROOT / "outputs" / "tworoom_direct_predictor_failure_atlas" / "direct_predictor_errors.pt"
H5_PATH = Path("/data/lzt26/stable-wm/datasets/tworoom.h5")

CLASS_LABELS = [
    "official_window_mse_top5",
    "official_window_mse_top10",
    "official_window_mse_top20",
    "direct_mse_top5",
    "direct_mse_top10",
    "direct_mse_top20",
]
REG_LABELS = ["log_official_window_mse", "log_direct_mse"]
PRIMARY_LABEL = "official_window_mse_top10"
V42_GROUP = [10, 13, 12, 1]


def ensure_dirs():
    for path in [OUT, REPORT, TABLE, MODEL_DIR, FIG]:
        path.mkdir(parents=True, exist_ok=True)


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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_base_and_errors():
    base = torch.load(BEST_H_DATASET, map_location="cpu")
    errors = torch.load(DIRECT_ERROR_PATH, map_location="cpu")
    return base, errors


def load_or_compute_q(base):
    _, q = compute_v42_q(base)
    return q.float()


def aligned_indices(base, errors):
    if "eval_base_idx" in errors:
        idx = errors["eval_base_idx"].long()
    else:
        idx = torch.nonzero(errors["valid_context"]).flatten().long()
    idx = idx[errors["valid_context"][idx]]
    finite = torch.isfinite(errors["official_window_mse"][idx]) & torch.isfinite(errors["last_step_mse"][idx])
    return idx[finite]


def split_by_episode(episode_id, idx, seed=3072, train_frac=0.70, val_frac=0.15):
    eps = torch.unique(episode_id[idx].long())
    g = torch.Generator().manual_seed(seed)
    eps = eps[torch.randperm(eps.numel(), generator=g)]
    n_train = int(train_frac * eps.numel())
    n_val = int(val_frac * eps.numel())
    train_eps = eps[:n_train]
    val_eps = eps[n_train : n_train + n_val]
    test_eps = eps[n_train + n_val :]

    def take(selected_eps):
        return idx[torch.isin(episode_id[idx].long(), selected_eps)]

    return {
        "head_train_idx": take(train_eps),
        "head_val_idx": take(val_eps),
        "head_test_idx": take(test_eps),
        "head_train_episodes": train_eps,
        "head_val_episodes": val_eps,
        "head_test_episodes": test_eps,
    }


def safe_corr(x, y):
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    m = torch.isfinite(x) & torch.isfinite(y)
    x = x[m]
    y = y[m]
    if x.numel() < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = x.norm() * y.norm()
    return (x @ y / den).item() if den > 0 else float("nan")


def build_aligned_dataset():
    ensure_dirs()
    base, errors = load_base_and_errors()
    idx = aligned_indices(base, errors)
    q = load_or_compute_q(base)
    split = split_by_episode(base["episode_id"], idx)
    action_key = "action" if "action" in base else "action_t"
    z_key = "z_t" if "z_t" in base else None
    out = {
        "source_base_idx": idx,
        "h": base["h_t"][idx].float(),
        "q": q[idx].float(),
        "action": base[action_key][idx].float(),
        "position": base["position_t"][idx].float(),
        "episode_id": base["episode_id"][idx].long(),
        "timestep": base["timestep"][idx].long(),
        "direct_mse": errors["last_step_mse"][idx].float(),
        "direct_cosine_error": errors["last_step_cosine_error"][idx].float(),
        "official_window_mse": errors["official_window_mse"][idx].float(),
        "split": {},
        "metadata": {
            "alignment": "official predictor errors are joined to h/q/action by source_base_idx produced from episode_id/timestep matching in direct predictor atlas",
            "direct_error_path": str(DIRECT_ERROR_PATH),
            "best_h_dataset": str(BEST_H_DATASET),
            "git_commit": git_text(["rev-parse", "HEAD"]),
            "sigreg_used_as_label": False,
            "label_target": "official predictor latent prediction error, not total train loss",
        },
    }
    if z_key:
        out["z"] = base[z_key][idx].float()
    old_to_new = torch.full((base["episode_id"].numel(),), -1, dtype=torch.long)
    old_to_new[idx] = torch.arange(idx.numel())
    for key in ["head_train_idx", "head_val_idx", "head_test_idx"]:
        out["split"][key] = old_to_new[split[key]].long()
    for key in ["head_train_episodes", "head_val_episodes", "head_test_episodes"]:
        out["split"][key] = split[key].long()
    return out, base, errors


def make_labels(data):
    tr = data["split"]["head_train_idx"]
    labels = {}
    thresholds = {}
    specs = {
        "official_window_mse": data["official_window_mse"],
        "direct_mse": data["direct_mse"],
        "direct_cosine_error": data["direct_cosine_error"],
    }
    for name, score in specs.items():
        for qv, suffix in [(0.95, "top5"), (0.90, "top10"), (0.80, "top20")]:
            key = f"{name}_{suffix}"
            threshold = torch.quantile(score[tr].float(), qv)
            thresholds[key] = threshold.item()
            labels[key] = (score >= threshold).float()
    labels["log_official_window_mse"] = torch.log(data["official_window_mse"].clamp_min(1e-8))
    labels["log_direct_mse"] = torch.log(data["direct_mse"].clamp_min(1e-8))
    data["labels"] = labels
    data["thresholds"] = thresholds
    data["primary_label"] = PRIMARY_LABEL if PRIMARY_LABEL in labels else "direct_mse_top10"
    return data


def save_dataset(data):
    ensure_dirs()
    path = OUT / "aligned_failure_dataset.pt"
    torch.save(data, path)
    return path


def load_dataset():
    return torch.load(OUT / "aligned_failure_dataset.pt", map_location="cpu")


def standardize_train(x, train_idx):
    mean = x[train_idx].float().mean(0, keepdim=True)
    std = x[train_idx].float().std(0, keepdim=True).clamp_min(1e-6)
    return (x.float() - mean) / std, mean.squeeze(0), std.squeeze(0)


def group_activation(q, mode="mean"):
    g = q[:, V42_GROUP].float()
    if mode == "sum":
        return g.sum(1, keepdim=True)
    if mode == "topk_avg":
        return torch.topk(g, k=min(2, g.shape[1]), dim=1).values.mean(1, keepdim=True)
    return g.mean(1, keepdim=True)


def make_feature(data, feature_group, seed=0):
    h, q, a = data["h"].float(), data["q"].float(), data["action"].float()
    gen = torch.Generator().manual_seed(1000 + seed)
    q_perm = q[torch.randperm(q.shape[0], generator=gen)]
    pieces = []
    group_slices = []

    def add(name, x):
        start = sum(p.shape[1] for p in pieces)
        pieces.append(x.float())
        group_slices.append((name, start, start + x.shape[1]))

    if feature_group == "action_only":
        add("action", a)
    elif feature_group == "h_action":
        add("h", h)
        add("action", a)
    elif feature_group == "q_action":
        add("q", q)
        add("action", a)
    elif feature_group == "h_q_action":
        add("h", h)
        add("q", q)
        add("action", a)
    elif feature_group == "h_q_group_action":
        add("h", h)
        add("q_group_mean", group_activation(q, "mean"))
        add("q_group_sum", group_activation(q, "sum"))
        add("q_group_topk_avg", group_activation(q, "topk_avg"))
        add("action", a)
    elif feature_group == "h_shuffled_q_action":
        add("h", h)
        add("q_shuffled", q_perm)
        add("action", a)
    elif feature_group == "q_shuffled_action":
        add("q_shuffled", q_perm)
        add("action", a)
    elif feature_group == "z_action":
        if "z" not in data:
            raise ValueError("z_action requested but aligned dataset has no z")
        add("z", data["z"].float())
        add("action", a)
    else:
        raise ValueError(f"unknown feature group: {feature_group}")
    return torch.cat(pieces, 1), group_slices


class ConfidenceHead(nn.Module):
    def __init__(self, in_dim, model_type="logistic_linear", group_slices=None, out_dim=8):
        super().__init__()
        self.model_type = model_type
        self.group_slices = group_slices or []
        if model_type in ("logistic_linear", "additive_linear"):
            if model_type == "additive_linear" and len(self.group_slices) > 1:
                self.parts = nn.ModuleList([nn.Linear(e - s, out_dim) for _, s, e in self.group_slices])
                self.bias = nn.Parameter(torch.zeros(out_dim))
            else:
                self.net = nn.Linear(in_dim, out_dim)
        elif model_type == "small_mlp_64":
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, out_dim))
        elif model_type == "small_mlp_128":
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, out_dim))
        else:
            raise ValueError(f"unknown model_type {model_type}")

    def forward(self, x):
        if self.model_type == "additive_linear" and hasattr(self, "parts"):
            out = self.bias[None].expand(x.shape[0], -1)
            for part, (_, s, e) in zip(self.parts, self.group_slices):
                out = out + part(x[:, s:e])
            return out
        return self.net(x)


def make_targets(data):
    cls = torch.stack([data["labels"][k].float() for k in CLASS_LABELS], 1)
    reg = torch.stack([data["labels"][k].float() for k in REG_LABELS], 1)
    return cls, reg


def train_one_model(data, feature_group, model_type, seed=0, max_epochs=100, batch_size=1024, lr=1e-3, patience=10, device="cuda"):
    torch.manual_seed(seed)
    x, group_slices = make_feature(data, feature_group, seed=seed)
    tr, va = data["split"]["head_train_idx"], data["split"]["head_val_idx"]
    x_std, mean, std = standardize_train(x, tr)
    y_cls, y_reg = make_targets(data)
    pos = y_cls[tr].sum(0)
    neg = y_cls[tr].shape[0] - pos
    pos_weight = (neg / pos.clamp_min(1)).clamp(1.0, 100.0)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ConfidenceHead(x.shape[1], model_type, group_slices=group_slices).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ds = TensorDataset(x_std[tr].float(), y_cls[tr].float(), y_reg[tr].float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    best_state, best_score = None, -float("inf")
    history = []
    bad = 0
    primary_i = CLASS_LABELS.index(data["primary_label"])
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb_cls, yb_reg in loader:
            xb, yb_cls, yb_reg = xb.to(device), yb_cls.to(device), yb_reg.to(device)
            out = model(xb)
            cls_logits, reg_pred = out[:, : len(CLASS_LABELS)], out[:, len(CLASS_LABELS) :]
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, yb_cls, pos_weight=pos_weight.to(device))
            reg_loss = F.smooth_l1_loss(reg_pred, yb_reg)
            loss = cls_loss + 0.1 * reg_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            pred = predict_model(model, x_std[va], device=device)
            score = average_precision(y_cls[va, primary_i], pred[:, primary_i].sigmoid())
        model.to(device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_primary_auprc": score})
        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.cpu()
    obj = {
        "state_dict": model.state_dict(),
        "feature_group": feature_group,
        "model_type": model_type,
        "seed": seed,
        "x_mean": mean,
        "x_std": std,
        "group_slices": group_slices,
        "history": history,
        "best_val_primary_auprc": best_score,
        "class_labels": CLASS_LABELS,
        "reg_labels": REG_LABELS,
        "primary_label": data["primary_label"],
    }
    return obj


def load_head(path):
    obj = torch.load(path, map_location="cpu")
    model = ConfidenceHead(int(obj["x_mean"].numel()), obj["model_type"], group_slices=obj.get("group_slices", []))
    model.load_state_dict(obj["state_dict"])
    return model, obj


@torch.no_grad()
def predict_model(model, x, batch_size=16384, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outs = []
    for i in range(0, x.shape[0], batch_size):
        outs.append(model(x[i : i + batch_size].float().to(device)).cpu())
    model.cpu()
    return torch.cat(outs, 0)


def sigmoid(x):
    return torch.sigmoid(x.float())


def roc_auc(y, s):
    y = y.float().reshape(-1)
    s = s.float().reshape(-1)
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float)
    return ((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def average_precision(y, s):
    y = y.float().reshape(-1)
    s = s.float().reshape(-1)
    pos_total = y.sum().item()
    if pos_total <= 0:
        return float("nan")
    order = torch.argsort(s, descending=True)
    yy = y[order]
    tp = torch.cumsum(yy, 0)
    precision = tp / torch.arange(1, yy.numel() + 1, dtype=torch.float)
    return ((precision * yy).sum() / pos_total).item()


def binary_eval(y, prob):
    y = y.float().reshape(-1)
    prob = prob.float().reshape(-1)
    pred = (prob >= 0.5).float()
    tp = ((pred == 1) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    acc = (tp + tn) / max(y.numel(), 1)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    brier = ((prob - y) ** 2).mean().item()
    return {
        "AUROC": roc_auc(y, prob),
        "AUPRC": average_precision(y, prob),
        "accuracy": acc,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": precision,
        "Brier": brier,
        "ECE": ece(y, prob),
    }


def ece(y, prob, bins=10):
    y = y.float()
    prob = prob.float()
    total = y.numel()
    out = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] if i < bins - 1 else prob <= edges[i + 1])
        if m.any():
            out += m.float().mean().item() * abs(prob[m].mean().item() - y[m].mean().item())
    return out


def precision_at_fraction(y, prob, frac):
    n = max(1, int(round(y.numel() * frac)))
    top = torch.argsort(prob, descending=True)[:n]
    return y[top].float().mean().item()


def recall_at_label_fraction(y, prob, frac):
    k = max(1, int(round(y.numel() * frac)))
    threshold = torch.topk(y.float(), k=min(k, y.numel())).values[-1]
    positives = y.float() >= threshold
    if positives.sum() == 0:
        return float("nan")
    top = torch.argsort(prob, descending=True)[: int(positives.sum().item())]
    return positives[top].float().sum().item() / positives.float().sum().item()


def eval_predictions(data, pred, split_idx, feature_group, model_type, seed):
    y_cls, _ = make_targets(data)
    rows = []
    probs = sigmoid(pred[:, : len(CLASS_LABELS)])
    for i, label in enumerate(CLASS_LABELS):
        y = y_cls[split_idx, i]
        p = probs[:, i]
        base = y.float().mean().item()
        row = {
            "feature_group": feature_group,
            "model_type": model_type,
            "seed": seed,
            "label": label,
            "split": "head_test",
            "base_rate": base,
            **binary_eval(y, p),
            "precision@top5_predicted_risk": precision_at_fraction(y, p, 0.05),
            "precision@top10_predicted_risk": precision_at_fraction(y, p, 0.10),
            "precision@top20_predicted_risk": precision_at_fraction(y, p, 0.20),
            "recall@top5_label": recall_at_label_fraction(y, p, 0.05),
            "recall@top10_label": recall_at_label_fraction(y, p, 0.10),
            "recall@top20_label": recall_at_label_fraction(y, p, 0.20),
        }
        row["precision@10/base_rate"] = row["precision@top10_predicted_risk"] / max(base, 1e-8)
        row["AUPRC/base_rate"] = row["AUPRC"] / max(base, 1e-8)
        rows.append(row)
    return rows


def summarize_best(rows, label=PRIMARY_LABEL):
    cand = [r for r in rows if r["label"] == label]
    return sorted(cand, key=lambda r: float(r["AUPRC"]), reverse=True)


def load_images(data, local_rows, image_size=64):
    import hdf5plugin  # noqa: F401
    import h5py

    local_rows = torch.as_tensor(local_rows, dtype=torch.long)
    with h5py.File(H5_PATH, "r") as f:
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
