import csv
import math
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v4_common import (
    REPO_ROOT,
    BEST_H_DATASET,
    RULE_DIR as V4_RULE_DIR,
    table_md,
    pearson,
    metric_dict,
    standardize_from_train,
    inverse_standardize,
    fit_linear,
    predict_linear,
    MLPRegressor,
    train_model,
    batched_predict,
    load_rule_model,
)
from tworoom_v41_common import V41_RULE_DIR, load_sharp_model, q_from_model


V5_OUT = REPO_ROOT / "outputs" / "tworoom_visualization_v5"
V5_REPORT = REPO_ROOT / "reports" / "tworoom_visualization_v5"
V5_FIG = V5_OUT / "figures"
VISUAL_DATASET = V5_OUT / "visual_dataset.pt"
H5_PATH = Path("/data/lzt26/stable-wm/datasets/tworoom.h5")


def ensure_dirs():
    V5_OUT.mkdir(parents=True, exist_ok=True)
    V5_REPORT.mkdir(parents=True, exist_ok=True)
    V5_FIG.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = V5_REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def git_text(args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def load_visual():
    return torch.load(VISUAL_DATASET, map_location="cpu")


def load_base_data():
    return torch.load(BEST_H_DATASET, map_location="cpu")


def compute_v4_q_for_indices(data, idx):
    summary = torch.load(V4_RULE_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    model, obj = load_rule_model(V4_RULE_DIR / f"{key}.pt")
    norm = obj["norm"]
    h = (data["h_t"][idx].float() - norm["h_t_mean"]) / norm["h_t_std"].clamp_min(1e-6)
    action = (data["action"][idx].float() - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    q = []
    with torch.no_grad():
        for i in range(0, h.shape[0], 16384):
            q.append(model(h[i : i + 16384], action[i : i + 16384])["q"])
    return key, torch.cat(q, 0).float()


def compute_v41_q_for_indices(data, idx):
    summary = torch.load(V41_RULE_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    model, obj = load_sharp_model(V41_RULE_DIR / f"{key}.pt")
    # Reuse helper on a small sliced dict so q extraction still uses h only.
    sliced = {
        "h_t": data["h_t"][idx].float(),
        "action": data["action"][idx].float(),
    }
    q = q_from_model(model, obj, sliced, hard_topk=True)
    return key, q.float()


def psnr_from_mse(mse):
    return 10.0 * math.log10(1.0 / max(float(mse), 1e-12))


def add_bias(x):
    return torch.cat([x.float(), torch.ones(x.shape[0], 1)], 1)


def ridge_predict_all(x, y, train_idx):
    xs, xm, xstd = standardize_from_train(x, train_idx)
    ys, ym, ystd = standardize_from_train(y, train_idx)
    w, b = fit_linear(xs, ys, train_idx)
    return inverse_standardize(predict_linear(xs, w, b), ym, ystd)


def binary_metrics_from_logits(logits, target):
    prob = torch.sigmoid(logits.float())
    pred = prob >= 0.5
    t = target.float() >= 0.5
    tp = (pred & t).sum(0).float()
    fp = (pred & ~t).sum(0).float()
    fn = (~pred & t).sum(0).float()
    tn = (~pred & ~t).sum(0).float()
    acc = ((tp + tn) / (tp + tn + fp + fn).clamp_min(1)).mean().item()
    precision = (tp / (tp + fp).clamp_min(1)).mean().item()
    recall = (tp / (tp + fn).clamp_min(1)).mean().item()
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"hard_acc": acc, "hard_f1": f1}


def save_image_grid(path, rows, col_titles, row_titles=None, figsize_scale=2.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = len(rows)
    ncol = len(rows[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize_scale * ncol, figsize_scale * nrow))
    if nrow == 1:
        axes = [axes]
    for r in range(nrow):
        for c in range(ncol):
            ax = axes[r][c]
            img = rows[r][c]
            if torch.is_tensor(img):
                img = img.detach().cpu()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)
                img = img.numpy()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
            if c == 0 and row_titles:
                ax.set_ylabel(row_titles[r], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def spatial_heatmap(path, pos, values, title, bins=48):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = pos.float()
    values = values.float().reshape(-1)
    x = pos[:, 0].numpy()
    y = pos[:, 1].numpy()
    v = values.numpy()
    sum_grid, xe, ye = np.histogram2d(x, y, bins=bins, weights=v)
    cnt_grid, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    grid = (sum_grid / np.maximum(cnt_grid, 1)).T
    grid[cnt_grid.T == 0] = np.nan
    plt.figure(figsize=(5, 4))
    plt.imshow(grid, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis")
    plt.colorbar(label="mean value")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def scatter_pos(path, pos, color=None, title="position", max_points=40000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = pos.float()
    if pos.shape[0] > max_points:
        idx = torch.linspace(0, pos.shape[0] - 1, max_points).long()
        pos = pos[idx]
        color = color[idx] if color is not None and torch.is_tensor(color) else color
    plt.figure(figsize=(5, 5))
    if color is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=2, alpha=0.25)
    else:
        plt.scatter(pos[:, 0], pos[:, 1], c=color.float(), s=2, alpha=0.45, cmap="viridis")
        plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def quiver_residual(path, pos, residual, title="residual direction", bins=16):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = pos.float()
    residual = residual.float()
    x_edges = torch.linspace(pos[:, 0].min(), pos[:, 0].max(), bins + 1)
    y_edges = torch.linspace(pos[:, 1].min(), pos[:, 1].max(), bins + 1)
    xs, ys, us, vs = [], [], [], []
    for i in range(bins):
        for j in range(bins):
            m = (pos[:, 0] >= x_edges[i]) & (pos[:, 0] < x_edges[i + 1]) & (pos[:, 1] >= y_edges[j]) & (pos[:, 1] < y_edges[j + 1])
            if m.sum() >= 20:
                xs.append(((x_edges[i] + x_edges[i + 1]) / 2).item())
                ys.append(((y_edges[j] + y_edges[j + 1]) / 2).item())
                us.append(residual[m, 0].mean().item())
                vs.append(residual[m, 1].mean().item())
    plt.figure(figsize=(5, 5))
    plt.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=0.25)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
