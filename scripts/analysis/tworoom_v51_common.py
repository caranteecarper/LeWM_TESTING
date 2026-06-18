import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tworoom_v5_common import (
    REPO_ROOT,
    V5_OUT,
    VISUAL_DATASET,
    load_visual,
    table_md,
    save_image_grid,
    scatter_pos,
    spatial_heatmap,
)


V51_OUT = REPO_ROOT / "outputs" / "tworoom_visualization_v51"
V51_REPORT = REPO_ROOT / "reports" / "tworoom_visualization_v51"
V51_FIG = V51_OUT / "figures"
V5_RULE_FIG = V5_OUT / "figures" / "rule_visual_link"


def ensure_dirs():
    V51_OUT.mkdir(parents=True, exist_ok=True)
    V51_REPORT.mkdir(parents=True, exist_ok=True)
    V51_FIG.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = V51_REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def git_text(args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def hard_enrichment(vis, score, mask=None):
    te = vis["test_idx"]
    if mask is None:
        mask = torch.ones(te.numel(), dtype=torch.bool)
    hard = vis["hard_labels"][te][mask]
    active = score[mask] >= torch.quantile(score[mask], 0.9)
    out = {}
    for i, name in enumerate(vis["hard_label_names"]):
        base = hard[:, i].mean().item() if hard.numel() else 0.0
        val = hard[active, i].mean().item() if active.any() else 0.0
        out[name] = val / max(base, 1e-8)
    return out


def compactness(pos, score):
    active = score >= torch.quantile(score, 0.9)
    if active.sum() < 3:
        return float("nan")
    p = pos[active].float()
    return p.std(0).mean().item()


def top_rule_rows(vis, q_name, k=8):
    te = vis["test_idx"]
    q = vis[q_name][te].float()
    pos = vis["position_t"][te]
    residual = vis["residual_t"][te]
    rows = []
    for r in range(q.shape[1]):
        score = q[:, r]
        active = score >= torch.quantile(score, 0.9)
        enrich = hard_enrichment(vis, score)
        top_hard, top_enrich = max(enrich.items(), key=lambda kv: kv[1])
        rows.append(
            {
                "unit_id": r,
                "top_hard_subset": top_hard,
                "hard_enrichment": top_enrich,
                "mean_residual_mag": residual[active].norm(dim=1).mean().item() if active.any() else 0.0,
                "spatial_compactness": compactness(pos, score),
                "coverage_top10": active.float().mean().item(),
            }
        )
    return sorted(rows, key=lambda r: r["hard_enrichment"] * r["mean_residual_mag"], reverse=True)[:k]


def save_top_images(path, vis, sample_idx, title):
    imgs = [[vis["image_t"][i].float() / 255.0] for i in sample_idx]
    save_image_grid(path, imgs, [title], row_titles=[f"#{int(i)}" for i in sample_idx], figsize_scale=2.0)


def save_residual_arrows(path, pos, residual, title, max_arrows=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = pos.float()
    residual = residual.float()
    if pos.shape[0] > max_arrows:
        idx = torch.linspace(0, pos.shape[0] - 1, max_arrows).long()
        pos = pos[idx]
        residual = residual[idx]
    mag = residual.norm(dim=1).clamp_min(1e-6)
    u = residual[:, 0] / mag
    v = residual[:, 1] / mag
    plt.figure(figsize=(5, 5))
    plt.scatter(pos[:, 0], pos[:, 1], c=mag, s=10, alpha=0.55, cmap="magma")
    plt.quiver(pos[:, 0], pos[:, 1], u, v, angles="xy", scale_units="xy", scale=0.08, width=0.004, color="black", alpha=0.65)
    plt.colorbar(label="residual magnitude")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_unit_visuals(fig_dir, vis, q_name, unit_id, prefix):
    te = vis["test_idx"]
    q = vis[q_name][te].float()
    score = q[:, unit_id]
    top_local = torch.topk(score, k=min(200, score.numel())).indices
    top_samples = te[top_local]
    save_top_images(fig_dir / f"{prefix}_top_images.png", vis, top_samples[:8], prefix)
    scatter_pos(fig_dir / f"{prefix}_positions.png", vis["position_t"][te], color=score, title=f"{prefix} activation over position")
    save_residual_arrows(fig_dir / f"{prefix}_residual_direction.png", vis["position_t"][top_samples], vis["residual_t"][top_samples], f"{prefix} residual directions")


def group_score(q, rules):
    return q[:, rules].mean(1)


def save_group_visuals(fig_dir, vis, group_id, rules):
    te = vis["test_idx"]
    score = group_score(vis["q_v4_t"][te].float(), rules)
    top_local = torch.topk(score, k=min(200, score.numel())).indices
    top_samples = te[top_local]
    prefix = f"v4_group_{group_id}_rules_{'-'.join(map(str, rules))}"
    save_top_images(fig_dir / f"{prefix}_top_images.png", vis, top_samples[:8], prefix)
    scatter_pos(fig_dir / f"{prefix}_positions.png", vis["position_t"][te], color=score, title=f"{prefix} activation over position")
    save_residual_arrows(fig_dir / f"{prefix}_residual_direction.png", vis["position_t"][top_samples], vis["residual_t"][top_samples], f"{prefix} residual directions")
    enrich = hard_enrichment(vis, score)
    top_hard, top_enrich = max(enrich.items(), key=lambda kv: kv[1])
    return {
        "group_id": group_id,
        "rules": ",".join(map(str, rules)),
        "top_hard_subset": top_hard,
        "hard_enrichment": top_enrich,
        "mean_residual_mag": vis["residual_t"][top_samples].norm(dim=1).mean().item(),
        "spatial_compactness": compactness(vis["position_t"][te], score),
        "coverage_top10": 0.1,
    }

