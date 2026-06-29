import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tworoom_v42_full_common import REPO_ROOT, build_visual_data, csv_write, maybe_float
from tworoom_v4_common import table_md


OUT = REPO_ROOT / "outputs" / "tworoom_failure_atlas"
REPORT = REPO_ROOT / "reports" / "tworoom_failure_atlas"
TABLE = OUT / "tables"
FIG = OUT / "figures"


FAILURE_MODES = [
    "high_residual_top10",
    "action_error_top10",
    "large_action_small_disp",
    "high_residual_top20",
    "action_error_top20",
]

V42_GROUP0 = [10, 13, 12, 1]
V42_BEST_RULE = 10
V4_GROUP0 = [15, 2, 8, 11]


def ensure_dirs():
    for p in [OUT, REPORT, TABLE, FIG]:
        p.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def save_image_grid(path, images, titles, ncols=4):
    n = len(images)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        arr = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(arr)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_failure(path, vis, score, title, active=None):
    te = vis["test_idx"]
    pos = vis["position_t"][te].float()
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(pos[:, 0], pos[:, 1], c=score, s=8, cmap="magma", alpha=0.62)
    if active is not None and active.any():
        p = pos[active]
        plt.scatter(p[:, 0], p[:, 1], facecolors="none", edgecolors="cyan", s=32, linewidths=0.8, label="active/top")
        plt.legend(loc="best", fontsize=8)
    plt.colorbar(label="score")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def residual_arrows(path, vis, idx, title):
    if idx.numel() == 0:
        return
    pos = vis["position_t"][idx].float()
    residual = vis["residual_t"][idx].float()
    mag = residual.norm(dim=1).clamp_min(1e-6)
    u = residual[:, 0] / mag
    v = residual[:, 1] / mag
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(pos[:, 0], pos[:, 1], c=mag, s=16, cmap="viridis", alpha=0.78)
    plt.quiver(pos[:, 0], pos[:, 1], u, v, angles="xy", scale_units="xy", scale=0.075, width=0.004, color="black", alpha=0.72)
    plt.colorbar(label="residual magnitude")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def rule_overlap_rows(vis):
    te = vis["test_idx"]
    q42 = vis["q_v42_t"][te].float()
    q4 = vis["q_v4_t"][te].float()
    hard = vis["hard_labels"][te].float()
    names = vis["hard_label_names"]
    scores = {
        "v42_rule_10": q42[:, V42_BEST_RULE],
        "v42_group_10_13_12_1": q42[:, V42_GROUP0].mean(1),
        "v4_group_15_2_8_11": q4[:, V4_GROUP0].mean(1),
    }
    rows = []
    for sname, score in scores.items():
        active = score >= torch.quantile(score, 0.9)
        for label in FAILURE_MODES:
            j = names.index(label)
            base = hard[:, j].mean().item()
            selected = hard[active, j].mean().item()
            rows.append(
                {
                    "rule_or_group": sname,
                    "failure_mode": label,
                    "base_rate": base,
                    "active_rate": selected,
                    "enrichment": selected / max(base, 1e-8),
                    "active_count": int(active.sum()),
                }
            )
    return rows


def failure_summary_rows(vis):
    te = vis["test_idx"]
    residual = vis["residual_t"][te].float()
    mag = residual.norm(dim=1)
    delta = vis["delta_position"][te].float()
    pos = vis["position_t"][te].float()
    hard = vis["hard_labels"][te].float()
    names = vis["hard_label_names"]
    q42_group = vis["q_v42_t"][te][:, V42_GROUP0].mean(1).float()
    q42_rule = vis["q_v42_t"][te][:, V42_BEST_RULE].float()
    rows = []
    for label in FAILURE_MODES:
        j = names.index(label)
        active = hard[:, j] > 0.5
        rows.append(
            {
                "failure_mode": label,
                "count": int(active.sum()),
                "rate": active.float().mean().item(),
                "residual_mag_mean": mag[active].mean().item(),
                "residual_mag_global_mean": mag.mean().item(),
                "delta_mag_mean": delta[active].norm(dim=1).mean().item(),
                "x_mean": pos[active, 0].mean().item(),
                "y_mean": pos[active, 1].mean().item(),
                "v42_group_mean": q42_group[active].mean().item(),
                "v42_group_global_mean": q42_group.mean().item(),
                "v42_rule10_mean": q42_rule[active].mean().item(),
                "v42_rule10_global_mean": q42_rule.mean().item(),
            }
        )
    return rows


def top_samples_for_mode(vis, label, k=12):
    te = vis["test_idx"]
    names = vis["hard_label_names"]
    mag = vis["residual_t"][te].float().norm(dim=1)
    if label in names:
        active = vis["hard_labels"][te, names.index(label)] > 0.5
        local = torch.nonzero(active).flatten()
        if local.numel() == 0:
            local = torch.arange(te.numel())
    else:
        local = torch.arange(te.numel())
    order = local[torch.argsort(mag[local], descending=True)]
    return te[order[:k]]


def make_failure_mode_figures(vis):
    te = vis["test_idx"]
    mag = vis["residual_t"][te].float().norm(dim=1)
    scatter_failure(FIG / "overall_residual_magnitude_map.png", vis, mag, "Overall residual magnitude on test")
    residual_arrows(FIG / "overall_top_residual_arrows.png", vis, te[torch.argsort(mag, descending=True)[:220]], "Top residual directions")

    rows = []
    for label in FAILURE_MODES:
        idx = top_samples_for_mode(vis, label, k=12)
        titles = []
        for sample_idx in idx:
            local = torch.nonzero(te == sample_idx).flatten()
            m = mag[local[0]].item() if local.numel() else float("nan")
            titles.append(f"{label}\nidx={int(sample_idx)} res={m:.2f}")
        save_image_grid(FIG / f"{label}_top_images.png", vis["image_t"][idx], titles)
        residual_arrows(FIG / f"{label}_residual_arrows.png", vis, idx, f"{label}: residual directions")
        active = vis["hard_labels"][te, vis["hard_label_names"].index(label)] > 0.5
        scatter_failure(FIG / f"{label}_spatial_map.png", vis, mag, f"{label}: spatial residual map", active=active)
        rows.append({"failure_mode": label, "top_images": str(FIG / f"{label}_top_images.png"), "spatial_map": str(FIG / f"{label}_spatial_map.png"), "residual_arrows": str(FIG / f"{label}_residual_arrows.png")})
    return rows


def make_rule_alignment_figures(vis):
    te = vis["test_idx"]
    q42 = vis["q_v42_t"][te].float()
    q4 = vis["q_v4_t"][te].float()
    specs = {
        "v42_rule_10": q42[:, V42_BEST_RULE],
        "v42_group_10_13_12_1": q42[:, V42_GROUP0].mean(1),
        "v4_group_15_2_8_11": q4[:, V4_GROUP0].mean(1),
    }
    rows = []
    for name, score in specs.items():
        active = score >= torch.quantile(score, 0.9)
        idx = te[torch.argsort(score, descending=True)[:12]]
        save_image_grid(FIG / f"{name}_top_images.png", vis["image_t"][idx], [f"{name}\nscore={score[torch.argsort(score, descending=True)[i]].item():.3f}" for i in range(idx.numel())])
        scatter_failure(FIG / f"{name}_activation_map.png", vis, score, f"{name}: activation over position", active=active)
        residual_arrows(FIG / f"{name}_active_residual_arrows.png", vis, te[active], f"{name}: active residual directions")
        rows.append({"rule_or_group": name, "top_images": str(FIG / f"{name}_top_images.png"), "activation_map": str(FIG / f"{name}_activation_map.png"), "residual_arrows": str(FIG / f"{name}_active_residual_arrows.png")})
    return rows


def make_episode_story_figures(vis, max_episodes=4):
    te = vis["test_idx"]
    mag_all = vis["residual_t"].float().norm(dim=1)
    q42_group_all = vis["q_v42_t"][:, V42_GROUP0].mean(1).float()
    hard_idx = vis["hard_label_names"].index("high_residual_top10")
    hard = vis["hard_labels"][:, hard_idx] > 0.5
    candidates = te[hard[te]]
    if candidates.numel() == 0:
        candidates = te[torch.argsort(mag_all[te], descending=True)[:200]]
    episode_ids = vis["episode_id"]
    rows = []
    used = set()
    for sample_idx in candidates[torch.argsort(mag_all[candidates], descending=True)]:
        ep = int(episode_ids[sample_idx])
        if ep in used:
            continue
        episode_mask = episode_ids == ep
        ep_idx = torch.nonzero(episode_mask).flatten()
        if ep_idx.numel() < 6:
            continue
        ep_idx = ep_idx[torch.argsort(vis["timestep"][ep_idx])]
        top_pos = torch.nonzero(ep_idx == sample_idx).flatten()
        if top_pos.numel() == 0:
            center = torch.argmax(mag_all[ep_idx]).item()
        else:
            center = int(top_pos[0])
        lo, hi = max(0, center - 3), min(ep_idx.numel(), center + 5)
        seq = ep_idx[lo:hi]
        prefix = f"episode_{ep}_around_t{int(vis['timestep'][sample_idx])}"

        titles = [f"t={int(vis['timestep'][i])}\nres={mag_all[i].item():.2f}\nqG={q42_group_all[i].item():.2f}" for i in seq]
        save_image_grid(FIG / f"{prefix}_image_strip.png", vis["image_t"][seq], titles, ncols=min(8, seq.numel()))

        pos = vis["position_t"][ep_idx].float()
        tim = vis["timestep"][ep_idx].float()
        plt.figure(figsize=(5.8, 4.8))
        plt.scatter(pos[:, 0], pos[:, 1], c=mag_all[ep_idx], s=18, cmap="magma")
        plt.plot(pos[:, 0], pos[:, 1], color="gray", linewidth=0.8, alpha=0.5)
        sp = vis["position_t"][sample_idx]
        plt.scatter([sp[0]], [sp[1]], marker="x", s=80, c="cyan", linewidths=2, label="selected failure")
        plt.colorbar(label="residual magnitude")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{prefix}: trajectory colored by residual")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / f"{prefix}_trajectory.png", dpi=180)
        plt.close()

        plt.figure(figsize=(6.2, 3.2))
        plt.plot(tim.numpy(), mag_all[ep_idx].numpy(), label="residual magnitude", linewidth=1.6)
        plt.plot(tim.numpy(), q42_group_all[ep_idx].numpy(), label="V4.2 group activation", linewidth=1.6)
        plt.axvline(float(vis["timestep"][sample_idx]), color="red", linestyle="--", linewidth=1)
        plt.xlabel("timestep")
        plt.title(f"{prefix}: failure score and rule activation over time")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / f"{prefix}_timeseries.png", dpi=180)
        plt.close()

        rows.append(
            {
                "episode_id": ep,
                "selected_timestep": int(vis["timestep"][sample_idx]),
                "selected_residual_mag": mag_all[sample_idx].item(),
                "selected_v42_group_activation": q42_group_all[sample_idx].item(),
                "image_strip": str(FIG / f"{prefix}_image_strip.png"),
                "trajectory": str(FIG / f"{prefix}_trajectory.png"),
                "timeseries": str(FIG / f"{prefix}_timeseries.png"),
            }
        )
        used.add(ep)
        if len(rows) >= max_episodes:
            break
    return rows


def main():
    ensure_dirs()
    vis = build_visual_data()
    summary = failure_summary_rows(vis)
    overlap = rule_overlap_rows(vis)
    figure_rows = make_failure_mode_figures(vis)
    rule_figures = make_rule_alignment_figures(vis)
    episode_rows = make_episode_story_figures(vis)

    csv_write(TABLE / "failure_mode_summary.csv", summary)
    csv_write(TABLE / "rule_failure_overlap.csv", overlap)
    csv_write(TABLE / "failure_visual_index.csv", figure_rows)
    csv_write(TABLE / "rule_visual_index.csv", rule_figures)
    csv_write(TABLE / "episode_story_index.csv", episode_rows)

    top_overlap = sorted(overlap, key=lambda r: r["enrichment"], reverse=True)[:12]
    text = f"""# TwoRoom Baseline Failure Atlas And Rule Alignment

## What This Atlas Shows

This is a diagnostic atlas, not a new training run. It visualizes where the current TwoRoom representation/prediction proxy has systematic residual failures, then checks whether V4/V4.2 rules align with those failures.

The first version uses the existing residual labels and hard labels from the TwoRoom analysis cache:

- `high_residual_top10/top20`
- `action_error_top10/top20`
- `large_action_small_disp`

These are the current objective failure-mode proxies. A later version can replace or augment them with direct LeWM rollout prediction errors.

## Failure Mode Summary

{table_md(summary, ["failure_mode", "count", "rate", "residual_mag_mean", "residual_mag_global_mean", "delta_mag_mean", "v42_group_mean", "v42_group_global_mean", "v42_rule10_mean", "v42_rule10_global_mean"])}

## Rule-Failure Alignment

{table_md(top_overlap, ["rule_or_group", "failure_mode", "base_rate", "active_rate", "enrichment", "active_count"])}

## Failure Visual Index

{table_md(figure_rows, ["failure_mode", "top_images", "spatial_map", "residual_arrows"])}

## Rule Visual Index

{table_md(rule_figures, ["rule_or_group", "top_images", "activation_map", "residual_arrows"])}

## Episode Execution Stories

These panels show how a failure looks over time within visualized episodes: an image strip, a trajectory colored by residual magnitude, and a time series comparing residual magnitude with V4.2 group activation.

{table_md(episode_rows, ["episode_id", "selected_timestep", "selected_residual_mag", "selected_v42_group_activation", "image_strip", "trajectory", "timeseries"])}

## Current Read

- If V4.2 group/rule enrichment is high on failure modes, the rules are not only visually meaningful; they are aligned with systematic error regions.
- The episode story plots are the most useful figures for human inspection because they show what the failure looks like, where it occurs, and whether the rule activation rises around the same event.
- This atlas should be used before designing `c_conf`; it tells us which failure scores `c_conf` should predict.
"""
    write_report("README.md", text)


if __name__ == "__main__":
    main()
