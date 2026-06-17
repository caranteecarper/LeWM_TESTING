import torch

from tworoom_v5_common import *


def main():
    ensure_dirs()
    fig_dir = V5_FIG / "rule_visual_link"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vis = load_visual()
    te = vis["test_idx"]
    q = vis["q_v41_t"]
    rows = []
    hard = vis["hard_labels"][te]
    for r in range(q.shape[1]):
        active = q[te, r] >= torch.quantile(q[te, r], 0.9)
        usage = active.float().mean().item()
        enrich = []
        for i, name in enumerate(vis["hard_label_names"]):
            base = hard[:, i].mean().item()
            act = hard[active, i].mean().item() if active.any() else 0.0
            enrich.append((name, act / max(base, 1e-8)))
        top_hard, top_enrich = max(enrich, key=lambda kv: kv[1])
        res_mag = vis["residual_t"][te][active].norm(dim=1).mean().item() if active.any() else 0.0
        rows.append({"rule": r, "usage_top10": usage, "top_hard_subset": top_hard, "hard_enrichment": top_enrich, "active_residual_mag": res_mag})
    top_rules = sorted(rows, key=lambda r: r["hard_enrichment"] * max(r["usage_top10"], 1e-6), reverse=True)[:6]
    for row in top_rules:
        r = row["rule"]
        score = q[te, r]
        top_local = torch.topk(score, k=min(8, score.numel())).indices
        sample_idx = te[top_local]
        imgs = [[vis["image_t"][i].float() / 255.0] for i in sample_idx]
        save_image_grid(fig_dir / f"rule_{r}_top_images.png", imgs, [f"q{r} top images"], row_titles=[f"#{int(i)}" for i in sample_idx], figsize_scale=2.0)
        scatter_pos(fig_dir / f"rule_{r}_positions.png", vis["position_t"][te], color=q[te, r], title=f"q_v41 rule {r} activation over position")
        quiver_residual(fig_dir / f"rule_{r}_residual_direction.png", vis["position_t"][sample_idx], vis["residual_t"][sample_idx], title=f"rule {r} top-sample residual directions", bins=8)
    text = f"""# V5 Rule Visual Link

This report links V4.1 q rules to images, true spatial positions, residual directions, and hard-subset enrichment.

## Top Linked Rules

{table_md(top_rules, ["rule", "usage_top10", "top_hard_subset", "hard_enrichment", "active_residual_mag"])}

## Figure Pattern

For each top rule, V5 saves:

- representative high-activation sample images
- spatial activation scatter
- residual direction map over top active samples

Output directory: `{fig_dir}`.

## Naming Caution

Rules are named conservatively as position-related, residual-correction, hard-motion correction, or state-partition rules. V5 does not name rules as walls, doors, or collisions without stronger evidence.
"""
    write_report("04_rule_visual_link.md", text)


if __name__ == "__main__":
    main()

