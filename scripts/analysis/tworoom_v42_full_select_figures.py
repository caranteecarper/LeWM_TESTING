from tworoom_v42_full_common import *
from tworoom_v51_common import (
    compactness,
    hard_enrichment,
    save_group_visuals,
    save_residual_arrows,
    save_top_images,
    save_unit_visuals,
    scatter_pos,
)


def save_named_group_visuals(fig_dir, vis, q_name, prefix, rules):
    te = vis["test_idx"]
    q = vis[q_name][te].float()
    score = q[:, rules].mean(1)
    top_local = torch.topk(score, k=min(200, score.numel())).indices
    top_samples = te[top_local]
    save_top_images(fig_dir / f"{prefix}_top_images.png", vis, top_samples[:8], prefix)
    scatter_pos(fig_dir / f"{prefix}_positions.png", vis["position_t"][te], color=score, title=f"{prefix} activation over position")
    save_residual_arrows(fig_dir / f"{prefix}_residual_direction.png", vis["position_t"][top_samples], vis["residual_t"][top_samples], f"{prefix} residual directions")
    enrich = hard_enrichment(vis, score)
    top_hard, top_enrich = max(enrich.items(), key=lambda kv: kv[1])
    return {
        "group_id": prefix,
        "rules": ",".join(map(str, rules)),
        "top_hard_subset": top_hard,
        "hard_enrichment": top_enrich,
        "mean_residual_mag": vis["residual_t"][top_samples].norm(dim=1).mean().item(),
        "spatial_compactness": compactness(vis["position_t"][te], score),
    }


def main():
    ensure_dirs()
    fig_dir = FULL_FIG / "recommended"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vis = build_visual_data()
    audit_rows = read_csv(FULL_TABLE / "v42_all_rule_audit.csv")
    group_rows = read_csv(FULL_TABLE / "v42_group_audit.csv")
    if audit_rows:
        best_rule = int(sorted(audit_rows, key=lambda r: maybe_float(r["top_hard_enrichment"]) * maybe_float(r["mean_residual_mag"]), reverse=True)[0]["rule"])
    else:
        best_rule = 10
    if group_rows:
        best_group = sorted(group_rows, key=lambda r: maybe_float(r["group_clarity_score"]), reverse=True)[0]
        best_rules = [int(x) for x in best_group["rules"].split(",")]
    else:
        best_rules = V42_GROUPS["group_0"]

    generated = []
    save_unit_visuals(fig_dir, vis, "q_v42_t", best_rule, f"v42_best_rule_{best_rule}")
    generated += [
        f"v42_best_rule_{best_rule}_top_images.png",
        f"v42_best_rule_{best_rule}_positions.png",
        f"v42_best_rule_{best_rule}_residual_direction.png",
    ]
    v42_group_prefix = f"v42_best_group_rules_{'-'.join(map(str, best_rules))}"
    save_named_group_visuals(fig_dir, vis, "q_v42_t", v42_group_prefix, best_rules)
    generated += [
        f"{v42_group_prefix}_top_images.png",
        f"{v42_group_prefix}_positions.png",
        f"{v42_group_prefix}_residual_direction.png",
    ]
    save_group_visuals(fig_dir, vis, 0, V4_GROUPS["group_0"])
    generated += [
        "v4_group_0_rules_15-2-8-11_top_images.png",
        "v4_group_0_rules_15-2-8-11_positions.png",
        "v4_group_0_rules_15-2-8-11_residual_direction.png",
    ]
    for r in [0, 12]:
        if r < vis["q_v41_t"].shape[1]:
            save_unit_visuals(fig_dir, vis, "q_v41_t", r, f"v41_sparse_rule_{r}")
            generated += [
                f"v41_sparse_rule_{r}_top_images.png",
                f"v41_sparse_rule_{r}_positions.png",
                f"v41_sparse_rule_{r}_residual_direction.png",
            ]

    rows = [{"figure": name, "path": str(fig_dir / name)} for name in generated if (fig_dir / name).exists()]
    csv_write(FULL_TABLE / "recommended_figures.csv", rows)
    text = f"""# Recommended V4.2 Full-Validation Figures

## Figure List

{table_md(rows, ["figure", "path"])}

## Answers

- Keep V4.2 best single-rule activation, top images, and residual-direction figures.
- Keep V4.2 best group figures and compare them directly with V4 group 0.
- Keep V4.1 sparse rule 0/12 figures as sparse white-box references.
- Use V4.2 as a main figure only if the final comparison accepts it as a replacement; otherwise present it as a compromise/control beside V4 group 0.
"""
    write_report("06_recommended_figures.md", text)


if __name__ == "__main__":
    main()
