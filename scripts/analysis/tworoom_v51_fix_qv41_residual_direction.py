from tworoom_v51_common import *


def main():
    ensure_dirs()
    vis = load_visual()
    fig_dir = V51_FIG / "qv41_rule_link_fixed"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rules = [0, 3, 7, 10, 12, 13]
    rows = []
    for r in rules:
        save_unit_visuals(fig_dir, vis, "q_v41_t", r, f"qv41_rule_{r}")
        row = top_rule_rows(vis, "q_v41_t", k=16)
        by_id = {int(x["unit_id"]): x for x in row}
        if r in by_id:
            rows.append(by_id[r])
        else:
            # Compute fallback if not in global top rows.
            q = vis["q_v41_t"][vis["test_idx"]].float()
            score = q[:, r]
            enrich = hard_enrichment(vis, score)
            top_hard, top_enrich = max(enrich.items(), key=lambda kv: kv[1])
            active = score >= torch.quantile(score, 0.9)
            rows.append(
                {
                    "unit_id": r,
                    "top_hard_subset": top_hard,
                    "hard_enrichment": top_enrich,
                    "mean_residual_mag": vis["residual_t"][vis["test_idx"]][active].norm(dim=1).mean().item(),
                    "spatial_compactness": compactness(vis["position_t"][vis["test_idx"]], score),
                    "coverage_top10": active.float().mean().item(),
                }
            )
    text = f"""# V5.1 q_v41 Residual Direction Fix

Residual direction figures are regenerated with true position coordinates and normalized residual arrows over the top 200 activated samples. Empty-arrow behavior from V5 was caused by the earlier coarse binned quiver path; V5.1 uses direct per-sample arrows.

{table_md(rows, ["unit_id", "top_hard_subset", "hard_enrichment", "mean_residual_mag", "spatial_compactness", "coverage_top10"])}

## Figures

Fixed q_v41 figures are saved under `{fig_dir}`.
Each rule has:

- top images
- activation over position
- residual direction over top samples
"""
    write_report("02_qv41_residual_direction_fix.md", text)


if __name__ == "__main__":
    main()

