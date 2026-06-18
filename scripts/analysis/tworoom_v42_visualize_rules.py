from tworoom_v42_common import *


def main():
    ensure_dirs()
    vis = load_visual()
    key = torch.load(V42_MODEL_DIR / "summary.pt", map_location="cpu")["best_key"]
    model, obj = load_sharp_model(V42_MODEL_DIR / f"{key}.pt")
    q = q_from_model(model, obj, vis, hard_topk=bool(obj["meta"].get("top_k")))
    vis = dict(vis)
    vis["q_v42_t"] = q
    fig_dir = V42_FIG / "rule_visualization"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Audit all rules on visual subset.
    rows = []
    te = vis["test_idx"]
    for r in range(q.shape[1]):
        score = q[te, r]
        from tworoom_v52_common import audit_score

        row = {"rule": r, **audit_score(vis, score)}
        rows.append(row)
    top = sorted(rows, key=lambda r: r["top_hard_enrichment"] * r["mean_residual_mag"], reverse=True)[:8]
    for row in top[:6]:
        save_unit_visuals(fig_dir, vis, "q_v42_t", int(row["rule"]), f"v42_rule_{row['rule']}")
    # Groups by top coactivation.
    group_rows = []
    top_rules = [int(r["rule"]) for r in top[:6]]
    if len(top_rules) >= 4:
        group_rows.append(save_group_visuals(fig_dir, {**vis, "q_v4_t": q}, 0, top_rules[:4]))
    csv_write(V42_TABLE / "v42_rule_audit.csv", rows)
    text = f"""# V4.2 Rule Visualization

Best V4.2 model: `{key}`.

## Top Rules

{table_md(top, ["rule", "top_hard_subset", "top_hard_enrichment", "mean_residual_mag", "spatial_compactness", "active_top10_count"])}

## Groups

{table_md(group_rows, ["group_id", "rules", "top_hard_subset", "hard_enrichment", "mean_residual_mag", "spatial_compactness"])}

Figures saved under `{fig_dir}`.
"""
    write_report("03_v42_rule_visualization.md", text)


if __name__ == "__main__":
    main()

