from tworoom_v42_full_common import *


def main():
    ensure_dirs()
    vis = build_visual_data()
    q = q_test(vis, "q_v42_t")
    rb = random_baseline(vis)
    rs = residual_stats(vis)
    audit_rows = []
    for r in range(q.shape[1]):
        row = {"rule": r, **audit_score(vis, q[:, r])}
        add_audit_derived(row, rs, rb)
        audit_rows.append(row)
    select_rule_rows(audit_rows)
    group_defs = discover_v42_groups(vis, audit_rows)
    rows = group_rows_from_rules(vis, group_defs, "q_v42_t")
    csv_write(FULL_TABLE / "v42_group_audit.csv", rows)
    best = sorted(rows, key=lambda r: r["group_clarity_score"], reverse=True)[:12]
    group0 = [r for r in rows if r["group"] == "group_0"]
    group0_best = sorted(group0, key=lambda r: r["group_clarity_score"], reverse=True)[:1]
    best_group = best[0] if best else {}
    text = f"""# V4.2 Group Audit

## V4.2 Group 0

{table_md(group0_best, ["group", "rules", "activation_definition", "top_hard_subset", "top_hard_enrichment", "mean_residual_mag", "residual_mag_ratio_to_global_mean", "compactness_ratio_to_random", "group_clarity_score", "physical_interpretation_label"])}

## Best Group Definitions

{table_md(best, ["group", "rules", "activation_definition", "top_hard_subset", "top_hard_enrichment", "mean_residual_mag", "residual_mag_ratio_to_global_mean", "compactness_ratio_to_random", "group_clarity_score", "physical_interpretation_label"])}

## Answers

- Preserved V4.2 group 0 is `[10, 13, 12, 1]`.
- Best audited group: `{best_group.get("group", "n/a")}` with rules `{best_group.get("rules", "n/a")}` and activation `{best_group.get("activation_definition", "n/a")}`.
- V4.2 can be discussed through both strong single rules and rule groups. The final replacement decision depends on whether group-only retention and mask damage are at least as convincing as V4 group 0.
"""
    write_report("02_v42_group_audit.md", text)


if __name__ == "__main__":
    main()
