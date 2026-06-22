from tworoom_v42_full_common import *


def main():
    ensure_dirs()
    vis = build_visual_data()
    q = q_test(vis, "q_v42_t")
    rb = random_baseline(vis)
    rs = residual_stats(vis)
    rows = []
    for r in range(q.shape[1]):
        row = {"rule": r, **audit_score(vis, q[:, r])}
        add_audit_derived(row, rs, rb)
        rows.append(row)
    selected = select_rule_rows(rows)
    top = sorted(rows, key=lambda r: r["top_hard_enrichment"] * r["mean_residual_mag"], reverse=True)[:8]
    top_rule_text = ", ".join(str(int(r["rule"])) for r in top[:5])
    csv_write(FULL_TABLE / "v42_all_rule_audit.csv", rows)
    text = f"""# V4.2 All-Rule Audit

## Global Baselines

{table_md([{**rs, **rb}], list({**rs, **rb}.keys()))}

## Selected Rules

{table_md(selected, ["rule", "top_hard_subset", "top_hard_enrichment", "mean_residual_mag", "residual_mag_ratio_to_global_mean", "compactness_ratio_to_random", "hard_enrichment_minus_random", "selection_reason"])}

## All Rules

{table_md(rows, ["rule", "usage_mean", "usage_std", "active_top10_count", "top_hard_subset", "top_hard_enrichment", "mean_residual_mag", "residual_mag_percentile", "residual_mag_ratio_to_global_mean", "compactness_ratio_to_random", "hard_enrichment_minus_random", "selected_for_reporting", "selection_reason"])}

## Answers

- Strongest V4.2 single-rule candidates by hard-enriched residual magnitude: `{top_rule_text}`.
- The selected rules are compared against random compactness, random hard enrichment, and random residual magnitude baselines, so the audit is not based only on visually attractive examples.
- V4.2 has clearer hard-enriched single-rule candidates than soft V4 single rules, but replacement still depends on group-only retention and mask damage versus V4 group 0.
"""
    write_report("01_v42_all_rule_audit.md", text)


if __name__ == "__main__":
    main()
