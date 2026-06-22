from tworoom_v42_full_common import *


def pick(rows, **conds):
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            return row
    return {}


def main():
    ensure_dirs()
    perf = read_csv(FULL_TABLE / "v42_group_only_performance.csv")
    retention_rows = read_csv(FULL_TABLE / "v42_group_only_retention.csv")
    group_rows = read_csv(FULL_TABLE / "v42_group_audit.csv")
    mask_rows = read_csv(FULL_TABLE / "v42_group_mask_ablation.csv")
    v4_perf = read_csv(REPO_ROOT / "outputs" / "tworoom_rule_group_v52" / "tables" / "v4_group_only_performance.csv")
    v4_group = read_csv(REPO_ROOT / "outputs" / "tworoom_rule_group_v52" / "tables" / "v4_group_audit.csv")
    v4_mask = read_csv(REPO_ROOT / "outputs" / "tworoom_rule_group_v52" / "tables" / "v4_group_mask_ablation.csv")

    q4 = pick(perf, input="q_v4_action", subset="all_test")
    q41 = pick(perf, input="q_v41_action", subset="all_test")
    q42 = pick(perf, input="full_q_v42_action", subset="all_test")
    v42_group0_ret = pick(retention_rows, input="group0_q_v42_action")
    v42_top_ret = pick(retention_rows, input="top_groups_q_v42_action")
    v4_group0_ret = {"retention_vs_full_q": "n/a"}
    if v4_perf:
        # Retention values are embedded in the V4 report, so recompute from all_test rows.
        v4_action = pick(v4_perf, input="action_only", subset="all_test")
        v4_full = pick(v4_perf, input="full_q_v4_action", subset="all_test")
        v4_g0 = pick(v4_perf, input="group0_q_v4_action", subset="all_test")
        if v4_action and v4_full and v4_g0:
            v4_group0_ret = {
                "retention_vs_full_q": (
                    maybe_float(v4_action["res_overall_mse"]) - maybe_float(v4_g0["res_overall_mse"])
                )
                / max(maybe_float(v4_action["res_overall_mse"]) - maybe_float(v4_full["res_overall_mse"]), 1e-8)
            }

    best_v42_group = sorted(group_rows, key=lambda r: maybe_float(r.get("group_clarity_score")), reverse=True)[0]
    best_v4_group = sorted(v4_group, key=lambda r: maybe_float(r.get("group_clarity_score")), reverse=True)[0] if v4_group else {}
    v42_mask_g0 = pick(mask_rows, mask="mask_group0", subset="action_error_top10")
    v42_mask_rand = pick(mask_rows, mask="mask_random_same_size", subset="action_error_top10")
    v4_mask_g0 = pick(v4_mask, mask="mask_group0", subset="action_error_top10")
    v4_mask_rand = pick(v4_mask, mask="mask_random_same_size", subset="action_error_top10")

    rows = [
        {
            "criterion": "full q all_test residual_mse",
            "v4": q4.get("res_overall_mse", "n/a"),
            "v41": q41.get("res_overall_mse", "n/a"),
            "v42": q42.get("res_overall_mse", "n/a"),
            "replacement_condition": "v42 clearly better than v4",
            "v42_pass": maybe_float(q42.get("res_overall_mse")) < maybe_float(q4.get("res_overall_mse")),
        },
        {
            "criterion": "group0 retention",
            "v4": v4_group0_ret.get("retention_vs_full_q", "n/a"),
            "v41": "n/a",
            "v42": v42_group0_ret.get("retention_vs_full_q_v42", "n/a"),
            "replacement_condition": "v42 >= v4 group0 retention",
            "v42_pass": maybe_float(v42_group0_ret.get("retention_vs_full_q_v42")) >= maybe_float(v4_group0_ret.get("retention_vs_full_q")),
        },
        {
            "criterion": "best group hard enrichment",
            "v4": best_v4_group.get("top_hard_enrichment", "n/a"),
            "v41": "n/a",
            "v42": best_v42_group.get("top_hard_enrichment", "n/a"),
            "replacement_condition": "> 2",
            "v42_pass": maybe_float(best_v42_group.get("top_hard_enrichment")) > 2.0,
        },
        {
            "criterion": "best group residual ratio",
            "v4": best_v4_group.get("residual_mag_ratio_to_global_mean", "n/a"),
            "v41": "n/a",
            "v42": best_v42_group.get("residual_mag_ratio_to_global_mean", "n/a"),
            "replacement_condition": "> 2",
            "v42_pass": maybe_float(best_v42_group.get("residual_mag_ratio_to_global_mean")) > 2.0,
        },
        {
            "criterion": "best group compactness ratio",
            "v4": best_v4_group.get("compactness_ratio_to_random", "n/a"),
            "v41": "n/a",
            "v42": best_v42_group.get("compactness_ratio_to_random", "n/a"),
            "replacement_condition": "< 0.5",
            "v42_pass": maybe_float(best_v42_group.get("compactness_ratio_to_random")) < 0.5,
        },
        {
            "criterion": "group0 mask damage vs random on action_error_top10",
            "v4": maybe_float(v4_mask_g0.get("residual_mse_increase")) - maybe_float(v4_mask_rand.get("residual_mse_increase")),
            "v41": "n/a",
            "v42": maybe_float(v42_mask_g0.get("residual_mse_increase")) - maybe_float(v42_mask_rand.get("residual_mse_increase")),
            "replacement_condition": "v42 positive and comparable to v4",
            "v42_pass": maybe_float(v42_mask_g0.get("residual_mse_increase")) > maybe_float(v42_mask_rand.get("residual_mse_increase")),
        },
        {
            "criterion": "top groups retention",
            "v4": "n/a",
            "v41": "n/a",
            "v42": v42_top_ret.get("retention_vs_full_q_v42", "n/a"),
            "replacement_condition": ">= 0.75",
            "v42_pass": maybe_float(v42_top_ret.get("retention_vs_full_q_v42")) >= 0.75,
        },
    ]
    csv_write(FULL_TABLE / "v42_vs_v4_final_comparison.csv", rows)
    pass_count = sum(1 for r in rows if str(r["v42_pass"]) == "True")
    decision = "V4 remains main; V4.2 is a compromise/control" if pass_count < 6 else "V4.2 can replace V4"
    text = f"""# V4.2 vs V4 / V4.1 Final Comparison

## Replacement Criteria

{table_md(rows, ["criterion", "v4", "v41", "v42", "replacement_condition", "v42_pass"])}

## Decision

{decision}.

## Answers

- V4.2 can replace V4 only if it passes performance, active-rule, group audit, group-only retention, and mask-damage criteria together.
- If V4.2 fails retention or does not clearly beat V4/V4.1, keep V4 full q as the main rule representation and use V4.2 as a compromise diagnostic.
"""
    write_report("05_v42_vs_v4_final_comparison.md", text)


if __name__ == "__main__":
    main()
