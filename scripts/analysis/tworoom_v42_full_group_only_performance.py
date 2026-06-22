from tworoom_v42_full_common import *


SUBSETS = [
    "all_test",
    "action_error_top20",
    "action_error_top10",
    "high_residual_top20",
    "high_residual_top10",
    "large_action_small_disp",
]


def main():
    ensure_dirs()
    vis = build_visual_data()
    action = vis["action_t"].float()
    h = vis["h_t"].float()
    q42 = vis["q_v42_t"].float()
    q4 = vis["q_v4_t"].float()
    q41 = vis["q_v41_t"].float()
    top_groups = sorted(set(V42_GROUPS["group_0"] + V42_GROUPS["best_rules"]))
    inputs = {
        "action_only": action,
        "full_q_v42_action": torch.cat([q42, action], 1),
        "group0_q_v42_action": torch.cat([q42[:, V42_GROUPS["group_0"]], action], 1),
        "top3_rules_q_v42_action": torch.cat([q42[:, V42_GROUPS["top3_rules"]], action], 1),
        "top4_rules_q_v42_action": torch.cat([q42[:, V42_GROUPS["best_rules"]], action], 1),
        "top_groups_q_v42_action": torch.cat([q42[:, top_groups], action], 1),
        "h_action": torch.cat([h, action], 1),
        "h_full_q_v42_action": torch.cat([h, q42, action], 1),
        "q_v4_action": torch.cat([q4, action], 1),
        "q_v41_action": torch.cat([q41, action], 1),
    }
    rows = []
    for name, x in inputs.items():
        r, _ = eval_input(vis, x, name, subset_names=SUBSETS)
        rows += r
    by = {(r["input"], r["subset"]): r["res_overall_mse"] for r in rows}
    action_mse = by[("action_only", "all_test")]
    full_mse = by[("full_q_v42_action", "all_test")]
    retention_rows = []
    for name in ["group0_q_v42_action", "top3_rules_q_v42_action", "top4_rules_q_v42_action", "top_groups_q_v42_action"]:
        retention_rows.append({"input": name, "retention_vs_full_q_v42": retention(action_mse, full_mse, by[(name, "all_test")])})
    csv_write(FULL_TABLE / "v42_group_only_performance.csv", rows)
    csv_write(FULL_TABLE / "v42_group_only_retention.csv", retention_rows)
    text = f"""# V4.2 Group-Only Performance

## Retention

{table_md(retention_rows, ["input", "retention_vs_full_q_v42"])}

## Metrics

{table_md(rows, ["input", "subset", "res_overall_mse", "res_x_r2", "res_y_r2", "delta_overall_mse", "hard_acc", "hard_f1"])}

## Answers

- Group-only and top-rule inputs are diagnostic compressed explanations, not new LeWM inputs.
- If group0/top-rule retention is below 0.75, V4.2 should not be treated as a compressed replacement input even if its individual rules are clearer.
"""
    write_report("03_v42_group_only_performance.md", text)


if __name__ == "__main__":
    main()
