from tworoom_v52_common import *


def main():
    ensure_dirs()
    vis = load_visual()
    action = vis["action"].float() if "action" in vis else vis["z_t"][:, :0]
    q = vis["q_v4_t"].float()
    h = vis["h_t"].float()
    inputs = {
        "action_only": action,
        "full_q_v4_action": torch.cat([q, action], 1),
        "group0_q_v4_action": torch.cat([q[:, GROUPS["group_0"]], action], 1),
        "top3_rules_q_v4_action": torch.cat([q[:, [15, 2, 14]], action], 1),
        "top_groups_q_v4_action": torch.cat([q[:, sorted(set(GROUPS["group_0"] + GROUPS["group_3"]))], action], 1),
        "h_action": torch.cat([h, action], 1),
        "h_full_q_v4_action": torch.cat([h, q, action], 1),
    }
    rows = []
    for name, x in inputs.items():
        r, _ = eval_input(vis, x, name)
        rows += r
    by = {(r["input"], r["subset"]): r["res_overall_mse"] for r in rows}
    action_mse = by[("action_only", "all_test")]
    full_mse = by[("full_q_v4_action", "all_test")]
    retention = []
    for name in ["group0_q_v4_action", "top3_rules_q_v4_action", "top_groups_q_v4_action"]:
        retention.append({"input": name, "retention_vs_full_q": (action_mse - by[(name, "all_test")]) / max(action_mse - full_mse, 1e-8)})
    csv_write(V52_TABLE / "v4_group_only_performance.csv", rows)
    text = f"""# V4 Group-Only Performance

## Retention

{table_md(retention, ["input", "retention_vs_full_q"])}

## Metrics

{table_md(rows, ["input", "subset", "res_overall_mse", "res_x_r2", "res_y_r2", "delta_overall_mse", "hard_acc", "hard_f1"])}

## Answers

Group-only inputs are diagnostic explanation inputs. If group0 retention is below 0.5, group 0 should be treated as an explanatory unit rather than a replacement input.
"""
    write_report("03_v4_group_only_performance.md", text)


if __name__ == "__main__":
    main()

