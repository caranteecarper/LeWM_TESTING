from tworoom_phase2a_common import *


def main():
    ensure_dirs()
    data = load_adapter_dataset()
    gates, meta = build_gates(data)
    test = data["split"]["head_test_idx"]
    hard = data["labels"][PRIMARY_LABEL] > 0.5
    group_active = data["group_10_13_12_1"] >= torch.quantile(data["group_10_13_12_1"][test], 0.9)
    rows = []
    for name, g in gates.items():
        gt = g[test].float()
        rows.append(
            {
                "gate": name,
                "mean": gt.mean().item(),
                "std": gt.std().item(),
                "p90": torch.quantile(gt, 0.9).item(),
                "p95": torch.quantile(gt, 0.95).item(),
                "high_risk_mean": g[test][hard[test]].mean().item(),
                "normal_mean": g[test][~hard[test]].mean().item(),
                "group_active_mean": g[test][group_active[test]].mean().item(),
                "group_inactive_mean": g[test][~group_active[test]].mean().item(),
            }
        )
    csv_write(TABLE / "gate_stats.csv", rows)
    text = f"""# Gate Design

## c_sigmoid 参数

- beta：`{meta['c_sigmoid_beta']}`
- tau：`{meta['c_sigmoid_tau']}`

## Gate 统计

{table_md(rows, ["gate", "mean", "std", "p90", "p95", "high_risk_mean", "normal_mean", "group_active_mean", "group_inactive_mean"])}

主线 gate 是 `c_soft_gate / c_sigmoid_gate`；`oracle_gate` 只作为 upper bound，`random_gate` 是控制实验。
"""
    write_report("02_gate_design.md", text)


if __name__ == "__main__":
    main()
