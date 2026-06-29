from tworoom_phase2a_common import *


def main():
    ensure_dirs()
    data = make_adapter_dataset()
    rows = []
    for split in ["head_train_idx", "head_val_idx", "head_test_idx"]:
        idx = data["split"][split]
        rows.append(
            {
                "split": split,
                "n": int(idx.numel()),
                "official_window_mse_top10_rate": data["labels"][PRIMARY_LABEL][idx].mean().item(),
                "direct_mse_top10_rate": data["labels"][DIRECT_LABEL][idx].mean().item(),
                "c_risk_mean": data["c_official_window_top10"][idx].mean().item(),
                "c_risk_p90": torch.quantile(data["c_official_window_top10"][idx], 0.9).item(),
                "group_activation_mean": data["group_10_13_12_1"][idx].mean().item(),
                "group_activation_p90": torch.quantile(data["group_10_13_12_1"][idx], 0.9).item(),
                "base_mse": ((data["base_pred_z"][idx] - data["target_z"][idx]) ** 2).mean().item(),
            }
        )
    csv_write(TABLE / "adapter_dataset_summary.csv", rows)
    text = f"""# Adapter Dataset

输出：`{ADAPTER_DATASET}`

{table_md(rows, ["split", "n", "official_window_mse_top10_rate", "direct_mse_top10_rate", "c_risk_mean", "c_risk_p90", "group_activation_mean", "group_activation_p90", "base_mse"])}

## 字段

包含 `h_t / q_v42_t / action_t / z_t / base_pred_z / target_z / target_delta_z / C risk / V4.2 group activation / failure labels`。

split 沿用 Phase 1 的 episode split，没有重新按样本随机划分。
"""
    write_report("01_adapter_dataset.md", text)


if __name__ == "__main__":
    main()
