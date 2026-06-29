from tworoom_failure_c_v1_common import *


def main():
    ensure_dirs()
    data, _, _ = build_aligned_dataset()
    data = make_labels(data)
    path = save_dataset(data)
    rows = []
    for label in [k for k in data["labels"] if k.endswith(("top5", "top10", "top20"))]:
        y = data["labels"][label]
        row = {
            "label": label,
            "threshold": data["thresholds"].get(label, ""),
            "overall_rate": y.mean().item(),
        }
        for split_name in ["head_train_idx", "head_val_idx", "head_test_idx"]:
            idx = data["split"][split_name]
            row[f"{split_name}_n"] = int(idx.numel())
            row[f"{split_name}_positive"] = int(y[idx].sum().item())
            row[f"{split_name}_rate"] = y[idx].mean().item()
        rows.append(row)
    corr = safe_corr(data["direct_mse"], data["official_window_mse"])
    overlap = ((data["labels"]["direct_mse_top10"] > 0.5) & (data["labels"]["official_window_mse_top10"] > 0.5)).float().mean().item()
    union = ((data["labels"]["direct_mse_top10"] > 0.5) | (data["labels"]["official_window_mse_top10"] > 0.5)).float().mean().item()
    csv_write(TABLE / "failure_label_stats.csv", rows)
    text = f"""# Failure Label 构造

## 数据集

- 输出：`{path}`
- 样本数：`{data['h'].shape[0]}`
- split：按 `episode_id` 重新划分 diagnostic split，`head_train/head_val/head_test = 70/15/15`。
- 主标签：`{data['primary_label']}`

阈值只在 `head_train` split 上计算，然后应用到 val/test，避免用全体样本泄漏。

## 标签统计

{table_md(rows, ["label", "threshold", "overall_rate", "head_train_idx_n", "head_train_idx_positive", "head_train_idx_rate", "head_val_idx_n", "head_val_idx_positive", "head_val_idx_rate", "head_test_idx_n", "head_test_idx_positive", "head_test_idx_rate"])}

## direct_mse 与 official_window_mse 关系

- Pearson：`{corr:.4f}`
- `direct_mse_top10` 与 `official_window_mse_top10` overlap：`{overlap:.4f}`
- top10 union rate：`{union:.4f}`

## 注意

SIGReg 不作为单样本 failure label。本实验预测的是 official predictor latent prediction error，不是官方 total training loss。
"""
    write_report("01_failure_labels.md", text)


if __name__ == "__main__":
    main()
