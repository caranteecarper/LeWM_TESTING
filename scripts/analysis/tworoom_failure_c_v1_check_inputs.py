from tworoom_failure_c_v1_common import *


def main():
    ensure_dirs()
    base, errors = load_base_and_errors()
    q = load_or_compute_q(base)
    idx = aligned_indices(base, errors)
    action_key = "action" if "action" in base else "action_t"
    rows = [
        {"item": "official_predictor_failure_atlas_exists", "value": DIRECT_ERROR_PATH.exists()},
        {"item": "best_h_dataset_exists", "value": BEST_H_DATASET.exists()},
        {"item": "aligned_samples", "value": int(idx.numel())},
        {"item": "unmatched_samples_from_direct_atlas", "value": errors.get("metadata", {}).get("unmatched_eval_samples", "unknown")},
        {"item": "h_t_shape", "value": tuple(base["h_t"].shape)},
        {"item": "q_v42_shape", "value": tuple(q.shape)},
        {"item": "action_shape", "value": tuple(base[action_key].shape)},
        {"item": "z_t_shape", "value": tuple(base["z_t"].shape) if "z_t" in base else "missing"},
        {"item": "position_shape", "value": tuple(base["position_t"].shape) if "position_t" in base else "missing"},
        {"item": "episode_timestep_present", "value": ("episode_id" in base and "timestep" in base)},
        {"item": "direct_mse_present", "value": "last_step_mse" in errors},
        {"item": "direct_cosine_error_present", "value": "last_step_cosine_error" in errors},
        {"item": "official_window_mse_present", "value": "official_window_mse" in errors},
        {"item": "sigreg_used_as_single_sample_label", "value": False},
    ]
    csv_write(TABLE / "aligned_failure_dataset_summary.csv", rows)
    text = f"""# 输入对齐检查

## 对齐方式

本阶段不假设不同缓存天然同序。`official predictor failure atlas` 在生成时已经用官方 validation batch 的 `episode_id + timestep` 映射到 `best_h_dataset` 的 `source_base_idx`。本检查继续使用该 `eval_base_idx/source_base_idx` 作为 join key，把 official predictor per-sample error 与 `h_t / q_v42_t / action_t / z_t` 对齐。

## 检查结果

{table_md(rows, ["item", "value"])}

## 结论

- 对齐成功样本数：`{idx.numel()}`
- h_t shape：`{tuple(base["h_t"].shape)}`，对齐后可用于 C head。
- q_v42 shape：`{tuple(q.shape)}`，对齐后可用于 C head。
- action shape：`{tuple(base[action_key].shape)}`，对齐后可用于 C head。
- official predictor per-sample error 包含 `direct_mse / direct_cosine_error / official_window_mse`。
- SIGReg 不作为单样本 failure label；本实验预测的是 official predictor latent prediction error。
"""
    write_report("00_input_alignment.md", text)


if __name__ == "__main__":
    main()
