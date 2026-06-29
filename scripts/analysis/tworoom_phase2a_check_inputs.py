from tworoom_phase2a_common import *


def main():
    ensure_dirs()
    phase1 = load_phase1()
    pred_cache, recomputed = compute_pred_target_cache()
    valid = pred_cache["valid_pred_target"]
    rows = [
        {"item": "phase1_aligned_failure_dataset", "value": PHASE1_DATASET.exists()},
        {"item": "phase1_best_c_model", "value": C_MODEL_PATH.exists()},
        {"item": "aligned_samples_phase1", "value": int(phase1["h"].shape[0])},
        {"item": "valid_pred_target_samples", "value": int(valid.sum().item())},
        {"item": "base_pred_z_shape", "value": tuple(pred_cache["base_pred_z"][valid].shape)},
        {"item": "target_z_shape", "value": tuple(pred_cache["target_z"][valid].shape)},
        {"item": "h_shape", "value": tuple(phase1["h"].shape)},
        {"item": "q_shape", "value": tuple(phase1["q"].shape)},
        {"item": "action_shape", "value": tuple(phase1["action"].shape)},
        {"item": "z_shape", "value": tuple(phase1["z"].shape) if "z" in phase1 else "missing"},
        {"item": "c_model_path", "value": str(C_MODEL_PATH)},
        {"item": "official_predictor_recomputed", "value": recomputed},
        {"item": "sigreg_used", "value": False},
    ]
    for split in ["head_train_idx", "head_val_idx", "head_test_idx"]:
        rows.append({"item": split, "value": int(phase1["split"][split].numel())})
    csv_write(TABLE / "input_alignment_summary.csv", rows)
    text = f"""# Phase 2A 输入检查

## 结果

{table_md(rows, ["item", "value"])}

## 说明

- 本阶段沿用 Phase 1 的 `source_base_idx / episode_id + timestep` 对齐，不假设缓存天然同序。
- `base_pred_z / target_z` 来自官方 checkpoint 对 validation batch 的重新 forward 或缓存。
- SIGReg 不作为本阶段单样本 correction target；本阶段修正的是 official predictor latent prediction error。
"""
    write_report("00_input_check.md", text)


if __name__ == "__main__":
    main()
