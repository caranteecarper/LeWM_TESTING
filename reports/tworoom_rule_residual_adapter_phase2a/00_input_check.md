# Phase 2A 输入检查

## 结果

|item|value|
|---|---|
|phase1_aligned_failure_dataset|True|
|phase1_best_c_model|True|
|aligned_samples_phase1|62048|
|valid_pred_target_samples|62048|
|base_pred_z_shape|(62048, 192)|
|target_z_shape|(62048, 192)|
|h_shape|(62048, 16)|
|q_shape|(62048, 16)|
|action_shape|(62048, 10)|
|z_shape|(62048, 192)|
|c_model_path|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/h_q_action__small_mlp_128__seed2.pt|
|official_predictor_recomputed|True|
|sigreg_used|False|
|head_train_idx|43562|
|head_val_idx|9327|
|head_test_idx|9159|

## 说明

- 本阶段沿用 Phase 1 的 `source_base_idx / episode_id + timestep` 对齐，不假设缓存天然同序。
- `base_pred_z / target_z` 来自官方 checkpoint 对 validation batch 的重新 forward 或缓存。
- SIGReg 不作为本阶段单样本 correction target；本阶段修正的是 official predictor latent prediction error。
