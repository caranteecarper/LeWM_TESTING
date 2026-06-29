# 输入对齐检查

## 对齐方式

本阶段不假设不同缓存天然同序。`official predictor failure atlas` 在生成时已经用官方 validation batch 的 `episode_id + timestep` 映射到 `best_h_dataset` 的 `source_base_idx`。本检查继续使用该 `eval_base_idx/source_base_idx` 作为 join key，把 official predictor per-sample error 与 `h_t / q_v42_t / action_t / z_t` 对齐。

## 检查结果

|item|value|
|---|---|
|official_predictor_failure_atlas_exists|True|
|best_h_dataset_exists|True|
|aligned_samples|62048|
|unmatched_samples_from_direct_atlas|11032|
|h_t_shape|(720809, 16)|
|q_v42_shape|(720809, 16)|
|action_shape|(720809, 10)|
|z_t_shape|(720809, 192)|
|position_shape|(720809, 2)|
|episode_timestep_present|True|
|direct_mse_present|True|
|direct_cosine_error_present|True|
|official_window_mse_present|True|
|sigreg_used_as_single_sample_label|False|

## 结论

- 对齐成功样本数：`62048`
- h_t shape：`(720809, 16)`，对齐后可用于 C head。
- q_v42 shape：`(720809, 16)`，对齐后可用于 C head。
- action shape：`(720809, 10)`，对齐后可用于 C head。
- official predictor per-sample error 包含 `direct_mse / direct_cosine_error / official_window_mse`。
- SIGReg 不作为单样本 failure label；本实验预测的是 official predictor latent prediction error。
