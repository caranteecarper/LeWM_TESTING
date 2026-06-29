# Adapter Dataset

输出：`/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/adapter_dataset.pt`

|split|n|official_window_mse_top10_rate|direct_mse_top10_rate|c_risk_mean|c_risk_p90|group_activation_mean|group_activation_p90|base_mse|
|---|---|---|---|---|---|---|---|---|
|head_train_idx|43562|0.10001836717128754|0.10001836717128754|0.3569580614566803|0.7330424189567566|0.059840355068445206|0.13951991498470306|0.0012379636755213141|
|head_val_idx|9327|0.09928165376186371|0.09735177457332611|0.3492838144302368|0.7233342528343201|0.058830346912145615|0.13548408448696136|0.0012628094991669059|
|head_test_idx|9159|0.1040506586432457|0.0989190936088562|0.3563029170036316|0.7313697934150696|0.0595141164958477|0.14453421533107758|0.0012896590633317828|

## 字段

包含 `h_t / q_v42_t / action_t / z_t / base_pred_z / target_z / target_delta_z / C risk / V4.2 group activation / failure labels`。

split 沿用 Phase 1 的 episode split，没有重新按样本随机划分。
