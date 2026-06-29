# C(h,q,a) Failure Confidence Head 训练

## 设置

- 输入组合：`['action_only', 'h_action', 'q_action', 'h_q_action', 'h_q_group_action', 'h_shuffled_q_action', 'q_shuffled_action', 'z_action']`
- 模型：`['logistic_linear', 'additive_linear', 'small_mlp_64', 'small_mlp_128']`
- seeds：`0,1,2`
- loss：weighted BCE 多分类标签 + `0.1 * SmoothL1(log error regression)`
- early stopping：验证集 `official_window_mse_top10` AUPRC，patience=10
- 本阶段只训练 C head，不训练 LeWM，不训练 h/q，不训练 residual adapter。

## 验证集 Top 20

|feature_group|model_type|seed|epoch|val_primary_auprc|model_path|
|---|---|---|---|---|---|
|z_action|small_mlp_128|2|16|0.34279945492744446|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed2.pt|
|z_action|small_mlp_128|0|17|0.34120991826057434|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|0|19|0.340180903673172|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|2|22|0.3389442265033722|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed2.pt|
|z_action|small_mlp_128|2|21|0.338405579328537|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed2.pt|
|z_action|small_mlp_128|0|15|0.3381185233592987|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|0|16|0.3378308415412903|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|0|12|0.33688467741012573|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|0|20|0.3368733823299408|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|1|21|0.3365207314491272|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed1.pt|
|z_action|small_mlp_128|2|18|0.3364194333553314|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed2.pt|
|z_action|small_mlp_128|2|19|0.3361189663410187|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed2.pt|
|z_action|small_mlp_128|0|14|0.33588650822639465|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|0|18|0.3352203369140625|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|0|11|0.33513861894607544|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|2|15|0.33512091636657715|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed2.pt|
|z_action|small_mlp_128|0|22|0.3349149525165558|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|0|25|0.3348274827003479|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|0|10|0.334669291973114|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed0.pt|
|z_action|small_mlp_128|2|26|0.33457258343696594|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/z_action__small_mlp_128__seed2.pt|
