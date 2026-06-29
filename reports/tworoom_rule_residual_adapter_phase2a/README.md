# TwoRoom Phase 2A Rule-Conditioned Residual Adapter

## 目的

验证规则分支是否能修正官方 LeWM predictor 的真实 latent prediction error。

## 约束

- 未修改 LeWM encoder / predictor / train.py / loss / module.py / environment。
- 未重新训练官方 LeWM、h extractor、V4.2 q、C failure head。
- 本阶段只训练外接 residual adapter：`final_pred_z = base_pred_z + gate * R(input)`。
- 本阶段不是因果证明，也不是完整 LeWM 替换。

## C gate 来源

`/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_confidence_head_v1/models/h_q_action__small_mlp_128__seed2.pt`

## 关键结果

|model|subset|base_mse|final_mse|relative_improvement|gate_mean|correction_norm|
|---|---|---|---|---|---|---|
|full_q_action|small_mlp_128|c_sigmoid_gate|hard_weighted|seed0|official_window_mse_top10|0.006026675924658775|0.005977303255349398|0.008192355110279654|0.8337525129318237|0.16876953840255737|
|full_q_action|small_mlp_128|c_sigmoid_gate|hard_weighted|seed0|direct_mse_top10|0.007103790994733572|0.007045791484415531|0.008164585692489975|0.8672284483909607|0.16766496002674103|
|full_q_action|small_mlp_128|c_sigmoid_gate|hard_weighted|seed0|all_test|0.0012896590633317828|0.0012573791900649667|0.025029772739643588|0.5073139071464539|0.25308170914649963|
|single_group_action|small_mlp_64|always_on|hard_weighted|seed2|V4.2_group_active_top10|0.0024085226468741894|0.0023605774622410536|0.019906470339965312|1.0|0.10808919370174408|
|shuffled_q_action|small_mlp_128|always_on|hard_weighted|seed0|official_window_mse_top10|0.006026675924658775|0.005976065061986446|0.008397807233212807|1.0|0.11504718661308289|
|single_group_action|small_mlp_128|random_gate|hard_weighted|seed2|official_window_mse_top10|0.006026675924658775|0.005992053542286158|0.005744855506657762|0.3583071827888489|0.22497643530368805|
|single_group_action|small_mlp_128|always_on|hard_weighted|seed2|official_window_mse_top10|0.006026675924658775|0.0059664552100002766|0.009992359869907623|1.0|0.12120094895362854|
|group_only_action|linear_adapter|oracle_gate|hard_weighted|seed0|official_window_mse_top10|0.006026675924658775|0.005991529673337936|0.005831780530463628|1.0|0.11174608767032623|

## 判断

`Partial`

Go 要求：full_q + C gate 在 official_window_mse_top10 上稳定改善，direct_mse_top10 不明显变差，all_test 不明显恶化，shuffled-q/random-gate 弱于真实 q/C gate。

## 下一步

如果 Go，进入 Phase 2B：

- 自动发现多个 q rule groups；
- 每个 group 做 audit；
- multi-group gated residual adapter；
- group-specific intervention。

如果 Partial，先继续调 gate / adapter，不做多组发现。
