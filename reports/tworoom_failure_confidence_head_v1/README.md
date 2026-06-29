# TwoRoom Failure Confidence Head V1

## 本阶段目的

训练一个外接诊断模块：

```text
c_{t,a} = C(h_t, q_t, a_t)
```

它预测的是官方 TwoRoom LeWM predictor 在当前 transition 上是否会出现高 latent prediction error。本阶段不改 LeWM encoder / predictor / train.py / loss / module.py / environment，也不训练 residual adapter。

## 使用的数据

- 官方 checkpoint：`/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt`
- official predictor failure labels：来自 `tworoom_direct_predictor_failure_atlas`
- 主标签：`official_window_mse_top10`
- 使用 direct_mse：是
- 使用 official_window_mse：是
- SIGReg：不作为单样本 failure label。SIGReg 是 batch/distribution regularizer，不适合作为 transition-level failure target。

## 关键结果

|feature_group|model_type|seed|base_rate|AUROC|AUPRC|precision@top10_predicted_risk|recall@top10_label|precision@10/base_rate|AUPRC/base_rate|
|---|---|---|---|---|---|---|---|---|---|
|h_q_action|small_mlp_128|2|0.1040506586432457|0.7972264289855957|0.32299110293388367|0.34934496879577637|0.3441762854144806|3.357450816275574|3.1041716327938897|
|h_action|small_mlp_128|2|0.1040506586432457|0.792782723903656|0.31013017892837524|0.3373362421989441|0.34207764952780695|3.242038508910888|2.980569109050103|
|q_action|small_mlp_128|1|0.1040506586432457|0.783268392086029|0.2953225076198578|0.33515283465385437|0.3315844700944386|3.221054426988102|2.8382569747339916|
|q_shuffled_action|logistic_linear|0|0.1040506586432457|0.5776466131210327|0.1409037858247757|0.1713973730802536|0.1720881427072403|1.6472492852536078|1.3541844680473079|
|h_shuffled_q_action|small_mlp_64|2|0.1040506586432457|0.7841760516166687|0.2970030903816223|0.3264192044734955|0.323189926547744|3.1371180992969574|2.854408556892896|
|z_action|small_mlp_128|0|0.1040506586432457|0.7948970794677734|0.3320993185043335|0.3569869101047516|0.35257082896117525|3.430895246215963|3.1917079894995095|
|h_q_group_action|small_mlp_128|1|0.1040506586432457|0.7971276640892029|0.3139319121837616|0.35480350255966187|0.3483735571878279|3.409911164293177|3.017106439086823|

## 当前判断

`Go`

- Go：进入第二阶段，训练 rule-conditioned residual adapter。
- Partial：继续优化 C 或 failure label / 输入，但暂不接 adapter。
- No-Go：h/q/action 无法可靠预测 official predictor failure，此路线暂时降级为 post-hoc diagnostics。

## 对第二阶段的建议

如果进入第二阶段，建议 gate 使用 `C(h,q,a)` 输出的 calibrated risk，adapter 输入优先从 `h_q_action` 开始，并保留 `h_action` 和 shuffled-q 对照；不要直接宣称因果修复，必须做 masking/intervention 或 adapter ablation。
