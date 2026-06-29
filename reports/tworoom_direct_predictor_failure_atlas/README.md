# TwoRoom 官方 Predictor 真实失败样本分析

## 目的

这个分析直接使用已经训练好的官方 TwoRoom LeWM baseline predictor，不是之前的 residual proxy 分析。

计算路径是：

```text
official validation batch pixels/action from train.py data path
official model.encode(batch) -> emb/action embeddings
official model.predict(ctx_emb, ctx_act_emb)
last predictor token -> pred_z[t+1]
compare with official encoded target emb[:, -1]
```

其中 `official_window_mse` 更接近 `train.py` 里的官方训练 loss 定义：把 predictor 的多个输出和 shifted targets `[t-1, t, t+1]` 做平均误差。

## 元信息

- checkpoint: `/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt`
- eval source: `official validation split from train.py random_split seed`
- joined eval samples: `62048`
- unmatched eval samples: `11032`
- valid context samples: `62048`
- analyzed eval samples: `62048`
- latent dim: `192`
- action block dim: `10`
- git commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- missing keys: `[]`
- unexpected keys: `[]`

## 官方 Predictor 失败样本统计

|failure_mode|count|rate|last_step_mse_mean|last_step_mse_global_mean|last_step_cosine_mean|official_window_mse_mean|official_window_mse_global_mean|
|---|---|---|---|---|---|---|---|
|direct_mse_top20|12410|0.20000644028186798|0.0039315358735620975|0.0012493293033912778|0.9978534579277039|0.0025314607191830873|0.0013947332045063376|
|direct_mse_top10|6205|0.10000322014093399|0.006592591758817434|0.0012493293033912778|0.9963011741638184|0.0034936845768243074|0.0013947332045063376|
|direct_mse_top5|3103|0.05000967159867287|0.01115331705659628|0.0012493293033912778|0.9936159253120422|0.005057379603385925|0.0013947332045063376|
|direct_cosine_error_top20|12414|0.20007091760635376|0.003906634636223316|0.0012493293033912778|0.9978412389755249|0.00254367315210402|0.0013947332045063376|
|direct_cosine_error_top10|6205|0.10000322014093399|0.006557890214025974|0.0012493293033912778|0.9962812066078186|0.0034961658529937267|0.0013947332045063376|
|direct_cosine_error_top5|3103|0.05000967159867287|0.011113111861050129|0.0012493293033912778|0.9935911297798157|0.005043996497988701|0.0013947332045063376|
|official_window_mse_top20|12410|0.20000644028186798|0.003581521799787879|0.0012493293033912778|0.9980291128158569|0.004328410141170025|0.0013947332045063376|
|official_window_mse_top10|6205|0.10000322014093399|0.005754008423537016|0.0012493293033912778|0.9967496395111084|0.007255614269524813|0.0013947332045063376|
|official_window_mse_top5|3103|0.05000967159867287|0.009216801263391972|0.0012493293033912778|0.9947051405906677|0.012260137125849724|0.0013947332045063376|

## 规则与官方 Predictor 失败样本的对应关系

下面的 `enrichment` 不是“加入规则后错误变高”，而是一个统计富集比例：

```text
enrichment = active_rate / base_rate
```

含义是：

- `base_rate`：全部分析样本里，属于某类高错误样本的比例。
- `active_rate`：只看某个 rule / rule group 高激活的样本，其中属于同类高错误样本的比例。
- `enrichment > 1`：rule 高激活时，更容易遇到官方 predictor 高错误样本。
- `enrichment = 1`：rule 激活和错误样本没有明显关系。
- `enrichment < 1`：rule 高激活时，反而较少落在这类错误样本里。

例子：`v42_group_10_13_12_1` 在 `direct_mse_top5` 上：

```text
base_rate   = 0.0500  约等于全部样本里 top5 高错误样本占 5%
active_rate = 0.1438  约等于该 rule group 高激活样本里有 14.4% 是 top5 高错误
enrichment  = 0.1438 / 0.0500 ≈ 2.87
```

所以它证明的是：V4.2 这个 rule group 像一个“错误雷达”，更常在官方 LeWM predictor 真正预测失败的位置亮起来。它不表示 rule 导致错误，也不表示把 rule 接入模型后错误变高。

最高富集项：

|rule_or_group|failure_mode|base_rate|active_rate|inactive_rate|enrichment|active_count|
|---|---|---|---|---|---|---|
|v42_group_10_13_12_1|direct_cosine_error_top5|0.05000967159867287|0.14681708812713623|0.039252907037734985|2.935773890005556|6205|
|v42_group_10_13_12_1|direct_mse_top5|0.05000967159867287|0.14375503361225128|0.03959314525127411|2.8745446434018613|6205|
|v42_rule_10|direct_cosine_error_top5|0.05000967159867287|0.13972602784633636|0.04004082828760147|2.7939801118398937|6205|
|v42_group_best_rules_10_13_12_6|direct_cosine_error_top5|0.05000967159867287|0.13924254477024078|0.04009455069899559|2.784312320377963|6205|
|v42_rule_10|direct_mse_top5|0.05000967159867287|0.13763093948364258|0.040273625403642654|2.7520864481600587|6205|
|v42_group_best_rules_10_13_12_6|direct_mse_top5|0.05000967159867287|0.13698630034923553|0.04034525528550148|2.7391961588660143|6205|
|v42_group_10_13_12_1|direct_mse_top10|0.10000322014093399|0.2718775272369385|0.08090539276599884|2.7186877267930267|6205|
|v42_group_10_13_12_1|direct_cosine_error_top10|0.10000322014093399|0.26752617955207825|0.08138889074325562|2.675175651094585|6205|
|v42_group_best_rules_10_13_12_6|direct_mse_top10|0.10000322014093399|0.26220789551734924|0.08197983354330063|2.6219945232545623|6205|
|v42_group_best_rules_10_13_12_6|direct_cosine_error_top10|0.10000322014093399|0.25801771879196167|0.082445427775383|2.580094105253198|6205|
|v4_group_15_2_8_11|direct_cosine_error_top5|0.05000967159867287|0.1289282888174057|0.041240621358156204|2.578067095742079|6205|
|v42_rule_10|direct_mse_top10|0.10000322014093399|0.25688961148262024|0.08257077634334564|2.568813395414539|6205|
|v42_rule_10|direct_cosine_error_top10|0.10000322014093399|0.25479450821876526|0.08280357718467712|2.5478630374070432|6205|
|v4_group_15_2_8_11|direct_mse_top5|0.05000967159867287|0.12699435651302338|0.04145551100373268|2.5393959298943587|6205|
|v4_group_15_2_8_11|direct_mse_top10|0.10000322014093399|0.24174052476882935|0.08425407111644745|2.4173274063389734|6205|
|v4_group_15_2_8_11|direct_cosine_error_top10|0.10000322014093399|0.24141821265220642|0.08428988605737686|2.414104388958446|6205|
|v42_group_10_13_12_1|official_window_mse_top5|0.05000967159867287|0.11055600643157959|0.04328205808997154|2.210692510016672|6205|
|v42_group_10_13_12_1|official_window_mse_top10|0.10000322014093399|0.22095084190368652|0.08656411617994308|2.2094372720428574|6205|

## 图片索引

|figure|type|source|path|
|---|---|---|---|
|direct_mse_spatial_map.png|spatial_map|direct_mse|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/direct_mse_spatial_map.png|
|official_window_mse_spatial_map.png|spatial_map|official_window_mse|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/official_window_mse_spatial_map.png|
|v42_rule_10_vs_direct_mse_overlay.png|rule_error_overlay|v42_rule_10|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/v42_rule_10_vs_direct_mse_overlay.png|
|v42_group_10_13_12_1_vs_direct_mse_overlay.png|rule_error_overlay|v42_group_10_13_12_1|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/v42_group_10_13_12_1_vs_direct_mse_overlay.png|
|v4_group_15_2_8_11_vs_direct_mse_overlay.png|rule_error_overlay|v4_group_15_2_8_11|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/v4_group_15_2_8_11_vs_direct_mse_overlay.png|
|direct_mse_top_images.png|top_images|direct_mse_top12|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/direct_mse_top_images.png|
|episode_5343_around_t78_direct_error_strip.png|episode_strip|episode_5343_around_t78|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_5343_around_t78_direct_error_strip.png|
|episode_5343_around_t78_direct_error_timeseries.png|episode_timeseries|episode_5343_around_t78|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_5343_around_t78_direct_error_timeseries.png|
|episode_5343_around_t78_direct_error_trajectory.png|episode_trajectory|episode_5343_around_t78|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_5343_around_t78_direct_error_trajectory.png|
|episode_3155_around_t65_direct_error_strip.png|episode_strip|episode_3155_around_t65|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_3155_around_t65_direct_error_strip.png|
|episode_3155_around_t65_direct_error_timeseries.png|episode_timeseries|episode_3155_around_t65|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_3155_around_t65_direct_error_timeseries.png|
|episode_3155_around_t65_direct_error_trajectory.png|episode_trajectory|episode_3155_around_t65|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_3155_around_t65_direct_error_trajectory.png|
|episode_2367_around_t35_direct_error_strip.png|episode_strip|episode_2367_around_t35|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_2367_around_t35_direct_error_strip.png|
|episode_2367_around_t35_direct_error_timeseries.png|episode_timeseries|episode_2367_around_t35|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_2367_around_t35_direct_error_timeseries.png|
|episode_2367_around_t35_direct_error_trajectory.png|episode_trajectory|episode_2367_around_t35|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_2367_around_t35_direct_error_trajectory.png|
|episode_5752_around_t27_direct_error_strip.png|episode_strip|episode_5752_around_t27|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_5752_around_t27_direct_error_strip.png|
|episode_5752_around_t27_direct_error_timeseries.png|episode_timeseries|episode_5752_around_t27|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_5752_around_t27_direct_error_timeseries.png|
|episode_5752_around_t27_direct_error_trajectory.png|episode_trajectory|episode_5752_around_t27|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_direct_predictor_failure_atlas/figures/episode_5752_around_t27_direct_error_trajectory.png|

## 当前结论

- `direct_mse_top10/top20` 是目前最硬的一步预测失败标签，因为它直接来自官方 predictor 输出和官方 encoder latent target 的误差。
- 如果 V4/V4.2 rule group 在 `direct_mse_top10` 上有高富集，说明它和官方 LeWM 的真实 next-latent 预测失败有关，而不只是和之前的物理 residual proxy 有关。
- 这次结果里，V4.2 group `[10,13,12,1]` 对 `direct_mse_top5` 的富集约为 `2.87`，对 `direct_cosine_error_top5` 的富集约为 `2.94`，说明它确实集中覆盖官方 predictor 最难预测的样本。
- episode 图片展示的是具体错题：图像状态、轨迹位置、direct predictor error 随时间变化，以及缩放后的 V4.2 group activation。

## 注意事项

- 这是一步 direct predictor failure analysis，和官方训练目标一致；它还不是多步 rollout failure。
- 本分析直接用官方 checkpoint 和官方 validation DataLoader transform 重新编码 validation pixels，再运行官方 predictor / action encoder。
- 高富集是强相关证据，不是因果证明。要证明规则真的能改变错误，还需要做 rule masking / intervention。
