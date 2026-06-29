# TwoRoom Baseline Failure Atlas And Rule Alignment

## What This Atlas Shows

This is a diagnostic atlas, not a new training run. It visualizes where the current TwoRoom representation/prediction proxy has systematic residual failures, then checks whether V4/V4.2 rules align with those failures.

The first version uses the existing residual labels and hard labels from the TwoRoom analysis cache:

- `high_residual_top10/top20`
- `action_error_top10/top20`
- `large_action_small_disp`

These are the current objective failure-mode proxies. A later version can replace or augment them with direct LeWM rollout prediction errors.

## Failure Mode Summary

|failure_mode|count|rate|residual_mag_mean|residual_mag_global_mean|delta_mag_mean|v42_group_mean|v42_group_global_mean|v42_rule10_mean|v42_rule10_global_mean|
|---|---|---|---|---|---|---|---|---|---|
|high_residual_top10|795|0.3179999887943268|3.0917201042175293|1.1533130407333374|5.059183120727539|0.13149042427539825|0.07689931988716125|0.057535964995622635|0.023569580167531967|
|action_error_top10|795|0.3179999887943268|3.0917201042175293|1.1533130407333374|5.059183120727539|0.13149042427539825|0.07689931988716125|0.057535964995622635|0.023569580167531967|
|large_action_small_disp|387|0.15479999780654907|2.56644606590271|1.1533130407333374|4.676944732666016|0.12183084338903427|0.07689931988716125|0.04989466443657875|0.023569580167531967|
|high_residual_top20|837|0.33480000495910645|2.9545834064483643|1.1533130407333374|5.126694202423096|0.12734617292881012|0.07689931988716125|0.05479836463928223|0.023569580167531967|
|action_error_top20|837|0.33480000495910645|2.9545834064483643|1.1533130407333374|5.126694202423096|0.12734617292881012|0.07689931988716125|0.05479836463928223|0.023569580167531967|

## Rule-Failure Alignment

|rule_or_group|failure_mode|base_rate|active_rate|enrichment|active_count|
|---|---|---|---|---|---|
|v4_group_15_2_8_11|high_residual_top10|0.3179999887943268|0.8119999766349792|2.553459136000654|250|
|v4_group_15_2_8_11|action_error_top10|0.3179999887943268|0.8119999766349792|2.553459136000654|250|
|v42_group_10_13_12_1|high_residual_top10|0.3179999887943268|0.7799999713897705|2.4528301851427172|250|
|v42_group_10_13_12_1|action_error_top10|0.3179999887943268|0.7799999713897705|2.4528301851427172|250|
|v4_group_15_2_8_11|large_action_small_disp|0.15479999780654907|0.37599998712539673|2.4289405197232465|250|
|v42_rule_10|high_residual_top10|0.3179999887943268|0.7720000147819519|2.42767308800523|250|
|v42_rule_10|action_error_top10|0.3179999887943268|0.7720000147819519|2.42767308800523|250|
|v4_group_15_2_8_11|high_residual_top20|0.33480000495910645|0.8119999766349792|2.4253284486484987|250|
|v4_group_15_2_8_11|action_error_top20|0.33480000495910645|0.8119999766349792|2.4253284486484987|250|
|v42_group_10_13_12_1|high_residual_top20|0.33480000495910645|0.7799999713897705|2.329748983979383|250|
|v42_group_10_13_12_1|action_error_top20|0.33480000495910645|0.7799999713897705|2.329748983979383|250|
|v42_rule_10|high_residual_top20|0.33480000495910645|0.7760000228881836|2.3178017066725154|250|

## Failure Visual Index

|failure_mode|top_images|spatial_map|residual_arrows|
|---|---|---|---|
|high_residual_top10|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/high_residual_top10_top_images.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/high_residual_top10_spatial_map.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/high_residual_top10_residual_arrows.png|
|action_error_top10|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/action_error_top10_top_images.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/action_error_top10_spatial_map.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/action_error_top10_residual_arrows.png|
|large_action_small_disp|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/large_action_small_disp_top_images.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/large_action_small_disp_spatial_map.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/large_action_small_disp_residual_arrows.png|
|high_residual_top20|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/high_residual_top20_top_images.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/high_residual_top20_spatial_map.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/high_residual_top20_residual_arrows.png|
|action_error_top20|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/action_error_top20_top_images.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/action_error_top20_spatial_map.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/action_error_top20_residual_arrows.png|

## Rule Visual Index

|rule_or_group|top_images|activation_map|residual_arrows|
|---|---|---|---|
|v42_rule_10|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v42_rule_10_top_images.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v42_rule_10_activation_map.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v42_rule_10_active_residual_arrows.png|
|v42_group_10_13_12_1|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v42_group_10_13_12_1_top_images.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v42_group_10_13_12_1_activation_map.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v42_group_10_13_12_1_active_residual_arrows.png|
|v4_group_15_2_8_11|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v4_group_15_2_8_11_top_images.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v4_group_15_2_8_11_activation_map.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/v4_group_15_2_8_11_active_residual_arrows.png|

## Episode Execution Stories

These panels show how a failure looks over time within visualized episodes: an image strip, a trajectory colored by residual magnitude, and a time series comparing residual magnitude with V4.2 group activation.

|episode_id|selected_timestep|selected_residual_mag|selected_v42_group_activation|image_strip|trajectory|timeseries|
|---|---|---|---|---|---|---|
|7218|0|11.347817420959473|0.18355156481266022|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_7218_around_t0_image_strip.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_7218_around_t0_trajectory.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_7218_around_t0_timeseries.png|
|2563|56|5.3426594734191895|0.2113354653120041|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_2563_around_t56_image_strip.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_2563_around_t56_trajectory.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_2563_around_t56_timeseries.png|
|9990|52|4.980496406555176|0.14913204312324524|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_9990_around_t52_image_strip.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_9990_around_t52_trajectory.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_9990_around_t52_timeseries.png|
|6173|6|4.966882705688477|0.19459861516952515|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_6173_around_t6_image_strip.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_6173_around_t6_trajectory.png|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_failure_atlas/figures/episode_6173_around_t6_timeseries.png|

## Current Read

- If V4.2 group/rule enrichment is high on failure modes, the rules are not only visually meaningful; they are aligned with systematic error regions.
- The episode story plots are the most useful figures for human inspection because they show what the failure looks like, where it occurs, and whether the rule activation rises around the same event.
- This atlas should be used before designing `c_conf`; it tells us which failure scores `c_conf` should predict.
