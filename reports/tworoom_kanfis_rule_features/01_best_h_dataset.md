# V4 Best-H Dataset

## Source

- best h key: `K16_pos_delta_residual_teacher_hard_hnext_light_aux`
- best h checkpoint: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v3/dynamics_aware_h/K16_pos_delta_residual_teacher_hard_hnext_light_aux.pt`
- output artifact: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/best_h_dataset.pt`

## Summary

|metric|value|
|---|---|
|samples|720809|
|h_dim|16|
|train_samples|576945|
|val_samples|72141|
|test_samples|71723|
|h_var_min_test|0.287949|
|h_var_mean_test|0.547713|
|h_abs_corr_max_offdiag_test|0.72911|
|delta_h_norm_mean_test|1.660933|

## H Dimension Top Correlations On Test

|h_dim|top_target|top_abs_pearson|
|---|---|---|
|0|position_x|0.4971|
|1|position_x|0.6259|
|2|position_x|0.4173|
|3|position_x|0.7867|
|4|position_y|0.5963|
|5|position_y|0.6025|
|6|hard_action_error_top10|0.3089|
|7|position_x|0.5089|
|8|hard_action_error_top10|0.3656|
|9|hard_action_error_top10|0.3611|
|10|position_y|0.4176|
|11|position_y|0.8048|
|12|position_y|0.3549|
|13|hard_action_error_top10|0.3706|
|14|position_y|0.6804|
|15|hard_action_error_top10|0.254|

## Constraint Check

- `h_t` and `h_next` are produced by the v3 best extractor from `z_t` and `z_next` only.
- Action is copied into the dataset for later diagnostic heads, but action is not used to generate h.
- No official LeWM encoder, predictor, loss, train script, module, or TwoRoom environment is modified.
