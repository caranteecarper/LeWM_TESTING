# V4.1 q Semantics

Best V4.1 q model: `sharp_tau_R16`.

Correlation matrix saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features_v41/q_semantics_corr_matrix.csv`.

|q_dim|top_target|pearson|abs_pearson|posthoc_label|
|---|---|---|---|---|
|15|position_x|-0.694|0.694|position-x-related factor|
|3|position_y|0.5854|0.5854|unknown / mixed factor|
|2|position_y|-0.552|0.552|unknown / mixed factor|
|4|position_x|0.551|0.551|unknown / mixed factor|
|14|position_y|-0.5209|0.5209|unknown / mixed factor|
|7|position_x|-0.5122|0.5122|unknown / mixed factor|
|8|position_y|-0.4472|0.4472|unknown / mixed factor|
|13|residual_y|0.4344|0.4344|residual correction factor|
|9|position_x|0.4239|0.4239|unknown / mixed factor|
|1|position_x|-0.4095|0.4095|unknown / mixed factor|
|12|residual_x|0.4006|0.4006|residual correction factor|
|0|residual_x|-0.3926|0.3926|residual correction factor|
|11|position_x|0.3196|0.3196|unknown / mixed factor|
|5|position_y|-0.2586|0.2586|unknown / mixed factor|
|6|position_x|-0.127|0.127|unknown / mixed factor|
|10|hard_action_error_top10|0.0507|0.0507|unknown / mixed factor|

## Naming Rule

No q dimension is named as wall, door, or collision unless enrichment, residual effect, and ablation jointly support it.
