# V4.1 Exported Rules

Best V4.1 q model: `sharp_tau_R16`.

Rule table saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features_v41/rules/v41_rule_table.csv`.

|rule_id|antecedent|usage_mean|top_hard_subset|hard_enrichment|residual_x_effect|residual_y_effect|physical_posthoc_label|confidence|
|---|---|---|---|---|---|---|---|---|
|12|h_7 is Low AND h_2 is Low AND h_12 is High AND h_6 is Low|0.07867|action_error_top10|2.4091|1.27824|-0.01699|hard-motion correction factor|candidate|
|0|h_3 is Low AND h_15 is High AND h_2 is High AND h_9 is High|0.09462|action_error_top10|1.9241|-1.21116|0.02019|hard-motion correction factor|candidate|
|2|h_7 is High AND h_5 is Low AND h_9 is Low AND h_15 is Low|0.16691|action_error_top20|0.8446|-0.00477|-0.0386|state-partition factor|candidate|
|15|h_3 is Low AND h_1 is High AND h_12 is Low AND h_7 is Low|0.18493|action_error_top20|0.6808|0.22666|-0.02297|state-partition factor|candidate|
|3|h_11 is High AND h_14 is Low AND h_4 is Low AND h_3 is High|0.13466|action_error_top20|0.6845|-0.03226|-0.06211|state-partition factor|candidate|
|8|h_9 is Low AND h_13 is Low AND h_2 is Low AND h_7 is Low|0.0801|action_error_top20|0.7953|0.29346|0.10015|state-partition factor|candidate|
|14|h_7 is High AND h_5 is Low AND h_1 is Low AND h_9 is Low|0.08364|action_error_top20|0.755|-0.26134|0.076|state-partition factor|candidate|
|13|h_13 is Low AND h_2 is Low AND h_8 is High AND h_12 is High|0.01702|action_error_top10|2.3872|0.45609|0.29966|hard-motion correction factor|candidate|
|4|h_11 is High AND h_3 is High AND h_8 is Low AND h_14 is Low|0.03916|action_error_top20|0.6962|-0.52936|-0.00344|state-partition factor|candidate|
|9|h_11 is High AND h_8 is Low AND h_3 is High AND h_1 is Low|0.03286|action_error_top20|0.7241|0.07877|0.08022|state-partition factor|candidate|
|6|h_3 is Low AND h_6 is High AND h_14 is Low AND h_2 is High|0.02643|action_error_top20|0.7214|0.0326|0.0331|state-partition factor|candidate|
|1|h_2 is High AND h_5 is High AND h_12 is Low AND h_1 is High|0.02615|action_error_top20|0.7123|-0.38411|-0.05087|state-partition factor|candidate|

## White-Box Check

Rule consequents are direct linear heads from `[q, action]` to physical targets. No hidden MLP and no q->h path is used in rule export.
