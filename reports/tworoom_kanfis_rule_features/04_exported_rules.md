# V4 Exported KANFIS Rules

Best q model: `q_pos_delta_residual_hard_hnext_R16`.

Raw rule table saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/rules/raw_rule_table.csv`.

## Top Raw Rules

|rule_id|antecedent|usage_mean|top_hard_subset|top_hard_enrichment|delta_effect_x|delta_effect_y|residual_effect_x|residual_effect_y|
|---|---|---|---|---|---|---|---|---|
|2|h_9 is High AND h_10 is Low AND h_8 is High AND h_0 is Low|0.05039|action_error_top10|3.5575|-0.95133|-2.42228|-1.0043|0.46317|
|14|h_3 is High AND h_1 is High AND h_2 is Low AND h_7 is Low|0.05385|action_error_top10|2.314|4.28554|-2.92427|14.7951|-0.3956|
|15|h_15 is High AND h_13 is Low AND h_10 is High AND h_4 is High|0.03273|action_error_top10|3.4184|-5.97644|0.9573|-15.89247|0.52045|
|12|h_8 is Low AND h_15 is High AND h_1 is High AND h_2 is Low|0.08594|action_error_top20|0.9819|-1.01816|-0.3137|-1.17492|0.18432|
|13|h_3 is High AND h_1 is High AND h_2 is Low AND h_14 is High|0.06075|action_error_top10|1.1031|4.49734|0.40711|9.84797|-0.04682|
|10|h_7 is High AND h_6 is High AND h_5 is Low AND h_9 is Low|0.07242|action_error_top20|0.8725|-1.42818|-0.87079|-0.18493|0.00945|
|3|h_5 is Low AND h_9 is Low AND h_14 is High AND h_1 is Low|0.07629|action_error_top20|0.804|-2.93677|0.75644|-2.644|0.18133|
|5|h_14 is High AND h_11 is Low AND h_0 is High AND h_3 is Low|0.07441|action_error_top20|0.8016|-3.02579|0.304|-3.0171|0.06813|
|1|h_6 is High AND h_9 is Low AND h_5 is Low AND h_4 is High|0.0617|action_error_top20|0.8956|-0.0333|0.75978|3.24027|0.00157|
|9|h_11 is High AND h_14 is Low AND h_8 is Low AND h_4 is Low|0.05848|large_action_small_disp|0.9324|-0.99207|-1.9957|-1.02972|-0.15248|
|11|h_7 is High AND h_15 is Low AND h_10 is High AND h_4 is Low|0.06802|action_error_top20|0.7801|-1.79181|-1.27189|-0.71714|0.03423|
|0|h_8 is Low AND h_11 is High AND h_13 is High AND h_14 is Low|0.06035|large_action_small_disp|0.8704|0.66417|-0.95617|-1.00748|-0.14153|

## Candidate Physical Interpretation

|rule_id|possible_meaning|evidence|confidence|
|---|---|---|---|
|2|hard-motion correction factor|usage=0.05039, enriched in action_error_top10 by 3.5575x, residual effect=(-1.0043, 0.46317)|candidate|
|14|hard-motion correction factor|usage=0.05385, enriched in action_error_top10 by 2.314x, residual effect=(14.7951, -0.3956)|candidate|
|15|hard-motion correction factor|usage=0.03273, enriched in action_error_top10 by 3.4184x, residual effect=(-15.89247, 0.52045)|candidate|
|12|state-partition factor|usage=0.08594, enriched in action_error_top20 by 0.9819x, residual effect=(-1.17492, 0.18432)|candidate|
|13|state-partition factor|usage=0.06075, enriched in action_error_top10 by 1.1031x, residual effect=(9.84797, -0.04682)|candidate|
|10|state-partition factor|usage=0.07242, enriched in action_error_top20 by 0.8725x, residual effect=(-0.18493, 0.00945)|candidate|
|3|state-partition factor|usage=0.07629, enriched in action_error_top20 by 0.804x, residual effect=(-2.644, 0.18133)|candidate|
|5|state-partition factor|usage=0.07441, enriched in action_error_top20 by 0.8016x, residual effect=(-3.0171, 0.06813)|candidate|
|1|state-partition factor|usage=0.0617, enriched in action_error_top20 by 0.8956x, residual effect=(3.24027, 0.00157)|candidate|
|9|state-partition factor|usage=0.05848, enriched in large_action_small_disp by 0.9324x, residual effect=(-1.02972, -0.15248)|candidate|
|11|state-partition factor|usage=0.06802, enriched in action_error_top20 by 0.7801x, residual effect=(-0.71714, 0.03423)|candidate|
|0|state-partition factor|usage=0.06035, enriched in large_action_small_disp by 0.8704x, residual effect=(-1.00748, -0.14153)|candidate|

## Caution

These are post-hoc interpretations. A rule is not named as an obstacle or doorway concept unless hard-subset enrichment, residual effect, and ablation all support it.
