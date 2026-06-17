# V4 q Semantics

Best q model: `q_pos_delta_residual_hard_hnext_R16`.

Full q-target correlation matrix saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/q_target_corr_matrix.csv`.

## Top q Semantic Alignments

|q_dim|top_target|pearson|abs_pearson|posthoc_label|
|---|---|---|---|---|
|7|position_x|0.8796|0.8796|position-related factor|
|9|position_y|0.8474|0.8474|position-related factor|
|6|position_x|-0.8438|0.8438|position-related factor|
|0|position_y|0.8273|0.8273|position-related factor|
|3|position_y|-0.7976|0.7976|position-related factor|
|5|position_x|-0.7832|0.7832|position-related factor|
|10|position_y|-0.7752|0.7752|position-related factor|
|8|position_x|0.7084|0.7084|position-related factor|
|4|position_x|-0.6718|0.6718|position-related factor|
|1|position_y|-0.6468|0.6468|position-related factor|
|12|position_y|-0.5871|0.5871|unknown or mixed factor|
|11|position_y|-0.4954|0.4954|unknown or mixed factor|
|2|hard_action_error_top10|0.4664|0.4664|hard-motion factor|
|13|position_x|0.4101|0.4101|unknown or mixed factor|
|15|hard_action_error_top10|0.3697|0.3697|hard-motion factor|
|14|hard_action_error_top10|0.2948|0.2948|hard-motion factor|

## Answers

- q dimensions are post-hoc labeled only when correlation evidence is strong enough.
- Weak or mixed q dimensions should remain unnamed; they may still be useful structural rules.
- q should not be called x/y directly unless the evidence is very strong and ablation supports it.
