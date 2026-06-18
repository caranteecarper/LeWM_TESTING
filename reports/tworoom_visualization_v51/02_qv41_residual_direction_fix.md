# V5.1 q_v41 Residual Direction Fix

Residual direction figures are regenerated with true position coordinates and normalized residual arrows over the top 200 activated samples. Empty-arrow behavior from V5 was caused by the earlier coarse binned quiver path; V5.1 uses direct per-sample arrows.

|unit_id|top_hard_subset|hard_enrichment|mean_residual_mag|spatial_compactness|coverage_top10|
|---|---|---|---|---|---|
|0|action_error_top10|2.528302038863167|3.2785284519195557|9.568315505981445|0.10000000149011612|
|3|large_action_small_disp|0.7235142444706872|0.5331056714057922|26.814599990844727|0.10000000149011612|
|7|action_error_top20|0.6810035744983276|0.5968091487884521|30.06220245361328|0.10000000149011612|
|10|action_error_top10|2.3270441371472934|3.252737283706665|30.18051528930664|0.10000000149011612|
|12|action_error_top10|2.4528301851427172|3.064133882522583|12.64031982421875|0.10000000149011612|
|13|action_error_top10|2.025157284573483|2.770914077758789|26.748191833496094|0.10000000149011612|

## Figures

Fixed q_v41 figures are saved under `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v51/figures/qv41_rule_link_fixed`.
Each rule has:

- top images
- activation over position
- residual direction over top samples
