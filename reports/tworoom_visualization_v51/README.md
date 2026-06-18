# TwoRoom V5.1 V4 vs V4.1 Rule Visualization Comparison

V5.1 adds visualization-only comparison. It does not train new LeWM, h, q, or KANFIS models.

## Recommendation

Use a dual-track presentation: V4 rule groups for performance-oriented rule representation, and V4.1 sparse rules for white-box single-rule explanations. V4 single rules are useful but should usually be interpreted with co-activation groups.

## Recommended Midterm Figures

- V4 group activation-over-position and top images from `outputs/tworoom_visualization_v51/figures/qv4_group_link/`
- V4.1 rule 0/12/13 activation-over-position, top images, and residual arrows from `outputs/tworoom_visualization_v51/figures/qv41_rule_link_fixed/`
- Comparison table in `03_v4_vs_v41_comparison.md`

## Current Caveats

- Region names remain conservative: hard-motion, residual-correction, position-related, or state-partition. Do not call rules wall/door/collision unless additional evidence is added.
- V4 groups are post-hoc co-activation groups, not newly trained rules.


---

# V5.1 Input Check

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

Execution note: the server clean clone still reports an older local HEAD because it was not fast-forwarded before this run. The V5.1 scripts used here were synchronized from local/GitHub commit `c465894 Add V5.1 V4 and V4.1 rule visualization comparison`.

## Inputs

|item|path|exists|
|---|---|---|
|visual_dataset|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/visual_dataset.pt|True|
|image_t|(12000, 3, 64, 64)|True|
|position_t|(12000, 2)|True|
|residual_t|(12000, 2)|True|
|q_v4_t|(12000, 16)|True|
|q_v41_t|(12000, 16)|True|
|v5_rule_visual_link_figures|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/rule_visual_link|True|
|hard_label_action_error_top20|4227|True|
|hard_label_action_error_top10|4031|True|
|hard_label_high_residual_top20|4227|True|
|hard_label_high_residual_top10|4031|True|
|hard_label_large_action_small_disp|1978|True|

## Constraint Check

- No new LeWM, h, q, or KANFIS training.
- V5.1 only reads existing V4/V4.1/V5 outputs and generates comparison figures/reports.


---

# V5.1 q_v4 Rule And Group Visual Link

## Top q_v4 Single Rules

|unit_id|top_hard_subset|hard_enrichment|mean_residual_mag|spatial_compactness|coverage_top10|
|---|---|---|---|---|---|
|15|action_error_top10|2.4150944457184886|3.318685531616211|10.925469398498535|0.10000000149011612|
|2|action_error_top10|2.4779874697162003|3.1892104148864746|17.94955825805664|0.10000000149011612|
|14|action_error_top10|2.389937161145005|2.731468439102173|12.335848808288574|0.10000000149011612|
|12|action_error_top10|1.119496914288049|1.470078945159912|16.790685653686523|0.10000000149011612|
|13|action_error_top20|0.9916367456576557|0.9943514466285706|23.454700469970703|0.10000000149011612|
|9|large_action_small_disp|0.956072377289706|0.9920293688774109|30.104246139526367|0.10000000149011612|
|0|large_action_small_disp|0.801033589990116|0.716162919998169|28.926673889160156|0.10000000149011612|
|7|action_error_top20|0.6212664313339548|0.49133580923080444|25.453041076660156|0.10000000149011612|

## q_v4 Rule Groups

V4 is a soft rule representation, so grouped/co-activated rules are often more interpretable than individual rules.

|group_id|rules|top_hard_subset|hard_enrichment|mean_residual_mag|spatial_compactness|coverage_top10|
|---|---|---|---|---|---|---|
|0|15,2,8,11|action_error_top10|2.553459136000654|3.3384041786193848|13.718653678894043|0.1|
|1|14,3,11,5|large_action_small_disp|0.6201550529376273|0.42694079875946045|20.368139266967773|0.1|
|2|12,9,7,5|large_action_small_disp|0.6459948508208923|0.4377276599407196|41.80109405517578|0.1|
|3|13,15,8,1|action_error_top20|0.9557944686605618|1.1360251903533936|23.930944442749023|0.1|
|4|0,9,3,7|large_action_small_disp|0.6718346487041573|0.4669261872768402|29.697147369384766|0.1|

## Figures

- single-rule figures: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v51/figures/qv4_rule_link`
- group figures: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v51/figures/qv4_group_link`

Each unit/group has top images, activation-over-position, and residual-direction figures.


---

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


---

# V5.1 V4 vs V4.1 Rule Comparison

## Unified Comparison Table

|method|unit_type|unit_id|region|hard_enrichment|residual_magnitude|compactness|clarity|comments|
|---|---|---|---|---|---|---|---|---|
|V4|rule|15|action_error_top10|2.415|3.319|10.925|high|soft single rule; best read with group context|
|V4|rule|2|action_error_top10|2.478|3.189|17.95|high|soft single rule; best read with group context|
|V4|rule|14|action_error_top10|2.39|2.731|12.336|high|soft single rule; best read with group context|
|V4|rule|12|action_error_top10|1.119|1.47|16.791|low|soft single rule; best read with group context|
|V4|rule|13|action_error_top20|0.992|0.994|23.455|low|soft single rule; best read with group context|
|V4|rule|9|large_action_small_disp|0.956|0.992|30.104|low|soft single rule; best read with group context|
|V4|group|0|action_error_top10|2.553|3.338|13.719|medium-high|co-activated rules 15,2,8,11|
|V4|group|1|large_action_small_disp|0.62|0.427|20.368|medium|co-activated rules 14,3,11,5|
|V4|group|2|large_action_small_disp|0.646|0.438|41.801|medium|co-activated rules 12,9,7,5|
|V4|group|3|action_error_top20|0.956|1.136|23.931|medium|co-activated rules 13,15,8,1|
|V4|group|4|large_action_small_disp|0.672|0.467|29.697|medium|co-activated rules 0,9,3,7|
|V4.1|sparse_rule|0|action_error_top10|2.528|3.279|9.568|high|sparse rule; easier single-rule story|
|V4.1|sparse_rule|10|action_error_top10|2.327|3.253|30.181|high|sparse rule; easier single-rule story|
|V4.1|sparse_rule|12|action_error_top10|2.453|3.064|12.64|high|sparse rule; easier single-rule story|
|V4.1|sparse_rule|13|action_error_top10|2.025|2.771|26.748|high|sparse rule; easier single-rule story|
|V4.1|sparse_rule|7|action_error_top20|0.681|0.597|30.062|low|sparse rule; easier single-rule story|
|V4.1|sparse_rule|3|large_action_small_disp|0.724|0.533|26.815|low|sparse rule; easier single-rule story|

## Interpretation

- V4 single rules tend to preserve more performant soft rule structure, but individual rules can be less sparse.
- V4 groups are the better way to explain V4 because co-activation captures the soft rule pattern.
- V4.1 sparse rules are easier to explain one by one and now have non-empty residual-direction figures.
- For reporting, use V4 groups when discussing performance-oriented rule representations and V4.1 rules when showing white-box sparse rule examples.
