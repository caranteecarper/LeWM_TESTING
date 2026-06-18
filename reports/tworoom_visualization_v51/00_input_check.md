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
