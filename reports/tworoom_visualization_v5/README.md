# TwoRoom V5 Representation Visualization

## Executive Summary

V5 visualizes representation layers after training. It does not modify LeWM, h, q, KANFIS, or the predictor. Image decoders are post-hoc and receive fixed z/h/q/hq tensors; current-frame image reconstruction does not use action.

The intended interpretation is qualitative plus diagnostic: z should retain richer visual detail, h should retain continuous state/room geometry, q should expose coarse rule partitions, and [h, q_v41] should be the strongest next-stage input candidate when both state continuity and white-box rules are useful.


---

# TwoRoom V5 Visualization Scope

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

Execution note: the server clean clone still reports an older local HEAD because it was not fast-forwarded before this run. The V5 scripts used here were synchronized from local/GitHub commits through `fa8e1ca Fix V5 decoder and rule grid execution`.

## Goal

V5 is post-hoc representation visualization. It compares what information is retained in:

- official LeWM latent `z_t`
- dynamics-aware factor `h_t`
- KANFIS rule factor `q_v4_t`
- sharper KANFIS rule factor `q_v41_t`
- combined `[h_t, q_v41_t]`

## Constraints

- No LeWM encoder / predictor / train.py / loss / environment modification.
- No KANFIS or h retraining.
- Image reconstruction trains only post-hoc decoders with fixed representations.
- No q->h reconstruction, h mimicry, or hidden-state distillation.
- Current-frame image reconstruction does not use action.

## Outputs

- visual dataset: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/visual_dataset.pt`
- figures: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures`
- reports: `/data/lzt26/lewm_official_tworoom_readout_git/reports/tworoom_visualization_v5`


---

# V5 Visual Dataset

Visual dataset saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/visual_dataset.pt`.

|item|value|
|---|---|
|samples|12000|
|train/val/test|8000/1500/2500|
|image_shape|(12000, 3, 64, 64)|
|z/h/qv4/qv41/hq|(12000, 192)/(12000, 16)/(12000, 16)/(12000, 16)/(12000, 32)|
|position_alignment_max_abs_error|0.0|
|flag_action_error_top20|4227|
|flag_action_error_top10|4031|
|flag_high_residual_top20|4227|
|flag_high_residual_top10|4031|
|flag_large_action_small_disp|1978|
|flag_normal|7080|

## Alignment

- HDF5 source: `/data/lzt26/stable-wm/datasets/tworoom.h5`
- Alignment rule: `h5_index = ep_offset[episode_id] + timestep`
- Max absolute position alignment error: `0.0`

## Notes

- Images are stored as uint8 64x64 RGB downsampled from original 224x224 pixels to keep decoder training and figure generation tractable.
- Representations are fixed post-hoc tensors; no LeWM, h, q, or KANFIS training is changed.
- Current-frame image reconstruction must use only z/h/q/hq, not action.


---

# V5 Image Reconstruction

Post-hoc decoders are trained with fixed representations. Action is not used for current-frame image reconstruction.

## Metrics

|representation|mse|psnr|val_mse|
|---|---|---|---|
|z|0.0020196076948195696|26.947329832868544|0.002008461393415928|
|h|0.002024041721597314|26.937805396223865|0.00201343628577888|
|q_v4|0.00202409690245986|26.937686997392518|0.002013478195294738|
|q_v41|0.0020274631679058075|26.93047026695381|0.0020171902142465115|
|hq|0.0020242698956280947|26.937315835478763|0.002013606484979391|

## Qualitative Grids

- `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/image_reconstruction/normal_grid.png`
- `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/image_reconstruction/action_error_top10_grid.png`
- `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/image_reconstruction/high_residual_top10_grid.png`
- `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/image_reconstruction/large_action_small_disp_grid.png`

## Interpretation Guide

- z is expected to retain the richest visual information.
- h should retain position and room-structure information more than texture/detail.
- q_v4/q_v41 are rule factors and should be read as coarse state partitions, not visual embeddings.
- [h, q_v41] tests whether continuous state plus rule structure improves visual recovery.


---

# V5 Semantic Maps

## Readout Metrics

|representation|pos_overall_mse|pos_x_r2|pos_y_r2|res_overall_mse|res_x_r2|res_y_r2|hard_acc|hard_f1|
|---|---|---|---|---|---|---|---|---|
|z|17.94780921936035|0.9967621564865112|0.9788062572479248|1.0738195180892944|0.5638768672943115|0.38840460777282715|0.615839958190918|0.502302151647695|
|h|48.331214904785156|0.9639533758163452|0.9664914011955261|1.0482959747314453|0.5751938819885254|0.3895072340965271|0.6269599795341492|0.5425891306688814|
|q_v4|56.783607482910156|0.9594323635101318|0.9590938091278076|1.2287983894348145|0.5035983324050903|0.26248377561569214|0.555679976940155|0.42208422918271676|
|q_v41|289.9206848144531|0.844194769859314|0.7468926310539246|1.1306427717208862|0.5403942465782166|0.3617544174194336|0.5245599746704102|0.41689967493687463|
|hq|37.848724365234375|0.9678633213043213|0.9771288633346558|1.0278671979904175|0.5813015699386597|0.43207573890686035|0.5743200182914734|0.4716826279562097|

## q Semantic Alignment

|q|dim|top_target|pearson|abs_pearson|
|---|---|---|---|---|
|q_v4|9|position_y|0.8436|0.8436|
|q_v4|7|position_x|0.8411|0.8411|
|q_v4|6|position_x|-0.8398|0.8398|
|q_v4|0|position_y|0.8184|0.8184|
|q_v4|10|position_y|-0.7444|0.7444|
|q_v4|3|position_y|-0.7386|0.7386|
|q_v4|5|position_x|-0.7351|0.7351|
|q_v41|15|position_x|-0.6668|0.6668|
|q_v4|8|position_x|0.6329|0.6329|
|q_v4|1|position_y|-0.6188|0.6188|
|q_v4|4|position_x|-0.6158|0.6158|
|q_v41|13|residual_y|0.6016|0.6016|
|q_v4|2|hard_action_error_top10|0.5998|0.5998|
|q_v41|12|residual_x|0.5668|0.5668|
|q_v41|0|residual_x|-0.5663|0.5663|
|q_v41|3|position_x|0.5497|0.5497|
|q_v41|7|position_x|-0.5316|0.5316|
|q_v4|12|position_y|-0.524|0.524|
|q_v41|4|position_x|0.5152|0.5152|
|q_v4|11|hard_action_error_top10|-0.5118|0.5118|

## Figures

- true position distribution: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/true_position_distribution.png`
- predicted position maps: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/pred_position_h.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/pred_position_q_v4.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/pred_position_q_v41.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/pred_position_hq.png`
- hard subset heatmaps: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/hard_subset_action_error_top10.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/hard_subset_high_residual_top10.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/hard_subset_large_action_small_disp.png`
- residual direction map: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/residual_direction_map.png`

## Notes

q heatmaps are post-hoc spatial activation maps. They should be interpreted as state partitions or correction factors unless rule enrichment and ablation provide stronger evidence.


---

# V5 Rule Visual Link

This report links V4.1 q rules to images, true spatial positions, residual directions, and hard-subset enrichment.

## Top Linked Rules

|rule|usage_top10|top_hard_subset|hard_enrichment|active_residual_mag|
|---|---|---|---|---|
|0|0.10000000149011612|action_error_top10|2.528302038863167|3.2785284519195557|
|12|0.10000000149011612|action_error_top10|2.4528301851427172|3.064133882522583|
|10|0.10000000149011612|action_error_top10|2.3270441371472934|3.252737283706665|
|13|0.10000000149011612|action_error_top10|2.025157284573483|2.770914077758789|
|3|0.10000000149011612|large_action_small_disp|0.7235142444706872|0.5331056714057922|
|7|0.10000000149011612|action_error_top20|0.6810035744983276|0.5968091487884521|

## Figure Pattern

For each top rule, V5 saves:

- representative high-activation sample images
- spatial activation scatter
- residual direction map over top active samples

Output directory: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/rule_visual_link`.

## Naming Caution

Rules are named conservatively as position-related, residual-correction, hard-motion correction, or state-partition rules. V5 does not name rules as walls, doors, or collisions without stronger evidence.
