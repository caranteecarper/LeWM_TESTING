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
