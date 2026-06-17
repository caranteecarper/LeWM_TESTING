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
