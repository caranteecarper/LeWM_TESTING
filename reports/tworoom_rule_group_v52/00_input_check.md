# V5.2 Input Check

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

|item|value|exists|
|---|---|---|
|visual_dataset|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/visual_dataset.pt|True|
|q_v4_t|(12000, 16)|True|
|q_v41_t|(12000, 16)|True|
|h_t|(12000, 16)|True|
|action_t|(12000, 10)|True|
|position_t|(12000, 2)|True|
|delta_position|(12000, 2)|True|
|residual_t|(12000, 2)|True|
|hard_labels|(12000, 5)|True|
|split|8000/1500/2500|True|

## Constraint Check

V5.2 reads existing q_v4/q_v41/h/action/physical labels. No LeWM, h, q, or KANFIS training is performed in phase A.
