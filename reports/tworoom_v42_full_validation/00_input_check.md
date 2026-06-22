# V4.2 Full Validation Input Check

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

## Inputs

|item|value|exists|
|---|---|---|
|visual_dataset|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/visual_dataset.pt|True|
|v42_model_path|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_v42_compromise/models/v42_tau05_topk6_hard_medium_R16.pt.pt|True|
|v42_compromise_outputs|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_v42_compromise|True|
|v42_compromise_reports|/data/lzt26/lewm_official_tworoom_readout_git/reports/tworoom_kanfis_v42_compromise|True|
|q_v42_t|(12000, 16)|True|
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

- Best V4.2 model: `v42_tau05_topk6_hard_medium_R16`
- Best V4.2 model path: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_v42_compromise/models/v42_tau05_topk6_hard_medium_R16.pt.pt`
- q_v42 shape: `(12000, 16)`
- hard label count: `5`
- split counts: `8000/1500/2500`
- No LeWM, h, V4, V4.1, or V4.2 q training is performed in this validation. Existing models and caches are read only.
