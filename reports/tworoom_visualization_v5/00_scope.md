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
