# V4.1 Input Check

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

Execution note: the server clean clone still reports an older local HEAD because it was not fast-forwarded before this run. The V4.1 scripts used here were synchronized from local/GitHub commit `3f0e9e7 Add V4.1 white-box KANFIS q strengthening`.

## Scope

V4.1 strengthens `q_t = KANFIS_rule_features(h_t)` without modifying LeWM and without connecting to the predictor.

## Inputs

|item|value|exists|
|---|---|---|
|best_h_dataset|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/best_h_dataset.pt|True|
|h_t_shape|(720809, 16)|True|
|action_shape|(720809, 10)|True|
|position_shape|(720809, 2)|True|
|delta_shape|(720809, 2)|True|
|residual_shape|(720809, 2)|True|
|hard_labels_shape|(720809, 5)|True|
|train/val/test|576945/72141/71723|True|
|v4_best_q|q_pos_delta_residual_hard_hnext_R16|True|
|v4_best_checkpoint|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/rule_feature_models/q_pos_delta_residual_hard_hnext_R16.pt|True|

## Constraint Check

- Action does not enter q extraction. q is produced from h only.
- This stage does not use q->h reconstruction, q->h_next reconstruction as a main loss, h mimicry, or MLP-hidden distillation.
- h_next / delta_h are not training losses in the main V4.1 variants.
