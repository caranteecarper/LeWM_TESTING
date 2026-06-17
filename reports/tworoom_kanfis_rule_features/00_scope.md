# TwoRoom KANFIS Rule-Feature Layer Scope

Generated: 2026-06-17T20:54:21

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

Execution note: the server clean clone could not be fast-forwarded during this run, so the server-side `git` metadata above still reports the older local HEAD. The v4 scripts used for this run were synchronized from local/GitHub commit `e203172 Add TwoRoom KANFIS rule-feature diagnostics`.

## Goal

Build an input-side interpretable factor path:

```text
z_t -> h_t -> KANFIS rule-feature layer -> q_t
```

## Constraints

- No official LeWM encoder / predictor / loss / train.py / module.py / TwoRoom environment modification.
- No FFN replacement and no ARPredictor integration.
- Action does not enter `F(z_t)` or `KANFIS(h_t) -> q_t`.
- Action only enters external diagnostic heads after q is produced.
- `q_t` is a KANFIS rule-activation physical factor candidate for future prediction modules.

## Inputs

- pair cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration/tworoom_pairs.pt` exists=True
- split: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v2/splits_episode_seed3072.pt` exists=True
- v3 teacher: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v3/v3_teacher_labels.pt` exists=True
- best h key: `K16_pos_delta_residual_teacher_hard_hnext_light_aux`
