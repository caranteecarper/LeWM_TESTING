# TwoRoom Pre-Integration V3 Scope

Generated: 2026-06-17T17:10:23

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

## Goal

Train a dynamics-aware physical factor extractor that combines C1-C4 diagnostics into one external path.

## Constraints

- No official LeWM encoder / predictor / loss / train.py / module.py / TwoRoom environment modifications.
- No new LeWM baseline training.
- No FFN replacement and no formal LeWM predictor integration.
- Extractor is strictly `h_t = F(z_t)`.
- Forbidden: `h_t = F(z_t, action_t)`.
- Action may only enter auxiliary heads after h is produced: delta, residual, hard labels, h_next / delta_h.
- KANFIS-style remains a low-dimensional external diagnostic only.

## Inputs

- pair cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration/tworoom_pairs.pt` exists=True
- v2 split: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v2/splits_episode_seed3072.pt` exists=True
- v2 reports: `/data/lzt26/lewm_official_tworoom_readout_git/reports/tworoom_preintegration_v2` exists=True
