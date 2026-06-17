# TwoRoom Pre-Integration V2 Scope

Generated: 2026-06-17T00:19:36

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

## Inputs

- latent cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_encoder_readout/tworoom_latents.pt` exists=True
- pair cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration/tworoom_pairs.pt` exists=True
- official checkpoint: `/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt` exists=True
- v1 reports: `/data/lzt26/lewm_official_tworoom_readout_git/reports/tworoom_preintegration` exists=True

## Scope

- This stage does not modify official LeWM encoder, predictor, loss, `train.py`, `module.py`, or TwoRoom environment logic.
- This stage does not start new LeWM baseline training.
- KANFIS-style models here are external diagnostic probes only; they are not inserted into LeWM.
- Factor extractors must obey `h_t = F(z_t)`. Action is not allowed to enter the extractor.
- Action may only be used after `h_t` exists, as an auxiliary/diagnostic input for transition, residual, or h dynamics checks.
- Outputs are written to `/data/lzt26/lewm_official_tworoom_readout_git/reports/tworoom_preintegration_v2` and `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v2`.
