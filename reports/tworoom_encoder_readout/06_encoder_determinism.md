# TwoRoom Encoder Determinism

## Checkpoint

```text
/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt
```

## Setup

- split/source: dataset sequential batch from TwoRoom HDF5 through official data config
- batch size: 16
- repeats: 5
- mode: `model.eval()` with `torch.no_grad()`
- missing keys: []
- unexpected keys: []

## Result

- latent shape: `(16, 4, 192)`
- max absolute difference: `0.0`
- mean absolute difference: `0.0`
- deterministic in eval mode: `True`

## BatchNorm / Dropout Note

Official LeWM contains Dropout in the predictor and BatchNorm1d in projector / pred_proj. Encoder readout uses `model.eval()` so BatchNorm uses stored running statistics and Dropout is disabled. This is required for deterministic repeated forward checks.

## Metadata

```json
{'checkpoint': '/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt', 'repo_commit': 'd7cc05827a274b4fdb54ef4bf2bd20a3f4f9626d', 'config': 'config/train/lewm.yaml data=tworoom', 'frameskip': 5, 'action_dim_raw': 2, 'action_dim_block': 10, 'latent_dim': 192}
```
