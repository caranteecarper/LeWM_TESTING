# TwoRoom Position Probe Result

## Input

```text
outputs/tworoom_encoder_readout/tworoom_latents.pt
```

Latent shape: `(730809, 192)`
Position shape: `(730809, 2)`

Split policy: deterministic random split by unique `episode_id`, seed `3072`.

- train episodes: `8000`, samples: `584945`
- val episodes: `1000`, samples: `73141`
- test episodes: `1000`, samples: `72723`

## Validation Metrics

- x MSE: `11.451403617858887`
- y MSE: `405.3571472167969`
- overall position MSE: `208.40426635742188`
- x R2: `0.9914700388908386`
- y R2: `0.7376241087913513`
- x Pearson: `0.9957255721092224`
- y Pearson: `0.8588843941688538`

## Test Metrics

- x MSE: `11.77698040008545`
- y MSE: `437.2347106933594`
- overall position MSE: `224.50584411621094`
- x R2: `0.9915762543678284`
- y R2: `0.725082516670227`
- x Pearson: `0.9957758784294128`
- y Pearson: `0.8524305820465088`

## Probe Weights

- W shape: `(2, 192)`
- w_x shape: `(192,)`
- w_y shape: `(192,)`
- w_x / w_y cosine similarity: `-0.2371075302362442`
- w_x top-20 latent dims: `[59, 160, 66, 106, 65, 61, 44, 19, 135, 40, 41, 32, 167, 42, 30, 103, 161, 21, 178, 110]`
- w_y top-20 latent dims: `[42, 110, 7, 30, 108, 126, 149, 19, 87, 84, 184, 73, 64, 76, 40, 152, 119, 92, 135, 164]`

Weights saved to:

```text
outputs/tworoom_encoder_readout/linear_probe_position.pt
```

Figures saved under:

```text
outputs/tworoom_encoder_readout/figures
```
