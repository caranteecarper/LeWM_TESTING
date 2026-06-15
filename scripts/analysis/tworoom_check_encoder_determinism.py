import argparse
from pathlib import Path

import torch

from tworoom_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REPORT_DIR,
    batch_to_device,
    compose_tworoom_cfg,
    load_model,
    load_tworoom_dataset,
    metadata,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--report", default=str(DEFAULT_REPORT_DIR / "06_encoder_determinism.md"))
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    cfg = compose_tworoom_cfg(batch_size=args.batch_size, num_workers=0)
    dataset = load_tworoom_dataset(cfg)
    model, missing, unexpected = load_model(cfg, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = batch_to_device(next(iter(loader)), device)

    outputs = []
    with torch.no_grad():
        for _ in range(args.repeats):
            outputs.append(model.encode(dict(batch))["emb"].detach().cpu())

    base = outputs[0]
    max_abs = max((x - base).abs().max().item() for x in outputs[1:]) if len(outputs) > 1 else 0.0
    mean_abs = max((x - base).abs().mean().item() for x in outputs[1:]) if len(outputs) > 1 else 0.0
    stable = max_abs == 0.0

    report = f"""# TwoRoom Encoder Determinism

## Checkpoint

```text
{checkpoint}
```

## Setup

- split/source: dataset sequential batch from TwoRoom HDF5 through official data config
- batch size: {args.batch_size}
- repeats: {args.repeats}
- mode: `model.eval()` with `torch.no_grad()`
- missing keys: {missing}
- unexpected keys: {unexpected}

## Result

- latent shape: `{tuple(base.shape)}`
- max absolute difference: `{max_abs}`
- mean absolute difference: `{mean_abs}`
- deterministic in eval mode: `{stable}`

## BatchNorm / Dropout Note

Official LeWM contains Dropout in the predictor and BatchNorm1d in projector / pred_proj. Encoder readout uses `model.eval()` so BatchNorm uses stored running statistics and Dropout is disabled. This is required for deterministic repeated forward checks.

## Metadata

```json
{metadata(checkpoint, cfg, dataset)}
```
"""
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report)
    print(report)


if __name__ == "__main__":
    main()

