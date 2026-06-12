# TwoRoom Baseline Training Plan

Created at: 2026-06-12 21:35 +08:00

## Objective

Train an official LeWM baseline on TwoRoom only after the smoke path succeeds. This plan does not introduce KAN/KANFIS, FFN replacement, predictor changes, encoder changes, loss changes, or environment changes.

## Official Entry Point

Training entrypoint:

```bash
python train.py data=tworoom
```

This is the official `train.py` Hydra entrypoint from the clean LeWM repository.

## Config Files

Main config:

```text
config/train/lewm.yaml
```

TwoRoom data config:

```text
config/train/data/tworoom.yaml
```

Model config:

```text
config/train/model/lewm.yaml
```

Relevant official settings:

```yaml
img_size: 224
embed_dim: 192
history_size: 3
num_preds: 1
optimizer:
  type: AdamW
  lr: 5e-5
  weight_decay: 1e-3
loss:
  sigreg:
    weight: 0.09
```

TwoRoom data config:

```yaml
dataset:
  num_steps: ${eval:'${num_preds} + ${history_size}'}
  frameskip: 5
  name: tworoom.h5
  keys_to_load:
    - pixels
    - action
    - proprio
  keys_to_cache:
    - action
    - proprio
```

## Data Path

Server dataset:

```text
/data/lzt26/stable-wm/datasets/tworoom.h5
```

Environment variables for this server:

```bash
export STABLEWM_HOME=/data/lzt26/stable-wm
export LOCAL_DATASET_DIR=/data/lzt26/stable-wm
```

`LOCAL_DATASET_DIR` must point to the cache root, not directly to `datasets/`, because `stable_worldmodel.data.load_dataset` resolves datasets under `<cache_root>/datasets/`.

## Checkpoint Output Path

Official `train.py` saves weights under:

```text
$STABLEWM_HOME/checkpoints/<subdir>/
```

For a full TwoRoom baseline run, use:

```text
/data/lzt26/stable-wm/checkpoints/tworoom_baseline_official/
```

Expected weight checkpoint pattern:

```text
/data/lzt26/stable-wm/checkpoints/tworoom_baseline_official/weights_epoch_<N>.pt
```

Lightning run checkpoints may also be created under the stable-pretraining run cache.

## Smoke Training Command

Use this first to validate the chain only:

```bash
STABLEWM_HOME=/data/lzt26/stable-wm \
LOCAL_DATASET_DIR=/data/lzt26/stable-wm \
CUDA_VISIBLE_DEVICES=0 \
/data/lzt26/envs/lewm-kan/bin/python train.py \
  data=tworoom \
  subdir=tworoom_encoder_readout_smoke \
  output_model_name=tworoom_encoder_readout_smoke \
  loader.batch_size=8 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  loader.pin_memory=false \
  trainer.devices=1 \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  +trainer.limit_val_batches=1 \
  wandb.enabled=false
```

## Full Baseline Training Command

Do not start this until explicitly approved:

```bash
STABLEWM_HOME=/data/lzt26/stable-wm \
LOCAL_DATASET_DIR=/data/lzt26/stable-wm \
CUDA_VISIBLE_DEVICES=0 \
/data/lzt26/envs/lewm-kan/bin/python train.py \
  data=tworoom \
  subdir=tworoom_baseline_official \
  output_model_name=tworoom_baseline_official \
  trainer.devices=1 \
  wandb.enabled=false
```

Optional resource override if the dataloader is too slow:

```bash
loader.num_workers=6 loader.persistent_workers=true loader.prefetch_factor=3
```

## Estimated Training Time

Smoke observation:

```text
2 / 82216 train batches completed in roughly 1.4 seconds for the tiny smoke loop.
```

This does not include realistic dataloader saturation or full validation overhead. A full 100-epoch official run would be very long on TwoRoom if run exactly as configured. Before launching full training, run a 100-500 step timing job to estimate stable throughput and decide whether to cap epochs or use a wall-clock limit.

## GPU / CPU Requirements

Minimum validated smoke setup:

```text
GPU: 1 CUDA GPU
precision: bf16
batch_size: 8 for smoke
num_workers: 0 for smoke
```

Official default batch size is 128. For full training, use one GPU first unless throughput or memory pressure requires changing the plan.

## Data Generation

No dataset generation is required. TwoRoom HDF5 data already exists at:

```text
/data/lzt26/stable-wm/datasets/tworoom.h5
```

## Encoder Gradient Confirmation

The smoke run must confirm:

1. `model.encoder` exists.
2. `emb` has latent dimension 192.
3. The optimizer includes `model` parameters.
4. The first backward pass gives gradients to all tracked parameters.

The official stable-pretraining callback already reported:

```text
all tracked parameters received gradients on the first backward pass
```

This should be rechecked for any full baseline run.

