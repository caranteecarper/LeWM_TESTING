# TwoRoom Full Official Baseline Training Log

## Launch Record

Launch time: 2026-06-12 21:46 +08:00

Server clean clone:

```text
/data/lzt26/lewm_official_tworoom_readout_git
```

Branch:

```text
exp/tworoom-official-encoder-readout
```

Git commit at launch:

```text
6421811 Add TwoRoom checkpoint inventory and smoke plan
```

Git synchronization note:

```text
git status was clean before launch.
git pull --ff-only against origin timed out due GitHub/network access from the server, but the local tracking state was already at the latest pushed commit 6421811.
```

## Data

TwoRoom dataset:

```text
/data/lzt26/stable-wm/datasets/tworoom.h5
```

Environment:

```bash
STABLEWM_HOME=/data/lzt26/stable-wm
LOCAL_DATASET_DIR=/data/lzt26/stable-wm
CUDA_VISIBLE_DEVICES=0
```

## Launch Method

`tmux` was not available on the server, so training was launched with `nohup`.

PID file:

```text
/data/lzt26/stable-wm/logs/tworoom_official_baseline_full/train.pid
```

PID observed after launch:

```text
3516794
```

Run script:

```text
/data/lzt26/stable-wm/logs/tworoom_official_baseline_full/run_train.sh
```

Full training command:

```bash
cd /data/lzt26/lewm_official_tworoom_readout_git
export STABLEWM_HOME=/data/lzt26/stable-wm
export LOCAL_DATASET_DIR=/data/lzt26/stable-wm
export CUDA_VISIBLE_DEVICES=0
/data/lzt26/envs/lewm-kan/bin/python train.py \
  data=tworoom \
  subdir=tworoom_official_baseline_full \
  output_model_name=tworoom_official_baseline_full \
  trainer.devices=1 \
  wandb.enabled=false \
  hydra.run.dir=/data/lzt26/stable-wm/logs/tworoom_official_baseline_full/hydra_run
```

Path correction from the earlier plan:

```text
hydra.run.dir was set under /data/lzt26/stable-wm/logs/tworoom_official_baseline_full/ to avoid writing Hydra outputs into the Git working tree.
```

## Output Paths

Checkpoint directory:

```text
/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/
```

Training log:

```text
/data/lzt26/stable-wm/logs/tworoom_official_baseline_full/train.log
```

Hydra run directory:

```text
/data/lzt26/stable-wm/logs/tworoom_official_baseline_full/hydra_run/
```

Stable-pretraining run summary:

```text
/home/lzt26/.cache/stable-pretraining/runs/20260612/214622/1f469fd8ba52/summary.json
```

## GPU / Resource State

GPU selected:

```text
GPU 0: NVIDIA L20
```

Initial free memory before launch:

```text
GPU0 free: 45444 MiB
GPU1 free: 45444 MiB
```

Observed during training:

```text
GPU0 memory used: about 14022 MiB
GPU0 utilization: 100%
GPU1 idle
```

The run follows the official default batch size 128 from `config/train/lewm.yaml`. This preserves the official baseline setting. The GPU still has spare memory; increasing batch size would be a training-setting change and was not done in this official baseline launch.

## Startup / Encoder Training Confirmation

Model and optimizer summary from the training log:

```text
Trainable params: 18.0 M
Non-trainable params: 0
Optimizer group model_opt matched 316 modules, 297 parameter tensors, 18,034,478 total parameters.
```

Encoder participation confirmation:

```text
all tracked parameters received gradients on the first backward pass
```

This confirms that the official `model` optimizer group includes active trainable parameters and that the first backward pass produced gradients for all tracked leaf parameters, consistent with the encoder audit.

## Early Validation / Loss Summary

Initial validation sanity metrics:

```text
validate/loss: 5.180151462554932
validate/pred_loss: 0.08640156686306
validate/sigreg_loss: 56.5
```

Early fit metrics from stable-pretraining summary around step 449:

```text
fit/pred_loss last: 0.3480035066604614
fit/pred_loss min: 0.20429880917072296
fit/pred_loss max: 0.35210391879081726
fit/sigreg_loss last: 22.5
fit/sigreg_loss min: 22.5
fit/sigreg_loss max: 42.0
fit/loss last: 2.379253387451172
fit/loss min: 2.379253387451172
fit/loss max: 3.985548734664917
```

Progress log:

```text
[Epoch 0/100] step 50/5138 (3.3 it/s)
[Epoch 0/100] step 100/5138 (3.3 it/s)
[Epoch 0/100] step 150/5138 (3.3 it/s)
[Epoch 0/100] step 200/5138 (3.4 it/s)
[Epoch 0/100] step 250/5138 (3.4 it/s)
[Epoch 0/100] step 300/5138 (3.4 it/s)
[Epoch 0/100] step 350/5138 (3.5 it/s)
[Epoch 0/100] step 400/5138 (3.5 it/s)
[Epoch 0/100] step 450/5138 (3.5 it/s)
```

## Error Check

No NaN, OOM, or data-read failure was observed in the startup window.

Warnings observed:

```text
lance is not fork-safe warning from dataloader multiprocessing.
stable-pretraining env_info git command timed out.
```

These warnings did not stop training. The dataloader continued and the model reached at least step 450.

## Estimated Completion Time

Training schedule:

```text
5138 steps per epoch
100 epochs
about 513800 optimizer steps total
```

Observed speed:

```text
about 3.4-3.5 steps/sec
```

Rough estimate:

```text
about 41-45 hours including validation/checkpoint overhead
```

## Completion / Checkpoint Validation

Pending. This section should be updated after full training finishes.

