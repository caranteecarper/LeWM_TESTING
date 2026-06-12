# TwoRoom Smoke Training Result

Created at: 2026-06-12 21:35 +08:00

## Scope

This was a short official-baseline smoke training run only. It did not modify the model architecture, encoder, predictor, loss, optimizer definition, or TwoRoom environment logic. It did not introduce KAN/KANFIS or FFN replacement.

## Code State

Server clean clone:

```text
/data/lzt26/lewm_official_tworoom_readout_git
```

Branch:

```text
exp/tworoom-official-encoder-readout
```

Commit used:

```text
5955620a Add TwoRoom encoder readout audit
```

## Dataset Check

Dataset:

```text
/data/lzt26/stable-wm/datasets/tworoom.h5
```

Load result:

```text
dataset type: stable_worldmodel.data.formats.hdf5.HDF5Dataset
usable samples: 730809
action_dim: 2
pixels: (4, 3, 224, 224), torch.uint8
action: (4, 10), torch.float32
proprio: (4, 2), torch.float32
```

The action field is a frameskip-packed block: `frameskip=5`, raw action dimension 2, loaded block dimension 10.

## Smoke Command

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

## Result

Smoke training succeeded.

Key log evidence:

```text
Optimizer group model_opt matched 316 modules, 297 parameter tensors, 18,034,478 total parameters.
Trainable params: 18.0 M
all tracked parameters received gradients on the first backward pass
Model saved to /data/lzt26/stable-wm/checkpoints/tworoom_encoder_readout_smoke/weights_epoch_1.pt
Trainer.fit stopped: max_steps=2 reached.
```

Validation sanity metric from the smoke run:

```text
validate/loss: 0.3752501308917999
validate/pred_loss: 0.06860950589179993
validate/sigreg_loss: 3.40625
```

These values are only chain-validation metrics and should not be interpreted as trained baseline quality.

## Checkpoint Save / Reload

Smoke checkpoint:

```text
/data/lzt26/stable-wm/checkpoints/tworoom_encoder_readout_smoke/weights_epoch_1.pt
```

Reload check:

```text
checkpoint exists: yes
size: 68.94 MB
load missing keys: []
load unexpected keys: []
encoder exists: yes
emb shape: (1, 4, 192)
act_emb shape: (1, 4, 192)
latent dim: 192
```

## Encoder Gradient Confirmation

The stable-pretraining unused-parameter/gradient hook reported:

```text
all tracked parameters received gradients on the first backward pass
```

Because the optimizer group is `modules: model`, this covers the official JEPA model parameters, including `model.encoder`.

## Next Step

Smoke training is successful. The next step is to ask for approval before launching any full baseline training. Do not start long training automatically.

