# Encoder Training Audit

Audit target: official LeWM baseline code at commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (`fix data config`).

## Summary

Under the normal official baseline training path, the encoder is trained end to end by default.

The encoder is instantiated as part of `JEPA`, participates in `model.encode(batch)`, produces `emb`, and `emb` is used directly in both loss terms:

```text
loss = pred_loss + lambda * SIGReg(emb)
```

No default freeze switch for the training encoder was found in the official training config or training entrypoint.

## Code Path Evidence

### Training entrypoint

File: `train.py`

- `run(cfg)` loads the dataset and instantiates the model with `hydra.utils.instantiate(cfg.model)`.
- The instantiated `world_model` is wrapped by `stable_pretraining.Module`.
- The training forward is `lejepa_forward`.

Relevant path:

```text
train.py -> run(cfg) -> world_model = hydra.utils.instantiate(cfg.model)
train.py -> spt.Module(model=world_model, forward=lejepa_forward, optim=optimizers)
```

### Model construction

File: `config/train/model/lewm.yaml`

The encoder is configured as:

```yaml
encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: ${img_size}
  pretrained: false
  use_mask_token: false
```

The model target is:

```yaml
_target_: jepa.JEPA
```

File: `jepa.py`

`JEPA.__init__` stores the encoder directly:

```python
self.encoder = encoder
```

There is no `requires_grad_(False)`, `.detach()`, `torch.no_grad()`, or freeze wrapper around the encoder in `JEPA.__init__` or `JEPA.encode`.

### Encoder forward path

File: `jepa.py`

`JEPA.encode(info)` runs:

```python
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]
emb = self.projector(pixels_emb)
info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
```

Because `pixels_emb` is not detached, gradients from losses using `emb` can flow through `projector` back into `encoder`.

### Predictor and loss path

File: `train.py`

`lejepa_forward` computes:

```python
output = self.model.encode(batch)
emb = output["emb"]
act_emb = output["act_emb"]
ctx_emb = emb[:, :ctx_len]
ctx_act = act_emb[:, : ctx_len]
tgt_emb = emb[:, n_preds:]
pred_emb = self.model.predict(ctx_emb, ctx_act)
output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
```

Both `pred_loss` and `sigreg_loss` depend on `emb`.

Important nuance: `tgt_emb` is not detached in the official code. Therefore `pred_loss` can backpropagate through both the predictor path and the target encoder embedding path. `sigreg_loss` also directly backpropagates through `emb`.

### Optimizer parameter source

File: `train.py`

The official training entrypoint defines:

```python
optimizers = {
    "model_opt": {
        "modules": "model",
        "optimizer": dict(cfg.optimizer),
        "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
        "interval": "epoch",
    },
}
```

This optimizer config is passed to:

```python
spt.Module(model=world_model, sigreg=SIGReg(...), forward=..., optim=optimizers)
```

`stable_pretraining.Module` groups optimizer parameters by module-name regex. In the installed implementation used with this project, `_collect_parameters_by_optimizer_groups` iterates over `self.named_modules()`, matches the configured regex, inherits the match to child modules, and collects each matched module's direct parameters. Since the top-level module name `model` matches `modules: "model"`, child modules under `model`, including `model.encoder`, are included.

The same implementation calls:

```python
self.manual_backward(state["loss"])
opt.step()
```

during training, so gradients from `state["loss"]` are applied to the optimizer's parameters.

### Optimizer config

File: `config/train/lewm.yaml`

Default optimizer:

```yaml
optimizer:
  type: AdamW
  lr: 5e-5
  weight_decay: 1e-3
```

No encoder-specific parameter exclusion was found in the official optimizer config.

## Freeze / Pretrained Encoder Conditions

### Training code

No freeze condition was found in:

- `train.py`
- `jepa.py`
- `config/train/model/lewm.yaml`
- `config/train/lewm.yaml`

### Evaluation code

`eval.py` contains:

```python
model.requires_grad_(False)
```

This is evaluation-only and not part of the official baseline training path.

### Pretrained encoder config

`config/train/model/lewm.yaml` sets:

```yaml
pretrained: false
```

This controls initialization, not freezing. It is false by default in the official training config.

## Required Answers

### Does the encoder train in normal official baseline training?

Yes. The encoder participates in the forward graph, its outputs are used in the official loss, and optimizer grouping includes the whole `model` module by default.

### Evidence code paths

- `config/train/model/lewm.yaml`: encoder is part of `jepa.JEPA`.
- `jepa.py`: `self.encoder = encoder`; `JEPA.encode` calls `self.encoder(...)`; no detach/freeze.
- `train.py`: `lejepa_forward` computes `emb` via `self.model.encode(batch)` and uses it in `pred_loss` and `SIGReg`.
- `train.py`: optimizer group is `modules: "model"`.
- `stable_pretraining.Module`: optimizer grouping matches `model` and children, then `manual_backward(state["loss"])` and `opt.step()` are used.

### Are there freeze encoder conditions?

No default training freeze condition was found. The only full-model freeze observed is in `eval.py`, which is not used for training.

### Is freeze enabled by default?

No. The official baseline training config has no encoder freeze option enabled by default.

