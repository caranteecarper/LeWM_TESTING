# TwoRoom Pre-Integration Inputs And Provenance

Generated: 2026-06-16T17:48:22

## Git

- Branch: `exp/tworoom-official-encoder-readout`
- Commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- Commit message: `4f42a43 Add TwoRoom pre-integration analysis pipeline`
- Status:

```text
?? scripts/analysis/__pycache__/
```

## Inputs

- Latent cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_encoder_readout/tworoom_latents.pt` exists=True
- Linear readout: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_encoder_readout/linear_probe_position.pt` exists=True
- Official baseline checkpoint: `/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt` exists=True

## Latent Cache Summary

|key|summary|
|---|---|
|z|shape=(730809, 192), dtype=torch.float32|
|position|shape=(730809, 2), dtype=torch.float32|
|action|shape=(730809, 10), dtype=torch.float32|
|episode_id|shape=(730809,), dtype=torch.int64|
|timestep|shape=(730809,), dtype=torch.int64|

## Linear Readout Summary

|key|summary|
|---|---|
|W|shape=(2, 192), dtype=torch.float32|
|b|shape=(2,), dtype=torch.float32|
|z_mean|shape=(192,), dtype=torch.float32|
|z_std|shape=(192,), dtype=torch.float32|

## Checkpoint Summary

```text
type=OrderedDict, keys=['encoder.embeddings.cls_token', 'encoder.embeddings.position_embeddings', 'encoder.embeddings.patch_embeddings.projection.weight', 'encoder.embeddings.patch_embeddings.projection.bias', 'encoder.layers.0.attention.q_proj.weight', 'encoder.layers.0.attention.q_proj.bias', 'encoder.layers.0.attention.k_proj.weight', 'encoder.layers.0.attention.k_proj.bias', 'encoder.layers.0.attention.v_proj.weight', 'encoder.layers.0.attention.v_proj.bias', 'encoder.layers.0.attention.o_proj.weight', 'encoder.layers.0.attention.o_proj.bias', 'encoder.layers.0.layernorm_before.weight', 'encoder.layers.0.layernorm_before.bias', 'encoder.layers.0.layernorm_after.weight', 'encoder.layers.0.layernorm_after.bias', 'encoder.layers.0.mlp.fc1.weight', 'encoder.layers.0.mlp.fc1.bias', 'encoder.layers.0.mlp.fc2.weight', 'encoder.layers.0.mlp.fc2.bias']
```

## Scope

These scripts are external diagnostics before LeWM predictor integration. They do not modify encoder, predictor, loss, TwoRoom environment logic, or any LeWM training code. KANFIS-style probes in this folder are analysis baselines only and are not inserted into LeWM.
