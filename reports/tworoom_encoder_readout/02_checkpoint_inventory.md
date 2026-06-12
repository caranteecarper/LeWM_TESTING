# TwoRoom Checkpoint Inventory

Created at: 2026-06-12 21:35 +08:00

## Scope

This inventory searched for an already available official TwoRoom LeWM baseline checkpoint that can be used with the clean official code lineage:

- Local clean branch: `exp/tworoom-official-encoder-readout`
- Official base commit: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- Official upstream: `https://github.com/lucas-maes/le-wm.git`
- Writable origin: `https://github.com/caranteecarper/LeWM_TESTING.git`

Modified KAN/KANFIS, FFN replacement, delta-input, and old experiment checkpoints were excluded.

## Search Locations

Searched on the server:

```text
/data/lzt26/stable-wm
/data/lzt26/hf_downloads
/data/lzt26/.cache/huggingface
/data/lzt26/lewm_official_tworoom_readout_git
```

## Data Found

TwoRoom dataset exists:

```text
/data/lzt26/stable-wm/datasets/tworoom.h5
```

Dataset smoke-open result:

```text
dataset type: stable_worldmodel.data.formats.hdf5.HDF5Dataset
usable samples: 730809
action_dim: 2
pixels: (4, 3, 224, 224), torch.uint8
action: (4, 10), torch.float32
proprio: (4, 2), torch.float32
```

The action tensor is a frameskip-packed block: TwoRoom raw action dimension is 2 and `frameskip=5`, so the loaded action block has dimension 10.

## Checkpoints Found

No usable official TwoRoom baseline checkpoint was found on this machine.

The search found Reacher/DMC baseline checkpoints and old experimental KAN/FFN-related checkpoints, but those do not satisfy the clean official TwoRoom baseline requirement.

The local Hugging Face cache only contains the TwoRoom dataset archive:

```text
/data/lzt26/hf_downloads/lewm/lewm-tworooms/tworoom.tar.zst
```

No local `weights.pt`, `_object.ckpt`, `_weight.ckpt`, or equivalent official TwoRoom model checkpoint was found for `quentinll/lewm-tworooms`.

## Required Fields

| Field | Result |
|---|---|
| checkpoint path | not available |
| corresponding config | not available |
| corresponding commit | not available |
| training log path | not available |
| load success | not applicable |
| latent dimension | not applicable |
| encoder exists and can forward | not applicable |

## Conclusion

未发现可用官方 TwoRoom baseline checkpoint。

The next step is to prepare and smoke-test official TwoRoom baseline training using the clean official code path.

