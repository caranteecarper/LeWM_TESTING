from pathlib import Path

import torch

from tworoom_preintegration_common import LATENTS_PATH, PREINT_OUT, PREINT_REPORT, ensure_dirs


def main():
    ensure_dirs()
    data = torch.load(LATENTS_PATH, map_location="cpu")
    z = data["z"].float()
    pos = data["position"].float()
    action = data["action"].float()
    ep = data["episode_id"].long()
    ts = data["timestep"].long()
    order = torch.argsort(ep * (ts.max() + 1) + ts)
    ep_s, ts_s = ep[order], ts[order]
    keep = (ep_s[1:] == ep_s[:-1]) & (ts_s[1:] == ts_s[:-1] + 1)
    left = order[:-1][keep]
    right = order[1:][keep]
    pairs = {
        "z_t": z[left],
        "z_next": z[right],
        "delta_z": z[right] - z[left],
        "position_t": pos[left],
        "position_next": pos[right],
        "delta_position": pos[right] - pos[left],
        "action_t": action[left],
        "episode_id": ep[left],
        "timestep": ts[left],
        "metadata": {
            "source": str(LATENTS_PATH),
            "num_source_samples": int(z.shape[0]),
            "num_pairs": int(left.numel()),
            "discarded_nonconsecutive": int((~keep).sum().item()),
            "action_shape": list(action[left].shape),
            "action_block_dim": int(action.shape[1]),
            "cross_episode_pairs": int((ep[left] != ep[right]).sum().item()),
        },
    }
    out = PREINT_OUT / "tworoom_pairs.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pairs, out)
    dp = pairs["delta_position"]
    stats = {k: getattr(dp, k)(0).tolist() for k in ["mean", "std"]}
    stats["min"] = dp.min(0).values.tolist()
    stats["max"] = dp.max(0).values.tolist()
    report = f"""# Pair Dataset

Input latent cache: `{LATENTS_PATH}`

Output pair cache: `{out}`

Method: sort by `(episode_id, timestep)` and keep only pairs where the episode is identical and `timestep_next = timestep + 1`.

- source samples: `{z.shape[0]}`
- pairs kept: `{left.numel()}`
- discarded non-consecutive candidates: `{(~keep).sum().item()}`
- cross-episode pairs after filtering: `{pairs['metadata']['cross_episode_pairs']}`
- action block shape: `{tuple(pairs['action_t'].shape)}`
- action block dim check: `{action.shape[1]}`

Action alignment note: the cached action comes from the official TwoRoom sequence reader using `frameskip=5`; it is a 10-D block at the sequence start. This is the best available alignment for transition from `z_t` to `z_next`, but the exact internal frameskip-to-next-sequence convention is not fully proven beyond the official loader/config path.

Delta position stats:

- mean: `{stats['mean']}`
- std: `{stats['std']}`
- min: `{stats['min']}`
- max: `{stats['max']}`
"""
    (PREINT_REPORT / "01_pair_dataset.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

