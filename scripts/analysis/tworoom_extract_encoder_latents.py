import argparse
from pathlib import Path

import torch

from tworoom_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    batch_to_device,
    compose_tworoom_cfg,
    load_model,
    load_tworoom_dataset,
    metadata,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "tworoom_latents.pt"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all available sequence starts")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    cfg = compose_tworoom_cfg(batch_size=args.batch_size, num_workers=0)
    dataset = load_tworoom_dataset(cfg)
    model, missing, unexpected = load_model(cfg, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    zs, positions, actions, episode_ids, timesteps = [], [], [], [], []
    seen = 0

    with torch.no_grad():
        for batch in loader:
            position_key = "pos_agent" if "pos_agent" in batch else "proprio"
            raw_position = batch[position_key][:, 0].detach().cpu().float()
            raw_action = batch["action"][:, 0].detach().cpu().float()
            batch_gpu = batch_to_device(batch, device)
            emb = model.encode(dict(batch_gpu))["emb"][:, 0].detach().cpu().float()
            bsz = emb.size(0)
            zs.append(emb)
            positions.append(raw_position)
            actions.append(raw_action)
            if "ep_idx" in batch:
                episode_ids.append(batch["ep_idx"][:, 0].detach().cpu().long())
            else:
                episode_ids.append(torch.arange(seen, seen + bsz, dtype=torch.long) // 1000000)
            if "step_idx" in batch:
                timesteps.append(batch["step_idx"][:, 0].detach().cpu().long())
            else:
                timesteps.append(torch.arange(seen, seen + bsz, dtype=torch.long))
            seen += bsz
            if args.max_samples and seen >= args.max_samples:
                break

    z = torch.cat(zs, dim=0)
    position = torch.cat(positions, dim=0)
    action = torch.cat(actions, dim=0)
    episode_id = torch.cat(episode_ids, dim=0)
    timestep = torch.cat(timesteps, dim=0)
    if args.max_samples:
        z = z[: args.max_samples]
        position = position[: args.max_samples]
        action = action[: args.max_samples]
        episode_id = episode_id[: args.max_samples]
        timestep = timestep[: args.max_samples]

    out = {
        "z": z,
        "position": position,
        "action": action,
        "episode_id": episode_id,
        "timestep": timestep,
        "metadata": metadata(
            checkpoint,
            cfg,
            dataset,
            {
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "position_source": "pos_agent[:, 0] from TwoRoom HDF5 sequence reader, falling back to proprio only if pos_agent is unavailable",
                "action_source": "raw frameskip-packed action[:, 0]",
                "sequence_policy": "one latent per dataset sequence start to avoid duplicated overlapping frames",
            },
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(f"saved {output}")
    print({k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in out.items() if k != "metadata"})
    print(out["metadata"])


if __name__ == "__main__":
    main()
