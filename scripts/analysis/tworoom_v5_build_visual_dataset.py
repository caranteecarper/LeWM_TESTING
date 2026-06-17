import numpy as np
import torch
import torch.nn.functional as F

from tworoom_v5_common import *


def choose_subset(data, per_split):
    g = torch.Generator().manual_seed(5005)
    labels = data["hard_labels"]
    names = data["hard_label_names"]
    selected = []
    split_names = []
    for split_name, base_idx in [("train", data["train_idx"]), ("val", data["val_idx"]), ("test", data["test_idx"])]:
        target_n = per_split[split_name]
        parts = []
        normal = base_idx[labels[base_idx].sum(1) == 0]
        hard_each = max(target_n // 8, 250)
        for name in ["action_error_top10", "high_residual_top10", "large_action_small_disp"]:
            pool = base_idx[labels[base_idx, names.index(name)] > 0.5]
            if pool.numel():
                parts.append(pool[torch.randperm(pool.numel(), generator=g)[: min(hard_each, pool.numel())]])
        if normal.numel():
            parts.append(normal[torch.randperm(normal.numel(), generator=g)[: min(target_n // 2, normal.numel())]])
        cur = torch.unique(torch.cat(parts)) if parts else torch.empty(0, dtype=torch.long)
        if cur.numel() < target_n:
            rest = base_idx[~torch.isin(base_idx, cur)]
            extra = rest[torch.randperm(rest.numel(), generator=g)[: min(target_n - cur.numel(), rest.numel())]]
            cur = torch.cat([cur, extra])
        cur = cur[torch.randperm(cur.numel(), generator=g)[: min(target_n, cur.numel())]]
        selected.append(cur)
        split_names.extend([split_name] * cur.numel())
    idx = torch.cat(selected)
    local_split = {}
    offset = 0
    for split_name, cur in zip(["train", "val", "test"], selected):
        local_split[f"{split_name}_idx"] = torch.arange(offset, offset + cur.numel())
        offset += cur.numel()
    return idx, split_names, local_split


def load_images_64(data, idx, image_size=64):
    import h5py

    with h5py.File(H5_PATH, "r") as f:
        offsets = f["ep_offset"][:]
        ep = data["episode_id"][idx].numpy()
        step = data["timestep"][idx].numpy()
        h5_idx = offsets[ep] + step
        order = np.argsort(h5_idx)
        sorted_idx = h5_idx[order]
        restored = np.empty_like(order)
        restored[order] = np.arange(len(order))
        chunks = []
        for start in range(0, len(sorted_idx), 2048):
            raw = f["pixels"][sorted_idx[start : start + 2048]]
            x = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0
            small = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
            chunks.append((small.clamp(0, 1) * 255).byte())
        imgs_sorted = torch.cat(chunks, 0)
        imgs = imgs_sorted[torch.from_numpy(restored).long()]
        pos_h5 = torch.from_numpy(f["pos_agent"][h5_idx]).float()
    align_error = (pos_h5 - data["position_t"][idx]).abs().max().item()
    return imgs, h5_idx, align_error


def main():
    ensure_dirs()
    data = load_base_data()
    idx, split_names, local_split = choose_subset(data, {"train": 30000, "val": 5000, "test": 8000})
    images, h5_idx, align_error = load_images_64(data, idx)
    v4_key, q4 = compute_v4_q_for_indices(data, idx)
    v41_key, q41 = compute_v41_q_for_indices(data, idx)
    hq = torch.cat([data["h_t"][idx].float(), q41], 1)
    flags = {name: data["hard_labels"][idx, data["hard_label_names"].index(name)].bool() for name in data["hard_label_names"]}
    flags["normal"] = data["hard_labels"][idx].sum(1) == 0
    out = {
        "indices": idx,
        "h5_indices": torch.from_numpy(h5_idx).long(),
        "image_t": images,
        "image_size": 64,
        "z_t": data["z_t"][idx].float(),
        "h_t": data["h_t"][idx].float(),
        "q_v4_t": q4.float(),
        "q_v41_t": q41.float(),
        "hq_t": hq.float(),
        "position_t": data["position_t"][idx].float(),
        "residual_t": data["residual"][idx].float(),
        "delta_position": data["delta_position"][idx].float(),
        "hard_labels": data["hard_labels"][idx].float(),
        "hard_label_names": data["hard_label_names"],
        "episode_id": data["episode_id"][idx].long(),
        "timestep": data["timestep"][idx].long(),
        "split_names": split_names,
        "sample_flags": flags,
        "train_idx": local_split["train_idx"],
        "val_idx": local_split["val_idx"],
        "test_idx": local_split["test_idx"],
        "metadata": {
            "h5_path": str(H5_PATH),
            "h5_alignment_rule": "h5_index = ep_offset[episode_id] + timestep",
            "position_alignment_max_abs_error": align_error,
            "image_source": "pixels uint8, downsampled from 224x224 to 64x64 for post-hoc decoder visualization",
            "v4_q": v4_key,
            "v41_q": v41_key,
            "no_action_for_image_reconstruction": True,
        },
    }
    torch.save(out, VISUAL_DATASET)
    rows = [
        {"item": "samples", "value": idx.numel()},
        {"item": "train/val/test", "value": f"{out['train_idx'].numel()}/{out['val_idx'].numel()}/{out['test_idx'].numel()}"},
        {"item": "image_shape", "value": tuple(images.shape)},
        {"item": "z/h/qv4/qv41/hq", "value": f"{tuple(out['z_t'].shape)}/{tuple(out['h_t'].shape)}/{tuple(q4.shape)}/{tuple(q41.shape)}/{tuple(hq.shape)}"},
        {"item": "position_alignment_max_abs_error", "value": align_error},
    ]
    for name, flag in flags.items():
        rows.append({"item": f"flag_{name}", "value": int(flag.sum())})
    text = f"""# V5 Visual Dataset

Visual dataset saved to `{VISUAL_DATASET}`.

{table_md(rows, ["item", "value"])}

## Alignment

- HDF5 source: `{H5_PATH}`
- Alignment rule: `h5_index = ep_offset[episode_id] + timestep`
- Max absolute position alignment error: `{align_error}`

## Notes

- Images are stored as uint8 64x64 RGB downsampled from original 224x224 pixels to keep decoder training and figure generation tractable.
- Representations are fixed post-hoc tensors; no LeWM, h, q, or KANFIS training is changed.
- Current-frame image reconstruction must use only z/h/q/hq, not action.
"""
    write_report("01_visual_dataset.md", text)


if __name__ == "__main__":
    main()

