import torch
import torch.nn.functional as F

from tworoom_v5_common import *


class ImageDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def train_decoder(name, x, y, tr, va, te):
    xs, xm, xstd = standardize_from_train(x, tr)
    out_dim = y.shape[1]
    model = ImageDecoder(xs.shape[1], out_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_idx = tr
    loader = DataLoader(TensorDataset(xs[train_idx].float(), y[train_idx].float()), batch_size=256, shuffle=True, num_workers=0)
    best = None
    best_val = float("inf")
    hist = []
    for epoch in range(8):
        total = 0.0
        count = 0
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]
        with torch.no_grad():
            val_pred = batched_predict(model, xs[va], batch_size=1024)
            val = F.mse_loss(val_pred, y[va]).item()
        hist.append({"epoch": epoch, "train_mse": total / max(count, 1), "val_mse": val})
        if val < best_val:
            best_val = val
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best:
        model.load_state_dict(best)
    pred_te = batched_predict(model, xs[te], batch_size=1024)
    mse = F.mse_loss(pred_te, y[te]).item()
    torch.save({"state_dict": model.cpu().state_dict(), "x_mean": xm, "x_std": xstd, "history": hist, "name": name}, V5_OUT / f"decoder_{name}.pt")
    return model, xs, pred_te, {"representation": name, "mse": mse, "psnr": psnr_from_mse(mse), "val_mse": best_val}


def pick_examples(vis, split_idx, flag_name, n=6):
    if flag_name == "normal":
        mask = vis["sample_flags"]["normal"][split_idx]
    else:
        mask = vis["sample_flags"][flag_name][split_idx]
    pool = split_idx[mask]
    if pool.numel() == 0:
        pool = split_idx
    return pool[torch.linspace(0, pool.numel() - 1, min(n, pool.numel())).long()]


def main():
    ensure_dirs()
    fig_dir = V5_FIG / "image_reconstruction"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vis = load_visual()
    tr, va, te = vis["train_idx"], vis["val_idx"], vis["test_idx"]
    y = vis["image_t"].float().reshape(vis["image_t"].shape[0], -1) / 255.0
    reps = {
        "z": vis["z_t"],
        "h": vis["h_t"],
        "q_v4": vis["q_v4_t"],
        "q_v41": vis["q_v41_t"],
        "hq": vis["hq_t"],
    }
    models = {}
    xs_all = {}
    pred_test = {}
    rows = []
    for name, x in reps.items():
        model, xs, pred, row = train_decoder(name, x.float(), y, tr, va, te)
        models[name] = model
        xs_all[name] = xs
        pred_test[name] = pred.reshape(te.numel(), 3, vis["image_size"], vis["image_size"]).clamp(0, 1)
        rows.append(row)

    # Build qualitative grids on test subset.
    te_pos = {int(v.item()): i for i, v in enumerate(te)}
    for flag in ["normal", "action_error_top10", "high_residual_top10", "large_action_small_disp"]:
        ex = pick_examples(vis, te, flag, n=6)
        grid_rows = []
        row_titles = []
        for idx in ex:
            local_test = te_pos[int(idx.item())]
            row = [vis["image_t"][idx].float() / 255.0]
            for name in ["z", "h", "q_v4", "q_v41", "hq"]:
                row.append(pred_test[name][local_test])
            grid_rows.append(row)
            row_titles.append(f"{flag}\\n#{idx.item()}")
        save_image_grid(fig_dir / f"{flag}_grid.png", grid_rows, ["Original", "z", "h", "q_v4", "q_v41", "h+q"], row_titles=row_titles, figsize_scale=1.7)

    text = f"""# V5 Image Reconstruction

Post-hoc decoders are trained with fixed representations. Action is not used for current-frame image reconstruction.

## Metrics

{table_md(rows, ["representation", "mse", "psnr", "val_mse"])}

## Qualitative Grids

- `{fig_dir / 'normal_grid.png'}`
- `{fig_dir / 'action_error_top10_grid.png'}`
- `{fig_dir / 'high_residual_top10_grid.png'}`
- `{fig_dir / 'large_action_small_disp_grid.png'}`

## Interpretation Guide

- z is expected to retain the richest visual information.
- h should retain position and room-structure information more than texture/detail.
- q_v4/q_v41 are rule factors and should be read as coarse state partitions, not visual embeddings.
- [h, q_v41] tests whether continuous state plus rule structure improves visual recovery.
"""
    write_report("02_image_reconstruction.md", text)


if __name__ == "__main__":
    main()

