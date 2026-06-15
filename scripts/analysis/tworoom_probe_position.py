import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tworoom_common import DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR


def pearson(x, y):
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    den = x.norm() * y.norm()
    return (x @ y / den).item() if den > 0 else float("nan")


def metrics(pred, y):
    mse_dim = ((pred - y) ** 2).mean(dim=0)
    ss_res = ((pred - y) ** 2).sum(dim=0)
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / ss_tot
    return {
        "x_mse": mse_dim[0].item(),
        "y_mse": mse_dim[1].item(),
        "overall_position_mse": ((pred - y) ** 2).mean().item(),
        "x_r2": r2[0].item(),
        "y_r2": r2[1].item(),
        "x_pearson": pearson(pred[:, 0], y[:, 0]),
        "y_pearson": pearson(pred[:, 1], y[:, 1]),
    }


def split_by_episode_or_order(n):
    idx = torch.arange(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def save_scatter(path, x, y, xlabel, ylabel, title):
    plt.figure(figsize=(5, 5))
    if x.numel() > 20000:
        take = torch.linspace(0, x.numel() - 1, 20000).long()
        x = x[take]
        y = y[take]
    plt.scatter(x.numpy(), y.numpy(), s=2, alpha=0.25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_bar(path, weights, title, topk):
    vals, idx = torch.topk(weights.abs(), k=min(topk, weights.numel()))
    plt.figure(figsize=(8, 4))
    plt.bar([str(i.item()) for i in idx], weights[idx].numpy())
    plt.xlabel("latent dimension")
    plt.ylabel("weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return idx.tolist(), vals.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", default=str(DEFAULT_OUTPUT_DIR / "tworoom_latents.pt"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "linear_probe_position.pt"))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_DIR / "07_position_probe_result.md"))
    parser.add_argument("--figures", default=str(DEFAULT_OUTPUT_DIR / "figures"))
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    data = torch.load(args.latents, map_location="cpu")
    z = data["z"].float()
    y = data["position"].float()
    train_idx, val_idx, test_idx = split_by_episode_or_order(z.size(0))
    mean = z[train_idx].mean(dim=0, keepdim=True)
    std = z[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    z_std = (z - mean) / std

    x_train = torch.cat([z_std[train_idx], torch.ones(len(train_idx), 1)], dim=1)
    y_train = y[train_idx]
    solution = torch.linalg.lstsq(x_train, y_train).solution
    w = solution[:-1].T.contiguous()
    b = solution[-1].contiguous()

    def predict(indices):
        return z_std[indices] @ w.T + b

    pred_val = predict(val_idx)
    pred_test = predict(test_idx)
    val_metrics = metrics(pred_val, y[val_idx])
    test_metrics = metrics(pred_test, y[test_idx])
    wx = w[0]
    wy = w[1]
    wx_wy_cosine = F.cosine_similarity(wx, wy, dim=0).item()

    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_scatter(fig_dir / "pred_x_vs_true_x.png", y[test_idx, 0], pred_test[:, 0], "true x", "pred x", "x readout")
    save_scatter(fig_dir / "pred_y_vs_true_y.png", y[test_idx, 1], pred_test[:, 1], "true y", "pred y", "y readout")
    save_scatter(fig_dir / "pred_xy_scatter.png", pred_test[:, 0], pred_test[:, 1], "pred x", "pred y", "predicted xy")
    wx_top_idx, wx_top_abs = save_bar(fig_dir / "wx_weight_bar_topk.png", wx, "w_x top latent dimensions", args.topk)
    wy_top_idx, wy_top_abs = save_bar(fig_dir / "wy_weight_bar_topk.png", wy, "w_y top latent dimensions", args.topk)

    out = {
        "W": w,
        "b": b,
        "z_mean": mean.squeeze(0),
        "z_std": std.squeeze(0),
        "w_x": wx,
        "w_y": wy,
        "wx_topk_dims": wx_top_idx,
        "wy_topk_dims": wy_top_idx,
        "wx_wy_cosine": wx_wy_cosine,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "metadata": data.get("metadata", {}),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)

    report = f"""# TwoRoom Position Probe Result

## Input

```text
{args.latents}
```

Latent shape: `{tuple(z.shape)}`
Position shape: `{tuple(y.shape)}`

Split policy: sequential 80/10/10 over sequence-start samples. The latent cache stores one frame per sequence start to avoid overlapping-frame duplication.

## Validation Metrics

- x MSE: `{val_metrics['x_mse']}`
- y MSE: `{val_metrics['y_mse']}`
- overall position MSE: `{val_metrics['overall_position_mse']}`
- x R2: `{val_metrics['x_r2']}`
- y R2: `{val_metrics['y_r2']}`
- x Pearson: `{val_metrics['x_pearson']}`
- y Pearson: `{val_metrics['y_pearson']}`

## Test Metrics

- x MSE: `{test_metrics['x_mse']}`
- y MSE: `{test_metrics['y_mse']}`
- overall position MSE: `{test_metrics['overall_position_mse']}`
- x R2: `{test_metrics['x_r2']}`
- y R2: `{test_metrics['y_r2']}`
- x Pearson: `{test_metrics['x_pearson']}`
- y Pearson: `{test_metrics['y_pearson']}`

## Probe Weights

- W shape: `{tuple(w.shape)}`
- w_x shape: `{tuple(wx.shape)}`
- w_y shape: `{tuple(wy.shape)}`
- w_x / w_y cosine similarity: `{wx_wy_cosine}`
- w_x top-{args.topk} latent dims: `{wx_top_idx}`
- w_y top-{args.topk} latent dims: `{wy_top_idx}`

Weights saved to:

```text
{args.output}
```

Figures saved under:

```text
{args.figures}
```
"""
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report)
    print(report)


if __name__ == "__main__":
    main()

