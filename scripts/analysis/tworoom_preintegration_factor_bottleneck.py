import torch

from tworoom_preintegration_common import *


def corr_matrix(h):
    h = (h - h.mean(0)) / h.std(0).clamp_min(1e-6)
    return (h.T @ h) / max(h.shape[0] - 1, 1)


def main():
    ensure_dirs()
    out = PREINT_OUT / "factor_bottleneck"
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = load_latents()
    z = data["z"].float()
    y = data["position"].float()
    train_idx, val_idx, test_idx, *_ = split_by_episode(data["episode_id"])
    z_std, z_mean, z_scale = standardize_from_train(z, train_idx)
    rows = []
    best = None

    for k in [2, 4, 6, 8, 16]:
        for kind in ["linear_bottleneck", "mlp_bottleneck"]:
            hidden = 384 if kind.startswith("mlp") else 128
            model = MLPRegressor(192, hidden=hidden, bottleneck_dim=k)
            model, _ = train_model(model, z_std, y, train_idx, val_idx, epochs=10, lr=1e-3, max_train=250000)
            pred, h = batched_predict(model, z_std[test_idx], return_h=True)
            met = metric_dict(pred, y[test_idx])
            h_var = h.var(0)
            c = corr_matrix(h)
            h_xy = []
            for i in range(k):
                h_xy.append({"dim": i, "pearson_x": pearson(h[:, i], y[test_idx, 0]), "pearson_y": pearson(h[:, i], y[test_idx, 1])})
                save_scatter(fig / f"{kind}_K{k}_h{i}_xy.png", y[test_idx, 0], y[test_idx, 1], "true x", "true y", f"{kind} K={k} h{i}")
            row = {"model": kind, "K": k, **met, "h_var_min": h_var.min().item(), "h_var_mean": h_var.mean().item(), "h_abs_corr_max_offdiag": (c - torch.eye(k)).abs().max().item()}
            rows.append(row)
            torch.save({"state_dict": model.state_dict(), "K": k, "kind": kind, "z_mean": z_mean, "z_std": z_scale, "metrics": row, "h_xy": h_xy}, out / f"{kind}_K{k}.pt")
            if best is None or row["overall_mse"] < best["overall_mse"]:
                best = row

    fields = ["model", "K", "x_r2", "y_r2", "x_mse", "y_mse", "overall_mse", "h_var_min", "h_var_mean", "h_abs_corr_max_offdiag"]
    report = f"""# Factor Bottleneck

Input: `{LATENTS_PATH}`

Split: episode-level 80/10/10. Models are external analysis bottlenecks; LeWM is not modified.

{table_md(rows, fields)}

## Key Answers

- Recommended K by overall MSE: `{best['K']}` using `{best['model']}`.
- K=2 is sufficient only if its R2/MSE is close to larger K; see table.
- Collapse check: `h_var_min` near zero indicates collapse.
- h dimensions are candidate physical factors only when correlated with x/y or spatial regions; no dimension is named directly as x or y.
"""
    (PREINT_REPORT / "03_factor_bottleneck.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

