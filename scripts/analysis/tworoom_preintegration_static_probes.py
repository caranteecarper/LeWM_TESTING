import torch

from tworoom_preintegration_common import *


def main():
    ensure_dirs()
    fig = PREINT_OUT / "figures_static"
    out_dir = PREINT_OUT / "static_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_latents()
    z = data["z"].float()
    y = data["position"].float()
    train_idx, val_idx, test_idx, *_ = split_by_episode(data["episode_id"])
    z_std, z_mean, z_scale = standardize_from_train(z, train_idx)

    rows = []
    w, b = fit_linear(z_std, y, train_idx)
    pred = predict_linear(z_std[test_idx], w, b)
    lin = metric_dict(pred, y[test_idx])
    rows.append({"model": "linear", **lin})
    torch.save({"W": w, "b": b, "z_mean": z_mean, "z_std": z_scale}, out_dir / "linear_static.pt")

    mlp = MLPRegressor(192, hidden=384)
    mlp, _ = train_model(mlp, z_std, y, train_idx, val_idx, epochs=10, lr=1e-3, max_train=250000)
    pred = batched_predict(mlp, z_std[test_idx])
    mlp_m = metric_dict(pred, y[test_idx])
    rows.append({"model": "mlp", **mlp_m})
    torch.save({"state_dict": mlp.state_dict(), "z_mean": z_mean, "z_std": z_scale}, out_dir / "mlp_static.pt")

    top32 = topk_from_weight(w, 32)
    for name, dims in [("kanfis_style_full192", torch.arange(192)), ("kanfis_style_top32", top32)]:
        x = z_std[:, dims]
        model = KANFISStyleRegressor(x.shape[1], rules=16)
        model, _ = train_model(model, x, y, train_idx, val_idx, epochs=10, lr=2e-3, max_train=200000)
        pred = batched_predict(model, x[test_idx])
        met = metric_dict(pred, y[test_idx])
        rows.append({"model": name, **met})
        imp = model.importance()
        save_bar(fig / f"{name}_importance.png", imp, f"{name} input importance")
        mapped_imp = torch.zeros(192)
        mapped_imp[dims] = imp
        torch.save(
            {"state_dict": model.state_dict(), "dims": dims, "importance_192": mapped_imp, "z_mean": z_mean, "z_std": z_scale},
            out_dir / f"{name}.pt",
        )

    fields = ["model", "x_mse", "y_mse", "x_rmse", "y_rmse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    best_y = max(rows, key=lambda r: r["y_r2"])
    overlap = set(top32.tolist())
    report = f"""# Static Probe Comparison

Input: `{LATENTS_PATH}`

Split: episode-level 80/10/10, seed 3072.

{table_md(rows, fields)}

## Answers

- MLP improves y over linear: `{mlp_m['y_r2'] > lin['y_r2']}`.
- Best y model: `{best_y['model']}` with y R2 `{best_y['y_r2']}`.
- KANFIS-style top32/full192 are external analysis probes, not LeWM modules.
- KANFIS-style top dims are saved in `outputs/tworoom_preintegration/static_probes/`.
- Linear top32 dims: `{top32.tolist()}`.
- Conclusion: continue bottleneck extraction if nonlinear/static probes improve y or reveal concentrated latent dimensions.
"""
    (PREINT_REPORT / "02_static_probe_comparison.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

