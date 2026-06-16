import torch

from tworoom_preintegration_common import *


def main():
    ensure_dirs()
    pairs_path = PREINT_OUT / "tworoom_pairs.pt"
    pairs = torch.load(pairs_path, map_location="cpu")
    dz = pairs["delta_z"].float()
    dp = pairs["delta_position"].float()
    train_idx, val_idx, test_idx, *_ = split_by_episode(pairs["episode_id"])
    dz_std, dz_mean, dz_scale = standardize_from_train(dz, train_idx)
    rows = []

    static = torch.load(LINEAR_PROBE_PATH, map_location="cpu")
    if "W" in static and "z_std" in static:
        w = static["W"].float()
        scale = static["z_std"].float().clamp_min(1e-6)
        pred = dz[test_idx] / scale @ w.T
        rows.append({"model": "static_W_delta_projection", **metric_dict(pred, dp[test_idx])})

    w, b = fit_linear(dz_std, dp, train_idx)
    pred = predict_linear(dz_std[test_idx], w, b)
    rows.append({"model": "linear_delta_z", **metric_dict(pred, dp[test_idx])})
    torch.save({"W": w, "b": b, "mean": dz_mean, "std": dz_scale}, PREINT_OUT / "delta_linear.pt")

    mlp = MLPRegressor(192, hidden=256)
    mlp, _ = train_model(mlp, dz_std, dp, train_idx, val_idx, epochs=8, max_train=250000)
    pred = batched_predict(mlp, dz_std[test_idx])
    rows.append({"model": "mlp_delta_z", **metric_dict(pred, dp[test_idx])})
    torch.save({"state_dict": mlp.state_dict(), "mean": dz_mean, "std": dz_scale}, PREINT_OUT / "delta_mlp.pt")

    dims = topk_from_weight(w, 32)
    kf = KANFISStyleRegressor(32, rules=16)
    kf, _ = train_model(kf, dz_std[:, dims], dp, train_idx, val_idx, epochs=8, lr=2e-3, max_train=200000)
    pred = batched_predict(kf, dz_std[test_idx][:, dims])
    rows.append({"model": "kanfis_style_delta_top32", **metric_dict(pred, dp[test_idx])})
    torch.save({"state_dict": kf.state_dict(), "dims": dims, "importance": kf.importance()}, PREINT_OUT / "delta_kanfis_style_top32.pt")

    fig = PREINT_OUT / "figures_delta"
    save_scatter(fig / "delta_x_pred_vs_true.png", dp[test_idx, 0], pred[:, 0], "true delta x", "pred delta x", "KANFIS-style delta x")
    save_scatter(fig / "delta_y_pred_vs_true.png", dp[test_idx, 1], pred[:, 1], "true delta y", "pred delta y", "KANFIS-style delta y")

    fields = ["model", "x_mse", "y_mse", "x_rmse", "y_rmse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    report = f"""# Delta Readout

Input pairs: `{pairs_path}`

{table_md(rows, fields)}

## Answers

- Static W delta projection is listed as a baseline when available.
- Retrained delta_z probes should be compared against static W projection in the table.
- If delta R2 is positive and Pearson high, latent changes contain physical motion information.
- If delta_y remains weak, h+action transition should be treated cautiously.
"""
    (PREINT_REPORT / "04_delta_readout.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

