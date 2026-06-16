import torch

from tworoom_preintegration_common import *


def run_regression(name, x, y, train_idx, val_idx, test_idx, rows, out_dir):
    x_std, mean, std = standardize_from_train(x, train_idx)
    w, b = fit_linear(x_std, y, train_idx)
    rows.append({"model": f"{name}_linear", **metric_dict(predict_linear(x_std[test_idx], w, b), y[test_idx])})
    mlp = MLPRegressor(x_std.shape[1], hidden=256)
    mlp, _ = train_model(mlp, x_std, y, train_idx, val_idx, epochs=8, max_train=250000)
    pred = batched_predict(mlp, x_std[test_idx])
    rows.append({"model": f"{name}_mlp", **metric_dict(pred, y[test_idx])})
    torch.save({"linear_W": w, "linear_b": b, "mlp": mlp.state_dict(), "mean": mean, "std": std}, out_dir / f"{name}.pt")
    return x_std


def main():
    ensure_dirs()
    out = PREINT_OUT / "action_transition"
    out.mkdir(parents=True, exist_ok=True)
    pairs = torch.load(PREINT_OUT / "tworoom_pairs.pt", map_location="cpu")
    z = pairs["z_t"].float()
    action = pairs["action_t"].float()
    target = pairs["delta_position"].float()
    train_idx, val_idx, test_idx, *_ = split_by_episode(pairs["episode_id"])
    rows = []

    run_regression("action_only", action, target, train_idx, val_idx, test_idx, rows, out)
    z_action = torch.cat([z, action], dim=1)
    run_regression("z_action", z_action, target, train_idx, val_idx, test_idx, rows, out)

    best_h = PREINT_OUT / "factor_bottleneck" / "mlp_bottleneck_K8.pt"
    if not best_h.exists():
        best_h = sorted((PREINT_OUT / "factor_bottleneck").glob("*_K*.pt"))[-1]
    hdata = torch.load(best_h, map_location="cpu")
    z_mean = hdata["z_mean"]
    z_std = hdata["z_std"]
    z_norm = (z - z_mean) / z_std.clamp_min(1e-6)
    model = MLPRegressor(192, hidden=384 if hdata.get("kind") == "mlp_bottleneck" else 128, bottleneck_dim=int(hdata["K"]))
    model.load_state_dict(hdata["state_dict"])
    _, h = batched_predict(model, z_norm, return_h=True)
    h_action = torch.cat([h.float(), action], dim=1)
    run_regression(f"hK{hdata['K']}_action", h_action, target, train_idx, val_idx, test_idx, rows, out)

    hx_std, _, _ = standardize_from_train(h_action, train_idx)
    kf = KANFISStyleRegressor(hx_std.shape[1], rules=16)
    kf, _ = train_model(kf, hx_std, target, train_idx, val_idx, epochs=8, lr=2e-3, max_train=200000)
    pred = batched_predict(kf, hx_std[test_idx])
    rows.append({"model": f"hK{hdata['K']}_action_kanfis_style", **metric_dict(pred, target[test_idx])})
    torch.save({"state_dict": kf.state_dict(), "source_h": str(best_h), "importance": kf.importance()}, out / "h_action_kanfis_style.pt")

    fields = ["model", "x_mse", "y_mse", "x_rmse", "y_rmse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    report = f"""# Action-Conditioned Transition

{table_md(rows, fields)}

## Answers

- Action-only shows how much displacement is predictable from action alone.
- z+action is the 192-D upper-bound diagnostic.
- h+action uses saved bottleneck `{best_h}`.
- Formal integration is supported only if h+action clearly improves over action-only and approaches z+action.
"""
    (PREINT_REPORT / "05_action_conditioned_transition.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

