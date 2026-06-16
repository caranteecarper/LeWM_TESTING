import torch

from tworoom_preintegration_common import *


def eval_masked_linear(z_std, y, test_idx, w, b, dims, name):
    x = z_std[test_idx].clone()
    x[:, dims] = 0.0
    return {"ablation": name, **metric_dict(predict_linear(x, w, b), y[test_idx])}


def main():
    ensure_dirs()
    data = load_latents()
    z = data["z"].float()
    y = data["position"].float()
    train_idx, _, test_idx, *_ = split_by_episode(data["episode_id"])
    z_std, _, _ = standardize_from_train(z, train_idx)
    lin = torch.load(PREINT_OUT / "static_probes" / "linear_static.pt", map_location="cpu")
    w, b = lin["W"], lin["b"]
    k = 32
    linear_top = topk_from_weight(w, k)
    var_top = torch.topk(z_std[train_idx].var(0), k=k).indices
    g = torch.Generator().manual_seed(3072)
    rand = torch.randperm(z_std.shape[1], generator=g)[:k]
    rows = []
    base = {"ablation": "none", **metric_dict(predict_linear(z_std[test_idx], w, b), y[test_idx])}
    rows.append(base)
    rows.append(eval_masked_linear(z_std, y, test_idx, w, b, linear_top, "mask_linear_top32"))
    rows.append(eval_masked_linear(z_std, y, test_idx, w, b, rand, "mask_random32"))
    rows.append(eval_masked_linear(z_std, y, test_idx, w, b, var_top, "mask_high_variance32"))
    kf = PREINT_OUT / "static_probes" / "kanfis_style_top32.pt"
    if kf.exists():
        kd = torch.load(kf, map_location="cpu")
        rows.append(eval_masked_linear(z_std, y, test_idx, w, b, kd["dims"], "mask_kanfis_top32"))

    # h-dim ablation for best h+action linear model if available.
    h_rows = []
    hfile = PREINT_OUT / "factor_bottleneck" / "mlp_bottleneck_K8.pt"
    if hfile.exists() and (PREINT_OUT / "tworoom_pairs.pt").exists():
        pairs = torch.load(PREINT_OUT / "tworoom_pairs.pt", map_location="cpu")
        hd = torch.load(hfile, map_location="cpu")
        model = MLPRegressor(192, hidden=384, bottleneck_dim=int(hd["K"]))
        model.load_state_dict(hd["state_dict"])
        z_norm = (pairs["z_t"].float() - hd["z_mean"]) / hd["z_std"].clamp_min(1e-6)
        _, h = batched_predict(model, z_norm, return_h=True)
        action = pairs["action_t"].float()
        target = pairs["delta_position"].float()
        tr, _, te, *_ = split_by_episode(pairs["episode_id"])
        hx = torch.cat([h, action], 1)
        hx_std, _, _ = standardize_from_train(hx, tr)
        ww, bb = fit_linear(hx_std, target, tr)
        h_rows.append({"ablation": "h_none", **metric_dict(predict_linear(hx_std[te], ww, bb), target[te])})
        for i in range(int(hd["K"])):
            xx = hx_std[te].clone()
            xx[:, i] = 0.0
            h_rows.append({"ablation": f"mask_h{i}", **metric_dict(predict_linear(xx, ww, bb), target[te])})

    fields = ["ablation", "x_mse", "y_mse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    report = f"""# Ablation Importance

## Static z -> position ablation

{table_md(rows, fields)}

## h + action transition h-dim ablation

{table_md(h_rows, fields) if h_rows else 'h ablation not available.'}

## Answers

- Top-k dims are meaningful if masking them degrades R2/MSE more than random-k.
- If KANFIS-selected dims do not hurt when masked, KANFIS importance is not reliable enough for integration.
- h dimensions are necessary only if single-dim masks measurably degrade transition prediction.
"""
    (PREINT_REPORT / "06_ablation_importance.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

