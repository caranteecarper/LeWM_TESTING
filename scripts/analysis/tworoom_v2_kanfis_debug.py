import torch

from tworoom_v2_common import *


def run_toy(name, x, y):
    n = x.shape[0]
    train_idx = torch.arange(0, int(0.7 * n))
    val_idx = torch.arange(int(0.7 * n), int(0.85 * n))
    test_idx = torch.arange(int(0.85 * n), n)
    x_std, _, _ = standardize_from_train(x, train_idx)
    y_std, y_mean, y_scale = standardize_from_train(y, train_idx)
    model = KANFISStyleTSKRegressor(x.shape[1], y.shape[1], rules=12)
    model, _, hist = train_model(model, x_std, y_std, train_idx, val_idx, epochs=80, batch_size=256, lr=2e-3, max_train=None)
    pred = inverse_standardize(batched_predict(model, x_std[test_idx]), y_mean, y_scale)
    target = y[test_idx]
    mse = ((pred - target) ** 2).mean().item()
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum().clamp_min(1e-12)
    met = {"overall_mse": mse, "r2": (1 - ss_res / ss_tot).item(), "pearson": pearson(pred.reshape(-1), target.reshape(-1))}
    return {"task": name, **met, **model.stats(), "first_loss": hist[0]["train_loss"], "last_loss": hist[-1]["train_loss"], "grad_norm": hist[-1]["grad_norm"]}, hist


def load_h_from_bottleneck(z, k, train_idx):
    path = REPO_ROOT / "outputs" / "tworoom_preintegration" / "factor_bottleneck" / f"mlp_bottleneck_K{k}.pt"
    if path.exists():
        obj = torch.load(path, map_location="cpu")
        model = MLPRegressor(192, hidden=384, bottleneck_dim=int(obj["K"]))
        model.load_state_dict(obj["state_dict"])
        z_norm = (z.float() - obj["z_mean"]) / obj["z_std"].clamp_min(1e-6)
        _, h = batched_predict(model, z_norm, return_h=True)
        return h.float(), str(path)

    z_std, z_mean, z_scale = standardize_from_train(z, train_idx)
    y = load_latents()["position"].float()
    model = MLPRegressor(192, hidden=384, bottleneck_dim=k)
    split = save_or_load_split(load_latents()["episode_id"])
    model, _, _ = train_model(model, z_std, y, split["train_idx"], split["val_idx"], epochs=8, max_train=200000)
    _, h = batched_predict(model, z_std, return_h=True)
    out = V2_OUT / "kanfis_debug"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "K": k, "z_mean": z_mean, "z_std": z_scale}, out / f"fallback_mlp_bottleneck_K{k}.pt")
    return h.float(), "trained fallback"


def main():
    ensure_dirs()
    out = V2_OUT / "kanfis_debug"
    out.mkdir(parents=True, exist_ok=True)

    # 2.2 toy sanity
    g = torch.Generator().manual_seed(3072)
    u1 = torch.linspace(-2.5, 2.5, 768).unsqueeze(1)
    u2 = torch.randn(768, 2, generator=g)
    toy_rows = []
    toy_hist = {}
    tasks = [
        ("toy_A_identity", u1, u1),
        ("toy_B_sin", u1, torch.sin(u1)),
        ("toy_C_additive", u2, (u2[:, :1] + u2[:, 1:2])),
        ("toy_D_product", u2, (u2[:, :1] * u2[:, 1:2])),
    ]
    for name, x, y in tasks:
        row, hist = run_toy(name, x.float(), y.float())
        toy_rows.append(row)
        toy_hist[name] = hist
    torch.save({"toy_rows": toy_rows, "toy_history": toy_hist}, out / "toy_sanity.pt")

    toy_pass = all(r["overall_mse"] < 0.05 for r in toy_rows if r["task"] in ["toy_A_identity", "toy_C_additive"])

    # 2.3 TwoRoom low-dimensional sanity
    data = load_latents()
    split = save_or_load_split(data["episode_id"])
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
    z = data["z"].float()
    pos = data["position"].float()
    z_std, _, _ = standardize_from_train(z, train_idx)
    w, _ = fit_linear(z_std, pos, train_idx)
    top32 = topk_from_weight(w, 32)
    h8, h8_source = load_h_from_bottleneck(z, 8, train_idx)
    h16, h16_source = load_h_from_bottleneck(z, 16, train_idx)

    pairs = load_pairs()
    p_split_train, p_split_val, p_split_test = indices_from_split(pairs["episode_id"], split)
    h16_t, _ = load_h_from_bottleneck(pairs["z_t"].float(), 16, p_split_train)
    h16_next, _ = load_h_from_bottleneck(pairs["z_next"].float(), 16, p_split_train)
    delta_h16 = h16_next - h16_t
    action = pairs["action_t"].float()
    delta_pos = pairs["delta_position"].float()
    z_action = torch.cat([pairs["z_t"].float(), action], 1)
    ha = torch.cat([h16_t, action], 1)
    row_action, state_action = train_eval_regressor("action_only_delta", "linear", action, delta_pos, p_split_train, p_split_val, p_split_test)
    action_pred_all = predict_linear(
        (action - state_action["x_mean"]) / state_action["x_std"].clamp_min(1e-6),
        state_action["W"],
        state_action["b"],
    )
    action_pred_all = inverse_standardize(action_pred_all, state_action["y_mean"], state_action["y_std"])
    residual = delta_pos - action_pred_all

    tasks = [
        ("true_position_to_position", pos, pos, train_idx, val_idx, test_idx),
        ("linear_top32_z_to_position", z[:, top32], pos, train_idx, val_idx, test_idx),
        ("h8_to_position", h8, pos, train_idx, val_idx, test_idx),
        ("h16_to_position", h16, pos, train_idx, val_idx, test_idx),
        ("delta_h16_to_delta_position", delta_h16, delta_pos, p_split_train, p_split_val, p_split_test),
        ("h16_action_to_delta_position", ha, delta_pos, p_split_train, p_split_val, p_split_test),
        ("h16_action_to_residual", ha, residual, p_split_train, p_split_val, p_split_test),
    ]

    rows = []
    states = {"h8_source": h8_source, "h16_source": h16_source, "top32": top32}
    for task_name, x, y, tr, va, te in tasks:
        for model_name in ["linear", "mlp", "kanfis"]:
            row, state = train_eval_regressor(task_name, model_name, x, y, tr, va, te, epochs=10 if model_name != "kanfis" else 16, lr=2e-3 if model_name == "kanfis" else 1e-3, max_train=160000)
            rows.append(row)
            states[f"{task_name}_{model_name}"] = state
    torch.save(states, out / "lowdim_sanity_models.pt")

    fields = ["task", "overall_mse", "r2", "pearson", "sigma_min", "sigma_mean", "sigma_max", "first_loss", "last_loss", "grad_norm"]
    low_fields = ["name", "model", "x_mse", "y_mse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    text = f"""# KANFIS-Style Debug

## Code Audit

- Current v1 failing class: `KANFISStyleRegressor` in `scripts/analysis/tworoom_preintegration_common.py`.
- v1 trained on raw position/delta targets; v2 trains targets in standardized units and inverse-transforms for metrics.
- v1 top32 KANFIS reused linear top32 dimensions by construction in `tworoom_preintegration_static_probes.py`, so `mask_kanfis_top32 == mask_linear_top32` does not prove independent KANFIS importance.
- v2 debug class: `KANFISStyleTSKRegressor` in `scripts/analysis/tworoom_v2_common.py`.
- v2 uses normalized firing strength, trainable centers/sigma/gates, local linear consequents, and bias via augmented input.

## Toy Sanity

{table_md(toy_rows, fields)}

Toy A/C pass threshold (`overall_mse < 0.05`): `{toy_pass}`.

## TwoRoom Low-Dimensional Sanity

- h8 source: `{h8_source}`
- h16 source: `{h16_source}`
- target standardization: train split mean/std, inverse transform for metrics.

{table_md(rows, low_fields)}

## Conclusion

- If toy A/C pass but TwoRoom h16 KANFIS is weak, the original failure is not a universal implementation failure; it is mainly training/conditioning/high-dimensional usage.
- If h16 succeeds while z192 failed, current KANFIS-style should only be considered on low-dimensional `h`, not raw z192.
- KANFIS remains external here and is not integrated into LeWM.
"""
    write_report("02_kanfis_debug.md", text)
    write_report("03_kanfis_lowdim_sanity.md", text)


if __name__ == "__main__":
    main()
