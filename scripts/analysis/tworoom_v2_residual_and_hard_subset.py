import torch

from tworoom_v2_common import *
from tworoom_v2_factor_extractor_no_action import AuxExtractor


def encode_pure_h(pairs, variant):
    obj = torch.load(V2_OUT / "factor_extractors" / "pure_extractors.pt", map_location="cpu")
    st = obj[variant]
    k = int(variant.split("K")[-1])
    if "linear" in variant:
        model = LinearBottleneckRegressor(192, bottleneck_dim=k)
    else:
        model = MLPRegressor(192, hidden=384, bottleneck_dim=k)
    model.load_state_dict(st["state_dict"])
    zt = (pairs["z_t"].float() - st["z_mean"]) / st["z_std"].clamp_min(1e-6)
    zn = (pairs["z_next"].float() - st["z_mean"]) / st["z_std"].clamp_min(1e-6)
    _, ht = batched_predict(model, zt, return_h=True)
    _, hn = batched_predict(model, zn, return_h=True)
    return ht.float(), hn.float()


def eval_rows(name, x, target, tr, va, te):
    rows = []
    states = {}
    for model_name in ["linear", "mlp"]:
        row, state = train_eval_regressor(name, model_name, x, target, tr, va, te, epochs=8, max_train=200000)
        rows.append(row)
        states[model_name] = state
    return rows, states


def subset_metrics(label, pred_map, target, subset):
    rows = []
    for name, pred in pred_map.items():
        rows.append({"subset": label, "model": name, **metric_dict(pred[subset], target[subset])})
    return rows


def main():
    ensure_dirs()
    out = V2_OUT / "residual_hard_subset"
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    split = torch.load(SPLIT_PATH, map_location="cpu") if SPLIT_PATH.exists() else save_or_load_split(pairs["episode_id"])
    tr, va, te = indices_from_split(pairs["episode_id"], split)
    action = pairs["action_t"].float()
    delta = pairs["delta_position"].float()

    rows = []
    states = {}
    r, st = eval_rows("action_only_delta", action, delta, tr, va, te)
    rows += r
    states["action_only"] = st
    action_linear = st["linear"]
    action_std = (action - action_linear["x_mean"]) / action_linear["x_std"].clamp_min(1e-6)
    pred_action = inverse_standardize(predict_linear(action_std, action_linear["W"], action_linear["b"]), action_linear["y_mean"], action_linear["y_std"])
    residual = delta - pred_action
    torch.save({"pred_action_linear": pred_action, "residual_linear": residual}, out / "action_only_predictions.pt")

    h8, _ = encode_pure_h(pairs, "pure_mlp_K8")
    h16, _ = encode_pure_h(pairs, "pure_mlp_K16")
    inputs = {
        "z_action": torch.cat([pairs["z_t"].float(), action], 1),
        "hK8_action": torch.cat([h8, action], 1),
        "hK16_action": torch.cat([h16, action], 1),
    }

    pred_delta_map = {"action_only_linear": pred_action}
    pred_res_map = {}
    for name, x in inputs.items():
        r_delta, st_delta = eval_rows(f"{name}_delta", x, delta, tr, va, te)
        rows += r_delta
        states[f"{name}_delta"] = st_delta
        r_res, st_res = eval_rows(f"{name}_residual", x, residual, tr, va, te)
        rows += r_res
        states[f"{name}_residual"] = st_res
        # Store linear residual reconstructed predictions for subset comparison.
        st_lin = st_res["linear"]
        x_std = (x - st_lin["x_mean"]) / st_lin["x_std"].clamp_min(1e-6)
        pred_res = inverse_standardize(predict_linear(x_std, st_lin["W"], st_lin["b"]), st_lin["y_mean"], st_lin["y_std"])
        pred_res_map[f"{name}_residual_linear"] = pred_res
        pred_delta_map[f"{name}_residual_recon_linear"] = pred_action + pred_res

    torch.save(states, out / "residual_models.pt")

    err = ((pred_action - delta) ** 2).sum(1).sqrt()
    residual_mag = residual.norm(dim=1)
    action_mag = action.norm(dim=1)
    disp_mag = delta.norm(dim=1)
    subsets = {
        "action_error_top20": err >= torch.quantile(err, 0.80),
        "action_error_top10": err >= torch.quantile(err, 0.90),
        "high_residual_top20": residual_mag >= torch.quantile(residual_mag, 0.80),
        "large_action_small_disp": (action_mag >= torch.quantile(action_mag, 0.75)) & (disp_mag <= torch.quantile(disp_mag, 0.25)),
    }
    subset_rows = []
    for label, mask in subsets.items():
        mask_te = mask & torch.isin(pairs["episode_id"].long(), split["test_eps"].long())
        subset_rows += subset_metrics(label, pred_delta_map, delta, mask_te)

    fields = ["name", "model", "x_mse", "y_mse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    subset_fields = ["subset", "model", "x_mse", "y_mse", "x_r2", "y_r2", "overall_mse"]
    text = f"""# Residual And Hard Subset

Residual is defined as `delta_position - action_only_linear_prediction`.

## Global Models

{table_md(rows, fields)}

## Hard Subsets

{table_md(subset_rows, subset_fields)}

## Answers

- hK16 is useful only if hK16 residual reconstruction improves over action-only on top-error/high-residual subsets and closes part of the z+action gap.
- If hK16 does not improve hard subsets, current h is mostly static-position readable but not yet a strong formal integration candidate.
- The saved action-only predictions are diagnostic outputs, not LeWM checkpoints.
"""
    write_report("05_residual_and_hard_subset.md", text)


if __name__ == "__main__":
    main()
