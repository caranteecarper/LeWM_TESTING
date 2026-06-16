import torch

from tworoom_v2_common import *
from tworoom_v2_residual_and_hard_subset import encode_pure_h


def vector_metrics(pred, target):
    pred = pred.float()
    target = target.float()
    mse = ((pred - target) ** 2).mean().item()
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return {"mse": mse, "mean_r2": (1 - ss_res / ss_tot).mean().item(), "dim_r2_min": (1 - ss_res / ss_tot).min().item()}


def train_vector(name, x, y, tr, va, te, model_name):
    x_std, _, _ = standardize_from_train(x, tr)
    y_std, y_mean, y_scale = standardize_from_train(y, tr)
    if model_name == "linear":
        w, b = fit_linear(x_std, y_std, tr)
        pred = inverse_standardize(predict_linear(x_std[te], w, b), y_mean, y_scale)
    else:
        model = MLPRegressor(x.shape[1], hidden=256, out_dim=y.shape[1])
        model, _, _ = train_model(model, x_std, y_std, tr, va, epochs=8, max_train=200000)
        pred = inverse_standardize(batched_predict(model, x_std[te]), y_mean, y_scale)
    return {"task": name, "model": model_name, **vector_metrics(pred, y[te])}


def main():
    ensure_dirs()
    pairs = load_pairs()
    split = torch.load(SPLIT_PATH, map_location="cpu") if SPLIT_PATH.exists() else save_or_load_split(pairs["episode_id"])
    tr, va, te = indices_from_split(pairs["episode_id"], split)
    h, hn = encode_pure_h(pairs, "pure_mlp_K16")
    dh = hn - h
    action = pairs["action_t"].float()
    ha = torch.cat([h, action], 1)
    delta = pairs["delta_position"].float()

    # Residual from saved action-only prediction if available.
    pred_file = V2_OUT / "residual_hard_subset" / "action_only_predictions.pt"
    if pred_file.exists():
        residual = delta - torch.load(pred_file, map_location="cpu")["pred_action_linear"]
    else:
        residual = delta

    rows2 = []
    rows_h = []
    for model_name in ["linear", "mlp"]:
        for task_name, x, y in [
            ("delta_h_to_delta_position", dh, delta),
            ("h_action_to_delta_position", ha, delta),
            ("h_action_to_residual", ha, residual),
        ]:
            row, _ = train_eval_regressor(task_name, model_name, x, y, tr, va, te, epochs=8, max_train=200000)
            rows2.append(row)
        rows_h.append(train_vector("h_action_to_h_next", ha, hn, tr, va, te, model_name))
        rows_h.append(train_vector("h_action_to_delta_h", ha, dh, tr, va, te, model_name))

    fields2 = ["name", "model", "x_mse", "y_mse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    fieldsh = ["task", "model", "mse", "mean_r2", "dim_r2_min"]
    text = f"""# H Dynamics Consistency

Extractor: `pure_mlp_K16`, with `h_t = F(z_t)` and `h_next = F(z_next)`.

## Physical Delta / Residual Tasks

{table_md(rows2, fields2)}

## H-Space Dynamics Tasks

{table_md(rows_h, fieldsh)}

## Answers

- `delta_h -> delta_position` tests whether changes in h correspond to real motion.
- `[h, action] -> h_next/delta_h` tests whether h carries a temporally usable state, not just a static coordinate readout.
- If h-space prediction is stable and physical delta/residual is competitive, formal integration should prefer h-next or delta-h objectives before direct delta-position replacement.
"""
    write_report("06_h_dynamics_consistency.md", text)


if __name__ == "__main__":
    main()
