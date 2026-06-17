import torch

from tworoom_v3_common import *


def predict_from_state(x, state):
    xs = (x.float() - state["x_mean"]) / state["x_std"].clamp_min(1e-6)
    return inverse_standardize(predict_linear(xs, state["W"], state["b"]), state["y_mean"], state["y_std"])


def train_eval_mlp_all(x, y, tr, va, te, name):
    x_std, xm, xs = standardize_from_train(x, tr)
    y_std, ym, ys = standardize_from_train(y, tr)
    model = MLPRegressor(x.shape[1], hidden=256, out_dim=y.shape[1])
    model, _, hist = train_model(model, x_std, y_std, tr, va, epochs=8, max_train=220000)
    pred = inverse_standardize(batched_predict(model, x_std), ym, ys)
    return pred, {"state_dict": model.state_dict(), "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys, "history": hist, "name": name}


def main():
    ensure_dirs()
    pairs = load_pairs()
    split = torch.load(SPLIT_PATH, map_location="cpu") if SPLIT_PATH.exists() else save_or_load_split(pairs["episode_id"])
    tr, va, te = indices_from_split(pairs["episode_id"], split)
    z = pairs["z_t"].float()
    action = pairs["action_t"].float()
    za = torch.cat([z, action], 1)
    delta = pairs["delta_position"].float()

    rows = []
    row_act_lin, st_act_lin = train_eval_regressor("action_only_delta", "linear", action, delta, tr, va, te)
    rows.append(row_act_lin)
    action_linear_pred = predict_from_state(action, st_act_lin)

    action_mlp_pred, st_act_mlp = train_eval_mlp_all(action, delta, tr, va, te, "action_only_mlp")
    rows.append({"name": "action_only_delta", "model": "mlp", **metric_dict(action_mlp_pred[te], delta[te])})

    residual = delta - action_linear_pred
    row_z_res_lin, st_z_res_lin = train_eval_regressor("z_action_residual", "linear", za, residual, tr, va, te)
    rows.append(row_z_res_lin)
    z_res_lin_pred = predict_from_state(za, st_z_res_lin)
    z_res_mlp_pred, st_z_res_mlp = train_eval_mlp_all(za, residual, tr, va, te, "z_action_residual_mlp")
    rows.append({"name": "z_action_residual", "model": "mlp", **metric_dict(z_res_mlp_pred[te], residual[te])})
    z_delta_teacher_pred = action_linear_pred + z_res_mlp_pred
    rows.append({"name": "z_action_delta_teacher", "model": "action_linear_plus_residual_mlp", **metric_dict(z_delta_teacher_pred[te], delta[te])})

    action_err = (delta - action_linear_pred).norm(dim=1)
    residual_mag = residual.norm(dim=1)
    action_mag = action.norm(dim=1)
    disp_mag = delta.norm(dim=1)

    thresholds = {
        "action_error_top20": torch.quantile(action_err[tr], 0.80).item(),
        "action_error_top10": torch.quantile(action_err[tr], 0.90).item(),
        "high_residual_top20": torch.quantile(residual_mag[tr], 0.80).item(),
        "high_residual_top10": torch.quantile(residual_mag[tr], 0.90).item(),
        "action_mag_p75": torch.quantile(action_mag[tr], 0.75).item(),
        "disp_mag_p25": torch.quantile(disp_mag[tr], 0.25).item(),
    }
    labels = torch.stack(
        [
            action_err >= thresholds["action_error_top20"],
            action_err >= thresholds["action_error_top10"],
            residual_mag >= thresholds["high_residual_top20"],
            residual_mag >= thresholds["high_residual_top10"],
            (action_mag >= thresholds["action_mag_p75"]) & (disp_mag <= thresholds["disp_mag_p25"]),
        ],
        dim=1,
    ).float()
    label_names = ["action_error_top20", "action_error_top10", "high_residual_top20", "high_residual_top10", "large_action_small_disp"]
    label_rows = []
    for i, name in enumerate(label_names):
        label_rows.append(
            {
                "label": name,
                "train_pos_rate": labels[tr, i].mean().item(),
                "val_pos_rate": labels[va, i].mean().item(),
                "test_pos_rate": labels[te, i].mean().item(),
            }
        )

    out = {
        "z_t": z,
        "z_next": pairs["z_next"].float(),
        "action": action,
        "position_t": pairs["position_t"].float(),
        "position_next": pairs["position_next"].float(),
        "delta_position": delta,
        "episode_id": pairs["episode_id"].long(),
        "timestep": pairs["timestep"].long(),
        "split": split,
        "train_idx": tr,
        "val_idx": va,
        "test_idx": te,
        "action_only_linear_pred_delta": action_linear_pred,
        "action_only_mlp_pred_delta": action_mlp_pred,
        "residual": residual,
        "z_action_residual_teacher_pred": z_res_mlp_pred,
        "z_action_delta_teacher_pred": z_delta_teacher_pred,
        "hard_labels": labels,
        "hard_label_names": label_names,
        "thresholds": thresholds,
        "states": {
            "action_linear": st_act_lin,
            "action_mlp": st_act_mlp,
            "z_residual_linear": st_z_res_lin,
            "z_residual_mlp": st_z_res_mlp,
        },
    }
    V3_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, TEACHER_PATH)

    fields = ["name", "model", "x_mse", "y_mse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    label_fields = ["label", "train_pos_rate", "val_pos_rate", "test_pos_rate"]
    threshold_rows = [{"threshold": k, "value": v} for k, v in thresholds.items()]
    text = f"""# V3 Teacher And Labels

## Split

- train samples: `{tr.numel()}`
- val samples: `{va.numel()}`
- test samples: `{te.numel()}`
- split source: `{SPLIT_PATH}`

## Teacher Metrics

{table_md(rows, fields)}

## Hard Label Rates

{table_md(label_rows, label_fields)}

## Train-Only Thresholds

{table_md(threshold_rows, ['threshold', 'value'])}

## Notes

- Thresholds are computed only on the train split and then applied to val/test.
- Residual uses action-only linear as baseline because it is stable, interpretable, and gives a fixed correction target.
- Data leakage risk is controlled by using train-only thresholds and fixed episode-level split.
- Teacher artifact saved to `{TEACHER_PATH}`.
"""
    write_report("01_teacher_and_labels.md", text)


if __name__ == "__main__":
    main()
