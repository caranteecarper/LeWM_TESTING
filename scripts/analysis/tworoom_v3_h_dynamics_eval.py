import torch

from tworoom_v3_common import *
from tworoom_v3_evaluate_h_hard_subset import get_h_for_key


def fit_vec_pred(x, y, tr):
    x_std, xm, xs = standardize_from_train(x, tr)
    y_std, ym, ys = standardize_from_train(y, tr)
    model = MLPRegressor(x.shape[1], hidden=256, out_dim=y.shape[1])
    model, _, _ = train_model(model, x_std, y_std, tr, tr[: min(tr.numel(), 20000)], epochs=6, max_train=160000)
    return inverse_standardize(batched_predict(model, x_std), ym, ys)


def eval_h(name, h, hn, data):
    tr, te = data["train_idx"], data["test_idx"]
    action = data["action"]
    ha = torch.cat([h, action], 1)
    dh = hn - h
    rows_h = []
    rows_phy = []
    pred_hn = fit_vec_pred(ha, hn, tr)
    pred_dh = fit_vec_pred(ha, dh, tr)
    rows_h.append({"candidate": name, "task": "h_action_to_h_next", **vector_r2(pred_hn[te], hn[te])})
    rows_h.append({"candidate": name, "task": "h_action_to_delta_h", **vector_r2(pred_dh[te], dh[te])})
    for task, target in [("delta_h_to_delta_position", data["delta_position"]), ("delta_h_to_residual", data["residual"])]:
        row, _ = train_eval_regressor(f"{name}_{task}", "linear", dh, target, tr, tr[: min(tr.numel(), 20000)], te)
        rows_phy.append(row)
    return rows_h, rows_phy


def main():
    ensure_dirs()
    data = load_teacher()
    summary = torch.load(DYN_DIR / "summary.pt", map_location="cpu")
    candidate_keys = [summary["best_key"]]
    for key in ["K16_pos_delta_residual_teacher_hard_hnext_hard_weighted", "K8_pos_delta_residual_teacher_hard_hnext_hard_weighted"]:
        if (DYN_DIR / f"{key}.pt").exists() and key not in candidate_keys:
            candidate_keys.append(key)

    rows_h = []
    rows_phy = []
    old_h, old_hn = load_v2_pure_h16({"z_t": data["z_t"], "z_next": data["z_next"]})
    r1, r2 = eval_h("old_pure_hK16", old_h, old_hn, data)
    rows_h += r1
    rows_phy += r2
    for key in candidate_keys:
        h, hn = get_h_for_key(data, key)
        r1, r2 = eval_h(key, h, hn, data)
        rows_h += r1
        rows_phy += r2

    text = f"""# H Dynamics Evaluation

## H-Space Prediction

{table_md(rows_h, ['candidate', 'task', 'mse', 'mean_r2', 'dim_r2_min'])}

## Delta-H Physical Prediction

{table_md(rows_phy, ['name', 'model', 'x_mse', 'y_mse', 'x_r2', 'y_r2', 'overall_mse'])}

## Answers

- v3 h improves over old pure hK16 only if h-next/delta-h mean R2 rises and delta_h better predicts physical residual/delta.
- Moderate h-next R2 remains a blocker for formal predictor integration.
"""
    write_report("04_h_dynamics_eval.md", text)


if __name__ == "__main__":
    main()
