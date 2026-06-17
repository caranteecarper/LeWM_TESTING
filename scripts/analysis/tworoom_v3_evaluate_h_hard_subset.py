import torch

from tworoom_v3_common import *


def predict_state(x, st):
    xs = (x - st["x_mean"]) / st["x_std"].clamp_min(1e-6)
    return inverse_standardize(predict_linear(xs, st["W"], st["b"]), st["y_mean"], st["y_std"])


def get_h_for_key(data, key):
    st = torch.load(DYN_DIR / f"{key}.pt", map_location="cpu")
    model = DynamicsAwareExtractor(h_dim=int(st["h_dim"]), hard_dim=int(st["hard_dim"]))
    model.load_state_dict(st["state_dict"])
    zt = (data["z_t"] - st["norm"]["z_mean"]) / st["norm"]["z_std"].clamp_min(1e-6)
    zn = (data["z_next"] - st["norm"]["z_mean"]) / st["norm"]["z_std"].clamp_min(1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    hs = []
    hns = []
    with torch.no_grad():
        for i in range(0, zt.shape[0], 16384):
            hs.append(model.encode(zt[i : i + 16384].to(device)).cpu())
            hns.append(model.encode(zn[i : i + 16384].to(device)).cpu())
    model.cpu()
    return torch.cat(hs, 0).float(), torch.cat(hns, 0).float()


def fit_linear_pred_all(x, y, tr):
    x_std, xm, xs = standardize_from_train(x, tr)
    y_std, ym, ys = standardize_from_train(y, tr)
    w, b = fit_linear(x_std, y_std, tr)
    return inverse_standardize(predict_linear(x_std, w, b), ym, ys)


def rows_for_subset(label, mask, pred_map, target, action_mse, teacher_mse):
    rows = []
    denom = action_mse - teacher_mse
    for name, pred in pred_map.items():
        mse = ((pred[mask] - target[mask]) ** 2).mean().item()
        closure = (action_mse - mse) / denom if denom > 1e-12 else float("nan")
        rows.append({"subset": label, "model": name, **metric_dict(pred[mask], target[mask]), "gap_closure": closure})
    return rows


def main():
    ensure_dirs()
    data = load_teacher()
    summary = torch.load(DYN_DIR / "summary.pt", map_location="cpu")
    candidate_keys = [summary["best_key"]]
    for key in ["K8_pos_delta_residual_teacher_hard_hnext_hard_weighted", "K16_pos_delta_residual_teacher_hard_hnext_hard_weighted"]:
        if (DYN_DIR / f"{key}.pt").exists() and key not in candidate_keys:
            candidate_keys.append(key)

    tr, te = data["train_idx"], data["test_idx"]
    action_pred = data["action_only_linear_pred_delta"]
    z_teacher = data["z_action_delta_teacher_pred"]
    target = data["delta_position"]
    residual = data["residual"]
    hard = data["hard_labels"].bool()

    pred_delta = {
        "action_only_linear": action_pred,
        "z_action_teacher": z_teacher,
    }
    pred_res = {}
    for key in candidate_keys:
        h, _ = get_h_for_key(data, key)
        ha = torch.cat([h, data["action"]], 1)
        pred_delta[f"{key}_h_action_delta_linear"] = fit_linear_pred_all(ha, target, tr)
        pred_res_lin = fit_linear_pred_all(ha, residual, tr)
        pred_res[f"{key}_h_action_residual_linear"] = pred_res_lin
        pred_delta[f"{key}_residual_recon_linear"] = action_pred + pred_res_lin

    subsets = {
        "all_test": torch.zeros(target.shape[0], dtype=torch.bool).index_fill(0, te, True),
        "action_error_top20": hard[:, 0] & torch.isin(data["episode_id"], data["split"]["test_eps"]),
        "action_error_top10": hard[:, 1] & torch.isin(data["episode_id"], data["split"]["test_eps"]),
        "high_residual_top20": hard[:, 2] & torch.isin(data["episode_id"], data["split"]["test_eps"]),
        "high_residual_top10": hard[:, 3] & torch.isin(data["episode_id"], data["split"]["test_eps"]),
        "large_action_small_disp": hard[:, 4] & torch.isin(data["episode_id"], data["split"]["test_eps"]),
    }
    rows = []
    for label, mask in subsets.items():
        action_mse = ((action_pred[mask] - target[mask]) ** 2).mean().item()
        teacher_mse = ((z_teacher[mask] - target[mask]) ** 2).mean().item()
        rows += rows_for_subset(label, mask, pred_delta, target, action_mse, teacher_mse)

    fields = ["subset", "model", "x_mse", "y_mse", "x_r2", "y_r2", "overall_mse", "gap_closure"]
    text = f"""# H Hard Subset Evaluation

Candidate keys: `{candidate_keys}`

{table_md(rows, fields)}

## Answers

- v3 h is better than old pure h only if its residual reconstruction improves hard-subset MSE and closes a meaningful fraction of the z+action teacher gap.
- Gap closure is `(MSE_action_only - MSE_h_model) / (MSE_action_only - MSE_z_teacher)`; it is NaN when denominator is not positive.
- If action-only remains close to h models on hard subsets, formal LeWM integration remains No-Go.
"""
    write_report("03_h_hard_subset_eval.md", text)


if __name__ == "__main__":
    main()
