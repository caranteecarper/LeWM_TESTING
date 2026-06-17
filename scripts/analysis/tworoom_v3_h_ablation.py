import torch

from tworoom_v3_common import *
from tworoom_v3_evaluate_h_hard_subset import get_h_for_key, fit_linear_pred_all


def linear_state(x, y, tr):
    x_std, xm, xs = standardize_from_train(x, tr)
    y_std, ym, ys = standardize_from_train(y, tr)
    w, b = fit_linear(x_std, y_std, tr)
    return {"W": w, "b": b, "xm": xm, "xs": xs, "ym": ym, "ys": ys}


def pred_state(st, x):
    xs = (x - st["xm"]) / st["xs"].clamp_min(1e-6)
    return inverse_standardize(predict_linear(xs, st["W"], st["b"]), st["ym"], st["ys"])


def ablate_rows(task, h, action, target, tr, mask, dims_to_mask):
    x = torch.cat([h, action], 1)
    st = linear_state(x, target, tr)
    base = metric_dict(pred_state(st, x)[mask], target[mask])
    rows = [{"task": task, "ablation": "none", **base, "mse_increase": 0.0}]
    for name, dims in dims_to_mask.items():
        xx = x.clone()
        xx[:, list(dims)] = 0.0
        met = metric_dict(pred_state(st, xx)[mask], target[mask])
        rows.append({"task": task, "ablation": name, **met, "mse_increase": met["overall_mse"] - base["overall_mse"]})
    return rows


def main():
    ensure_dirs()
    data = load_teacher()
    summary = torch.load(DYN_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    if "K16" not in key and (DYN_DIR / "K16_pos_delta_residual_teacher_hard_hnext_hard_weighted.pt").exists():
        key = "K16_pos_delta_residual_teacher_hard_hnext_hard_weighted"
    h, hn = get_h_for_key(data, key)
    dh = hn - h
    tr, te = data["train_idx"], data["test_idx"]
    test_mask = torch.zeros(h.shape[0], dtype=torch.bool).index_fill(0, te, True)
    hard_mask = data["hard_labels"][:, 1].bool() & torch.isin(data["episode_id"], data["split"]["test_eps"])

    # Correlation-based groups.
    groups = {}
    scores = {}
    for i in range(h.shape[1]):
        scores[i] = {
            "position": max(abs(pearson(h[:, i], data["position_t"][:, 0])), abs(pearson(h[:, i], data["position_t"][:, 1]))),
            "residual": max(abs(pearson(h[:, i], data["residual"][:, 0])), abs(pearson(h[:, i], data["residual"][:, 1]))),
            "hard": max(abs(pearson(h[:, i], data["hard_labels"][:, j])) for j in range(data["hard_labels"].shape[1])),
            "dynamics": max(abs(pearson(dh[:, i], data["delta_position"][:, 0])), abs(pearson(dh[:, i], data["delta_position"][:, 1]))),
        }
    for group in ["position", "residual", "hard", "dynamics"]:
        groups[f"group_{group}_top4"] = [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1][group], reverse=True)[:4]]
    g = torch.Generator().manual_seed(3072)
    for r in range(5):
        groups[f"random4_{r}"] = torch.randperm(h.shape[1], generator=g)[:4].tolist()
    for i in range(h.shape[1]):
        groups[f"mask_h{i}"] = [i]

    rows = []
    rows += ablate_rows("delta_all_test", h, data["action"], data["delta_position"], tr, test_mask, groups)
    rows += ablate_rows("residual_all_test", h, data["action"], data["residual"], tr, test_mask, groups)
    rows += ablate_rows("residual_hard_top10", h, data["action"], data["residual"], tr, hard_mask, groups)
    # h-next first two dims for compact scalar table.
    rows += ablate_rows("delta_h_first2_all_test", h, data["action"], dh[:, :2], tr, test_mask, groups)

    group_rows = [{"dim": i, **scores[i]} for i in range(h.shape[1])]
    text = f"""# H Ablation

Candidate: `{key}`

## Dimension Group Scores

{table_md(group_rows, ['dim', 'position', 'residual', 'hard', 'dynamics'])}

## Mask Results

{table_md(rows, ['task', 'ablation', 'x_mse', 'y_mse', 'x_r2', 'y_r2', 'overall_mse', 'mse_increase'])}

## Answers

- h dimensions are necessary only when masking them increases residual or hard-subset MSE more than random masks.
- Group masks compare position/residual/hard/dynamics-correlated h dimensions against random controls.
- If mask effects are small, h remains useful as a readout but not yet a necessary predictor state.
"""
    write_report("05_h_ablation.md", text)


if __name__ == "__main__":
    main()
