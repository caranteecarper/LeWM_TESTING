import torch
import torch.nn.functional as F
from torch import nn

from tworoom_v41_common import *


def q_from_v4(data):
    summary = torch.load(V4_RULE_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    model, obj = load_rule_model(V4_RULE_DIR / f"{key}.pt")
    norm = obj["norm"]
    h = (data["h_t"].float() - norm["h_t_mean"]) / norm["h_t_std"].clamp_min(1e-6)
    action = (data["action"].float() - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    q = []
    with torch.no_grad():
        for i in range(0, h.shape[0], 16384):
            q.append(model(h[i : i + 16384], action[i : i + 16384])["q"])
    return key, torch.cat(q, 0)


def q_from_v41(data):
    summary = torch.load(V41_RULE_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    model, obj = load_sharp_model(V41_RULE_DIR / f"{key}.pt")
    return key, q_from_model(model, obj, data, hard_topk=True)


def train_eval_vec(name, model_name, x, y, tr, va, te):
    x_std, xm, xs = standardize_from_train(x, tr)
    y_std, ym, ys = standardize_from_train(y, tr)
    if model_name == "linear":
        w, b = fit_linear(x_std, y_std, tr)
        pred = inverse_standardize(predict_linear(x_std[te], w, b), ym, ys)
    else:
        model = MLPRegressor(x.shape[1], hidden=256, out_dim=y.shape[1])
        model, _, _ = train_model(model, x_std, y_std, tr, va, epochs=7, max_train=180000)
        pred = inverse_standardize(batched_predict(model, x_std[te]), ym, ys)
    return pred


def train_eval_hard(x, y, tr, te):
    x_std, _, _ = standardize_from_train(x, tr)
    model = MLPRegressor(x.shape[1], hidden=256, out_dim=y.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_idx = tr
    if train_idx.numel() > 180000:
        g = torch.Generator().manual_seed(4101)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:180000]]
    loader = DataLoader(TensorDataset(x_std[train_idx], y[train_idx].float()), batch_size=4096, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(6):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return batched_predict(model, x_std[te])


def active_count(q):
    return (q > (1.0 / q.shape[1])).float().sum(1).mean().item()


def main():
    ensure_dirs()
    data = load_data()
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    v4_key, q4 = q_from_v4(data)
    v41_key, q41 = q_from_v41(data)
    inputs = {
        "action_only": data["action"].float(),
        "q_v4_action": torch.cat([q4, data["action"].float()], 1),
        "q_v41_action": torch.cat([q41, data["action"].float()], 1),
        "h_action": torch.cat([data["h_t"].float(), data["action"].float()], 1),
        "h_q_v41_action": torch.cat([data["h_t"].float(), q41, data["action"].float()], 1),
        "z_action_teacher": torch.cat([data["z_t"].float(), data["action"].float()], 1),
    }
    rows = []
    subset_names = ["all_test", *data["hard_label_names"]]
    for input_name, x in inputs.items():
        for task, y in [("delta", data["delta_position"].float()), ("residual", data["residual"].float())]:
            pred = train_eval_vec(input_name, "mlp", x, y, tr, va, te)
            for subset in subset_names:
                idx = subset_index(data, subset)
                mask = torch.isin(te, idx)
                m = metric_dict(pred[mask], y[idx])
                rows.append({"input": input_name, "task": task, "subset": subset, **m})
        logits = train_eval_hard(x, data["hard_labels"].float(), tr, te)
        rows.append({"input": input_name, "task": "hard", "subset": "all_test", **binary_metrics(logits, data["hard_labels"][te])})

    by = {(r["input"], r["task"], r["subset"]): r for r in rows}
    action_mse = by[("action_only", "residual", "all_test")]["overall_mse"]
    teacher_mse = by[("z_action_teacher", "residual", "all_test")]["overall_mse"]
    gap_rows = []
    for name in ["q_v4_action", "q_v41_action", "h_action", "h_q_v41_action"]:
        mse = by[(name, "residual", "all_test")]["overall_mse"]
        gap = (action_mse - mse) / max(action_mse - teacher_mse, 1e-12)
        gap_rows.append({"input": name, "residual_mse": mse, "residual_gap_closure": gap})

    summary_rows = [
        {"metric": "v4_best", "value": v4_key},
        {"metric": "v41_best", "value": v41_key},
        {"metric": "v4_active_rule_count", "value": active_count(q4[te])},
        {"metric": "v41_active_rule_count", "value": active_count(q41[te])},
        {"metric": "v41_all_residual_mse_target", "value": "<=0.34"},
        {"metric": "v41_action_error_top10_target", "value": "<=2.10"},
        {"metric": "v41_large_action_small_disp_target", "value": "<=0.75"},
    ]
    fields = ["input", "task", "subset", "overall_mse", "x_r2", "y_r2", "hard_auc", "hard_f1"]
    text = f"""# V4 vs V4.1 q Evaluation

## Summary

{table_md(summary_rows, ["metric", "value"])}

## Residual Gap Closure

{table_md(gap_rows, ["input", "residual_mse", "residual_gap_closure"])}

## Metrics

{table_md(rows, fields)}
"""
    write_report("03_v4_vs_v41_eval.md", text)


if __name__ == "__main__":
    main()
