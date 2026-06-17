import torch
import torch.nn.functional as F
from torch import nn

from tworoom_v4_common import *


def predict_rule_q(data, model_path):
    model, obj = load_rule_model(model_path)
    norm = obj["norm"]
    h = (data["h_t"].float() - norm["h_t_mean"]) / norm["h_t_std"].clamp_min(1e-6)
    action = (data["action"].float() - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qs = []
    with torch.no_grad():
        model.to(device).eval()
        for i in range(0, h.shape[0], 16384):
            qs.append(model(h[i : i + 16384].to(device), action[i : i + 16384].to(device))["q"].cpu())
        model.cpu()
    return torch.cat(qs, 0), obj


def subset_index(data, name):
    te = data["test_idx"]
    if name == "all_test":
        return te
    labels = data["hard_labels"]
    names = data["hard_label_names"]
    idx = names.index(name)
    return te[labels[te, idx] > 0.5]


def eval_vector_model(x, y, tr, va, te, model_name, epochs=6):
    x_std, xm, xs = standardize_from_train(x, tr)
    y_std, ym, ys = standardize_from_train(y, tr)
    if model_name == "linear":
        w, b = fit_linear(x_std, y_std, tr)
        pred = inverse_standardize(predict_linear(x_std[te], w, b), ym, ys)
        return pred
    model = MLPRegressor(x.shape[1], hidden=256, out_dim=y.shape[1])
    model, _, _ = train_model(model, x_std, y_std, tr, va, epochs=epochs, max_train=160000)
    return inverse_standardize(batched_predict(model, x_std[te]), ym, ys)


def eval_binary(x, y, tr, va, te, model_name):
    x_std, _, _ = standardize_from_train(x, tr)
    if model_name == "linear":
        model = nn.Linear(x.shape[1], y.shape[1])
    else:
        model = MLPRegressor(x.shape[1], hidden=256, out_dim=y.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_idx = tr
    if train_idx.numel() > 160000:
        g = torch.Generator().manual_seed(3072)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:160000]]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_std[train_idx], y[train_idx].float()), batch_size=4096, shuffle=True)
    for _ in range(5):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    logits = batched_predict(model, x_std[te])
    return logits


def gap_closure(mse_action, mse_model, mse_teacher):
    denom = max(mse_action - mse_teacher, 1e-12)
    return (mse_action - mse_model) / denom


def main():
    ensure_dirs()
    data = load_best_h_dataset()
    summary = torch.load(RULE_DIR / "summary.pt", map_location="cpu")
    best_key = summary["best_key"]
    q, q_obj = predict_rule_q(data, RULE_DIR / f"{best_key}.pt")
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    inputs = {
        "action_only": data["action"].float(),
        "h_action": torch.cat([data["h_t"].float(), data["action"].float()], 1),
        "q_action": torch.cat([q.float(), data["action"].float()], 1),
        "h_q_action": torch.cat([data["h_t"].float(), q.float(), data["action"].float()], 1),
        "z_action_teacher": torch.cat([data["z_t"].float(), data["action"].float()], 1),
    }
    tasks = {
        "delta": data["delta_position"].float(),
        "residual": data["residual"].float(),
        "h_next": data["h_next"].float(),
        "delta_h": data["delta_h"].float(),
    }
    subset_names = ["all_test", *data["hard_label_names"]]
    rows = []
    pred_cache = {}
    for input_name, x in inputs.items():
        for task_name, y in tasks.items():
            for model_name in ["linear", "mlp"]:
                pred = eval_vector_model(x, y, tr, va, te, model_name)
                pred_cache[(input_name, task_name, model_name)] = pred
                for subset in subset_names:
                    idx_abs = subset_index(data, subset)
                    if idx_abs.numel() == 0:
                        continue
                    # pred is ordered by te, so map absolute test ids to local positions.
                    mask = torch.isin(te, idx_abs)
                    p = pred[mask]
                    target = y[idx_abs]
                    if task_name in ["delta", "residual"]:
                        m = metric_dict(p, target)
                        rows.append({"input": input_name, "task": task_name, "model": model_name, "subset": subset, **m})
                    else:
                        m = vector_r2(p, target)
                        rows.append({"input": input_name, "task": task_name, "model": model_name, "subset": subset, **m})
        for model_name in ["linear", "mlp"]:
            logits = eval_binary(x, data["hard_labels"].float(), tr, va, te, model_name)
            bm = binary_metrics(logits, data["hard_labels"][te])
            rows.append({"input": input_name, "task": "hard_labels", "model": model_name, "subset": "all_test", **bm})

    # Gap closure for residual on all_test using MLP.
    residual_rows = [r for r in rows if r.get("task") == "residual" and r.get("model") == "mlp" and r.get("subset") == "all_test"]
    by_input = {r["input"]: r["overall_mse"] for r in residual_rows}
    closure_rows = []
    if {"action_only", "q_action", "h_action", "z_action_teacher"}.issubset(by_input):
        for name in ["q_action", "h_action", "h_q_action"]:
            closure_rows.append(
                {
                    "input": name,
                    "residual_gap_closure_vs_action_to_zteacher": round(gap_closure(by_input["action_only"], by_input[name], by_input["z_action_teacher"]), 4),
                }
            )

    fields = ["input", "task", "model", "subset", "overall_mse", "x_r2", "y_r2", "mean_r2", "hard_auc", "hard_f1"]
    text = f"""# V4 q + Action Transition Diagnostics

Best q model: `{best_key}`.

## Transition Metrics

{table_md(rows, fields)}

## Residual Gap Closure

{table_md(closure_rows, ["input", "residual_gap_closure_vs_action_to_zteacher"])}

## Answers

- `q+action` should be read against `action_only`, `h+action`, and `z+action_teacher`.
- If `q+action` is clearly above `action_only` and close to `h+action`, q keeps transition-critical information.
- If `q+action` is much weaker than `h+action`, q is interpretable but lossy and should not replace h as the only prediction input yet.
"""
    write_report("03_q_transition_eval.md", text)


if __name__ == "__main__":
    main()
