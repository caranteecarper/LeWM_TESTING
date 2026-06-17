import torch

from tworoom_v3_common import *
from tworoom_v3_evaluate_h_hard_subset import get_h_for_key


def calibration(pred, target):
    rows = {}
    for i, axis in enumerate(["x", "y"]):
        x = pred[:, i]
        y = target[:, i]
        xb = torch.stack([x, torch.ones_like(x)], 1)
        sol = torch.linalg.lstsq(xb, y).solution
        rows[f"{axis}_slope"] = sol[0].item()
        rows[f"{axis}_intercept"] = sol[1].item()
    return rows


def run_task(name, x, y, tr, va, te):
    rows = []
    states = {}
    for model in ["linear", "mlp", "kanfis"]:
        row, state = train_eval_regressor(name, model, x, y, tr, va, te, epochs=12 if model == "kanfis" else 8, lr=2e-3 if model == "kanfis" else 1e-3, max_train=160000)
        rows.append(row)
        states[model] = state
    return rows, states


def main():
    ensure_dirs()
    data = load_teacher()
    summary = torch.load(DYN_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    h, hn = get_h_for_key(data, key)
    dh = hn - h
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    ha = torch.cat([h, data["action"]], 1)
    tasks = [
        ("h_to_position", h, data["position_t"]),
        ("h_action_to_delta", ha, data["delta_position"]),
        ("h_action_to_residual", ha, data["residual"]),
        ("h_action_to_delta_h_first2", ha, dh[:, :2]),
    ]
    rows = []
    states = {}
    for name, x, y in tasks:
        r, st = run_task(name, x, y, tr, va, te)
        rows += r
        states[name] = st

    # Importance mask check for KANFIS h_action residual.
    kstate = states["h_action_to_residual"].get("kanfis", {})
    mask_row = {}
    if "state_dict" in kstate:
        model = KANFISStyleTSKRegressor(ha.shape[1], out_dim=2, rules=16)
        model.load_state_dict(kstate["state_dict"])
        imp = model.importance()
        top = torch.topk(imp, k=min(4, imp.numel())).indices
        x_std = (ha - kstate["x_mean"]) / kstate["x_std"].clamp_min(1e-6)
        pred = inverse_standardize(batched_predict(model, x_std[te]), kstate["y_mean"], kstate["y_std"])
        xx = x_std.clone()
        xx[:, top] = 0.0
        pred_mask = inverse_standardize(batched_predict(model, xx[te]), kstate["y_mean"], kstate["y_std"])
        base = metric_dict(pred, data["residual"][te])
        masked = metric_dict(pred_mask, data["residual"][te])
        mask_row = {"top_importance_dims": top.tolist(), "base_mse": base["overall_mse"], "masked_mse": masked["overall_mse"], "mse_increase": masked["overall_mse"] - base["overall_mse"], **model.stats(), **calibration(pred, data["residual"][te])}

    torch.save({"candidate": key, "states": states, "mask_row": mask_row}, V3_OUT / "kanfis_on_best_h.pt")
    fields = ["name", "model", "x_mse", "y_mse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    text = f"""# KANFIS On Best H

Candidate h: `{key}`. Raw z192 is not used.

{table_md(rows, fields)}

## KANFIS Importance Mask Check

```text
{mask_row}
```

## Answers

- KANFIS is useful only if it approaches Linear/MLP on low-dimensional h and its important dimensions matter under masking.
- If KANFIS remains weaker or mask effects are small, it remains exploratory and should not drive formal integration.
"""
    write_report("06_kanfis_on_best_h.md", text)


if __name__ == "__main__":
    main()
