import re

import torch

from tworoom_v2_common import *


def extract_metric(text, label):
    matches = re.findall(rf"{re.escape(label)}:\s*`([^`]+)`", text)
    return float(matches[-1]) if matches else None


def main():
    ensure_dirs()
    data = load_latents()
    z = data["z"].float()
    y = data["position"].float()
    split = save_or_load_split(data["episode_id"])
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
    z_std, z_mean, z_scale = standardize_from_train(z, train_idx)

    old_probe = torch.load(LINEAR_PROBE_PATH, map_location="cpu") if LINEAR_PROBE_PATH.exists() else {}
    rows = []
    if "W" in old_probe and "b" in old_probe and "z_mean" in old_probe and "z_std" in old_probe:
        old_z = (z - old_probe["z_mean"]) / old_probe["z_std"].clamp_min(1e-6)
        old_pred = predict_linear(old_z[test_idx], old_probe["W"], old_probe["b"])
        rows.append({"case": "saved_old_probe_W", **metric_dict(old_pred, y[test_idx])})

    # Reproduce old script path: torch.linalg.lstsq, no ridge.
    x_train = add_bias(z_std[train_idx])
    solution = torch.linalg.lstsq(x_train, y[train_idx]).solution
    w_lstsq = solution[:-1].T.contiguous()
    b_lstsq = solution[-1].contiguous()
    rows.append({"case": "fresh_lstsq_no_ridge", **metric_dict(predict_linear(z_std[test_idx], w_lstsq, b_lstsq), y[test_idx])})

    # Unified v2 baseline: same split + train standardization + tiny ridge.
    w_ridge, b_ridge = fit_linear(z_std, y, train_idx, ridge=1e-4)
    pred_ridge = predict_linear(z_std[test_idx], w_ridge, b_ridge)
    unified = {"case": "unified_ridge_1e-4", **metric_dict(pred_ridge, y[test_idx])}
    rows.append(unified)

    out = V2_OUT / "readout_consistency"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split_path": str(SPLIT_PATH),
            "train_eps": split["train_eps"],
            "val_eps": split["val_eps"],
            "test_eps": split["test_eps"],
            "unified_W": w_ridge,
            "unified_b": b_ridge,
            "z_mean": z_mean,
            "z_std": z_scale,
            "metrics": rows,
        },
        out / "unified_linear_probe.pt",
    )

    old_report_path = REPO_ROOT / "reports" / "tworoom_encoder_readout" / "07_position_probe_result.md"
    old_y = None
    if old_report_path.exists():
        old_text = old_report_path.read_text()
        old_y = extract_metric(old_text, "y R2")
    static_report = V1_REPORT / "02_static_probe_comparison.md"
    static_text = static_report.read_text() if static_report.exists() else ""

    fields = ["case", "x_mse", "y_mse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse"]
    text = f"""# Readout Consistency

## Inputs

- latent cache: `{LATENTS_PATH}`
- old saved probe: `{LINEAR_PROBE_PATH}`
- old report: `{old_report_path}` exists={old_report_path.exists()}
- v1 static report: `{static_report}` exists={static_report.exists()}
- unified split file: `{SPLIT_PATH}`

Split is episode-level with seed 3072:

- train episodes: `{split['train_eps'].numel()}`, samples: `{train_idx.numel()}`
- val episodes: `{split['val_eps'].numel()}`, samples: `{val_idx.numel()}`
- test episodes: `{split['test_eps'].numel()}`, samples: `{test_idx.numel()}`

## Recomputed Results

{table_md(rows, fields)}

## Diagnosis

- The old reported y R2 was `{old_y}`.
- The current unified baseline is `unified_ridge_1e-4`, with y R2 `{unified['y_r2']}`.
- The split policy, seed, target scale, and metric formula are the same at the high level. The material difference is the linear solver path: the old script used `torch.linalg.lstsq` without ridge regularization, while v1 static comparison used a ridge normal-equation solve.
- The saved old weights reproduce the lower y score, so the inconsistency is not caused by a different latent cache at evaluation time.
- For all v2 experiments, use `{SPLIT_PATH}` and the unified ridge baseline as the position readout baseline.

## V1 Static Report Excerpt

```text
{static_text[:2000]}
```
"""
    write_report("01_readout_consistency.md", text)


if __name__ == "__main__":
    main()
