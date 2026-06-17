import torch

from tworoom_v4_common import *


def safe_corr_table(h, data, idx):
    rows = []
    targets = {
        "position_x": data["position_t"][:, 0],
        "position_y": data["position_t"][:, 1],
        "delta_x": data["delta_position"][:, 0],
        "delta_y": data["delta_position"][:, 1],
        "residual_x": data["residual"][:, 0],
        "residual_y": data["residual"][:, 1],
    }
    for i, name in enumerate(data["hard_label_names"]):
        targets[f"hard_{name}"] = data["hard_labels"][:, i]
    for dim in range(h.shape[1]):
        vals = []
        for name, target in targets.items():
            vals.append((name, abs(pearson(h[idx, dim], target[idx]))))
        best_name, best_abs = max(vals, key=lambda x: x[1])
        rows.append({"h_dim": dim, "top_target": best_name, "top_abs_pearson": round(best_abs, 4)})
    return rows


def main():
    ensure_dirs()
    data = torch.load(TEACHER_PATH, map_location="cpu")
    split = torch.load(SPLIT_PATH, map_location="cpu")
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    model, obj, model_path = load_best_h_model()

    h = encode_best_h(data["z_t"])
    h_next = encode_best_h(data["z_next"])
    delta_h = h_next - h
    out = dict(data)
    out.update(
        {
            "h_t": h,
            "h_next": h_next,
            "delta_h": delta_h,
            "best_h_key": BEST_H_KEY,
            "best_h_checkpoint": str(model_path),
            "split": split,
        }
    )
    torch.save(out, BEST_H_DATASET)

    c = corr_matrix(h[te])
    max_offdiag = (c - torch.eye(c.shape[0])).abs().max().item()
    corr_rows = safe_corr_table(h, out, te)
    summary_rows = [
        {"metric": "samples", "value": h.shape[0]},
        {"metric": "h_dim", "value": h.shape[1]},
        {"metric": "train_samples", "value": tr.numel()},
        {"metric": "val_samples", "value": va.numel()},
        {"metric": "test_samples", "value": te.numel()},
        {"metric": "h_var_min_test", "value": round(h[te].var(0).min().item(), 6)},
        {"metric": "h_var_mean_test", "value": round(h[te].var(0).mean().item(), 6)},
        {"metric": "h_abs_corr_max_offdiag_test", "value": round(max_offdiag, 6)},
        {"metric": "delta_h_norm_mean_test", "value": round(delta_h[te].norm(dim=1).mean().item(), 6)},
    ]

    text = f"""# V4 Best-H Dataset

## Source

- best h key: `{BEST_H_KEY}`
- best h checkpoint: `{model_path}`
- output artifact: `{BEST_H_DATASET}`

## Summary

{table_md(summary_rows, ["metric", "value"])}

## H Dimension Top Correlations On Test

{table_md(corr_rows, ["h_dim", "top_target", "top_abs_pearson"])}

## Constraint Check

- `h_t` and `h_next` are produced by the v3 best extractor from `z_t` and `z_next` only.
- Action is copied into the dataset for later diagnostic heads, but action is not used to generate h.
- No official LeWM encoder, predictor, loss, train script, module, or TwoRoom environment is modified.
"""
    write_report("01_best_h_dataset.md", text)


if __name__ == "__main__":
    main()
