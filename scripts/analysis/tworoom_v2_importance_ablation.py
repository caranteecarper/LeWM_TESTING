import torch

from tworoom_v2_common import *
from tworoom_v2_residual_and_hard_subset import encode_pure_h


def linear_state(x, y, tr):
    x_std, xm, xs = standardize_from_train(x, tr)
    y_std, ym, ys = standardize_from_train(y, tr)
    w, b = fit_linear(x_std, y_std, tr)
    return {"W": w, "b": b, "xm": xm, "xs": xs, "ym": ym, "ys": ys}


def pred_state(st, x):
    xs = (x - st["xm"]) / st["xs"].clamp_min(1e-6)
    return inverse_standardize(predict_linear(xs, st["W"], st["b"]), st["ym"], st["ys"])


def ablate_h_dims(task, x, y, tr, te, subset=None, h_dim=16):
    st = linear_state(x, y, tr)
    mask = subset if subset is not None else torch.zeros(x.shape[0], dtype=torch.bool).index_fill(0, te, True)
    base = metric_dict(pred_state(st, x)[mask], y[mask])
    rows = [{"task": task, "ablation": "none", **base}]
    for i in range(h_dim):
        xx = x.clone()
        xx[:, i] = 0.0
        met = metric_dict(pred_state(st, xx)[mask], y[mask])
        rows.append({"task": task, "ablation": f"mask_h{i}", "delta_overall_mse": met["overall_mse"] - base["overall_mse"], **met})
    return rows


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
    pred_file = V2_OUT / "residual_hard_subset" / "action_only_predictions.pt"
    pred_action = torch.load(pred_file, map_location="cpu")["pred_action_linear"] if pred_file.exists() else torch.zeros_like(delta)
    residual = delta - pred_action

    rows = []
    rows += ablate_h_dims("h_action_to_delta", ha, delta, tr, te)
    rows += ablate_h_dims("h_action_to_residual", ha, residual, tr, te)
    # h_next/delta_h summarized with MSE over h dims, but report as x/y slots using first two dims for compact table.
    rows += ablate_h_dims("h_action_to_delta_h_first2", ha, dh[:, :2], tr, te)

    err = ((pred_action - delta) ** 2).sum(1).sqrt()
    hard = (err >= torch.quantile(err, 0.90)) & torch.isin(pairs["episode_id"].long(), split["test_eps"].long())
    hard_rows = []
    hard_rows += ablate_h_dims("hard_top10_h_action_to_delta", ha, delta, tr, te, subset=hard)
    hard_rows += ablate_h_dims("hard_top10_h_action_to_residual", ha, residual, tr, te, subset=hard)

    # Static latent ablation with unified linear baseline.
    data = load_latents()
    lsplit = torch.load(SPLIT_PATH, map_location="cpu")
    ltr, _, lte = lsplit["train_idx"], lsplit["val_idx"], lsplit["test_idx"]
    z = data["z"].float()
    pos = data["position"].float()
    z_std, _, _ = standardize_from_train(z, ltr)
    w, b = fit_linear(z_std, pos, ltr)
    top32 = topk_from_weight(w, 32)
    var32 = torch.topk(z_std[ltr].var(0), 32).indices
    g = torch.Generator().manual_seed(3072)
    rand32 = torch.randperm(z_std.shape[1], generator=g)[:32]
    latent_rows = []
    base = metric_dict(predict_linear(z_std[lte], w, b), pos[lte])
    latent_rows.append({"ablation": "none", **base})
    for name, dims in [("mask_linear_top32", top32), ("mask_random32", rand32), ("mask_high_variance32", var32)]:
        zz = z_std.clone()
        zz[:, dims] = 0.0
        latent_rows.append({"ablation": name, **metric_dict(predict_linear(zz[lte], w, b), pos[lte])})

    fields = ["task", "ablation", "x_mse", "y_mse", "x_r2", "y_r2", "overall_mse", "delta_overall_mse"]
    lat_fields = ["ablation", "x_mse", "y_mse", "x_r2", "y_r2", "overall_mse"]
    text = f"""# Importance Ablation

## hK16 Dimension Ablation

{table_md(rows, fields)}

## Hard Subset hK16 Ablation

{table_md(hard_rows, fields)}

## Static Latent Dimension Ablation

{table_md(latent_rows, lat_fields)}

## Answers

- h dimensions with the largest positive `delta_overall_mse` are the most necessary under the linear diagnostic head.
- Hard subset rows show whether h is specifically needed where action-only has high error.
- KANFIS top32 is not reported unless independent KANFIS importance is established; v1 reused linear top32 for KANFIS top32.
"""
    write_report("07_importance_ablation.md", text)


if __name__ == "__main__":
    main()
