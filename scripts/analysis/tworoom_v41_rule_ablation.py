import torch

from tworoom_v41_common import *


def eval_mask(model, obj, data, q, mask=None, subset=None):
    te = data["test_idx"] if subset is None else subset
    qx = q[te].clone()
    if mask is not None:
        qx[:, mask] = 0.0
        qx = qx / qx.sum(1, keepdim=True).clamp_min(1e-8)
    action = (data["action"][te].float() - obj["norm"]["action_mean"]) / obj["norm"]["action_std"].clamp_min(1e-6)
    qa = torch.cat([qx, action], 1)
    with torch.no_grad():
        delta = inverse_standardize(model.delta_head(qa), obj["norm"]["delta_position_mean"], obj["norm"]["delta_position_std"])
        res = inverse_standardize(model.res_head(qa), obj["norm"]["residual_mean"], obj["norm"]["residual_std"])
        hard_logits = model.hard_head(qa)
    return {
        "residual_mse": vector_mse(res, data["residual"][te]),
        "delta_mse": vector_mse(delta, data["delta_position"][te]),
        **binary_metrics(hard_logits, data["hard_labels"][te]),
    }


def main():
    ensure_dirs()
    data = load_data()
    key = torch.load(V41_RULE_DIR / "summary.pt", map_location="cpu")["best_key"]
    model, obj = load_sharp_model(V41_RULE_DIR / f"{key}.pt")
    q = q_from_model(model, obj, data, hard_topk=True)
    te = data["test_idx"]
    usage = q[te].mean(0)
    hard = data["hard_labels"][te]
    enrich = []
    for r in range(q.shape[1]):
        active = q[te, r] >= torch.quantile(q[te, r], 0.8)
        enrich.append(max(((hard[active, i].mean().item() if active.any() else 0.0) / max(hard[:, i].mean().item(), 1e-8)) for i in range(hard.shape[1])))
    effects = []
    with torch.no_grad():
        for r in range(q.shape[1]):
            q_one = torch.zeros(1, q.shape[1])
            q_one[0, r] = 1.0
            qa = torch.cat([q_one, torch.zeros(1, data["action"].shape[1])], 1)
            effects.append(model.res_head(qa).norm().item())
    k = min(4, q.shape[1])
    masks = {
        "none": None,
        "top_usage": torch.topk(usage, k=k).indices,
        "top_hard_enriched": torch.topk(torch.tensor(enrich), k=k).indices,
        "top_residual_effect": torch.topk(torch.tensor(effects), k=k).indices,
        "random": torch.randperm(q.shape[1], generator=torch.Generator().manual_seed(4101))[:k],
    }
    base = eval_mask(model, obj, data, q)
    rows = []
    for name, mask in masks.items():
        m = eval_mask(model, obj, data, q, mask)
        rows.append(
            {
                "mask": name,
                "rules": "" if mask is None else ",".join(map(str, mask.tolist())),
                **m,
                "residual_mse_increase": m["residual_mse"] - base["residual_mse"],
                "delta_mse_increase": m["delta_mse"] - base["delta_mse"],
                "hard_auc_drop": base["hard_auc"] - m["hard_auc"],
            }
        )
    subset_rows = []
    for subset in data["hard_label_names"]:
        idx = subset_index(data, subset)
        b = eval_mask(model, obj, data, q, subset=idx)
        for name in ["top_hard_enriched", "top_residual_effect", "random"]:
            m = eval_mask(model, obj, data, q, masks[name], idx)
            subset_rows.append({"subset": subset, "mask": name, "residual_mse_increase": m["residual_mse"] - b["residual_mse"]})
    text = f"""# V4.1 Rule Ablation

Best V4.1 q model: `{key}`.

## All-Test Rule Mask

{table_md(rows, ["mask", "rules", "residual_mse", "residual_mse_increase", "delta_mse", "delta_mse_increase", "hard_auc", "hard_auc_drop"])}

## Hard-Subset Rule Mask

{table_md(subset_rows, ["subset", "mask", "residual_mse_increase"])}

## Interpretation

Top hard-enriched and residual-effect masks should exceed random-mask damage if V4.1 made rules more functional.
"""
    write_report("05_rule_ablation.md", text)


if __name__ == "__main__":
    main()
