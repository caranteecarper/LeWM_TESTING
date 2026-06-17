import torch

from tworoom_v4_common import *


def load_best_q():
    data = load_best_h_dataset()
    summary = torch.load(RULE_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    model, obj = load_rule_model(RULE_DIR / f"{key}.pt")
    norm = obj["norm"]
    h = (data["h_t"].float() - norm["h_t_mean"]) / norm["h_t_std"].clamp_min(1e-6)
    action = (data["action"].float() - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    q = []
    with torch.no_grad():
        for i in range(0, h.shape[0], 16384):
            q.append(model(h[i : i + 16384], action[i : i + 16384])["q"])
    return data, key, model, obj, torch.cat(q, 0)


def eval_heads(model, obj, data, q, rule_mask=None, subset=None):
    norm = obj["norm"]
    te = data["test_idx"] if subset is None else subset
    qx = q[te].clone()
    if rule_mask is not None:
        qx[:, rule_mask] = 0.0
        qx = qx / qx.sum(1, keepdim=True).clamp_min(1e-8)
    action = (data["action"][te].float() - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    qa = torch.cat([qx, action], 1)
    with torch.no_grad():
        delta = inverse_standardize(model.delta_head(qa), norm["delta_position_mean"], norm["delta_position_std"])
        residual = inverse_standardize(model.res_head(qa), norm["residual_mean"], norm["residual_std"])
        hard_logits = model.hard_head(qa)
        hnext = inverse_standardize(model.hnext_head(qa), norm["h_next_mean"], norm["h_next_std"])
    return {
        "delta_mse": vector_mse(delta, data["delta_position"][te]),
        "residual_mse": vector_mse(residual, data["residual"][te]),
        "hnext_mse": vector_mse(hnext, data["h_next"][te]),
        **binary_metrics(hard_logits, data["hard_labels"][te]),
    }


def main():
    ensure_dirs()
    data, key, model, obj, q = load_best_q()
    te = data["test_idx"]
    hard = data["hard_labels"][te]
    hard_names = data["hard_label_names"]
    usage = q[te].mean(0)
    high_usage = torch.topk(usage, k=min(4, usage.numel())).indices
    enrich_score = []
    for r in range(q.shape[1]):
        active = q[te, r] >= torch.quantile(q[te, r], 0.8)
        score = 0.0
        for i in range(hard.shape[1]):
            base = hard[:, i].mean().item()
            score = max(score, (hard[active, i].mean().item() if active.any() else 0.0) / max(base, 1e-8))
        enrich_score.append(score)
    high_enrich = torch.topk(torch.tensor(enrich_score), k=min(4, q.shape[1])).indices
    with torch.no_grad():
        effects = []
        for r in range(q.shape[1]):
            q_one = torch.zeros(1, q.shape[1])
            q_one[0, r] = 1.0
            qa = torch.cat([q_one, torch.zeros(1, data["action"].shape[1])], 1)
            effects.append(model.res_head(qa).norm().item())
    high_effect = torch.topk(torch.tensor(effects), k=min(4, q.shape[1])).indices
    g = torch.Generator().manual_seed(3072)
    random_rules = torch.randperm(q.shape[1], generator=g)[: min(4, q.shape[1])]
    masks = {
        "none": None,
        "top_usage": high_usage,
        "top_hard_enriched": high_enrich,
        "top_residual_effect": high_effect,
        "random": random_rules,
    }
    rows = []
    base = eval_heads(model, obj, data, q)
    for name, mask in masks.items():
        m = eval_heads(model, obj, data, q, mask)
        rows.append(
            {
                "mask": name,
                "rules": "" if mask is None else ",".join(map(str, mask.tolist())),
                **m,
                "residual_mse_increase": m["residual_mse"] - base["residual_mse"],
                "delta_mse_increase": m["delta_mse"] - base["delta_mse"],
                "hnext_mse_increase": m["hnext_mse"] - base["hnext_mse"],
            }
        )

    subset_rows = []
    for i, name in enumerate(hard_names):
        subset = te[hard[:, i] > 0.5]
        if subset.numel() == 0:
            continue
        b = eval_heads(model, obj, data, q, subset=subset)
        for mask_name in ["top_usage", "top_hard_enriched", "top_residual_effect", "random"]:
            m = eval_heads(model, obj, data, q, masks[mask_name], subset)
            subset_rows.append(
                {
                    "subset": name,
                    "mask": mask_name,
                    "residual_mse_increase": m["residual_mse"] - b["residual_mse"],
                    "delta_mse_increase": m["delta_mse"] - b["delta_mse"],
                }
            )

    text = f"""# V4 Rule And q Ablation

Best q model: `{key}`.

## All-Test Ablation

{table_md(rows, ["mask", "rules", "residual_mse", "residual_mse_increase", "delta_mse", "delta_mse_increase", "hnext_mse", "hnext_mse_increase", "hard_auc", "hard_f1"])}

## Hard-Subset Ablation

{table_md(subset_rows, ["subset", "mask", "residual_mse_increase", "delta_mse_increase"])}

## Answers

- If top rule masks increase residual or hard-subset MSE more than random masks, the rules have functional contribution.
- If random and top masks are similar, q may still compress h but rule identity is not yet functionally stable.
"""
    write_report("05_rule_ablation.md", text)


if __name__ == "__main__":
    main()
