import csv

import torch
import torch.nn.functional as F

from tworoom_v4_common import *


def load_best():
    data = load_best_h_dataset()
    summary = torch.load(RULE_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    model, obj = load_rule_model(RULE_DIR / f"{key}.pt")
    return data, key, model, obj


def compute_q(data, model, norm):
    h = (data["h_t"].float() - norm["h_t_mean"]) / norm["h_t_std"].clamp_min(1e-6)
    action = (data["action"].float() - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    q = []
    outs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.to(device).eval()
        for i in range(0, h.shape[0], 16384):
            out = model(h[i : i + 16384].to(device), action[i : i + 16384].to(device))
            q.append(out["q"].cpu())
            outs.append({k: v.cpu() for k, v in out.items() if k != "q"})
        model.cpu()
    return q and torch.cat(q, 0), {k: torch.cat([o[k] for o in outs], 0) for k in outs[0]}


def fuzzy_label(center):
    if center < -0.5:
        return "Low"
    if center > 0.5:
        return "High"
    return "Medium"


def main():
    ensure_dirs()
    rules_dir = V4_OUT / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    data, key, model, obj = load_best()
    norm = obj["norm"]
    q, outs = compute_q(data, model, norm)
    te = data["test_idx"]
    centers = model.rule_layer.centers.detach().cpu()
    sigma = F.softplus(model.rule_layer.log_sigma.detach().cpu())
    gates = torch.sigmoid(model.rule_layer.input_gate_logits.detach().cpu())
    usage = q[te].mean(0)
    hard = data["hard_labels"][te]
    hard_names = data["hard_label_names"]
    hard_enrich = []
    for r in range(q.shape[1]):
        top = q[te, r] >= torch.quantile(q[te, r], 0.8)
        vals = {}
        for i, name in enumerate(hard_names):
            base = hard[:, i].mean().item()
            active = hard[top, i].mean().item() if top.any() else 0.0
            vals[name] = active / max(base, 1e-8)
        hard_enrich.append(vals)

    rows = []
    for r in range(centers.shape[0]):
        score = (centers[r].abs() * gates).float()
        top_dims = torch.topk(score, k=min(4, score.numel())).indices.tolist()
        antecedent = " AND ".join([f"h_{d} is {fuzzy_label(centers[r, d].item())}" for d in top_dims])
        with torch.no_grad():
            q_one = torch.zeros(1, q.shape[1])
            q_one[0, r] = 1.0
            zero_action = torch.zeros(1, data["action"].shape[1])
            qa = torch.cat([q_one, zero_action], 1)
            delta_effect = inverse_standardize(model.delta_head(qa), norm["delta_position_mean"], norm["delta_position_std"])[0]
            residual_effect = inverse_standardize(model.res_head(qa), norm["residual_mean"], norm["residual_std"])[0]
        top_hard = max(hard_enrich[r].items(), key=lambda kv: kv[1])
        rows.append(
            {
                "rule_id": r,
                "antecedent": antecedent,
                "top_h_dims": ",".join(map(str, top_dims)),
                "usage_mean": round(usage[r].item(), 5),
                "top_hard_subset": top_hard[0],
                "top_hard_enrichment": round(top_hard[1], 4),
                "delta_effect_x": round(delta_effect[0].item(), 5),
                "delta_effect_y": round(delta_effect[1].item(), 5),
                "residual_effect_x": round(residual_effect[0].item(), 5),
                "residual_effect_y": round(residual_effect[1].item(), 5),
                "mean_sigma_top_dims": round(sigma[r, top_dims].mean().item(), 5),
            }
        )

    csv_path = rules_dir / "raw_rule_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top_rows = sorted(rows, key=lambda x: x["usage_mean"] * x["top_hard_enrichment"], reverse=True)[:12]
    interpretations = []
    for row in top_rows:
        label = "hard-motion correction factor" if row["top_hard_enrichment"] >= 1.5 else "state-partition factor"
        evidence = f"usage={row['usage_mean']}, enriched in {row['top_hard_subset']} by {row['top_hard_enrichment']}x, residual effect=({row['residual_effect_x']}, {row['residual_effect_y']})"
        interpretations.append({"rule_id": row["rule_id"], "possible_meaning": label, "evidence": evidence, "confidence": "candidate"})

    text = f"""# V4 Exported KANFIS Rules

Best q model: `{key}`.

Raw rule table saved to `{csv_path}`.

## Top Raw Rules

{table_md(top_rows, ["rule_id", "antecedent", "usage_mean", "top_hard_subset", "top_hard_enrichment", "delta_effect_x", "delta_effect_y", "residual_effect_x", "residual_effect_y"])}

## Candidate Physical Interpretation

{table_md(interpretations, ["rule_id", "possible_meaning", "evidence", "confidence"])}

## Caution

These are post-hoc interpretations. A rule is not named as an obstacle or doorway concept unless hard-subset enrichment, residual effect, and ablation all support it.
"""
    write_report("04_exported_rules.md", text)


if __name__ == "__main__":
    main()
