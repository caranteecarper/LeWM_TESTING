import csv

import torch
import torch.nn.functional as F

from tworoom_v41_common import *


def label(v):
    if v < -0.5:
        return "Low"
    if v > 0.5:
        return "High"
    return "Medium"


def main():
    ensure_dirs()
    data = load_data()
    summary = torch.load(V41_RULE_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    model, obj = load_sharp_model(V41_RULE_DIR / f"{key}.pt")
    q = q_from_model(model, obj, data, hard_topk=True)
    te = data["test_idx"]
    centers = model.rule_layer.centers.detach().cpu()
    gates = torch.sigmoid(model.rule_layer.input_gate_logits.detach().cpu())
    usage = q[te].mean(0)
    hard = data["hard_labels"][te]
    rows = []
    for r in range(q.shape[1]):
        score = centers[r].abs() * gates
        top_dims = torch.topk(score, k=min(4, score.numel())).indices.tolist()
        ant = " AND ".join([f"h_{d} is {label(centers[r, d].item())}" for d in top_dims])
        top_active = q[te, r] >= torch.quantile(q[te, r], 0.8)
        enrich = []
        for i, name in enumerate(data["hard_label_names"]):
            base = hard[:, i].mean().item()
            active = hard[top_active, i].mean().item() if top_active.any() else 0.0
            enrich.append((name, active / max(base, 1e-8)))
        top_hard, top_enrich = max(enrich, key=lambda x: x[1])
        q_one = torch.zeros(1, q.shape[1])
        q_one[0, r] = 1.0
        qa = torch.cat([q_one, torch.zeros(1, data["action"].shape[1])], 1)
        with torch.no_grad():
            delta = inverse_standardize(model.delta_head(qa), obj["norm"]["delta_position_mean"], obj["norm"]["delta_position_std"])[0]
            res = inverse_standardize(model.res_head(qa), obj["norm"]["residual_mean"], obj["norm"]["residual_std"])[0]
        phys = "hard-motion correction factor" if top_enrich >= 1.5 else "state-partition factor"
        rows.append(
            {
                "rule_id": r,
                "antecedent": ant,
                "usage_mean": round(usage[r].item(), 5),
                "top_hard_subset": top_hard,
                "hard_enrichment": round(top_enrich, 4),
                "delta_x_effect": round(delta[0].item(), 5),
                "delta_y_effect": round(delta[1].item(), 5),
                "residual_x_effect": round(res[0].item(), 5),
                "residual_y_effect": round(res[1].item(), 5),
                "associated_h_dims": ",".join(map(str, top_dims)),
                "physical_posthoc_label": phys,
                "confidence": "candidate",
            }
        )
    csv_path = V41_RULE_TABLE_DIR / "v41_rule_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    top_rows = sorted(rows, key=lambda r: r["usage_mean"] * r["hard_enrichment"], reverse=True)[:12]
    text = f"""# V4.1 Exported Rules

Best V4.1 q model: `{key}`.

Rule table saved to `{csv_path}`.

{table_md(top_rows, ["rule_id", "antecedent", "usage_mean", "top_hard_subset", "hard_enrichment", "residual_x_effect", "residual_y_effect", "physical_posthoc_label", "confidence"])}

## White-Box Check

Rule consequents are direct linear heads from `[q, action]` to physical targets. No hidden MLP and no q->h path is used in rule export.
"""
    write_report("04_exported_rules.md", text)


if __name__ == "__main__":
    main()
