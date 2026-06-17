import csv

from tworoom_v41_common import *


def main():
    ensure_dirs()
    data = load_data()
    key = torch.load(V41_RULE_DIR / "summary.pt", map_location="cpu")["best_key"]
    model, obj = load_sharp_model(V41_RULE_DIR / f"{key}.pt")
    q = q_from_model(model, obj, data, hard_topk=True)
    te = data["test_idx"]
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
    rows = []
    matrix = []
    for i in range(q.shape[1]):
        vals = []
        for name, target in targets.items():
            c = pearson(q[te, i], target[te])
            matrix.append({"q_dim": i, "target": name, "pearson": c, "abs_pearson": abs(c)})
            vals.append((name, c, abs(c)))
        top, c, a = max(vals, key=lambda x: x[2])
        if "position_x" == top and a >= 0.6:
            label = "position-x-related factor"
        elif "position_y" == top and a >= 0.6:
            label = "position-y-related factor"
        elif "residual" in top and a >= 0.35:
            label = "residual correction factor"
        elif "hard" in top and a >= 0.25:
            label = "hard-motion correction factor"
        else:
            label = "unknown / mixed factor"
        rows.append({"q_dim": i, "top_target": top, "pearson": round(c, 4), "abs_pearson": round(a, 4), "posthoc_label": label})
    csv_path = V41_OUT / "q_semantics_corr_matrix.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["q_dim", "target", "pearson", "abs_pearson"])
        writer.writeheader()
        writer.writerows(matrix)
    top_rows = sorted(rows, key=lambda r: r["abs_pearson"], reverse=True)
    text = f"""# V4.1 q Semantics

Best V4.1 q model: `{key}`.

Correlation matrix saved to `{csv_path}`.

{table_md(top_rows, ["q_dim", "top_target", "pearson", "abs_pearson", "posthoc_label"])}

## Naming Rule

No q dimension is named as wall, door, or collision unless enrichment, residual effect, and ablation jointly support it.
"""
    write_report("06_q_semantics.md", text)


if __name__ == "__main__":
    main()
