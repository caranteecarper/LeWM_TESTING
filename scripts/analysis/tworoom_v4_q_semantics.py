import csv

import torch

from tworoom_v4_common import *


def main():
    ensure_dirs()
    fig_dir = V4_OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
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
    q = torch.cat(q, 0)
    te = data["test_idx"]
    targets = {
        "position_x": data["position_t"][:, 0],
        "position_y": data["position_t"][:, 1],
        "delta_x": data["delta_position"][:, 0],
        "delta_y": data["delta_position"][:, 1],
        "residual_x": data["residual"][:, 0],
        "residual_y": data["residual"][:, 1],
        "action_norm": data["action"].float().norm(dim=1),
        "delta_h_norm": data["delta_h"].float().norm(dim=1),
    }
    for i, name in enumerate(data["hard_label_names"]):
        targets[f"hard_{name}"] = data["hard_labels"][:, i]

    rows = []
    matrix_rows = []
    for r in range(q.shape[1]):
        vals = []
        for name, target in targets.items():
            c = pearson(q[te, r], target[te])
            matrix_rows.append({"q_dim": r, "target": name, "pearson": c, "abs_pearson": abs(c)})
            vals.append((name, c, abs(c)))
        top_name, top_c, top_abs = max(vals, key=lambda x: x[2])
        if "position" in top_name and top_abs >= 0.6:
            label = "position-related factor"
        elif "residual" in top_name and top_abs >= 0.4:
            label = "residual-correction factor"
        elif "hard" in top_name and top_abs >= 0.25:
            label = "hard-motion factor"
        elif "delta" in top_name and top_abs >= 0.4:
            label = "dynamics-related factor"
        else:
            label = "unknown or mixed factor"
        rows.append({"q_dim": r, "top_target": top_name, "pearson": round(top_c, 4), "abs_pearson": round(top_abs, 4), "posthoc_label": label})

    csv_path = V4_OUT / "q_target_corr_matrix.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["q_dim", "target", "pearson", "abs_pearson"])
        writer.writeheader()
        writer.writerows(matrix_rows)

    top_rows = sorted(rows, key=lambda x: x["abs_pearson"], reverse=True)[:16]
    text = f"""# V4 q Semantics

Best q model: `{key}`.

Full q-target correlation matrix saved to `{csv_path}`.

## Top q Semantic Alignments

{table_md(top_rows, ["q_dim", "top_target", "pearson", "abs_pearson", "posthoc_label"])}

## Answers

- q dimensions are post-hoc labeled only when correlation evidence is strong enough.
- Weak or mixed q dimensions should remain unnamed; they may still be useful structural rules.
- q should not be called x/y directly unless the evidence is very strong and ablation supports it.
"""
    write_report("06_q_semantics.md", text)


if __name__ == "__main__":
    main()
