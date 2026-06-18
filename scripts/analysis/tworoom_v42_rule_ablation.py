from tworoom_v42_common import *


def main():
    ensure_dirs()
    data = load_base_data()
    key, q = compute_v42_q(data)
    action = data["action"].float()
    x = torch.cat([q, action], 1)
    tr, te = data["train_idx"], data["test_idx"]
    wr, br = fit_ridge(x, data["residual"], tr)
    wh, bh = fit_ridge(x, data["hard_labels"], tr)
    # choose top rules by q/residual enrichment on test
    mag = data["residual"][te].norm(dim=1)
    scores = []
    for r in range(q.shape[1]):
        active = q[te, r] >= torch.quantile(q[te, r], 0.9)
        scores.append((r, mag[active].mean().item()))
    top_rules = [r for r, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:4]]
    masks = {
        "none": [],
        "mask_best_rules": top_rules,
        "mask_random": list(range(min(4, q.shape[1]))),
    }
    rows = []
    subsets = ["all_test", "action_error_top10", "high_residual_top10", "large_action_small_disp"]
    base = {}
    for subset in subsets:
        idx = te if subset == "all_test" else te[data["hard_labels"][te, data["hard_label_names"].index(subset)] > 0.5]
        pr = pred_ridge(x[idx], wr, br)
        ph = pred_ridge(x[idx], wh, bh)
        base[subset] = {"res": metric_dict(pr, data["residual"][idx])["overall_mse"], "hard": binary_metrics(ph, data["hard_labels"][idx])["hard_f1"]}
    for name, rules in masks.items():
        qm = q.clone()
        qm[:, rules] = 0.0
        xm = torch.cat([qm, action], 1)
        for subset in subsets:
            idx = te if subset == "all_test" else te[data["hard_labels"][te, data["hard_label_names"].index(subset)] > 0.5]
            pr = pred_ridge(xm[idx], wr, br)
            ph = pred_ridge(xm[idx], wh, bh)
            rows.append({"mask": name, "rules": ",".join(map(str, rules)), "subset": subset, "residual_mse_increase": metric_dict(pr, data["residual"][idx])["overall_mse"] - base[subset]["res"], "hard_f1_drop": base[subset]["hard"] - binary_metrics(ph, data["hard_labels"][idx])["hard_f1"]})
    text = f"""# V4.2 Rule Ablation

Best V4.2 model: `{key}`.

{table_md(rows, ["mask", "rules", "subset", "residual_mse_increase", "hard_f1_drop"])}
"""
    write_report("04_v42_rule_ablation.md", text)


if __name__ == "__main__":
    main()

