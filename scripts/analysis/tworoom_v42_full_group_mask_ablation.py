from tworoom_v42_full_common import *


def main():
    ensure_dirs()
    vis = build_visual_data()
    action = vis["action_t"].float()
    q = vis["q_v42_t"].float()
    x_full = torch.cat([q, action], 1)
    _, states = eval_input(vis, x_full, "full_q_v42_action", subset_names=["all_test"])
    wr, br = states["res"]
    wd, bd = states["delta"]
    wh, bh = states["hard"]
    masks = {
        "none": [],
        "mask_group0": V42_GROUPS["group_0"],
        "mask_best_rules": V42_GROUPS["best_rules"],
        "mask_top3_rules": V42_GROUPS["top3_rules"],
        "mask_top_hard_rules": [12, 10, 13, 1],
        "mask_top_residual_rules": [10, 13, 12, 6],
        "mask_random_same_size": [0, 2, 4, 8],
    }
    for r in range(q.shape[1]):
        masks[f"mask_rule_{r}"] = [r]
    subsets = ["all_test", "action_error_top10", "high_residual_top10", "large_action_small_disp"]
    base = {}
    for subset in subsets:
        idx = subset_idx(vis, subset)
        pr = pred_ridge(x_full[idx], wr, br)
        pd = pred_ridge(x_full[idx], wd, bd)
        ph = pred_ridge(x_full[idx], wh, bh)
        base[subset] = {
            "res": metric_dict(pr, vis["residual_t"][idx])["overall_mse"],
            "delta": metric_dict(pd, vis["delta_position"][idx])["overall_mse"],
            "hard": binary_metrics(ph, vis["hard_labels"][idx])["hard_f1"],
        }
    rows = []
    for name, rules in masks.items():
        qm = apply_mask(q, rules)
        xm = torch.cat([qm, action], 1)
        for subset in subsets:
            idx = subset_idx(vis, subset)
            pr = pred_ridge(xm[idx], wr, br)
            pd = pred_ridge(xm[idx], wd, bd)
            ph = pred_ridge(xm[idx], wh, bh)
            res_mse = metric_dict(pr, vis["residual_t"][idx])["overall_mse"]
            delta_mse = metric_dict(pd, vis["delta_position"][idx])["overall_mse"]
            hard_f1 = binary_metrics(ph, vis["hard_labels"][idx])["hard_f1"]
            action_mse = metric_dict(torch.zeros_like(vis["residual_t"][idx]), vis["residual_t"][idx])["overall_mse"]
            rows.append(
                {
                    "mask": name,
                    "rules": ",".join(map(str, rules)),
                    "subset": subset,
                    "residual_mse_increase": res_mse - base[subset]["res"],
                    "delta_mse_increase": delta_mse - base[subset]["delta"],
                    "hard_f1_drop": base[subset]["hard"] - hard_f1,
                    "gap_closure_drop": (res_mse - base[subset]["res"]) / max(action_mse - base[subset]["res"], 1e-8),
                }
            )
    csv_write(FULL_TABLE / "v42_group_mask_ablation.csv", rows)
    focus = [r for r in rows if r["mask"] in ["none", "mask_group0", "mask_best_rules", "mask_random_same_size"]]
    text = f"""# V4.2 Group Mask Ablation

## Focus Masks

{table_md(focus, ["mask", "rules", "subset", "residual_mse_increase", "delta_mse_increase", "hard_f1_drop", "gap_closure_drop"])}

## All Masks

{table_md(rows[:96], ["mask", "rules", "subset", "residual_mse_increase", "delta_mse_increase", "hard_f1_drop"])}

## Answers

- Compare `mask_group0` and `mask_best_rules` against `mask_random_same_size`.
- V4.2 group/rules are functionally supported only if their residual increase is clearly above random on hard subsets.
"""
    write_report("04_v42_group_mask_ablation.md", text)


if __name__ == "__main__":
    main()
