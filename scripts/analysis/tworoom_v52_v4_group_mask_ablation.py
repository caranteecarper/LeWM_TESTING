from tworoom_v52_common import *


def apply_mask(q, mask_rules):
    x = q.clone()
    x[:, mask_rules] = 0.0
    return x


def main():
    ensure_dirs()
    vis = load_visual()
    action = vis["action"].float() if "action" in vis else vis["z_t"][:, :0]
    q = vis["q_v4_t"].float()
    x_full = torch.cat([q, action], 1)
    _, states = eval_input(vis, x_full, "full_q_v4_action", subset_names=["all_test"])
    wr, br = states["res"]
    wd, bd = states["delta"]
    wh, bh = states["hard"]
    masks = {
        "none": [],
        "mask_group0": GROUPS["group_0"],
        "mask_top3_rules": [15, 2, 14],
        "mask_top_hard_rules": [15, 2, 14, 12],
        "mask_top_residual_rules": [15, 2, 14, 0],
        "mask_random_same_size": [4, 6, 9, 13],
    }
    for r in range(16):
        masks[f"mask_rule_{r}"] = [r]
    rows = []
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
            rows.append(
                {
                    "mask": name,
                    "rules": ",".join(map(str, rules)),
                    "subset": subset,
                    "residual_mse_increase": res_mse - base[subset]["res"],
                    "delta_mse_increase": delta_mse - base[subset]["delta"],
                    "hard_f1_drop": base[subset]["hard"] - hard_f1,
                }
            )
    csv_write(V52_TABLE / "v4_group_mask_ablation.csv", rows)
    text = f"""# V4 Group Mask Ablation

{table_md(rows[:80], ["mask", "rules", "subset", "residual_mse_increase", "delta_mse_increase", "hard_f1_drop"])}

## Answers

Compare `mask_group0` against `mask_random_same_size`. If group 0 causes larger residual increase, it has functional support beyond visualization.
"""
    write_report("04_v4_group_mask_ablation.md", text)


if __name__ == "__main__":
    main()

