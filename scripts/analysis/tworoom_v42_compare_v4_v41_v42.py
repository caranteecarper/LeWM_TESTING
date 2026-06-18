from tworoom_v42_common import *


def eval_rep(data, qv4, qv41, qv42):
    # Use visual-sized diagnostics if full cache is too large? Here use full test with ridge heads.
    action = data["action"].float()
    h = data["h_t"].float()
    inputs = {
        "action_only": action,
        "q_v4_action": torch.cat([qv4, action], 1),
        "q_v41_action": torch.cat([qv41, action], 1),
        "q_v42_action": torch.cat([qv42, action], 1),
        "h_action": torch.cat([h, action], 1),
        "h_q_v4_action": torch.cat([h, qv4, action], 1),
        "h_q_v42_action": torch.cat([h, qv42, action], 1),
    }
    # Adapt V52 evaluator needs visual split names; use direct indices.
    tr, te = data["train_idx"], data["test_idx"]
    rows = []
    for name, x in inputs.items():
        wr, br = fit_ridge(x, data["residual"], tr)
        wd, bd = fit_ridge(x, data["delta_position"], tr)
        wh, bh = fit_ridge(x, data["hard_labels"], tr)
        for subset in ["all_test", "action_error_top10", "high_residual_top10", "large_action_small_disp"]:
            idx = te if subset == "all_test" else te[data["hard_labels"][te, data["hard_label_names"].index(subset)] > 0.5]
            pr = pred_ridge(x[idx], wr, br)
            pd = pred_ridge(x[idx], wd, bd)
            ph = pred_ridge(x[idx], wh, bh)
            row = {"input": name, "subset": subset}
            row.update({"res_overall_mse": metric_dict(pr, data["residual"][idx])["overall_mse"]})
            row.update({"delta_overall_mse": metric_dict(pd, data["delta_position"][idx])["overall_mse"]})
            row.update(binary_metrics(ph, data["hard_labels"][idx]))
            rows.append(row)
    return rows


def main():
    ensure_dirs()
    data = load_base_data()
    v4_key, qv4 = compute_v4_q(data)
    v41_key, qv41 = compute_v41_q(data)
    v42_key, qv42 = compute_v42_q(data)
    rows = eval_rep(data, qv4, qv41, qv42)
    csv_write(V42_TABLE / "v42_comparison.csv", rows)
    text = f"""# V4 / V4.1 / V4.2 Comparison

- V4: `{v4_key}`
- V4.1: `{v41_key}`
- V4.2: `{v42_key}`

{table_md(rows, ["input", "subset", "res_overall_mse", "delta_overall_mse", "hard_acc", "hard_f1"])}

## Answers

V4.2 is useful only if q_v42+action improves over q_v41+action while keeping active rule count in the compromise range and offering clearer rules than V4.
"""
    write_report("02_compare_v4_v41_v42.md", text)


if __name__ == "__main__":
    main()

