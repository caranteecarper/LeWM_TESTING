from tworoom_failure_c_v1_common import *


def best_by_group(rows, label):
    out = {}
    for r in rows:
        if r["label"] != label:
            continue
        g = r["feature_group"]
        if g not in out or float(r["AUPRC"]) > float(out[g]["AUPRC"]):
            out[g] = r
    return out


def delta(a, b, key):
    return float(a[key]) - float(b[key]) if a and b else float("nan")


def main():
    ensure_dirs()
    rows = read_csv(TABLE / "eval_metrics.csv")
    label = PRIMARY_LABEL
    best = best_by_group(rows, label)
    summary = []
    for group, row in sorted(best.items()):
        summary.append(row)
    csv_write(TABLE / "q_contribution_summary.csv", summary)
    hqa = best.get("h_q_action")
    ha = best.get("h_action")
    qa = best.get("q_action")
    act = best.get("action_only")
    hshuf = best.get("h_shuffled_q_action")
    qshuf = best.get("q_shuffled_action")
    group_only = best.get("h_q_group_action")
    za = best.get("z_action")
    p10_ratio = float(hqa["precision@10/base_rate"]) if hqa else 0.0
    go = (
        hqa
        and ha
        and float(hqa["AUPRC"]) > float(ha["AUPRC"]) + 0.01
        and hshuf
        and float(hqa["AUPRC"]) > float(hshuf["AUPRC"]) + 0.01
        and qa
        and act
        and float(qa["AUPRC"]) > float(act["AUPRC"]) + 0.01
        and p10_ratio >= 1.5
    )
    partial = (hqa and ha and float(hqa["AUPRC"]) > float(ha["AUPRC"])) or (qa and act and float(qa["AUPRC"]) > float(act["AUPRC"]))
    verdict = "Go" if go else ("Partial" if partial else "No-Go")
    text = f"""# q_v42 独立贡献分析

## 各输入组合最佳结果（标签：{label}）

{table_md(summary, ["feature_group", "model_type", "seed", "base_rate", "AUROC", "AUPRC", "precision@top10_predicted_risk", "recall@top10_label", "precision@10/base_rate", "AUPRC/base_rate"])}

## 关键对比

- h_q_action vs h_action AUPRC 差值：`{delta(hqa, ha, 'AUPRC'):.4f}`
- q_action vs action_only AUPRC 差值：`{delta(qa, act, 'AUPRC'):.4f}`
- h_q_action vs h_shuffled_q_action AUPRC 差值：`{delta(hqa, hshuf, 'AUPRC'):.4f}`
- q_action vs q_shuffled_action AUPRC 差值：`{delta(qa, qshuf, 'AUPRC'):.4f}`
- h_q_group_action 最佳 AUPRC：`{float(group_only['AUPRC']) if group_only else float('nan'):.4f}`
- z_action 上限 AUPRC：`{float(za['AUPRC']) if za else float('nan'):.4f}`

## 判断

当前阶段判断：`{verdict}`

解释：Go 需要 q 在 full/shuffled/action 对照中都有明确增益，且 precision@10 至少达到 base rate 的 1.5 倍。本阶段仍只是 failure confidence head，不是 residual adapter，也不是 causal intervention。
"""
    write_report("04_q_contribution.md", text)


if __name__ == "__main__":
    main()
