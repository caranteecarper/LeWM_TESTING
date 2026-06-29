from tworoom_failure_c_v1_common import *


def main():
    ensure_dirs()
    eval_rows = read_csv(TABLE / "eval_metrics.csv")
    contrib = read_csv(TABLE / "q_contribution_summary.csv")
    label = PRIMARY_LABEL
    best = sorted([r for r in eval_rows if r["label"] == label], key=lambda r: float(r["AUPRC"]), reverse=True)
    def best_group(name):
        cand = [r for r in best if r["feature_group"] == name]
        return cand[0] if cand else None
    hqa, ha, qa, qshuf, hshuf, za, group_only = [best_group(x) for x in ["h_q_action", "h_action", "q_action", "q_shuffled_action", "h_shuffled_q_action", "z_action", "h_q_group_action"]]
    verdict = "No-Go"
    if hqa and ha and qshuf and qa:
        if float(hqa["AUPRC"]) > float(ha["AUPRC"]) + 0.01 and float(hqa["AUPRC"]) > float(hshuf["AUPRC"]) + 0.01 and float(qa["AUPRC"]) > float(best_group("action_only")["AUPRC"]) + 0.01 and float(hqa["precision@10/base_rate"]) >= 1.5:
            verdict = "Go"
        elif float(hqa["AUPRC"]) > float(ha["AUPRC"]) or float(qa["AUPRC"]) > float(best_group("action_only")["AUPRC"]):
            verdict = "Partial"
    rows = [r for r in [hqa, ha, qa, qshuf, hshuf, za, group_only] if r]
    text = f"""# TwoRoom Failure Confidence Head V1

## 本阶段目的

训练一个外接诊断模块：

```text
c_{{t,a}} = C(h_t, q_t, a_t)
```

它预测的是官方 TwoRoom LeWM predictor 在当前 transition 上是否会出现高 latent prediction error。本阶段不改 LeWM encoder / predictor / train.py / loss / module.py / environment，也不训练 residual adapter。

## 使用的数据

- 官方 checkpoint：`/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt`
- official predictor failure labels：来自 `tworoom_direct_predictor_failure_atlas`
- 主标签：`{label}`
- 使用 direct_mse：是
- 使用 official_window_mse：是
- SIGReg：不作为单样本 failure label。SIGReg 是 batch/distribution regularizer，不适合作为 transition-level failure target。

## 关键结果

{table_md(rows, ["feature_group", "model_type", "seed", "base_rate", "AUROC", "AUPRC", "precision@top10_predicted_risk", "recall@top10_label", "precision@10/base_rate", "AUPRC/base_rate"])}

## 当前判断

`{verdict}`

- Go：进入第二阶段，训练 rule-conditioned residual adapter。
- Partial：继续优化 C 或 failure label / 输入，但暂不接 adapter。
- No-Go：h/q/action 无法可靠预测 official predictor failure，此路线暂时降级为 post-hoc diagnostics。

## 对第二阶段的建议

如果进入第二阶段，建议 gate 使用 `C(h,q,a)` 输出的 calibrated risk，adapter 输入优先从 `h_q_action` 开始，并保留 `h_action` 和 shuffled-q 对照；不要直接宣称因果修复，必须做 masking/intervention 或 adapter ablation。
"""
    write_report("README.md", text)


if __name__ == "__main__":
    main()
