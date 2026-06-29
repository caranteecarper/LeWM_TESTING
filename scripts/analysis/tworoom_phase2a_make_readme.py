from tworoom_phase2a_common import *


def pick(rows, subset, pred):
    cand = [r for r in rows if r.get("subset") == subset and pred(r)]
    return sorted(cand, key=lambda r: float(r["relative_improvement"]), reverse=True)[0] if cand else None


def main():
    ensure_dirs()
    rows = read_csv(TABLE / "adapter_eval_metrics.csv")
    ab = read_csv(TABLE / "adapter_ablation.csv")
    best_full = pick(rows, PRIMARY_LABEL, lambda r: r.get("input_group") == "full_q_action" and r.get("gate_name") in ("c_soft_gate", "c_sigmoid_gate"))
    best_single = pick(rows, "V4.2_group_active_top10", lambda r: r.get("input_group") in ("single_group_action", "group_only_action"))
    best_shuf = pick(rows, PRIMARY_LABEL, lambda r: r.get("input_group") == "shuffled_q_action")
    best_random = pick(rows, PRIMARY_LABEL, lambda r: r.get("gate_name") == "random_gate")
    best_always = pick(rows, PRIMARY_LABEL, lambda r: r.get("gate_name") == "always_on")
    best_oracle = pick(rows, PRIMARY_LABEL, lambda r: r.get("gate_name") == "oracle_gate")
    all_row = None
    if best_full:
        all_matches = [r for r in rows if r["subset"] == "all_test" and r["model"] == best_full["model"]]
        all_row = all_matches[0] if all_matches else None
    direct_row = None
    if best_full:
        dm = [r for r in rows if r["subset"] == DIRECT_LABEL and r["model"] == best_full["model"]]
        direct_row = dm[0] if dm else None
    go = (
        best_full
        and float(best_full["relative_improvement"]) >= 0.05
        and direct_row
        and float(direct_row["relative_improvement"]) >= 0.0
        and all_row
        and float(all_row["relative_improvement"]) >= -0.01
        and best_shuf
        and float(best_full["relative_improvement"]) > float(best_shuf["relative_improvement"])
        and best_random
        and float(best_full["relative_improvement"]) > float(best_random["relative_improvement"])
    )
    partial = best_full and float(best_full["relative_improvement"]) > 0 and all_row and float(all_row["relative_improvement"]) >= -0.03
    verdict = "Go" if go else ("Partial" if partial else "No-Go")
    summary = [r for r in [best_full, direct_row, all_row, best_single, best_shuf, best_random, best_always, best_oracle] if r]
    text = f"""# TwoRoom Phase 2A Rule-Conditioned Residual Adapter

## 目的

验证规则分支是否能修正官方 LeWM predictor 的真实 latent prediction error。

## 约束

- 未修改 LeWM encoder / predictor / train.py / loss / module.py / environment。
- 未重新训练官方 LeWM、h extractor、V4.2 q、C failure head。
- 本阶段只训练外接 residual adapter：`final_pred_z = base_pred_z + gate * R(input)`。
- 本阶段不是因果证明，也不是完整 LeWM 替换。

## C gate 来源

`{C_MODEL_PATH}`

## 关键结果

{table_md(summary, ["model", "subset", "base_mse", "final_mse", "relative_improvement", "gate_mean", "correction_norm"])}

## 判断

`{verdict}`

Go 要求：full_q + C gate 在 official_window_mse_top10 上稳定改善，direct_mse_top10 不明显变差，all_test 不明显恶化，shuffled-q/random-gate 弱于真实 q/C gate。

## 下一步

如果 Go，进入 Phase 2B：

- 自动发现多个 q rule groups；
- 每个 group 做 audit；
- multi-group gated residual adapter；
- group-specific intervention。

如果 Partial，先继续调 gate / adapter，不做多组发现。
"""
    write_report("README.md", text)


if __name__ == "__main__":
    main()
