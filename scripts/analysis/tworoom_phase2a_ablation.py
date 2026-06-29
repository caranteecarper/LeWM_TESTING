from tworoom_phase2a_common import *


ABLATIONS = ["none", "mask_group", "mask_full_q", "shuffled_q", "clamp_group_high", "clamp_group_low", "random_gate", "always_on_gate", "oracle_gate"]


def best_model_path():
    rows = read_csv(TABLE / "adapter_eval_metrics.csv")
    cand = [r for r in rows if r.get("subset") == PRIMARY_LABEL and r.get("input_group") == "full_q_action" and r.get("gate_name") in ("c_soft_gate", "c_sigmoid_gate")]
    if not cand:
        cand = [r for r in rows if r.get("subset") == PRIMARY_LABEL and r.get("model") != "base_official_predictor"]
    best = sorted(cand, key=lambda r: float(r["relative_improvement"]), reverse=True)[0]
    name = f"{best['input_group']}__{best['model_type']}__{best['gate_name']}__{best['loss_mode']}__seed{best['seed']}.pt"
    return MODEL_DIR / name, best


def main():
    ensure_dirs()
    data = load_adapter_dataset()
    path, best = best_model_path()
    model, obj = load_adapter(path)
    masks = subset_masks(data)
    rows = []
    for ab in ABLATIONS:
        final, corr, gate = evaluate_adapter_obj(data, model, obj, ablation=None if ab == "none" else ab)
        for subset in ["all_test", PRIMARY_LABEL, DIRECT_LABEL, "V4.2_group_active_top10", "rule_active_and_high_risk", "rule_inactive_low_risk"]:
            row = metrics_for_final(data, final, corr, gate, f"best_full_q_ablation:{ab}", subset, masks[subset])
            if row:
                row["ablation"] = ab
                row.update({k: obj[k] for k in ["input_group", "model_type", "gate_name", "loss_mode", "seed"]})
                rows.append(row)
    csv_write(TABLE / "adapter_ablation.csv", rows)
    text = f"""# q/gate 消融

消融对象：`{path}`

{table_md(rows, ["ablation", "subset", "base_mse", "final_mse", "relative_improvement", "gate_mean", "correction_norm"])}

## 需要关注

- `mask_group/mask_full_q/shuffled_q` 如果变弱，说明 adapter 使用了 q。
- `random_gate/always_on_gate` 如果弱于原 gate，说明 C gate 有贡献。
- `oracle_gate` 只是 upper bound，不作为主结果。
"""
    write_report("05_ablation.md", text)


if __name__ == "__main__":
    main()
