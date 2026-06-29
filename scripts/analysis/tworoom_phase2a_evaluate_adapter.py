from tworoom_phase2a_common import *


def evaluate_all():
    data = load_adapter_dataset()
    masks = subset_masks(data)
    rows = []
    # Base row.
    for subset, mask in masks.items():
        base_row = metrics_for_final(data, data["base_pred_z"], torch.zeros_like(data["base_pred_z"]), torch.zeros(data["h"].shape[0]), "base_official_predictor", subset, mask)
        if base_row:
            rows.append(base_row)
    for path in sorted(MODEL_DIR.glob("*.pt")):
        model, obj = load_adapter(path)
        final, corr, gate = evaluate_adapter_obj(data, model, obj)
        model_name = f"{obj['input_group']}|{obj['model_type']}|{obj['gate_name']}|{obj['loss_mode']}|seed{obj['seed']}"
        for subset, mask in masks.items():
            row = metrics_for_final(data, final, corr, gate, model_name, subset, mask)
            if row:
                row.update({k: obj[k] for k in ["input_group", "model_type", "gate_name", "loss_mode", "seed"]})
                rows.append(row)
    csv_write(TABLE / "adapter_eval_metrics.csv", rows)
    return rows


def main():
    ensure_dirs()
    rows = evaluate_all()
    top_ow = sorted(
        [r for r in rows if r["subset"] == PRIMARY_LABEL and r["model"] != "base_official_predictor"],
        key=lambda r: float(r["relative_improvement"]),
        reverse=True,
    )[:20]
    top_direct = sorted(
        [r for r in rows if r["subset"] == DIRECT_LABEL and r["model"] != "base_official_predictor"],
        key=lambda r: float(r["relative_improvement"]),
        reverse=True,
    )[:12]
    all_test = sorted(
        [r for r in rows if r["subset"] == "all_test" and r["model"] != "base_official_predictor"],
        key=lambda r: float(r["relative_improvement"]),
        reverse=True,
    )[:12]
    text = f"""# Adapter Evaluation

## official_window_mse_top10 Top

{table_md(top_ow, ["model", "n", "base_mse", "final_mse", "relative_improvement", "gate_mean", "correction_norm"])}

## direct_mse_top10 Top

{table_md(top_direct, ["model", "n", "base_mse", "final_mse", "relative_improvement", "gate_mean", "correction_norm"])}

## all_test Top

{table_md(all_test, ["model", "n", "base_mse", "final_mse", "relative_improvement", "gate_mean", "correction_norm"])}

Go 参考：hard subset >=5% improvement，all_test 不恶化超过 1%，shuffled/random 控制弱于 full-q/C gate。
"""
    write_report("04_adapter_evaluation.md", text)


if __name__ == "__main__":
    main()
