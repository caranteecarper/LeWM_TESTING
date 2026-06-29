from tworoom_failure_c_v1_common import *


def evaluate_all():
    data = load_dataset()
    test_idx = data["split"]["head_test_idx"]
    rows = []
    for path in sorted(MODEL_DIR.glob("*.pt")):
        model, obj = load_head(path)
        x, _ = make_feature(data, obj["feature_group"], seed=int(obj["seed"]))
        x_std = (x.float() - obj["x_mean"]) / obj["x_std"].clamp_min(1e-6)
        pred = predict_model(model, x_std[test_idx])
        rows.extend(eval_predictions(data, pred, test_idx, obj["feature_group"], obj["model_type"], int(obj["seed"])))
    csv_write(TABLE / "eval_metrics.csv", rows)
    return data, rows


def main():
    ensure_dirs()
    data, rows = evaluate_all()
    primary = sorted([r for r in rows if r["label"] == data["primary_label"]], key=lambda r: float(r["AUPRC"]), reverse=True)[:20]
    direct = sorted([r for r in rows if r["label"] == "direct_mse_top10"], key=lambda r: float(r["AUPRC"]), reverse=True)[:12]
    text = f"""# C Head 评估

## 重点标签：{data['primary_label']}

{table_md(primary, ["feature_group", "model_type", "seed", "base_rate", "AUROC", "AUPRC", "precision@top10_predicted_risk", "recall@top10_label", "precision@10/base_rate", "AUPRC/base_rate"])}

## direct_mse_top10

{table_md(direct, ["feature_group", "model_type", "seed", "base_rate", "AUROC", "AUPRC", "precision@top10_predicted_risk", "recall@top10_label", "precision@10/base_rate", "AUPRC/base_rate"])}

## 解读

`precision@top10_predicted_risk / base_rate` 表示 C head 挑出的 top10% 风险样本相比随机样本富集了多少倍。该指标不是性能修复，只是 failure risk 预测能力。
"""
    write_report("03_evaluation.md", text)


if __name__ == "__main__":
    main()
