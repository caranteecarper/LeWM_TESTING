from tworoom_failure_c_v1_common import *


FEATURE_GROUPS = [
    "action_only",
    "h_action",
    "q_action",
    "h_q_action",
    "h_q_group_action",
    "h_shuffled_q_action",
    "q_shuffled_action",
    "z_action",
]
MODEL_TYPES = ["logistic_linear", "additive_linear", "small_mlp_64", "small_mlp_128"]


def main():
    ensure_dirs()
    data = load_dataset()
    rows = []
    for feature_group in FEATURE_GROUPS:
        if feature_group == "z_action" and "z" not in data:
            continue
        for model_type in MODEL_TYPES:
            for seed in [0, 1, 2]:
                print(f"training {feature_group} {model_type} seed={seed}", flush=True)
                obj = train_one_model(data, feature_group, model_type, seed=seed)
                path = MODEL_DIR / f"{feature_group}__{model_type}__seed{seed}.pt"
                torch.save(obj, path)
                hist = obj["history"]
                for h in hist:
                    rows.append(
                        {
                            "feature_group": feature_group,
                            "model_type": model_type,
                            "seed": seed,
                            "epoch": h["epoch"],
                            "train_loss": h["train_loss"],
                            "val_primary_auprc": h["val_primary_auprc"],
                            "model_path": str(path),
                        }
                    )
    csv_write(TABLE / "train_metrics_all.csv", rows)
    best = sorted(rows, key=lambda r: float(r["val_primary_auprc"]), reverse=True)[:20]
    text = f"""# C(h,q,a) Failure Confidence Head 训练

## 设置

- 输入组合：`{FEATURE_GROUPS}`
- 模型：`{MODEL_TYPES}`
- seeds：`0,1,2`
- loss：weighted BCE 多分类标签 + `0.1 * SmoothL1(log error regression)`
- early stopping：验证集 `{data['primary_label']}` AUPRC，patience=10
- 本阶段只训练 C head，不训练 LeWM，不训练 h/q，不训练 residual adapter。

## 验证集 Top 20

{table_md(best, ["feature_group", "model_type", "seed", "epoch", "val_primary_auprc", "model_path"])}
"""
    write_report("02_train_failure_head.md", text)


if __name__ == "__main__":
    main()
