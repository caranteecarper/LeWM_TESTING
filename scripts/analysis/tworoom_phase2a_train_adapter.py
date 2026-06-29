from tworoom_phase2a_common import *


TRAIN_GATES = ["c_soft_gate", "c_sigmoid_gate", "always_on", "random_gate", "oracle_gate"]
LOSS_MODES = ["hard_weighted"]


def main():
    ensure_dirs()
    data = load_adapter_dataset()
    rows = []
    jobs = []
    # Full grid for the most relevant gates, single loss mode to keep Phase 2A bounded.
    for input_group in INPUT_GROUPS:
        for model_type in MODEL_TYPES:
            for gate_name in TRAIN_GATES:
                for seed in [0, 1, 2]:
                    jobs.append((input_group, model_type, gate_name, "hard_weighted", seed))
    for input_group, model_type, gate_name, loss_mode, seed in jobs:
        print(f"train {input_group} {model_type} {gate_name} {loss_mode} seed={seed}", flush=True)
        obj = train_adapter(data, input_group, model_type, gate_name, loss_mode=loss_mode, seed=seed)
        path = MODEL_DIR / f"{input_group}__{model_type}__{gate_name}__{loss_mode}__seed{seed}.pt"
        torch.save(obj, path)
        for h in obj["history"]:
            rows.append(
                {
                    "input_group": input_group,
                    "model_type": model_type,
                    "gate_name": gate_name,
                    "loss_mode": loss_mode,
                    "seed": seed,
                    "epoch": h["epoch"],
                    "train_loss": h["train_loss"],
                    "val_hard_mse": h["val_hard_mse"],
                    "model_path": str(path),
                }
            )
    csv_write(TABLE / "train_metrics.csv", rows)
    best = sorted(rows, key=lambda r: float(r["val_hard_mse"]))[:20]
    text = f"""# Residual Adapter 训练

训练目标：`final_pred_z = base_pred_z + gate * R(input)`，只训练 R，不修改 LeWM。

- input groups：`{INPUT_GROUPS}`
- model types：`{MODEL_TYPES}`
- gates：`{TRAIN_GATES}`
- loss mode：`hard_weighted`
- seeds：`0,1,2`

## Val hard subset Top 20

{table_md(best, ["input_group", "model_type", "gate_name", "loss_mode", "seed", "epoch", "val_hard_mse", "model_path"])}
"""
    write_report("03_train_adapter.md", text)


if __name__ == "__main__":
    main()
