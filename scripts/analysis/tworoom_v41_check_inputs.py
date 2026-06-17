from tworoom_v41_common import *


def main():
    ensure_dirs()
    data = load_data()
    v4_summary = V4_RULE_DIR / "summary.pt"
    v4_best = None
    if v4_summary.exists():
        v4_best = torch.load(v4_summary, map_location="cpu").get("best_key")
    rows = [
        {"item": "best_h_dataset", "value": str(BEST_H_DATASET), "exists": BEST_H_DATASET.exists()},
        {"item": "h_t_shape", "value": tuple(data["h_t"].shape), "exists": True},
        {"item": "action_shape", "value": tuple(data["action"].shape), "exists": True},
        {"item": "position_shape", "value": tuple(data["position_t"].shape), "exists": True},
        {"item": "delta_shape", "value": tuple(data["delta_position"].shape), "exists": True},
        {"item": "residual_shape", "value": tuple(data["residual"].shape), "exists": True},
        {"item": "hard_labels_shape", "value": tuple(data["hard_labels"].shape), "exists": True},
        {"item": "train/val/test", "value": f"{data['train_idx'].numel()}/{data['val_idx'].numel()}/{data['test_idx'].numel()}", "exists": True},
        {"item": "v4_best_q", "value": v4_best, "exists": bool(v4_best)},
        {"item": "v4_best_checkpoint", "value": str(V4_RULE_DIR / f'{v4_best}.pt') if v4_best else "", "exists": (V4_RULE_DIR / f"{v4_best}.pt").exists() if v4_best else False},
    ]
    text = f"""# V4.1 Input Check

## Git

- branch: `{git_text(['branch', '--show-current'])}`
- commit: `{git_text(['rev-parse', 'HEAD'])}`
- log: `{git_text(['log', '-1', '--oneline'])}`

## Scope

V4.1 strengthens `q_t = KANFIS_rule_features(h_t)` without modifying LeWM and without connecting to the predictor.

## Inputs

{table_md(rows, ["item", "value", "exists"])}

## Constraint Check

- Action does not enter q extraction. q is produced from h only.
- This stage does not use q->h reconstruction, q->h_next reconstruction as a main loss, h mimicry, or MLP-hidden distillation.
- h_next / delta_h are not training losses in the main V4.1 variants.
"""
    write_report("01_input_check.md", text)


if __name__ == "__main__":
    main()

