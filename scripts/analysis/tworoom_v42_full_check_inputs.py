from tworoom_v42_full_common import *


def main():
    ensure_dirs()
    vis = build_visual_data()
    key = best_v42_key()
    model_path = v42_model_path(key)
    rows = [
        {"item": "visual_dataset", "value": str(REPO_ROOT / "outputs" / "tworoom_visualization_v5" / "visual_dataset.pt"), "exists": True},
        {"item": "v42_model_path", "value": str(model_path), "exists": model_path.exists()},
        {"item": "v42_compromise_outputs", "value": str(V42_OUT), "exists": V42_OUT.exists()},
        {"item": "v42_compromise_reports", "value": str(V42_REPORT), "exists": V42_REPORT.exists()},
        {"item": "q_v42_t", "value": str(tuple(vis["q_v42_t"].shape)), "exists": "q_v42_t" in vis},
        {"item": "q_v4_t", "value": str(tuple(vis["q_v4_t"].shape)), "exists": "q_v4_t" in vis},
        {"item": "q_v41_t", "value": str(tuple(vis["q_v41_t"].shape)), "exists": "q_v41_t" in vis},
        {"item": "h_t", "value": str(tuple(vis["h_t"].shape)), "exists": "h_t" in vis},
        {"item": "action_t", "value": str(tuple(vis["action_t"].shape)), "exists": "action_t" in vis},
        {"item": "position_t", "value": str(tuple(vis["position_t"].shape)), "exists": "position_t" in vis},
        {"item": "delta_position", "value": str(tuple(vis["delta_position"].shape)), "exists": "delta_position" in vis},
        {"item": "residual_t", "value": str(tuple(vis["residual_t"].shape)), "exists": "residual_t" in vis},
        {"item": "hard_labels", "value": str(tuple(vis["hard_labels"].shape)), "exists": "hard_labels" in vis},
        {
            "item": "split",
            "value": f"{vis['train_idx'].numel()}/{vis['val_idx'].numel()}/{vis['test_idx'].numel()}",
            "exists": True,
        },
    ]
    text = f"""# V4.2 Full Validation Input Check

## Git

- branch: `{git_text(["branch", "--show-current"])}`
- commit: `{git_text(["rev-parse", "HEAD"])}`
- log: `{git_text(["log", "-1", "--oneline"])}`

## Inputs

{table_md(rows, ["item", "value", "exists"])}

## Constraint Check

- Best V4.2 model: `{key}`
- Best V4.2 model path: `{model_path}`
- q_v42 shape: `{tuple(vis["q_v42_t"].shape)}`
- hard label count: `{vis["hard_labels"].shape[1]}`
- split counts: `{vis["train_idx"].numel()}/{vis["val_idx"].numel()}/{vis["test_idx"].numel()}`
- No LeWM, h, V4, V4.1, or V4.2 q training is performed in this validation. Existing models and caches are read only.
"""
    write_report("00_input_check.md", text)


if __name__ == "__main__":
    main()
