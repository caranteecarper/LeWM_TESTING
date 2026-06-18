from tworoom_v52_common import *


def main():
    ensure_dirs()
    vis = load_visual()
    rows = [
        {"item": "visual_dataset", "value": str(VISUAL_DATASET), "exists": VISUAL_DATASET.exists()},
        {"item": "q_v4_t", "value": str(tuple(vis["q_v4_t"].shape)), "exists": True},
        {"item": "q_v41_t", "value": str(tuple(vis["q_v41_t"].shape)), "exists": True},
        {"item": "h_t", "value": str(tuple(vis["h_t"].shape)), "exists": True},
        {"item": "action_t", "value": str(tuple(vis.get("action", vis.get("action_t", vis["z_t"][:, :0])).shape)), "exists": "action" in vis or "action_t" in vis},
        {"item": "position_t", "value": str(tuple(vis["position_t"].shape)), "exists": True},
        {"item": "delta_position", "value": str(tuple(vis["delta_position"].shape)), "exists": True},
        {"item": "residual_t", "value": str(tuple(vis["residual_t"].shape)), "exists": True},
        {"item": "hard_labels", "value": str(tuple(vis["hard_labels"].shape)), "exists": True},
        {"item": "split", "value": f"{vis['train_idx'].numel()}/{vis['val_idx'].numel()}/{vis['test_idx'].numel()}", "exists": True},
    ]
    text = f"""# V5.2 Input Check

## Git

- branch: `{git_text(['branch', '--show-current'])}`
- commit: `{git_text(['rev-parse', 'HEAD'])}`
- log: `{git_text(['log', '-1', '--oneline'])}`

{table_md(rows, ["item", "value", "exists"])}

## Constraint Check

V5.2 reads existing q_v4/q_v41/h/action/physical labels. No LeWM, h, q, or KANFIS training is performed in phase A.
"""
    write_report("00_input_check.md", text)


if __name__ == "__main__":
    main()

