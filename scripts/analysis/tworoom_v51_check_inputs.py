from tworoom_v51_common import *


def main():
    ensure_dirs()
    vis = load_visual()
    rows = [
        {"item": "visual_dataset", "path": str(VISUAL_DATASET), "exists": VISUAL_DATASET.exists()},
        {"item": "image_t", "path": str(tuple(vis["image_t"].shape)), "exists": True},
        {"item": "position_t", "path": str(tuple(vis["position_t"].shape)), "exists": True},
        {"item": "residual_t", "path": str(tuple(vis["residual_t"].shape)), "exists": True},
        {"item": "q_v4_t", "path": str(tuple(vis["q_v4_t"].shape)), "exists": True},
        {"item": "q_v41_t", "path": str(tuple(vis["q_v41_t"].shape)), "exists": True},
        {"item": "v5_rule_visual_link_figures", "path": str(V5_RULE_FIG), "exists": V5_RULE_FIG.exists()},
    ]
    for name in vis["hard_label_names"]:
        rows.append({"item": f"hard_label_{name}", "path": str(int(vis["sample_flags"][name].sum())), "exists": True})
    text = f"""# V5.1 Input Check

## Git

- branch: `{git_text(['branch', '--show-current'])}`
- commit: `{git_text(['rev-parse', 'HEAD'])}`
- log: `{git_text(['log', '-1', '--oneline'])}`

## Inputs

{table_md(rows, ["item", "path", "exists"])}

## Constraint Check

- No new LeWM, h, q, or KANFIS training.
- V5.1 only reads existing V4/V4.1/V5 outputs and generates comparison figures/reports.
"""
    write_report("00_input_check.md", text)


if __name__ == "__main__":
    main()

