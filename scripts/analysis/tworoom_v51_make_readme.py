from tworoom_v51_common import *


def read(name):
    p = V51_REPORT / name
    return p.read_text() if p.exists() else f"# {name}\n\nMissing.\n"


def main():
    ensure_dirs()
    text = "\n\n---\n\n".join(
        [
            "# TwoRoom V5.1 V4 vs V4.1 Rule Visualization Comparison\n\n"
            "V5.1 adds visualization-only comparison. It does not train new LeWM, h, q, or KANFIS models.\n\n"
            "## Recommendation\n\n"
            "Use a dual-track presentation: V4 rule groups for performance-oriented rule representation, and V4.1 sparse rules for white-box single-rule explanations. V4 single rules are useful but should usually be interpreted with co-activation groups.\n\n"
            "## Recommended Midterm Figures\n\n"
            "- V4 group activation-over-position and top images from `outputs/tworoom_visualization_v51/figures/qv4_group_link/`\n"
            "- V4.1 rule 0/12/13 activation-over-position, top images, and residual arrows from `outputs/tworoom_visualization_v51/figures/qv41_rule_link_fixed/`\n"
            "- Comparison table in `03_v4_vs_v41_comparison.md`\n\n"
            "## Current Caveats\n\n"
            "- Region names remain conservative: hard-motion, residual-correction, position-related, or state-partition. Do not call rules wall/door/collision unless additional evidence is added.\n"
            "- V4 groups are post-hoc co-activation groups, not newly trained rules.\n",
            read("00_input_check.md"),
            read("01_qv4_rule_and_group_visual_link.md"),
            read("02_qv41_residual_direction_fix.md"),
            read("03_v4_vs_v41_comparison.md"),
        ]
    )
    write_report("README.md", text)


if __name__ == "__main__":
    main()
