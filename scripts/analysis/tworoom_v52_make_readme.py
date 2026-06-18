from tworoom_v52_common import *


def read(name):
    p = V52_REPORT / name
    return p.read_text() if p.exists() else f"# {name}\n\nMissing.\n"


def main():
    ensure_dirs()
    text = "\n\n---\n\n".join(
        [
            "# TwoRoom V5.2 V4 Rule Group Validation\n\n"
            "V5.2 validates whether V4 rule groups are useful beyond visualization. It audits all V4 rules, audits co-activation groups, tests group-only performance, and masks groups in a fixed full-q diagnostic head.\n\n"
            "## Provisional Judgment\n\n"
            "V4 full q remains useful. Group 0 is the main V4 explanatory unit if it keeps high hard enrichment and its mask damage exceeds random. If group-only retention is limited, group 0 should be used as explanation rather than replacement input.\n",
            read("00_input_check.md"),
            read("01_v4_all_rule_audit.md"),
            read("02_v4_group_audit.md"),
            read("03_v4_group_only_performance.md"),
            read("04_v4_group_mask_ablation.md"),
        ]
    )
    write_report("README.md", text)


if __name__ == "__main__":
    main()
