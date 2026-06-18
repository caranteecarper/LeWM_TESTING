from pathlib import Path

from tworoom_v42_common import REPO_ROOT


def read(path):
    p = REPO_ROOT / path
    return p.read_text() if p.exists() else f"# {path}\n\nMissing.\n"


def main():
    out = REPO_ROOT / "reports" / "tworoom_rule_and_compromise_summary.md"
    text = "\n\n---\n\n".join(
        [
            "# TwoRoom Rule Group And Compromise Summary\n\n"
            "This summary combines V5.2 V4 rule-group validation and V4.2 compromise q diagnostics.\n\n"
            "Recommended decision should be made from: V4 group functional evidence, V4.2 performance relative to V4/V4.1, active rule count, and ablation strength.",
            read("reports/tworoom_rule_group_v52/README.md"),
            read("reports/tworoom_kanfis_v42_compromise/README.md"),
        ]
    )
    out.write_text(text)
    print(text[:7000])


if __name__ == "__main__":
    main()
