from tworoom_v42_common import *


def read(name):
    p = V42_REPORT / name
    return p.read_text() if p.exists() else f"# {name}\n\nMissing.\n"


def main():
    ensure_dirs()
    text = "\n\n---\n\n".join(
        [
            "# TwoRoom V4.2 Compromise KANFIS q\n\n"
            "V4.2 trains only an external compromise q layer from h. It does not modify LeWM, does not connect to ARPredictor, and does not use q->h reconstruction, h mimicry, hidden distillation, or image reconstruction.\n\n"
            "## Judgment Template\n\n"
            "V4.2 is Go only if it is closer to V4 performance than V4.1 while improving active rule count and visual clarity. Otherwise, keep V4 full q as main and V4.1/V4.2 as explanatory baselines.\n",
            read("01_train_v42.md"),
            read("02_compare_v4_v41_v42.md"),
            read("03_v42_rule_visualization.md"),
            read("04_v42_rule_ablation.md"),
        ]
    )
    write_report("README.md", text)


if __name__ == "__main__":
    main()

