from tworoom_v41_common import *


def read(name):
    p = V41_REPORT / name
    return p.read_text() if p.exists() else f"# {name}\n\nMissing.\n"


def main():
    ensure_dirs()
    parts = [
        "# TwoRoom V4.1 White-Box-Preserving KANFIS q Strengthening\n",
        """## Executive Summary

V4.1 sharpens `q_t = KANFIS_rule_features(h_t)` with temperature annealing, top-k rule gating, entropy, usage balance, and rule diversity. It does not modify LeWM, does not connect to the predictor, and does not use q->h reconstruction or h distillation.

The final judgment should compare V4.1 to V4 on residual MSE, hard-subset residual MSE, gap closure, active rule count, and rule ablation strength.
""",
    ]
    for name in [
        "01_input_check.md",
        "02_train_sharp_q.md",
        "03_v4_vs_v41_eval.md",
        "04_exported_rules.md",
        "05_rule_ablation.md",
        "06_q_semantics.md",
    ]:
        parts += ["\n---\n", read(name)]
    write_report("README.md", "\n".join(parts))


if __name__ == "__main__":
    main()
