from pathlib import Path

from tworoom_v4_common import *


def read(name):
    path = V4_REPORT / name
    return path.read_text() if path.exists() else f"# {name}\n\nMissing.\n"


def main():
    ensure_dirs()
    sections = [
        "00_scope.md",
        "01_best_h_dataset.md",
        "02_train_rule_features.md",
        "03_q_transition_eval.md",
        "04_exported_rules.md",
        "05_rule_ablation.md",
        "06_q_semantics.md",
    ]
    body = ["# TwoRoom KANFIS Rule-Feature Layer README\n"]
    body.append(
        """## Executive Summary

This stage tests an external input-side path:

```text
z_t -> h_t -> KANFIS rule-feature layer -> q_t
```

`q_t` is a candidate interpretable input for a later prediction module. This run does not modify LeWM, does not replace FFN, and does not feed action into q extraction.

The final Go / Partial / No-Go judgment should be based on whether q keeps position information, whether q+action beats action-only on residual/delta/hard subsets, whether it approaches h+action, and whether rule ablation shows functional contribution.
"""
    )
    for name in sections:
        body.append("\n---\n")
        body.append(read(name))
    out = "\n".join(body)
    write_report("README.md", out)


if __name__ == "__main__":
    main()
