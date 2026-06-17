from tworoom_v3_common import *


def read(name):
    p = V3_REPORT / name
    return p.read_text() if p.exists() else f"Missing `{name}`."


def main():
    ensure_dirs()
    text = f"""# TwoRoom Pre-Integration V3

## 1. Goal

Combine C1-C4 into one external dynamics-aware physical factor extractor. This is still pre-integration; LeWM is untouched.

## 2. Scope

{read('00_scope.md')}

## 3. Teacher And Labels

{read('01_teacher_and_labels.md')}

## 4. Dynamics-Aware H Training

{read('02_dynamics_aware_h_training.md')}

## 5. Hard Subset Evaluation

{read('03_h_hard_subset_eval.md')}

## 6. H Dynamics

{read('04_h_dynamics_eval.md')}

## 7. H Ablation

{read('05_h_ablation.md')}

## 8. KANFIS On Best H

{read('06_kanfis_on_best_h.md')}

## 9. Go / Partial Go / No-Go

Formal integration requires hard-subset improvement over action-only, positive gap closure toward z+action teacher, improved h dynamics over old pure hK16, and h ablation evidence that selected h dimensions are necessary.

Do not modify LeWM unless those conditions are satisfied.
"""
    write_report("README.md", text)


if __name__ == "__main__":
    main()
