from tworoom_v2_common import *


def read(name):
    path = V2_REPORT / name
    return path.read_text() if path.exists() else f"Missing `{name}`."


def main():
    ensure_dirs()
    content = f"""# TwoRoom Pre-Integration V2

## 1. Questions Addressed

1. Why did v1 KANFIS-style fail?
2. Can a clean factor extractor obey `h_t = F(z_t)` while using action only as auxiliary diagnostics?

## 2. Scope

{read('00_v2_scope.md')}

## 3. Readout Consistency

{read('01_readout_consistency.md')}

## 4. KANFIS Debug

{read('02_kanfis_debug.md')}

## 5. Low-Dimensional KANFIS Sanity

{read('03_kanfis_lowdim_sanity.md')}

## 6. Factor Extractor Without Action

{read('04_factor_extractor_no_action.md')}

## 7. Residual And Hard Subset

{read('05_residual_and_hard_subset.md')}

## 8. H Dynamics Consistency

{read('06_h_dynamics_consistency.md')}

## 9. Importance Ablation

{read('07_importance_ablation.md')}

## 10. Go / Partial Go / No-Go

Decision rule:

- Go requires KANFIS toy/lowdim sanity, non-collapsed hK16, hK16+action residual or hard-subset gain over action-only, and meaningful h-dim ablation.
- Partial Go means external factor work may continue, but formal LeWM predictor integration remains blocked.
- No-Go means h or KANFIS does not add reliable transition/residual value beyond action-only.

Final decision should be read from sections 4-9 above. Do not formally integrate into LeWM unless the hard-subset and h-dynamics evidence is positive.
"""
    write_report("README.md", content)


if __name__ == "__main__":
    main()
