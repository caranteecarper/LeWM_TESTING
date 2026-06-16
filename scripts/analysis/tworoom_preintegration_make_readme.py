from pathlib import Path

from tworoom_preintegration_common import PREINT_REPORT


def read(name):
    p = PREINT_REPORT / name
    return p.read_text() if p.exists() else f"Missing `{name}`."


def main():
    PREINT_REPORT.mkdir(parents=True, exist_ok=True)
    content = f"""# TwoRoom Pre-Integration Report

## 1. Goal

This stage validates an external physical-factor extraction path before any formal LeWM predictor integration. No official LeWM encoder, predictor, loss, FFN, or TwoRoom environment logic is modified.

## 2. Official Baseline And Readout

Official checkpoint: `/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt`

Existing linear readout test result: x R2 about 0.9916, y R2 about 0.7251.

## 3. Static Probe Comparison

{read('02_static_probe_comparison.md')}

## 4. Factor Bottleneck

{read('03_factor_bottleneck.md')}

## 5. Delta Readout

{read('04_delta_readout.md')}

## 6. Action-Conditioned Transition

{read('05_action_conditioned_transition.md')}

## 7. Ablation / Masking

{read('06_ablation_importance.md')}

## 8. Region Diagnostics

{read('07_region_diagnostics.md')}

## 9. Go / No-Go

The final Go / No-Go decision should be based on whether h+action approaches z+action, whether h is non-collapsed and interpretable, and whether ablation confirms important dimensions are necessary.

## 10. Next Step

If Go: test `z_t -> h_t`, then `[h_t, action] -> delta_position / h_next` as a formal integration candidate. If No-Go: improve the external extractor before touching LeWM predictor integration.
"""
    (PREINT_REPORT / "README.md").write_text(content)
    print(content[:4000])


if __name__ == "__main__":
    main()

