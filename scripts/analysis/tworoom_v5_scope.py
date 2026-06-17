from tworoom_v5_common import *


def main():
    ensure_dirs()
    text = f"""# TwoRoom V5 Visualization Scope

## Git

- branch: `{git_text(['branch', '--show-current'])}`
- commit: `{git_text(['rev-parse', 'HEAD'])}`
- log: `{git_text(['log', '-1', '--oneline'])}`

## Goal

V5 is post-hoc representation visualization. It compares what information is retained in:

- official LeWM latent `z_t`
- dynamics-aware factor `h_t`
- KANFIS rule factor `q_v4_t`
- sharper KANFIS rule factor `q_v41_t`
- combined `[h_t, q_v41_t]`

## Constraints

- No LeWM encoder / predictor / train.py / loss / environment modification.
- No KANFIS or h retraining.
- Image reconstruction trains only post-hoc decoders with fixed representations.
- No q->h reconstruction, h mimicry, or hidden-state distillation.
- Current-frame image reconstruction does not use action.

## Outputs

- visual dataset: `{VISUAL_DATASET}`
- figures: `{V5_FIG}`
- reports: `{V5_REPORT}`
"""
    write_report("00_scope.md", text)


if __name__ == "__main__":
    main()

