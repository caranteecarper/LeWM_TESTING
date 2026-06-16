from datetime import datetime

from tworoom_v2_common import *


def main():
    ensure_dirs()
    text = f"""# TwoRoom Pre-Integration V2 Scope

Generated: {datetime.now().isoformat(timespec="seconds")}

## Git

- branch: `{git_text(['branch', '--show-current'])}`
- commit: `{git_text(['rev-parse', 'HEAD'])}`
- log: `{git_text(['log', '-1', '--oneline'])}`

## Inputs

- latent cache: `{LATENTS_PATH}` exists={LATENTS_PATH.exists()}
- pair cache: `{PAIR_PATH}` exists={PAIR_PATH.exists()}
- official checkpoint: `{CHECKPOINT_PATH}` exists={CHECKPOINT_PATH.exists()}
- v1 reports: `{V1_REPORT}` exists={V1_REPORT.exists()}

## Scope

- This stage does not modify official LeWM encoder, predictor, loss, `train.py`, `module.py`, or TwoRoom environment logic.
- This stage does not start new LeWM baseline training.
- KANFIS-style models here are external diagnostic probes only; they are not inserted into LeWM.
- Factor extractors must obey `h_t = F(z_t)`. Action is not allowed to enter the extractor.
- Action may only be used after `h_t` exists, as an auxiliary/diagnostic input for transition, residual, or h dynamics checks.
- Outputs are written to `{V2_REPORT}` and `{V2_OUT}`.
"""
    write_report("00_v2_scope.md", text)


if __name__ == "__main__":
    main()
