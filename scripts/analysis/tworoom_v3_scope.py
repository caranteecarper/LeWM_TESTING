from datetime import datetime

from tworoom_v3_common import *


def main():
    ensure_dirs()
    text = f"""# TwoRoom Pre-Integration V3 Scope

Generated: {datetime.now().isoformat(timespec="seconds")}

## Git

- branch: `{git_text(['branch', '--show-current'])}`
- commit: `{git_text(['rev-parse', 'HEAD'])}`
- log: `{git_text(['log', '-1', '--oneline'])}`

## Goal

Train a dynamics-aware physical factor extractor that combines C1-C4 diagnostics into one external path.

## Constraints

- No official LeWM encoder / predictor / loss / train.py / module.py / TwoRoom environment modifications.
- No new LeWM baseline training.
- No FFN replacement and no formal LeWM predictor integration.
- Extractor is strictly `h_t = F(z_t)`.
- Forbidden: `h_t = F(z_t, action_t)`.
- Action may only enter auxiliary heads after h is produced: delta, residual, hard labels, h_next / delta_h.
- KANFIS-style remains a low-dimensional external diagnostic only.

## Inputs

- pair cache: `{PAIR_PATH}` exists={PAIR_PATH.exists()}
- v2 split: `{SPLIT_PATH}` exists={SPLIT_PATH.exists()}
- v2 reports: `{V2_REPORT}` exists={V2_REPORT.exists()}
"""
    write_report("00_scope.md", text)


if __name__ == "__main__":
    main()
