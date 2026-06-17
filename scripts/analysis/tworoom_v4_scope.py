from datetime import datetime

from tworoom_v4_common import *


def main():
    ensure_dirs()
    text = f"""# TwoRoom KANFIS Rule-Feature Layer Scope

Generated: {datetime.now().isoformat(timespec="seconds")}

## Git

- branch: `{git_text(['branch', '--show-current'])}`
- commit: `{git_text(['rev-parse', 'HEAD'])}`
- log: `{git_text(['log', '-1', '--oneline'])}`

## Goal

Build an input-side interpretable factor path:

```text
z_t -> h_t -> KANFIS rule-feature layer -> q_t
```

## Constraints

- No official LeWM encoder / predictor / loss / train.py / module.py / TwoRoom environment modification.
- No FFN replacement and no ARPredictor integration.
- Action does not enter `F(z_t)` or `KANFIS(h_t) -> q_t`.
- Action only enters external diagnostic heads after q is produced.
- `q_t` is a KANFIS rule-activation physical factor candidate for future prediction modules.

## Inputs

- pair cache: `{PAIR_PATH}` exists={PAIR_PATH.exists()}
- split: `{SPLIT_PATH}` exists={SPLIT_PATH.exists()}
- v3 teacher: `{TEACHER_PATH}` exists={TEACHER_PATH.exists()}
- best h key: `{BEST_H_KEY}`
"""
    write_report("00_scope.md", text)


if __name__ == "__main__":
    main()
