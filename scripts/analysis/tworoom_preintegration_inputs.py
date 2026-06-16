import subprocess
from datetime import datetime

import torch

from tworoom_preintegration_common import *


def git_text(args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def tensor_summary(obj, key):
    if key not in obj:
        return "missing"
    value = obj[key]
    if hasattr(value, "shape"):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}"
    return type(value).__name__


def main():
    ensure_dirs()
    latents_exists = LATENTS_PATH.exists()
    probe_exists = LINEAR_PROBE_PATH.exists()
    checkpoint_exists = CHECKPOINT_PATH.exists()

    latent_info = {}
    if latents_exists:
        latents = torch.load(LATENTS_PATH, map_location="cpu")
        latent_info = {k: tensor_summary(latents, k) for k in ["z", "position", "action", "episode_id", "timestep"]}

    probe_info = {}
    if probe_exists:
        probe = torch.load(LINEAR_PROBE_PATH, map_location="cpu")
        probe_info = {k: tensor_summary(probe, k) for k in ["W", "b", "z_mean", "z_std"]}
        for key in ["test_x_r2", "test_y_r2", "test_x_pearson", "test_y_pearson", "test_overall_mse"]:
            if key in probe:
                probe_info[key] = str(probe[key])

    checkpoint_info = "not found"
    if checkpoint_exists:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        checkpoint_info = f"type={type(ckpt).__name__}, keys={list(ckpt.keys())[:20] if isinstance(ckpt, dict) else 'n/a'}"

    report = f"""# TwoRoom Pre-Integration Inputs And Provenance

Generated: {datetime.now().isoformat(timespec="seconds")}

## Git

- Branch: `{git_text(["branch", "--show-current"])}`
- Commit: `{git_text(["rev-parse", "HEAD"])}`
- Commit message: `{git_text(["log", "-1", "--oneline"])}`
- Status:

```text
{git_text(["status", "--short"])}
```

## Inputs

- Latent cache: `{LATENTS_PATH}` exists={latents_exists}
- Linear readout: `{LINEAR_PROBE_PATH}` exists={probe_exists}
- Official baseline checkpoint: `{CHECKPOINT_PATH}` exists={checkpoint_exists}

## Latent Cache Summary

{table_md([{"key": k, "summary": v} for k, v in latent_info.items()], ["key", "summary"]) if latent_info else "Latent cache missing."}

## Linear Readout Summary

{table_md([{"key": k, "summary": v} for k, v in probe_info.items()], ["key", "summary"]) if probe_info else "Linear probe missing."}

## Checkpoint Summary

```text
{checkpoint_info}
```

## Scope

These scripts are external diagnostics before LeWM predictor integration. They do not modify encoder, predictor, loss, TwoRoom environment logic, or any LeWM training code. KANFIS-style probes in this folder are analysis baselines only and are not inserted into LeWM.
"""
    path = PREINT_REPORT / "00_inputs_and_provenance.md"
    path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
