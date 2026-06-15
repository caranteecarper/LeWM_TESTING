import json
import os
import sys
from pathlib import Path

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import get_img_preprocessor

DEFAULT_CHECKPOINT = Path("/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tworoom_encoder_readout"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "tworoom_encoder_readout"


def compose_tworoom_cfg(batch_size=128, num_workers=0):
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "config" / "train"), version_base=None):
        cfg = hydra.compose(
            config_name="lewm",
            overrides=[
                "data=tworoom",
                f"loader.batch_size={batch_size}",
                f"loader.num_workers={num_workers}",
                "loader.persistent_workers=false",
                "loader.prefetch_factor=null",
                "wandb.enabled=false",
            ],
        )
    return cfg


def load_tworoom_dataset(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    for key in ("pos_agent", "ep_idx", "step_idx"):
        if key not in dataset_cfg["keys_to_load"]:
            dataset_cfg["keys_to_load"].append(key)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", "/data/lzt26/stable-wm")
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    )
    return dataset


def load_model(cfg, checkpoint_path):
    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()
    return model, list(missing), list(unexpected)


def batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def metadata(checkpoint_path, cfg, dataset, extra=None):
    meta = {
        "checkpoint": str(checkpoint_path),
        "repo_commit": git_commit(),
        "config": "config/train/lewm.yaml data=tworoom",
        "frameskip": int(cfg.data.dataset.frameskip),
        "action_dim_raw": int(dataset.get_dim("action")),
        "action_dim_block": int(cfg.model.action_encoder.input_dim),
        "latent_dim": int(cfg.embed_dim),
    }
    if extra:
        meta.update(extra)
    return meta


def git_commit():
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
