import csv
import math
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v41_common import (
    REPO_ROOT,
    BEST_H_DATASET,
    V41_RULE_DIR,
    SharpQModel,
    hard_sample_weight,
    standardize_v41,
    eval_model_heads,
    save_sharp_model,
    load_sharp_model,
    q_from_model,
)
from tworoom_v4_common import RULE_DIR as V4_RULE_DIR, load_rule_model, table_md, metric_dict
from tworoom_v5_common import load_visual
from tworoom_v51_common import save_unit_visuals, save_group_visuals
from tworoom_v52_common import binary_metrics, fit_ridge, pred_ridge, subset_idx, csv_write, GROUPS


V42_OUT = REPO_ROOT / "outputs" / "tworoom_kanfis_v42_compromise"
V42_REPORT = REPO_ROOT / "reports" / "tworoom_kanfis_v42_compromise"
V42_MODEL_DIR = V42_OUT / "models"
V42_TABLE = V42_OUT / "tables"
V42_FIG = V42_OUT / "figures"


def ensure_dirs():
    V42_OUT.mkdir(parents=True, exist_ok=True)
    V42_REPORT.mkdir(parents=True, exist_ok=True)
    V42_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    V42_TABLE.mkdir(parents=True, exist_ok=True)
    V42_FIG.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = V42_REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def git_text(args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


V42_VARIANTS = {
    "v42_tau07_R16": {"rules": 16, "top_k": None, "tau": 0.7, "hard": "light", "entropy": 0.006},
    "v42_tau05_R16": {"rules": 16, "top_k": None, "tau": 0.5, "hard": "light", "entropy": 0.008},
    "v42_tau07_topk8_R16": {"rules": 16, "top_k": 8, "tau": 0.7, "hard": "light", "entropy": 0.008},
    "v42_tau05_topk8_R16": {"rules": 16, "top_k": 8, "tau": 0.5, "hard": "light", "entropy": 0.01},
    "v42_tau07_topk6_R16": {"rules": 16, "top_k": 6, "tau": 0.7, "hard": "light", "entropy": 0.01},
    "v42_tau05_topk6_R16": {"rules": 16, "top_k": 6, "tau": 0.5, "hard": "light", "entropy": 0.012},
    "v42_tau07_topk6_hard_light_R16": {"rules": 16, "top_k": 6, "tau": 0.7, "hard": "light", "entropy": 0.01},
    "v42_tau05_topk6_hard_light_R16": {"rules": 16, "top_k": 6, "tau": 0.5, "hard": "light", "entropy": 0.012},
    "v42_tau07_topk6_hard_medium_R16": {"rules": 16, "top_k": 6, "tau": 0.7, "hard": "medium", "entropy": 0.012},
    "v42_tau05_topk6_hard_medium_R16": {"rules": 16, "top_k": 6, "tau": 0.5, "hard": "medium", "entropy": 0.014},
    "v42_tau07_R20": {"rules": 20, "top_k": None, "tau": 0.7, "hard": "light", "entropy": 0.008},
    "v42_tau05_topk8_R20": {"rules": 20, "top_k": 8, "tau": 0.5, "hard": "light", "entropy": 0.012},
}


def load_base_data():
    return torch.load(BEST_H_DATASET, map_location="cpu")


def tau_for_epoch(epoch, epochs, end):
    t = epoch / max(epochs - 1, 1)
    return end + 0.5 * (1.0 - end) * (1 + math.cos(math.pi * t))


def score_v42(row):
    active = row["active_rule_count"]
    return (
        row["res_overall_mse"]
        + 0.25 * row["action_error_top10_res_mse"]
        + 0.15 * row["large_action_small_disp_res_mse"]
        + 0.05 * abs(active - 5.0)
        - 0.02 * row["hard_auc"]
    )


def compute_v4_q(data):
    summary = torch.load(V4_RULE_DIR / "summary.pt", map_location="cpu")
    key = summary["best_key"]
    model, obj = load_rule_model(V4_RULE_DIR / f"{key}.pt")
    norm = obj["norm"]
    h = (data["h_t"].float() - norm["h_t_mean"]) / norm["h_t_std"].clamp_min(1e-6)
    action = (data["action"].float() - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    q = []
    with torch.no_grad():
        for i in range(0, h.shape[0], 32768):
            q.append(model(h[i : i + 32768], action[i : i + 32768])["q"])
    return key, torch.cat(q, 0)


def compute_v41_q(data):
    key = torch.load(V41_RULE_DIR / "summary.pt", map_location="cpu")["best_key"]
    model, obj = load_sharp_model(V41_RULE_DIR / f"{key}.pt")
    return key, q_from_model(model, obj, data, hard_topk=True)


def v42_model_path(key):
    path = V42_MODEL_DIR / f"{key}.pt"
    if path.exists():
        return path
    legacy_path = V42_MODEL_DIR / f"{key}.pt.pt"
    if legacy_path.exists():
        return legacy_path
    return path


def compute_v42_q(data):
    key = torch.load(V42_MODEL_DIR / "summary.pt", map_location="cpu")["best_key"]
    model, obj = load_sharp_model(v42_model_path(key))
    return key, q_from_model(model, obj, data, hard_topk=bool(obj["meta"].get("top_k")))
