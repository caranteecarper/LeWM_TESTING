from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v2_common import (
    REPO_ROOT,
    PAIR_PATH,
    V2_OUT,
    V2_REPORT,
    SPLIT_PATH,
    ensure_dirs as ensure_v2_dirs,
    git_text,
    load_pairs,
    save_or_load_split,
    indices_from_split,
    standardize_from_train,
    inverse_standardize,
    metric_dict,
    fit_linear,
    predict_linear,
    train_eval_regressor,
    train_model,
    batched_predict,
    corr_matrix,
    pearson,
    table_md,
    MLPRegressor,
    LinearBottleneckRegressor,
    KANFISStyleTSKRegressor,
)


V3_OUT = REPO_ROOT / "outputs" / "tworoom_preintegration_v3"
V3_REPORT = REPO_ROOT / "reports" / "tworoom_preintegration_v3"
TEACHER_PATH = V3_OUT / "v3_teacher_labels.pt"
DYN_DIR = V3_OUT / "dynamics_aware_h"


def ensure_dirs():
    ensure_v2_dirs()
    V3_OUT.mkdir(parents=True, exist_ok=True)
    V3_REPORT.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = V3_REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def load_teacher():
    return torch.load(TEACHER_PATH, map_location="cpu")


def binary_metrics(logits, target):
    logits = logits.float()
    target = target.float()
    prob = torch.sigmoid(logits)
    pred = prob >= 0.5
    t = target >= 0.5
    tp = (pred & t).sum(0).float()
    fp = (pred & ~t).sum(0).float()
    fn = (~pred & t).sum(0).float()
    tn = (~pred & ~t).sum(0).float()
    acc = ((tp + tn) / (tp + tn + fp + fn).clamp_min(1)).mean().item()
    precision = (tp / (tp + fp).clamp_min(1)).mean().item()
    recall = (tp / (tp + fn).clamp_min(1)).mean().item()
    f1 = (2 * precision * recall / max(precision + recall, 1e-12))
    aucs = []
    for i in range(target.shape[1]):
        y = target[:, i].float()
        s = prob[:, i].float()
        pos = y == 1
        neg = y == 0
        if pos.any() and neg.any():
            ranks = torch.argsort(torch.argsort(s)).float() + 1
            auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
            aucs.append(auc.item())
    return {"hard_acc": acc, "hard_f1": f1, "hard_auc": sum(aucs) / len(aucs) if aucs else float("nan")}


def vector_r2(pred, target):
    pred = pred.float()
    target = target.float()
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    r2 = 1 - ss_res / ss_tot
    return {"mse": ((pred - target) ** 2).mean().item(), "mean_r2": r2.mean().item(), "dim_r2_min": r2.min().item()}


class DynamicsAwareExtractor(nn.Module):
    def __init__(self, z_dim=192, h_dim=16, action_dim=10, hard_dim=5):
        super().__init__()
        self.h_dim = h_dim
        self.encoder = nn.Sequential(nn.Linear(z_dim, 384), nn.GELU(), nn.Linear(384, h_dim))
        self.pos_head = nn.Sequential(nn.LayerNorm(h_dim), nn.Linear(h_dim, 128), nn.GELU(), nn.Linear(128, 2))
        ha = h_dim + action_dim
        self.delta_head = nn.Sequential(nn.LayerNorm(ha), nn.Linear(ha, 128), nn.GELU(), nn.Linear(128, 2))
        self.res_head = nn.Sequential(nn.LayerNorm(ha), nn.Linear(ha, 128), nn.GELU(), nn.Linear(128, 2))
        self.teacher_res_head = nn.Sequential(nn.LayerNorm(ha), nn.Linear(ha, 128), nn.GELU(), nn.Linear(128, 2))
        self.hard_head = nn.Sequential(nn.LayerNorm(ha), nn.Linear(ha, 128), nn.GELU(), nn.Linear(128, hard_dim))
        self.hnext_head = nn.Sequential(nn.LayerNorm(ha), nn.Linear(ha, 128), nn.GELU(), nn.Linear(128, h_dim))
        self.dh_head = nn.Sequential(nn.LayerNorm(ha), nn.Linear(ha, 128), nn.GELU(), nn.Linear(128, h_dim))

    def encode(self, z):
        return self.encoder(z)

    def forward(self, z, action):
        h = self.encode(z)
        ha = torch.cat([h, action], 1)
        return {
            "h": h,
            "pos": self.pos_head(h),
            "delta": self.delta_head(ha),
            "residual": self.res_head(ha),
            "teacher_residual": self.teacher_res_head(ha),
            "hard_logits": self.hard_head(ha),
            "h_next": self.hnext_head(ha),
            "delta_h": self.dh_head(ha),
        }


PROFILES = {
    "light_aux": {
        "pos": 1.0,
        "delta": 0.2,
        "res": 0.2,
        "teacher": 0.2,
        "hard": 0.1,
        "hnext": 0.1,
        "dh": 0.1,
        "decor": 0.001,
        "var": 0.01,
        "hard_alpha": 0.0,
        "hard_beta": 0.0,
        "hard_gamma": 0.0,
    },
    "medium_aux": {
        "pos": 1.0,
        "delta": 0.6,
        "res": 0.5,
        "teacher": 0.5,
        "hard": 0.3,
        "hnext": 0.3,
        "dh": 0.3,
        "decor": 0.002,
        "var": 0.01,
        "hard_alpha": 0.5,
        "hard_beta": 0.5,
        "hard_gamma": 0.5,
    },
    "hard_weighted": {
        "pos": 1.0,
        "delta": 0.8,
        "res": 0.8,
        "teacher": 0.8,
        "hard": 0.8,
        "hnext": 0.4,
        "dh": 0.4,
        "decor": 0.002,
        "var": 0.02,
        "hard_alpha": 1.0,
        "hard_beta": 1.0,
        "hard_gamma": 1.0,
    },
}


VARIANTS = {
    "pos_only": {"delta": False, "res": False, "teacher": False, "hard": False, "hnext": False},
    "pos_delta": {"delta": True, "res": False, "teacher": False, "hard": False, "hnext": False},
    "pos_residual_teacher": {"delta": False, "res": True, "teacher": True, "hard": False, "hnext": False},
    "pos_hard_label": {"delta": False, "res": False, "teacher": False, "hard": True, "hnext": False},
    "pos_hnext": {"delta": False, "res": False, "teacher": False, "hard": False, "hnext": True},
    "pos_delta_residual_teacher_hard_hnext": {"delta": True, "res": True, "teacher": True, "hard": True, "hnext": True},
}


def load_best_model_state(path):
    obj = torch.load(path, map_location="cpu")
    model = DynamicsAwareExtractor(h_dim=int(obj["h_dim"]), hard_dim=int(obj["hard_dim"]))
    model.load_state_dict(obj["state_dict"])
    return model, obj


def encode_with_state(model, state, z):
    z_std = (z.float() - state["z_mean"]) / state["z_std"].clamp_min(1e-6)
    with torch.no_grad():
        return batched_predict(model, z_std, return_h=True)[1].float()


def load_v2_pure_h16(pairs):
    path = V2_OUT / "factor_extractors" / "pure_extractors.pt"
    obj = torch.load(path, map_location="cpu")
    st = obj["pure_mlp_K16"]
    model = MLPRegressor(192, hidden=384, bottleneck_dim=16)
    model.load_state_dict(st["state_dict"])
    zt = (pairs["z_t"].float() - st["z_mean"]) / st["z_std"].clamp_min(1e-6)
    zn = (pairs["z_next"].float() - st["z_mean"]) / st["z_std"].clamp_min(1e-6)
    return batched_predict(model, zt, return_h=True)[1].float(), batched_predict(model, zn, return_h=True)[1].float()

