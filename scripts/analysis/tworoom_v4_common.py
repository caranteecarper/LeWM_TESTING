from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v2_common import (
    REPO_ROOT,
    SPLIT_PATH,
    PAIR_PATH,
    git_text,
    pearson,
    metric_dict,
    table_md,
    standardize_from_train,
    inverse_standardize,
    fit_linear,
    predict_linear,
    train_eval_regressor,
    train_model,
    batched_predict,
    corr_matrix,
    KANFISStyleTSKRegressor,
    MLPRegressor,
)
from tworoom_v3_common import (
    V3_OUT,
    TEACHER_PATH,
    DYN_DIR,
    DynamicsAwareExtractor,
    binary_metrics,
    vector_r2,
)


V4_OUT = REPO_ROOT / "outputs" / "tworoom_kanfis_rule_features"
V4_REPORT = REPO_ROOT / "reports" / "tworoom_kanfis_rule_features"
BEST_H_KEY = "K16_pos_delta_residual_teacher_hard_hnext_light_aux"
BEST_H_DATASET = V4_OUT / "best_h_dataset.pt"
RULE_DIR = V4_OUT / "rule_feature_models"


def ensure_dirs():
    V4_OUT.mkdir(parents=True, exist_ok=True)
    V4_REPORT.mkdir(parents=True, exist_ok=True)
    RULE_DIR.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = V4_REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def load_best_h_model():
    path = DYN_DIR / f"{BEST_H_KEY}.pt"
    obj = torch.load(path, map_location="cpu")
    model = DynamicsAwareExtractor(h_dim=int(obj["h_dim"]), hard_dim=int(obj["hard_dim"]))
    model.load_state_dict(obj["state_dict"])
    return model, obj, path


@torch.no_grad()
def encode_best_h(z):
    model, obj, _ = load_best_h_model()
    norm = obj["norm"]
    z_std = (z.float() - norm["z_mean"]) / norm["z_std"].clamp_min(1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outs = []
    for i in range(0, z_std.shape[0], 16384):
        outs.append(model.encode(z_std[i : i + 16384].to(device)).cpu())
    model.cpu()
    return torch.cat(outs, 0).float()


class KANFISRuleFeatureLayer(nn.Module):
    """Rule-feature extractor. q is normalized rule activation, not a LeWM module."""

    def __init__(self, in_dim=16, num_rules=16):
        super().__init__()
        self.in_dim = in_dim
        self.num_rules = num_rules
        self.centers = nn.Parameter(torch.randn(num_rules, in_dim) * 0.5)
        self.log_sigma = nn.Parameter(torch.zeros(num_rules, in_dim))
        self.rule_logits = nn.Parameter(torch.zeros(num_rules))
        self.input_gate_logits = nn.Parameter(torch.zeros(in_dim))

    @torch.no_grad()
    def initialize_from_data(self, h):
        g = torch.Generator().manual_seed(3072)
        idx = torch.randperm(h.shape[0], generator=g)[: self.num_rules]
        if idx.numel() < self.num_rules:
            idx = idx.repeat((self.num_rules + idx.numel() - 1) // idx.numel())[: self.num_rules]
        self.centers.copy_(h[idx].float())
        sigma = h.float().std(0, keepdim=True).clamp_min(0.2).repeat(self.num_rules, 1)
        self.log_sigma.copy_(torch.log(torch.expm1(sigma)))

    def forward(self, h):
        gate = torch.sigmoid(self.input_gate_logits)
        hg = h.float() * gate
        sigma = F.softplus(self.log_sigma) + 1e-4
        dist = ((hg[:, None, :] - self.centers[None, :, :]) / sigma[None, :, :]).pow(2).mean(-1)
        logits = -0.5 * dist + self.rule_logits[None, :]
        q = torch.softmax(logits, dim=-1)
        return q

    def stats(self, q):
        ent = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum(1)
        active = (q > (1.0 / self.num_rules)).float().sum(1)
        usage = q.mean(0)
        sigma = F.softplus(self.log_sigma).detach().cpu()
        return {
            "q_var_min": q.var(0).min().item(),
            "q_var_mean": q.var(0).mean().item(),
            "q_entropy_mean": ent.mean().item(),
            "active_rule_count_mean": active.mean().item(),
            "rule_usage_min": usage.min().item(),
            "rule_usage_max": usage.max().item(),
            "sigma_min": sigma.min().item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_max": sigma.max().item(),
        }


class RuleFeatureModel(nn.Module):
    def __init__(self, h_dim=16, action_dim=10, num_rules=16, hard_dim=5):
        super().__init__()
        self.rule_layer = KANFISRuleFeatureLayer(h_dim, num_rules)
        q_dim = num_rules
        self.pos_head = nn.Sequential(nn.LayerNorm(q_dim), nn.Linear(q_dim, 64), nn.GELU(), nn.Linear(64, 2))
        qa = q_dim + action_dim
        self.delta_head = nn.Sequential(nn.LayerNorm(qa), nn.Linear(qa, 128), nn.GELU(), nn.Linear(128, 2))
        self.res_head = nn.Sequential(nn.LayerNorm(qa), nn.Linear(qa, 128), nn.GELU(), nn.Linear(128, 2))
        self.hard_head = nn.Sequential(nn.LayerNorm(qa), nn.Linear(qa, 128), nn.GELU(), nn.Linear(128, hard_dim))
        self.hnext_head = nn.Sequential(nn.LayerNorm(qa), nn.Linear(qa, 128), nn.GELU(), nn.Linear(128, h_dim))
        self.dh_head = nn.Sequential(nn.LayerNorm(qa), nn.Linear(qa, 128), nn.GELU(), nn.Linear(128, h_dim))

    def forward(self, h, action):
        q = self.rule_layer(h)
        qa = torch.cat([q, action.float()], 1)
        return {
            "q": q,
            "pos": self.pos_head(q),
            "delta": self.delta_head(qa),
            "residual": self.res_head(qa),
            "hard_logits": self.hard_head(qa),
            "h_next": self.hnext_head(qa),
            "delta_h": self.dh_head(qa),
        }


V4_VARIANTS = {
    "q_pos_only": {"delta": False, "res": False, "hard": False, "hnext": False},
    "q_pos_delta": {"delta": True, "res": False, "hard": False, "hnext": False},
    "q_pos_residual": {"delta": False, "res": True, "hard": False, "hnext": False},
    "q_pos_residual_hard": {"delta": False, "res": True, "hard": True, "hnext": False},
    "q_pos_delta_residual_hard_hnext": {"delta": True, "res": True, "hard": True, "hnext": True},
}


def load_best_h_dataset():
    return torch.load(BEST_H_DATASET, map_location="cpu")


def vector_mse(pred, target):
    return ((pred.float() - target.float()) ** 2).mean().item()


def fit_linear_pred_all(x, y, tr):
    x_std, xm, xs = standardize_from_train(x, tr)
    y_std, ym, ys = standardize_from_train(y, tr)
    w, b = fit_linear(x_std, y_std, tr)
    return inverse_standardize(predict_linear(x_std, w, b), ym, ys)


def save_model_state(path, model, norm, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "norm": norm, "meta": meta}, path)


def load_rule_model(path):
    obj = torch.load(path, map_location="cpu")
    meta = obj["meta"]
    model = RuleFeatureModel(h_dim=meta["h_dim"], action_dim=meta["action_dim"], num_rules=meta["num_rules"], hard_dim=meta["hard_dim"])
    model.load_state_dict(obj["state_dict"])
    return model, obj

