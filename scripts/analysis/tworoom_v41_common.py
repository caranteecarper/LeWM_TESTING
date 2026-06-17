import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v4_common import (
    REPO_ROOT,
    V4_OUT,
    V4_REPORT,
    BEST_H_DATASET,
    RULE_DIR as V4_RULE_DIR,
    git_text,
    pearson,
    metric_dict,
    table_md,
    standardize_from_train,
    inverse_standardize,
    fit_linear,
    predict_linear,
    train_model,
    batched_predict,
    MLPRegressor,
    RuleFeatureModel,
    load_rule_model,
    binary_metrics,
)


V41_OUT = REPO_ROOT / "outputs" / "tworoom_kanfis_rule_features_v41"
V41_REPORT = REPO_ROOT / "reports" / "tworoom_kanfis_rule_features_v41"
V41_RULE_DIR = V41_OUT / "sharp_q_models"
V41_RULE_TABLE_DIR = V41_OUT / "rules"


def ensure_dirs():
    V41_OUT.mkdir(parents=True, exist_ok=True)
    V41_REPORT.mkdir(parents=True, exist_ok=True)
    V41_RULE_DIR.mkdir(parents=True, exist_ok=True)
    V41_RULE_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = V41_REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def load_data():
    return torch.load(BEST_H_DATASET, map_location="cpu")


def vector_mse(pred, target):
    return ((pred.float() - target.float()) ** 2).mean().item()


def standardize_v41(data, tr):
    keys = ["h_t", "action", "position_t", "delta_position", "residual"]
    std_data = {}
    norm = {}
    for key in keys:
        xs, mean, std = standardize_from_train(data[key].float(), tr)
        std_data[key] = xs
        norm[f"{key}_mean"] = mean
        norm[f"{key}_std"] = std
    return std_data, norm


class SharpKANFISRuleLayer(nn.Module):
    """White-box q extractor: q is only rule firing strengths from h."""

    def __init__(self, in_dim=16, num_rules=16, top_k=None):
        super().__init__()
        self.in_dim = in_dim
        self.num_rules = num_rules
        self.top_k = top_k
        self.centers = nn.Parameter(torch.randn(num_rules, in_dim) * 0.5)
        self.log_sigma = nn.Parameter(torch.zeros(num_rules, in_dim))
        self.rule_logits = nn.Parameter(torch.zeros(num_rules))
        self.input_gate_logits = nn.Parameter(torch.zeros(in_dim))

    @torch.no_grad()
    def initialize_from_data(self, h):
        g = torch.Generator().manual_seed(4101 + self.num_rules)
        idx = torch.randperm(h.shape[0], generator=g)[: self.num_rules]
        if idx.numel() < self.num_rules:
            idx = idx.repeat(math.ceil(self.num_rules / max(idx.numel(), 1)))[: self.num_rules]
        self.centers.copy_(h[idx].float())
        sigma = h.float().std(0, keepdim=True).clamp_min(0.15).repeat(self.num_rules, 1)
        self.log_sigma.copy_(torch.log(torch.expm1(sigma)))

    def logits(self, h):
        gate = torch.sigmoid(self.input_gate_logits)
        hg = h.float() * gate
        sigma = F.softplus(self.log_sigma) + 1e-4
        dist = ((hg[:, None, :] - self.centers[None, :, :]) / sigma[None, :, :]).pow(2).mean(-1)
        return -0.5 * dist + self.rule_logits[None, :]

    def forward(self, h, tau=1.0, hard_topk=False):
        logits = self.logits(h) / max(float(tau), 1e-4)
        if self.top_k and hard_topk:
            top = torch.topk(logits, k=min(self.top_k, logits.shape[1]), dim=1).indices
            mask = torch.full_like(logits, -1e9)
            logits = mask.scatter(1, top, logits.gather(1, top))
        return torch.softmax(logits, dim=-1)

    def regularizers(self, q, w_entropy, w_balance, w_diversity, w_gate):
        entropy = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum(1).mean()
        usage = q.mean(0).clamp_min(1e-8)
        uniform = torch.full_like(usage, 1.0 / usage.numel())
        balance = (usage * (usage / uniform).log()).sum()
        c = F.normalize(self.centers, dim=1)
        gram = c @ c.T
        diversity = (gram - torch.eye(gram.shape[0], device=gram.device)).pow(2).mean()
        gate_sparse = torch.sigmoid(self.input_gate_logits).mean()
        return w_entropy * entropy + w_balance * balance + w_diversity * diversity + w_gate * gate_sparse

    def stats(self, q):
        ent = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum(1)
        usage = q.mean(0)
        active = (q > (1.0 / q.shape[1])).float().sum(1)
        usage_ent = -(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum()
        c = F.normalize(self.centers.detach().cpu(), dim=1)
        dist = torch.cdist(c, c)
        dist = dist + torch.eye(dist.shape[0]) * 10.0
        return {
            "q_entropy": ent.mean().item(),
            "active_rule_count": active.mean().item(),
            "q_var_mean": q.var(0).mean().item(),
            "usage_min": usage.min().item(),
            "usage_max": usage.max().item(),
            "usage_entropy": usage_ent.item(),
            "center_min_distance": dist.min().item(),
            "gate_mean": torch.sigmoid(self.input_gate_logits.detach()).mean().item(),
        }


class SharpQModel(nn.Module):
    def __init__(self, h_dim=16, action_dim=10, num_rules=16, hard_dim=5, top_k=None):
        super().__init__()
        self.rule_layer = SharpKANFISRuleLayer(h_dim, num_rules, top_k=top_k)
        self.pos_head = nn.Linear(num_rules, 2)
        qa = num_rules + action_dim
        self.delta_head = nn.Linear(qa, 2)
        self.res_head = nn.Linear(qa, 2)
        self.hard_head = nn.Linear(qa, hard_dim)

    def forward(self, h, action, tau=1.0, hard_topk=False):
        q = self.rule_layer(h, tau=tau, hard_topk=hard_topk)
        qa = torch.cat([q, action.float()], 1)
        return {
            "q": q,
            "pos": self.pos_head(q),
            "delta": self.delta_head(qa),
            "residual": self.res_head(qa),
            "hard_logits": self.hard_head(qa),
        }


V41_VARIANTS = {
    "baseline_v4_reproduce": {"rules": 16, "top_k": None, "tau_end": 1.0, "hard": "light", "entropy": 0.005},
    "sharp_tau_R16": {"rules": 16, "top_k": None, "tau_end": 0.2, "hard": "light", "entropy": 0.015},
    "sharp_tau_topk3_R16": {"rules": 16, "top_k": 3, "tau_end": 0.2, "hard": "light", "entropy": 0.02},
    "sharp_tau_topk4_R16": {"rules": 16, "top_k": 4, "tau_end": 0.2, "hard": "light", "entropy": 0.02},
    "sharp_tau_topk3_hard_medium_R16": {"rules": 16, "top_k": 3, "tau_end": 0.2, "hard": "medium", "entropy": 0.025},
    "sharp_tau_topk4_hard_medium_R16": {"rules": 16, "top_k": 4, "tau_end": 0.2, "hard": "medium", "entropy": 0.025},
    "sharp_tau_topk3_hard_strong_R16": {"rules": 16, "top_k": 3, "tau_end": 0.2, "hard": "strong", "entropy": 0.03},
    "sharp_tau_topk4_hard_strong_R16": {"rules": 16, "top_k": 4, "tau_end": 0.2, "hard": "strong", "entropy": 0.03},
    "sharp_tau_topk3_hard_medium_R24": {"rules": 24, "top_k": 3, "tau_end": 0.2, "hard": "medium", "entropy": 0.025},
    "sharp_tau_topk4_hard_medium_R24": {"rules": 24, "top_k": 4, "tau_end": 0.2, "hard": "medium", "entropy": 0.025},
}


HARD_WEIGHTS = {
    "light": (1.0, 1.0, 2.0),
    "medium": (2.0, 2.0, 4.0),
    "strong": (4.0, 4.0, 6.0),
}


def hard_sample_weight(data, mode):
    names = data["hard_label_names"]
    labels = data["hard_labels"].float()
    a, b, c = HARD_WEIGHTS[mode]
    return (
        1.0
        + a * labels[:, names.index("action_error_top10")]
        + b * labels[:, names.index("high_residual_top10")]
        + c * labels[:, names.index("large_action_small_disp")]
    )


def subset_index(data, name):
    te = data["test_idx"]
    if name == "all_test":
        return te
    names = data["hard_label_names"]
    labels = data["hard_labels"]
    return te[labels[te, names.index(name)] > 0.5]


def eval_model_heads(model, data, std_data, norm, idx, tau=0.2, hard_topk=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outs = []
    with torch.no_grad():
        model.to(device).eval()
        for i in range(0, idx.numel(), 16384):
            j = idx[i : i + 16384]
            out = model(std_data["h_t"][j].to(device), std_data["action"][j].to(device), tau=tau, hard_topk=hard_topk)
            outs.append({k: v.cpu() for k, v in out.items()})
        model.cpu()
    merged = {k: torch.cat([o[k] for o in outs], 0) for k in outs[0]}
    pos = inverse_standardize(merged["pos"], norm["position_t_mean"], norm["position_t_std"])
    delta = inverse_standardize(merged["delta"], norm["delta_position_mean"], norm["delta_position_std"])
    residual = inverse_standardize(merged["residual"], norm["residual_mean"], norm["residual_std"])
    row = {
        **metric_dict(pos, data["position_t"][idx], prefix="pos_"),
        **metric_dict(delta, data["delta_position"][idx], prefix="delta_"),
        **metric_dict(residual, data["residual"][idx], prefix="res_"),
        **binary_metrics(merged["hard_logits"], data["hard_labels"][idx]),
        **model.rule_layer.stats(merged["q"]),
    }
    return row, merged


def save_sharp_model(name, model, norm, meta):
    path = V41_RULE_DIR / f"{name}.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "norm": norm, "meta": meta}, path)
    return path


def load_sharp_model(path):
    obj = torch.load(path, map_location="cpu")
    meta = obj["meta"]
    model = SharpQModel(
        h_dim=meta["h_dim"],
        action_dim=meta["action_dim"],
        num_rules=meta["num_rules"],
        hard_dim=meta["hard_dim"],
        top_k=meta.get("top_k"),
    )
    model.load_state_dict(obj["state_dict"])
    return model, obj


def q_from_model(model, obj, data, hard_topk=True):
    norm = obj["norm"]
    h = (data["h_t"].float() - norm["h_t_mean"]) / norm["h_t_std"].clamp_min(1e-6) if "h_t_mean" in norm else (data["h_t"].float() - norm["h_t_mean"])
    action = (data["action"].float() - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    tau = obj["meta"].get("tau_end", 0.2)
    q = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.to(device).eval()
        for i in range(0, h.shape[0], 16384):
            q.append(model(h[i : i + 16384].to(device), action[i : i + 16384].to(device), tau=tau, hard_topk=hard_topk)["q"].cpu())
        model.cpu()
    return torch.cat(q, 0)

