import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
LATENTS_PATH = REPO_ROOT / "outputs" / "tworoom_encoder_readout" / "tworoom_latents.pt"
LINEAR_PROBE_PATH = REPO_ROOT / "outputs" / "tworoom_encoder_readout" / "linear_probe_position.pt"
PREINT_OUT = REPO_ROOT / "outputs" / "tworoom_preintegration"
PREINT_REPORT = REPO_ROOT / "reports" / "tworoom_preintegration"
CHECKPOINT_PATH = Path("/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt")


def ensure_dirs():
    PREINT_OUT.mkdir(parents=True, exist_ok=True)
    PREINT_REPORT.mkdir(parents=True, exist_ok=True)


def load_latents(path=LATENTS_PATH):
    return torch.load(path, map_location="cpu")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def pearson(x, y):
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    den = x.norm() * y.norm()
    return (x @ y / den).item() if den > 0 else float("nan")


def metric_dict(pred, target, prefix=""):
    pred = pred.float()
    target = target.float()
    mse = ((pred - target) ** 2).mean(dim=0)
    rmse = mse.sqrt()
    ss_res = ((pred - target) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = 1 - ss_res / ss_tot
    out = {
        f"{prefix}x_mse": mse[0].item(),
        f"{prefix}y_mse": mse[1].item(),
        f"{prefix}x_rmse": rmse[0].item(),
        f"{prefix}y_rmse": rmse[1].item(),
        f"{prefix}x_r2": r2[0].item(),
        f"{prefix}y_r2": r2[1].item(),
        f"{prefix}x_pearson": pearson(pred[:, 0], target[:, 0]),
        f"{prefix}y_pearson": pearson(pred[:, 1], target[:, 1]),
        f"{prefix}overall_mse": ((pred - target) ** 2).mean().item(),
    }
    return out


def split_by_episode(episode_id, seed=3072):
    episode_id = episode_id.long()
    unique = torch.unique(episode_id)
    g = torch.Generator().manual_seed(seed)
    unique = unique[torch.randperm(unique.numel(), generator=g)]
    n_train = int(0.8 * unique.numel())
    n_val = int(0.1 * unique.numel())
    train_eps = unique[:n_train]
    val_eps = unique[n_train : n_train + n_val]
    test_eps = unique[n_train + n_val :]

    def idx(eps):
        return torch.nonzero(torch.isin(episode_id, eps), as_tuple=False).squeeze(1)

    return idx(train_eps), idx(val_eps), idx(test_eps), train_eps, val_eps, test_eps


def standardize_from_train(x, train_idx):
    mean = x[train_idx].float().mean(0, keepdim=True)
    std = x[train_idx].float().std(0, keepdim=True).clamp_min(1e-6)
    return (x.float() - mean) / std, mean.squeeze(0), std.squeeze(0)


def add_bias(x):
    return torch.cat([x.float(), torch.ones(x.shape[0], 1)], dim=1)


def fit_linear(x, y, train_idx, ridge=1e-4):
    xt = add_bias(x[train_idx])
    yt = y[train_idx].float()
    eye = torch.eye(xt.shape[1])
    eye[-1, -1] = 0.0
    w = torch.linalg.solve(xt.T @ xt + ridge * eye, xt.T @ yt)
    return w[:-1].T.contiguous(), w[-1].contiguous()


def predict_linear(x, weight, bias):
    return x.float() @ weight.T + bias


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=2, bottleneck_dim=None):
        super().__init__()
        if bottleneck_dim is None:
            self.encoder = nn.Identity()
            mid = in_dim
        else:
            self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, bottleneck_dim))
            mid = bottleneck_dim
        self.head = nn.Sequential(nn.LayerNorm(mid), nn.Linear(mid, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x, return_h=False):
        h = self.encoder(x)
        y = self.head(h)
        return (y, h) if return_h else y


class KANFISStyleRegressor(nn.Module):
    def __init__(self, in_dim, out_dim=2, rules=16):
        super().__init__()
        self.in_dim = in_dim
        self.rules = rules
        self.gate_logits = nn.Parameter(torch.zeros(in_dim))
        self.centers = nn.Parameter(torch.randn(rules, in_dim) * 0.2)
        self.log_sigma = nn.Parameter(torch.zeros(rules, in_dim))
        self.rule_weight = nn.Parameter(torch.randn(rules, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        gate = torch.sigmoid(self.gate_logits)
        xg = x * gate
        sigma = F.softplus(self.log_sigma) + 1e-4
        dist = ((xg[:, None, :] - self.centers[None, :, :]) / sigma[None, :, :]).pow(2).mean(-1)
        act = torch.softmax(-0.5 * dist, dim=-1)
        return act @ self.rule_weight + self.bias

    def importance(self):
        return torch.sigmoid(self.gate_logits).detach().cpu()


def train_model(model, x, y, train_idx, val_idx, *, epochs=8, batch_size=4096, lr=1e-3, max_train=200000, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    if max_train and train_idx.numel() > max_train:
        g = torch.Generator().manual_seed(3072)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:max_train]]
    ds = TensorDataset(x[train_idx].float(), y[train_idx].float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = batched_predict(model, x[val_idx], device=device)
            val = F.mse_loss(pred.cpu(), y[val_idx].float()).item()
        model.to(device)
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.cpu()
    return model, best_val


@torch.no_grad()
def batched_predict(model, x, batch_size=16384, device="cuda", return_h=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    outs = []
    hs = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].float().to(device)
        if return_h:
            yb, hb = model(xb, return_h=True)
            outs.append(yb.cpu())
            hs.append(hb.cpu())
        else:
            outs.append(model(xb).cpu())
    model.cpu()
    if return_h:
        return torch.cat(outs, 0), torch.cat(hs, 0)
    return torch.cat(outs, 0)


def topk_from_weight(weight, k):
    score = weight.abs().sum(0)
    return torch.topk(score, k=min(k, score.numel())).indices


def save_scatter(path, x, y, xlabel, ylabel, title, max_points=30000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu().float().reshape(-1)
    y = y.detach().cpu().float().reshape(-1)
    if x.numel() > max_points:
        idx = torch.linspace(0, x.numel() - 1, max_points).long()
        x = x[idx]
        y = y[idx]
    plt.figure(figsize=(5, 5))
    plt.scatter(x.numpy(), y.numpy(), s=2, alpha=0.25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_bar(path, values, title, topk=32):
    values = values.detach().cpu().float()
    vals, idx = torch.topk(values.abs(), k=min(topk, values.numel()))
    plt.figure(figsize=(9, 4))
    plt.bar([str(i.item()) for i in idx], values[idx].numpy())
    plt.title(title)
    plt.xlabel("dimension")
    plt.ylabel("importance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return idx.tolist()


def table_md(rows, fields):
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for r in rows:
        lines.append("|" + "|".join(str(r.get(f, "")) for f in fields) + "|")
    return "\n".join(lines)
