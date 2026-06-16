import math
import subprocess
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
PAIR_PATH = REPO_ROOT / "outputs" / "tworoom_preintegration" / "tworoom_pairs.pt"
V1_REPORT = REPO_ROOT / "reports" / "tworoom_preintegration"
V2_OUT = REPO_ROOT / "outputs" / "tworoom_preintegration_v2"
V2_REPORT = REPO_ROOT / "reports" / "tworoom_preintegration_v2"
SPLIT_PATH = V2_OUT / "splits_episode_seed3072.pt"
CHECKPOINT_PATH = Path("/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt")


def ensure_dirs():
    V2_OUT.mkdir(parents=True, exist_ok=True)
    V2_REPORT.mkdir(parents=True, exist_ok=True)


def git_text(args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def load_latents():
    return torch.load(LATENTS_PATH, map_location="cpu")


def load_pairs():
    return torch.load(PAIR_PATH, map_location="cpu")


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


def save_or_load_split(episode_id):
    ensure_dirs()
    if SPLIT_PATH.exists():
        return torch.load(SPLIT_PATH, map_location="cpu")
    train_idx, val_idx, test_idx, train_eps, val_eps, test_eps = split_by_episode(episode_id)
    split = {
        "seed": 3072,
        "train_eps": train_eps,
        "val_eps": val_eps,
        "test_eps": test_eps,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    torch.save(split, SPLIT_PATH)
    return split


def indices_from_split(episode_id, split):
    episode_id = episode_id.long()

    def idx(eps):
        return torch.nonzero(torch.isin(episode_id, eps.long()), as_tuple=False).squeeze(1)

    return idx(split["train_eps"]), idx(split["val_eps"]), idx(split["test_eps"])


def standardize_from_train(x, train_idx):
    mean = x[train_idx].float().mean(0, keepdim=True)
    std = x[train_idx].float().std(0, keepdim=True).clamp_min(1e-6)
    return (x.float() - mean) / std, mean.squeeze(0), std.squeeze(0)


def inverse_standardize(x, mean, std):
    return x.float() * std + mean


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


def add_bias(x):
    return torch.cat([x.float(), torch.ones(x.shape[0], 1)], dim=1)


def fit_linear(x, y, train_idx, ridge=1e-4):
    xt = add_bias(x[train_idx])
    yt = y[train_idx].float()
    eye = torch.eye(xt.shape[1])
    eye[-1, -1] = 0.0
    w = torch.linalg.solve(xt.T @ xt + ridge * eye, xt.T @ yt)
    return w[:-1].T.contiguous(), w[-1].contiguous()


def predict_linear(x, w, b):
    return x.float() @ w.T + b


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


class LinearBottleneckRegressor(nn.Module):
    def __init__(self, in_dim, out_dim=2, bottleneck_dim=16):
        super().__init__()
        self.encoder = nn.Linear(in_dim, bottleneck_dim)
        self.head = nn.Linear(bottleneck_dim, out_dim)

    def forward(self, x, return_h=False):
        h = self.encoder(x)
        y = self.head(h)
        return (y, h) if return_h else y


class KANFISStyleTSKRegressor(nn.Module):
    """External neuro-fuzzy sanity probe, not a LeWM module or paper-claim implementation."""

    def __init__(self, in_dim, out_dim=2, rules=16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rules = rules
        self.gate_logits = nn.Parameter(torch.zeros(in_dim))
        self.centers = nn.Parameter(torch.randn(rules, in_dim) * 0.5)
        self.log_sigma = nn.Parameter(torch.full((rules, in_dim), math.log(math.e - 1.0)))
        self.local_weight = nn.Parameter(torch.randn(rules, in_dim + 1, out_dim) * 0.02)

    @torch.no_grad()
    def initialize_from_data(self, x):
        if x.shape[0] == 0:
            return
        g = torch.Generator().manual_seed(3072)
        idx = torch.randperm(x.shape[0], generator=g)[: self.rules]
        if idx.numel() < self.rules:
            idx = idx.repeat(math.ceil(self.rules / max(idx.numel(), 1)))[: self.rules]
        self.centers.copy_(x[idx].float())
        sigma = x.float().std(0, keepdim=True).clamp_min(0.2).repeat(self.rules, 1)
        self.log_sigma.copy_(torch.log(torch.expm1(sigma)))

    def firing(self, x):
        gate = torch.sigmoid(self.gate_logits)
        xg = x * gate
        sigma = F.softplus(self.log_sigma) + 1e-4
        dist = ((xg[:, None, :] - self.centers[None, :, :]) / sigma[None, :, :]).pow(2).mean(-1)
        return torch.softmax(-0.5 * dist, dim=-1)

    def forward(self, x):
        act = self.firing(x)
        x_aug = torch.cat([x.float(), torch.ones(x.shape[0], 1, device=x.device)], dim=1)
        local = torch.einsum("bi,rio->bro", x_aug, self.local_weight)
        return (act.unsqueeze(-1) * local).sum(1)

    def stats(self):
        sigma = F.softplus(self.log_sigma).detach().cpu()
        return {
            "sigma_min": sigma.min().item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_max": sigma.max().item(),
            "gate_mean": torch.sigmoid(self.gate_logits).detach().cpu().mean().item(),
        }

    def importance(self):
        return torch.sigmoid(self.gate_logits).detach().cpu()


def train_model(model, x, y, train_idx, val_idx, *, epochs=8, batch_size=4096, lr=1e-3, max_train=200000, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if max_train and train_idx.numel() > max_train:
        g = torch.Generator().manual_seed(3072)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:max_train]]
    model.to(device)
    if hasattr(model, "initialize_from_data"):
        model.initialize_from_data(x[train_idx[: min(train_idx.numel(), 8192)]].float())
    ds = TensorDataset(x[train_idx].float(), y[train_idx].float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        grad_norm = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = math.sqrt(sum((p.grad.detach().norm().item() ** 2) for p in model.parameters() if p.grad is not None))
            opt.step()
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]
        model.eval()
        with torch.no_grad():
            pred = batched_predict(model, x[val_idx], device=device)
            val = F.mse_loss(pred.cpu(), y[val_idx].float()).item()
        model.to(device)
        train_loss = total / max(count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val, "grad_norm": grad_norm})
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.cpu()
    return model, best_val, history


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


def train_eval_regressor(name, model, x, y, train_idx, val_idx, test_idx, *, epochs=8, lr=1e-3, max_train=200000):
    x_std, x_mean, x_scale = standardize_from_train(x, train_idx)
    y_std, y_mean, y_scale = standardize_from_train(y, train_idx)
    if model == "linear":
        w, b = fit_linear(x_std, y_std, train_idx)
        pred_std = predict_linear(x_std[test_idx], w, b)
        pred = inverse_standardize(pred_std, y_mean, y_scale)
        return {"name": name, "model": "linear", **metric_dict(pred, y[test_idx])}, {"W": w, "b": b, "x_mean": x_mean, "x_std": x_scale, "y_mean": y_mean, "y_std": y_scale}
    if model == "mlp":
        net = MLPRegressor(x.shape[1], hidden=256, out_dim=y.shape[1])
    elif model == "kanfis":
        net = KANFISStyleTSKRegressor(x.shape[1], out_dim=y.shape[1], rules=16)
    else:
        raise ValueError(model)
    net, _, hist = train_model(net, x_std, y_std, train_idx, val_idx, epochs=epochs, lr=lr, max_train=max_train)
    pred = inverse_standardize(batched_predict(net, x_std[test_idx]), y_mean, y_scale)
    row = {"name": name, "model": model, **metric_dict(pred, y[test_idx])}
    state = {"state_dict": net.state_dict(), "x_mean": x_mean, "x_std": x_scale, "y_mean": y_mean, "y_std": y_scale, "history": hist}
    if hasattr(net, "stats"):
        state["kanfis_stats"] = net.stats()
    return row, state


def topk_from_weight(weight, k):
    score = weight.abs().sum(0)
    return torch.topk(score, k=min(k, score.numel())).indices


def corr_matrix(h):
    h = h.float()
    h = (h - h.mean(0)) / h.std(0).clamp_min(1e-6)
    return (h.T @ h) / max(h.shape[0] - 1, 1)


def save_scatter(path, x, y, xlabel, ylabel, title, color=None, max_points=30000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu().float().reshape(-1)
    y = y.detach().cpu().float().reshape(-1)
    c = color.detach().cpu().float().reshape(-1) if color is not None else None
    if x.numel() > max_points:
        idx = torch.linspace(0, x.numel() - 1, max_points).long()
        x = x[idx]
        y = y[idx]
        c = c[idx] if c is not None else None
    plt.figure(figsize=(5, 5))
    if c is None:
        plt.scatter(x.numpy(), y.numpy(), s=2, alpha=0.25)
    else:
        plt.scatter(x.numpy(), y.numpy(), c=c.numpy(), s=2, alpha=0.4, cmap="viridis")
        plt.colorbar(label="h value")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def table_md(rows, fields):
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(f, "")) for f in fields) + "|")
    return "\n".join(lines)


def write_report(name, text):
    ensure_dirs()
    path = V2_REPORT / name
    path.write_text(text)
    print(text[:6000])
    return path
