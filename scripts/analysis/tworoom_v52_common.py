import csv
import math
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v5_common import REPO_ROOT, VISUAL_DATASET, table_md, metric_dict, pearson
from tworoom_v51_common import hard_enrichment, compactness
from tworoom_v4_common import BEST_H_DATASET


V52_OUT = REPO_ROOT / "outputs" / "tworoom_rule_group_v52"
V52_REPORT = REPO_ROOT / "reports" / "tworoom_rule_group_v52"
V52_TABLE = V52_OUT / "tables"

GROUPS = {
    "group_0": [15, 2, 8, 11],
    "group_1": [14, 3, 11, 5],
    "group_2": [12, 9, 7, 5],
    "group_3": [13, 15, 8, 1],
    "group_4": [0, 9, 3, 7],
}


def load_visual():
    vis = torch.load(VISUAL_DATASET, map_location="cpu")
    if "action" not in vis:
        base = torch.load(BEST_H_DATASET, map_location="cpu")
        idx = vis["indices"].long()
        vis["action"] = base["action"][idx].float()
    return vis


def ensure_dirs():
    V52_OUT.mkdir(parents=True, exist_ok=True)
    V52_REPORT.mkdir(parents=True, exist_ok=True)
    V52_TABLE.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = V52_REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def git_text(args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def subset_idx(vis, name):
    te = vis["test_idx"]
    if name == "all_test":
        return te
    return te[vis["sample_flags"][name][te]]


def group_activation(q, rules, mode="mean"):
    x = q[:, rules].float()
    if mode == "max":
        return x.max(1).values
    if mode == "sum":
        return x.sum(1)
    if mode == "topk_avg":
        return torch.topk(x, k=min(2, x.shape[1]), dim=1).values.mean(1)
    return x.mean(1)


def residual_stats(vis):
    mag = vis["residual_t"][vis["test_idx"]].norm(dim=1)
    return {
        "global_mean": mag.mean().item(),
        "global_median": mag.median().item(),
        "global_p80": torch.quantile(mag, 0.8).item(),
        "global_p90": torch.quantile(mag, 0.9).item(),
        "global_p95": torch.quantile(mag, 0.95).item(),
    }


def random_baseline(vis, n=128, top_frac=0.1):
    te = vis["test_idx"]
    pos = vis["position_t"][te]
    hard = vis["hard_labels"][te]
    mag = vis["residual_t"][te].norm(dim=1)
    g = torch.Generator().manual_seed(5202)
    comp = []
    enrich = []
    res = []
    k = max(1, int(top_frac * te.numel()))
    for _ in range(n):
        idx = torch.randperm(te.numel(), generator=g)[:k]
        mask = torch.zeros(te.numel(), dtype=torch.bool)
        mask[idx] = True
        comp.append(pos[mask].std(0).mean().item())
        vals = []
        for j in range(hard.shape[1]):
            vals.append((hard[mask, j].mean().item()) / max(hard[:, j].mean().item(), 1e-8))
        enrich.append(max(vals))
        res.append(mag[mask].mean().item())
    return {
        "random_compactness_mean": float(torch.tensor(comp).mean()),
        "random_compactness_std": float(torch.tensor(comp).std()),
        "random_hard_enrichment_mean": float(torch.tensor(enrich).mean()),
        "random_hard_enrichment_std": float(torch.tensor(enrich).std()),
        "random_residual_mag_mean": float(torch.tensor(res).mean()),
        "random_residual_mag_std": float(torch.tensor(res).std()),
    }


def audit_score(vis, score):
    te = vis["test_idx"]
    pos = vis["position_t"][te]
    residual = vis["residual_t"][te]
    mag = residual.norm(dim=1)
    active10 = score >= torch.quantile(score, 0.9)
    active20 = score >= torch.quantile(score, 0.8)
    hard = hard_enrichment(vis, score)
    top_hard, top_enrich = max(hard.items(), key=lambda kv: kv[1])
    mean_res = mag[active10].mean().item()
    return {
        "usage_mean": score.mean().item(),
        "usage_std": score.std().item(),
        "active_top10_count": int(active10.sum()),
        "active_top20_count": int(active20.sum()),
        "hard_enrichment_action_error_top10": hard.get("action_error_top10", float("nan")),
        "hard_enrichment_high_residual_top10": hard.get("high_residual_top10", float("nan")),
        "hard_enrichment_large_action_small_disp": hard.get("large_action_small_disp", float("nan")),
        "top_hard_subset": top_hard,
        "top_hard_enrichment": top_enrich,
        "mean_residual_mag": mean_res,
        "residual_mag_percentile": (mag <= mean_res).float().mean().item(),
        "spatial_compactness": compactness(pos, score),
    }


def add_bias(x):
    return torch.cat([x.float(), torch.ones(x.shape[0], 1)], 1)


def fit_ridge(x, y, tr, ridge=1e-3):
    xt = add_bias(x[tr])
    yt = y[tr].float()
    eye = torch.eye(xt.shape[1])
    eye[-1, -1] = 0.0
    w = torch.linalg.solve(xt.T @ xt + ridge * eye, xt.T @ yt)
    return w[:-1].T.contiguous(), w[-1].contiguous()


def pred_ridge(x, w, b):
    return x.float() @ w.T + b


def binary_metrics(logits, target):
    prob = torch.sigmoid(logits.float())
    pred = prob >= 0.5
    t = target.float() >= 0.5
    tp = (pred & t).sum(0).float()
    fp = (pred & ~t).sum(0).float()
    fn = (~pred & t).sum(0).float()
    tn = (~pred & ~t).sum(0).float()
    precision = (tp / (tp + fp).clamp_min(1)).mean().item()
    recall = (tp / (tp + fn).clamp_min(1)).mean().item()
    return {
        "hard_acc": ((tp + tn) / (tp + tn + fp + fn).clamp_min(1)).mean().item(),
        "hard_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def eval_input(vis, x, name, subset_names=None):
    tr, te = vis["train_idx"], vis["test_idx"]
    y_res = vis["residual_t"].float()
    y_delta = vis["delta_position"].float()
    y_hard = vis["hard_labels"].float()
    wr, br = fit_ridge(x, y_res, tr)
    wd, bd = fit_ridge(x, y_delta, tr)
    wh, bh = fit_ridge(x, y_hard, tr)
    subset_names = subset_names or ["all_test", "action_error_top10", "high_residual_top10", "large_action_small_disp"]
    rows = []
    for subset in subset_names:
        idx = subset_idx(vis, subset)
        pr = pred_ridge(x[idx], wr, br)
        pd = pred_ridge(x[idx], wd, bd)
        ph = pred_ridge(x[idx], wh, bh)
        row = {"input": name, "subset": subset}
        row.update({f"res_{k}": v for k, v in metric_dict(pr, y_res[idx]).items()})
        row.update({f"delta_{k}": v for k, v in metric_dict(pd, y_delta[idx]).items()})
        row.update(binary_metrics(ph, y_hard[idx]))
        rows.append(row)
    return rows, {"res": (wr, br), "delta": (wd, bd), "hard": (wh, bh)}
