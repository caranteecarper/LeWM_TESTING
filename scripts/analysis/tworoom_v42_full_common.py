import csv
import shutil
import subprocess
from pathlib import Path

import torch

from tworoom_v4_common import table_md, metric_dict
from tworoom_v42_common import (
    REPO_ROOT,
    V42_MODEL_DIR,
    V42_REPORT,
    V42_OUT,
    compute_v42_q,
    compute_v4_q,
    compute_v41_q,
    load_base_data,
    v42_model_path,
)
from tworoom_v52_common import (
    GROUPS as V4_GROUPS,
    audit_score,
    binary_metrics,
    csv_write,
    eval_input,
    fit_ridge,
    group_activation,
    load_visual,
    pred_ridge,
    random_baseline,
    residual_stats,
    subset_idx,
)


FULL_OUT = REPO_ROOT / "outputs" / "tworoom_v42_full_validation"
FULL_REPORT = REPO_ROOT / "reports" / "tworoom_v42_full_validation"
FULL_TABLE = FULL_OUT / "tables"
FULL_FIG = FULL_OUT / "figures"

V42_GROUPS = {
    "group_0": [10, 13, 12, 1],
    "best_rules": [10, 13, 12, 6],
    "top3_rules": [10, 13, 12],
}


def ensure_dirs():
    FULL_OUT.mkdir(parents=True, exist_ok=True)
    FULL_REPORT.mkdir(parents=True, exist_ok=True)
    FULL_TABLE.mkdir(parents=True, exist_ok=True)
    FULL_FIG.mkdir(parents=True, exist_ok=True)


def write_report(name, text):
    ensure_dirs()
    path = FULL_REPORT / name
    path.write_text(text)
    print(text[:7000])
    return path


def git_text(args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def maybe_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def best_v42_key():
    summary = torch.load(V42_MODEL_DIR / "summary.pt", map_location="cpu")
    return summary["best_key"]


def build_visual_data():
    vis = load_visual()
    if "action_t" not in vis:
        vis["action_t"] = vis["action"].float()
    if "residual" not in vis:
        vis["residual"] = vis["residual_t"]
    if "q_v42_t" not in vis:
        base = load_base_data()
        key, q_all = compute_v42_q(base)
        vis["q_v42_t"] = q_all[vis["indices"].long()].float()
        vis["v42_key"] = key
    if "q_v4_t" not in vis or "q_v41_t" not in vis:
        base = load_base_data()
        if "q_v4_t" not in vis:
            _, q_v4 = compute_v4_q(base)
            vis["q_v4_t"] = q_v4[vis["indices"].long()].float()
        if "q_v41_t" not in vis:
            _, q_v41 = compute_v41_q(base)
            vis["q_v41_t"] = q_v41[vis["indices"].long()].float()
    return vis


def q_test(vis, name="q_v42_t"):
    return vis[name][vis["test_idx"]].float()


def add_audit_derived(row, rs, rb):
    row["residual_mag_ratio_to_global_mean"] = row["mean_residual_mag"] / max(rs["global_mean"], 1e-8)
    row["residual_mag_zscore"] = (row["mean_residual_mag"] - rb["random_residual_mag_mean"]) / max(
        rb["random_residual_mag_std"], 1e-8
    )
    row["compactness_ratio_to_random"] = row["spatial_compactness"] / max(rb["random_compactness_mean"], 1e-8)
    row["hard_enrichment_minus_random"] = row["top_hard_enrichment"] - rb["random_hard_enrichment_mean"]
    return row


def select_rule_rows(rows):
    selected = []
    for row in rows:
        strong = row["top_hard_enrichment"] > 2.0 or row["residual_mag_ratio_to_global_mean"] > 2.0
        compact = row["compactness_ratio_to_random"] < 0.8
        row["selected_for_reporting"] = strong and compact
        if row["selected_for_reporting"]:
            row["selection_reason"] = "hard/residual candidate"
            selected.append(row)
        elif row["top_hard_enrichment"] > 1.5:
            row["selection_reason"] = "medium hard candidate"
        else:
            row["selection_reason"] = "ordinary state partition"
    return selected


def group_rows_from_rules(vis, group_defs, q_name="q_v42_t"):
    rb = random_baseline(vis)
    rs = residual_stats(vis)
    q = q_test(vis, q_name)
    rows = []
    for gid, rules in group_defs.items():
        rules = [int(r) for r in rules]
        for mode in ["mean", "sum", "max", "topk_avg"]:
            score = group_activation(q, rules, mode)
            row = {
                "group": gid,
                "rules": ",".join(map(str, rules)),
                "activation_definition": mode,
                **audit_score(vis, score),
            }
            add_audit_derived(row, rs, rb)
            row["group_clarity_score"] = (
                row["top_hard_enrichment"]
                * row["residual_mag_ratio_to_global_mean"]
                / max(row["compactness_ratio_to_random"], 1e-6)
            )
            row["physical_interpretation_label"] = (
                "hard-motion correction group" if row["top_hard_enrichment"] > 2 else "state/correction group"
            )
            rows.append(row)
    return rows


def discover_v42_groups(vis, audit_rows):
    q = q_test(vis)
    selected = [int(r["rule"]) for r in sorted(audit_rows, key=lambda r: r["top_hard_enrichment"], reverse=True)[:6]]
    out = dict(V42_GROUPS)
    out["top_hard_rules"] = selected[:4]
    selected_res = [
        int(r["rule"]) for r in sorted(audit_rows, key=lambda r: r["mean_residual_mag"], reverse=True)[:4]
    ]
    out["top_residual_rules"] = selected_res
    corr = torch.corrcoef(q.T).nan_to_num()
    for seed in selected[:3]:
        partners = torch.argsort(corr[seed], descending=True).tolist()
        group = []
        for p in partners:
            if p not in group:
                group.append(int(p))
            if len(group) == 4:
                break
        out[f"coactivation_seed_{seed}"] = group
    return out


def retention(action_mse, full_mse, row_mse):
    return (action_mse - row_mse) / max(action_mse - full_mse, 1e-8)


def apply_mask(q, rules):
    x = q.clone()
    x[:, [int(r) for r in rules]] = 0.0
    return x


def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False
