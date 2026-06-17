import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v4_common import *


def normalize_targets(data, tr):
    keys = ["h_t", "action", "position_t", "delta_position", "residual", "h_next", "delta_h"]
    norm = {}
    std_data = {}
    for key in keys:
        xs, mean, std = standardize_from_train(data[key].float(), tr)
        std_data[key] = xs
        norm[f"{key}_mean"] = mean
        norm[f"{key}_std"] = std
    return std_data, norm


def eval_rule_model(model, std_data, data, norm, idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, idx.numel(), 16384):
            j = idx[i : i + 16384]
            out = model(std_data["h_t"][j].to(device), std_data["action"][j].to(device))
            outs.append({k: v.cpu() for k, v in out.items()})
    model.cpu()
    merged = {k: torch.cat([o[k] for o in outs], 0) for k in outs[0]}

    def inv(key, pred):
        return inverse_standardize(pred, norm[f"{key}_mean"], norm[f"{key}_std"])

    pos = inv("position_t", merged["pos"])
    delta = inv("delta_position", merged["delta"])
    residual = inv("residual", merged["residual"])
    hnext = inv("h_next", merged["h_next"])
    dh = inv("delta_h", merged["delta_h"])
    q_stats = model.rule_layer.stats(merged["q"])
    return {
        **metric_dict(pos, data["position_t"][idx], prefix="pos_"),
        **metric_dict(delta, data["delta_position"][idx], prefix="delta_"),
        **metric_dict(residual, data["residual"][idx], prefix="res_"),
        **binary_metrics(merged["hard_logits"], data["hard_labels"][idx]),
        "hnext_mse": vector_r2(hnext, data["h_next"][idx])["mse"],
        "hnext_mean_r2": vector_r2(hnext, data["h_next"][idx])["mean_r2"],
        "delta_h_mse": vector_r2(dh, data["delta_h"][idx])["mse"],
        "delta_h_mean_r2": vector_r2(dh, data["delta_h"][idx])["mean_r2"],
        **q_stats,
    }


def regularization(model, q):
    ent = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum(1).mean()
    usage = q.mean(0)
    usage_target = torch.full_like(usage, 1.0 / usage.numel())
    usage_loss = F.mse_loss(usage, usage_target)
    centers = F.normalize(model.rule_layer.centers, dim=1)
    gram = centers @ centers.T
    offdiag = gram - torch.eye(gram.shape[0], device=gram.device)
    diversity_loss = offdiag.pow(2).mean()
    q_var_loss = (0.005 - q.var(0)).relu().mean()
    gate = torch.sigmoid(model.rule_layer.input_gate_logits)
    gate_sparse = gate.mean()
    return 0.005 * ent + 0.05 * usage_loss + 0.01 * diversity_loss + 0.05 * q_var_loss + 0.001 * gate_sparse


def train_one(data, num_rules, variant_name):
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    std_data, norm = normalize_targets(data, tr)
    hard = data["hard_labels"].float()
    variant = V4_VARIANTS[variant_name]
    model = RuleFeatureModel(
        h_dim=std_data["h_t"].shape[1],
        action_dim=std_data["action"].shape[1],
        num_rules=num_rules,
        hard_dim=hard.shape[1],
    )
    model.rule_layer.initialize_from_data(std_data["h_t"][tr[: min(tr.numel(), 8192)]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_idx = tr
    if train_idx.numel() > 180000:
        g = torch.Generator().manual_seed(3072)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:180000]]
    weights = 1.0 + 0.5 * hard[:, 0] + 0.5 * hard[:, 2] + 0.25 * hard[:, 4]
    ds = TensorDataset(
        std_data["h_t"][train_idx],
        std_data["action"][train_idx],
        std_data["position_t"][train_idx],
        std_data["delta_position"][train_idx],
        std_data["residual"][train_idx],
        hard[train_idx],
        std_data["h_next"][train_idx],
        std_data["delta_h"][train_idx],
        weights[train_idx],
    )
    loader = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=0)
    hist = []
    for epoch in range(8):
        model.train()
        total = 0.0
        count = 0
        for hb, ab, pb, db, rb, yhard, hnb, dhb, wb in loader:
            hb, ab, pb, db, rb, yhard, hnb, dhb, wb = [x.to(device) for x in (hb, ab, pb, db, rb, yhard, hnb, dhb, wb)]
            out = model(hb, ab)
            w = wb.unsqueeze(1)
            loss = ((out["pos"] - pb).pow(2) * w).mean()
            if variant["delta"]:
                loss = loss + 0.5 * ((out["delta"] - db).pow(2) * w).mean()
            if variant["res"]:
                loss = loss + 0.7 * ((out["residual"] - rb).pow(2) * w).mean()
            if variant["hard"]:
                loss = loss + 0.3 * F.binary_cross_entropy_with_logits(out["hard_logits"], yhard, weight=w.expand_as(yhard))
            if variant["hnext"]:
                loss = loss + 0.25 * F.mse_loss(out["h_next"], hnb)
                loss = loss + 0.25 * F.mse_loss(out["delta_h"], dhb)
            loss = loss + regularization(model, out["q"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * hb.shape[0]
            count += hb.shape[0]
        hist.append({"epoch": epoch, "loss": total / max(count, 1)})

    row = eval_rule_model(model, std_data, data, norm, te)
    row.update({"variant": variant_name, "num_rules": num_rules, "train_loss": hist[-1]["loss"]})
    meta = {
        "variant": variant_name,
        "num_rules": num_rules,
        "h_dim": std_data["h_t"].shape[1],
        "action_dim": std_data["action"].shape[1],
        "hard_dim": hard.shape[1],
        "history": hist,
        "metrics": row,
        "q_definition": "softmax-normalized KANFIS rule firing strengths from h_t only",
    }
    save_model_state(RULE_DIR / f"{variant_name}_R{num_rules}.pt", model, norm, meta)
    return row, meta


def score(row):
    return (
        row["res_overall_mse"]
        + 0.35 * row["delta_overall_mse"]
        + 0.2 * row["hnext_mse"]
        - 0.05 * row["hard_auc"]
        + max(0.0, 0.90 - row["pos_x_r2"]) * 2.0
        + max(0.0, 0.90 - row["pos_y_r2"]) * 2.0
        + max(0.0, 0.002 - row["q_var_mean"]) * 100.0
    )


def main():
    ensure_dirs()
    data = load_best_h_dataset()
    rows = []
    best_key = None
    best_score = float("inf")
    for rules in [16, 32]:
        for variant in V4_VARIANTS:
            row, _ = train_one(data, rules, variant)
            rows.append(row)
            s = score(row)
            if s < best_score:
                best_score = s
                best_key = f"{variant}_R{rules}"
    torch.save({"best_key": best_key, "rows": rows}, RULE_DIR / "summary.pt")

    fields = [
        "variant",
        "num_rules",
        "pos_x_r2",
        "pos_y_r2",
        "delta_x_r2",
        "delta_y_r2",
        "res_x_r2",
        "res_y_r2",
        "hard_auc",
        "hard_f1",
        "hnext_mean_r2",
        "delta_h_mean_r2",
        "q_var_mean",
        "q_entropy_mean",
        "active_rule_count_mean",
        "rule_usage_min",
        "rule_usage_max",
    ]
    text = f"""# V4 KANFIS Rule-Feature Training

`q_t` is defined as the softmax-normalized firing strength of learnable KANFIS-style rule prototypes over `h_t`.
Action is not an input to the rule-feature layer; action only enters diagnostic heads after q is produced.

Recommended best q model: `{best_key}`.

{table_md(rows, fields)}

## Collapse Check

- `q_var_mean`, `q_entropy_mean`, active rule count, and rule usage range are reported for every variant.
- A low `q_var_mean` with near-uniform usage would indicate q collapse.
- Models are saved under `{RULE_DIR}` and are not committed to git.
"""
    write_report("02_train_rule_features.md", text)


if __name__ == "__main__":
    main()
