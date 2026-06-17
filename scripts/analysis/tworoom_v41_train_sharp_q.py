import torch
import torch.nn.functional as F

from tworoom_v41_common import *


def tau_for_epoch(epoch, epochs, tau_end):
    if tau_end >= 1.0:
        return 1.0
    t = epoch / max(epochs - 1, 1)
    return tau_end + 0.5 * (1.0 - tau_end) * (1 + math.cos(math.pi * t))


def train_one(name, cfg, data):
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    std_data, norm = standardize_v41(data, tr)
    hard = data["hard_labels"].float()
    model = SharpQModel(
        h_dim=std_data["h_t"].shape[1],
        action_dim=std_data["action"].shape[1],
        num_rules=cfg["rules"],
        hard_dim=hard.shape[1],
        top_k=cfg["top_k"],
    )
    model.rule_layer.initialize_from_data(std_data["h_t"][tr[: min(tr.numel(), 8192)]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_idx = tr
    if train_idx.numel() > 220000:
        g = torch.Generator().manual_seed(4101)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:220000]]
    weights = hard_sample_weight(data, cfg["hard"])
    ds = TensorDataset(
        std_data["h_t"][train_idx],
        std_data["action"][train_idx],
        std_data["position_t"][train_idx],
        std_data["delta_position"][train_idx],
        std_data["residual"][train_idx],
        hard[train_idx],
        weights[train_idx],
    )
    loader = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    hist = []
    epochs = 10
    for epoch in range(epochs):
        tau = tau_for_epoch(epoch, epochs, cfg["tau_end"])
        hard_topk = cfg["top_k"] is not None
        total = 0.0
        count = 0
        model.train()
        for hb, ab, pos, delta, residual, yhard, wb in loader:
            hb, ab, pos, delta, residual, yhard, wb = [x.to(device) for x in (hb, ab, pos, delta, residual, yhard, wb)]
            out = model(hb, ab, tau=tau, hard_topk=hard_topk)
            w = wb.unsqueeze(1)
            loss = 0.8 * ((out["pos"] - pos).pow(2) * w).mean()
            loss = loss + 0.8 * ((out["delta"] - delta).pow(2) * w).mean()
            loss = loss + 1.2 * ((out["residual"] - residual).pow(2) * w).mean()
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(out["hard_logits"], yhard, weight=w.expand_as(yhard))
            loss = loss + model.rule_layer.regularizers(
                out["q"],
                w_entropy=cfg["entropy"],
                w_balance=0.08,
                w_diversity=0.03,
                w_gate=0.002,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * hb.shape[0]
            count += hb.shape[0]
        hist.append({"epoch": epoch, "tau": tau, "train_loss": total / max(count, 1)})

    row, _ = eval_model_heads(model, data, std_data, norm, te, tau=cfg["tau_end"], hard_topk=cfg["top_k"] is not None)
    # Required hard subset residual metrics.
    for subset in ["action_error_top10", "large_action_small_disp", "high_residual_top10"]:
        idx = subset_index(data, subset)
        sr, _ = eval_model_heads(model, data, std_data, norm, idx, tau=cfg["tau_end"], hard_topk=cfg["top_k"] is not None)
        row[f"{subset}_res_mse"] = sr["res_overall_mse"]
    row.update({"variant": name, "num_rules": cfg["rules"], "top_k": cfg["top_k"], "tau_end": cfg["tau_end"], "hard_weight": cfg["hard"]})
    meta = {
        "variant": name,
        "num_rules": cfg["rules"],
        "top_k": cfg["top_k"],
        "tau_end": cfg["tau_end"],
        "hard_weight": cfg["hard"],
        "h_dim": std_data["h_t"].shape[1],
        "action_dim": std_data["action"].shape[1],
        "hard_dim": hard.shape[1],
        "history": hist,
        "metrics": row,
        "no_q_to_h": True,
        "q_definition": "softmax/top-k KANFIS rule firing strengths from h_t only",
    }
    save_sharp_model(name, model, norm, meta)
    return row


def score(row):
    active_penalty = abs(row["active_rule_count"] - 3.0) * 0.02
    return (
        row["res_overall_mse"]
        + 0.35 * row["action_error_top10_res_mse"]
        + 0.2 * row["large_action_small_disp_res_mse"]
        + active_penalty
        + max(0.0, 0.95 - row["pos_x_r2"])
        + max(0.0, 0.95 - row["pos_y_r2"])
        - 0.02 * row["hard_auc"]
    )


def main():
    ensure_dirs()
    data = load_data()
    rows = []
    best_key = None
    best_score = float("inf")
    for name, cfg in V41_VARIANTS.items():
        row = train_one(name, cfg, data)
        rows.append(row)
        s = score(row)
        if s < best_score:
            best_score = s
            best_key = name
    torch.save({"best_key": best_key, "rows": rows}, V41_RULE_DIR / "summary.pt")
    fields = [
        "variant",
        "num_rules",
        "top_k",
        "tau_end",
        "hard_weight",
        "res_overall_mse",
        "action_error_top10_res_mse",
        "large_action_small_disp_res_mse",
        "delta_overall_mse",
        "pos_x_r2",
        "pos_y_r2",
        "hard_auc",
        "hard_f1",
        "q_entropy",
        "active_rule_count",
        "usage_min",
        "usage_max",
        "usage_entropy",
        "center_min_distance",
    ]
    text = f"""# V4.1 Sharp q Training

Best V4.1 q model: `{best_key}`.

No q->h reconstruction, h mimicry, or MLP hidden distillation is used. h_next / delta_h are not used as training losses.

{table_md(rows, fields)}

## Notes

- `tau` is annealed from 1.0 to the listed `tau_end`.
- `top_k` variants keep only the top-k firing strengths and renormalize q.
- Entropy lowers per-sample active rule count; balance regularization keeps dataset-level rule usage from collapsing.
- Model files are saved under `{V41_RULE_DIR}` and are not committed to git.
"""
    write_report("02_train_sharp_q.md", text)


if __name__ == "__main__":
    main()
