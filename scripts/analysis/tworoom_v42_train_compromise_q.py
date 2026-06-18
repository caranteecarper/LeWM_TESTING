import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v42_common import *


def train_one(name, cfg, data):
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    std_data, norm = standardize_v41(data, tr)
    hard = data["hard_labels"].float()
    model = SharpQModel(std_data["h_t"].shape[1], std_data["action"].shape[1], cfg["rules"], hard.shape[1], top_k=cfg["top_k"])
    model.rule_layer.initialize_from_data(std_data["h_t"][tr[:8192]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_idx = tr
    if train_idx.numel() > 220000:
        g = torch.Generator().manual_seed(4202)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:220000]]
    weights = hard_sample_weight(data, cfg["hard"])
    ds = TensorDataset(std_data["h_t"][train_idx], std_data["action"][train_idx], std_data["position_t"][train_idx], std_data["delta_position"][train_idx], std_data["residual"][train_idx], hard[train_idx], weights[train_idx])
    loader = DataLoader(ds, batch_size=4096, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 8
    hist = []
    for epoch in range(epochs):
        tau = tau_for_epoch(epoch, epochs, cfg["tau"])
        total = 0.0
        count = 0
        model.train()
        for hb, ab, pos, delta, residual, yhard, wb in loader:
            hb, ab, pos, delta, residual, yhard, wb = [x.to(device) for x in (hb, ab, pos, delta, residual, yhard, wb)]
            out = model(hb, ab, tau=tau, hard_topk=cfg["top_k"] is not None)
            w = wb.unsqueeze(1)
            loss = 0.8 * ((out["pos"] - pos).pow(2) * w).mean()
            loss = loss + 0.9 * ((out["delta"] - delta).pow(2) * w).mean()
            loss = loss + 1.1 * ((out["residual"] - residual).pow(2) * w).mean()
            loss = loss + 0.45 * F.binary_cross_entropy_with_logits(out["hard_logits"], yhard, weight=w.expand_as(yhard))
            loss = loss + model.rule_layer.regularizers(out["q"], cfg["entropy"], 0.06, 0.02, 0.001)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * hb.shape[0]
            count += hb.shape[0]
        hist.append({"epoch": epoch, "tau": tau, "train_loss": total / max(count, 1)})
    row, _ = eval_model_heads(model, data, std_data, norm, te, tau=cfg["tau"], hard_topk=cfg["top_k"] is not None)
    for subset in ["action_error_top10", "large_action_small_disp", "high_residual_top10"]:
        idx = te[data["hard_labels"][te, data["hard_label_names"].index(subset)] > 0.5]
        sr, _ = eval_model_heads(model, data, std_data, norm, idx, tau=cfg["tau"], hard_topk=cfg["top_k"] is not None)
        row[f"{subset}_res_mse"] = sr["res_overall_mse"]
    row.update({"variant": name, "num_rules": cfg["rules"], "top_k": cfg["top_k"], "tau_end": cfg["tau"], "hard_weight": cfg["hard"]})
    meta = {"variant": name, "num_rules": cfg["rules"], "top_k": cfg["top_k"], "tau_end": cfg["tau"], "hard_weight": cfg["hard"], "h_dim": std_data["h_t"].shape[1], "action_dim": std_data["action"].shape[1], "hard_dim": hard.shape[1], "history": hist, "metrics": row, "no_q_to_h": True}
    save_sharp_model(V42_MODEL_DIR / name, model, norm, meta)
    return row


def main():
    ensure_dirs()
    data = load_base_data()
    rows = []
    best = None
    best_score = float("inf")
    for name, cfg in V42_VARIANTS.items():
        row = train_one(name, cfg, data)
        rows.append(row)
        s = score_v42(row)
        if s < best_score:
            best_score = s
            best = name
    torch.save({"best_key": best, "rows": rows}, V42_MODEL_DIR / "summary.pt")
    csv_write(V42_TABLE / "v42_training_metrics.csv", rows)
    fields = ["variant", "num_rules", "top_k", "tau_end", "hard_weight", "res_overall_mse", "action_error_top10_res_mse", "large_action_small_disp_res_mse", "delta_overall_mse", "pos_x_r2", "pos_y_r2", "hard_auc", "hard_f1", "active_rule_count", "q_entropy", "usage_min", "usage_max", "usage_entropy"]
    text = f"""# V4.2 Compromise q Training

Best V4.2 model: `{best}`.

No q->h reconstruction, h mimicry, hidden distillation, image reconstruction, or LeWM modification is used.

{table_md(rows, fields)}
"""
    write_report("01_train_v42.md", text)


if __name__ == "__main__":
    main()
