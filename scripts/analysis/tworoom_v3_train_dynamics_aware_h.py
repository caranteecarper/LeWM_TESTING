import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tworoom_v3_common import *


def norm_all(x, tr):
    xs, mean, std = standardize_from_train(x, tr)
    return xs, mean, std


def eval_model(model, norm, data, te):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    z = (data["z_t"] - norm["z_mean"]) / norm["z_std"].clamp_min(1e-6)
    zn = (data["z_next"] - norm["z_mean"]) / norm["z_std"].clamp_min(1e-6)
    action = (data["action"] - norm["action_mean"]) / norm["action_std"].clamp_min(1e-6)
    outs = []
    hns = []
    with torch.no_grad():
        for i in range(0, z.shape[0], 16384):
            out = model(z[i : i + 16384].to(device), action[i : i + 16384].to(device))
            outs.append({k: v.cpu() for k, v in out.items()})
            hns.append(model.encode(zn[i : i + 16384].to(device)).cpu())
    model.cpu()
    merged = {k: torch.cat([o[k] for o in outs], 0) for k in outs[0]}
    merged["h_next_target"] = torch.cat(hns, 0)
    pos = inverse_standardize(merged["pos"], norm["pos_mean"], norm["pos_std"])
    delta = inverse_standardize(merged["delta"], norm["delta_mean"], norm["delta_std"])
    residual = inverse_standardize(merged["residual"], norm["res_mean"], norm["res_std"])
    teacher = inverse_standardize(merged["teacher_residual"], norm["teacher_mean"], norm["teacher_std"])
    row = {
        **metric_dict(pos[te], data["position_t"][te], prefix="pos_"),
        **metric_dict(delta[te], data["delta_position"][te], prefix="delta_"),
        **metric_dict(residual[te], data["residual"][te], prefix="res_"),
        "teacher_res_mse": ((teacher[te] - data["z_action_residual_teacher_pred"][te]) ** 2).mean().item(),
        **binary_metrics(merged["hard_logits"][te], data["hard_labels"][te]),
        **vector_r2(merged["h_next"][te], merged["h_next_target"][te]),
        "h_var_min": merged["h"][te].var(0).min().item(),
        "h_var_mean": merged["h"][te].var(0).mean().item(),
        "h_abs_corr_max_offdiag": (corr_matrix(merged["h"][te]) - torch.eye(model.h_dim)).abs().max().item(),
    }
    return row


def train_one(data, k, variant_name, profile_name):
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    z, z_mean, z_std = norm_all(data["z_t"], tr)
    zn = (data["z_next"] - z_mean) / z_std.clamp_min(1e-6)
    action, action_mean, action_std = norm_all(data["action"], tr)
    pos, pos_mean, pos_std = norm_all(data["position_t"], tr)
    delta, delta_mean, delta_std = norm_all(data["delta_position"], tr)
    res, res_mean, res_std = norm_all(data["residual"], tr)
    teacher, teacher_mean, teacher_std = norm_all(data["z_action_residual_teacher_pred"], tr)
    hard = data["hard_labels"].float()

    norm = {
        "z_mean": z_mean,
        "z_std": z_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "res_mean": res_mean,
        "res_std": res_std,
        "teacher_mean": teacher_mean,
        "teacher_std": teacher_std,
    }
    variant = VARIANTS[variant_name]
    profile = PROFILES[profile_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicsAwareExtractor(h_dim=k, hard_dim=hard.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_idx = tr
    if train_idx.numel() > 160000:
        g = torch.Generator().manual_seed(3072)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:160000]]
    weights = 1.0 + profile["hard_alpha"] * hard[:, 0] + profile["hard_gamma"] * hard[:, 2] + profile["hard_beta"] * hard[:, 4]
    ds = TensorDataset(z[train_idx], zn[train_idx], action[train_idx], pos[train_idx], delta[train_idx], res[train_idx], teacher[train_idx], hard[train_idx], weights[train_idx])
    loader = DataLoader(ds, batch_size=4096, shuffle=True)
    hist = []
    for epoch in range(6):
        model.train()
        total = 0.0
        count = 0
        for zb, znb, ab, pb, db, rb, tb, hb, wb in loader:
            zb, znb, ab, pb, db, rb, tb, hb, wb = [t.to(device) for t in (zb, znb, ab, pb, db, rb, tb, hb, wb)]
            out = model(zb, ab)
            with torch.no_grad():
                hnext_target = model.encode(znb)
            dh_target = hnext_target - out["h"].detach()
            sample_w = wb.unsqueeze(1)
            loss = profile["pos"] * ((out["pos"] - pb).pow(2) * sample_w).mean()
            if variant["delta"]:
                loss = loss + profile["delta"] * ((out["delta"] - db).pow(2) * sample_w).mean()
            if variant["res"]:
                loss = loss + profile["res"] * ((out["residual"] - rb).pow(2) * sample_w).mean()
            if variant["teacher"]:
                loss = loss + profile["teacher"] * ((out["teacher_residual"] - tb).pow(2) * sample_w).mean()
            if variant["hard"]:
                loss = loss + profile["hard"] * F.binary_cross_entropy_with_logits(out["hard_logits"], hb, weight=wb.unsqueeze(1).expand_as(hb))
            if variant["hnext"]:
                loss = loss + profile["hnext"] * F.mse_loss(out["h_next"], hnext_target.detach())
                loss = loss + profile["dh"] * F.mse_loss(out["delta_h"], dh_target.detach())
            h = out["h"]
            h_std = h.std(0)
            c = corr_matrix(h.detach().cpu()).to(device)
            loss = loss + profile["var"] * (1.0 - h_std.clamp_max(1.0)).relu().mean()
            loss = loss + profile["decor"] * (c - torch.eye(k, device=device)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * zb.shape[0]
            count += zb.shape[0]
        hist.append({"epoch": epoch, "loss": total / max(count, 1)})

    row = eval_model(model, norm, data, te)
    row.update({"K": k, "variant": variant_name, "profile": profile_name, "train_loss": hist[-1]["loss"]})
    state = {"state_dict": model.cpu().state_dict(), "h_dim": k, "hard_dim": hard.shape[1], "norm": norm, "history": hist, "metrics": row}
    return row, state


def main():
    ensure_dirs()
    DYN_DIR.mkdir(parents=True, exist_ok=True)
    data = load_teacher()
    rows = []
    best = None
    best_score = float("inf")
    states = {}
    for k in [8, 16]:
        for variant in VARIANTS:
            for profile in PROFILES:
                row, state = train_one(data, k, variant, profile)
                key = f"K{k}_{variant}_{profile}"
                rows.append(row)
                states[key] = state
                # Favor hard/residual usefulness while preserving position and h variance.
                score = row["res_overall_mse"] + 0.2 * row["delta_overall_mse"] + max(0.0, 0.95 - row["pos_y_r2"]) * 100.0 + max(0.0, 0.05 - row["h_var_min"]) * 50.0
                if score < best_score:
                    best_score = score
                    best = key
                torch.save(state, DYN_DIR / f"{key}.pt")
    torch.save({"best_key": best, "rows": rows, "states_index": list(states)}, DYN_DIR / "summary.pt")

    fields = [
        "K",
        "variant",
        "profile",
        "pos_x_r2",
        "pos_y_r2",
        "delta_x_r2",
        "delta_y_r2",
        "res_x_r2",
        "res_y_r2",
        "teacher_res_mse",
        "hard_auc",
        "hard_acc",
        "mean_r2",
        "h_var_min",
        "h_abs_corr_max_offdiag",
    ]
    text = f"""# Dynamics-Aware H Training

Trained K=8 and K=16 across C1-C4 variants and loss profiles. Action is only provided after `h = F(z)`.

Recommended candidate by diagnostic score: `{best}`.

{table_md(rows, fields)}

## Notes

- `mean_r2` is h_next mean R2.
- All saved model states are under `{DYN_DIR}` and are not committed to git.
- No LeWM module is modified or called for training.
"""
    write_report("02_dynamics_aware_h_training.md", text)


if __name__ == "__main__":
    main()
