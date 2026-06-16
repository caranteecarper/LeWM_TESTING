import torch
from torch import nn
import torch.nn.functional as F

from tworoom_v2_common import *


class AuxExtractor(nn.Module):
    def __init__(self, z_dim=192, h_dim=16, action_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(z_dim, 384), nn.GELU(), nn.Linear(384, h_dim))
        self.pos_head = nn.Sequential(nn.LayerNorm(h_dim), nn.Linear(h_dim, 128), nn.GELU(), nn.Linear(128, 2))
        self.delta_head = nn.Sequential(nn.LayerNorm(h_dim + action_dim), nn.Linear(h_dim + action_dim, 128), nn.GELU(), nn.Linear(128, 2))
        self.hnext_head = nn.Sequential(nn.LayerNorm(h_dim + action_dim), nn.Linear(h_dim + action_dim, 128), nn.GELU(), nn.Linear(128, h_dim))
        self.residual_head = nn.Sequential(nn.LayerNorm(h_dim + action_dim), nn.Linear(h_dim + action_dim, 128), nn.GELU(), nn.Linear(128, 2))

    def encode(self, z):
        return self.encoder(z)

    def forward(self, z, action=None):
        h = self.encode(z)
        pos = self.pos_head(h)
        if action is None:
            return pos, h
        ha = torch.cat([h, action], 1)
        return pos, h, self.delta_head(ha), self.hnext_head(ha), self.residual_head(ha)


def train_pure(kind, k, z_std, pos_std, train_idx, val_idx, test_idx, pos_mean, pos_scale):
    if kind == "linear":
        model = LinearBottleneckRegressor(192, out_dim=2, bottleneck_dim=k)
        hidden = None
    else:
        model = MLPRegressor(192, hidden=384, out_dim=2, bottleneck_dim=k)
        hidden = 384
    model, _, hist = train_model(model, z_std, pos_std, train_idx, val_idx, epochs=10, max_train=250000)
    pred_std, h = batched_predict(model, z_std[test_idx], return_h=True)
    pred = inverse_standardize(pred_std, pos_mean, pos_scale)
    met = {"variant": f"pure_{kind}_K{k}", **metric_dict(pred, inverse_standardize(pos_std[test_idx], pos_mean, pos_scale))}
    h_all = batched_predict(model, z_std, return_h=True)[1]
    met.update({"h_var_min": h.var(0).min().item(), "h_var_mean": h.var(0).mean().item(), "h_abs_corr_max_offdiag": (corr_matrix(h) - torch.eye(k)).abs().max().item()})
    return model, h_all, met, hist


def train_aux(variant, z, pos, pairs, split, residual, h_dim=16):
    pair_train, pair_val, pair_test = indices_from_split(pairs["episode_id"], split)
    z_std, z_mean, z_scale = standardize_from_train(pairs["z_t"].float(), pair_train)
    zn_std = (pairs["z_next"].float() - z_mean) / z_scale.clamp_min(1e-6)
    pos_std, pos_mean, pos_scale = standardize_from_train(pairs["position_t"].float(), pair_train)
    delta_std, delta_mean, delta_scale = standardize_from_train(pairs["delta_position"].float(), pair_train)
    res_std, res_mean, res_scale = standardize_from_train(residual, pair_train)
    action_std, action_mean, action_scale = standardize_from_train(pairs["action_t"].float(), pair_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuxExtractor(h_dim=h_dim, action_dim=action_std.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_idx = pair_train
    if train_idx.numel() > 220000:
        g = torch.Generator().manual_seed(3072)
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:220000]]
    loader = DataLoader(TensorDataset(z_std[train_idx], zn_std[train_idx], action_std[train_idx], pos_std[train_idx], delta_std[train_idx], res_std[train_idx]), batch_size=4096, shuffle=True)
    history = []
    for epoch in range(10):
        model.train()
        total = 0.0
        count = 0
        for zb, znb, ab, pb, db, rb in loader:
            zb, znb, ab, pb, db, rb = zb.to(device), znb.to(device), ab.to(device), pb.to(device), db.to(device), rb.to(device)
            pos_hat, h, delta_hat, hnext_hat, res_hat = model(zb, ab)
            with torch.no_grad():
                hnext = model.encode(znb)
            loss = F.mse_loss(pos_hat, pb)
            if "delta" in variant:
                loss = loss + F.mse_loss(delta_hat, db)
            if "hnext" in variant:
                loss = loss + 0.5 * F.mse_loss(hnext_hat, hnext.detach())
            if "residual" in variant:
                loss = loss + F.mse_loss(res_hat, rb)
            h_std = h.std(0)
            c = corr_matrix(h.detach().cpu())
            loss = loss + 0.01 * (1.0 - h_std.clamp_max(1.0)).relu().mean() + 0.001 * (c - torch.eye(h_dim)).pow(2).mean().to(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * zb.shape[0]
            count += zb.shape[0]
        history.append({"epoch": epoch, "loss": total / max(count, 1)})

    model.eval()
    with torch.no_grad():
        pos_hat, h, delta_hat, hnext_hat, res_hat = model(z_std[pair_test].to(device), action_std[pair_test].to(device))
        pred_pos = inverse_standardize(pos_hat.cpu(), pos_mean, pos_scale)
        pred_delta = inverse_standardize(delta_hat.cpu(), delta_mean, delta_scale)
        pred_res = inverse_standardize(res_hat.cpu(), res_mean, res_scale)
        h_cpu = h.cpu()
    met = {
        "variant": variant,
        **metric_dict(pred_pos, pairs["position_t"].float()[pair_test], prefix="pos_"),
        **metric_dict(pred_delta, pairs["delta_position"].float()[pair_test], prefix="delta_"),
        **metric_dict(pred_res, residual[pair_test], prefix="res_"),
        "h_var_min": h_cpu.var(0).min().item(),
        "h_var_mean": h_cpu.var(0).mean().item(),
        "h_abs_corr_max_offdiag": (corr_matrix(h_cpu) - torch.eye(h_dim)).abs().max().item(),
    }
    model.cpu()
    return model, met, history, {"z_mean": z_mean, "z_std": z_scale, "action_mean": action_mean, "action_std": action_scale}


def main():
    ensure_dirs()
    out = V2_OUT / "factor_extractors"
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = load_latents()
    split = save_or_load_split(data["episode_id"])
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
    z = data["z"].float()
    pos = data["position"].float()
    z_std, z_mean, z_scale = standardize_from_train(z, train_idx)
    pos_std, pos_mean, pos_scale = standardize_from_train(pos, train_idx)

    pure_rows = []
    pure_states = {}
    for k in [4, 8, 16]:
        for kind in ["linear", "mlp"]:
            model, h_all, met, hist = train_pure(kind, k, z_std, pos_std, train_idx, val_idx, test_idx, pos_mean, pos_scale)
            pure_rows.append(met)
            pure_states[met["variant"]] = {"state_dict": model.state_dict(), "h_all": h_all, "history": hist, "z_mean": z_mean, "z_std": z_scale}
            if k == 16 and kind == "mlp":
                for i in range(16):
                    save_scatter(fig / f"pure_mlp_K16_h{i}_spatial.png", pos[test_idx, 0], pos[test_idx, 1], "x", "y", f"h{i}", color=h_all[test_idx, i])
    torch.save(pure_states, out / "pure_extractors.pt")

    pairs = load_pairs()
    p_train, p_val, p_test = indices_from_split(pairs["episode_id"], split)
    row_action, st_action = train_eval_regressor("action_only_delta", "linear", pairs["action_t"].float(), pairs["delta_position"].float(), p_train, p_val, p_test)
    action_std = (pairs["action_t"].float() - st_action["x_mean"]) / st_action["x_std"].clamp_min(1e-6)
    action_pred = inverse_standardize(predict_linear(action_std, st_action["W"], st_action["b"]), st_action["y_mean"], st_action["y_std"])
    residual = pairs["delta_position"].float() - action_pred

    aux_rows = []
    aux_states = {"action_only": st_action}
    for variant in ["aux_no_aux", "aux_delta", "aux_hnext", "aux_residual", "aux_delta_hnext_residual"]:
        model, met, hist, norm = train_aux(variant, z, pos, pairs, split, residual, h_dim=16)
        aux_rows.append(met)
        aux_states[variant] = {"state_dict": model.state_dict(), "history": hist, "norm": norm}
    torch.save(aux_states, out / "aux_extractors.pt")

    pure_fields = ["variant", "x_mse", "y_mse", "x_r2", "y_r2", "x_pearson", "y_pearson", "overall_mse", "h_var_min", "h_var_mean", "h_abs_corr_max_offdiag"]
    aux_fields = ["variant", "pos_x_r2", "pos_y_r2", "delta_x_r2", "delta_y_r2", "res_x_r2", "res_y_r2", "h_var_min", "h_var_mean", "h_abs_corr_max_offdiag"]
    text = f"""# Factor Extractor Without Action

Constraint: `h_t = F(z_t)`. Action is not an input to the extractor. Action is only used after h exists in auxiliary heads.

## Pure Extractor

{table_md(pure_rows, pure_fields)}

## Auxiliary Constrained Extractor

{table_md(aux_rows, aux_fields)}

## Answers

- K=16 pure MLP is the main candidate if it reconstructs position without h collapse.
- Auxiliary variants are only diagnostics: they test whether h can support delta/residual/h-next prediction after action is appended.
- If auxiliary improves delta/residual without collapsing h variance, it is a better candidate than pure only; otherwise prefer pure K16.
- None of these models are LeWM predictor integrations.
"""
    write_report("04_factor_extractor_no_action.md", text)


if __name__ == "__main__":
    main()
