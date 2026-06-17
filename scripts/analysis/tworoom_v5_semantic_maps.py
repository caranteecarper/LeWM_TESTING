import torch
import torch.nn.functional as F

from tworoom_v5_common import *


def train_hard_linear(x, hard, tr, te):
    xs, xm, xstd = standardize_from_train(x, tr)
    model = nn.Linear(xs.shape[1], hard.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(xs[tr].float(), hard[tr].float()), batch_size=2048, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(5):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    logits = batched_predict(model, xs[te], batch_size=4096)
    return logits


def main():
    ensure_dirs()
    fig_dir = V5_FIG / "semantic_maps"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vis = load_visual()
    tr, te = vis["train_idx"], vis["test_idx"]
    reps = {
        "z": vis["z_t"].float(),
        "h": vis["h_t"].float(),
        "q_v4": vis["q_v4_t"].float(),
        "q_v41": vis["q_v41_t"].float(),
        "hq": vis["hq_t"].float(),
    }
    rows = []
    pred_pos = {}
    pred_res = {}
    for name, x in reps.items():
        pp = ridge_predict_all(x, vis["position_t"], tr)
        rr = ridge_predict_all(x, vis["residual_t"], tr)
        pred_pos[name] = pp
        pred_res[name] = rr
        pm = metric_dict(pp[te], vis["position_t"][te], prefix="pos_")
        rm = metric_dict(rr[te], vis["residual_t"][te], prefix="res_")
        logits = train_hard_linear(x, vis["hard_labels"], tr, te)
        hm = binary_metrics_from_logits(logits, vis["hard_labels"][te])
        rows.append({"representation": name, **pm, **rm, **hm})

    scatter_pos(fig_dir / "true_position_distribution.png", vis["position_t"][te], title="True position distribution")
    for name in ["h", "q_v4", "q_v41", "hq"]:
        scatter_pos(fig_dir / f"pred_position_{name}.png", pred_pos[name][te], title=f"Predicted position from {name}")

    # Select q dimensions by post-hoc correlation.
    sem_rows = []
    targets = {
        "position_x": vis["position_t"][:, 0],
        "position_y": vis["position_t"][:, 1],
        "residual_x": vis["residual_t"][:, 0],
        "residual_y": vis["residual_t"][:, 1],
    }
    for i, label in enumerate(vis["hard_label_names"]):
        targets[f"hard_{label}"] = vis["hard_labels"][:, i]
    for qname in ["q_v4", "q_v41"]:
        q = vis[f"{qname}_t"]
        for d in range(q.shape[1]):
            vals = [(k, pearson(q[te, d], v[te])) for k, v in targets.items()]
            top, c = max(vals, key=lambda kv: abs(kv[1]))
            sem_rows.append({"q": qname, "dim": d, "top_target": top, "pearson": round(c, 4), "abs_pearson": round(abs(c), 4)})
        for row in sorted([r for r in sem_rows if r["q"] == qname], key=lambda r: r["abs_pearson"], reverse=True)[:6]:
            spatial_heatmap(fig_dir / f"{qname}_dim{row['dim']}_{row['top_target']}.png", vis["position_t"][te], q[te, row["dim"]], f"{qname} dim {row['dim']} over position")

    for label in ["action_error_top10", "high_residual_top10", "large_action_small_disp"]:
        spatial_heatmap(fig_dir / f"hard_subset_{label}.png", vis["position_t"][te], vis["sample_flags"][label][te].float(), f"{label} spatial density")
    quiver_residual(fig_dir / "residual_direction_map.png", vis["position_t"][te], vis["residual_t"][te], title="Mean residual correction direction")

    text = f"""# V5 Semantic Maps

## Readout Metrics

{table_md(rows, ["representation", "pos_overall_mse", "pos_x_r2", "pos_y_r2", "res_overall_mse", "res_x_r2", "res_y_r2", "hard_acc", "hard_f1"])}

## q Semantic Alignment

{table_md(sorted(sem_rows, key=lambda r: r["abs_pearson"], reverse=True)[:20], ["q", "dim", "top_target", "pearson", "abs_pearson"])}

## Figures

- true position distribution: `{fig_dir / 'true_position_distribution.png'}`
- predicted position maps: `{fig_dir / 'pred_position_h.png'}`, `{fig_dir / 'pred_position_q_v4.png'}`, `{fig_dir / 'pred_position_q_v41.png'}`, `{fig_dir / 'pred_position_hq.png'}`
- hard subset heatmaps: `{fig_dir / 'hard_subset_action_error_top10.png'}`, `{fig_dir / 'hard_subset_high_residual_top10.png'}`, `{fig_dir / 'hard_subset_large_action_small_disp.png'}`
- residual direction map: `{fig_dir / 'residual_direction_map.png'}`

## Notes

q heatmaps are post-hoc spatial activation maps. They should be interpreted as state partitions or correction factors unless rule enrichment and ablation provide stronger evidence.
"""
    write_report("03_semantic_maps.md", text)


if __name__ == "__main__":
    main()

