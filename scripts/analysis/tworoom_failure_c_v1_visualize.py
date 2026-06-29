from tworoom_failure_c_v1_common import *


def best_model_for(data, label=PRIMARY_LABEL, group="h_q_action"):
    rows = read_csv(TABLE / "eval_metrics.csv")
    cand = [r for r in rows if r["label"] == label and r["feature_group"] == group]
    if not cand:
        cand = [r for r in rows if r["label"] == label]
    best = sorted(cand, key=lambda r: float(r["AUPRC"]), reverse=True)[0]
    path = MODEL_DIR / f"{best['feature_group']}__{best['model_type']}__seed{best['seed']}.pt"
    return path, best


def main():
    ensure_dirs()
    data = load_dataset()
    path, best = best_model_for(data)
    model, obj = load_head(path)
    x, _ = make_feature(data, obj["feature_group"], seed=int(obj["seed"]))
    x_std = (x.float() - obj["x_mean"]) / obj["x_std"].clamp_min(1e-6)
    test_idx = data["split"]["head_test_idx"]
    pred = predict_model(model, x_std[test_idx])
    label_i = CLASS_LABELS.index(data["primary_label"])
    risk = sigmoid(pred[:, label_i])
    y = data["labels"][data["primary_label"]][test_idx].float()

    plt.figure(figsize=(5.2, 3.6))
    plt.hist(risk[y < 0.5].numpy(), bins=40, alpha=0.65, label="negative")
    plt.hist(risk[y > 0.5].numpy(), bins=40, alpha=0.65, label="positive")
    plt.xlabel("predicted failure risk")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "risk_histogram_positive_vs_negative.png", dpi=170)
    plt.close()

    order = torch.argsort(risk, descending=True)
    yy = y[order]
    precision = torch.cumsum(yy, 0) / torch.arange(1, yy.numel() + 1, dtype=torch.float)
    recall = torch.cumsum(yy, 0) / yy.sum().clamp_min(1)
    plt.figure(figsize=(4.5, 4.2))
    plt.plot(recall.numpy(), precision.numpy())
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(data["primary_label"])
    plt.tight_layout()
    plt.savefig(FIG / "precision_recall_primary.png", dpi=170)
    plt.close()

    pos = data["position"][test_idx]
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(pos[:, 0], pos[:, 1], c=risk, s=7, cmap="magma", alpha=0.65)
    plt.colorbar(label="C predicted risk")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("C(h,q,a) predicted official predictor failure risk")
    plt.tight_layout()
    plt.savefig(FIG / "spatial_risk_map.png", dpi=170)
    plt.close()

    qg = group_activation(data["q"][test_idx], "mean").reshape(-1)
    top_rule = qg >= torch.quantile(qg, 0.9)
    top_risk = risk >= torch.quantile(risk, 0.9)
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(pos[:, 0], pos[:, 1], c="lightgray", s=5, alpha=0.25)
    plt.scatter(pos[top_risk, 0], pos[top_risk, 1], c="red", s=10, alpha=0.5, label="top C risk")
    plt.scatter(pos[top_rule, 0], pos[top_rule, 1], facecolors="none", edgecolors="blue", s=25, linewidths=0.7, label="top V4.2 group")
    both = top_rule & top_risk
    plt.scatter(pos[both, 0], pos[both, 1], c="gold", s=16, alpha=0.8, label="both")
    plt.legend(fontsize=7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(FIG / "v42_group_vs_predicted_risk_overlay.png", dpi=170)
    plt.close()

    top = test_idx[torch.argsort(risk, descending=True)[:12]]
    imgs = load_images(data, top)
    titles = [f"risk={risk[torch.argsort(risk, descending=True)[i]]:.2f}\ny={int(data['labels'][data['primary_label']][top[i]].item())}" for i in range(top.numel())]
    save_image_grid(FIG / "top_predicted_risk_images.png", imgs, titles)

    fp_local = torch.nonzero((risk >= torch.quantile(risk, 0.9)) & (y < 0.5)).flatten()[:12]
    fn_local = torch.nonzero((risk < torch.quantile(risk, 0.5)) & (y > 0.5)).flatten()[:12]
    if fp_local.numel():
        rows = test_idx[fp_local]
        save_image_grid(FIG / "false_positive_examples.png", load_images(data, rows), [f"risk={risk[i]:.2f}" for i in fp_local])
    if fn_local.numel():
        rows = test_idx[fn_local]
        save_image_grid(FIG / "false_negative_examples.png", load_images(data, rows), [f"risk={risk[i]:.2f}" for i in fn_local])

    text = f"""# 可视化

本页展示最佳 `{obj['feature_group']} / {obj['model_type']} / seed={obj['seed']}` 的 C head。

C 模块预测的是 official predictor failure risk：

- 不是环境 reward；
- 不是 episode continuation；
- 不是 SIGReg loss；
- 不是已经接入 LeWM 的修正模块。

输出图片目录：`{FIG}`
"""
    write_report("05_visualization.md", text)


if __name__ == "__main__":
    main()
