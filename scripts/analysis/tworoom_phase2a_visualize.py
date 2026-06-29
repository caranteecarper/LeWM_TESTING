from tworoom_phase2a_common import *


def best_overall_path():
    rows = read_csv(TABLE / "adapter_eval_metrics.csv")
    cand = [r for r in rows if r.get("subset") == PRIMARY_LABEL and r.get("input_group") == "full_q_action" and r.get("gate_name") in ("c_soft_gate", "c_sigmoid_gate")]
    if not cand:
        cand = [r for r in rows if r.get("subset") == PRIMARY_LABEL and r.get("model") != "base_official_predictor"]
    best = sorted(cand, key=lambda r: float(r["relative_improvement"]), reverse=True)[0]
    return MODEL_DIR / f"{best['input_group']}__{best['model_type']}__{best['gate_name']}__{best['loss_mode']}__seed{best['seed']}.pt", best


def scatter(path, pos, values, title, label):
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(pos[:, 0], pos[:, 1], c=values, s=6, cmap="magma", alpha=0.6)
    plt.colorbar(label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def main():
    ensure_dirs()
    data = load_adapter_dataset()
    path, best = best_overall_path()
    model, obj = load_adapter(path)
    final, corr, gate = evaluate_adapter_obj(data, model, obj)
    test = data["split"]["head_test_idx"]
    base_err = ((data["base_pred_z"] - data["target_z"]) ** 2).mean(1)
    final_err = ((final - data["target_z"]) ** 2).mean(1)
    improvement = base_err - final_err
    pos = data["position"][test].float()
    scatter(FIG / "base_error_spatial_map.png", pos, base_err[test], "base official predictor error", "MSE")
    scatter(FIG / "final_error_spatial_map.png", pos, final_err[test], "adapter final error", "MSE")
    scatter(FIG / "improvement_spatial_map.png", pos, improvement[test], "base - final improvement", "MSE improvement")
    scatter(FIG / "gate_spatial_map.png", pos, gate[test], "adapter gate", "gate")
    scatter(FIG / "correction_norm_spatial_map.png", pos, corr[test].norm(dim=1), "correction norm", "norm")

    plt.figure(figsize=(5.2, 3.6))
    plt.hist(base_err[test].numpy(), bins=60, alpha=0.65, label="base")
    plt.hist(final_err[test].numpy(), bins=60, alpha=0.65, label="final")
    plt.legend()
    plt.xlabel("per-sample latent MSE")
    plt.tight_layout()
    plt.savefig(FIG / "before_after_error_histogram.png", dpi=170)
    plt.close()

    corrected = test[torch.argsort(improvement[test], descending=True)[:12]]
    over = test[torch.argsort(improvement[test])[:12]]
    save_image_grid(FIG / "top_corrected_examples.png", load_images(data, corrected), [f"impr={improvement[i]:.4g}" for i in corrected])
    save_image_grid(FIG / "over_corrected_examples.png", load_images(data, over), [f"impr={improvement[i]:.4g}" for i in over])

    text = f"""# Visualization

展示模型：`{path}`

本阶段展示的是 adapter correction 效果，不是环境 reward，不是 continuation，不是 SIGReg，不是多步 rollout。

图片目录：`{FIG}`
"""
    write_report("06_visualization.md", text)


if __name__ == "__main__":
    main()
