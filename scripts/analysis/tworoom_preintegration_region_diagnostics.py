import torch

from tworoom_preintegration_common import *


def bin_errors(pos, err, bins=20):
    x = pos[:, 0]
    y = pos[:, 1]
    xi = torch.bucketize(x, torch.linspace(x.min(), x.max(), bins + 1)[1:-1])
    yi = torch.bucketize(y, torch.linspace(y.min(), y.max(), bins + 1)[1:-1])
    heat = torch.full((bins, bins), float("nan"))
    for i in range(bins):
        for j in range(bins):
            m = (xi == i) & (yi == j)
            if m.any():
                heat[j, i] = err[m].mean()
    return heat


def main():
    ensure_dirs()
    fig = PREINT_OUT / "figures_region"
    fig.mkdir(parents=True, exist_ok=True)
    data = load_latents()
    z = data["z"].float()
    pos = data["position"].float()
    train_idx, _, test_idx, *_ = split_by_episode(data["episode_id"])
    z_std, _, _ = standardize_from_train(z, train_idx)
    lin = torch.load(PREINT_OUT / "static_probes" / "linear_static.pt", map_location="cpu")
    pred = predict_linear(z_std[test_idx], lin["W"], lin["b"])
    true = pos[test_idx]
    err = pred - true
    abs_y = err[:, 1].abs()
    heat = bin_errors(true, abs_y)
    plt.figure(figsize=(6, 5))
    plt.imshow(heat.numpy(), origin="lower", aspect="auto")
    plt.colorbar(label="mean |y error|")
    plt.title("Linear probe y error by spatial bin")
    plt.xlabel("x bin")
    plt.ylabel("y bin")
    plt.tight_layout()
    plt.savefig(fig / "linear_y_error_spatial_bins.png", dpi=160)
    plt.close()
    save_scatter(fig / "linear_y_error_vs_true_y.png", true[:, 1], abs_y, "true y", "|y error|", "y error vs y")
    save_scatter(fig / "linear_y_error_vs_timestep.png", data["timestep"][test_idx].float(), abs_y, "timestep", "|y error|", "y error vs timestep")

    q = torch.quantile(abs_y, torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99]))
    report = f"""# Region Diagnostics

Wall / door constants were not read from a reliable TwoRoom env source in this clean-code analysis. Spatial quantile/bin diagnostics are used instead.

Linear probe y absolute error quantiles:

- 50%: `{q[0].item()}`
- 75%: `{q[1].item()}`
- 90%: `{q[2].item()}`
- 95%: `{q[3].item()}`
- 99%: `{q[4].item()}`

Figures:

- `outputs/tworoom_preintegration/figures_region/linear_y_error_spatial_bins.png`
- `outputs/tworoom_preintegration/figures_region/linear_y_error_vs_true_y.png`
- `outputs/tworoom_preintegration/figures_region/linear_y_error_vs_timestep.png`

## Answers

- This diagnostic identifies whether y error is concentrated in spatial bins rather than globally uniform.
- Compare with MLP/KANFIS reports to determine whether nonlinear probes repair y.
- If y error concentrates near implicit obstacle/border regions, formal integration should include region-aware transition checks.
"""
    (PREINT_REPORT / "07_region_diagnostics.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

