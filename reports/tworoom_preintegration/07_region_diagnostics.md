# Region Diagnostics

Wall / door constants were not read from a reliable TwoRoom env source in this clean-code analysis. Spatial quantile/bin diagnostics are used instead.

Linear probe y absolute error quantiles:

- 50%: `3.5501747131347656`
- 75%: `6.149787902832031`
- 90%: `9.01250171661377`
- 95%: `10.962159156799316`
- 99%: `16.12163543701172`

Figures:

- `outputs/tworoom_preintegration/figures_region/linear_y_error_spatial_bins.png`
- `outputs/tworoom_preintegration/figures_region/linear_y_error_vs_true_y.png`
- `outputs/tworoom_preintegration/figures_region/linear_y_error_vs_timestep.png`

## Answers

- This diagnostic identifies whether y error is concentrated in spatial bins rather than globally uniform.
- Compare with MLP/KANFIS reports to determine whether nonlinear probes repair y.
- If y error concentrates near implicit obstacle/border regions, formal integration should include region-aware transition checks.
