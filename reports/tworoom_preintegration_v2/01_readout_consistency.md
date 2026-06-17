# Readout Consistency

## Inputs

- latent cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_encoder_readout/tworoom_latents.pt`
- old saved probe: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_encoder_readout/linear_probe_position.pt`
- old report: `/data/lzt26/lewm_official_tworoom_readout_git/reports/tworoom_encoder_readout/07_position_probe_result.md` exists=True
- v1 static report: `/data/lzt26/lewm_official_tworoom_readout_git/reports/tworoom_preintegration/02_static_probe_comparison.md` exists=True
- unified split file: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v2/splits_episode_seed3072.pt`

Split is episode-level with seed 3072:

- train episodes: `8000`, samples: `584945`
- val episodes: `1000`, samples: `73141`
- test episodes: `1000`, samples: `72723`

## Recomputed Results

|case|x_mse|y_mse|x_r2|y_r2|x_pearson|y_pearson|overall_mse|
|---|---|---|---|---|---|---|---|
|saved_old_probe_W|11.77698040008545|437.2347106933594|0.9915762543678284|0.725082516670227|0.9957758784294128|0.8524305820465088|224.50584411621094|
|fresh_lstsq_no_ridge|8.210810661315918|157.36854553222656|0.9941270351409912|0.9010522961616516|0.9970589280128479|0.9495103359222412|82.78968811035156|
|unified_ridge_1e-4|4.282838821411133|31.945621490478516|0.9969366192817688|0.9799137115478516|0.9984649419784546|0.9899068474769592|18.11423110961914|

## Diagnosis

- The old reported y R2 was `0.725082516670227`.
- The current unified baseline is `unified_ridge_1e-4`, with y R2 `0.9799137115478516`.
- The split policy, seed, target scale, and metric formula are the same at the high level. The material difference is the linear solver path: the old script used `torch.linalg.lstsq` without ridge regularization, while v1 static comparison used a ridge normal-equation solve.
- The saved old weights reproduce the lower y score, so the inconsistency is not caused by a different latent cache at evaluation time.
- For all v2 experiments, use `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v2/splits_episode_seed3072.pt` and the unified ridge baseline as the position readout baseline.

## V1 Static Report Excerpt

```text
# Static Probe Comparison

Input: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_encoder_readout/tworoom_latents.pt`

Split: episode-level 80/10/10, seed 3072.

|model|x_mse|y_mse|x_rmse|y_rmse|x_r2|y_r2|x_pearson|y_pearson|overall_mse|
|---|---|---|---|---|---|---|---|---|---|
|linear|4.282838821411133|31.945621490478516|2.069502115249634|5.652045726776123|0.9969366192817688|0.9799137115478516|0.9984649419784546|0.9899068474769592|18.11423110961914|
|mlp|3.1298909187316895|4.564048767089844|1.7691497802734375|2.1363635063171387|0.9977613091468811|0.997130274772644|0.9988769888877869|0.9985749125480652|3.8469698429107666|
|kanfis_style_full192|13131.7080078125|8673.8701171875|114.59366607666016|93.13361358642578|-8.392720222473145|-4.4538187980651855|-0.10235268622636795|0.4772135317325592|10902.7890625|
|kanfis_style_top32|13153.4462890625|8694.3525390625|114.6884765625|93.24351501464844|-8.408268928527832|-4.4666972160339355|-0.304726779460907|0.27517369389533997|10923.8994140625|

## Answers

- MLP improves y over linear: `True`.
- Best y model: `mlp` with y R2 `0.997130274772644`.
- KANFIS-style top32/full192 are external analysis probes, not LeWM modules.
- KANFIS-style top dims are saved in `outputs/tworoom_preintegration/static_probes/`.
- Linear top32 dims: `[136, 137, 55, 91, 152, 59, 44, 39, 57, 64, 77, 113, 74, 71, 110, 90, 19, 30, 41, 14, 79, 179, 6, 178, 36, 147, 40, 169, 149, 37, 0, 160]`.
- Conclusion: continue bottleneck extraction if nonlinear/static probes improve y or reveal concentrated latent dimensions.

```
