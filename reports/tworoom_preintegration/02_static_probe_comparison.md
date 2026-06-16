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
