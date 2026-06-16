# TwoRoom Pre-Integration Report

## 1. Goal

This stage validates an external physical-factor extraction path before any formal LeWM predictor integration. No official LeWM encoder, predictor, loss, FFN, or TwoRoom environment logic is modified.

## 2. Official Baseline And Readout

Official checkpoint: `/data/lzt26/stable-wm/checkpoints/tworoom_official_baseline_full/weights_epoch_100.pt`

Existing linear readout test result: x R2 about 0.9916, y R2 about 0.7251.

## 3. Static Probe Comparison

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


## 4. Factor Bottleneck

# Factor Bottleneck

Input: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_encoder_readout/tworoom_latents.pt`

Split: episode-level 80/10/10. Models are external analysis bottlenecks; LeWM is not modified.

|model|K|x_r2|y_r2|x_mse|y_mse|overall_mse|h_var_min|h_var_mean|h_abs_corr_max_offdiag|
|---|---|---|---|---|---|---|---|---|---|
|linear_bottleneck|2|-0.00032007694244384766|-8.654594421386719e-05|1398.5205078125|1590.55908203125|1494.5399169921875|0.35222986340522766|0.5248579978942871|0.8436780571937561|
|mlp_bottleneck|2|0.005716085433959961|0.016792356967926025|1390.08154296875|1563.714599609375|1476.8980712890625|1.9297293424606323|2.193216323852539|0.9648138880729675|
|linear_bottleneck|4|0.8228600025177002|0.7379582524299622|247.65460205078125|416.7568359375|332.2057189941406|0.5285527110099792|0.641357958316803|0.8345525860786438|
|mlp_bottleneck|4|0.9585137367248535|0.9668743014335632|58.00083923339844|52.68382263183594|55.34233093261719|0.509711742401123|4.377853870391846|0.8648676872253418|
|linear_bottleneck|6|0.9332078695297241|0.9250738620758057|93.38031768798828|119.16414642333984|106.27223205566406|0.20902833342552185|0.9915308952331543|0.8746181726455688|
|mlp_bottleneck|6|0.9896620512008667|0.988715410232544|14.45323371887207|17.9472713470459|16.200252532958984|0.6833245158195496|1.9976673126220703|0.8757295608520508|
|linear_bottleneck|8|0.9608485698699951|0.9566750526428223|54.73652267456055|68.90489196777344|61.820709228515625|0.09973671287298203|0.7472873330116272|0.7728251218795776|
|mlp_bottleneck|8|0.9934296011924744|0.9936822056770325|9.185866355895996|10.04791259765625|9.616889953613281|0.9732953310012817|2.686260223388672|0.9287861585617065|
|linear_bottleneck|16|0.9911820292472839|0.9907965064048767|12.32816219329834|14.6373929977417|13.482778549194336|0.14222894608974457|0.8826844692230225|0.9246177077293396|
|mlp_bottleneck|16|0.9980376362800598|0.9981366395950317|2.7435128688812256|2.963564395904541|2.8535385131835938|0.3945061266422272|1.4516279697418213|0.9363141655921936|

## Key Answers

- Recommended K by overall MSE: `16` using `mlp_bottleneck`.
- K=2 is sufficient only if its R2/MSE is close to larger K; see table.
- Collapse check: `h_var_min` near zero indicates collapse.
- h dimensions are candidate physical factors only when correlated with x/y or spatial regions; no dimension is named directly as x or y.


## 5. Delta Readout

# Delta Readout

Input pairs: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration/tworoom_pairs.pt`

|model|x_mse|y_mse|x_rmse|y_rmse|x_r2|y_r2|x_pearson|y_pearson|overall_mse|
|---|---|---|---|---|---|---|---|---|---|
|static_W_delta_projection|10.023714065551758|149.8037872314453|3.166024923324585|12.239436149597168|0.4443133473396301|-7.041701316833496|0.8002591133117676|0.3077795207500458|79.91374969482422|
|linear_delta_z|1.8970062732696533|7.124406337738037|1.3773185014724731|2.669158458709717|0.8948352932929993|0.6175507307052612|0.9459576606750488|0.7858632802963257|4.510706424713135|
|mlp_delta_z|1.502173900604248|1.2579960823059082|1.2256320714950562|1.1216042041778564|0.9167236685752869|0.932468831539154|0.9575675129890442|0.9669322967529297|1.3800849914550781|
|kanfis_style_delta_top32|16.491212844848633|18.38098907470703|4.060937404632568|4.28730583190918|0.08577334880828857|0.013279855251312256|0.349660724401474|0.16373707354068756|17.43610191345215|

## Answers

- Static W delta projection is listed as a baseline when available.
- Retrained delta_z probes should be compared against static W projection in the table.
- If delta R2 is positive and Pearson high, latent changes contain physical motion information.
- If delta_y remains weak, h+action transition should be treated cautiously.


## 6. Action-Conditioned Transition

# Action-Conditioned Transition

|model|x_mse|y_mse|x_rmse|y_rmse|x_r2|y_r2|x_pearson|y_pearson|overall_mse|
|---|---|---|---|---|---|---|---|---|---|
|action_only_linear|1.6350470781326294|0.0823291540145874|1.2786896228790283|0.2869305908679962|0.9093575477600098|0.9955804347991943|0.9536048173904419|0.9977798461914062|0.8586881160736084|
|action_only_mlp|3.597834825515747|2.299891710281372|1.8967958688735962|1.516539454460144|0.8005461096763611|0.8765382170677185|0.8947473168373108|0.9362383484840393|2.9488635063171387|
|z_action_linear|1.1204280853271484|0.06773268431425095|1.0585027933120728|0.26025503873825073|0.9378865957260132|0.9963639974594116|0.9684463143348694|0.9981722831726074|0.59408038854599|
|z_action_mlp|0.49970120191574097|0.07486989349126816|0.7068954706192017|0.27362361550331116|0.9722979664802551|0.9959808588027954|0.9860589504241943|0.9979894161224365|0.28728556632995605|
|hK8_action_linear|1.527862787246704|0.08086106181144714|1.236067533493042|0.28436079621315|0.9152995347976685|0.9956592321395874|0.9567149877548218|0.99781334400177|0.8043619394302368|
|hK8_action_mlp|1.6932649612426758|0.28317737579345703|1.3012551069259644|0.5321441292762756|0.9061301350593567|0.9847986102104187|0.9519200325012207|0.9923575520515442|0.9882211685180664|
|hK8_action_kanfis_style|9.54210376739502|11.134133338928223|3.089029550552368|3.336784839630127|0.4710124731063843|0.4023023843765259|0.9435075521469116|0.9855694770812988|10.338118553161621|

## Answers

- Action-only shows how much displacement is predictable from action alone.
- z+action is the 192-D upper-bound diagnostic.
- h+action uses saved bottleneck `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration/factor_bottleneck/mlp_bottleneck_K8.pt`.
- Formal integration is supported only if h+action clearly improves over action-only and approaches z+action.


## 7. Ablation / Masking

# Ablation Importance

## Static z -> position ablation

|ablation|x_mse|y_mse|x_r2|y_r2|x_pearson|y_pearson|overall_mse|
|---|---|---|---|---|---|---|---|
|none|4.282838821411133|31.945621490478516|0.9969366192817688|0.9799137115478516|0.9984649419784546|0.9899068474769592|18.11423110961914|
|mask_linear_top32|1291.97802734375|20365.73046875|0.0758865475654602|-11.805240631103516|0.4982459247112274|0.2465152144432068|10828.8544921875|
|mask_random32|833.1935424804688|12836.4267578125|0.40404146909713745|-7.071084976196289|0.7640621066093445|0.19867748022079468|6834.810546875|
|mask_high_variance32|691.092529296875|13808.5654296875|0.505682110786438|-7.682331085205078|0.8183394074440002|0.1333647519350052|7249.82958984375|
|mask_kanfis_top32|1291.97802734375|20365.73046875|0.0758865475654602|-11.805240631103516|0.4982459247112274|0.2465152144432068|10828.8544921875|

## h + action transition h-dim ablation

|ablation|x_mse|y_mse|x_r2|y_r2|x_pearson|y_pearson|overall_mse|
|---|---|---|---|---|---|---|---|
|h_none|1.527862787246704|0.08086106181144714|0.9152995347976685|0.9956592321395874|0.9567149877548218|0.99781334400177|0.8043619394302368|
|mask_h0|1.5436151027679443|0.08105386793613434|0.914426326751709|0.9956489205360413|0.956256091594696|0.9978076219558716|0.812334418296814|
|mask_h1|1.7896203994750977|0.08086220175027847|0.900788426399231|0.9956591725349426|0.949626624584198|0.99781334400177|0.9352412223815918|
|mask_h2|1.553971767425537|0.08092810213565826|0.9138521552085876|0.9956556558609009|0.9559570550918579|0.9978113174438477|0.8174498677253723|
|mask_h3|1.5373021364212036|0.08086522668600082|0.9147762656211853|0.9956590533256531|0.9564440250396729|0.9978134632110596|0.8090836405754089|
|mask_h4|1.6627546548843384|0.08107593655586243|0.907821536064148|0.9956477284431458|0.9529330730438232|0.997806966304779|0.8719152808189392|
|mask_h5|1.527994155883789|0.08222813904285431|0.9152922630310059|0.9955858588218689|0.9567111134529114|0.9977767467498779|0.8051111698150635|
|mask_h6|1.591861605644226|0.08146998286247253|0.9117516279220581|0.9956265687942505|0.9548652768135071|0.9977964758872986|0.8366657495498657|
|mask_h7|1.5879168510437012|0.08107458055019379|0.911970317363739|0.9956477880477905|0.9549800157546997|0.9978076815605164|0.8344956636428833|

## Answers

- Top-k dims are meaningful if masking them degrades R2/MSE more than random-k.
- If KANFIS-selected dims do not hurt when masked, KANFIS importance is not reliable enough for integration.
- h dimensions are necessary only if single-dim masks measurably degrade transition prediction.


## 8. Region Diagnostics

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


## 9. Go / No-Go

Current decision: partial Go for external factor diagnostics, No-Go for direct KANFIS-style predictor integration as currently implemented.

- Static MLP readout is strong (`x_r2=0.9978`, `y_r2=0.9971`), so the official encoder latent contains readable position information.
- Current KANFIS-style static probes are not usable as integration candidates (`x_r2/y_r2` are negative), despite being useful as a failed baseline.
- MLP bottleneck needs at least `K=8` for a compact factor that preserves position well; `K=16` is best overall. `K=2` is not sufficient.
- Delta readout supports physical transition information in latent changes when using MLP (`x_r2=0.9167`, `y_r2=0.9325`), but static `W` delta projection and KANFIS-style delta are weak.
- `hK8 + action` linear transition is stable and close to action-only, but it does not clearly approach the best `z + action` MLP upper bound. This is not strong enough yet for formal LeWM predictor replacement.
- Ablation confirms the static top latent dimensions are meaningful: masking linear top-32 dimensions hurts more than random/high-variance masking.

## 10. Next Step

Recommended next step: keep LeWM untouched and improve the external factor/transition path first. Test `K=16` factor transition, add an explicit `h_next` or `delta_h` prediction objective outside LeWM, and only consider formal predictor-side integration if `[h_t, action]` closes more of the gap to `z + action`. Do not integrate the current KANFIS-style probe into LeWM yet.
