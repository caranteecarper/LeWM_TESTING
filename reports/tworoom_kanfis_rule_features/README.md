# TwoRoom KANFIS Rule-Feature Layer README

## Executive Summary

This stage tests an external input-side path:

```text
z_t -> h_t -> KANFIS rule-feature layer -> q_t
```

`q_t` is a candidate interpretable input for a later prediction module. This run does not modify LeWM, does not replace FFN, and does not feed action into q extraction.

The final Go / Partial / No-Go judgment should be based on whether q keeps position information, whether q+action beats action-only on residual/delta/hard subsets, whether it approaches h+action, and whether rule ablation shows functional contribution.


---

# TwoRoom KANFIS Rule-Feature Layer Scope

Generated: 2026-06-17T20:54:21

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

Execution note: the server clean clone could not be fast-forwarded during this run, so the server-side `git` metadata above still reports the older local HEAD. The v4 scripts used for this run were synchronized from local/GitHub commit `e203172 Add TwoRoom KANFIS rule-feature diagnostics`.

## Goal

Build an input-side interpretable factor path:

```text
z_t -> h_t -> KANFIS rule-feature layer -> q_t
```

## Constraints

- No official LeWM encoder / predictor / loss / train.py / module.py / TwoRoom environment modification.
- No FFN replacement and no ARPredictor integration.
- Action does not enter `F(z_t)` or `KANFIS(h_t) -> q_t`.
- Action only enters external diagnostic heads after q is produced.
- `q_t` is a KANFIS rule-activation physical factor candidate for future prediction modules.

## Inputs

- pair cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration/tworoom_pairs.pt` exists=True
- split: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v2/splits_episode_seed3072.pt` exists=True
- v3 teacher: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v3/v3_teacher_labels.pt` exists=True
- best h key: `K16_pos_delta_residual_teacher_hard_hnext_light_aux`


---

# V4 Best-H Dataset

## Source

- best h key: `K16_pos_delta_residual_teacher_hard_hnext_light_aux`
- best h checkpoint: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v3/dynamics_aware_h/K16_pos_delta_residual_teacher_hard_hnext_light_aux.pt`
- output artifact: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/best_h_dataset.pt`

## Summary

|metric|value|
|---|---|
|samples|720809|
|h_dim|16|
|train_samples|576945|
|val_samples|72141|
|test_samples|71723|
|h_var_min_test|0.287949|
|h_var_mean_test|0.547713|
|h_abs_corr_max_offdiag_test|0.72911|
|delta_h_norm_mean_test|1.660933|

## H Dimension Top Correlations On Test

|h_dim|top_target|top_abs_pearson|
|---|---|---|
|0|position_x|0.4971|
|1|position_x|0.6259|
|2|position_x|0.4173|
|3|position_x|0.7867|
|4|position_y|0.5963|
|5|position_y|0.6025|
|6|hard_action_error_top10|0.3089|
|7|position_x|0.5089|
|8|hard_action_error_top10|0.3656|
|9|hard_action_error_top10|0.3611|
|10|position_y|0.4176|
|11|position_y|0.8048|
|12|position_y|0.3549|
|13|hard_action_error_top10|0.3706|
|14|position_y|0.6804|
|15|hard_action_error_top10|0.254|

## Constraint Check

- `h_t` and `h_next` are produced by the v3 best extractor from `z_t` and `z_next` only.
- Action is copied into the dataset for later diagnostic heads, but action is not used to generate h.
- No official LeWM encoder, predictor, loss, train script, module, or TwoRoom environment is modified.


---

# V4 KANFIS Rule-Feature Training

`q_t` is defined as the softmax-normalized firing strength of learnable KANFIS-style rule prototypes over `h_t`.
Action is not an input to the rule-feature layer; action only enters diagnostic heads after q is produced.

Recommended best q model: `q_pos_delta_residual_hard_hnext_R16`.

|variant|num_rules|pos_x_r2|pos_y_r2|delta_x_r2|delta_y_r2|res_x_r2|res_y_r2|hard_auc|hard_f1|hnext_mean_r2|delta_h_mean_r2|q_var_mean|q_entropy_mean|active_rule_count_mean|rule_usage_min|rule_usage_max|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|q_pos_only|16|0.9934326410293579|0.9879767894744873|-0.17335569858551025|0.05450516939163208|-0.07401609420776367|-0.023489713668823242|0.5115492045879364|0.21259232025814218|-0.03537558764219284|-0.05124741047620773|0.0001218492616317235|2.7376708984375|8.710762023925781|0.030664458870887756|0.0787622332572937|
|q_pos_delta|16|0.9908133745193481|0.9919946789741516|0.9041892886161804|0.9884588718414307|-0.013581395149230957|-0.0285872220993042|0.4891046643257141|0.20441393437948463|-0.03988632187247276|-0.04313597083091736|0.00011878721124958247|2.734084129333496|8.689346313476562|0.028689729049801826|0.0770290419459343|
|q_pos_residual|16|0.993518590927124|0.9905507564544678|0.03749716281890869|0.014225900173187256|0.15438073873519897|-0.00020992755889892578|0.49089112877845764|0.21351842197194434|-0.048198699951171875|-0.04333131015300751|0.00015352608170360327|2.7336111068725586|8.536675453186035|0.032098960131406784|0.07588627189397812|
|q_pos_residual_hard|16|0.9919958114624023|0.990554690361023|-0.031928420066833496|0.04124641418457031|0.07637882232666016|-7.712841033935547e-05|0.835535752773285|0.4147876144363578|-0.03834347426891327|-0.034038983285427094|0.00014769771951250732|2.734633684158325|8.481672286987305|0.0339241661131382|0.07534891366958618|
|q_pos_delta_residual_hard_hnext|16|0.9917436242103577|0.9897725582122803|0.9055083990097046|0.9884514808654785|0.10448968410491943|-0.0003751516342163086|0.831038773059845|0.4099653278008526|0.4584018886089325|0.07962966710329056|0.00016203560517169535|2.7319564819335938|7.80605936050415|0.03273436054587364|0.08593717217445374|
|q_pos_only|32|0.9941663146018982|0.9940518736839294|0.021301686763763428|0.011127293109893799|-0.0329892635345459|-0.043205857276916504|0.4987886309623718|0.20998579108020268|-0.036322616040706635|-0.04468858242034912|2.505562406440731e-05|3.4284353256225586|17.494415283203125|0.01210976392030716|0.04042847082018852|
|q_pos_delta|32|0.9947845339775085|0.9935483932495117|0.9043104648590088|0.9889758229255676|-0.026985526084899902|-0.01780855655670166|0.4954301655292511|0.19398401234331553|-0.03881213068962097|-0.04284796863794327|2.541898538765963e-05|3.4267373085021973|17.045257568359375|0.01155837532132864|0.040809761732816696|
|q_pos_residual|32|0.994556725025177|0.9937136769294739|0.020123720169067383|-0.2508375644683838|0.03700685501098633|0.00018137693405151367|0.49802528619766234|0.18850732993513514|-0.055208466947078705|-0.053355470299720764|2.8722037313855253e-05|3.427001953125|17.417774200439453|0.013347680680453777|0.04123397916555405|
|q_pos_residual_hard|32|0.9956916570663452|0.9934489727020264|-0.11920225620269775|-0.11170852184295654|0.02931082248687744|-0.0018752813339233398|0.7821497082710266|0.40435378229348784|-0.03854915872216225|-0.04086620360612869|2.8089023544453084e-05|3.4294164180755615|17.29898452758789|0.013115634210407734|0.03994457796216011|
|q_pos_delta_residual_hard_hnext|32|0.9944911003112793|0.9926695823669434|0.9065605401992798|0.9889777302742004|0.08014732599258423|0.0006392002105712891|0.8172608494758606|0.41965997493826734|0.464184045791626|0.0719972550868988|3.99145828851033e-05|3.424309730529785|17.021108627319336|0.014865750446915627|0.039393842220306396|

## Collapse Check

- `q_var_mean`, `q_entropy_mean`, active rule count, and rule usage range are reported for every variant.
- A low `q_var_mean` with near-uniform usage would indicate q collapse.
- Models are saved under `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/rule_feature_models` and are not committed to git.


---

# V4 q + Action Transition Diagnostics

Best q model: `q_pos_delta_residual_hard_hnext_R16`.

## Transition Metrics

|input|task|model|subset|overall_mse|x_r2|y_r2|mean_r2|hard_auc|hard_f1|
|---|---|---|---|---|---|---|---|---|---|
|action_only|delta|linear|all_test|0.8586020469665527|0.9093542695045471|0.9955928921699524||||
|action_only|delta|linear|action_error_top20|3.992807149887085|0.6007741093635559|0.9787328839302063||||
|action_only|delta|linear|action_error_top10|7.850186824798584|-0.1279747486114502|0.9572795033454895||||
|action_only|delta|linear|high_residual_top20|3.992807149887085|0.6007741093635559|0.9787328839302063||||
|action_only|delta|linear|high_residual_top10|7.850186824798584|-0.1279747486114502|0.9572795033454895||||
|action_only|delta|linear|large_action_small_disp|4.868738651275635|-0.5099008083343506|0.9689277410507202||||
|action_only|delta|mlp|all_test|3.1344945430755615|0.7920833230018616|0.8648030757904053||||
|action_only|delta|mlp|action_error_top20|6.770258903503418|0.5165055990219116|0.7649365067481995||||
|action_only|delta|mlp|action_error_top10|9.464415550231934|-0.121712327003479|0.7755905985832214||||
|action_only|delta|mlp|high_residual_top20|6.770258903503418|0.5165055990219116|0.7649365067481995||||
|action_only|delta|mlp|high_residual_top10|9.464415550231934|-0.121712327003479|0.7755905985832214||||
|action_only|delta|mlp|large_action_small_disp|6.078220367431641|-0.40031158924102783|0.778338611125946||||
|action_only|residual|linear|all_test|0.8586003184318542|-3.1113624572753906e-05|-8.976459503173828e-05||||
|action_only|residual|linear|action_error_top20|3.9925155639648438|-0.0002595186233520508|-0.0055124759674072266||||
|action_only|residual|linear|action_error_top10|7.84941291809082|-0.00042116641998291016|-0.011784076690673828||||
|action_only|residual|linear|high_residual_top20|3.9925155639648438|-0.0002595186233520508|-0.0055124759674072266||||
|action_only|residual|linear|high_residual_top10|7.84941291809082|-0.00042116641998291016|-0.011784076690673828||||
|action_only|residual|linear|large_action_small_disp|4.867892742156982|-0.00019431114196777344|-0.004806876182556152||||
|action_only|residual|mlp|all_test|0.8579976558685303|0.0007119178771972656|-0.0002071857452392578||||
|action_only|residual|mlp|action_error_top20|3.996086835861206|-0.001249074935913086|-0.004575014114379883||||
|action_only|residual|mlp|action_error_top10|7.845913887023926|-1.1324882507324219e-05|-0.010630488395690918||||
|action_only|residual|mlp|high_residual_top20|3.996086835861206|-0.001249074935913086|-0.004575014114379883||||
|action_only|residual|mlp|high_residual_top10|7.845913887023926|-1.1324882507324219e-05|-0.010630488395690918||||
|action_only|residual|mlp|large_action_small_disp|4.864312171936035|0.0005211234092712402|-0.003693103790283203||||
|action_only|h_next|linear|all_test||||0.04288192838430405|||
|action_only|h_next|linear|action_error_top20||||-0.04167364910244942|||
|action_only|h_next|linear|action_error_top10||||-0.47710660099983215|||
|action_only|h_next|linear|high_residual_top20||||-0.04167364910244942|||
|action_only|h_next|linear|high_residual_top10||||-0.47710660099983215|||
|action_only|h_next|linear|large_action_small_disp||||-0.1778143346309662|||
|action_only|h_next|mlp|all_test||||0.02711866796016693|||
|action_only|h_next|mlp|action_error_top20||||-0.05928707867860794|||
|action_only|h_next|mlp|action_error_top10||||-0.4877689778804779|||
|action_only|h_next|mlp|high_residual_top20||||-0.05928707867860794|||
|action_only|h_next|mlp|high_residual_top10||||-0.4877689778804779|||
|action_only|h_next|mlp|large_action_small_disp||||-0.19736069440841675|||
|action_only|delta_h|linear|all_test||||0.019750256091356277|||
|action_only|delta_h|linear|action_error_top20||||0.006439119577407837|||
|action_only|delta_h|linear|action_error_top10||||-0.015354696661233902|||
|action_only|delta_h|linear|high_residual_top20||||0.006439119577407837|||
|action_only|delta_h|linear|high_residual_top10||||-0.015354696661233902|||
|action_only|delta_h|linear|large_action_small_disp||||0.006792135536670685|||
|action_only|delta_h|mlp|all_test||||0.01861850917339325|||
|action_only|delta_h|mlp|action_error_top20||||0.005415927618741989|||
|action_only|delta_h|mlp|action_error_top10||||-0.014649759978055954|||
|action_only|delta_h|mlp|high_residual_top20||||0.005415927618741989|||
|action_only|delta_h|mlp|high_residual_top10||||-0.014649759978055954|||
|action_only|delta_h|mlp|large_action_small_disp||||0.005666155368089676|||
|action_only|hard_labels|linear|all_test|||||0.505612188577652|0.10319168676525961|
|action_only|hard_labels|mlp|all_test|||||0.6761267423629761|0.24014100237875544|
|h_action|delta|linear|all_test|0.5895889401435852|0.9383858442306519|0.9963628053665161||||
|h_action|delta|linear|action_error_top20|2.134845733642578|0.7907103896141052|0.9843435883522034||||
|h_action|delta|linear|action_error_top10|4.086558818817139|0.4247422218322754|0.9691013693809509||||
|h_action|delta|linear|high_residual_top20|2.134845733642578|0.7907103896141052|0.9843435883522034||||
|h_action|delta|linear|high_residual_top10|4.086558818817139|0.4247422218322754|0.9691013693809509||||
|h_action|delta|linear|large_action_small_disp|2.055114507675171|0.38892805576324463|0.9769750237464905||||
|h_action|delta|mlp|all_test|0.5714344382286072|0.9520309567451477|0.9850988984107971||||
|h_action|delta|mlp|action_error_top20|1.379456639289856|0.8744552135467529|0.9799144268035889||||
|h_action|delta|mlp|action_error_top10|2.3372104167938232|0.6819784641265869|0.9743558168411255||||
|h_action|delta|mlp|high_residual_top20|1.379456639289856|0.8744552135467529|0.9799144268035889||||
|h_action|delta|mlp|high_residual_top10|2.3372104167938232|0.6819784641265869|0.9743558168411255||||
|h_action|delta|mlp|large_action_small_disp|0.6837363243103027|0.832714855670929|0.978749692440033||||
|h_action|residual|linear|all_test|0.5895897746086121|0.32025665044784546|0.17453104257583618||||
|h_action|residual|linear|action_error_top20|2.134857654571533|0.4756256341934204|0.2592095136642456||||
|h_action|residual|linear|action_error_top10|4.08649206161499|0.48979902267456055|0.26758497953414917||||
|h_action|residual|linear|high_residual_top20|2.134857654571533|0.4756256341934204|0.2592095136642456||||
|h_action|residual|linear|high_residual_top10|4.08649206161499|0.48979902267456055|0.26758497953414917||||
|h_action|residual|linear|large_action_small_disp|2.0548970699310303|0.5952520370483398|0.2547590136528015||||
|h_action|residual|mlp|all_test|0.3298569917678833|0.6261810064315796|0.4091951847076416||||
|h_action|residual|mlp|action_error_top20|1.0496822595596313|0.747913122177124|0.5244767665863037||||
|h_action|residual|mlp|action_error_top10|1.9653137922286987|0.7603045701980591|0.537950873374939||||
|h_action|residual|mlp|high_residual_top20|1.0496822595596313|0.747913122177124|0.5244767665863037||||
|h_action|residual|mlp|high_residual_top10|1.9653137922286987|0.7603045701980591|0.537950873374939||||
|h_action|residual|mlp|large_action_small_disp|0.6507176160812378|0.8847253918647766|0.5269564390182495||||
|h_action|h_next|linear|all_test||||0.6531738638877869|||
|h_action|h_next|linear|action_error_top20||||0.6785727739334106|||
|h_action|h_next|linear|action_error_top10||||0.6055081486701965|||
|h_action|h_next|linear|high_residual_top20||||0.6785727739334106|||
|h_action|h_next|linear|high_residual_top10||||0.6055081486701965|||
|h_action|h_next|linear|large_action_small_disp||||0.7133491039276123|||
|h_action|h_next|mlp|all_test||||0.7697052955627441|||
|h_action|h_next|mlp|action_error_top20||||0.8019784688949585|||
|h_action|h_next|mlp|action_error_top10||||0.7826145887374878|||
|h_action|h_next|mlp|high_residual_top20||||0.8019784688949585|||
|h_action|h_next|mlp|high_residual_top10||||0.7826145887374878|||
|h_action|h_next|mlp|large_action_small_disp||||0.8135092854499817|||
|h_action|delta_h|linear|all_test||||0.30880963802337646|||
|h_action|delta_h|linear|action_error_top20||||0.16251124441623688|||
|h_action|delta_h|linear|action_error_top10||||-0.038560472428798676|||
|h_action|delta_h|linear|high_residual_top20||||0.16251124441623688|||
|h_action|delta_h|linear|high_residual_top10||||-0.038560472428798676|||
|h_action|delta_h|linear|large_action_small_disp||||-0.04512682557106018|||
|h_action|delta_h|mlp|all_test||||0.5503928661346436|||
|h_action|delta_h|mlp|action_error_top20||||0.5143727660179138|||
|h_action|delta_h|mlp|action_error_top10||||0.46407538652420044|||
|h_action|delta_h|mlp|high_residual_top20||||0.5143727660179138|||
|h_action|delta_h|mlp|high_residual_top10||||0.46407538652420044|||
|h_action|delta_h|mlp|large_action_small_disp||||0.40538010001182556|||
|h_action|hard_labels|linear|all_test|||||0.7336256504058838|0.4128227388851185|
|h_action|hard_labels|mlp|all_test|||||0.9126852869987487|0.5521645475246414|
|q_action|delta|linear|all_test|0.6464666724205017|0.9323626756668091|0.9960886240005493||||
|q_action|delta|linear|action_error_top20|2.511718511581421|0.7529582977294922|0.9824082255363464||||
|q_action|delta|linear|action_error_top10|4.850465297698975|0.31488364934921265|0.9650130271911621||||
|q_action|delta|linear|high_residual_top20|2.511718511581421|0.7529582977294922|0.9824082255363464||||
|q_action|delta|linear|high_residual_top10|4.850465297698975|0.31488364934921265|0.9650130271911621||||
|q_action|delta|linear|large_action_small_disp|2.5063283443450928|0.24920529127120972|0.9740167260169983||||
|q_action|delta|mlp|all_test|0.7009938359260559|0.946154773235321|0.9768791198730469||||
|q_action|delta|mlp|action_error_top20|1.719794750213623|0.8574546575546265|0.9605832695960999||||
|q_action|delta|mlp|action_error_top10|2.9401378631591797|0.637703537940979|0.9403286576271057||||
|q_action|delta|mlp|high_residual_top20|1.719794750213623|0.8574546575546265|0.9605832695960999||||
|q_action|delta|mlp|high_residual_top10|2.9401378631591797|0.637703537940979|0.9403286576271057||||
|q_action|delta|mlp|large_action_small_disp|0.988437294960022|0.8054232597351074|0.9514491558074951||||
|q_action|residual|linear|all_test|0.6464666724205017|0.2538068890571594|0.11234176158905029||||
|q_action|residual|linear|action_error_top20|2.5117483139038086|0.38103044033050537|0.1676122546195984||||
|q_action|residual|linear|action_error_top10|4.850429058074951|0.3923547863960266|0.1706472635269165||||
|q_action|residual|linear|high_residual_top20|2.5117483139038086|0.38103044033050537|0.1676122546195984||||
|q_action|residual|linear|high_residual_top10|4.850429058074951|0.3923547863960266|0.1706472635269165||||
|q_action|residual|linear|large_action_small_disp|2.506174087524414|0.5026645064353943|0.159121572971344||||
|q_action|residual|mlp|all_test|0.3628973066806793|0.5872660279273987|0.3793148398399353||||
|q_action|residual|mlp|action_error_top20|1.24930739402771|0.6978349685668945|0.4754677414894104||||
|q_action|residual|mlp|action_error_top10|2.366708278656006|0.7091991901397705|0.48518478870391846||||
|q_action|residual|mlp|high_residual_top20|1.24930739402771|0.6978349685668945|0.4754677414894104||||
|q_action|residual|mlp|high_residual_top10|2.366708278656006|0.7091991901397705|0.48518478870391846||||
|q_action|residual|mlp|large_action_small_disp|0.8894811868667603|0.8371999263763428|0.44949740171432495||||
|q_action|h_next|linear|all_test||||0.6315220594406128|||
|q_action|h_next|linear|action_error_top20||||0.6583857536315918|||
|q_action|h_next|linear|action_error_top10||||0.5769167542457581|||
|q_action|h_next|linear|high_residual_top20||||0.6583857536315918|||
|q_action|h_next|linear|high_residual_top10||||0.5769167542457581|||
|q_action|h_next|linear|large_action_small_disp||||0.6901562809944153|||
|q_action|h_next|mlp|all_test||||0.7445112466812134|||
|q_action|h_next|mlp|action_error_top20||||0.7822334170341492|||
|q_action|h_next|mlp|action_error_top10||||0.7631053924560547|||
|q_action|h_next|mlp|high_residual_top20||||0.7822334170341492|||
|q_action|h_next|mlp|high_residual_top10||||0.7631053924560547|||
|q_action|h_next|mlp|large_action_small_disp||||0.7987642884254456|||
|q_action|delta_h|linear|all_test||||0.24809299409389496|||
|q_action|delta_h|linear|action_error_top20||||0.11181849241256714|||
|q_action|delta_h|linear|action_error_top10||||-0.0756264179944992|||
|q_action|delta_h|linear|high_residual_top20||||0.11181849241256714|||
|q_action|delta_h|linear|high_residual_top10||||-0.0756264179944992|||
|q_action|delta_h|linear|large_action_small_disp||||-0.09790247678756714|||
|q_action|delta_h|mlp|all_test||||0.4780755341053009|||
|q_action|delta_h|mlp|action_error_top20||||0.42787861824035645|||
|q_action|delta_h|mlp|action_error_top10||||0.36636263132095337|||
|q_action|delta_h|mlp|high_residual_top20||||0.42787861824035645|||
|q_action|delta_h|mlp|high_residual_top10||||0.36636263132095337|||
|q_action|delta_h|mlp|large_action_small_disp||||0.31237906217575073|||
|q_action|hard_labels|linear|all_test|||||0.7371000289916992|0.41309393025182567|
|q_action|hard_labels|mlp|all_test|||||0.9074273347854614|0.5492894225635098|
|h_q_action|delta|linear|all_test|0.5825316309928894|0.9390001893043518|0.9965255856513977||||
|h_q_action|delta|linear|action_error_top20|2.100421905517578|0.793304443359375|0.9853991866111755||||
|h_q_action|delta|linear|action_error_top10|4.021421909332275|0.43153947591781616|0.9713156223297119||||
|h_q_action|delta|linear|high_residual_top20|2.100421905517578|0.793304443359375|0.9853991866111755||||
|h_q_action|delta|linear|high_residual_top10|4.021421909332275|0.43153947591781616|0.9713156223297119||||
|h_q_action|delta|linear|large_action_small_disp|1.948390007019043|0.41994065046310425|0.9784427881240845||||
|h_q_action|delta|mlp|all_test|0.5868015289306641|0.9515613317489624|0.9839038252830505||||
|h_q_action|delta|mlp|action_error_top20|1.3528813123703003|0.8790668845176697|0.9780452251434326||||
|h_q_action|delta|mlp|action_error_top10|2.2896106243133545|0.6939802765846252|0.9708678126335144||||
|h_q_action|delta|mlp|high_residual_top20|1.3528813123703003|0.8790668845176697|0.9780452251434326||||
|h_q_action|delta|mlp|high_residual_top10|2.2896106243133545|0.6939802765846252|0.9708678126335144||||
|h_q_action|delta|mlp|large_action_small_disp|0.7205452919006348|0.8260982632637024|0.9767042398452759||||
|h_q_action|residual|linear|all_test|0.5825234651565552|0.3270440101623535|0.2114970088005066||||
|h_q_action|residual|linear|action_error_top20|2.1004390716552734|0.48212432861328125|0.3090552091598511||||
|h_q_action|residual|linear|action_error_top10|4.021362781524658|0.49582719802856445|0.3199526071548462||||
|h_q_action|residual|linear|high_residual_top20|2.1004390716552734|0.48212432861328125|0.3090552091598511||||
|h_q_action|residual|linear|high_residual_top10|4.021362781524658|0.49582719802856445|0.3199526071548462||||
|h_q_action|residual|linear|large_action_small_disp|1.9480321407318115|0.6158299446105957|0.3021748661994934||||
|h_q_action|residual|mlp|all_test|0.320722758769989|0.6331659555435181|0.4926060438156128||||
|h_q_action|residual|mlp|action_error_top20|1.010923981666565|0.7540410757064819|0.6036748886108398||||
|h_q_action|residual|mlp|action_error_top10|1.8882131576538086|0.7665913701057434|0.6163815259933472||||
|h_q_action|residual|mlp|high_residual_top20|1.010923981666565|0.7540410757064819|0.6036748886108398||||
|h_q_action|residual|mlp|high_residual_top10|1.8882131576538086|0.7665913701057434|0.6163815259933472||||
|h_q_action|residual|mlp|large_action_small_disp|0.6587852239608765|0.8786124587059021|0.6071900129318237||||
|h_q_action|h_next|linear|all_test||||0.6720461249351501|||
|h_q_action|h_next|linear|action_error_top20||||0.7004910111427307|||
|h_q_action|h_next|linear|action_error_top10||||0.6302943229675293|||
|h_q_action|h_next|linear|high_residual_top20||||0.7004910111427307|||
|h_q_action|h_next|linear|high_residual_top10||||0.6302943229675293|||
|h_q_action|h_next|linear|large_action_small_disp||||0.7121158242225647|||
|h_q_action|h_next|mlp|all_test||||0.7762791514396667|||
|h_q_action|h_next|mlp|action_error_top20||||0.8074753284454346|||
|h_q_action|h_next|mlp|action_error_top10||||0.7878085970878601|||
|h_q_action|h_next|mlp|high_residual_top20||||0.8074753284454346|||
|h_q_action|h_next|mlp|high_residual_top10||||0.7878085970878601|||
|h_q_action|h_next|mlp|large_action_small_disp||||0.8196238875389099|||
|h_q_action|delta_h|linear|all_test||||0.34383532404899597|||
|h_q_action|delta_h|linear|action_error_top20||||0.21682167053222656|||
|h_q_action|delta_h|linear|action_error_top10||||0.0240032821893692|||
|h_q_action|delta_h|linear|high_residual_top20||||0.21682167053222656|||
|h_q_action|delta_h|linear|high_residual_top10||||0.0240032821893692|||
|h_q_action|delta_h|linear|large_action_small_disp||||-0.05304597318172455|||
|h_q_action|delta_h|mlp|all_test||||0.5670982003211975|||
|h_q_action|delta_h|mlp|action_error_top20||||0.5287611484527588|||
|h_q_action|delta_h|mlp|action_error_top10||||0.4722910225391388|||
|h_q_action|delta_h|mlp|high_residual_top20||||0.5287611484527588|||
|h_q_action|delta_h|mlp|high_residual_top10||||0.4722910225391388|||
|h_q_action|delta_h|mlp|large_action_small_disp||||0.4125065207481384|||
|h_q_action|hard_labels|linear|all_test|||||0.7466477990150452|0.41782865835046307|
|h_q_action|hard_labels|mlp|all_test|||||0.9155881643295288|0.5697046738543817|
|z_action_teacher|delta|linear|all_test|0.5939823985099792|0.9378831386566162|0.9963778853416443||||
|z_action_teacher|delta|linear|action_error_top20|2.1869194507598877|0.7852046489715576|0.984373927116394||||
|z_action_teacher|delta|linear|action_error_top10|4.203769207000732|0.40693938732147217|0.9691612720489502||||
|z_action_teacher|delta|linear|high_residual_top20|2.1869194507598877|0.7852046489715576|0.984373927116394||||
|z_action_teacher|delta|linear|high_residual_top10|4.203769207000732|0.40693938732147217|0.9691612720489502||||
|z_action_teacher|delta|linear|large_action_small_disp|1.8876981735229492|0.44387370347976685|0.976901650428772||||
|z_action_teacher|delta|mlp|all_test|0.32882723212242126|0.9683130383491516|0.9953795671463013||||
|z_action_teacher|delta|mlp|action_error_top20|1.0771619081497192|0.8983763456344604|0.9880101084709167||||
|z_action_teacher|delta|mlp|action_error_top10|2.0213165283203125|0.7234939932823181|0.9788872599601746||||
|z_action_teacher|delta|mlp|high_residual_top20|1.0771619081497192|0.8983763456344604|0.9880101084709167||||
|z_action_teacher|delta|mlp|high_residual_top10|2.0213165283203125|0.7234939932823181|0.9788872599601746||||
|z_action_teacher|delta|mlp|large_action_small_disp|0.4637244641780853|0.8871771097183228|0.9853485822677612||||
|z_action_teacher|residual|linear|all_test|0.59397953748703|0.314713716506958|0.17798089981079102||||
|z_action_teacher|residual|linear|action_error_top20|2.186887741088867|0.46184414625167847|0.2605709433555603||||
|z_action_teacher|residual|linear|action_error_top10|4.203606605529785|0.4740234613418579|0.26891791820526123||||
|z_action_teacher|residual|linear|high_residual_top20|2.186887741088867|0.46184414625167847|0.2605709433555603||||
|z_action_teacher|residual|linear|high_residual_top10|4.203606605529785|0.4740234613418579|0.26891791820526123||||
|z_action_teacher|residual|linear|large_action_small_disp|1.887470006942749|0.6316680312156677|0.2522745132446289||||
|z_action_teacher|residual|mlp|all_test|0.2681210935115814|0.6960217952728271|0.5222184658050537||||
|z_action_teacher|residual|mlp|action_error_top20|0.8854632377624512|0.7858260273933411|0.6284335851669312||||
|z_action_teacher|residual|mlp|action_error_top10|1.6751327514648438|0.7938848733901978|0.6412147283554077||||
|z_action_teacher|residual|mlp|high_residual_top20|0.8854632377624512|0.7858260273933411|0.6284335851669312||||
|z_action_teacher|residual|mlp|high_residual_top10|1.6751327514648438|0.7938848733901978|0.6412147283554077||||
|z_action_teacher|residual|mlp|large_action_small_disp|0.37305158376693726|0.9399188756942749|0.6184235215187073||||
|z_action_teacher|h_next|linear|all_test||||0.7115768194198608|||
|z_action_teacher|h_next|linear|action_error_top20||||0.7284806966781616|||
|z_action_teacher|h_next|linear|action_error_top10||||0.6435953378677368|||
|z_action_teacher|h_next|linear|high_residual_top20||||0.7284806966781616|||
|z_action_teacher|h_next|linear|high_residual_top10||||0.6435953378677368|||
|z_action_teacher|h_next|linear|large_action_small_disp||||0.7276641130447388|||
|z_action_teacher|h_next|mlp|all_test||||0.9029451608657837|||
|z_action_teacher|h_next|mlp|action_error_top20||||0.9194856286048889|||
|z_action_teacher|h_next|mlp|action_error_top10||||0.907427191734314|||
|z_action_teacher|h_next|mlp|high_residual_top20||||0.9194856286048889|||
|z_action_teacher|h_next|mlp|high_residual_top10||||0.907427191734314|||
|z_action_teacher|h_next|mlp|large_action_small_disp||||0.9130755066871643|||
|z_action_teacher|delta_h|linear|all_test||||0.41688278317451477|||
|z_action_teacher|delta_h|linear|action_error_top20||||0.2998967170715332|||
|z_action_teacher|delta_h|linear|action_error_top10||||0.09615778177976608|||
|z_action_teacher|delta_h|linear|high_residual_top20||||0.2998967170715332|||
|z_action_teacher|delta_h|linear|high_residual_top10||||0.09615778177976608|||
|z_action_teacher|delta_h|linear|large_action_small_disp||||0.014653030782938004|||
|z_action_teacher|delta_h|mlp|all_test||||0.8102624416351318|||
|z_action_teacher|delta_h|mlp|action_error_top20||||0.8029427528381348|||
|z_action_teacher|delta_h|mlp|action_error_top10||||0.7697857618331909|||
|z_action_teacher|delta_h|mlp|high_residual_top20||||0.8029427528381348|||
|z_action_teacher|delta_h|mlp|high_residual_top10||||0.7697857618331909|||
|z_action_teacher|delta_h|mlp|large_action_small_disp||||0.7027373313903809|||
|z_action_teacher|hard_labels|linear|all_test|||||0.7529670834541321|0.42741816648656067|
|z_action_teacher|hard_labels|mlp|all_test|||||0.9221143126487732|0.6478666696145827|

## Residual Gap Closure

|input|residual_gap_closure_vs_action_to_zteacher|
|---|---|
|q_action|0.8393|
|h_action|0.8953|
|h_q_action|0.9108|

## Answers

- `q+action` should be read against `action_only`, `h+action`, and `z+action_teacher`.
- If `q+action` is clearly above `action_only` and close to `h+action`, q keeps transition-critical information.
- If `q+action` is much weaker than `h+action`, q is interpretable but lossy and should not replace h as the only prediction input yet.


---

# V4 Exported KANFIS Rules

Best q model: `q_pos_delta_residual_hard_hnext_R16`.

Raw rule table saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/rules/raw_rule_table.csv`.

## Top Raw Rules

|rule_id|antecedent|usage_mean|top_hard_subset|top_hard_enrichment|delta_effect_x|delta_effect_y|residual_effect_x|residual_effect_y|
|---|---|---|---|---|---|---|---|---|
|2|h_9 is High AND h_10 is Low AND h_8 is High AND h_0 is Low|0.05039|action_error_top10|3.5575|-0.95133|-2.42228|-1.0043|0.46317|
|14|h_3 is High AND h_1 is High AND h_2 is Low AND h_7 is Low|0.05385|action_error_top10|2.314|4.28554|-2.92427|14.7951|-0.3956|
|15|h_15 is High AND h_13 is Low AND h_10 is High AND h_4 is High|0.03273|action_error_top10|3.4184|-5.97644|0.9573|-15.89247|0.52045|
|12|h_8 is Low AND h_15 is High AND h_1 is High AND h_2 is Low|0.08594|action_error_top20|0.9819|-1.01816|-0.3137|-1.17492|0.18432|
|13|h_3 is High AND h_1 is High AND h_2 is Low AND h_14 is High|0.06075|action_error_top10|1.1031|4.49734|0.40711|9.84797|-0.04682|
|10|h_7 is High AND h_6 is High AND h_5 is Low AND h_9 is Low|0.07242|action_error_top20|0.8725|-1.42818|-0.87079|-0.18493|0.00945|
|3|h_5 is Low AND h_9 is Low AND h_14 is High AND h_1 is Low|0.07629|action_error_top20|0.804|-2.93677|0.75644|-2.644|0.18133|
|5|h_14 is High AND h_11 is Low AND h_0 is High AND h_3 is Low|0.07441|action_error_top20|0.8016|-3.02579|0.304|-3.0171|0.06813|
|1|h_6 is High AND h_9 is Low AND h_5 is Low AND h_4 is High|0.0617|action_error_top20|0.8956|-0.0333|0.75978|3.24027|0.00157|
|9|h_11 is High AND h_14 is Low AND h_8 is Low AND h_4 is Low|0.05848|large_action_small_disp|0.9324|-0.99207|-1.9957|-1.02972|-0.15248|
|11|h_7 is High AND h_15 is Low AND h_10 is High AND h_4 is Low|0.06802|action_error_top20|0.7801|-1.79181|-1.27189|-0.71714|0.03423|
|0|h_8 is Low AND h_11 is High AND h_13 is High AND h_14 is Low|0.06035|large_action_small_disp|0.8704|0.66417|-0.95617|-1.00748|-0.14153|

## Candidate Physical Interpretation

|rule_id|possible_meaning|evidence|confidence|
|---|---|---|---|
|2|hard-motion correction factor|usage=0.05039, enriched in action_error_top10 by 3.5575x, residual effect=(-1.0043, 0.46317)|candidate|
|14|hard-motion correction factor|usage=0.05385, enriched in action_error_top10 by 2.314x, residual effect=(14.7951, -0.3956)|candidate|
|15|hard-motion correction factor|usage=0.03273, enriched in action_error_top10 by 3.4184x, residual effect=(-15.89247, 0.52045)|candidate|
|12|state-partition factor|usage=0.08594, enriched in action_error_top20 by 0.9819x, residual effect=(-1.17492, 0.18432)|candidate|
|13|state-partition factor|usage=0.06075, enriched in action_error_top10 by 1.1031x, residual effect=(9.84797, -0.04682)|candidate|
|10|state-partition factor|usage=0.07242, enriched in action_error_top20 by 0.8725x, residual effect=(-0.18493, 0.00945)|candidate|
|3|state-partition factor|usage=0.07629, enriched in action_error_top20 by 0.804x, residual effect=(-2.644, 0.18133)|candidate|
|5|state-partition factor|usage=0.07441, enriched in action_error_top20 by 0.8016x, residual effect=(-3.0171, 0.06813)|candidate|
|1|state-partition factor|usage=0.0617, enriched in action_error_top20 by 0.8956x, residual effect=(3.24027, 0.00157)|candidate|
|9|state-partition factor|usage=0.05848, enriched in large_action_small_disp by 0.9324x, residual effect=(-1.02972, -0.15248)|candidate|
|11|state-partition factor|usage=0.06802, enriched in action_error_top20 by 0.7801x, residual effect=(-0.71714, 0.03423)|candidate|
|0|state-partition factor|usage=0.06035, enriched in large_action_small_disp by 0.8704x, residual effect=(-1.00748, -0.14153)|candidate|

## Caution

These are post-hoc interpretations. A rule is not named as an obstacle or doorway concept unless hard-subset enrichment, residual effect, and ablation all support it.


---

# V4 Rule And q Ablation

Best q model: `q_pos_delta_residual_hard_hnext_R16`.

## All-Test Ablation

|mask|rules|residual_mse|residual_mse_increase|delta_mse|delta_mse_increase|hnext_mse|hnext_mse_increase|hard_auc|hard_f1|
|---|---|---|---|---|---|---|---|---|---|
|none||0.7731637954711914|0.0|0.9598051905632019|0.0|0.28313496708869934|0.0|0.8310387372970581|0.4099653278008526|
|top_usage|12,3,5,10|0.7936097383499146|0.020445942878723145|0.9671851396560669|0.00737994909286499|1.8256527185440063|1.542517751455307|0.8467088937759399|0.43203687080162023|
|top_hard_enriched|2,15,14,13|0.8726325631141663|0.09946876764297485|1.0012556314468384|0.041450440883636475|0.7974365949630737|0.5143016278743744|0.7375025987625122|0.2428076736691301|
|top_residual_effect|15,14,13,1|0.8866499662399292|0.11348617076873779|0.9981177449226379|0.038312554359436035|0.44072115421295166|0.15758618712425232|0.8215697646141052|0.36116369704087153|
|random|15,4,14,9|0.8189355731010437|0.045771777629852295|0.9835954308509827|0.02379024028778076|0.8553423285484314|0.5722073614597321|0.8204071879386902|0.36890500106832613|

## Hard-Subset Ablation

|subset|mask|residual_mse_increase|delta_mse_increase|
|---|---|---|---|
|action_error_top20|top_usage|-0.08913373947143555|-0.07145428657531738|
|action_error_top20|top_hard_enriched|0.5577783584594727|0.23864269256591797|
|action_error_top20|top_residual_effect|0.340609073638916|0.18696379661560059|
|action_error_top20|random|0.25979161262512207|0.1326732635498047|
|action_error_top10|top_usage|-0.2314314842224121|-0.19330120086669922|
|action_error_top10|top_hard_enriched|1.1554450988769531|0.49710893630981445|
|action_error_top10|top_residual_effect|0.6485700607299805|0.3782172203063965|
|action_error_top10|random|0.5325307846069336|0.2741241455078125|
|high_residual_top20|top_usage|-0.08913373947143555|-0.07145428657531738|
|high_residual_top20|top_hard_enriched|0.5577783584594727|0.23864269256591797|
|high_residual_top20|top_residual_effect|0.340609073638916|0.18696379661560059|
|high_residual_top20|random|0.25979161262512207|0.1326732635498047|
|high_residual_top10|top_usage|-0.2314314842224121|-0.19330120086669922|
|high_residual_top10|top_hard_enriched|1.1554450988769531|0.49710893630981445|
|high_residual_top10|top_residual_effect|0.6485700607299805|0.3782172203063965|
|high_residual_top10|random|0.5325307846069336|0.2741241455078125|
|large_action_small_disp|top_usage|-0.16375041007995605|-0.10813784599304199|
|large_action_small_disp|top_hard_enriched|0.8972864151000977|0.3274385929107666|
|large_action_small_disp|top_residual_effect|0.4863772392272949|0.24120068550109863|
|large_action_small_disp|random|0.3748178482055664|0.167464017868042|

## Answers

- If top rule masks increase residual or hard-subset MSE more than random masks, the rules have functional contribution.
- If random and top masks are similar, q may still compress h but rule identity is not yet functionally stable.


---

# V4 q Semantics

Best q model: `q_pos_delta_residual_hard_hnext_R16`.

Full q-target correlation matrix saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/q_target_corr_matrix.csv`.

## Top q Semantic Alignments

|q_dim|top_target|pearson|abs_pearson|posthoc_label|
|---|---|---|---|---|
|7|position_x|0.8796|0.8796|position-related factor|
|9|position_y|0.8474|0.8474|position-related factor|
|6|position_x|-0.8438|0.8438|position-related factor|
|0|position_y|0.8273|0.8273|position-related factor|
|3|position_y|-0.7976|0.7976|position-related factor|
|5|position_x|-0.7832|0.7832|position-related factor|
|10|position_y|-0.7752|0.7752|position-related factor|
|8|position_x|0.7084|0.7084|position-related factor|
|4|position_x|-0.6718|0.6718|position-related factor|
|1|position_y|-0.6468|0.6468|position-related factor|
|12|position_y|-0.5871|0.5871|unknown or mixed factor|
|11|position_y|-0.4954|0.4954|unknown or mixed factor|
|2|hard_action_error_top10|0.4664|0.4664|hard-motion factor|
|13|position_x|0.4101|0.4101|unknown or mixed factor|
|15|hard_action_error_top10|0.3697|0.3697|hard-motion factor|
|14|hard_action_error_top10|0.2948|0.2948|hard-motion factor|

## Answers

- q dimensions are post-hoc labeled only when correlation evidence is strong enough.
- Weak or mixed q dimensions should remain unnamed; they may still be useful structural rules.
- q should not be called x/y directly unless the evidence is very strong and ablation supports it.
