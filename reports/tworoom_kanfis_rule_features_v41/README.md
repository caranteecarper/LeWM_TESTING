# TwoRoom V4.1 White-Box-Preserving KANFIS q Strengthening

## Executive Summary

V4.1 sharpens `q_t = KANFIS_rule_features(h_t)` with temperature annealing, top-k rule gating, entropy, usage balance, and rule diversity. It does not modify LeWM, does not connect to the predictor, and does not use q->h reconstruction or h distillation.

The final judgment should compare V4.1 to V4 on residual MSE, hard-subset residual MSE, gap closure, active rule count, and rule ablation strength.


---

# V4.1 Input Check

## Git

- branch: `exp/tworoom-official-encoder-readout`
- commit: `4f42a43e222ea8ea1d80975204e818af0d4bb335`
- log: `4f42a43 Add TwoRoom pre-integration analysis pipeline`

Execution note: the server clean clone still reports an older local HEAD because it was not fast-forwarded before this run. The V4.1 scripts used here were synchronized from local/GitHub commit `3f0e9e7 Add V4.1 white-box KANFIS q strengthening`.

## Scope

V4.1 strengthens `q_t = KANFIS_rule_features(h_t)` without modifying LeWM and without connecting to the predictor.

## Inputs

|item|value|exists|
|---|---|---|
|best_h_dataset|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/best_h_dataset.pt|True|
|h_t_shape|(720809, 16)|True|
|action_shape|(720809, 10)|True|
|position_shape|(720809, 2)|True|
|delta_shape|(720809, 2)|True|
|residual_shape|(720809, 2)|True|
|hard_labels_shape|(720809, 5)|True|
|train/val/test|576945/72141/71723|True|
|v4_best_q|q_pos_delta_residual_hard_hnext_R16|True|
|v4_best_checkpoint|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features/rule_feature_models/q_pos_delta_residual_hard_hnext_R16.pt|True|

## Constraint Check

- Action does not enter q extraction. q is produced from h only.
- This stage does not use q->h reconstruction, q->h_next reconstruction as a main loss, h mimicry, or MLP-hidden distillation.
- h_next / delta_h are not training losses in the main V4.1 variants.


---

# V4.1 Sharp q Training

Best V4.1 q model: `sharp_tau_R16`.

No q->h reconstruction, h mimicry, or MLP hidden distillation is used. h_next / delta_h are not used as training losses.

|variant|num_rules|top_k|tau_end|hard_weight|res_overall_mse|action_error_top10_res_mse|large_action_small_disp_res_mse|delta_overall_mse|pos_x_r2|pos_y_r2|hard_auc|hard_f1|q_entropy|active_rule_count|usage_min|usage_max|usage_entropy|center_min_distance|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|baseline_v4_reproduce|16|None|1.0|light|0.760982871055603|5.46574592590332|2.841115713119507|6.046706676483154|0.5614453554153442|0.4225621223449707|0.650470244884491|0.032927154868684395|2.143198013305664|4.8809027671813965|0.016077550128102303|0.1633179932832718|2.6183717250823975|0.6788238883018494|
|sharp_tau_R16|16|None|0.2|light|0.6727852821350098|4.882841110229492|2.4942681789398193|4.077075958251953|0.6511949300765991|0.6457587480545044|0.6855162024497986|0.19846013480949415|0.9822302460670471|2.6087448596954346|0.00027623469941318035|0.18492546677589417|2.359973430633545|0.5207082033157349|
|sharp_tau_topk3_R16|16|3|0.2|light|0.7568138837814331|4.987070560455322|2.5437240600585938|6.0698676109313965|0.688459038734436|0.5738338232040405|0.694217038154602|0.3427111357754868|0.6560923457145691|2.285919427871704|3.882442251779139e-05|0.21139302849769592|2.0879061222076416|0.7573006749153137|
|sharp_tau_topk4_R16|16|4|0.2|light|0.701028048992157|4.970098495483398|2.564117193222046|5.4680681228637695|0.5837750434875488|0.6172689199447632|0.7010053396224976|0.23041723968157635|0.8953432440757751|2.882687568664551|5.065994264441542e-05|0.15396331250667572|2.3046493530273438|0.5538146495819092|
|sharp_tau_topk3_hard_medium_R16|16|3|0.2|medium|0.8007550835609436|4.694546699523926|2.318310499191284|4.750671863555908|0.6476146578788757|0.49690598249435425|0.7053280949592591|0.3511379330802858|0.5436818599700928|1.9800621271133423|3.258178549003787e-05|0.277835875749588|1.8061063289642334|0.6111775040626526|
|sharp_tau_topk4_hard_medium_R16|16|4|0.2|medium|0.7340476512908936|4.984316825866699|2.5410752296447754|6.223606586456299|0.6601971387863159|0.5962164998054504|0.685577118396759|0.2913568607508521|0.8491964936256409|2.6939475536346436|1.9691671695909463e-05|0.1846831738948822|2.1730003356933594|0.763097882270813|
|sharp_tau_topk3_hard_strong_R16|16|3|0.2|strong|0.7790927290916443|4.864404201507568|2.4269120693206787|6.955224514007568|0.6619951128959656|0.44676506519317627|0.7071952104568482|0.3629799548315626|0.6944206357002258|2.33803653717041|3.874008325510658e-05|0.22832971811294556|2.0303092002868652|0.6041937470436096|
|sharp_tau_topk4_hard_strong_R16|16|4|0.2|strong|0.7548397183418274|5.219217300415039|2.727914810180664|5.730182647705078|0.6732255220413208|0.6337131261825562|0.7096405625343323|0.3653176193248883|0.7388776540756226|2.4010009765625|1.9064069419982843e-05|0.2280522584915161|2.1221346855163574|0.6710748076438904|
|sharp_tau_topk3_hard_medium_R24|24|3|0.2|medium|0.7840986251831055|4.827001571655273|2.3899941444396973|3.807657241821289|0.6549410820007324|0.6062314510345459|0.7233364105224609|0.3721060361560039|0.6588847041130066|2.4700722694396973|2.8371080134093063e-06|0.2030390352010727|2.2299437522888184|0.21642066538333893|
|sharp_tau_topk4_hard_medium_R24|24|4|0.2|medium|0.730663001537323|4.884976863861084|2.4425597190856934|4.9196648597717285|0.6692236065864563|0.5861597061157227|0.7078008651733398|0.369484586089801|1.0396902561187744|3.3947129249572754|5.49264677829342e-06|0.12996913492679596|2.654979705810547|0.027535226196050644|

## Notes

- `tau` is annealed from 1.0 to the listed `tau_end`.
- `top_k` variants keep only the top-k firing strengths and renormalize q.
- Entropy lowers per-sample active rule count; balance regularization keeps dataset-level rule usage from collapsing.
- Model files are saved under `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features_v41/sharp_q_models` and are not committed to git.


---

# V4 vs V4.1 q Evaluation

## Summary

|metric|value|
|---|---|
|v4_best|q_pos_delta_residual_hard_hnext_R16|
|v41_best|sharp_tau_R16|
|v4_active_rule_count|7.80605936050415|
|v41_active_rule_count|2.6087448596954346|
|v41_all_residual_mse_target|<=0.34|
|v41_action_error_top10_target|<=2.10|
|v41_large_action_small_disp_target|<=0.75|

## Residual Gap Closure

|input|residual_mse|residual_gap_closure|
|---|---|---|
|q_v4_action|0.33114564418792725|0.8416564784656003|
|q_v41_action|0.37577688694000244|0.7703134173862005|
|h_action|0.31345701217651367|0.8699317644801855|
|h_q_v41_action|0.26934918761253357|0.9404381413567552|

## Metrics

|input|task|subset|overall_mse|x_r2|y_r2|hard_auc|hard_f1|
|---|---|---|---|---|---|---|---|
|action_only|delta|all_test|3.07647705078125|0.7951762676239014|0.8680370450019836|||
|action_only|delta|action_error_top20|6.680119037628174|0.5193653702735901|0.7717465758323669|||
|action_only|delta|action_error_top10|9.408997535705566|-0.12079620361328125|0.7810072898864746|||
|action_only|delta|high_residual_top20|6.680119037628174|0.5193653702735901|0.7717465758323669|||
|action_only|delta|high_residual_top10|9.408997535705566|-0.12079620361328125|0.7810072898864746|||
|action_only|delta|large_action_small_disp|6.027553558349609|-0.3994406461715698|0.7842618227005005|||
|action_only|residual|all_test|0.8576744198799133|0.0011118650436401367|-0.00030052661895751953|||
|action_only|residual|action_error_top20|3.9787395000457764|0.0033472180366516113|-0.005056142807006836|||
|action_only|residual|action_error_top10|7.820668697357178|0.003401339054107666|-0.011154890060424805|||
|action_only|residual|high_residual_top20|3.9787395000457764|0.0033472180366516113|-0.005056142807006836|||
|action_only|residual|high_residual_top10|7.820668697357178|0.003401339054107666|-0.011154890060424805|||
|action_only|residual|large_action_small_disp|4.844427585601807|0.004865050315856934|-0.004335880279541016|||
|action_only|hard|all_test||||0.6781967639923095|0.26490420317343744|
|q_v4_action|delta|all_test|0.658778965473175|0.9489622116088867|0.978692889213562|||
|q_v4_action|delta|action_error_top20|1.618186354637146|0.8652125597000122|0.9635951519012451|||
|q_v4_action|delta|action_error_top10|2.7747457027435303|0.6566489934921265|0.9447268843650818|||
|q_v4_action|delta|high_residual_top20|1.618186354637146|0.8652125597000122|0.9635951519012451|||
|q_v4_action|delta|high_residual_top10|2.7747457027435303|0.6566489934921265|0.9447268843650818|||
|q_v4_action|delta|large_action_small_disp|0.9342666864395142|0.8107617497444153|0.956119179725647|||
|q_v4_action|residual|all_test|0.33114564418792725|0.6210042238235474|0.4809025526046753|||
|q_v4_action|residual|action_error_top20|1.0853612422943115|0.7353030443191528|0.586652934551239|||
|q_v4_action|residual|action_error_top10|2.040271043777466|0.7471562623977661|0.5978452563285828|||
|q_v4_action|residual|high_residual_top20|1.0853612422943115|0.7353030443191528|0.586652934551239|||
|q_v4_action|residual|high_residual_top10|2.040271043777466|0.7471562623977661|0.5978452563285828|||
|q_v4_action|residual|large_action_small_disp|0.7345505356788635|0.8646917343139648|0.5612826347351074|||
|q_v4_action|hard|all_test||||0.9124212741851807|0.5746073219133307|
|q_v41_action|delta|all_test|0.6712770462036133|0.942789614200592|0.9833282232284546|||
|q_v41_action|delta|action_error_top20|1.6647378206253052|0.8440840840339661|0.980294942855835|||
|q_v41_action|delta|action_error_top10|2.8899085521698|0.5978186726570129|0.9747915267944336|||
|q_v41_action|delta|high_residual_top20|1.6647378206253052|0.8440840840339661|0.980294942855835|||
|q_v41_action|delta|high_residual_top10|2.8899085521698|0.5978186726570129|0.9747915267944336|||
|q_v41_action|delta|large_action_small_disp|1.0024787187576294|0.7235838174819946|0.9805949330329895|||
|q_v41_action|residual|all_test|0.37577688694000244|0.5715422034263611|0.3787078261375427|||
|q_v41_action|residual|action_error_top20|1.3672723770141602|0.6671397089958191|0.46787571907043457|||
|q_v41_action|residual|action_error_top10|2.595625162124634|0.6788928508758545|0.4775514602661133|||
|q_v41_action|residual|high_residual_top20|1.3672723770141602|0.6671397089958191|0.46787571907043457|||
|q_v41_action|residual|high_residual_top10|2.595625162124634|0.6788928508758545|0.4775514602661133|||
|q_v41_action|residual|large_action_small_disp|1.0028730630874634|0.8114939332008362|0.4703494906425476|||
|q_v41_action|hard|all_test||||0.9144314169883728|0.5795189940485242|
|h_action|delta|all_test|0.530762255191803|0.9549211859703064|0.9866669178009033|||
|h_action|delta|action_error_top20|1.2607741355895996|0.8843737244606018|0.9825506806373596|||
|h_action|delta|action_error_top10|2.13374400138855|0.7079470157623291|0.9778345227241516|||
|h_action|delta|high_residual_top20|1.2607741355895996|0.8843737244606018|0.9825506806373596|||
|h_action|delta|high_residual_top10|2.13374400138855|0.7079470157623291|0.9778345227241516|||
|h_action|delta|large_action_small_disp|0.6099202036857605|0.8465621471405029|0.9826333522796631|||
|h_action|residual|all_test|0.31345701217651367|0.6422677040100098|0.48833900690078735|||
|h_action|residual|action_error_top20|0.9978105425834656|0.757642388343811|0.6008539795875549|||
|h_action|residual|action_error_top10|1.8661229610443115|0.7696589231491089|0.6143508553504944|||
|h_action|residual|high_residual_top20|0.9978105425834656|0.757642388343811|0.6008539795875549|||
|h_action|residual|high_residual_top10|1.8661229610443115|0.7696589231491089|0.6143508553504944|||
|h_action|residual|large_action_small_disp|0.6202133297920227|0.8873341679573059|0.6005108952522278|||
|h_action|hard|all_test||||0.9195507287979126|0.5867762722230272|
|h_q_v41_action|delta|all_test|0.5040305256843567|0.9574247002601624|0.9871127009391785|||
|h_q_v41_action|delta|action_error_top20|1.1568859815597534|0.8936602473258972|0.9842365980148315|||
|h_q_v41_action|delta|action_error_top10|1.9719514846801758|0.7296537756919861|0.9798333644866943|||
|h_q_v41_action|delta|high_residual_top20|1.1568859815597534|0.8936602473258972|0.9842365980148315|||
|h_q_v41_action|delta|high_residual_top10|1.9719514846801758|0.7296537756919861|0.9798333644866943|||
|h_q_v41_action|delta|large_action_small_disp|0.6077831983566284|0.840012788772583|0.9853681921958923|||
|h_q_v41_action|residual|all_test|0.26934918761253357|0.6944184303283691|0.5242332220077515|||
|h_q_v41_action|residual|action_error_top20|0.8594462871551514|0.7927212715148926|0.6276751160621643|||
|h_q_v41_action|residual|action_error_top10|1.6097736358642578|0.8027184009552002|0.639898955821991|||
|h_q_v41_action|residual|high_residual_top20|0.8594462871551514|0.7927212715148926|0.6276751160621643|||
|h_q_v41_action|residual|high_residual_top10|1.6097736358642578|0.8027184009552002|0.639898955821991|||
|h_q_v41_action|residual|large_action_small_disp|0.556934654712677|0.8987821936607361|0.6421331167221069|||
|h_q_v41_action|hard|all_test||||0.9284636735916137|0.6192869063428872|
|z_action_teacher|delta|all_test|0.2984561026096344|0.9705101251602173|0.996512770652771|||
|z_action_teacher|delta|action_error_top20|0.9772688150405884|0.9069610834121704|0.9899857044219971|||
|z_action_teacher|delta|action_error_top10|1.834725022315979|0.74775230884552|0.9817554950714111|||
|z_action_teacher|delta|high_residual_top20|0.9772688150405884|0.9069610834121704|0.9899857044219971|||
|z_action_teacher|delta|high_residual_top10|1.834725022315979|0.74775230884552|0.9817554950714111|||
|z_action_teacher|delta|large_action_small_disp|0.4289551377296448|0.8940348625183105|0.9870513677597046|||
|z_action_teacher|residual|all_test|0.232088103890419|0.7339664697647095|0.6443296670913696|||
|z_action_teacher|residual|action_error_top20|0.753089427947998|0.8148143887519836|0.7427136301994324|||
|z_action_teacher|residual|action_error_top10|1.4228312969207764|0.8218532800674438|0.7547667026519775|||
|z_action_teacher|residual|high_residual_top20|0.753089427947998|0.8148143887519836|0.7427136301994324|||
|z_action_teacher|residual|high_residual_top10|1.4228312969207764|0.8218532800674438|0.7547667026519775|||
|z_action_teacher|residual|large_action_small_disp|0.3229884207248688|0.9443937540054321|0.7355853319168091|||
|z_action_teacher|hard|all_test||||0.936367928981781|0.6409248417406692|


---

# V4.1 Exported Rules

Best V4.1 q model: `sharp_tau_R16`.

Rule table saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features_v41/rules/v41_rule_table.csv`.

|rule_id|antecedent|usage_mean|top_hard_subset|hard_enrichment|residual_x_effect|residual_y_effect|physical_posthoc_label|confidence|
|---|---|---|---|---|---|---|---|---|
|12|h_7 is Low AND h_2 is Low AND h_12 is High AND h_6 is Low|0.07867|action_error_top10|2.4091|1.27824|-0.01699|hard-motion correction factor|candidate|
|0|h_3 is Low AND h_15 is High AND h_2 is High AND h_9 is High|0.09462|action_error_top10|1.9241|-1.21116|0.02019|hard-motion correction factor|candidate|
|2|h_7 is High AND h_5 is Low AND h_9 is Low AND h_15 is Low|0.16691|action_error_top20|0.8446|-0.00477|-0.0386|state-partition factor|candidate|
|15|h_3 is Low AND h_1 is High AND h_12 is Low AND h_7 is Low|0.18493|action_error_top20|0.6808|0.22666|-0.02297|state-partition factor|candidate|
|3|h_11 is High AND h_14 is Low AND h_4 is Low AND h_3 is High|0.13466|action_error_top20|0.6845|-0.03226|-0.06211|state-partition factor|candidate|
|8|h_9 is Low AND h_13 is Low AND h_2 is Low AND h_7 is Low|0.0801|action_error_top20|0.7953|0.29346|0.10015|state-partition factor|candidate|
|14|h_7 is High AND h_5 is Low AND h_1 is Low AND h_9 is Low|0.08364|action_error_top20|0.755|-0.26134|0.076|state-partition factor|candidate|
|13|h_13 is Low AND h_2 is Low AND h_8 is High AND h_12 is High|0.01702|action_error_top10|2.3872|0.45609|0.29966|hard-motion correction factor|candidate|
|4|h_11 is High AND h_3 is High AND h_8 is Low AND h_14 is Low|0.03916|action_error_top20|0.6962|-0.52936|-0.00344|state-partition factor|candidate|
|9|h_11 is High AND h_8 is Low AND h_3 is High AND h_1 is Low|0.03286|action_error_top20|0.7241|0.07877|0.08022|state-partition factor|candidate|
|6|h_3 is Low AND h_6 is High AND h_14 is Low AND h_2 is High|0.02643|action_error_top20|0.7214|0.0326|0.0331|state-partition factor|candidate|
|1|h_2 is High AND h_5 is High AND h_12 is Low AND h_1 is High|0.02615|action_error_top20|0.7123|-0.38411|-0.05087|state-partition factor|candidate|

## White-Box Check

Rule consequents are direct linear heads from `[q, action]` to physical targets. No hidden MLP and no q->h path is used in rule export.


---

# V4.1 Rule Ablation

Best V4.1 q model: `sharp_tau_R16`.

## All-Test Rule Mask

|mask|rules|residual_mse|residual_mse_increase|delta_mse|delta_mse_increase|hard_auc|hard_auc_drop|
|---|---|---|---|---|---|---|---|
|none||0.6727852821350098|0.0|4.077075958251953|0.0|0.6855161964893342|0.0|
|top_usage|15,2,3,0|0.7981470823287964|0.12536180019378662|4.288814544677734|0.21173858642578125|0.6401577055454254|0.04535849094390876|
|top_hard_enriched|10,12,13,0|0.8456970453262329|0.17291176319122314|4.352104187011719|0.2750282287597656|0.5967176556587219|0.08879854083061223|
|top_residual_effect|13,12,0,7|0.8618593215942383|0.18907403945922852|4.359899520874023|0.2828235626220703|0.6164631366729736|0.06905305981636056|
|random|7,2,12,9|0.7457232475280762|0.0729379653930664|4.181859970092773|0.10478401184082031|0.6725868225097656|0.012929373979568504|

## Hard-Subset Rule Mask

|subset|mask|residual_mse_increase|
|---|---|---|
|action_error_top20|top_hard_enriched|1.0118961334228516|
|action_error_top20|top_residual_effect|1.0863513946533203|
|action_error_top20|random|0.41771697998046875|
|action_error_top10|top_hard_enriched|2.0492849349975586|
|action_error_top10|top_residual_effect|2.1977319717407227|
|action_error_top10|random|0.8550186157226562|
|high_residual_top20|top_hard_enriched|1.0118961334228516|
|high_residual_top20|top_residual_effect|1.0863513946533203|
|high_residual_top20|random|0.41771697998046875|
|high_residual_top10|top_hard_enriched|2.0492849349975586|
|high_residual_top10|top_residual_effect|2.1977319717407227|
|high_residual_top10|random|0.8550186157226562|
|large_action_small_disp|top_hard_enriched|1.5503456592559814|
|large_action_small_disp|top_residual_effect|1.6552526950836182|
|large_action_small_disp|random|0.6312947273254395|

## Interpretation

Top hard-enriched and residual-effect masks should exceed random-mask damage if V4.1 made rules more functional.


---

# V4.1 q Semantics

Best V4.1 q model: `sharp_tau_R16`.

Correlation matrix saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_kanfis_rule_features_v41/q_semantics_corr_matrix.csv`.

|q_dim|top_target|pearson|abs_pearson|posthoc_label|
|---|---|---|---|---|
|15|position_x|-0.694|0.694|position-x-related factor|
|3|position_y|0.5854|0.5854|unknown / mixed factor|
|2|position_y|-0.552|0.552|unknown / mixed factor|
|4|position_x|0.551|0.551|unknown / mixed factor|
|14|position_y|-0.5209|0.5209|unknown / mixed factor|
|7|position_x|-0.5122|0.5122|unknown / mixed factor|
|8|position_y|-0.4472|0.4472|unknown / mixed factor|
|13|residual_y|0.4344|0.4344|residual correction factor|
|9|position_x|0.4239|0.4239|unknown / mixed factor|
|1|position_x|-0.4095|0.4095|unknown / mixed factor|
|12|residual_x|0.4006|0.4006|residual correction factor|
|0|residual_x|-0.3926|0.3926|residual correction factor|
|11|position_x|0.3196|0.3196|unknown / mixed factor|
|5|position_y|-0.2586|0.2586|unknown / mixed factor|
|6|position_x|-0.127|0.127|unknown / mixed factor|
|10|hard_action_error_top10|0.0507|0.0507|unknown / mixed factor|

## Naming Rule

No q dimension is named as wall, door, or collision unless enrichment, residual effect, and ablation jointly support it.
