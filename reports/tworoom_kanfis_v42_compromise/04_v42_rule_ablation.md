# V4.2 Rule Ablation

Best V4.2 model: `v42_tau05_topk6_hard_medium_R16`.

|mask|rules|subset|residual_mse_increase|hard_f1_drop|
|---|---|---|---|---|
|none||all_test|0.0|0.0|
|none||action_error_top10|0.0|0.0|
|none||high_residual_top10|0.0|0.0|
|none||large_action_small_disp|0.0|0.0|
|mask_best_rules|10,13,12,6|all_test|0.3632109761238098|-0.016342979078269426|
|mask_best_rules|10,13,12,6|action_error_top10|3.7577786445617676|0.007553540251871826|
|mask_best_rules|10,13,12,6|high_residual_top10|3.7577786445617676|0.007553540251871826|
|mask_best_rules|10,13,12,6|large_action_small_disp|3.051063299179077|-0.017563753626715806|
|mask_random|0,1,2,3|all_test|0.038156867027282715|0.004424713474318803|
|mask_random|0,1,2,3|action_error_top10|-0.1318502426147461|-0.0031156554339786258|
|mask_random|0,1,2,3|high_residual_top10|-0.1318502426147461|-0.0031156554339786258|
|mask_random|0,1,2,3|large_action_small_disp|-0.0835576057434082|0.004620270031599372|
