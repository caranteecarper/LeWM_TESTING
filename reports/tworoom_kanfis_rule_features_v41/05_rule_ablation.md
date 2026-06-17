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
