# V4.2 Group Audit

## V4.2 Group 0

|group|rules|activation_definition|top_hard_subset|top_hard_enrichment|mean_residual_mag|residual_mag_ratio_to_global_mean|compactness_ratio_to_random|group_clarity_score|physical_interpretation_label|
|---|---|---|---|---|---|---|---|---|---|
|group_0|10,13,12,1|topk_avg|action_error_top10|2.4779874697162003|3.056112766265869|2.649855380394061|0.4737019782109357|13.861686738096362|hard-motion correction group|

## Best Group Definitions

|group|rules|activation_definition|top_hard_subset|top_hard_enrichment|mean_residual_mag|residual_mag_ratio_to_global_mean|compactness_ratio_to_random|group_clarity_score|physical_interpretation_label|
|---|---|---|---|---|---|---|---|---|---|
|group_0|10,13,12,1|topk_avg|action_error_top10|2.4779874697162003|3.056112766265869|2.649855380394061|0.4737019782109357|13.861686738096362|hard-motion correction group|
|best_rules|10,13,12,6|topk_avg|action_error_top10|2.4779874697162003|3.056112766265869|2.649855380394061|0.4737019782109357|13.861686738096362|hard-motion correction group|
|top3_rules|10,13,12|topk_avg|action_error_top10|2.4779874697162003|3.056112766265869|2.649855380394061|0.4737019782109357|13.861686738096362|hard-motion correction group|
|top_hard_rules|12,10,13,1|topk_avg|action_error_top10|2.4779874697162003|3.056112766265869|2.649855380394061|0.4737019782109357|13.861686738096362|hard-motion correction group|
|top_residual_rules|10,13,1,12|topk_avg|action_error_top10|2.4779874697162003|3.056112766265869|2.649855380394061|0.4737019782109357|13.861686738096362|hard-motion correction group|
|coactivation_seed_12|12,13,8,10|topk_avg|action_error_top10|2.4779874697162003|3.056112766265869|2.649855380394061|0.4737019782109357|13.861686738096362|hard-motion correction group|
|coactivation_seed_13|13,12,10,8|topk_avg|action_error_top10|2.4779874697162003|3.056112766265869|2.649855380394061|0.4737019782109357|13.861686738096362|hard-motion correction group|
|best_rules|10,13,12,6|mean|action_error_top10|2.4654088274294588|3.140653371810913|2.7231577732043326|0.49000609028949027|13.701252587645861|hard-motion correction group|
|best_rules|10,13,12,6|sum|action_error_top10|2.4654088274294588|3.140653371810913|2.7231577732043326|0.49000609028949027|13.701252587645861|hard-motion correction group|
|group_0|10,13,12,1|mean|action_error_top10|2.4528301851427172|3.135301351547241|2.7185172115574545|0.4892414969498817|13.629385726495755|hard-motion correction group|
|group_0|10,13,12,1|sum|action_error_top10|2.4528301851427172|3.135301351547241|2.7185172115574545|0.4892414969498817|13.629385726495755|hard-motion correction group|
|top_hard_rules|12,10,13,1|mean|action_error_top10|2.4528301851427172|3.135301351547241|2.7185172115574545|0.4892414969498817|13.629385726495755|hard-motion correction group|

## Answers

- Preserved V4.2 group 0 is `[10, 13, 12, 1]`.
- Best audited group: `group_0` with rules `10,13,12,1` and activation `topk_avg`.
- V4.2 can be discussed through both strong single rules and rule groups. The final replacement decision depends on whether group-only retention and mask damage are at least as convincing as V4 group 0.
