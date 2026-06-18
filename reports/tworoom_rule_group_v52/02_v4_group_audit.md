# V4 Group Audit

## Best Group Definitions

|group|rules|activation_definition|top_hard_subset|top_hard_enrichment|mean_residual_mag|residual_mag_ratio_to_global_mean|compactness_ratio_to_random|group_clarity_score|physical_interpretation_label|
|---|---|---|---|---|---|---|---|---|---|
|group_0|15,2,8,11|mean|action_error_top10|2.553459136000654|3.4407596588134766|2.9833701148698184|0.3687873586550869|20.6566561925199|hard-motion correction group|
|group_0|15,2,8,11|sum|action_error_top10|2.553459136000654|3.4407596588134766|2.9833701148698184|0.3687873586550869|20.6566561925199|hard-motion correction group|
|group_0|15,2,8,11|max|action_error_top10|2.402515803431747|3.314497232437134|2.8738920963987376|0.46875883509856703|14.729474224211094|hard-motion correction group|
|group_0|15,2,8,11|topk_avg|action_error_top10|2.364779876571522|3.2476465702056885|2.8159280746020725|0.46070389143759033|14.454078136637243|hard-motion correction group|
|group_1|14,3,11,5|max|large_action_small_disp|1.912144754579412|2.1871237754821777|1.8963834607224148|0.7983061017030147|4.542317388475278|state/correction group|
|group_3|13,15,8,1|max|large_action_small_disp|1.2661498556281532|1.7442867755889893|1.512413988209029|0.7116292534478227|2.690927535009305|state/correction group|
|group_3|13,15,8,1|mean|action_error_top20|0.9557944686605618|1.0090844631195068|0.8749441196623231|0.6433160277776268|1.2999314704613625|state/correction group|
|group_3|13,15,8,1|sum|action_error_top20|0.9557944686605618|1.0090844631195068|0.8749441196623231|0.6433160277776268|1.2999314704613625|state/correction group|

## Answers

- Group 0 is more stable than any one soft rule when interpreted as a co-activation unit.
- Group 0 with rules `[15,2,8,11]` is the current main V4 interpretation unit because it is hard-enriched and high-residual.
- V4 should be explained through groups first, single rules second.
