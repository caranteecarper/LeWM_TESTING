# V5.1 V4 vs V4.1 Rule Comparison

## Unified Comparison Table

|method|unit_type|unit_id|region|hard_enrichment|residual_magnitude|compactness|clarity|comments|
|---|---|---|---|---|---|---|---|---|
|V4|rule|15|action_error_top10|2.415|3.319|10.925|high|soft single rule; best read with group context|
|V4|rule|2|action_error_top10|2.478|3.189|17.95|high|soft single rule; best read with group context|
|V4|rule|14|action_error_top10|2.39|2.731|12.336|high|soft single rule; best read with group context|
|V4|rule|12|action_error_top10|1.119|1.47|16.791|low|soft single rule; best read with group context|
|V4|rule|13|action_error_top20|0.992|0.994|23.455|low|soft single rule; best read with group context|
|V4|rule|9|large_action_small_disp|0.956|0.992|30.104|low|soft single rule; best read with group context|
|V4|group|0|action_error_top10|2.553|3.338|13.719|medium-high|co-activated rules 15,2,8,11|
|V4|group|1|large_action_small_disp|0.62|0.427|20.368|medium|co-activated rules 14,3,11,5|
|V4|group|2|large_action_small_disp|0.646|0.438|41.801|medium|co-activated rules 12,9,7,5|
|V4|group|3|action_error_top20|0.956|1.136|23.931|medium|co-activated rules 13,15,8,1|
|V4|group|4|large_action_small_disp|0.672|0.467|29.697|medium|co-activated rules 0,9,3,7|
|V4.1|sparse_rule|0|action_error_top10|2.528|3.279|9.568|high|sparse rule; easier single-rule story|
|V4.1|sparse_rule|10|action_error_top10|2.327|3.253|30.181|high|sparse rule; easier single-rule story|
|V4.1|sparse_rule|12|action_error_top10|2.453|3.064|12.64|high|sparse rule; easier single-rule story|
|V4.1|sparse_rule|13|action_error_top10|2.025|2.771|26.748|high|sparse rule; easier single-rule story|
|V4.1|sparse_rule|7|action_error_top20|0.681|0.597|30.062|low|sparse rule; easier single-rule story|
|V4.1|sparse_rule|3|large_action_small_disp|0.724|0.533|26.815|low|sparse rule; easier single-rule story|

## Interpretation

- V4 single rules tend to preserve more performant soft rule structure, but individual rules can be less sparse.
- V4 groups are the better way to explain V4 because co-activation captures the soft rule pattern.
- V4.1 sparse rules are easier to explain one by one and now have non-empty residual-direction figures.
- For reporting, use V4 groups when discussing performance-oriented rule representations and V4.1 rules when showing white-box sparse rule examples.
