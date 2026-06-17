# V5 Rule Visual Link

This report links V4.1 q rules to images, true spatial positions, residual directions, and hard-subset enrichment.

## Top Linked Rules

|rule|usage_top10|top_hard_subset|hard_enrichment|active_residual_mag|
|---|---|---|---|---|
|0|0.10000000149011612|action_error_top10|2.528302038863167|3.2785284519195557|
|12|0.10000000149011612|action_error_top10|2.4528301851427172|3.064133882522583|
|10|0.10000000149011612|action_error_top10|2.3270441371472934|3.252737283706665|
|13|0.10000000149011612|action_error_top10|2.025157284573483|2.770914077758789|
|3|0.10000000149011612|large_action_small_disp|0.7235142444706872|0.5331056714057922|
|7|0.10000000149011612|action_error_top20|0.6810035744983276|0.5968091487884521|

## Figure Pattern

For each top rule, V5 saves:

- representative high-activation sample images
- spatial activation scatter
- residual direction map over top active samples

Output directory: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/rule_visual_link`.

## Naming Caution

Rules are named conservatively as position-related, residual-correction, hard-motion correction, or state-partition rules. V5 does not name rules as walls, doors, or collisions without stronger evidence.
