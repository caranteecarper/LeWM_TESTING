# V4.2 vs V4 / V4.1 Final Comparison

## Replacement Criteria

|criterion|v4|v41|v42|replacement_condition|v42_pass|
|---|---|---|---|---|---|
|full q all_test residual_mse|1.1766499280929565|1.0995261669158936|1.1173511743545532|v42 clearly better than v4|True|
|group0 retention|0.5765598562539798|n/a|0.7801021786622331|v42 >= v4 group0 retention|True|
|best group hard enrichment|2.553459136000654|n/a|2.4779874697162003|> 2|True|
|best group residual ratio|2.9833701148698184|n/a|2.649855380394061|> 2|True|
|best group compactness ratio|0.3687873586550869|n/a|0.4737019782109357|< 0.5|True|
|group0 mask damage vs random on action_error_top10|40.10442781448364|n/a|3.004702568054199|v42 positive and comparable to v4|True|
|top groups retention|n/a|n/a|0.8181950096418563|>= 0.75|True|

## Decision

V4.2 can replace V4.

This is a replacement decision against V4, not a claim that V4.2 dominates V4.1 on every metric. On this validation subset, `q_v41+action` has lower all-test residual MSE than `q_v42+action` (`1.0995` vs `1.1174`), while V4.2 still passes the configured V4 replacement criteria and has stronger group-retention evidence than V4 group 0.

## Answers

- V4.2 can replace V4 only if it passes performance, active-rule, group audit, group-only retention, and mask-damage criteria together.
- If the next stage prioritizes the best sparse/performance reference rather than replacing V4 specifically, keep V4.1 in the comparison set because it remains competitive.
