# V5 Semantic Maps

## Readout Metrics

|representation|pos_overall_mse|pos_x_r2|pos_y_r2|res_overall_mse|res_x_r2|res_y_r2|hard_acc|hard_f1|
|---|---|---|---|---|---|---|---|---|
|z|17.94780921936035|0.9967621564865112|0.9788062572479248|1.0738195180892944|0.5638768672943115|0.38840460777282715|0.615839958190918|0.502302151647695|
|h|48.331214904785156|0.9639533758163452|0.9664914011955261|1.0482959747314453|0.5751938819885254|0.3895072340965271|0.6269599795341492|0.5425891306688814|
|q_v4|56.783607482910156|0.9594323635101318|0.9590938091278076|1.2287983894348145|0.5035983324050903|0.26248377561569214|0.555679976940155|0.42208422918271676|
|q_v41|289.9206848144531|0.844194769859314|0.7468926310539246|1.1306427717208862|0.5403942465782166|0.3617544174194336|0.5245599746704102|0.41689967493687463|
|hq|37.848724365234375|0.9678633213043213|0.9771288633346558|1.0278671979904175|0.5813015699386597|0.43207573890686035|0.5743200182914734|0.4716826279562097|

## q Semantic Alignment

|q|dim|top_target|pearson|abs_pearson|
|---|---|---|---|---|
|q_v4|9|position_y|0.8436|0.8436|
|q_v4|7|position_x|0.8411|0.8411|
|q_v4|6|position_x|-0.8398|0.8398|
|q_v4|0|position_y|0.8184|0.8184|
|q_v4|10|position_y|-0.7444|0.7444|
|q_v4|3|position_y|-0.7386|0.7386|
|q_v4|5|position_x|-0.7351|0.7351|
|q_v41|15|position_x|-0.6668|0.6668|
|q_v4|8|position_x|0.6329|0.6329|
|q_v4|1|position_y|-0.6188|0.6188|
|q_v4|4|position_x|-0.6158|0.6158|
|q_v41|13|residual_y|0.6016|0.6016|
|q_v4|2|hard_action_error_top10|0.5998|0.5998|
|q_v41|12|residual_x|0.5668|0.5668|
|q_v41|0|residual_x|-0.5663|0.5663|
|q_v41|3|position_x|0.5497|0.5497|
|q_v41|7|position_x|-0.5316|0.5316|
|q_v4|12|position_y|-0.524|0.524|
|q_v41|4|position_x|0.5152|0.5152|
|q_v4|11|hard_action_error_top10|-0.5118|0.5118|

## Figures

- true position distribution: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/true_position_distribution.png`
- predicted position maps: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/pred_position_h.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/pred_position_q_v4.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/pred_position_q_v41.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/pred_position_hq.png`
- hard subset heatmaps: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/hard_subset_action_error_top10.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/hard_subset_high_residual_top10.png`, `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/hard_subset_large_action_small_disp.png`
- residual direction map: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_visualization_v5/figures/semantic_maps/residual_direction_map.png`

## Notes

q heatmaps are post-hoc spatial activation maps. They should be interpreted as state partitions or correction factors unless rule enrichment and ablation provide stronger evidence.
