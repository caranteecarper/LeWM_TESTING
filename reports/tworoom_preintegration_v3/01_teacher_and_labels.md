# V3 Teacher And Labels

## Split

- train samples: `576945`
- val samples: `72141`
- test samples: `71723`
- split source: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v2/splits_episode_seed3072.pt`

## Teacher Metrics

|name|model|x_mse|y_mse|x_r2|y_r2|x_pearson|y_pearson|overall_mse|
|---|---|---|---|---|---|---|---|---|
|action_only_delta|linear|1.6351069211959839|0.08209721744060516|0.9093542695045471|0.9955928921699524|0.9536049962043762|0.9977800250053406|0.8586020469665527|
|action_only_delta|mlp|3.1635775566101074|1.9130173921585083|0.8246201276779175|0.8973062634468079|0.9081665873527527|0.9473984837532043|2.538297414779663|
|z_action_residual|linear|1.1204755306243896|0.06748365610837936|0.314713716506958|0.17798089981079102|0.5609991550445557|0.4234798550605774|0.59397953748703|
|z_action_residual|mlp|0.37398818135261536|0.01955048367381096|0.7712676525115967|0.7618553638458252|0.8782550692558289|0.8748380541801453|0.1967693418264389|
|z_action_delta_teacher|action_linear_plus_residual_mlp|0.37398818135261536|0.019550485536456108|0.9792671203613281|0.9989504814147949|0.9895813465118408|0.9994609951972961|0.1967693418264389|

## Hard Label Rates

|label|train_pos_rate|val_pos_rate|test_pos_rate|
|---|---|---|---|
|action_error_top20|0.20000173151493073|0.19479908049106598|0.2076600193977356|
|action_error_top10|0.10000086575746536|0.0970044806599617|0.10477810353040695|
|high_residual_top20|0.20000173151493073|0.19479908049106598|0.2076600193977356|
|high_residual_top10|0.10000086575746536|0.0970044806599617|0.10477810353040695|
|large_action_small_disp|0.025381969287991524|0.025242233648896217|0.02474798820912838|

## Train-Only Thresholds

|threshold|value|
|---|---|
|action_error_top20|0.3499937951564789|
|action_error_top10|0.3695480525493622|
|high_residual_top20|0.3499937951564789|
|high_residual_top10|0.3695480525493622|
|action_mag_p75|2.8932225704193115|
|disp_mag_p25|5.104437351226807|

## Notes

- Thresholds are computed only on the train split and then applied to val/test.
- Residual uses action-only linear as baseline because it is stable, interpretable, and gives a fixed correction target.
- Data leakage risk is controlled by using train-only thresholds and fixed episode-level split.
- Teacher artifact saved to `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration_v3/v3_teacher_labels.pt`.
