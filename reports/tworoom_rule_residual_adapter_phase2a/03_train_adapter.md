# Residual Adapter 训练

训练目标：`final_pred_z = base_pred_z + gate * R(input)`，只训练 R，不修改 LeWM。

- input groups：`['h_action', 'single_group_action', 'full_q_action', 'q_only_action', 'shuffled_q_action', 'z_action', 'group_only_action']`
- model types：`['linear_adapter', 'small_mlp_64', 'small_mlp_128']`
- gates：`['c_soft_gate', 'c_sigmoid_gate', 'always_on', 'random_gate', 'oracle_gate']`
- loss mode：`hard_weighted`
- seeds：`0,1,2`

## Val hard subset Top 20

|input_group|model_type|gate_name|loss_mode|seed|epoch|val_hard_mse|model_path|
|---|---|---|---|---|---|---|---|
|q_only_action|small_mlp_64|always_on|hard_weighted|1|97|0.005907590966671705|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed1.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|95|0.005908405873924494|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|90|0.005908552557229996|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|1|90|0.005908602848649025|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed1.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|92|0.005908748600631952|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|97|0.0059089804999530315|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|98|0.005908982362598181|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|83|0.005909040104597807|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|94|0.005909093655645847|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|99|0.0059091863222420216|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|h_action|small_mlp_64|always_on|hard_weighted|1|75|0.005909250117838383|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/h_action__small_mlp_64__always_on__hard_weighted__seed1.pt|
|full_q_action|small_mlp_64|always_on|hard_weighted|2|88|0.005909277591854334|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/full_q_action__small_mlp_64__always_on__hard_weighted__seed2.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|96|0.005909346975386143|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|full_q_action|small_mlp_64|always_on|hard_weighted|2|87|0.005909453611820936|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/full_q_action__small_mlp_64__always_on__hard_weighted__seed2.pt|
|h_action|small_mlp_64|always_on|hard_weighted|1|84|0.005909469444304705|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/h_action__small_mlp_64__always_on__hard_weighted__seed1.pt|
|full_q_action|small_mlp_64|always_on|hard_weighted|2|96|0.00590949272736907|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/full_q_action__small_mlp_64__always_on__hard_weighted__seed2.pt|
|h_action|small_mlp_64|always_on|hard_weighted|1|79|0.005909575615078211|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/h_action__small_mlp_64__always_on__hard_weighted__seed1.pt|
|full_q_action|small_mlp_64|always_on|hard_weighted|0|55|0.005909713916480541|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/full_q_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
|full_q_action|small_mlp_64|always_on|hard_weighted|2|77|0.0059098489582538605|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/full_q_action__small_mlp_64__always_on__hard_weighted__seed2.pt|
|q_only_action|small_mlp_64|always_on|hard_weighted|0|79|0.0059099807403981686|/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_rule_residual_adapter_phase2a/models/q_only_action__small_mlp_64__always_on__hard_weighted__seed0.pt|
