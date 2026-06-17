# H Dynamics Evaluation

## H-Space Prediction

|candidate|task|mse|mean_r2|dim_r2_min|
|---|---|---|---|---|
|old_pure_hK16|h_action_to_h_next|0.06884817034006119|0.6511854529380798|0.4369358420372009|
|old_pure_hK16|h_action_to_delta_h|0.06526435166597366|0.4060770273208618|0.35502856969833374|
|K16_pos_delta_residual_teacher_hard_hnext_light_aux|h_action_to_h_next|0.11827948689460754|0.770660936832428|0.6651503443717957|
|K16_pos_delta_residual_teacher_hard_hnext_light_aux|h_action_to_delta_h|0.11148940026760101|0.5512418746948242|0.44024014472961426|
|K16_pos_delta_residual_teacher_hard_hnext_hard_weighted|h_action_to_h_next|0.12465327233076096|0.7237664461135864|0.5223915576934814|
|K16_pos_delta_residual_teacher_hard_hnext_hard_weighted|h_action_to_delta_h|0.11697612702846527|0.5181655287742615|0.3718299865722656|
|K8_pos_delta_residual_teacher_hard_hnext_hard_weighted|h_action_to_h_next|0.2390974909067154|0.743419885635376|0.5953128337860107|
|K8_pos_delta_residual_teacher_hard_hnext_hard_weighted|h_action_to_delta_h|0.2223084717988968|0.47682255506515503|0.3423495292663574|

## Delta-H Physical Prediction

|name|model|x_mse|y_mse|x_r2|y_r2|overall_mse|
|---|---|---|---|---|---|---|
|old_pure_hK16_delta_h_to_delta_position|linear|7.955322742462158|10.132416725158691|0.5589791536331177|0.45607608556747437|9.043869018554688|
|old_pure_hK16_delta_h_to_residual|linear|1.567223310470581|0.08198980242013931|0.0414813756942749|0.0012813806533813477|0.8246065974235535|
|K16_pos_delta_residual_teacher_hard_hnext_light_aux_delta_h_to_delta_position|linear|8.292628288269043|9.787611961364746|0.5402798652648926|0.47458571195602417|9.040119171142578|
|K16_pos_delta_residual_teacher_hard_hnext_light_aux_delta_h_to_residual|linear|1.5248005390167236|0.08193518221378326|0.06742733716964722|0.0019466876983642578|0.8033678531646729|
|K16_pos_delta_residual_teacher_hard_hnext_hard_weighted_delta_h_to_delta_position|linear|9.897695541381836|11.638020515441895|0.45129942893981934|0.375252902507782|10.767858505249023|
|K16_pos_delta_residual_teacher_hard_hnext_hard_weighted_delta_h_to_residual|linear|1.550749659538269|0.08194884657859802|0.05155670642852783|0.0017802715301513672|0.8163492679595947|
|K8_pos_delta_residual_teacher_hard_hnext_hard_weighted_delta_h_to_delta_position|linear|11.998598098754883|11.832710266113281|0.3348313570022583|0.36480164527893066|11.915655136108398|
|K8_pos_delta_residual_teacher_hard_hnext_hard_weighted_delta_h_to_residual|linear|1.5627658367156982|0.08196184784173965|0.044207632541656494|0.0016219019889831543|0.8223637938499451|

## Answers

- v3 h improves over old pure hK16 only if h-next/delta-h mean R2 rises and delta_h better predicts physical residual/delta.
- Moderate h-next R2 remains a blocker for formal predictor integration.
