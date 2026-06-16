# Pair Dataset

Input latent cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_encoder_readout/tworoom_latents.pt`

Output pair cache: `/data/lzt26/lewm_official_tworoom_readout_git/outputs/tworoom_preintegration/tworoom_pairs.pt`

Method: sort by `(episode_id, timestep)` and keep only pairs where the episode is identical and `timestep_next = timestep + 1`.

- source samples: `730809`
- pairs kept: `720809`
- discarded non-consecutive candidates: `9999`
- cross-episode pairs after filtering: `0`
- action block shape: `(720809, 10)`
- action block dim check: `10`

Action alignment note: the cached action comes from the official TwoRoom sequence reader using `frameskip=5`; it is a 10-D block at the sequence start. This is the best available alignment for transition from `z_t` to `z_next`, but the exact internal frameskip-to-next-sequence convention is not fully proven beyond the official loader/config path.

Delta position stats:

- mean: `[0.02035294659435749, -0.5140467882156372]`
- std: `[4.2403435707092285, 4.311271667480469]`
- min: `[-12.497772216796875, -5.965789794921875]`
- max: `[12.4981689453125, 6.975152969360352]`
