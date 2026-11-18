PROFILE trial configurations for 512 pixels = 531.337 um (1.0378 um/pixel), mouse GCaMP8m neurons.

These files use released PROFILE mouse checkpoints and differ only in post-processing thresholds, except the m110 comparison model. PROFILE has no physical pixel-size field; do not interpret these as retrained models at this exact scale.

Recommended order:
1. profile_gcamp8m_1p038um_m124_balanced.json
2. profile_gcamp8m_1p038um_m124_sensitive.json
3. profile_gcamp8m_1p038um_m124_conservative.json
4. profile_gcamp8m_1p038um_m110_balanced.json

The checkpoint paths are relative to the configured PROFILE repository root. In TwoPhoton Studio, choose the JSON and keep the default PROFILE checkpoint override empty so the matching m124 or m110 checkpoint is used.

The vector-field clip values follow the released matching inference configs: m124 uses the n115 value (the checkpoint metadata reports 115 annotated training sessions), and m110 uses the n100 value.

Approximate interpretation at 1.0378 um/pixel:
- min_size 25: permissive, about 5 pixels equivalent diameter
- min_size 40: balanced, about 7 pixels equivalent diameter
- min_size 80: conservative, about 10 pixels equivalent diameter

These are trial settings, not validated GCaMP8m-specific models. Compare candidates against the Registered mean and Suite2p/Cellpose masks before accepting them.
