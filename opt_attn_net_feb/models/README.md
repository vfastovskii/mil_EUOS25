# Models Directory

This folder is intentionally split by responsibility:

- `models/multimodal_mil/`
  - Main multimodal MIL model and its components.
  - Contains:
    - `model.py`: `MILTaskAttnMixerWithAux`
    - `embedders.py`: 2D/3D embedder registries + builders
    - `embedder_mlp_v3_base.py`: shared V3 residual MLP embedder implementation
    - `aggregators.py`: 3D aggregator registry + builders
    - `predictors.py`: predictor-head registry + builders
    - `head_mlp_v3.py`: V3-style predictor head implementation
    - `head_utils.py`: head application/projection helpers
    - `heads.py`, `training.py`, `constants.py`: model internals and compatibility re-exports

- `models/attention_pooling/`
  - Reusable attention pooling primitives used by multimodal MIL.
  - Contains:
    - `pool.py`: `TaskAttentionPool`
    - `masking.py`: attention mask/normalization helpers

- `models/__init__.py`
  - Stable package-level exports (`MILTaskAttnMixerWithAux`, `TaskAttentionPool`).
