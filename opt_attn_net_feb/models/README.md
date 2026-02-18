# Models Directory

This folder is intentionally split by responsibility:

- `models/multimodal_mil/`
  - Main multimodal MIL model and its components.
  - Contains:
    - `model.py`: `MILTaskAttnMixerWithAux`
    - `embedders.py`: 2D/3D embedder registries + builders
    - `aggregators.py`: 3D aggregator registry + builders
    - `predictors.py`: predictor-head registry + builders
    - `make_2d_embedder.py`: `make_2d_embedder`
    - `make_3d_embedder.py`: `make_3d_embedder`
    - `make_aggregator.py`: `make_aggregator`
    - `make_mixer.py`: `make_mixer`
    - `make_pred_head.py`: `make_pred_head`
    - `make_aux_pred_head.py`: `make_aux_pred_head`
    - `heads.py`, `training.py`, `constants.py`: model internals

- `models/attention_pooling/`
  - Reusable attention pooling primitives used by multimodal MIL.
  - Contains:
    - `pool.py`: `TaskAttentionPool`
    - `masking.py`: attention mask/normalization helpers

- `models/__init__.py`
  - Stable package-level exports (`MILTaskAttnMixerWithAux`, `TaskAttentionPool`).
