from __future__ import annotations

# Compatibility facade: concrete head implementations are split into dedicated files.
from .head_mlp_v3 import MLPPredictorV3Like, make_predictor_heads
from .head_utils import apply_shared_heads, apply_task_heads, make_projection

__all__ = [
    "make_projection",
    "MLPPredictorV3Like",
    "make_predictor_heads",
    "apply_task_heads",
    "apply_shared_heads",
]
