from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .head_mlp_v3 import make_predictor_heads

PredictorBuilder = Callable[..., nn.ModuleList]

_PREDICTOR_REGISTRY: Dict[str, PredictorBuilder] = {}


def _build_mlp_v3_predictor_heads(
    *,
    in_dim: int,
    count: int,
    activation: str,
    num_layers: int,
    dropout: float,
    stochastic_depth: float,
    fc2_gain_non_last: float,
) -> nn.ModuleList:
    return make_predictor_heads(
        int(in_dim),
        int(count),
        activation=str(activation),
        num_layers=int(num_layers),
        dropout=float(dropout),
        stochastic_depth=float(stochastic_depth),
        fc2_gain_non_last=float(fc2_gain_non_last),
    )


def register_predictor(name: str, builder: PredictorBuilder) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Predictor name must be non-empty")
    _PREDICTOR_REGISTRY[key] = builder


def build_predictor_heads(
    *,
    name: str,
    in_dim: int,
    count: int,
    activation: str,
    num_layers: int,
    dropout: float,
    stochastic_depth: float,
    fc2_gain_non_last: float,
) -> nn.ModuleList:
    key = str(name).strip().lower()
    if key not in _PREDICTOR_REGISTRY:
        raise ValueError(f"Unknown predictor '{name}'. Available: {sorted(_PREDICTOR_REGISTRY)}")
    return _PREDICTOR_REGISTRY[key](
        in_dim=int(in_dim),
        count=int(count),
        activation=str(activation),
        num_layers=int(num_layers),
        dropout=float(dropout),
        stochastic_depth=float(stochastic_depth),
        fc2_gain_non_last=float(fc2_gain_non_last),
    )


def available_predictors() -> list[str]:
    return sorted(_PREDICTOR_REGISTRY)


# Default implementation
register_predictor("mlp_v3", _build_mlp_v3_predictor_heads)


__all__ = [
    "register_predictor",
    "build_predictor_heads",
    "available_predictors",
]
