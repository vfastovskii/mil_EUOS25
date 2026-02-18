from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from ...utils.mlp import make_residual_mlp_embedder_v3

EmbedderBuilder = Callable[..., nn.Module]

_EMBEDDER_2D_REGISTRY: Dict[str, EmbedderBuilder] = {}
_EMBEDDER_3D_REGISTRY: Dict[str, EmbedderBuilder] = {}
_LEGACY_2D_ALIASES: Dict[str, str] = {"mlp_v3": "mlp_v3_2d"}
_LEGACY_3D_ALIASES: Dict[str, str] = {"mlp_v3": "mlp_v3_3d"}


def _build_mlp_v3_embedder(
    *,
    input_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    activation: str,
) -> nn.Module:
    return make_residual_mlp_embedder_v3(
        int(input_dim),
        int(hidden_dim),
        int(layers),
        float(dropout),
        activation=str(activation),
    )


def register_2d_embedder(name: str, builder: EmbedderBuilder) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("2D embedder name must be non-empty")
    _EMBEDDER_2D_REGISTRY[key] = builder


def register_3d_embedder(name: str, builder: EmbedderBuilder) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("3D embedder name must be non-empty")
    _EMBEDDER_3D_REGISTRY[key] = builder


def _resolve_2d_name(name: str) -> str:
    key = str(name).strip().lower()
    return _LEGACY_2D_ALIASES.get(key, key)


def _resolve_3d_name(name: str) -> str:
    key = str(name).strip().lower()
    return _LEGACY_3D_ALIASES.get(key, key)


def build_2d_embedder(
    *,
    name: str,
    input_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    activation: str,
) -> nn.Module:
    key = _resolve_2d_name(name)
    if key not in _EMBEDDER_2D_REGISTRY:
        raise ValueError(
            f"Unknown 2D embedder '{name}' (resolved='{key}'). Available: {sorted(_EMBEDDER_2D_REGISTRY)}"
        )
    return _EMBEDDER_2D_REGISTRY[key](
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        layers=int(layers),
        dropout=float(dropout),
        activation=str(activation),
    )


def build_3d_embedder(
    *,
    name: str,
    input_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    activation: str,
) -> nn.Module:
    key = _resolve_3d_name(name)
    if key not in _EMBEDDER_3D_REGISTRY:
        raise ValueError(
            f"Unknown 3D embedder '{name}' (resolved='{key}'). Available: {sorted(_EMBEDDER_3D_REGISTRY)}"
        )
    return _EMBEDDER_3D_REGISTRY[key](
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        layers=int(layers),
        dropout=float(dropout),
        activation=str(activation),
    )


def available_2d_embedders() -> list[str]:
    return sorted(_EMBEDDER_2D_REGISTRY)


def available_3d_embedders() -> list[str]:
    return sorted(_EMBEDDER_3D_REGISTRY)


# Default implementations
register_2d_embedder("mlp_v3_2d", _build_mlp_v3_embedder)
register_3d_embedder("mlp_v3_3d", _build_mlp_v3_embedder)


__all__ = [
    "register_2d_embedder",
    "register_3d_embedder",
    "build_2d_embedder",
    "build_3d_embedder",
    "available_2d_embedders",
    "available_3d_embedders",
]
