from __future__ import annotations

import torch.nn as nn

from .embedders import build_3d_embedder


def make_3d_embedder(
    *,
    name: str,
    input_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    activation: str,
) -> nn.Module:
    return build_3d_embedder(
        name=str(name),
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        layers=int(layers),
        dropout=float(dropout),
        activation=str(activation),
    )


__all__ = ["make_3d_embedder"]
