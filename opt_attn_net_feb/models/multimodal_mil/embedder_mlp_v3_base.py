from __future__ import annotations

import torch.nn as nn

from ...utils.mlp import make_residual_mlp_embedder_v3


def build_mlp_v3_embedder(
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


__all__ = ["build_mlp_v3_embedder"]
