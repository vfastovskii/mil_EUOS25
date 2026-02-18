from __future__ import annotations

import torch.nn as nn

from .predictors import build_predictor_heads


def make_aux_pred_head(
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
    return build_predictor_heads(
        name=str(name),
        in_dim=int(in_dim),
        count=int(count),
        activation=str(activation),
        num_layers=int(num_layers),
        dropout=float(dropout),
        stochastic_depth=float(stochastic_depth),
        fc2_gain_non_last=float(fc2_gain_non_last),
    )


__all__ = ["make_aux_pred_head"]
