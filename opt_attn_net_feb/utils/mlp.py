from __future__ import annotations

from typing import List
import torch
import torch.nn as nn


def _act_factory(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "mish":
        # Use native Mish if available, otherwise define a lightweight fallback
        if hasattr(nn, "Mish"):
            return nn.Mish()
        class _Mish(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * torch.tanh(torch.nn.functional.softplus(x))
        return _Mish()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "leakyrelu" or name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=False)
    # default
    return nn.GELU()


def make_mlp(in_dim: int, hidden_dim: int, n_layers: int, dropout: float, activation: str = "GELU") -> nn.Module:
    """Build a simple MLP with LayerNorm, activation, and Dropout per layer.

    Parameters
    ----------
    in_dim: input feature dimension
    hidden_dim: hidden layer size (reused for all layers)
    n_layers: number of stacked layers
    dropout: dropout probability per layer
    activation: activation function name in {GELU, SiLU, Mish, ReLU, LeakyReLU}
    """
    layers: List[nn.Module] = []
    d = in_dim
    act = _act_factory(activation)
    for _ in range(int(n_layers)):
        layers += [
            nn.Linear(d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act.__class__(**getattr(act, "__dict__", {})) if isinstance(act, (nn.LeakyReLU,)) else _act_factory(activation),
            nn.Dropout(float(dropout)),
        ]
        d = hidden_dim
    return nn.Sequential(*layers)
