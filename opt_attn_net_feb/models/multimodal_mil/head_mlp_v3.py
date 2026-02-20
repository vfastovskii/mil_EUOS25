from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.mlp import DropPath


def _normalize_activation(name: str) -> Literal["relu", "gelu", "silu"]:
    n = str(name).strip().lower()
    if n == "gelu":
        return "gelu"
    if n in {"relu", "leakyrelu", "leaky_relu"}:
        return "relu"
    # silu / mish / unknowns map to silu for gated blocks
    return "silu"


class _ResFFNBlock(nn.Module):
    """Residual FFN block matching MLP predictor V3 design."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        inner_dim: int,
        *,
        activation: Literal["relu", "gelu", "silu"] = "silu",
        use_glu: bool = True,
        use_layernorm: bool = True,
        pre_layer_norm: bool = True,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        res_scale_init: float | None = 0.1,
        is_last_block: bool = False,
        fc2_gain_non_last: float = 1e-2,
        proj_gain: float = 0.5,
    ):
        super().__init__()
        self.pre_layer_norm = bool(pre_layer_norm)
        self.use_glu = bool(use_glu)
        self.is_last_block = bool(is_last_block)
        self._use_kaiming = activation in {"relu", "silu"}
        self.fc2_gain_non_last = float(fc2_gain_non_last)
        self.proj_gain = float(proj_gain)

        self.pre_norm = nn.LayerNorm(in_dim) if (use_layernorm and pre_layer_norm) else nn.Identity()
        self.post_norm = nn.LayerNorm(out_dim) if (use_layernorm and not pre_layer_norm) else nn.Identity()

        self.fc1 = nn.Linear(in_dim, inner_dim * (2 if self.use_glu else 1))
        self.fc2 = nn.Linear(inner_dim, out_dim)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.drop_path = DropPath(float(drop_path)) if drop_path > 0.0 else nn.Identity()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init))) if res_scale_init is not None else None

        self._init_weights()

    def _init_weights(self) -> None:
        if self.use_glu:
            w = self.fc1.weight
            mid = w.size(0) // 2
            nn.init.xavier_uniform_(w[:mid, :])
            nn.init.kaiming_uniform_(w[mid:, :], nonlinearity="relu")
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            with torch.no_grad():
                w[:mid, :].mul_(1.0 / math.sqrt(2.0))
        else:
            if self._use_kaiming:
                nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(self.fc1.weight)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)

        if self.is_last_block:
            nn.init.zeros_(self.fc2.weight)
        else:
            nn.init.xavier_uniform_(self.fc2.weight, gain=self.fc2_gain_non_last)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

        if isinstance(self.proj, nn.Linear):
            nn.init.xavier_uniform_(self.proj.weight, gain=self.proj_gain)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        y = self.pre_norm(x) if self.pre_layer_norm else x
        y = self.fc1(y)
        if self.use_glu:
            v, g = y.chunk(2, dim=-1)
            y = F.silu(g) * v
        else:
            y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)

        if not self.pre_layer_norm:
            y = self.post_norm(y)

        if isinstance(self.proj, nn.Linear):
            shortcut = self.proj(shortcut)
        if self.res_scale is not None:
            y = y * self.res_scale.tanh()
        y = self.drop_path(y)
        return shortcut + y


class MLPPredictorV3Like(nn.Module):
    """Residual MLP head based on `example_code/mlp_predictor_v3.py`."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        expansion: float = 2.0,
        activation: str = "silu",
        use_glu: bool = True,
        dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        use_layernorm: bool = True,
        pre_layer_norm: bool = True,
        output_dim: int = 1,
        input_layernorm: bool = True,
        final_layernorm: bool = False,
        res_scale_init: float | None = 0.1,
        inner_multiple: int = 64,
        head_dropout: float = 0.0,
        output_bias: float | None = None,
        fc2_gain_non_last: float = 1e-2,
        proj_gain: float = 0.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else input_dim)
        self.num_layers = int(num_layers)
        self.output_dim = int(output_dim)
        self.use_glu = bool(use_glu)
        self.expansion = float(expansion)
        self.stochastic_depth = float(stochastic_depth)
        self.inner_multiple = int(inner_multiple)

        act = _normalize_activation(activation)

        self.input_norm = nn.LayerNorm(self.input_dim) if (input_layernorm and use_layernorm) else nn.Identity()
        self.blocks = nn.ModuleList()
        d_in = self.input_dim
        for i in range(self.num_layers):
            dp = self.stochastic_depth * (i / max(1, self.num_layers - 1)) if self.num_layers > 1 else 0.0
            inner = max(4, int(round(self.hidden_dim * self.expansion)))
            if self.inner_multiple > 1:
                inner = int(math.ceil(inner / self.inner_multiple) * self.inner_multiple)
            self.blocks.append(
                _ResFFNBlock(
                    in_dim=d_in,
                    out_dim=self.hidden_dim,
                    inner_dim=inner,
                    activation=act,
                    use_glu=self.use_glu,
                    use_layernorm=bool(use_layernorm),
                    pre_layer_norm=bool(pre_layer_norm),
                    dropout=float(dropout),
                    drop_path=float(dp),
                    res_scale_init=res_scale_init,
                    is_last_block=(i == self.num_layers - 1),
                    fc2_gain_non_last=float(fc2_gain_non_last),
                    proj_gain=float(proj_gain),
                )
            )
            d_in = self.hidden_dim

        self.out_norm = nn.LayerNorm(self.hidden_dim) if (final_layernorm and use_layernorm) else nn.Identity()
        self.pre_head_drop = nn.Dropout(float(head_dropout)) if head_dropout > 0.0 else nn.Identity()
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)
        self._init_head(output_bias=output_bias)

    def _init_head(self, output_bias: float | None = None) -> None:
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.01)
        if self.fc_out.bias is not None:
            if self.output_dim == 1 and output_bias is not None:
                nn.init.constant_(self.fc_out.bias, float(output_bias))
            else:
                nn.init.zeros_(self.fc_out.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        h = self.pre_head_drop(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        return self.fc_out(h)


def make_predictor_heads(
    in_dim: int,
    count: int,
    *,
    activation: str,
    num_layers: int = 2,
    dropout: float = 0.1,
    stochastic_depth: float = 0.1,
    fc2_gain_non_last: float = 1e-2,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            MLPPredictorV3Like(
                input_dim=int(in_dim),
                hidden_dim=int(in_dim),
                num_layers=int(num_layers),
                expansion=2.0,
                activation=str(activation),
                use_glu=True,
                dropout=float(dropout),
                stochastic_depth=float(stochastic_depth),
                use_layernorm=True,
                pre_layer_norm=True,
                output_dim=1,
                input_layernorm=True,
                final_layernorm=False,
                res_scale_init=0.1,
                inner_multiple=64,
                head_dropout=0.0,
                output_bias=None,
                fc2_gain_non_last=float(fc2_gain_non_last),
                proj_gain=0.5,
            )
            for _ in range(int(count))
        ]
    )


__all__ = [
    "MLPPredictorV3Like",
    "make_predictor_heads",
]
