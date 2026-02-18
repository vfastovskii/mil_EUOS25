from __future__ import annotations

import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act_factory(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "mish":
        if hasattr(nn, "Mish"):
            return nn.Mish()

        class _Mish(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * torch.tanh(torch.nn.functional.softplus(x))

        return _Mish()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name in {"leakyrelu", "leaky_relu"}:
        return nn.LeakyReLU(negative_slope=0.01, inplace=False)
    return nn.GELU()


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = rand.floor()
        return x.div(keep_prob) * mask


class MLPEmbedderV3Like(nn.Module):
    """Residual MLP stack inspired by `example_code/mlp_embedder_v3.py`.

    The module is token-wise: it does not mix tokens/instances. It only maps
    feature vectors to embedding vectors with residual FFN blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str = "GELU",
        *,
        expansion: float = 2.0,
        use_layernorm: bool = True,
        block_norm: Literal["pre", "post", "none"] = "pre",
        gated: bool = True,
        residual_dropout: float = 0.05,
        stochastic_depth: float = 0.05,
        residual_scale_init: Optional[float] = 0.1,
        residual_scale_warmup: Optional[float] = 0.01,
        residual_scale_last_k: int = 2,
        zero_init_last_block: bool = True,
        depth_scale_ff2: bool = False,
        output_dim: Optional[int] = None,
        final_norm: Literal["none", "layernorm", "l2"] = "none",
        rescale_after_l2: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.expansion = float(expansion)
        self.use_layernorm = bool(use_layernorm)
        self.block_norm = str(block_norm)
        self.gated = bool(gated)
        self.residual_dropout = float(residual_dropout)
        self.stochastic_depth = float(stochastic_depth)
        self.residual_scale_init = residual_scale_init
        self.residual_scale_warmup = residual_scale_warmup
        self.residual_scale_last_k = int(residual_scale_last_k)
        self.zero_init_last_block = bool(zero_init_last_block)
        self.depth_scale_ff2 = bool(depth_scale_ff2)
        self.output_dim = int(output_dim) if output_dim is not None else int(hidden_dim)
        self.final_norm = str(final_norm)
        self.rescale_after_l2 = bool(rescale_after_l2)
        self.act = _act_factory(activation)

        if self.use_layernorm and self.block_norm == "pre":
            self.input_norm = nn.Identity()
        else:
            self.input_norm = nn.LayerNorm(self.input_dim) if self.use_layernorm else nn.Identity()

        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim) if self.input_dim != self.hidden_dim else nn.Identity()
        inner = int(math.ceil(max(4.0, self.hidden_dim * self.expansion) / 64.0) * 64.0)

        self.blocks = nn.ModuleList()
        self.pre_norms = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.residual_drops = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        self.res_scales = nn.ParameterList()
        self._res_scale_no_tanh: List[bool] = []

        for layer_idx in range(self.num_layers):
            if self.block_norm == "pre":
                self.pre_norms.append(nn.LayerNorm(self.hidden_dim) if self.use_layernorm else nn.Identity())
                self.post_norms.append(nn.Identity())
            elif self.block_norm == "post":
                self.pre_norms.append(nn.Identity())
                self.post_norms.append(nn.LayerNorm(self.hidden_dim) if self.use_layernorm else nn.Identity())
            else:
                self.pre_norms.append(nn.Identity())
                self.post_norms.append(nn.Identity())

            self.residual_drops.append(nn.Dropout(self.residual_dropout) if self.residual_dropout > 0.0 else nn.Identity())

            if self.num_layers > 1:
                dp_rate = self.stochastic_depth * layer_idx / max(1, self.num_layers - 1)
            else:
                dp_rate = 0.0
            self.drop_paths.append(DropPath(dp_rate))

            if self.residual_scale_init is None:
                self.res_scales.append(nn.Parameter(torch.tensor(1.0), requires_grad=False))
                self._res_scale_no_tanh.append(True)
            else:
                if (self.residual_scale_warmup is not None) and (
                    layer_idx < max(0, self.num_layers - self.residual_scale_last_k)
                ):
                    init_val = float(self.residual_scale_warmup)
                else:
                    init_val = float(self.residual_scale_init)
                self.res_scales.append(nn.Parameter(torch.tensor(init_val)))
                self._res_scale_no_tanh.append(False)

            ff1_out = 2 * inner if self.gated else inner
            ff1 = nn.Linear(self.hidden_dim, ff1_out)
            ff2 = nn.Linear(inner, self.hidden_dim)
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ff1": ff1,
                        "ff2": ff2,
                        "drop": nn.Dropout(self.dropout),
                    }
                )
            )

        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim) if self.output_dim != self.hidden_dim else None
        if self.final_norm == "layernorm":
            self.out_norm = nn.LayerNorm(self.output_dim)
        else:
            self.out_norm = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                gain = 0.01 if (self.fc_out is not None and mod is self.fc_out) else 1.0
                nn.init.xavier_uniform_(mod.weight, gain=gain)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

        if self.gated:
            for blk in self.blocks:
                ff1 = blk["ff1"]
                w = ff1.weight
                mid = w.size(0) // 2
                nn.init.kaiming_uniform_(w[:mid, :], nonlinearity="relu")
                nn.init.xavier_uniform_(w[mid:, :])
                if ff1.bias is not None:
                    nn.init.zeros_(ff1.bias)
                with torch.no_grad():
                    w[mid:, :].mul_(1.0 / math.sqrt(2.0))

        if self.zero_init_last_block and len(self.blocks) > 0:
            last_ff2 = self.blocks[-1]["ff2"]
            nn.init.zeros_(last_ff2.weight)
            if last_ff2.bias is not None:
                nn.init.zeros_(last_ff2.bias)

        if self.depth_scale_ff2 and self.num_layers > 0:
            scale = 1.0 / math.sqrt(self.num_layers)
            for blk in self.blocks:
                with torch.no_grad():
                    blk["ff2"].weight.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param0 = next(self.parameters(), None)
        target_dtype = param0.dtype if param0 is not None else x.dtype
        x = x.to(target_dtype)

        orig_shape = x.shape
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        x = self.input_norm(x)
        x = self.in_proj(x)

        for idx, (pre_n, block, post_n, res_drop, drop_path) in enumerate(
            zip(self.pre_norms, self.blocks, self.post_norms, self.residual_drops, self.drop_paths)
        ):
            h = pre_n(x)
            if self.gated:
                u, v = torch.chunk(block["ff1"](h), 2, dim=-1)
                h = F.silu(u) * v
            else:
                h = self.act(block["ff1"](h))
            h = block["drop"](h)
            h = block["ff2"](h)
            h = post_n(h)
            h = res_drop(h)

            if self._res_scale_no_tanh[idx]:
                scale = self.res_scales[idx]
            else:
                scale = torch.tanh(self.res_scales[idx])
            x = x + drop_path(h * scale)

        out = self.fc_out(x) if self.fc_out is not None else x

        if len(orig_shape) > 2:
            out = out.view(*orig_shape[:-1], -1)
        elif was_1d and self.output_dim != 1:
            out = out.squeeze(0)

        if self.output_dim == 1:
            out = out.squeeze(-1)
        elif self.final_norm == "l2":
            out = F.normalize(out, p=2, dim=-1)
            if self.rescale_after_l2:
                out = out * math.sqrt(self.output_dim)
        elif self.final_norm == "layernorm":
            out = self.out_norm(out)

        return out


def make_residual_mlp_embedder_v3(
    in_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float,
    activation: str = "GELU",
) -> nn.Module:
    """Factory for V3-like residual encoders used by 2D/3D branches."""

    return MLPEmbedderV3Like(
        input_dim=int(in_dim),
        hidden_dim=int(hidden_dim),
        num_layers=int(n_layers),
        dropout=float(dropout),
        activation=str(activation),
        expansion=2.0,
        use_layernorm=True,
        block_norm="pre",
        gated=True,
        residual_dropout=0.05,
        stochastic_depth=0.05,
        residual_scale_init=0.1,
        residual_scale_warmup=0.01,
        residual_scale_last_k=2,
        zero_init_last_block=True,
        depth_scale_ff2=False,
        output_dim=int(hidden_dim),
        final_norm="none",
        rescale_after_l2=True,
    )


def make_mlp(in_dim: int, hidden_dim: int, n_layers: int, dropout: float, activation: str = "GELU") -> nn.Module:
    """Build a simple MLP with LayerNorm, activation, and Dropout per layer."""

    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(int(n_layers)):
        layers += [
            nn.Linear(d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            _act_factory(activation),
            nn.Dropout(float(dropout)),
        ]
        d = hidden_dim
    return nn.Sequential(*layers)


__all__ = ["DropPath", "MLPEmbedderV3Like", "make_residual_mlp_embedder_v3", "make_mlp"]
