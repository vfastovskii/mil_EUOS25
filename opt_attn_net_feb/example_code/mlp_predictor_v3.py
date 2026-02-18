import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

from ppl.utils.model_builder.model_components_registry import register
from ppl.utils.model_builder.model_components_factory.predictors.base_predictor import PredictorBase


class DropPath(nn.Module):
    """Stochastic depth per sample (a.k.a. DropPath)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast on feature dims
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = rand.floor()
        return x.div(keep_prob) * mask


class _ResFFNBlock(nn.Module):
    """
    Residual FFN block: Pre/Post-LN, (Swi)GLU gating, expansion, DropPath, learnable res scale.

    Enhancements:
      • GLU split-init: value half (v)=Xavier, gate half (g)=Kaiming (then 1/√2 scaling)
      • fc2 init: tiny-nonzero for all blocks except last; last block fc2 is zero-initialized
      • Single dropout site per block (after GLU/activation)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        inner_dim: int,                                   # pre-gating inner width
        activation: Literal["relu", "gelu", "silu"] = "silu",
        use_glu: bool = True,
        use_layernorm: bool = True,
        pre_layer_norm: bool = True,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        res_scale_init: Optional[float] = 0.1,
        *,
        is_last_block: bool = False,
        fc2_gain_non_last: float = 1e-2,                  # NEW: faster “wake up”
        proj_gain: float = 0.5,                           # NEW: softer residual proj
    ):
        super().__init__()
        self.pre_layer_norm = pre_layer_norm
        self.use_glu = bool(use_glu)
        self.is_last_block = bool(is_last_block)
        self.fc2_gain_non_last = float(fc2_gain_non_last)
        self.proj_gain = float(proj_gain)

        # Norms
        self.pre_norm = nn.LayerNorm(in_dim) if (use_layernorm and pre_layer_norm) else nn.Identity()
        self.post_norm = nn.LayerNorm(out_dim) if (use_layernorm and not pre_layer_norm) else nn.Identity()

        # MLP
        self.fc1 = nn.Linear(in_dim, inner_dim * (2 if self.use_glu else 1))
        self.fc2 = nn.Linear(inner_dim, out_dim)

        # Activation (non-gated path); gate ALWAYS uses SiLU
        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
            self._use_kaiming = False
        elif act == "silu":
            self.act = nn.SiLU()
            self._use_kaiming = True
        else:
            self.act = nn.ReLU()
            self._use_kaiming = True

        # Single dropout site (post-activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0.0 else nn.Identity()

        # Projection for residual if dims differ
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # Optional learnable residual scale (tanh-bounded at runtime)
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init))) if res_scale_init is not None else None

        self._init_weights()

    def _init_weights(self):
        # ---- FC1: GLU split-init (fixed halves) ----
        if self.use_glu:
            W = self.fc1.weight
            mid = W.size(0) // 2
            # first half -> value (v), second half -> gate (g)
            nn.init.xavier_uniform_(W[:mid, :])                        # value half (v)
            nn.init.kaiming_uniform_(W[mid:, :], nonlinearity="relu")  # gate half (g)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            # IMPORTANT: scale ONLY the value half, not the gate half
            with torch.no_grad():
                W[:mid, :].mul_(1.0 / math.sqrt(2.0))
        else:
            if self._use_kaiming:
                nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(self.fc1.weight)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)

        # ---- FC2: tiny-nonzero for all but last; last = zero ----
        if self.is_last_block:
            nn.init.zeros_(self.fc2.weight)
        else:
            nn.init.xavier_uniform_(self.fc2.weight, gain=self.fc2_gain_non_last)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

        # ---- Residual projection: softened gain ----
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
            y = F.silu(g) * v                     # SwiGLU (gate always SiLU)
        else:
            y = self.act(y)

        # single dropout site (post-activation, pre-fc2)
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


@register("predictor_type", "mlp_predictor_v3")
class MLPPredictorV3(PredictorBase):
    r"""
    Residual MLP predictor for MIL bag-level outputs.

    This version uses:
      • SwiGLU split-init for fc1 halves (+ 1/√2 scaling)
      • fc2 tiny-nonzero init for all but the last block; last block fc2 zero-init
      • Single dropout site per block (after GLU/activation)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
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
        res_scale_init: Optional[float] = 0.1,
        inner_multiple: Optional[int] = 64,
        head_dropout: float = 0.0,
        output_bias: Optional[float] = None,
        # NEW knobs propagated into blocks:
        fc2_gain_non_last: float = 1e-2,
        proj_gain: float = 0.5,
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"

        self._input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.expansion = float(expansion)
        self.activation_type = activation.lower()
        self.use_glu = bool(use_glu)
        self.dropout = float(dropout)
        self.stochastic_depth = float(stochastic_depth)
        self.use_layernorm = bool(use_layernorm)
        self.pre_layer_norm = bool(pre_layer_norm)
        self._output_dim = int(output_dim)
        self.input_layernorm = bool(input_layernorm)
        self.final_layernorm = bool(final_layernorm)
        self.res_scale_init = res_scale_init
        self.inner_multiple = inner_multiple
        self.head_dropout = float(head_dropout)
        self.output_bias = output_bias
        self.fc2_gain_non_last = float(fc2_gain_non_last)
        self.proj_gain = float(proj_gain)

        # Optional input normalization (stabilizes bag_repr scale)
        self.input_norm = nn.LayerNorm(self._input_dim) if (self.input_layernorm and self.use_layernorm) else nn.Identity()

        # Helper to compute inner width with optional rounding
        def _inner_width(base: int, mult: float, multiple: Optional[int]) -> int:
            w = max(4, int(round(base * mult)))
            if multiple and multiple > 1:
                w = int(math.ceil(w / multiple) * multiple)
            return w

        # Build residual blocks
        self.blocks = nn.ModuleList()
        d_in = self._input_dim
        for i in range(self.num_layers):
            dp = self.stochastic_depth * (i / max(1, self.num_layers - 1)) if self.num_layers > 1 else 0.0
            inner = _inner_width(self.hidden_dim, self.expansion, self.inner_multiple)
            block = _ResFFNBlock(
                in_dim=d_in,
                out_dim=self.hidden_dim,
                inner_dim=inner,
                activation=self.activation_type,
                use_glu=self.use_glu,
                use_layernorm=self.use_layernorm,
                pre_layer_norm=self.pre_layer_norm,
                dropout=self.dropout,
                drop_path=dp,
                res_scale_init=self.res_scale_init,
                is_last_block=(i == self.num_layers - 1),
                fc2_gain_non_last=self.fc2_gain_non_last,
                proj_gain=self.proj_gain,
            )
            self.blocks.append(block)
            d_in = self.hidden_dim

        # Pre-head norm + dropout
        self.out_norm = nn.LayerNorm(self.hidden_dim) if (self.final_layernorm and self.use_layernorm) else nn.Identity()
        self.pre_head_drop = nn.Dropout(self.head_dropout) if self.head_dropout > 0.0 else nn.Identity()

        # Output head
        self.fc_out = nn.Linear(self.hidden_dim, self._output_dim)
        self._init_head()

    # ----- Initialization -----
    def _init_head(self):
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.01)
        if self.fc_out.bias is not None:
            if self._output_dim == 1 and self.output_bias is not None:
                nn.init.constant_(self.fc_out.bias, float(self.output_bias))
            else:
                nn.init.zeros_(self.fc_out.bias)

    # ----- Forward -----
    def forward_features(self, bag_repr: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(bag_repr)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        x = self.pre_head_drop(x)
        return x

    def forward(self, bag_repr: torch.Tensor) -> torch.Tensor:
        """
        bag_repr: [D] or [B, D]
        returns: scalar [] or [B]  (or [B, C] if output_dim=C>1, without activation)
        """
        single = bag_repr.dim() == 1
        if single:
            bag_repr = bag_repr.unsqueeze(0)  # [1, D]

        x = self.forward_features(bag_repr)
        y = self.fc_out(x)

        if single:
            y = y.squeeze(0)
        return y.squeeze(-1) if self._output_dim == 1 else y

    # ----- Introspection / dynamic resize -----
    @property
    def output_dim(self) -> int:
        return self.fc_out.out_features

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value: int) -> None:
        """Rebuilds the first block projection & input norm; keeps the rest intact."""
        value = int(value)
        if self._input_dim == value:
            return
        self._input_dim = value
        self.input_norm = nn.LayerNorm(self._input_dim) if (self.input_layernorm and self.use_layernorm) else nn.Identity()

        # Rebuild first block to accept new input dim, preserving inner / gating config
        first = self.blocks[0]
        current_inner = first.fc1.out_features // (2 if self.use_glu else 1)
        is_single = (self.num_layers == 1)  # KEEP last-block semantics if single-layer model
        self.blocks[0] = _ResFFNBlock(
            in_dim=self._input_dim,
            out_dim=first.fc2.out_features,      # equals hidden_dim
            inner_dim=current_inner,
            activation=self.activation_type,
            use_glu=self.use_glu,
            use_layernorm=self.use_layernorm,
            pre_layer_norm=self.pre_layer_norm,
            dropout=self.dropout,
            drop_path=0.0,                       # first block: usually no drop-path
            res_scale_init=self.res_scale_init,
            is_last_block=is_single,
            fc2_gain_non_last=self.fc2_gain_non_last,
            proj_gain=self.proj_gain,
        )

    def describe(self) -> dict:
        inners = [blk.fc1.out_features // (2 if self.use_glu else 1) for blk in self.blocks]
        return {
            "class": self.__class__.__name__,
            "input_dim": self._input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "expansion": self.expansion,
            "inner_multiple": self.inner_multiple,
            "inner_widths": inners,
            "activation": self.activation_type,
            "use_glu": self.use_glu,
            "dropout": self.dropout,
            "stochastic_depth": self.stochastic_depth,
            "use_layernorm": self.use_layernorm,
            "pre_layer_norm": self.pre_layer_norm,
            "input_layernorm": self.input_layernorm,
            "final_layernorm": self.final_layernorm,
            "head_dropout": self.head_dropout,
            "output_dim": self.fc_out.out_features,
            "output_bias": self.output_bias,
            "output_head": str(self.fc_out),
        }
