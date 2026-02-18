#!/usr/bin/env python3
"""
mlp_embedder.py

Transformer-style residual MLP embedder for MIL instance embeddings.

This module defines:

- DropPath      : per-sample stochastic depth (used inside residual blocks)
- MLPEmbedderV3 : a token-wise (instance-wise) gated MLP stack that mimics the
                  feed-forward (FFN) sublayer of a Transformer encoder.

Design goals
------------
- Take a fixed-size input descriptor (e.g., 3D fingerprints per conformer)
  and map it to a learned embedding space of dimension `output_dim`.
- Use Transformer-like architecture for stability:
    * Pre-LayerNorm or Post-LayerNorm inside blocks
    * SwiGLU gating (gated MLP)
    * Learnable residual scaling (with warmup schedule)
    * Stochastic depth (DropPath) across layers
    * Optional zero-init of the last block for near-identity start
    * Optional depth-aware scaling of ff2 weights

The embedder is strictly per-instance: it does NOT mix information across
instances; that is the job of the MIL aggregator.
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppl.utils.model_builder.model_components_factory.embedders.base_embedder import (
    EmbedderBase,
)
from ppl.utils.model_builder.model_components_registry import register


class DropPath(nn.Module):
    """
    Per-sample stochastic depth.

    During training, this layer randomly drops the residual branch with
    probability `drop_prob` on a *per-sample* basis, and rescales the
    remaining paths so that the expected output stays unchanged.

    Mathematically, for each sample x:
        with prob 1 - p: y = x / (1 - p)
        with prob p    : y = 0

    This is commonly used in deep residual networks (e.g., DeiT, ConvNeXt)
    to improve generalization.

    Args
    ----
    drop_prob : float, default = 0.0
        Stochastic depth probability. 0.0 means no dropping (identity).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth to the input.

        Args
        ----
        x : torch.Tensor
            Input tensor of arbitrary shape. DropPath treats the *first*
            dimension as batch and broadcasts a binary mask across all
            remaining dimensions.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as `x`, where some residual paths
            may be zeroed during training.
        """
        if self.drop_prob == 0.0 or not self.training:
            # In eval mode or if no dropping is requested, act as identity.
            return x

        keep_prob = 1.0 - self.drop_prob

        # Mask is of shape (batch_size, 1, ..., 1), broadcastable to x.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = rand.floor()  # {0, 1} mask

        # Scale surviving paths by 1 / keep_prob to keep the expectation.
        return x.div(keep_prob) * mask


@register("embedder_type", "mlp_embedder_v3")
class MLPEmbedderV3(EmbedderBase):
    r"""
    Transformer-style residual MLP embedder for MIL instance embeddings.

    This module takes an input feature vector (e.g., conformer fingerprint)
    and transforms it into a learned embedding of dimension `output_dim`
    using a stack of residual MLP blocks.

    The architecture closely mimics the FFN sublayer of a Transformer encoder:
    - (Optional) input LayerNorm
    - Linear projection from input_dim → hidden_dim
    - L residual FFN blocks (SwiGLU-gated MLP + LayerNorm + DropPath +
      learnable residual scale)
    - Optional final Linear head from hidden_dim → output_dim
    - Optional final normalization (LayerNorm or L2-normalization)

    Output dimensionality (default)
    --------------------------------
    - `output_dim = 256` → output shape `[..., 256]`.

    Intended usage
    --------------
    - Applied independently to each instance (e.g., conformer or molecule).
    - The downstream MIL aggregator is responsible for mixing instance
      information and enforcing permutation invariance.

    Key hyperparameters
    -------------------
    input_dim : int
        Dimension of the input descriptor (F).
    hidden_dim : int, default = 256
        Width of the hidden representation (H). All blocks operate in this
        space before the final projection to output_dim.
    num_layers : int, default = 2
        Number of residual FFN blocks.
    expansion : float, default = 2.0
        Expansion factor for the inner width of the MLPs:
            inner ≈ ceil(expansion * hidden_dim / 64) * 64
        (rounded to a multiple of 64 for efficient matmul tiling).
    dropout : float, default = 0.1
        Dropout applied inside the MLP (on the hidden activations).
    use_layernorm : bool, default = True
        Whether to use LayerNorm at all (input + blocks). If False, all
        norms are identities.
    block_norm : {"pre", "post", "none"}, default = "pre"
        Placement of LayerNorm inside residual blocks:
            - "pre"  : LN(x) → MLP → + residual
            - "post" : MLP(x) → LN → + residual
            - "none" : no per-block norm
    activation : {"relu", "gelu", "silu"}, default = "gelu"
        Activation for the non-gated MLP case. Ignored when `gated=True`.
    gated : bool, default = True
        If True, use SwiGLU-style gating (u, v = split(ff1(h)); silu(u) * v).
        If False, use a single-branch MLP with chosen activation.
    output_dim : int, default = 256
        Final embedding dimensionality (D).
    output_bias : Optional[float], default = None
        Optional bias initialization for the final head (`fc_out`).
        If `output_dim == 1`, all biases are filled with this value.
    residual_dropout : float, default = 0.05
        Dropout applied to the MLP output before adding the residual.
    stochastic_depth : float, default = 0.05
        Maximum DropPath probability at the *deepest* block. The per-layer
        probability increases linearly from 0 to this value.
    residual_scale_init : Optional[float], default = 0.1
        Initial value of the learnable residual scale parameter per block.
        If None, residual scale is fixed at 1.0 and not passed through tanh.
    final_norm : {"none", "layernorm", "l2"}, default = "none"
        Optional normalization applied *after* the final projection:
            - "none"      : no normalization
            - "layernorm" : LayerNorm(output_dim)
            - "l2"        : L2-normalize each embedding along the last dim
    rescale_after_l2 : bool, default = True
        When `final_norm == "l2"`, re-scale the L2-normalized embedding by
        sqrt(output_dim) to restore a typical vector length.
    zero_init_last_block : bool, default = True
        If True, zero-initialize the weights (and biases) of the last block's
        ff2 layer so that the model starts near an identity mapping.
    residual_scale_warmup : Optional[float], default = 0.01
        If set, earlier blocks use this smaller residual scale, and only the
        last `residual_scale_last_k` blocks use `residual_scale_init`.
    residual_scale_last_k : int, default = 2
        Number of final blocks that use `residual_scale_init` instead of
        `residual_scale_warmup`.
    depth_scale_ff2 : bool, default = False
        If True, multiply each ff2 weight by 1 / sqrt(num_layers) after init
        for depth-aware scaling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        expansion: float = 2,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        block_norm: Literal["pre", "post", "none"] = "pre",
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        gated: bool = True,
        output_dim: int = 256,
        output_bias: Optional[float] = None,
        residual_dropout: float = 0.05,
        stochastic_depth: float = 0.05,
        residual_scale_init: Optional[float] = 0.1,
        final_norm: Literal["none", "layernorm", "l2"] = "none",
        rescale_after_l2: bool = True,
        zero_init_last_block: bool = True,
        # --- extra knobs ------------------------------------------------------
        residual_scale_warmup: Optional[float] = 0.01,
        residual_scale_last_k: int = 2,
        depth_scale_ff2: bool = False,
    ) -> None:
        super().__init__()

        # Store scalar config values
        self._input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.expansion = float(expansion)
        self.dropout = float(dropout)
        self.residual_dropout = float(residual_dropout)
        self.use_layernorm = bool(use_layernorm)
        self.block_norm = block_norm
        self.gated = bool(gated)
        self.stochastic_depth = float(stochastic_depth)
        self.residual_scale_init = residual_scale_init
        self.final_norm = final_norm
        self.rescale_after_l2 = bool(rescale_after_l2)
        self.zero_init_last_block = bool(zero_init_last_block)
        self.residual_scale_warmup = residual_scale_warmup
        self.residual_scale_last_k = int(residual_scale_last_k)
        self.depth_scale_ff2 = bool(depth_scale_ff2)

        # Output dimensionality (embedding size)
        self._output_dim = int(output_dim)

        # ---------------------------------------------------------------------
        # 1) Activation choice for non-gated (single-branch) MLP
        # ---------------------------------------------------------------------
        # Note: for gated=True we always use SiLU as part of SwiGLU.
        if activation == "gelu":
            self.act = nn.GELU()
            # GELU often pairs better with Xavier-style init.
            self._use_kaiming_base = False
        elif activation in ("silu", "swish"):
            self.act = nn.SiLU()
            self._use_kaiming_base = True
        else:  # "relu"
            self.act = nn.ReLU()
            self._use_kaiming_base = True

        # ---------------------------------------------------------------------
        # 2) Input normalization & stem projection
        # ---------------------------------------------------------------------
        # If we are using pre-norm blocks, we typically do NOT want
        # an extra LayerNorm at the stem.
        if self.use_layernorm and self.block_norm == "pre":
            self.input_norm = nn.Identity()
        else:
            self.input_norm = (
                nn.LayerNorm(self._input_dim) if self.use_layernorm else nn.Identity()
            )

        # Optional linear projection from input_dim → hidden_dim.
        self.in_proj = (
            nn.Linear(self._input_dim, self.hidden_dim)
            if self._input_dim != self.hidden_dim
            else nn.Identity()
        )

        # ---------------------------------------------------------------------
        # 3) Residual FFN blocks
        # ---------------------------------------------------------------------
        self.blocks = nn.ModuleList()
        self.pre_norms = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.residual_drops = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        self.res_scales = nn.ParameterList()
        self._res_scale_no_tanh = []  # flags to skip tanh on residual scale

        # Inner MLP width: expansion * hidden_dim, rounded to multiple of 64.
        inner_float = max(4.0, self.hidden_dim * self.expansion)
        inner = int(math.ceil(inner_float / 64.0) * 64)

        for layer_idx in range(self.num_layers):
            # --- 3a) Norm placement inside the block ------------------------
            if self.block_norm == "pre":
                self.pre_norms.append(
                    nn.LayerNorm(self.hidden_dim) if self.use_layernorm else nn.Identity()
                )
                self.post_norms.append(nn.Identity())
            elif self.block_norm == "post":
                self.pre_norms.append(nn.Identity())
                self.post_norms.append(
                    nn.LayerNorm(self.hidden_dim) if self.use_layernorm else nn.Identity()
                )
            else:  # "none"
                self.pre_norms.append(nn.Identity())
                self.post_norms.append(nn.Identity())

            # --- 3b) Dropout on residual branch -----------------------------
            if self.residual_dropout > 0.0:
                self.residual_drops.append(nn.Dropout(self.residual_dropout))
            else:
                self.residual_drops.append(nn.Identity())

            # --- 3c) DropPath schedule across depth -------------------------
            # Stochastic depth probability increases linearly from 0 at
            # the shallowest block to `stochastic_depth` at the deepest.
            if self.num_layers > 1:
                dp_rate = self.stochastic_depth * layer_idx / max(
                    1, self.num_layers - 1
                )
            else:
                dp_rate = 0.0
            self.drop_paths.append(DropPath(dp_rate) if dp_rate > 0.0 else DropPath(0.0))

            # --- 3d) Learnable residual scale per block ---------------------
            if self.residual_scale_init is None:
                # Fixed scale of 1.0, not passed through tanh.
                scale_param = nn.Parameter(
                    torch.tensor(1.0), requires_grad=False
                )
                self._res_scale_no_tanh.append(True)
            else:
                # Warmup schedule: early blocks use residual_scale_warmup,
                # last K blocks use residual_scale_init.
                if (self.residual_scale_warmup is not None) and (
                    layer_idx < max(0, self.num_layers - self.residual_scale_last_k)
                ):
                    init_val = float(self.residual_scale_warmup)
                else:
                    init_val = float(self.residual_scale_init)
                scale_param = nn.Parameter(torch.tensor(init_val))
                self._res_scale_no_tanh.append(False)

            self.res_scales.append(scale_param)

            # --- 3e) Feed-forward block (gated or not) ----------------------
            if self.gated:
                # SwiGLU: ff1 produces [u, v] and we compute silu(u) * v.
                ff1 = nn.Linear(self.hidden_dim, 2 * inner)
                ff1._is_ff1 = True  # tag for custom initialization
                ff2 = nn.Linear(inner, self.hidden_dim)
            else:
                ff1 = nn.Linear(self.hidden_dim, inner)
                ff1._is_ff1 = True
                ff2 = nn.Linear(inner, self.hidden_dim)

            block = nn.ModuleDict(
                {
                    "ff1": ff1,
                    "ff2": ff2,
                    "drop": nn.Dropout(self.dropout),
                }
            )
            self.blocks.append(block)

        # ---------------------------------------------------------------------
        # 4) Output head: map hidden_dim → output_dim
        # ---------------------------------------------------------------------
        if self._output_dim != self.hidden_dim:
            self.fc_out = nn.Linear(self.hidden_dim, self._output_dim)
            # Optional bias initialization for the final head.
            if output_bias is not None and self.fc_out.bias is not None:
                with torch.no_grad():
                    if self._output_dim == 1:
                        self.fc_out.bias.fill_(float(output_bias))
                    else:
                        self.fc_out.bias.zero_()
        else:
            self.fc_out = None

        # ---------------------------------------------------------------------
        # 5) Final normalization (if any)
        # ---------------------------------------------------------------------
        if self.final_norm == "layernorm":
            self.out_norm = nn.LayerNorm(self._output_dim)
        else:
            self.out_norm = nn.Identity()

        # ---------------------------------------------------------------------
        # 6) Parameter initialization
        # ---------------------------------------------------------------------
        self.reset_parameters()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """
        Initialize all parameters of the embedder.

        Strategy
        --------
        1) Generic init for all Linear layers:
           - `fc_out` uses a very small Xavier gain (0.01) for stability.
           - Other linears use:
               * Kaiming uniform if `_use_kaiming_base=True`
               * Xavier uniform otherwise.
        2) For gated FFN blocks (SwiGLU):
           - First half of ff1 (gate `u`) gets Kaiming init.
           - Second half of ff1 (value `v`) gets Xavier init,
             then scaled by 1/sqrt(2) to balance variances.
        3) Optionally zero-init the last block's ff2 for near-identity start.
        4) Optionally scale all ff2 weights by 1/sqrt(num_layers) for
           depth-aware scaling (similar to DeepNet-style inits).
        """
        # 1) Generic Linear initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Special case: final head
                if self.fc_out is not None and module is self.fc_out:
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    continue

                # Default FFN inits (overridden for gated ff1 below).
                if self._use_kaiming_base:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 2) Custom init for gated FFN (SwiGLU) first layer
        if self.gated:
            for blk in self.blocks:
                ff1 = blk["ff1"]
                W = ff1.weight
                mid = W.size(0) // 2

                # Gate part (u): Kaiming
                nn.init.kaiming_uniform_(W[:mid, :], nonlinearity="relu")

                # Value part (v): Xavier, then scaled by 1/sqrt(2)
                nn.init.xavier_uniform_(W[mid:, :])
                if ff1.bias is not None:
                    nn.init.zeros_(ff1.bias)
                with torch.no_grad():
                    W[mid:, :].mul_(1.0 / math.sqrt(2.0))

        # 3) Optionally zero-init last block's ff2
        if self.zero_init_last_block and len(self.blocks) > 0:
            last_ff2 = self.blocks[-1]["ff2"]
            nn.init.zeros_(last_ff2.weight)
            if last_ff2.bias is not None:
                nn.init.zeros_(last_ff2.bias)

        # 4) Optional depth-aware scaling for ff2
        if self.depth_scale_ff2 and self.num_layers > 0:
            scale = 1.0 / math.sqrt(self.num_layers)
            for blk in self.blocks:
                with torch.no_grad():
                    blk["ff2"].weight.mul_(scale)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def input_dim(self) -> int:
        """Dimension of the input descriptor (F)."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Dimension of the output embedding (D)."""
        return self._output_dim

    @property
    def out_dim(self) -> int:
        """Alias for `output_dim` (for compatibility with some callers)."""
        return self._output_dim

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map input descriptors to instance embeddings.

        High-level view
        ----------------
        1) Optionally cast input to the module's parameter dtype (AMP-safe).
        2) Flatten all leading dimensions except the last (feature) dimension.
           The network then operates on shape [-1, F].
        3) Apply:
            - input_norm
            - in_proj (F → H)
            - L residual FFN blocks (pre/post-LN, SwiGLU, DropPath, residual)
        4) Optionally project H → D via fc_out.
        5) Restore the original leading dimensions.
        6) Apply final normalization (none / L2 / LayerNorm).

        Args
        ----
        x : torch.Tensor
            Input tensor of shape:
                - [F]            (single instance), or
                - [..., F]       (batch / bag / instances, last dim = features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape:
                - [D]            if input was [F] and D != 1, or
                - [..., D]       mirroring the leading dims of the input, or
                - scalar         if output_dim == 1 and leading dims collapse.
        """
        # 1) AMP-safe casting: match the dtype of the module parameters.
        param0 = next(self.parameters(), None)
        target_dtype = param0.dtype if param0 is not None else x.dtype
        x = x.to(target_dtype)

        # 2) Flatten leading dimensions into a single batch dimension.
        orig_shape = x.shape
        was_1d = x.dim() == 1
        if was_1d:
            # [F] -> [1, F]
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            # [..., F] -> [-1, F]
            x = x.reshape(-1, x.size(-1))

        # 3a) Stem: optional input norm + projection to hidden_dim.
        x = self.input_norm(x)   # (-1, F)
        x = self.in_proj(x)      # (-1, H)

        # 3b) Residual FFN blocks
        for idx, (pre_n, block, post_n, res_drop, drop_path) in enumerate(
            zip(self.pre_norms, self.blocks, self.post_norms, self.residual_drops, self.drop_paths)
        ):
            # Normalization (pre- or post-, depending on config)
            h = pre_n(x)

            # First FFN layer: either gated (SwiGLU) or simple activation.
            if self.gated:
                # ff1: H → 2 * inner, split into (u, v)
                u, v = torch.chunk(block["ff1"](h), 2, dim=-1)
                h = F.silu(u) * v  # SwiGLU: silu(u) * v
            else:
                h = self.act(block["ff1"](h))

            # Dropout inside MLP
            h = block["drop"](h)

            # Second FFN layer: inner → H
            h = block["ff2"](h)

            # Optional post-norm
            h = post_n(h)

            # Dropout on residual branch
            h = res_drop(h)

            # Residual scale (tanh-bounded unless fixed at 1.0) + DropPath
            if self._res_scale_no_tanh[idx]:
                scale = self.res_scales[idx]
            else:
                scale = torch.tanh(self.res_scales[idx])
            h = drop_path(h * scale)

            # Residual connection: x ← x + scaled, dropped h
            x = x + h

        # 3c) Optional final head: hidden_dim → output_dim
        out = self.fc_out(x) if self.fc_out is not None else x  # (-1, D or H)

        # 4) Restore original leading dims.
        if len(orig_shape) > 2:
            # [..., F] → [..., D]
            out = out.view(*orig_shape[:-1], -1)
        elif was_1d and self._output_dim != 1:
            # [1, D] → [D]
            out = out.squeeze(0)

        # 5) Final normalization.
        if self._output_dim == 1:
            # Treat scalar outputs specially.
            out = out.squeeze(-1)
        else:
            if self.final_norm == "l2":
                # L2-normalize embeddings along the last dimension.
                out = F.normalize(out, p=2, dim=-1)
                if self.rescale_after_l2:
                    # Restore typical magnitude ~ sqrt(D).
                    out = out * math.sqrt(self._output_dim)
            elif self.final_norm == "layernorm":
                out = self.out_norm(out)

        return out

    # -------------------------------------------------------------------------
    # Introspection helper
    # -------------------------------------------------------------------------
    def describe(self) -> dict:
        """
        Provide a lightweight dictionary summarizing the architecture and
        hyperparameters of this embedder instance.

        Returns
        -------
        dict
            Dictionary with human-readable configuration fields, useful
            for logging, experiment tracking, and debugging.
        """
        inner = int(
            math.ceil(max(4.0, self.hidden_dim * self.expansion) / 64.0) * 64
        )
        return {
            "class": self.__class__.__name__,
            "input_dim": self._input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "expansion": self.expansion,
            "inner_width_rounded": inner,
            "dropout": self.dropout,
            "residual_dropout": self.residual_dropout,
            "stochastic_depth": self.stochastic_depth,
            "residual_scale_init": self.residual_scale_init,
            "residual_scale_warmup": self.residual_scale_warmup,
            "residual_scale_last_k": self.residual_scale_last_k,
            "use_layernorm": self.use_layernorm,
            "block_norm": self.block_norm,
            "norm_type_in_blocks": (
                "LayerNorm" if self.use_layernorm and self.block_norm != "none" else "Identity"
            ),
            "activation_non_gated": self.act.__class__.__name__.lower(),
            "gated": self.gated,
            "output_dim": self._output_dim,
            "final_norm": self.final_norm,
            "rescale_after_l2": self.rescale_after_l2,
            "zero_init_last_block": self.zero_init_last_block,
            "depth_scale_ff2": self.depth_scale_ff2,
        }
