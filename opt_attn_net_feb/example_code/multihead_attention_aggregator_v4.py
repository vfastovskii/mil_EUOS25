import math
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ppl.utils.model_builder.model_components_registry import register


@register("aggregator_type", "mha_att_v4")
class MultiHeadAttentionAggregatorV4(nn.Module):
    r"""
    Multi-head self-attention aggregator with optional multi-CLS queries.

    New features:
    - topk_n: keep only the top-k instances by attention when pooling (k=1 → pass the single best instance's features)
    - topk_strategy: how to combine the selected top-k ("renorm" | "mean" | "sum" | "argmax")

    Inputs
    ------
    h : [N, D] or [B, N, D]
    key_padding_mask : Optional[BoolTensor] with shape [B, N] where True marks PAD (ignored by attention)

    Outputs
    -------
    bag_repr : [D] or [B, D]
    extras : {
        "alpha":     [N] or [B, N],                      # full attention over all instances
        "alpha_std": [N] or [B, N],
        "entropy":   scalar or [B],
        "attn":      [B, H, 1, N] (CLS mode) or [B, H, N, N] (self-attn mode), if requested
        "alpha_topk": [B, k] or [k] (when topk_n>0),     # selected weights (pre-renorm) in descending order
        "alpha_topk_idx": [B, k] or [k] (when topk_n>0)  # indices of selected instances
    }
    """

    def __init__(
        self,
        input_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.00,
        attn_dropout: float = 0.0,                          # NEW: separate attention dropout
        use_layer_norm: bool = True,
        std_correction: float = 0.0,
        use_checkpoint: bool = True,
        eps: float = 1e-8,
        prune_below = 0.0, #0.001 - paper
        pre_layer_norm: bool = True,
        use_cls_token: bool = True,
        num_cls_tokens: int = 2,
        use_weighted_sum: bool = True,
        # --- pooling config ---
        pool_from: Literal["attn_out", "inputs", "normed_inputs"] = "normed_inputs", # "attn_out", # "normed_inputs - paper",
        pool_v_proj: Union[bool, Literal[False, "linear", "tie_mha_v"]] = "tie_mha_v", # "tie_mha_v - paper",
        pool_v_bias: bool = False,
        residual_pooling: bool = False,
        residual_pool_from: Literal["same", "inputs", "normed_inputs"] = "same",   # NEW
        residual_mix_learnable: bool = True,                                         # NEW (learnable blend)
        # --- top-k pooling knobs ---
        topk_n: int = 0,  # 0 → disabled; k>=1 keeps only top-k instances by attention
        topk_strategy: Literal["renorm", "mean", "sum", "argmax"] = "renorm",
        # --- temperature on Q / CLS Q ---
        use_temperature: bool = False,
        temperature_init: float = 0.3,
        # --- out-proj init ---
        out_proj_init: Literal["zero", "tiny"] = "tiny",                              # NEW
        # --- multi-CLS mixer ---
        multi_cls_mixer: bool = True,                                               # NEW: learned convex mix over K
    ):
        super().__init__()

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self._input_dim = self.out_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.head_dim = input_dim // num_heads
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.use_layernorm = bool(use_layer_norm)
        self.std_correction = float(std_correction)
        self.use_checkpoint = bool(use_checkpoint)
        self.eps = float(eps)
        self.pre_layer_norm = bool(pre_layer_norm)
        self.prune_below = float(prune_below)

        # ---- CLS queries config ----
        if num_cls_tokens > 0:
            self.num_cls_tokens = int(num_cls_tokens)
            self.use_cls_token = True
        else:
            self.use_cls_token = bool(use_cls_token)
            self.num_cls_tokens = 1 if self.use_cls_token else 0

        self.use_weighted_sum = bool(use_weighted_sum)

        # --- pooling knobs
        self.pool_from = pool_from
        # normalize legacy bool into mode
        if isinstance(pool_v_proj, bool):
            self.pool_v_mode = "linear" if pool_v_proj else False
        else:
            self.pool_v_mode = pool_v_proj
        self.residual_pooling = bool(residual_pooling)
        self.residual_pool_from = residual_pool_from
        self.residual_mix_learnable = bool(residual_mix_learnable)
        if self.residual_mix_learnable:
            # sigmoid(0)=0.5 → equal blend at start
            self.residual_mix = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("residual_mix", None)

        # --- top-k pooling
        self.topk_n = int(topk_n)
        self.topk_strategy = topk_strategy

        # Multi-head attention (batch_first for [B, N, D])
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=self.attn_dropout,   # use dedicated attn dropout
            bias=True,
            batch_first=True,
        )

        # Layer norms
        self.pre_ln = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.post_ln = nn.LayerNorm(input_dim) if (use_layer_norm and not pre_layer_norm) else nn.Identity()

        # Residual dropout after attention
        self.post_dropout = nn.Dropout(dropout)

        # Output projection with residual
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout * 0.5),
        )

        # CLS query parameters (K learnable queries)
        if self.use_cls_token:
            self.cls_tokens = nn.Parameter(torch.randn(1, self.num_cls_tokens, input_dim) * 0.02)
            self.cls_ln = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        else:
            self.register_parameter("cls_tokens", None)
            self.cls_ln = nn.Identity()

        # Multi-CLS learned convex mixing over queries
        self.multi_cls_mixer = bool(multi_cls_mixer)
        if self.use_cls_token and self.num_cls_tokens > 1 and self.multi_cls_mixer:
            # logits for a convex combination across K queries
            self.cls_mix_logits = nn.Parameter(torch.zeros(self.num_cls_tokens))
        else:
            self.register_parameter("cls_mix_logits", None)

        # 1) Value projection for pooling
        if self.pool_v_mode == "linear":
            self.pool_v = nn.Linear(input_dim, input_dim, bias=pool_v_bias)
            nn.init.xavier_uniform_(self.pool_v.weight)
            if self.pool_v.bias is not None:
                nn.init.zeros_(self.pool_v.bias)
        else:
            self.pool_v = nn.Identity()  # for False or "tie_mha_v"

        # 2) Learnable temperature τ (query scaling only)
        self.use_temperature = bool(use_temperature)
        self.log_tau = nn.Parameter(torch.tensor(math.log(float(temperature_init))), requires_grad=True)

        # out-proj init mode
        self.out_proj_init = out_proj_init

        # Init
        self._init_weights()

        # For debugging/visualization
        self.last_attn = None  # [B, H, K, N] (CLS mode) or [B, H, N, N] (self-attn mode)

    # --- helpers ---------------------------------------------------------------

    def _tau(self):
        return self.log_tau.exp().clamp(0.1, 10.0)

    def _project_for_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Project tokens into pooling value space according to the selected mode."""
        if self.pool_v_mode == "linear":
            return self.pool_v(x)
        elif self.pool_v_mode == "tie_mha_v":
            D = self._input_dim
            # V slice of in_proj: [3D, D] → rows [2D:3D]
            W = self.mha.in_proj_weight[2 * D : 3 * D]
            b = None if self.mha.in_proj_bias is None else self.mha.in_proj_bias[2 * D : 3 * D]
            return F.linear(x, W, b)
        else:
            return x

    def _init_weights(self):
        """Stable defaults."""
        # Attention out-proj
        if hasattr(self.mha, "out_proj") and hasattr(self.mha.out_proj, "weight"):
            if self.out_proj_init == "zero":
                nn.init.zeros_(self.mha.out_proj.weight)
            else:  # "tiny"
                nn.init.xavier_uniform_(self.mha.out_proj.weight, gain=1e-3)
            if self.mha.out_proj.bias is not None:
                nn.init.zeros_(self.mha.out_proj.bias)

        # Small projection after residual
        if hasattr(self.output_projection[0], "weight"):
            nn.init.xavier_uniform_(self.output_projection[0].weight, gain=0.02)
            if self.output_projection[0].bias is not None:
                nn.init.constant_(self.output_projection[0].bias, 0)

    def _topk_pool(
        self, alpha: torch.Tensor, pool_source: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply top-k selection over instances using attention weights.
        alpha: [B, N] or [N]
        pool_source: [B, N, D] or [N, D]
        Returns [B, D] or [D]
        """
        squeeze_out = False
        if alpha.dim() == 1:  # [N]
            alpha = alpha.unsqueeze(0)
            pool_source = pool_source.unsqueeze(0)
            squeeze_out = True

        B, N = alpha.shape
        D = pool_source.shape[-1]

        # --- HARD PRUNE (exact zeros) + renorm; safe fallback if all are zero ---
        if getattr(self, "prune_below", 0.0) > 0.0:
            a = alpha.clone()
            a = a.masked_fill(a < self.prune_below, 0.0)  # zero-out small weights
            s = a.sum(dim=-1, keepdim=True)
            need_fallback = (s <= self.eps).squeeze(
                -1
            )  # all pruned? fallback to argmax
            if need_fallback.any():
                onehot = torch.zeros_like(a)
                argmax_idx = alpha.argmax(dim=-1, keepdim=True)  # use pre-pruned argmax
                onehot.scatter_(1, argmax_idx, 1.0)
                a = torch.where(need_fallback.unsqueeze(-1), onehot, a)
                s = a.sum(dim=-1, keepdim=True)
            alpha = a / s.clamp_min(self.eps)  # re-normalize

        # No top-k or k>=N → standard weighted sum
        if self.topk_n <= 0 or self.topk_n >= N:
            out = torch.bmm(alpha.unsqueeze(1), pool_source).squeeze(1)  # [B, D]
            return out.squeeze(0) if squeeze_out else out

        k = max(1, min(self.topk_n, N))
        topv, topi = torch.topk(alpha, k, dim=-1, largest=True, sorted=True)  # [B, k]

        # Gather selected token features: [B, k, D]
        idx = topi.unsqueeze(-1).expand(-1, -1, D)
        selected = torch.gather(pool_source, dim=1, index=idx)

        if k == 1:
            out = selected.squeeze(1)  # [B, D] — "pass the best instance"
            return out.squeeze(0) if squeeze_out else out

        if self.topk_strategy == "mean":
            out = selected.mean(dim=1)
        elif self.topk_strategy == "sum":
            out = selected.sum(dim=1)
        elif self.topk_strategy == "argmax":
            out = selected[:, 0, :]  # strictly the top-1
        else:  # "renorm" — re-normalize weights within the top-k
            w = topv / topv.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # [B, k]
            out = (w.unsqueeze(-1) * selected).sum(dim=1)

        return out.squeeze(0) if squeeze_out else out

    # --- Core attention blocks -------------------------------------------------

    def _attention_block(self, h_seq, key_padding_mask=None):
        """
        One MHA block with residuals.
        h_seq: [B, N, D]
        key_padding_mask: [B, N] True=PAD (ignored by attention)
        """
        normed_h = self.pre_ln(h_seq)

        # Scale queries by τ
        q_in = normed_h / self._tau() if self.use_temperature else normed_h

        # Self-attention
        attn_out, attn_w = self.mha(
            q_in, normed_h, normed_h,
            need_weights=True,
            average_attn_weights=False,
            key_padding_mask=key_padding_mask,
        )  # attn_out: [B, N, D], attn_w: [B, H, N, N]

        # Residual + (optional) post-LN
        attn_out = self.post_dropout(attn_out) + h_seq        # [B, N, D]
        attn_out = self.post_ln(attn_out)                     # [B, N, D]

        # Output projection with residual
        output = self.output_projection(attn_out) + attn_out  # [B, N, D]
        return output, attn_w

    def _checkpointed_attention(self, h_seq, key_padding_mask=None):
        """Memory-efficient block; recomputes attn_w under no_grad for logging."""
        def block_only(x, mask):
            out, _ = self._attention_block(x, key_padding_mask=mask)
            return out

        attn_out = checkpoint.checkpoint(block_only, h_seq, key_padding_mask, use_reentrant=False)

        with torch.no_grad():
            normed_h = self.pre_ln(h_seq)
            q_in = normed_h / self._tau() if self.use_temperature else normed_h
            _, attn_w = self.mha(
                q_in, normed_h, normed_h,
                need_weights=True,
                average_attn_weights=False,
                key_padding_mask=key_padding_mask,
            )
        return attn_out, attn_w

    # --- Forward ---------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_entropy: bool = True,
        return_attn: bool = False,
    ):
        # Ensure batch
        if h.dim() == 2:  # [N, D]
            h_seq = h.unsqueeze(0)        # [1, N, D]
        else:
            h_seq = h                     # [B, N, D]
        B, N, D = h_seq.shape

        extras_attn = None  # ensure defined
        alpha = None
        alpha_std = None
        alpha_topk = None
        alpha_topk_idx = None

        if self.use_cls_token:
            # --- K learnable CLS queries attending into inputs ---
            normed_h = self.pre_ln(h_seq)                        # [B, N, D]
            queries = self.cls_ln(self.cls_tokens).expand(B, -1, -1)  # [B, K, D]
            if self.use_temperature:
                queries = queries / self._tau()

            cls_out, attn_w = self.mha(
                query=queries,
                key=normed_h,
                value=normed_h,
                need_weights=True,
                average_attn_weights=False,
                key_padding_mask=key_padding_mask,
            )   # cls_out: [B, K, D], attn_w: [B, H, K, N]

            self.last_attn = attn_w.detach()

            # Head-averaged per-CLS weights over instances → [B, K, N]
            alpha_q = attn_w.mean(dim=1)
            if key_padding_mask is not None:
                alpha_q = alpha_q.masked_fill(key_padding_mask.unsqueeze(1), 0.0)

            # Normalize each CLS's weights across N
            alpha_q = alpha_q / alpha_q.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            # Aggregate the K query distributions
            if self.num_cls_tokens > 1:
                if self.multi_cls_mixer and self.cls_mix_logits is not None:
                    # convex combination over K (softmax-normalized weights)
                    w = torch.softmax(self.cls_mix_logits, dim=0)  # [K]
                    alpha = (alpha_q * w.view(1, -1, 1)).sum(dim=1)  # [B, N]
                else:
                    alpha = alpha_q.mean(dim=1)                      # [B, N]
                alpha_std = alpha_q.std(dim=1, correction=self.std_correction)
            else:
                alpha = alpha_q.squeeze(1)                           # [B, N]
                alpha_std = torch.zeros_like(alpha)

            # --- Pooling source and learnable blend with mean ---
            if self.pool_from == "inputs":
                pool_source = self._project_for_pool(h_seq)
            elif self.pool_from == "normed_inputs":
                pool_source = self._project_for_pool(self.pre_ln(h_seq))
            else:
                # CLS path has no per-instance attn_out; use normed inputs by default
                pool_source = self._project_for_pool(self.pre_ln(h_seq))

            # Record top-k indices/weights for extras (before any renorm)
            if self.topk_n > 0 and self.topk_n < N:
                k = max(1, min(self.topk_n, N))
                alpha_topk, alpha_topk_idx = torch.topk(alpha, k, dim=-1, largest=True, sorted=True)

            # Apply top-k selection + pooling
            attn_pool = self._topk_pool(alpha, pool_source)  # [B, D] or [D]

            if self.residual_pooling:
                if self.residual_pool_from == "same":
                    res_source = pool_source
                elif self.residual_pool_from == "inputs":
                    res_source = self._project_for_pool(h_seq)
                else:  # "normed_inputs"
                    res_source = self._project_for_pool(self.pre_ln(h_seq))

                mean_pool = res_source.mean(dim=1)  # [B, D]
                if self.residual_mix_learnable and self.residual_mix is not None:
                    m = torch.sigmoid(self.residual_mix)
                    bag_repr = (1.0 - m) * attn_pool + m * mean_pool
                else:
                    bag_repr = attn_pool + mean_pool
            else:
                bag_repr = attn_pool

            if return_attn:
                extras_attn = self.last_attn.mean(dim=2, keepdim=True)  # [B, H, 1, N]

        else:
            # --- No CLS: self-attention + pooling path ---
            if self.use_checkpoint and self.training:
                attn_out, attn_w = self._checkpointed_attention(h_seq, key_padding_mask)
            else:
                attn_out, attn_w = self._attention_block(h_seq, key_padding_mask)

            self.last_attn = attn_w.detach()  # [B, H, N, N]

            attn_head_mean = attn_w.mean(dim=1)              # [B, N, N]
            if key_padding_mask is not None:
                # Mask out contributions from padded KEYS (columns)
                key_valid_f = (~key_padding_mask).float()     # [B, N]
                attn_head_mean = attn_head_mean * key_valid_f.unsqueeze(1)

                # Compute mean over QUERIES using only valid (un-padded) queries
                query_valid_f = (~key_padding_mask).float()   # [B, N]
                sum_over_q = (attn_head_mean * query_valid_f.unsqueeze(-1)).sum(dim=1)  # [B, N]
                denom_q = query_valid_f.sum(dim=-1, keepdim=True).clamp_min(self.eps)   # [B, 1]
                alpha_raw = sum_over_q / denom_q                                                # [B, N]

                # Ensure padded KEYS are strictly zero before final renorm
                alpha_raw = alpha_raw.masked_fill(key_padding_mask, 0.0)
            else:
                # No mask provided: simple mean over queries
                alpha_raw = attn_head_mean.mean(dim=1)           # [B, N]

            # Final normalization across valid positions (padded keys remain zero)
            alpha = alpha_raw / alpha_raw.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            alpha_std = (
                attn_head_mean.std(dim=1, correction=self.std_correction)
                if attn_head_mean.size(1) > 1 else torch.zeros_like(alpha)
            )

            if self.use_weighted_sum:
                if self.pool_from == "attn_out":
                    pool_source = attn_out
                elif self.pool_from == "normed_inputs":
                    pool_source = self._project_for_pool(self.pre_ln(h_seq))
                else:
                    pool_source = self._project_for_pool(h_seq)

                # Record top-k indices/weights for extras (before any renorm)
                if self.topk_n > 0 and self.topk_n < N:
                    k = max(1, min(self.topk_n, N))
                    alpha_topk, alpha_topk_idx = torch.topk(alpha, k, dim=-1, largest=True, sorted=True)

                # Apply top-k selection + pooling
                attn_pool = self._topk_pool(alpha, pool_source).squeeze(0) if h.dim()==2 else self._topk_pool(alpha, pool_source)

                if self.residual_pooling:
                    if self.residual_pool_from == "same":
                        res_source = pool_source
                    elif self.residual_pool_from == "inputs":
                        res_source = self._project_for_pool(h_seq)
                    else:
                        res_source = self._project_for_pool(self.pre_ln(h_seq))
                    mean_pool = res_source.mean(dim=1)
                    if self.residual_mix_learnable and self.residual_mix is not None:
                        m = torch.sigmoid(self.residual_mix)
                        bag_repr = (1.0 - m) * attn_pool + m * mean_pool
                    else:
                        bag_repr = attn_pool + mean_pool
                else:
                    bag_repr = attn_pool
            else:
                # Mean pooling over tokens when not using attention weights
                bag_repr = attn_out.mean(dim=1)  # [B, D]

            if return_attn:
                extras_attn = self.last_attn  # [B, H, N, N]

        # Final safety: enforce zero on padded positions and renormalize across valid tokens
        if key_padding_mask is not None:
            if alpha.dim() == 1 and key_padding_mask.dim() == 2:
                # Handle rare case where single-bag input passed with [1,N] mask
                km = key_padding_mask.squeeze(0)
            else:
                km = key_padding_mask
            # Zero-out padded positions and renormalize per bag
            alpha = alpha.masked_fill(km, 0.0)
            denom = alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            alpha = alpha / denom

        # Final numeric safety: clamp and renormalize in-place (preserve dtype/device; MPS has no float64)
        with torch.no_grad():
            a = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            # clip negatives and re-normalize
            a = a.clamp(min=0.0)
            s = a.sum(dim=-1, keepdim=True)
            a = torch.where(s > 0, a / s.clamp_min(self.eps), a)
            # clip to [0,1] and renormalize again to counter tiny drifts
            a = a.clamp(0.0, 1.0)
            s = a.sum(dim=-1, keepdim=True)
            a = torch.where(s > 0, a / s.clamp_min(self.eps), a)
            alpha = a

        # Squeeze for single-bag inputs
        if h.dim() == 2:
            bag_repr = bag_repr.squeeze(0)    # [D]
            alpha = alpha.squeeze(0)          # [N]
            alpha_std = alpha_std.squeeze(0)  # [N]
            if alpha_topk is not None:
                alpha_topk = alpha_topk.squeeze(0)
                alpha_topk_idx = alpha_topk_idx.squeeze(0)

        # Extras
        extras = {"alpha": alpha, "alpha_std": alpha_std}
        if return_entropy:
            alpha_safe = alpha.clamp_min(self.eps)
            log_alpha = alpha_safe.log()
            extras["entropy"] = (-(alpha_safe * log_alpha).sum()
                                 if alpha.dim() == 1 else
                                 -(alpha_safe * log_alpha).sum(dim=1))
        if return_attn and extras_attn is not None:
            extras["attn"] = extras_attn
        if alpha_topk is not None:
            extras["alpha_topk"] = alpha_topk
            extras["alpha_topk_idx"] = alpha_topk_idx

        return bag_repr, extras

    # --- Properties & dynamic resizing ---------------------------------------

    @property
    def output_dim(self):
        return self.out_dim

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value: int):
        if self._input_dim == value:
            return
        assert value > 0
        self._input_dim = int(value)
        self.out_dim = int(value)

        # Ensure num_heads divides new dim
        if value % self.num_heads != 0:
            for i in range(self.num_heads, 0, -1):
                if value % i == 0:
                    self.num_heads = i
                    break
        self.head_dim = value // self.num_heads

        # Rebuild attention (uses attn_dropout)
        self.mha = nn.MultiheadAttention(
            embed_dim=value,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            bias=True,
            batch_first=True,
        )
        # Norms & post block
        self.pre_ln = nn.LayerNorm(value) if self.use_layernorm else nn.Identity()
        self.post_ln = nn.LayerNorm(value) if (self.use_layernorm and not self.pre_layer_norm) else nn.Identity()
        self.output_projection = nn.Sequential(
            nn.Linear(value, value),
            nn.Dropout(self.dropout * 0.5),
        )

        # CLS path
        if self.use_cls_token:
            self.cls_tokens = nn.Parameter(torch.randn(1, self.num_cls_tokens, value) * 0.02)
            self.cls_ln = nn.LayerNorm(value) if self.use_layernorm else nn.Identity()

        # pool_v: only rebuild the explicit linear
        if self.pool_v_mode == "linear":
            self.pool_v = nn.Linear(value, value, bias=self.pool_v.bias is not None)
            nn.init.xavier_uniform_(self.pool_v.weight)
            if self.pool_v.bias is not None:
                nn.init.zeros_(self.pool_v.bias)
        else:
            self.pool_v = nn.Identity()

        self._init_weights()

    # --- Introspection ---------------------------------------------------------

    def describe(self) -> dict:
        # expose residual mix (sigmoid) if learnable
        res_mix = float(torch.sigmoid(self.residual_mix).detach().cpu()) if (
            self.residual_mix_learnable and self.residual_mix is not None
        ) else None

        # expose multi-CLS mixer weights if present
        cls_mix = None
        if self.use_cls_token and self.num_cls_tokens > 1 and self.multi_cls_mixer and self.cls_mix_logits is not None:
            w = torch.softmax(self.cls_mix_logits.detach().cpu(), dim=0)
            cls_mix = [float(x) for x in w.tolist()]

        return {
            "class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.out_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "use_layernorm": self.use_layernorm,
            "pre_layer_norm": self.pre_layer_norm,
            "use_checkpoint": self.use_checkpoint,
            "use_cls_token": self.use_cls_token,
            "num_cls_tokens": self.num_cls_tokens,
            "multi_cls_mixer": self.multi_cls_mixer,
            "multi_cls_weights": cls_mix,
            "use_weighted_sum": self.use_weighted_sum,
            "residual_pooling": self.residual_pooling,
            "residual_pool_from": self.residual_pool_from,
            "residual_mix_learnable": self.residual_mix_learnable,
            "residual_mix_sigma": res_mix,
            "pool_from": self.pool_from,
            "pool_v_mode": self.pool_v_mode,
            "use_temperature": self.use_temperature,
            "tau": float(self._tau().detach().cpu()) if self.use_temperature else None,
            "out_proj_init": self.out_proj_init,
            # NEW
            "topk_n": self.topk_n,
            "topk_strategy": self.topk_strategy,
            "architecture": {
                "out_proj": str(self.mha.out_proj),
                "post_projection": str(self.output_projection[0]),
                "pool_v_is_linear": isinstance(self.pool_v, nn.Linear),
                "pooling_method": (
                    f"{self.num_cls_tokens}×CLS → weights on instances (pool: {self.pool_from})"
                    if self.use_cls_token
                    else (f"Weighted sum from {self.pool_from}" + (" + residual mean/blend" if self.residual_pooling else ""))
                    if self.use_weighted_sum
                    else "Mean pooling"
                ),
            },
        }
