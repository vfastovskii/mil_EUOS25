from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masking import mask_and_normalize_attention


class TaskAttentionPool(nn.Module):
    """
    Task-query attention pooling with V4-style aggregator logic.

    Inputs:
      - tokens: [B, N, D]
      - key_padding_mask: [B, N] (True marks padding)

    Outputs:
      - pooled: [B, T, D]
      - attn:   [B, T, N] (full per-task attention over instances) or None
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float,
        n_tasks: int = 4,
        *,
        attn_dropout: Optional[float] = None,
        use_layer_norm: bool = True,
        pre_layer_norm: bool = True,
        pool_from: Literal["attn_out", "inputs", "normed_inputs"] = "normed_inputs",
        pool_v_mode: Literal[False, "linear", "tie_mha_v"] = "tie_mha_v",
        pool_v_bias: bool = False,
        residual_pooling: bool = False,
        residual_pool_from: Literal["same", "inputs", "normed_inputs"] = "same",
        residual_mix_learnable: bool = True,
        topk_n: int = 0,
        topk_strategy: Literal["renorm", "mean", "sum", "argmax"] = "renorm",
        use_temperature: bool = False,
        temperature_init: float = 0.3,
        prune_below: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        dim = int(dim)
        n_heads = int(n_heads)
        n_tasks = int(n_tasks)
        if dim % n_heads != 0:
            raise ValueError(f"TaskAttentionPool: dim={dim} must be divisible by n_heads={n_heads}")

        self.n_tasks = n_tasks
        self.dim = dim
        self.eps = float(eps)
        self.pre_layer_norm = bool(pre_layer_norm)
        self.pool_from = pool_from
        self.pool_v_mode = pool_v_mode
        self.residual_pooling = bool(residual_pooling)
        self.residual_pool_from = residual_pool_from
        self.residual_mix_learnable = bool(residual_mix_learnable)
        self.topk_n = int(topk_n)
        self.topk_strategy = topk_strategy
        self.use_temperature = bool(use_temperature)
        self.prune_below = float(prune_below)

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=float(attn_dropout if attn_dropout is not None else dropout),
            batch_first=True,
            bias=True,
        )

        self.pre_ln = nn.LayerNorm(dim) if (use_layer_norm and self.pre_layer_norm) else nn.Identity()

        self.q = nn.Parameter(torch.randn(1, n_tasks, dim) * 0.02)
        self.q_ln = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()

        if pool_v_mode == "linear":
            self.pool_v = nn.Linear(dim, dim, bias=bool(pool_v_bias))
            nn.init.xavier_uniform_(self.pool_v.weight)
            if self.pool_v.bias is not None:
                nn.init.zeros_(self.pool_v.bias)
        else:
            self.pool_v = nn.Identity()

        if self.residual_mix_learnable:
            # sigmoid(0)=0.5 (equal blend at initialization)
            self.residual_mix = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("residual_mix", None)

        self.log_tau = nn.Parameter(torch.tensor(math.log(float(temperature_init))), requires_grad=True)
        self.last_attn: Optional[torch.Tensor] = None

    def _tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(0.1, 10.0)

    def _project_for_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_v_mode == "linear":
            return self.pool_v(x)
        if self.pool_v_mode == "tie_mha_v":
            d = self.dim
            w = self.mha.in_proj_weight[2 * d : 3 * d]
            b = None if self.mha.in_proj_bias is None else self.mha.in_proj_bias[2 * d : 3 * d]
            return F.linear(x, w, b)
        return x

    def _apply_prune_and_renorm(
        self,
        alpha: torch.Tensor,                # [B,T,N]
        key_padding_mask: Optional[torch.Tensor],  # [B,N]
    ) -> torch.Tensor:
        if self.prune_below <= 0.0:
            return alpha

        a = alpha.masked_fill(alpha < self.prune_below, 0.0)
        if key_padding_mask is not None:
            a = a.masked_fill(key_padding_mask.unsqueeze(1), 0.0)

        s = a.sum(dim=-1, keepdim=True)
        need_fallback = s <= self.eps  # [B,T,1]
        if need_fallback.any():
            base = alpha
            if key_padding_mask is not None:
                base = base.masked_fill(key_padding_mask.unsqueeze(1), -1.0)
            argmax_idx = base.argmax(dim=-1, keepdim=True)  # [B,T,1]
            onehot = torch.zeros_like(a).scatter_(-1, argmax_idx, 1.0)
            a = torch.where(need_fallback, onehot, a)
            if key_padding_mask is not None:
                a = a.masked_fill(key_padding_mask.unsqueeze(1), 0.0)

        return a / a.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _topk_pool(self, alpha: torch.Tensor, pool_source: torch.Tensor) -> torch.Tensor:
        """
        alpha: [B,T,N]
        pool_source: [B,N,D]
        returns [B,T,D]
        """
        bsz, n_tasks, n_inst = alpha.shape
        dim = pool_source.shape[-1]

        if self.topk_n <= 0 or self.topk_n >= n_inst:
            return torch.einsum("btn,bnd->btd", alpha, pool_source)

        k = max(1, min(self.topk_n, n_inst))
        topv, topi = torch.topk(alpha, k, dim=-1, largest=True, sorted=True)  # [B,T,k]

        pool_src = pool_source.unsqueeze(1).expand(bsz, n_tasks, n_inst, dim)  # [B,T,N,D]
        idx = topi.unsqueeze(-1).expand(bsz, n_tasks, k, dim)
        selected = torch.gather(pool_src, dim=2, index=idx)  # [B,T,k,D]

        if k == 1:
            return selected.squeeze(2)

        if self.topk_strategy == "mean":
            return selected.mean(dim=2)
        if self.topk_strategy == "sum":
            return selected.sum(dim=2)
        if self.topk_strategy == "argmax":
            return selected[:, :, 0, :]

        # renorm strategy
        w = topv / topv.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # [B,T,k]
        return (w.unsqueeze(-1) * selected).sum(dim=2)

    def forward(
        self,
        tokens: torch.Tensor,                   # [B, N, D]
        key_padding_mask: Optional[torch.Tensor],  # [B, N] True=PAD
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,N,D], got {tuple(tokens.shape)}")

        batch_size = tokens.shape[0]
        normed_tokens = self.pre_ln(tokens)
        query = self.q_ln(self.q).expand(batch_size, -1, -1)  # [B,T,D]
        if self.use_temperature:
            query = query / self._tau()

        cls_out, attn_w = self.mha(
            query=query,
            key=normed_tokens,
            value=normed_tokens,
            need_weights=True,
            average_attn_weights=False,
            key_padding_mask=key_padding_mask,
        )  # cls_out: [B,T,D], attn_w: [B,H,T,N]
        self.last_attn = attn_w.detach()

        alpha = attn_w.mean(dim=1)  # [B,T,N]
        if key_padding_mask is not None:
            alpha = alpha.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
            alpha = mask_and_normalize_attention(alpha, key_padding_mask=key_padding_mask, eps=self.eps)
        else:
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        alpha = self._apply_prune_and_renorm(alpha, key_padding_mask=key_padding_mask)

        if self.pool_from == "attn_out":
            attn_pool = cls_out
        else:
            if self.pool_from == "inputs":
                pool_tokens = tokens
            else:  # "normed_inputs"
                pool_tokens = normed_tokens
            pool_source = self._project_for_pool(pool_tokens)
            attn_pool = self._topk_pool(alpha, pool_source)

        if not self.residual_pooling:
            return attn_pool, (alpha if return_attn else None)

        if self.residual_pool_from == "same":
            if self.pool_from == "attn_out":
                res_tokens = normed_tokens
            elif self.pool_from == "inputs":
                res_tokens = tokens
            else:
                res_tokens = normed_tokens
        elif self.residual_pool_from == "inputs":
            res_tokens = tokens
        else:
            res_tokens = normed_tokens

        res_source = self._project_for_pool(res_tokens)
        mean_pool = res_source.mean(dim=1, keepdim=True).expand(-1, self.n_tasks, -1)
        if self.residual_mix_learnable and self.residual_mix is not None:
            mix = torch.sigmoid(self.residual_mix)
            pooled = (1.0 - mix) * attn_pool + mix * mean_pool
        else:
            pooled = attn_pool + mean_pool
        return pooled, (alpha if return_attn else None)
