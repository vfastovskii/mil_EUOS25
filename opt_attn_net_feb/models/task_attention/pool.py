from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .masking import mask_and_normalize_attention


class TaskAttentionPool(nn.Module):
    """
    Multi-query attention pooling:
      - learned queries: [T=n_tasks]
      - outputs:
        pooled: [B, T, D]
        attn:   [B, T, N] (padding masked and renormalized)
    """

    def __init__(self, dim: int, n_heads: int, dropout: float, n_tasks: int = 4):
        super().__init__()
        dim = int(dim)
        n_heads = int(n_heads)
        n_tasks = int(n_tasks)
        if dim % n_heads != 0:
            raise ValueError(f"TaskAttentionPool: dim={dim} must be divisible by n_heads={n_heads}")

        self.n_tasks = n_tasks
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.q = nn.Parameter(torch.randn(1, n_tasks, dim) * 0.02)

    def forward(
        self,
        tokens: torch.Tensor,           # [B, N, D]
        key_padding_mask: torch.Tensor, # [B, N] True=PAD
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = tokens.shape[0]
        query = self.q.expand(batch_size, -1, -1)  # [B, T, D]

        if not return_attn:
            pooled, _ = self.mha(query, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
            return pooled, None

        pooled, attn = self._forward_with_attention(query=query, tokens=tokens, key_padding_mask=key_padding_mask)
        attn = mask_and_normalize_attention(attn=attn, key_padding_mask=key_padding_mask)
        return pooled, attn

    def _forward_with_attention(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prefer per-head attention (torch>=2), then average over heads.
        try:
            pooled, attn = self.mha(
                query,
                tokens,
                tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # [B, H, T, N]
            )
            attn = attn.mean(dim=1)  # [B, T, N]
        except TypeError:
            pooled, attn = self.mha(
                query,
                tokens,
                tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,  # [B, T, N]
            )
        return pooled, attn
