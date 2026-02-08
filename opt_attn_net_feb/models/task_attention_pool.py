from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAttentionPool(nn.Module):
    """
    Multi-query attention pooling:
      - queries: [T=4] learned vectors
      - returns:
        pooled: [B, T, D]
        attn:   [B, T, N] (masked PAD=0 and renormalized to sum 1 over N)
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
        B = tokens.shape[0]
        q = self.q.expand(B, -1, -1)  # [B, T, D]

        if not return_attn:
            out, _ = self.mha(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
            return out, None

        # Try to get per-head weights (torch>=2); fallback to averaged weights.
        try:
            out, attn = self.mha(
                q, tokens, tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # [B, H, T, N]
            )
            attn = attn.mean(dim=1)  # -> [B, T, N]
        except TypeError:
            out, attn = self.mha(
                q, tokens, tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,  # [B, T, N]
            )

        # Mask PAD to 0 and renormalize so sum over instances == 1 for each (B,T)
        pad = key_padding_mask.unsqueeze(1).expand(-1, attn.shape[1], -1)  # [B,T,N]
        attn = attn.masked_fill(pad, 0.0)
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        attn = attn / denom
        return out, attn
