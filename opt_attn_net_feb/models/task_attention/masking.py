from __future__ import annotations

import torch


def mask_and_normalize_attention(
    attn: torch.Tensor,
    key_padding_mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Apply padding mask and renormalize attention to sum to 1 over instances.

    Expected shapes:
      - attn: [B, T, N]
      - key_padding_mask: [B, N], where True marks PAD positions.
    """
    if attn.ndim != 3:
        raise ValueError(f"Expected attn with shape [B,T,N], got ndim={attn.ndim}")
    if key_padding_mask.ndim != 2:
        raise ValueError(f"Expected key_padding_mask with shape [B,N], got ndim={key_padding_mask.ndim}")
    if attn.shape[0] != key_padding_mask.shape[0] or attn.shape[2] != key_padding_mask.shape[1]:
        raise ValueError(
            "Incompatible shapes for attention masking: "
            f"attn={tuple(attn.shape)} key_padding_mask={tuple(key_padding_mask.shape)}"
        )

    pad = key_padding_mask.bool().unsqueeze(1).expand(-1, attn.shape[1], -1)  # [B,T,N]
    attn = attn.masked_fill(pad, 0.0)
    denom = attn.sum(dim=-1, keepdim=True).clamp(min=float(eps))
    return attn / denom
