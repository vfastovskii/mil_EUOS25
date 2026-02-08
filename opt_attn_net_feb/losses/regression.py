from __future__ import annotations

import torch
import torch.nn.functional as F


def reg_loss_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Masked, weighted regression loss averaged across outputs.

    Parameters
    ----------
    pred: [B, C] predictions
    target: [B, C] targets
    mask: [B, C] boolean mask indicating valid targets
    w: [B, C] non-negative weights
    loss_type: 'smoothl1' or 'mse'
    """
    # Ensure boolean mask and avoid computing loss on invalid targets (prevents NaNs)
    mask_b = mask.bool()
    mask_f = mask_b.float()
    w_eff = torch.clamp(w, min=0.0) * mask_f

    target_safe = torch.where(mask_b, target, pred.detach())

    if loss_type == "smoothl1":
        per = F.smooth_l1_loss(pred, target_safe, reduction="none")
    elif loss_type == "mse":
        per = (pred - target_safe).pow(2)
    else:
        raise ValueError(f"Unknown reg loss type: {loss_type}")

    per = per * mask_f
    num = (per * w_eff).sum(dim=0)
    den = w_eff.sum(dim=0).clamp(min=1e-6)
    return (num / den).mean()
