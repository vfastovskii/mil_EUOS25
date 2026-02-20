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
    """
    Computes a weighted regression loss between predicted and target tensors using the provided mask
    and weights. Supports different types of loss functions such as Smooth L1 and MSE. Invalid
    targets are ignored using a specified mask to prevent numerical instabilities.

    Parameters:
        pred (torch.Tensor): The predicted values.
        target (torch.Tensor): The target values for comparison against predictions.
        mask (torch.Tensor): A tensor indicating valid elements for loss computation as a boolean mask.
        w (torch.Tensor): The weight tensor for scaling the loss computation.
        loss_type (str): Specifies the type of loss to calculate. Valid options are "smoothl1"
                         and "mse".

    Returns:
        torch.Tensor: The computed weighted regression loss as a single scalar tensor.

    Raises:
        ValueError: If the specified loss_type is not "smoothl1" or "mse".
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
