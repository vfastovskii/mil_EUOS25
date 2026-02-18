from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...losses.regression import reg_loss_weighted


@dataclass(frozen=True)
class TrainingLossBreakdown:
    total: torch.Tensor
    cls: torch.Tensor
    abs: torch.Tensor
    fluo: torch.Tensor
    per_task: torch.Tensor
    weighted_per_task: torch.Tensor


def compute_training_losses(
    *,
    cls_loss_fn: nn.Module,
    logits: torch.Tensor,
    y_cls: torch.Tensor,
    w_cls: torch.Tensor,
    lam: torch.Tensor,
    abs_out: torch.Tensor,
    y_abs: torch.Tensor,
    m_abs: torch.Tensor,
    w_abs: torch.Tensor,
    fluo_out: torch.Tensor,
    y_fluo: torch.Tensor,
    m_fluo: torch.Tensor,
    w_fluo: torch.Tensor,
    reg_loss_type: str,
    lambda_aux_abs: float,
    lambda_aux_fluo: float,
) -> TrainingLossBreakdown:
    per_task = cls_loss_fn(logits.float(), y_cls.float(), w_cls.float())
    weighted_per_task = per_task * lam.float()
    loss_cls = weighted_per_task.mean()

    loss_abs = reg_loss_weighted(abs_out.float(), y_abs.float(), m_abs, w_abs.float(), reg_loss_type)
    loss_fluo = reg_loss_weighted(fluo_out.float(), y_fluo.float(), m_fluo, w_fluo.float(), reg_loss_type)

    loss_total = loss_cls + float(lambda_aux_abs) * loss_abs + float(lambda_aux_fluo) * loss_fluo
    return TrainingLossBreakdown(
        total=loss_total,
        cls=loss_cls,
        abs=loss_abs,
        fluo=loss_fluo,
        per_task=per_task,
        weighted_per_task=weighted_per_task,
    )
