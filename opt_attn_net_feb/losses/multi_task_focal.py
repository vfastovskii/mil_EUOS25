from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskFocal(nn.Module):
    """
    Computes the Multi-Task Focal Loss.

    The Multi-Task Focal Loss is an adaptation of the focal loss designed to handle
    multi-task learning scenarios, where tasks might have varying levels of
    importance and difficulty. It uses task-specific weights and a focal adjustment
    to focus training on more difficult examples in each task.

    Attributes:
        pos_weight (torch.Tensor): A tensor representing the positive class weighting for
            each task.
        gamma (torch.Tensor): A tensor that controls the focusing parameter for the focal
            loss, per task.

    Methods:
        forward: Computes the weighted multi-task focal loss given the logits, targets,
                 and task weights.
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("gamma", gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = (1 - pt).pow(self.gamma.view(1, -1))
        loss = focal * bce

        w = torch.clamp(w, min=0.0)
        num = (loss * w).sum(dim=0)
        den = w.sum(dim=0).clamp(min=1e-6)
        return num / den
