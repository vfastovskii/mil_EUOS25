from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskFocal(nn.Module):
    """
    Weighted focal BCE per task:
      - w: per-sample per-task weights
      - returns per-task scalar losses (shape [4])
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
