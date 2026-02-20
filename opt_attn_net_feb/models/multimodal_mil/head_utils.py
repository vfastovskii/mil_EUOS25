from __future__ import annotations

import torch
import torch.nn as nn


def make_projection(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(int(in_dim), int(out_dim)), nn.LayerNorm(int(out_dim)))


def apply_task_heads(z_tasks: torch.Tensor, heads: nn.ModuleList) -> torch.Tensor:
    """
    Apply one predictor head per task.

    Args:
      - z_tasks: [B, T, D]
      - heads: length T
    """
    return torch.cat([head(z_tasks[:, task_idx, :]) for task_idx, head in enumerate(heads)], dim=1)


def apply_shared_heads(x: torch.Tensor, heads: nn.ModuleList) -> torch.Tensor:
    """
    Apply multiple predictor heads to shared features.

    Args:
      - x: [B, D]
      - heads: list of predictor heads
    """
    return torch.cat([head(x) for head in heads], dim=1)


__all__ = [
    "make_projection",
    "apply_task_heads",
    "apply_shared_heads",
]
