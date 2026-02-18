from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from ..attention_pooling import TaskAttentionPool

AggregatorBuilder = Callable[..., nn.Module]

_AGGREGATOR_REGISTRY: Dict[str, AggregatorBuilder] = {}


def _build_task_attention_pool(
    *,
    dim: int,
    n_heads: int,
    dropout: float,
    n_tasks: int,
    **kwargs,
) -> nn.Module:
    return TaskAttentionPool(
        dim=int(dim),
        n_heads=int(n_heads),
        dropout=float(dropout),
        n_tasks=int(n_tasks),
        **kwargs,
    )


def register_aggregator(name: str, builder: AggregatorBuilder) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Aggregator name must be non-empty")
    _AGGREGATOR_REGISTRY[key] = builder


def build_aggregator(
    *,
    name: str,
    dim: int,
    n_heads: int,
    dropout: float,
    n_tasks: int,
    **kwargs,
) -> nn.Module:
    key = str(name).strip().lower()
    if key not in _AGGREGATOR_REGISTRY:
        raise ValueError(f"Unknown aggregator '{name}'. Available: {sorted(_AGGREGATOR_REGISTRY)}")
    return _AGGREGATOR_REGISTRY[key](
        dim=int(dim),
        n_heads=int(n_heads),
        dropout=float(dropout),
        n_tasks=int(n_tasks),
        **kwargs,
    )


def available_aggregators() -> list[str]:
    return sorted(_AGGREGATOR_REGISTRY)


# Default implementation
register_aggregator("task_attention_pool", _build_task_attention_pool)


__all__ = [
    "register_aggregator",
    "build_aggregator",
    "available_aggregators",
]
