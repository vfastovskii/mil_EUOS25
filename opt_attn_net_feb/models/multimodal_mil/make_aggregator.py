from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from .aggregators import build_aggregator


def make_aggregator(
    *,
    name: str,
    dim: int,
    n_heads: int,
    dropout: float,
    n_tasks: int,
    aggregator_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    agg_kwargs = dict(aggregator_kwargs or {})
    reserved_agg_keys = {"dim", "n_heads", "dropout", "n_tasks"}
    overlap = reserved_agg_keys.intersection(agg_kwargs.keys())
    if overlap:
        raise ValueError(f"aggregator_kwargs cannot override reserved keys: {sorted(overlap)}")
    return build_aggregator(
        name=str(name),
        dim=int(dim),
        n_heads=int(n_heads),
        dropout=float(dropout),
        n_tasks=int(n_tasks),
        **agg_kwargs,
    )


__all__ = ["make_aggregator"]
