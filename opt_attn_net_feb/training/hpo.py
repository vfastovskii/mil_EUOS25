from __future__ import annotations

"""Compatibility facade for historical `training.hpo` imports.

Core logic now lives in:
- `training.search_space`
- `training.pipeline`
"""

from .pipeline import (
    drop_ids_without_bags,
    objective_mil_cv,
    save_best_fold_metrics,
    save_study_artifacts,
    train_best_and_export,
)
from .search_space import search_space

__all__ = [
    "search_space",
    "objective_mil_cv",
    "save_study_artifacts",
    "save_best_fold_metrics",
    "drop_ids_without_bags",
    "train_best_and_export",
]
