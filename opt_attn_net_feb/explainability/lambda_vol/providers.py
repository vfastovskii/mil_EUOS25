from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

import pandas as pd


class ConceptProvider(Protocol):
    """Provides concept family universe and optional metadata."""

    def task_ids(self) -> Sequence[str]:
        """Return ordered task IDs."""

    def concept_ids(self) -> Sequence[str]:
        """Return ordered concept family IDs."""

    def concept_metadata(self) -> Mapping[str, Mapping[str, Any]]:
        """Return metadata per concept ID."""


class TCAVProvider(Protocol):
    """Provides TCAV scores at epoch granularity."""

    def tcav_for_epoch(self, epoch: int) -> pd.DataFrame:
        """Return DataFrame with columns: task_id, concept_id, tcav."""


class AttentionProvider(Protocol):
    """Provides MIL attention-derived concept metrics."""

    def concept_attention_for_epoch(self, epoch: int) -> pd.DataFrame:
        """Return DataFrame with columns: task_id, concept_id, attention_support, prevalence."""

    def task_attention_for_epoch(self, epoch: int) -> pd.DataFrame:
        """Return DataFrame with columns: task_id, attention_entropy, witness_rate."""


class TrainingMetricsProvider(Protocol):
    """Provides training/validation covariates for regime and dynamics."""

    def task_metrics_for_epoch(self, epoch: int) -> pd.DataFrame:
        """Return DataFrame with columns: task_id, train_metric, val_metric, loss, calibration_error."""

    def context_covariates_for_epoch(self, epoch: int) -> Mapping[str, float]:
        """Return global context covariates c(t)."""


class ModelAdapter(Protocol):
    """Generic adapter for model outputs/gradients where needed."""

    def task_ids(self) -> Sequence[str]:
        """Return known task IDs."""

    def get_layer_activations(self, layer_name: str, batch: Any) -> Any:
        """Return activations at a layer for a batch."""

    def get_task_scalar(self, outputs: Any, task_id: str) -> Any:
        """Return task-specific scalar objective from model outputs."""


@dataclass(frozen=True)
class OfflineFrameProvider:
    """Simple offline provider from precomputed frames by epoch."""

    tcav_by_epoch: Mapping[int, pd.DataFrame]
    concept_attention_by_epoch: Mapping[int, pd.DataFrame]
    task_attention_by_epoch: Mapping[int, pd.DataFrame]
    task_metrics_by_epoch: Mapping[int, pd.DataFrame]
    context_by_epoch: Mapping[int, Mapping[str, float]]

    def tcav_for_epoch(self, epoch: int) -> pd.DataFrame:
        return self.tcav_by_epoch[int(epoch)].copy()

    def concept_attention_for_epoch(self, epoch: int) -> pd.DataFrame:
        return self.concept_attention_by_epoch[int(epoch)].copy()

    def task_attention_for_epoch(self, epoch: int) -> pd.DataFrame:
        return self.task_attention_by_epoch[int(epoch)].copy()

    def task_metrics_for_epoch(self, epoch: int) -> pd.DataFrame:
        return self.task_metrics_by_epoch[int(epoch)].copy()

    def context_covariates_for_epoch(self, epoch: int) -> Mapping[str, float]:
        return dict(self.context_by_epoch[int(epoch)])


__all__ = [
    "AttentionProvider",
    "ConceptProvider",
    "ModelAdapter",
    "OfflineFrameProvider",
    "TCAVProvider",
    "TrainingMetricsProvider",
]
