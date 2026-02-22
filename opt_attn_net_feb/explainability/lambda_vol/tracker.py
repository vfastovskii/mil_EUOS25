from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import TrackerConfig
from .providers import AttentionProvider, TCAVProvider, TrainingMetricsProvider
from .types import EpochConceptMetrics, EpochTaskMetrics, PressureTensorIndex, RegimeLabel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PressureEpochState:
    """Internal epoch state matrices for concept pressure tracking."""

    epoch: int
    regime_label: RegimeLabel
    tcav: np.ndarray
    tcav_smoothed: np.ndarray
    delta_tcav: np.ndarray
    attention_support: np.ndarray
    prevalence: np.ndarray
    rho: np.ndarray
    drift: np.ndarray
    attention_entropy: np.ndarray
    witness_rate: np.ndarray
    train_metric: np.ndarray
    val_metric: np.ndarray
    loss: np.ndarray
    calibration_error: np.ndarray
    context_covariates: dict[str, float]


class ConceptPressureTracker:
    """Tracks concept pressure rho[x,y,t] and related metrics across epochs."""

    def __init__(
        self,
        *,
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        config: TrackerConfig,
    ):
        self.task_ids = tuple(str(x) for x in task_ids)
        self.concept_ids = tuple(str(x) for x in concept_ids)
        self.config = config

        if len(self.task_ids) == 0:
            raise ValueError("task_ids cannot be empty")
        if len(self.concept_ids) == 0:
            raise ValueError("concept_ids cannot be empty")

        self._task_to_idx = {tid: i for i, tid in enumerate(self.task_ids)}
        self._concept_to_idx = {cid: j for j, cid in enumerate(self.concept_ids)}
        self._states: list[PressureEpochState] = []

    @property
    def states(self) -> Sequence[PressureEpochState]:
        """Return immutable view of tracked states."""
        return tuple(self._states)

    def update_from_providers(
        self,
        *,
        epoch: int,
        regime_label: RegimeLabel,
        tcav_provider: TCAVProvider,
        attention_provider: AttentionProvider,
        metrics_provider: TrainingMetricsProvider,
    ) -> tuple[list[EpochConceptMetrics], list[EpochTaskMetrics], PressureEpochState]:
        """Collect provider outputs for one epoch and update tracker state."""
        tcav_df = tcav_provider.tcav_for_epoch(int(epoch))
        concept_attn_df = attention_provider.concept_attention_for_epoch(int(epoch))
        task_attn_df = attention_provider.task_attention_for_epoch(int(epoch))
        task_metrics_df = metrics_provider.task_metrics_for_epoch(int(epoch))
        context = metrics_provider.context_covariates_for_epoch(int(epoch))
        return self.update_from_frames(
            epoch=epoch,
            regime_label=regime_label,
            tcav_df=tcav_df,
            concept_attention_df=concept_attn_df,
            task_attention_df=task_attn_df,
            task_metrics_df=task_metrics_df,
            context_covariates=context,
        )

    def update_from_frames(
        self,
        *,
        epoch: int,
        regime_label: RegimeLabel,
        tcav_df: pd.DataFrame,
        concept_attention_df: pd.DataFrame,
        task_attention_df: pd.DataFrame,
        task_metrics_df: pd.DataFrame,
        context_covariates: Mapping[str, float],
    ) -> tuple[list[EpochConceptMetrics], list[EpochTaskMetrics], PressureEpochState]:
        """Update tracker from explicit epoch DataFrames (online/offline mode)."""
        tcav = self._matrix_from_frame(
            df=tcav_df,
            value_col="tcav",
            default=0.0,
        )
        attention_support = self._matrix_from_frame(
            df=concept_attention_df,
            value_col="attention_support",
            default=0.0,
        )
        prevalence = self._matrix_from_frame(
            df=concept_attention_df,
            value_col="prevalence",
            default=0.0,
        )

        attention_entropy = self._task_vector_from_frame(task_attention_df, "attention_entropy", default=0.0)
        witness_rate = self._task_vector_from_frame(task_attention_df, "witness_rate", default=0.0)

        train_metric = self._task_vector_from_frame(task_metrics_df, "train_metric", default=0.0)
        val_metric = self._task_vector_from_frame(task_metrics_df, "val_metric", default=0.0)
        loss = self._task_vector_from_frame(task_metrics_df, "loss", default=0.0)
        calibration_error = self._task_vector_from_frame(task_metrics_df, "calibration_error", default=0.0)

        prev = self._states[-1] if self._states else None
        if prev is None:
            tcav_smoothed = tcav.copy()
            delta_tcav = np.zeros_like(tcav, dtype=np.float32)
        else:
            beta = float(self.config.tcav_ema_beta)
            tcav_smoothed = (beta * prev.tcav_smoothed + (1.0 - beta) * tcav).astype(np.float32)
            delta_tcav = (tcav - prev.tcav).astype(np.float32)

        alpha = float(self.config.alpha)
        rho = (alpha * tcav_smoothed + (1.0 - alpha) * attention_support).astype(np.float32)
        if prev is None:
            drift = np.zeros_like(rho, dtype=np.float32)
        else:
            drift = (rho - prev.rho).astype(np.float32)

        clip = float(self.config.drift_clip)
        if clip > 0:
            drift = np.clip(drift, -clip, clip).astype(np.float32)

        state = PressureEpochState(
            epoch=int(epoch),
            regime_label=RegimeLabel(regime_label),
            tcav=tcav,
            tcav_smoothed=tcav_smoothed,
            delta_tcav=delta_tcav,
            attention_support=attention_support,
            prevalence=prevalence,
            rho=rho,
            drift=drift,
            attention_entropy=attention_entropy,
            witness_rate=witness_rate,
            train_metric=train_metric,
            val_metric=val_metric,
            loss=loss,
            calibration_error=calibration_error,
            context_covariates={str(k): float(v) for k, v in context_covariates.items()},
        )
        self._states.append(state)

        concept_rows = self._build_concept_rows(state)
        task_rows = self._build_task_rows(state)
        logger.info(
            "Updated concept pressure state",
            extra={
                "epoch": int(epoch),
                "regime": str(state.regime_label.value),
                "rho_mean": float(np.mean(state.rho)),
                "rho_max": float(np.max(state.rho)),
            },
        )
        return concept_rows, task_rows, state

    def pressure_tensor(self) -> tuple[np.ndarray, PressureTensorIndex]:
        """Return rho tensor [T, C, E] and axis metadata."""
        if not self._states:
            return (
                np.zeros((len(self.task_ids), len(self.concept_ids), 0), dtype=np.float32),
                PressureTensorIndex(task_ids=self.task_ids, concept_ids=self.concept_ids, epochs=()),
            )
        tensor = np.stack([s.rho for s in self._states], axis=-1).astype(np.float32)
        index = PressureTensorIndex(
            task_ids=self.task_ids,
            concept_ids=self.concept_ids,
            epochs=tuple(int(s.epoch) for s in self._states),
        )
        return tensor, index

    def component_tensor(self, component: str) -> tuple[np.ndarray, PressureTensorIndex]:
        """Return tensor for one matrix component stored in state."""
        if not self._states:
            return self.pressure_tensor()
        valid = {
            "tcav": lambda s: s.tcav,
            "tcav_smoothed": lambda s: s.tcav_smoothed,
            "delta_tcav": lambda s: s.delta_tcav,
            "attention_support": lambda s: s.attention_support,
            "prevalence": lambda s: s.prevalence,
            "rho": lambda s: s.rho,
            "drift": lambda s: s.drift,
        }
        if component not in valid:
            raise KeyError(f"Unknown component '{component}'")
        tensor = np.stack([valid[component](s) for s in self._states], axis=-1).astype(np.float32)
        index = PressureTensorIndex(
            task_ids=self.task_ids,
            concept_ids=self.concept_ids,
            epochs=tuple(int(s.epoch) for s in self._states),
        )
        return tensor, index

    def to_long_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export tracked history into long-form concept/task dataframes."""
        concept_rows: list[dict[str, object]] = []
        task_rows: list[dict[str, object]] = []
        for s in self._states:
            for row in self._build_concept_rows(s):
                concept_rows.append(row.__dict__)
            for row in self._build_task_rows(s):
                task_rows.append(row.__dict__)
        return pd.DataFrame(concept_rows), pd.DataFrame(task_rows)

    def _matrix_from_frame(self, *, df: pd.DataFrame, value_col: str, default: float) -> np.ndarray:
        required = {"task_id", "concept_id", value_col}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Expected columns {sorted(required)} in frame")
        out = np.full((len(self.task_ids), len(self.concept_ids)), float(default), dtype=np.float32)
        for row in df.itertuples(index=False):
            task_id = str(getattr(row, "task_id"))
            concept_id = str(getattr(row, "concept_id"))
            if task_id not in self._task_to_idx or concept_id not in self._concept_to_idx:
                continue
            out[self._task_to_idx[task_id], self._concept_to_idx[concept_id]] = float(getattr(row, value_col))
        return out

    def _task_vector_from_frame(self, df: pd.DataFrame, value_col: str, default: float) -> np.ndarray:
        required = {"task_id", value_col}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Expected columns {sorted(required)} in task frame")
        out = np.full((len(self.task_ids),), float(default), dtype=np.float32)
        for row in df.itertuples(index=False):
            task_id = str(getattr(row, "task_id"))
            if task_id not in self._task_to_idx:
                continue
            out[self._task_to_idx[task_id]] = float(getattr(row, value_col))
        return out

    def _build_concept_rows(self, state: PressureEpochState) -> list[EpochConceptMetrics]:
        rows: list[EpochConceptMetrics] = []
        for ti, task_id in enumerate(self.task_ids):
            for ci, concept_id in enumerate(self.concept_ids):
                rows.append(
                    EpochConceptMetrics(
                        epoch=int(state.epoch),
                        task_id=str(task_id),
                        concept_id=str(concept_id),
                        tcav=float(state.tcav[ti, ci]),
                        tcav_smoothed=float(state.tcav_smoothed[ti, ci]),
                        delta_tcav=float(state.delta_tcav[ti, ci]),
                        attention_support=float(state.attention_support[ti, ci]),
                        prevalence=float(state.prevalence[ti, ci]),
                        rho=float(state.rho[ti, ci]),
                        drift=float(state.drift[ti, ci]),
                        regime_label=RegimeLabel(state.regime_label),
                    )
                )
        return rows

    def _build_task_rows(self, state: PressureEpochState) -> list[EpochTaskMetrics]:
        rows: list[EpochTaskMetrics] = []
        for ti, task_id in enumerate(self.task_ids):
            rows.append(
                EpochTaskMetrics(
                    epoch=int(state.epoch),
                    task_id=str(task_id),
                    attention_entropy=float(state.attention_entropy[ti]),
                    witness_rate=float(state.witness_rate[ti]),
                    train_metric=float(state.train_metric[ti]),
                    val_metric=float(state.val_metric[ti]),
                    loss=float(state.loss[ti]),
                    calibration_error=float(state.calibration_error[ti]),
                    regime_label=RegimeLabel(state.regime_label),
                )
            )
        return rows


__all__ = ["ConceptPressureTracker", "PressureEpochState"]
