from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import LambdaVolConfig
from .db.repository import LambdaVolRepository
from .detectors import ConceptPressureDetector, DetectorOutput
from .dynamics import LinearConceptDynamicsModel
from .exporters import LambdaVolArtifactExporter
from .policy import RecommendationEngine
from .providers import AttentionProvider, TCAVProvider, TrainingMetricsProvider
from .ricci import ConceptRicciFlowAnalyzer, RicciEpochOutput
from .regime import HybridRegimeInferer, RuleBasedRegimeInferer
from .tracker import ConceptPressureTracker
from .types import (
    AlertRecord,
    EpochConceptMetrics,
    EpochTaskMetrics,
    PressureRunArtifacts,
    RicciEdgeMetrics,
    RicciTaskSummary,
    RecommendationRecord,
    RegimeLabel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochFrames:
    """Epoch-level input frames for concept-pressure monitoring."""

    tcav_df: pd.DataFrame
    concept_attention_df: pd.DataFrame
    task_attention_df: pd.DataFrame
    task_metrics_df: pd.DataFrame
    context_covariates: Mapping[str, float]


@dataclass(frozen=True)
class EpochStepResult:
    """Result of one Lambda-Vol monitoring step."""

    epoch: int
    regime_label: RegimeLabel
    n_alerts: int
    n_recommendations: int
    rho_mean: float
    rho_max: float


class LambdaVolMonitor:
    """Orchestrates tracker, regime inference, dynamics, alerts, policy, and persistence."""

    def __init__(
        self,
        *,
        config: LambdaVolConfig,
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        concept_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
        repository: Optional[LambdaVolRepository] = None,
    ) -> None:
        self.config = config
        self.task_ids = tuple(str(x) for x in task_ids)
        self.concept_ids = tuple(str(x) for x in concept_ids)
        self.concept_metadata = {
            str(k): dict(v) for k, v in (concept_metadata or {}).items()
        }

        self._task_to_idx = {tid: i for i, tid in enumerate(self.task_ids)}
        self._concept_to_idx = {cid: j for j, cid in enumerate(self.concept_ids)}

        self.repository = repository or LambdaVolRepository(db_uri=self.config.store.db_uri)
        self.run_id = self.repository.create_run(
            run_name=self.config.run_name,
            config=self.config.to_dict(),
        )

        self.tracker = ConceptPressureTracker(
            task_ids=self.task_ids,
            concept_ids=self.concept_ids,
            config=self.config.tracker,
        )
        self.regime_inferer = HybridRegimeInferer(
            rules=RuleBasedRegimeInferer(config=self.config.regime),
            classifier=None,
        )
        self.dynamics = LinearConceptDynamicsModel(
            config=self.config.dynamics,
            n_tasks=len(self.task_ids),
            n_concepts=len(self.concept_ids),
        )
        self.detector = ConceptPressureDetector(self.config.detector)
        self.policy = RecommendationEngine(self.config.policy)
        self.exporter = LambdaVolArtifactExporter(config=self.config.exporter)
        self.ricci_analyzer = (
            ConceptRicciFlowAnalyzer(
                task_ids=self.task_ids,
                concept_ids=self.concept_ids,
                concept_modalities=tuple(
                    str(self.concept_metadata.get(cid, {}).get("modality", "2d"))
                    for cid in self.concept_ids
                ),
                config=self.config.ricci,
            )
            if bool(self.config.ricci.enabled)
            else None
        )

        self._task_history_df = pd.DataFrame()
        self._last_regime: Optional[RegimeLabel] = None
        self._prev_concentration: Optional[dict[str, Any]] = None

        self._concept_rows: list[EpochConceptMetrics] = []
        self._task_rows: list[EpochTaskMetrics] = []
        self._alerts: list[AlertRecord] = []
        self._recommendations: list[RecommendationRecord] = []
        self._ricci_edges: list[RicciEdgeMetrics] = []
        self._ricci_summaries: list[RicciTaskSummary] = []
        self._detector_outputs: dict[int, DetectorOutput] = {}

        logger.info(
            "Initialized LambdaVolMonitor",
            extra={
                "run_id": self.run_id,
                "n_tasks": len(self.task_ids),
                "n_concepts": len(self.concept_ids),
            },
        )

    @property
    def concept_rows(self) -> Sequence[EpochConceptMetrics]:
        return tuple(self._concept_rows)

    @property
    def task_rows(self) -> Sequence[EpochTaskMetrics]:
        return tuple(self._task_rows)

    @property
    def alerts(self) -> Sequence[AlertRecord]:
        return tuple(self._alerts)

    @property
    def recommendations(self) -> Sequence[RecommendationRecord]:
        return tuple(self._recommendations)

    @property
    def ricci_edges(self) -> Sequence[RicciEdgeMetrics]:
        return tuple(self._ricci_edges)

    @property
    def ricci_summaries(self) -> Sequence[RicciTaskSummary]:
        return tuple(self._ricci_summaries)

    def step_from_frames(
        self,
        *,
        epoch: int,
        tcav_df: pd.DataFrame,
        concept_attention_df: Optional[pd.DataFrame] = None,
        task_attention_df: Optional[pd.DataFrame] = None,
        task_metrics_df: Optional[pd.DataFrame] = None,
        context_covariates: Optional[Mapping[str, float]] = None,
        ricci_payload: Optional[Mapping[str, Any]] = None,
    ) -> EpochStepResult:
        """Run one monitoring step from explicit frames (online or offline)."""
        ep = int(epoch)

        concept_attention_df = self._normalize_concept_attention_df(concept_attention_df)
        task_attention_df = self._normalize_task_attention_df(task_attention_df)
        task_metrics_df = self._normalize_task_metrics_df(task_metrics_df)
        context_covariates = (
            {} if context_covariates is None else {str(k): float(v) for k, v in context_covariates.items()}
        )

        current_for_regime = self._build_current_task_regime_frame(
            epoch=ep,
            task_attention_df=task_attention_df,
            task_metrics_df=task_metrics_df,
        )
        regime = self.regime_inferer.infer(
            epoch=ep,
            task_df_history=self._task_history_df,
            current_task_df=current_for_regime,
            context=context_covariates,
            last_regime=self._last_regime,
        )
        self._last_regime = regime

        concept_rows, task_rows, state = self.tracker.update_from_frames(
            epoch=ep,
            regime_label=regime,
            tcav_df=tcav_df,
            concept_attention_df=concept_attention_df,
            task_attention_df=task_attention_df,
            task_metrics_df=task_metrics_df,
            context_covariates=context_covariates,
        )

        ricci_out: Optional[RicciEpochOutput] = None
        if self.ricci_analyzer is not None:
            ricci_interval = int(max(1, int(getattr(self.config.ricci, "update_interval_epochs", 1))))
            if (ep % ricci_interval) != 0:
                ricci_out = None
            else:
                activity_samples = None
                if ricci_payload is not None:
                    try:
                        raw = ricci_payload.get("concept_activity_samples")
                        if raw is not None:
                            arr = np.asarray(raw, dtype=np.float32)
                            expected = (len(self.task_ids), -1, len(self.concept_ids))
                            if arr.ndim == 3 and arr.shape[0] == expected[0] and arr.shape[2] == expected[2]:
                                activity_samples = arr
                    except Exception:
                        activity_samples = None
                ricci_out = self.ricci_analyzer.analyze_epoch(
                    epoch=ep,
                    tcav_smoothed=state.tcav_smoothed,
                    attention_support=state.attention_support,
                    concept_activity_samples=activity_samples,
                )
                self._ricci_edges.extend(ricci_out.edge_rows)
                self._ricci_summaries.extend(ricci_out.task_summaries)
                if bool(self.config.ricci.use_flow_as_concept_coupling):
                    coupling = LinearConceptDynamicsModel.make_similarity_coupling(
                        ricci_out.mean_flowed_similarity,
                        strength=float(self.config.ricci.coupling_strength),
                    )
                    self.dynamics.set_concept_coupling(coupling)

        decomp = self.dynamics.predict_next(
            rho=state.rho,
            context_covariates=state.context_covariates,
            regime_label=regime,
            prev_drift=self._prev_drift(),
            running_mean_rho=self._running_mean_rho(),
        )

        concept_rows = self._attach_dynamics(concept_rows=concept_rows, decomp=decomp)

        det_out = self.detector.detect(
            run_id=self.run_id,
            epoch=ep,
            task_ids=self.task_ids,
            concept_ids=self.concept_ids,
            rho=state.rho,
            drift=state.drift,
            dissipation=decomp.dissipation,
            blocked_concepts=self.config.blocked_concepts,
            prev_concentration=self._prev_concentration,
            ricci_summaries=(None if ricci_out is None else ricci_out.task_summaries),
        )
        self._prev_concentration = det_out.concentration_by_task
        self._detector_outputs[ep] = det_out

        task_rows = self._attach_concentration(task_rows=task_rows, detector_output=det_out)

        recommendations = self.policy.recommend(
            run_id=self.run_id,
            epoch=ep,
            alerts=det_out.alerts,
        )

        self.repository.upsert_epoch_rows(
            run_id=self.run_id,
            concept_rows=concept_rows,
            task_rows=task_rows,
            context_covariates=state.context_covariates,
        )
        self.repository.insert_alerts(det_out.alerts)
        self.repository.insert_recommendations(recommendations)

        self._concept_rows.extend(concept_rows)
        self._task_rows.extend(task_rows)
        self._alerts.extend(det_out.alerts)
        self._recommendations.extend(recommendations)

        self._task_history_df = pd.DataFrame([self._task_row_to_dict(x) for x in self._task_rows])

        logger.info(
            "Lambda-Vol epoch complete",
            extra={
                "run_id": self.run_id,
                "epoch": ep,
                "regime": regime.value,
                "n_alerts": len(det_out.alerts),
                "n_recommendations": len(recommendations),
                "n_ricci_edges": (0 if ricci_out is None else len(ricci_out.edge_rows)),
                "rho_mean": float(np.mean(state.rho)),
                "rho_max": float(np.max(state.rho)),
            },
        )

        return EpochStepResult(
            epoch=ep,
            regime_label=regime,
            n_alerts=len(det_out.alerts),
            n_recommendations=len(recommendations),
            rho_mean=float(np.mean(state.rho)),
            rho_max=float(np.max(state.rho)),
        )

    def step_from_providers(
        self,
        *,
        epoch: int,
        tcav_provider: TCAVProvider,
        metrics_provider: TrainingMetricsProvider,
        attention_provider: Optional[AttentionProvider] = None,
    ) -> EpochStepResult:
        """Run one monitoring step by querying provider interfaces."""
        ep = int(epoch)
        tcav_df = tcav_provider.tcav_for_epoch(ep)
        task_metrics_df = metrics_provider.task_metrics_for_epoch(ep)
        context_covariates = metrics_provider.context_covariates_for_epoch(ep)

        if attention_provider is None:
            concept_attention_df = None
            task_attention_df = None
        else:
            concept_attention_df = attention_provider.concept_attention_for_epoch(ep)
            task_attention_df = attention_provider.task_attention_for_epoch(ep)

        return self.step_from_frames(
            epoch=ep,
            tcav_df=tcav_df,
            concept_attention_df=concept_attention_df,
            task_attention_df=task_attention_df,
            task_metrics_df=task_metrics_df,
            context_covariates=context_covariates,
            ricci_payload=None,
        )

    def finalize(self) -> PressureRunArtifacts:
        """Export all run artifacts after epoch processing."""
        return self.exporter.export(
            run_id=self.run_id,
            run_name=self.config.run_name,
            task_ids=self.task_ids,
            concept_ids=self.concept_ids,
            concept_rows=self._concept_rows,
            task_rows=self._task_rows,
            alerts=self._alerts,
            recommendations=self._recommendations,
            concept_metadata=self.concept_metadata,
            concept_coupling=self.dynamics.concept_coupling,
            ricci_edges=self._ricci_edges,
            ricci_task_summaries=self._ricci_summaries,
        )

    def _prev_drift(self) -> Optional[np.ndarray]:
        states = self.tracker.states
        if len(states) < 2:
            return None
        return states[-2].drift

    def _running_mean_rho(self) -> Optional[np.ndarray]:
        states = self.tracker.states
        if len(states) < 2:
            return None
        mats = np.stack([s.rho for s in states[:-1]], axis=0)
        return np.mean(mats, axis=0).astype(np.float32)

    def _tcav_history_by_task(self) -> dict[str, np.ndarray]:
        states = self.tracker.states
        if len(states) == 0:
            return {}
        out: dict[str, np.ndarray] = {}
        for ti, task_id in enumerate(self.task_ids):
            hist = np.stack([s.tcav[ti] for s in states], axis=0).astype(np.float32)
            out[str(task_id)] = hist
        return out

    def _attach_dynamics(
        self,
        *,
        concept_rows: Sequence[EpochConceptMetrics],
        decomp,
    ) -> list[EpochConceptMetrics]:
        out: list[EpochConceptMetrics] = []
        for row in concept_rows:
            ti = self._task_to_idx[str(row.task_id)]
            ci = self._concept_to_idx[str(row.concept_id)]
            out.append(
                replace(
                    row,
                    regime_core=float(decomp.regime_core[ti, ci]),
                    feedback=float(decomp.feedback[ti, ci]),
                    trend_loop=float(decomp.trend_loop[ti, ci]),
                    revert_loop=float(decomp.revert_loop[ti, ci]),
                    dissipation=float(decomp.dissipation[ti, ci]),
                    context_term=float(decomp.context_term[ti, ci]),
                )
            )
        return out

    def _attach_concentration(
        self,
        *,
        task_rows: Sequence[EpochTaskMetrics],
        detector_output: DetectorOutput,
    ) -> list[EpochTaskMetrics]:
        out: list[EpochTaskMetrics] = []
        for row in task_rows:
            cm = detector_output.concentration_by_task.get(str(row.task_id))
            if cm is None:
                out.append(row)
                continue
            out.append(
                replace(
                    row,
                    concentration_entropy=float(cm.entropy),
                    concentration_gini=float(cm.gini),
                    topk_mass=float(cm.topk_mass),
                )
            )
        return out

    @staticmethod
    def _task_row_to_dict(row: EpochTaskMetrics) -> dict[str, Any]:
        d = dict(row.__dict__)
        d["regime_label"] = str(row.regime_label.value)
        return d

    def _normalize_concept_attention_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(
                {
                    "task_id": np.repeat(self.task_ids, len(self.concept_ids)),
                    "concept_id": list(self.concept_ids) * len(self.task_ids),
                    "attention_support": np.zeros(len(self.task_ids) * len(self.concept_ids), dtype=np.float32),
                    "prevalence": np.zeros(len(self.task_ids) * len(self.concept_ids), dtype=np.float32),
                }
            )
        out = df.copy()
        for col in ("attention_support", "prevalence"):
            if col not in out.columns:
                out[col] = 0.0
        return out[["task_id", "concept_id", "attention_support", "prevalence"]]

    def _normalize_task_attention_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(
                {
                    "task_id": list(self.task_ids),
                    "attention_entropy": np.zeros(len(self.task_ids), dtype=np.float32),
                    "witness_rate": np.zeros(len(self.task_ids), dtype=np.float32),
                }
            )
        out = df.copy()
        if "attention_entropy" not in out.columns:
            out["attention_entropy"] = 0.0
        if "witness_rate" not in out.columns:
            out["witness_rate"] = 0.0
        return out[["task_id", "attention_entropy", "witness_rate"]]

    def _normalize_task_metrics_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(
                {
                    "task_id": list(self.task_ids),
                    "train_metric": np.zeros(len(self.task_ids), dtype=np.float32),
                    "val_metric": np.zeros(len(self.task_ids), dtype=np.float32),
                    "loss": np.zeros(len(self.task_ids), dtype=np.float32),
                    "calibration_error": np.zeros(len(self.task_ids), dtype=np.float32),
                }
            )
        out = df.copy()
        for col in ("train_metric", "val_metric", "loss", "calibration_error"):
            if col not in out.columns:
                out[col] = 0.0
        return out[["task_id", "train_metric", "val_metric", "loss", "calibration_error"]]

    def _build_current_task_regime_frame(
        self,
        *,
        epoch: int,
        task_attention_df: pd.DataFrame,
        task_metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        base = pd.DataFrame({"task_id": list(self.task_ids)})
        merged = base.merge(task_metrics_df, on="task_id", how="left")
        merged = merged.merge(task_attention_df, on="task_id", how="left")
        merged = merged.fillna(0.0)
        merged["epoch"] = int(epoch)
        if "topk_mass" not in merged.columns:
            merged["topk_mass"] = 0.0
        return merged[
            [
                "epoch",
                "task_id",
                "train_metric",
                "val_metric",
                "loss",
                "attention_entropy",
                "topk_mass",
            ]
        ]


__all__ = ["EpochFrames", "EpochStepResult", "LambdaVolMonitor"]
