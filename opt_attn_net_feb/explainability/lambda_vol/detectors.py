from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Optional, Sequence

import numpy as np

from .config import DetectorConfig
from .types import AlertRecord, ConcentrationMetrics, RicciTaskSummary

logger = logging.getLogger(__name__)



def concentration_metrics(values: np.ndarray, top_k: int) -> ConcentrationMetrics:
    """Compute entropy/gini/top-k mass for non-negative vector."""
    v = np.asarray(values, dtype=np.float64)
    v = np.maximum(v, 0.0)
    s = float(np.sum(v))
    if s <= 1e-12:
        return ConcentrationMetrics(entropy=0.0, gini=0.0, topk_mass=0.0)

    p = v / s
    entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
    entropy_norm = float(entropy / max(np.log(len(p)), 1e-12))

    sorted_p = np.sort(p)
    n = len(sorted_p)
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = float((np.sum((2 * idx - n - 1) * sorted_p)) / max(n - 1, 1))

    kk = max(1, min(int(top_k), n))
    topk_mass = float(np.sum(np.sort(p)[-kk:]))
    return ConcentrationMetrics(entropy=entropy_norm, gini=gini, topk_mass=topk_mass)


@dataclass(frozen=True)
class DetectorOutput:
    """Detector output for one epoch."""

    concentration_by_task: dict[str, ConcentrationMetrics]
    runaway_scores: np.ndarray
    trend_vs_dissipation: np.ndarray
    alerts: list[AlertRecord]


class ConceptPressureDetector:
    """Runaway and concept-collapse detector for rho dynamics."""

    def __init__(self, config: DetectorConfig):
        self.config = config

    def detect(
        self,
        *,
        run_id: str,
        epoch: int,
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        rho: np.ndarray,
        drift: np.ndarray,
        dissipation: np.ndarray,
        blocked_concepts: Sequence[str],
        prev_concentration: Optional[Mapping[str, ConcentrationMetrics]] = None,
        ricci_summaries: Optional[Sequence[RicciTaskSummary]] = None,
    ) -> DetectorOutput:
        """Detect runaway/collapse signals and return alerts."""
        task_ids = tuple(str(x) for x in task_ids)
        concept_ids = tuple(str(x) for x in concept_ids)
        rho_m = np.asarray(rho, dtype=np.float32)
        drift_m = np.asarray(drift, dtype=np.float32)
        diss_m = np.asarray(dissipation, dtype=np.float32)

        if rho_m.shape != drift_m.shape or rho_m.shape != diss_m.shape:
            raise ValueError("rho/drift/dissipation shape mismatch")
        if rho_m.shape != (len(task_ids), len(concept_ids)):
            raise ValueError("rho shape does not match task/concept dimensions")

        runaway = np.maximum(drift_m, 0.0) / (np.abs(diss_m) + 1e-6)
        trend_vs_diss = drift_m - diss_m

        concentration: dict[str, ConcentrationMetrics] = {}
        alerts: list[AlertRecord] = []

        for ti, task_id in enumerate(task_ids):
            cm = concentration_metrics(rho_m[ti], top_k=int(self.config.concentration_top_k))
            concentration[task_id] = cm

            prev = None if prev_concentration is None else prev_concentration.get(task_id)
            if cm.topk_mass >= float(self.config.concentration_topk_mass_alert):
                alerts.append(
                    AlertRecord(
                        run_id=str(run_id),
                        epoch=int(epoch),
                        code="concept_collapse_topk_mass",
                        severity="high",
                        task_id=str(task_id),
                        concept_id=None,
                        score=float(cm.topk_mass),
                        message=(
                            f"Task '{task_id}' has high concept concentration (top-k mass={cm.topk_mass:.3f})."
                        ),
                        details={
                            "entropy": float(cm.entropy),
                            "gini": float(cm.gini),
                            "topk_mass": float(cm.topk_mass),
                        },
                    )
                )

            if prev is not None:
                entropy_drop = float(prev.entropy - cm.entropy)
                if entropy_drop >= float(self.config.concentration_entropy_drop_alert):
                    alerts.append(
                        AlertRecord(
                            run_id=str(run_id),
                            epoch=int(epoch),
                            code="concept_entropy_drop",
                            severity="medium",
                            task_id=str(task_id),
                            concept_id=None,
                            score=float(entropy_drop),
                            message=(
                                f"Task '{task_id}' concept entropy dropped by {entropy_drop:.3f} vs previous epoch."
                            ),
                            details={
                                "prev_entropy": float(prev.entropy),
                                "curr_entropy": float(cm.entropy),
                                "delta": float(entropy_drop),
                            },
                        )
                    )

        blocked = set(str(x) for x in blocked_concepts)
        for ti, task_id in enumerate(task_ids):
            for ci, concept_id in enumerate(concept_ids):
                rv = float(runaway[ti, ci])
                if rv >= float(self.config.runaway_threshold):
                    alerts.append(
                        AlertRecord(
                            run_id=str(run_id),
                            epoch=int(epoch),
                            code="runaway_concept_pressure",
                            severity="high",
                            task_id=str(task_id),
                            concept_id=str(concept_id),
                            score=rv,
                            message=(
                                f"Runaway pressure detected for task='{task_id}', concept='{concept_id}' "
                                f"(score={rv:.3f})."
                            ),
                            details={
                                "drift": float(drift_m[ti, ci]),
                                "dissipation": float(diss_m[ti, ci]),
                                "rho": float(rho_m[ti, ci]),
                            },
                        )
                    )

                if concept_id in blocked and float(drift_m[ti, ci]) >= float(self.config.blocked_concept_positive_drift):
                    alerts.append(
                        AlertRecord(
                            run_id=str(run_id),
                            epoch=int(epoch),
                            code="blocked_concept_positive_drift",
                            severity="critical",
                            task_id=str(task_id),
                            concept_id=str(concept_id),
                            score=float(drift_m[ti, ci]),
                            message=(
                                f"Blocked concept '{concept_id}' shows positive drift on task '{task_id}'."
                            ),
                            details={
                                "drift": float(drift_m[ti, ci]),
                                "rho": float(rho_m[ti, ci]),
                            },
                        )
                    )

        if ricci_summaries is not None:
            for summary in ricci_summaries:
                if float(summary.negative_edge_fraction) >= float(self.config.ricci_negative_edge_fraction_alert):
                    alerts.append(
                        AlertRecord(
                            run_id=str(run_id),
                            epoch=int(epoch),
                            code="ricci_negative_curvature_surge",
                            severity="high",
                            task_id=str(summary.task_id),
                            concept_id=None,
                            score=float(summary.negative_edge_fraction),
                            message=(
                                f"Task '{summary.task_id}' has high negative-curvature edge fraction "
                                f"({summary.negative_edge_fraction:.3f})."
                            ),
                            details={
                                "negative_edge_fraction": float(summary.negative_edge_fraction),
                                "strong_negative_edge_fraction": float(summary.strong_negative_edge_fraction),
                                "mean_curvature": float(summary.mean_curvature),
                            },
                        )
                    )

                if float(summary.strong_negative_edge_fraction) >= float(self.config.ricci_strong_negative_fraction_alert):
                    alerts.append(
                        AlertRecord(
                            run_id=str(run_id),
                            epoch=int(epoch),
                            code="ricci_bridge_concentration",
                            severity="high",
                            task_id=str(summary.task_id),
                            concept_id=None,
                            score=float(summary.strong_negative_edge_fraction),
                            message=(
                                f"Task '{summary.task_id}' has concentrated strong-negative curvature bridges "
                                f"({summary.strong_negative_edge_fraction:.3f})."
                            ),
                            details={
                                "strong_negative_edge_fraction": float(summary.strong_negative_edge_fraction),
                                "min_curvature": float(summary.min_curvature),
                                "top_negative_src": summary.top_negative_src,
                                "top_negative_dst": summary.top_negative_dst,
                            },
                        )
                    )

                if float(summary.min_curvature) <= float(self.config.ricci_min_curvature_alert):
                    alerts.append(
                        AlertRecord(
                            run_id=str(run_id),
                            epoch=int(epoch),
                            code="ricci_extreme_negative_bridge",
                            severity="critical",
                            task_id=str(summary.task_id),
                            concept_id=None,
                            score=float(summary.min_curvature),
                            message=(
                                f"Task '{summary.task_id}' has an extreme negative-curvature bridge edge "
                                f"({summary.top_negative_src}->{summary.top_negative_dst}, "
                                f"k={summary.min_curvature:.3f})."
                            ),
                            details={
                                "min_curvature": float(summary.min_curvature),
                                "top_negative_src": summary.top_negative_src,
                                "top_negative_dst": summary.top_negative_dst,
                            },
                        )
                    )

        logger.info(
            "Detector run complete",
            extra={
                "epoch": int(epoch),
                "n_alerts": len(alerts),
                "max_runaway": float(np.max(runaway)) if runaway.size else 0.0,
            },
        )

        return DetectorOutput(
            concentration_by_task=concentration,
            runaway_scores=runaway.astype(np.float32),
            trend_vs_dissipation=trend_vs_diss.astype(np.float32),
            alerts=alerts,
        )



def summarize_alerts_markdown(alerts: Sequence[AlertRecord]) -> str:
    """Build markdown summary from alert records."""
    if not alerts:
        return "# Alerts\n\nNo alerts triggered."

    lines = ["# Alerts", ""]
    for a in alerts:
        loc = []
        if a.task_id is not None:
            loc.append(f"task={a.task_id}")
        if a.concept_id is not None:
            loc.append(f"concept={a.concept_id}")
        loc_str = ", ".join(loc) if loc else "global"
        lines.append(f"- [{a.severity.upper()}] `{a.code}` ({loc_str}) score={a.score:.4f}: {a.message}")
    return "\n".join(lines)


__all__ = [
    "ConceptPressureDetector",
    "DetectorOutput",
    "concentration_metrics",
    "summarize_alerts_markdown",
]
