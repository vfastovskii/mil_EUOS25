from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Optional, Protocol, Sequence

from .config import PolicyConfig
from .types import AlertRecord, RecommendationRecord

logger = logging.getLogger(__name__)


class InterventionActionHook(Protocol):
    """Action hook protocol for optional automatic intervention application."""

    def apply(self, recommendation: RecommendationRecord) -> bool:
        """Apply one recommendation. Return True if applied."""


@dataclass
class LoggingActionHook:
    """No-op hook that only logs requested interventions."""

    def apply(self, recommendation: RecommendationRecord) -> bool:
        logger.info(
            "Intervention hook invoked",
            extra={
                "type": recommendation.recommendation_type,
                "task_id": recommendation.task_id,
                "concept_id": recommendation.concept_id,
                "action_level": recommendation.action_level,
            },
        )
        return False


@dataclass
class RecommendationEngine:
    """Alert-driven recommendation generator and optional action dispatcher."""

    config: PolicyConfig
    action_hooks: tuple[InterventionActionHook, ...] = ()

    def recommend(
        self,
        *,
        run_id: str,
        epoch: int,
        alerts: Sequence[AlertRecord],
    ) -> list[RecommendationRecord]:
        """Generate recommendations from detector alerts."""
        if not self.config.enabled:
            return []

        recs: list[RecommendationRecord] = []
        for alert in alerts:
            if alert.code == "runaway_concept_pressure":
                recs.extend(
                    [
                        self._mk(
                            run_id=run_id,
                            epoch=epoch,
                            recommendation_type="concept_balanced_batching",
                            action_level="data",
                            task_id=alert.task_id,
                            concept_id=alert.concept_id,
                            rationale="Runaway concept pressure detected; rebalance batches toward counterexamples.",
                            params={"target_concept": alert.concept_id, "priority": "high"},
                        ),
                        self._mk(
                            run_id=run_id,
                            epoch=epoch,
                            recommendation_type="increase_attention_entropy_regularization",
                            action_level="loss",
                            task_id=alert.task_id,
                            concept_id=alert.concept_id,
                            rationale="Runaway pressure + low dissipation; increase attention entropy regularization.",
                            params={"delta": 0.05},
                        ),
                    ]
                )
            elif alert.code == "concept_collapse_topk_mass":
                recs.extend(
                    [
                        self._mk(
                            run_id=run_id,
                            epoch=epoch,
                            recommendation_type="hard_negative_mining",
                            action_level="data",
                            task_id=alert.task_id,
                            concept_id=None,
                            rationale="High top-k concept mass indicates collapse; mine hard negatives for underused concepts.",
                            params={"task_id": alert.task_id},
                        ),
                        self._mk(
                            run_id=run_id,
                            epoch=epoch,
                            recommendation_type="increase_dropout_damping",
                            action_level="model",
                            task_id=alert.task_id,
                            concept_id=None,
                            rationale="Concept concentration surge; add damping via dropout/regularization.",
                            params={"dropout_delta": 0.05, "lambda_damping_delta": 0.02},
                        ),
                    ]
                )
            elif alert.code == "blocked_concept_positive_drift":
                recs.extend(
                    [
                        self._mk(
                            run_id=run_id,
                            epoch=epoch,
                            recommendation_type="activate_blocked_concept_penalty",
                            action_level="loss",
                            task_id=alert.task_id,
                            concept_id=alert.concept_id,
                            rationale="Blocked concept exhibits positive drift; activate explicit concept penalty.",
                            params={"concept_id": alert.concept_id, "weight": 0.1},
                        ),
                        self._mk(
                            run_id=run_id,
                            epoch=epoch,
                            recommendation_type="oversample_counterexamples",
                            action_level="data",
                            task_id=alert.task_id,
                            concept_id=alert.concept_id,
                            rationale="Increase counterexamples to reduce blocked concept reliance.",
                            params={"concept_id": alert.concept_id, "multiplier": 2.0},
                        ),
                    ]
                )
            elif alert.code in {
                "ricci_negative_curvature_surge",
                "ricci_bridge_concentration",
                "ricci_extreme_negative_bridge",
            }:
                recs.extend(
                    [
                        self._mk(
                            run_id=run_id,
                            epoch=epoch,
                            recommendation_type="enable_concept_graph_rebalancing",
                            action_level="analysis",
                            task_id=alert.task_id,
                            concept_id=alert.concept_id,
                            rationale=(
                                "Ricci curvature indicates bottleneck-like concept bridges; "
                                "apply stronger concept rebalancing and monitor bridge families."
                            ),
                            params={
                                "edge_focus": "negative_curvature_bridges",
                                "task_id": alert.task_id,
                            },
                        ),
                        self._mk(
                            run_id=run_id,
                            epoch=epoch,
                            recommendation_type="increase_hard_negative_mining_for_bridge_families",
                            action_level="data",
                            task_id=alert.task_id,
                            concept_id=alert.concept_id,
                            rationale=(
                                "Bridge-heavy concept geometry can cause shortcut reliance; "
                                "mine counterexamples around affected concept families."
                            ),
                            params={"priority": "high", "task_id": alert.task_id},
                        ),
                    ]
                )

        dedup: dict[tuple[str, Optional[str], Optional[str], str], RecommendationRecord] = {}
        for rec in recs:
            key = (rec.recommendation_type, rec.task_id, rec.concept_id, rec.action_level)
            dedup[key] = rec

        out = list(dedup.values())
        if self.config.auto_action and self.action_hooks:
            self._apply_actions(out)

        logger.info(
            "Generated recommendations",
            extra={"epoch": int(epoch), "n_recommendations": len(out)},
        )
        return out

    def _apply_actions(self, recommendations: Sequence[RecommendationRecord]) -> None:
        for rec in recommendations:
            for hook in self.action_hooks:
                try:
                    applied = bool(hook.apply(rec))
                    if applied:
                        break
                except Exception:
                    logger.exception("Action hook failed", extra={"recommendation": rec.recommendation_type})

    @staticmethod
    def _mk(
        *,
        run_id: str,
        epoch: int,
        recommendation_type: str,
        action_level: str,
        task_id: Optional[str],
        concept_id: Optional[str],
        rationale: str,
        params: Mapping[str, object],
    ) -> RecommendationRecord:
        return RecommendationRecord(
            run_id=str(run_id),
            epoch=int(epoch),
            recommendation_type=str(recommendation_type),
            action_level=str(action_level),
            task_id=(None if task_id is None else str(task_id)),
            concept_id=(None if concept_id is None else str(concept_id)),
            rationale=str(rationale),
            params=dict(params),
        )


__all__ = ["InterventionActionHook", "LoggingActionHook", "RecommendationEngine"]
