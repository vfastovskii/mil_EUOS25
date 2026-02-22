from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .db.repository import LambdaVolRepository


@dataclass(frozen=True)
class LambdaVolQueryService:
    """High-level query API for Lambda-Vol analytics."""

    repository: LambdaVolRepository

    def planar_conjugated_rising_pressure(
        self,
        *,
        run_id: str,
        task_id: str,
        last_n_epochs: int,
    ) -> Sequence[Mapping[str, Any]]:
        return self.repository.query_planar_conjugated_rising_pressure(
            run_id=run_id,
            task_id=task_id,
            last_n_epochs=last_n_epochs,
        )

    def high_tcav_low_prevalence(
        self,
        *,
        run_id: str,
        task_id: str,
        epoch: int,
        tcav_thr: float,
        prevalence_thr: float,
    ) -> Sequence[Mapping[str, Any]]:
        return self.repository.query_high_tcav_low_prevalence(
            run_id=run_id,
            task_id=task_id,
            epoch=epoch,
            tcav_thr=tcav_thr,
            prevalence_thr=prevalence_thr,
        )

    def top_trend_loops(
        self,
        *,
        run_id: str,
        last_n_epochs: int,
        limit: int = 20,
    ) -> Sequence[Mapping[str, Any]]:
        return self.repository.query_top_trend_loops(
            run_id=run_id,
            last_n_epochs=last_n_epochs,
            limit=limit,
        )

    def recommendations(
        self,
        *,
        run_id: str,
        last_n_epochs: int,
    ) -> Sequence[Mapping[str, Any]]:
        return self.repository.query_recommendations(
            run_id=run_id,
            last_n_epochs=last_n_epochs,
        )


__all__ = ["LambdaVolQueryService"]
