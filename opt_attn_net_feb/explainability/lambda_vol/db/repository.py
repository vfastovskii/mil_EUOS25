from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import json
import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from sqlalchemy import and_, desc, func, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from ..types import AlertRecord, EpochConceptMetrics, EpochTaskMetrics, RecommendationRecord
from .models import PressureAlertORM, PressureEpochORM, PressureRunORM, RecommendationORM, TaskEpochORM
from .session import build_engine, initialize_database, make_session_factory

logger = logging.getLogger(__name__)



def _j(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class LambdaVolRepository:
    """Persistence and query API for Lambda-Vol monitoring artifacts."""

    def __init__(self, *, db_uri: str):
        self.db_uri = str(db_uri)
        self.engine = build_engine(self.db_uri)
        initialize_database(self.engine)
        self.SessionFactory = make_session_factory(self.engine)

    def _new_run_id(self, run_name: str, config: Mapping[str, Any]) -> str:
        payload = f"{run_name}|{_j(config)}|{datetime.now(timezone.utc).isoformat()}"
        return sha1(payload.encode("utf-8")).hexdigest()

    def create_run(self, *, run_name: str, config: Mapping[str, Any], run_id: Optional[str] = None) -> str:
        rid = str(run_id or self._new_run_id(run_name, config))
        with self.SessionFactory() as s:
            s.add(
                PressureRunORM(
                    id=rid,
                    name=str(run_name),
                    config_json=_j(dict(config)),
                    created_at=datetime.now(timezone.utc),
                )
            )
            s.commit()
        return rid

    def upsert_epoch_rows(
        self,
        *,
        run_id: str,
        concept_rows: Sequence[EpochConceptMetrics],
        task_rows: Sequence[EpochTaskMetrics],
        context_covariates: Mapping[str, float],
    ) -> None:
        with self.SessionFactory() as s:
            for row in concept_rows:
                existing = s.execute(
                    select(PressureEpochORM).where(
                        PressureEpochORM.run_id == str(run_id),
                        PressureEpochORM.epoch == int(row.epoch),
                        PressureEpochORM.task_id == str(row.task_id),
                        PressureEpochORM.concept_id == str(row.concept_id),
                    )
                ).scalar_one_or_none()
                if existing is None:
                    s.add(
                        PressureEpochORM(
                            run_id=str(run_id),
                            epoch=int(row.epoch),
                            task_id=str(row.task_id),
                            concept_id=str(row.concept_id),
                            tcav=float(row.tcav),
                            tcav_smoothed=float(row.tcav_smoothed),
                            delta_tcav=float(row.delta_tcav),
                            attention_support=float(row.attention_support),
                            prevalence=float(row.prevalence),
                            rho=float(row.rho),
                            drift=float(row.drift),
                            regime_label=str(row.regime_label.value),
                            regime_core=float(row.regime_core),
                            feedback=float(row.feedback),
                            trend_loop=float(row.trend_loop),
                            revert_loop=float(row.revert_loop),
                            dissipation=float(row.dissipation),
                            context_term=float(row.context_term),
                        )
                    )
                else:
                    existing.tcav = float(row.tcav)
                    existing.tcav_smoothed = float(row.tcav_smoothed)
                    existing.delta_tcav = float(row.delta_tcav)
                    existing.attention_support = float(row.attention_support)
                    existing.prevalence = float(row.prevalence)
                    existing.rho = float(row.rho)
                    existing.drift = float(row.drift)
                    existing.regime_label = str(row.regime_label.value)
                    existing.regime_core = float(row.regime_core)
                    existing.feedback = float(row.feedback)
                    existing.trend_loop = float(row.trend_loop)
                    existing.revert_loop = float(row.revert_loop)
                    existing.dissipation = float(row.dissipation)
                    existing.context_term = float(row.context_term)

            ctx_json = _j({str(k): float(v) for k, v in context_covariates.items()})
            for row in task_rows:
                existing = s.execute(
                    select(TaskEpochORM).where(
                        TaskEpochORM.run_id == str(run_id),
                        TaskEpochORM.epoch == int(row.epoch),
                        TaskEpochORM.task_id == str(row.task_id),
                    )
                ).scalar_one_or_none()
                if existing is None:
                    s.add(
                        TaskEpochORM(
                            run_id=str(run_id),
                            epoch=int(row.epoch),
                            task_id=str(row.task_id),
                            attention_entropy=float(row.attention_entropy),
                            witness_rate=float(row.witness_rate),
                            train_metric=float(row.train_metric),
                            val_metric=float(row.val_metric),
                            loss=float(row.loss),
                            calibration_error=float(row.calibration_error),
                            concentration_entropy=float(row.concentration_entropy),
                            concentration_gini=float(row.concentration_gini),
                            topk_mass=float(row.topk_mass),
                            regime_label=str(row.regime_label.value),
                            context_json=ctx_json,
                        )
                    )
                else:
                    existing.attention_entropy = float(row.attention_entropy)
                    existing.witness_rate = float(row.witness_rate)
                    existing.train_metric = float(row.train_metric)
                    existing.val_metric = float(row.val_metric)
                    existing.loss = float(row.loss)
                    existing.calibration_error = float(row.calibration_error)
                    existing.concentration_entropy = float(row.concentration_entropy)
                    existing.concentration_gini = float(row.concentration_gini)
                    existing.topk_mass = float(row.topk_mass)
                    existing.regime_label = str(row.regime_label.value)
                    existing.context_json = ctx_json

            s.commit()

    def insert_alerts(self, alerts: Sequence[AlertRecord]) -> None:
        if not alerts:
            return
        with self.SessionFactory() as s:
            for a in alerts:
                s.add(
                    PressureAlertORM(
                        run_id=str(a.run_id),
                        epoch=int(a.epoch),
                        code=str(a.code),
                        severity=str(a.severity),
                        task_id=(None if a.task_id is None else str(a.task_id)),
                        concept_id=(None if a.concept_id is None else str(a.concept_id)),
                        score=float(a.score),
                        message=str(a.message),
                        details_json=_j(dict(a.details)),
                    )
                )
            s.commit()

    def insert_recommendations(self, recommendations: Sequence[RecommendationRecord]) -> None:
        if not recommendations:
            return
        with self.SessionFactory() as s:
            for r in recommendations:
                s.add(
                    RecommendationORM(
                        run_id=str(r.run_id),
                        epoch=int(r.epoch),
                        recommendation_type=str(r.recommendation_type),
                        action_level=str(r.action_level),
                        task_id=(None if r.task_id is None else str(r.task_id)),
                        concept_id=(None if r.concept_id is None else str(r.concept_id)),
                        rationale=str(r.rationale),
                        params_json=_j(dict(r.params)),
                    )
                )
            s.commit()

    def query_planar_conjugated_rising_pressure(
        self,
        *,
        run_id: str,
        task_id: str,
        last_n_epochs: int,
    ) -> list[dict[str, Any]]:
        """Query concepts with planar+conjugated tags and rising rho trend."""
        with self.SessionFactory() as s:
            max_epoch = s.execute(
                select(func.max(PressureEpochORM.epoch)).where(PressureEpochORM.run_id == str(run_id))
            ).scalar_one_or_none()
            if max_epoch is None:
                return []
            min_epoch = max(0, int(max_epoch) - int(last_n_epochs) + 1)

            rows = s.execute(
                select(
                    PressureEpochORM.concept_id,
                    PressureEpochORM.epoch,
                    PressureEpochORM.rho,
                ).where(
                    and_(
                        PressureEpochORM.run_id == str(run_id),
                        PressureEpochORM.task_id == str(task_id),
                        PressureEpochORM.epoch >= int(min_epoch),
                    )
                )
            ).all()

            concept_tags: dict[str, set[str]] = {}
            insp = inspect(self.engine)
            if insp.has_table("concept_tags"):
                tag_rows = s.execute(text("SELECT concept_id, tag FROM concept_tags")).all()
                for cid, tag in tag_rows:
                    concept_tags.setdefault(str(cid), set()).add(str(tag))

        grouped: dict[str, list[tuple[int, float]]] = {}
        for cid, ep, rho in rows:
            grouped.setdefault(str(cid), []).append((int(ep), float(rho)))

        out: list[dict[str, Any]] = []
        for cid, pts in grouped.items():
            tags = concept_tags.get(cid, set())
            if tags and ("planar" not in tags or not ({"extended conjugation", "aromatic pi-system"} & tags)):
                continue
            pts = sorted(pts)
            if len(pts) < 2:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            slope = float(np.polyfit(x, y, deg=1)[0])
            if slope > 0:
                out.append({"concept_id": cid, "slope": slope, "rho_last": float(y[-1]), "tags": sorted(tags)})

        out.sort(key=lambda r: r["slope"], reverse=True)
        return out

    def query_high_tcav_low_prevalence(
        self,
        *,
        run_id: str,
        task_id: str,
        epoch: int,
        tcav_thr: float,
        prevalence_thr: float,
    ) -> list[dict[str, Any]]:
        with self.SessionFactory() as s:
            rows = s.execute(
                select(
                    PressureEpochORM.concept_id,
                    PressureEpochORM.tcav,
                    PressureEpochORM.prevalence,
                ).where(
                    and_(
                        PressureEpochORM.run_id == str(run_id),
                        PressureEpochORM.task_id == str(task_id),
                        PressureEpochORM.epoch == int(epoch),
                        PressureEpochORM.tcav >= float(tcav_thr),
                        PressureEpochORM.prevalence <= float(prevalence_thr),
                    )
                )
            ).all()
        return [
            {
                "concept_id": str(cid),
                "tcav": float(tcav),
                "prevalence": float(prev),
            }
            for cid, tcav, prev in rows
        ]

    def query_top_trend_loops(
        self,
        *,
        run_id: str,
        last_n_epochs: int,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self.SessionFactory() as s:
            max_epoch = s.execute(
                select(func.max(PressureEpochORM.epoch)).where(PressureEpochORM.run_id == str(run_id))
            ).scalar_one_or_none()
            if max_epoch is None:
                return []
            min_epoch = max(0, int(max_epoch) - int(last_n_epochs) + 1)

            rows = s.execute(
                select(
                    PressureEpochORM.task_id,
                    PressureEpochORM.concept_id,
                    func.avg(PressureEpochORM.trend_loop),
                    func.avg(PressureEpochORM.dissipation),
                )
                .where(
                    and_(
                        PressureEpochORM.run_id == str(run_id),
                        PressureEpochORM.epoch >= int(min_epoch),
                    )
                )
                .group_by(PressureEpochORM.task_id, PressureEpochORM.concept_id)
                .order_by(desc(func.avg(PressureEpochORM.trend_loop)))
                .limit(int(limit))
            ).all()

        out: list[dict[str, Any]] = []
        for task_id, concept_id, tr, dis in rows:
            trf = float(tr)
            disf = float(dis)
            ratio = float(trf / (abs(disf) + 1e-6))
            out.append(
                {
                    "task_id": str(task_id),
                    "concept_id": str(concept_id),
                    "trend_loop": trf,
                    "dissipation": disf,
                    "trend_over_dissipation": ratio,
                }
            )
        return out

    def query_recommendations(self, *, run_id: str, last_n_epochs: int) -> list[dict[str, Any]]:
        with self.SessionFactory() as s:
            max_epoch = s.execute(
                select(func.max(RecommendationORM.epoch)).where(RecommendationORM.run_id == str(run_id))
            ).scalar_one_or_none()
            if max_epoch is None:
                return []
            min_epoch = max(0, int(max_epoch) - int(last_n_epochs) + 1)
            rows = s.execute(
                select(
                    RecommendationORM.epoch,
                    RecommendationORM.recommendation_type,
                    RecommendationORM.action_level,
                    RecommendationORM.task_id,
                    RecommendationORM.concept_id,
                    RecommendationORM.rationale,
                    RecommendationORM.params_json,
                )
                .where(
                    and_(
                        RecommendationORM.run_id == str(run_id),
                        RecommendationORM.epoch >= int(min_epoch),
                    )
                )
                .order_by(RecommendationORM.epoch.asc())
            ).all()

        return [
            {
                "epoch": int(ep),
                "recommendation_type": str(rt),
                "action_level": str(level),
                "task_id": (None if task_id is None else str(task_id)),
                "concept_id": (None if concept_id is None else str(concept_id)),
                "rationale": str(rationale),
                "params": json.loads(params_json),
            }
            for ep, rt, level, task_id, concept_id, rationale, params_json in rows
        ]


__all__ = ["LambdaVolRepository"]
