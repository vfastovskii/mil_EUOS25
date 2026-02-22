from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, sessionmaker

from ..db.models import ConceptORM, ConceptTagORM, MILConceptEpochORM, TCAVEpochORM


@dataclass(frozen=True)
class ConceptQueryService:
    """Query/analytics service for Chem-ACE concept database."""

    session_factory: sessionmaker[Session]

    def _fetch_tcav_series(self, *, task_id: str, last_n_epochs: int) -> dict[str, list[tuple[int, float]]]:
        with self.session_factory() as s:
            max_epoch = s.execute(select(func.max(TCAVEpochORM.epoch))).scalar_one_or_none()
            if max_epoch is None:
                return {}
            min_epoch = max(0, int(max_epoch) - int(last_n_epochs) + 1)
            rows = s.execute(
                select(
                    TCAVEpochORM.concept_id,
                    TCAVEpochORM.epoch,
                    func.avg(TCAVEpochORM.tcav_sign_rate),
                )
                .where(
                    and_(
                        TCAVEpochORM.task_id == str(task_id),
                        TCAVEpochORM.epoch >= int(min_epoch),
                    )
                )
                .group_by(TCAVEpochORM.concept_id, TCAVEpochORM.epoch)
                .order_by(TCAVEpochORM.concept_id.asc(), TCAVEpochORM.epoch.asc())
            ).all()

        out: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for cid, ep, val in rows:
            out[str(cid)].append((int(ep), float(val)))
        return out

    def planar_conjugated_rising_tcav(self, task_id: str, last_n_epochs: int) -> Sequence[Mapping[str, Any]]:
        series = self._fetch_tcav_series(task_id=task_id, last_n_epochs=last_n_epochs)
        if not series:
            return []

        with self.session_factory() as s:
            tag_rows = s.execute(
                select(ConceptTagORM.concept_id, ConceptTagORM.tag)
                .where(ConceptTagORM.tag.in_(["planar", "extended conjugation", "aromatic pi-system"]))
            ).all()
            label_rows = s.execute(select(ConceptORM.id, ConceptORM.label_auto)).all()

        tags_by_concept: dict[str, set[str]] = defaultdict(set)
        for cid, tag in tag_rows:
            tags_by_concept[str(cid)].add(str(tag))
        labels_by_concept = {str(cid): (None if lbl is None else str(lbl)) for cid, lbl in label_rows}

        out: list[dict[str, Any]] = []
        for cid, pts in series.items():
            tags = tags_by_concept.get(cid, set())
            if "planar" not in tags:
                continue
            if not ({"extended conjugation", "aromatic pi-system"} & tags):
                continue
            if len(pts) < 2:
                continue
            xs = np.asarray([p[0] for p in pts], dtype=np.float64)
            ys = np.asarray([p[1] for p in pts], dtype=np.float64)
            slope = float(np.polyfit(xs, ys, deg=1)[0])
            if slope <= 0:
                continue
            out.append(
                {
                    "concept_id": cid,
                    "label_auto": labels_by_concept.get(cid),
                    "slope": slope,
                    "tcav_last": float(ys[-1]),
                    "tags": sorted(tags),
                    "series": pts,
                }
            )
        out.sort(key=lambda r: r["slope"], reverse=True)
        return out

    def top_attention_support(self, task_id: str, epoch: int, limit: int = 20) -> Sequence[Mapping[str, Any]]:
        with self.session_factory() as s:
            rows = s.execute(
                select(
                    MILConceptEpochORM.concept_id,
                    ConceptORM.label_auto,
                    MILConceptEpochORM.attention_support,
                    MILConceptEpochORM.witness_rate,
                    MILConceptEpochORM.attention_entropy,
                    MILConceptEpochORM.prevalence,
                )
                .join(ConceptORM, ConceptORM.id == MILConceptEpochORM.concept_id)
                .where(
                    and_(
                        MILConceptEpochORM.task_id == str(task_id),
                        MILConceptEpochORM.epoch == int(epoch),
                    )
                )
                .order_by(MILConceptEpochORM.attention_support.desc())
                .limit(int(limit))
            ).all()

        return [
            {
                "concept_id": str(cid),
                "label_auto": (None if label is None else str(label)),
                "attention_support": float(sup),
                "witness_rate": float(wit),
                "attention_entropy": float(ent),
                "prevalence": float(prev),
            }
            for cid, label, sup, wit, ent, prev in rows
        ]

    def high_tcav_low_prevalence(
        self,
        task_id: str,
        epoch: int,
        tcav_thr: float,
        prevalence_thr: float,
    ) -> Sequence[Mapping[str, Any]]:
        with self.session_factory() as s:
            tcav_rows = s.execute(
                select(TCAVEpochORM.concept_id, func.avg(TCAVEpochORM.tcav_sign_rate))
                .where(
                    and_(
                        TCAVEpochORM.task_id == str(task_id),
                        TCAVEpochORM.epoch == int(epoch),
                    )
                )
                .group_by(TCAVEpochORM.concept_id)
            ).all()
            prev_rows = s.execute(
                select(MILConceptEpochORM.concept_id, func.avg(MILConceptEpochORM.prevalence))
                .where(
                    and_(
                        MILConceptEpochORM.task_id == str(task_id),
                        MILConceptEpochORM.epoch == int(epoch),
                    )
                )
                .group_by(MILConceptEpochORM.concept_id)
            ).all()

        tcav_map = {str(cid): float(v) for cid, v in tcav_rows}
        prev_map = {str(cid): float(v) for cid, v in prev_rows}

        out: list[dict[str, Any]] = []
        for cid, tcav in tcav_map.items():
            prev = prev_map.get(cid)
            if prev is None:
                continue
            if tcav >= float(tcav_thr) and prev <= float(prevalence_thr):
                out.append({"concept_id": cid, "tcav_sign_rate": tcav, "prevalence": prev})

        out.sort(key=lambda r: (r["tcav_sign_rate"], -r["prevalence"]), reverse=True)
        return out

    def concept_collapse_indicator(self, task_id: str, epoch: int, top_k: int = 5) -> Mapping[str, Any]:
        with self.session_factory() as s:
            rows = s.execute(
                select(TCAVEpochORM.concept_id, func.avg(TCAVEpochORM.tcav_sign_rate))
                .where(
                    and_(
                        TCAVEpochORM.task_id == str(task_id),
                        TCAVEpochORM.epoch == int(epoch),
                    )
                )
                .group_by(TCAVEpochORM.concept_id)
            ).all()

        vals = sorted([max(0.0, float(v)) for _, v in rows], reverse=True)
        if not vals:
            return {
                "task_id": str(task_id),
                "epoch": int(epoch),
                "collapse_index": 0.0,
                "top_k": int(top_k),
                "n_concepts": 0,
            }

        total = float(np.sum(vals))
        top_mass = float(np.sum(vals[: max(1, int(top_k))]))
        collapse = float(top_mass / max(total, 1e-12))
        return {
            "task_id": str(task_id),
            "epoch": int(epoch),
            "collapse_index": collapse,
            "top_k": int(top_k),
            "n_concepts": int(len(vals)),
            "total_tcav_mass": total,
            "top_tcav_mass": top_mass,
        }


__all__ = ["ConceptQueryService"]
