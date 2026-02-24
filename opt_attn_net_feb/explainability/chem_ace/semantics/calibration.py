from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ....utils.progress import log_event
from ...types import TagAssignment
from .taggers import SemanticTaggingResult


@dataclass(frozen=True)
class ActivityCalibrationConfig:
    """Configuration for activity-aware semantic tag calibration."""

    enabled: bool = True
    min_concept_support: int = 12
    min_tag_support: int = 24
    prior_strength: float = 32.0
    min_w: float = 0.40
    task_weight: float = 0.70
    bitmask_weight: float = 0.30
    bitmask_min_count: int = 20
    bitmask_exclude_zero: bool = True
    mix_base: float = 0.60
    keep_threshold: float = 0.55
    min_confidence: float = 0.05
    max_confidence: float = 0.99
    ratio_cap: float = 8.0
    fallback_top1_if_empty: bool = True


@dataclass(frozen=True)
class ActivityCalibratedTagRecord:
    """One calibrated concept-tag row with train-only activity evidence."""

    concept_id: str
    tag: str
    provenance: str
    base_confidence: float
    calibrated_confidence: float
    keep: bool
    concept_support: int
    tag_support: int
    concept_task_score: float
    concept_bitmask_score: float
    tag_task_score: float
    tag_bitmask_score: float
    selected_task: str
    selected_task_ratio: float
    selected_bitmask: int
    selected_bitmask_ratio: float


class ActivityAwareSemanticCalibrator:
    """
    Calibrate semantic tag confidence using train-scope activity labels.

    This class is leakage-safe if caller provides only CV-train/final-train IDs and labels.
    """

    def __init__(
        self,
        *,
        config: ActivityCalibrationConfig,
        task_cols: Sequence[str],
    ) -> None:
        self.config = config
        self.task_cols = tuple(str(x) for x in task_cols)

    def calibrate(
        self,
        *,
        tagging_results: Sequence[SemanticTaggingResult],
        concept_mol_map_train: Mapping[str, set[str]],
        ids_train: Sequence[str],
        y_train: np.ndarray,
    ) -> tuple[list[SemanticTaggingResult], list[ActivityCalibratedTagRecord], dict[str, Any]]:
        """Fit train-scope activity statistics and return calibrated semantic tags."""
        if len(tagging_results) == 0:
            return [], [], {"enabled": bool(self.config.enabled), "n_tagging_results": 0}
        if (not bool(self.config.enabled)) or len(ids_train) == 0:
            return list(tagging_results), [], {
                "enabled": bool(self.config.enabled),
                "reason": "disabled_or_empty_train_ids",
                "n_tagging_results": int(len(tagging_results)),
            }

        ids = [str(x) for x in ids_train]
        y = np.asarray(y_train, dtype=np.int64)
        if y.ndim != 2 or y.shape[0] != len(ids) or y.shape[1] != len(self.task_cols):
            raise ValueError(
                "Activity calibration requires y_train with shape "
                f"[len(ids_train), {len(self.task_cols)}], got {tuple(y.shape)}"
            )
        y = (y > 0).astype(np.int64)
        bitmask_ids = self._bitmask_ids(y)

        concept_profiles = self._compute_concept_profiles(
            concept_mol_map_train=concept_mol_map_train,
            ids=ids,
            y=y,
            bitmask_ids=bitmask_ids,
        )
        tag_profiles = self._compute_tag_profiles(
            tagging_results=tagging_results,
            concept_mol_map_train=concept_mol_map_train,
            ids=ids,
            y=y,
            bitmask_ids=bitmask_ids,
        )

        records: list[ActivityCalibratedTagRecord] = []
        calibrated_results: list[SemanticTaggingResult] = []
        concept_to_rows: dict[str, list[ActivityCalibratedTagRecord]] = {}

        tag_by_concept: dict[str, dict[str, TagAssignment]] = {}
        for result in tagging_results:
            tag_by_concept[str(result.concept_id)] = {str(t.tag): t for t in result.tags}

        for result in tagging_results:
            cid = str(result.concept_id)
            cprof = concept_profiles.get(cid, self._empty_profile())
            rows: list[ActivityCalibratedTagRecord] = []
            for tag in result.tags:
                tag_name = str(tag.tag)
                tprof = tag_profiles.get(tag_name, self._empty_profile())
                base_conf = float(np.clip(float(tag.confidence), 0.0, 1.0))
                fused_concept = self._weighted_mix(
                    task_score=float(cprof["task_score"]),
                    bitmask_score=float(cprof["bitmask_score"]),
                )
                fused_tag = self._weighted_mix(
                    task_score=float(tprof["task_score"]),
                    bitmask_score=float(tprof["bitmask_score"]),
                )
                support_concept = int(cprof["support"])
                support_tag = int(tprof["support"])
                support_scale = 0.5 * (
                    min(1.0, np.sqrt(float(support_concept) / max(1.0, float(self.config.min_concept_support))))
                    + min(1.0, np.sqrt(float(support_tag) / max(1.0, float(self.config.min_tag_support))))
                )
                total_score = ((0.6 * fused_concept) + (0.4 * fused_tag)) * float(support_scale)
                calibrated_conf = float(
                    np.clip(
                        (1.0 - float(self.config.mix_base)) * base_conf
                        + float(self.config.mix_base) * total_score,
                        float(self.config.min_confidence),
                        float(self.config.max_confidence),
                    )
                )
                keep = bool(calibrated_conf >= float(self.config.keep_threshold))
                row = ActivityCalibratedTagRecord(
                    concept_id=cid,
                    tag=tag_name,
                    provenance=str(tag.provenance),
                    base_confidence=float(base_conf),
                    calibrated_confidence=float(calibrated_conf),
                    keep=bool(keep),
                    concept_support=int(cprof["support"]),
                    tag_support=int(tprof["support"]),
                    concept_task_score=float(cprof["task_score"]),
                    concept_bitmask_score=float(cprof["bitmask_score"]),
                    tag_task_score=float(tprof["task_score"]),
                    tag_bitmask_score=float(tprof["bitmask_score"]),
                    selected_task=str(cprof["selected_task"]),
                    selected_task_ratio=float(cprof["selected_task_ratio"]),
                    selected_bitmask=int(cprof["selected_bitmask"]),
                    selected_bitmask_ratio=float(cprof["selected_bitmask_ratio"]),
                )
                rows.append(row)
                records.append(row)
            concept_to_rows[cid] = rows

        for result in tagging_results:
            cid = str(result.concept_id)
            rows = sorted(
                concept_to_rows.get(cid, []),
                key=lambda r: float(r.calibrated_confidence),
                reverse=True,
            )
            kept_rows = [r for r in rows if bool(r.keep)]
            if len(kept_rows) == 0 and bool(self.config.fallback_top1_if_empty) and len(rows) > 0:
                kept_rows = [rows[0]]

            base_tag_map = tag_by_concept.get(cid, {})
            new_tags: list[TagAssignment] = []
            for row in kept_rows:
                base_tag = base_tag_map.get(str(row.tag))
                if base_tag is None:
                    continue
                new_evidence = dict(base_tag.evidence_json)
                new_evidence["activity_calibration"] = {
                    "base_confidence": float(row.base_confidence),
                    "calibrated_confidence": float(row.calibrated_confidence),
                    "concept_support_train": int(row.concept_support),
                    "tag_support_train": int(row.tag_support),
                    "selected_task": str(row.selected_task),
                    "selected_task_ratio": float(row.selected_task_ratio),
                    "selected_bitmask": int(row.selected_bitmask),
                    "selected_bitmask_ratio": float(row.selected_bitmask_ratio),
                    "min_w": float(self.config.min_w),
                    "task_weight": float(self.config.task_weight),
                    "bitmask_weight": float(self.config.bitmask_weight),
                    "keep_threshold": float(self.config.keep_threshold),
                }
                new_tags.append(
                    TagAssignment(
                        concept_id=str(base_tag.concept_id),
                        tag=str(base_tag.tag),
                        confidence=float(row.calibrated_confidence),
                        provenance=f"{base_tag.provenance}|activity_calibrated",
                        evidence_json=new_evidence,
                    )
                )

            ev = dict(result.evidence_json)
            ev["activity_calibration"] = {
                "enabled": True,
                "n_input_tags": int(len(result.tags)),
                "n_kept_tags": int(len(new_tags)),
                "kept_tags": [str(t.tag) for t in new_tags],
                "fallback_top1_applied": bool(
                    len(result.tags) > 0 and len(new_tags) == 1 and len([r for r in rows if bool(r.keep)]) == 0
                ),
            }
            calibrated_results.append(
                SemanticTaggingResult(
                    concept_id=str(result.concept_id),
                    label_auto=str(result.label_auto),
                    tags=new_tags,
                    evidence_json=ev,
                )
            )

        summary = {
            "enabled": True,
            "n_train_ids": int(len(ids)),
            "n_tasks": int(len(self.task_cols)),
            "n_concepts": int(len(tagging_results)),
            "n_calibrated_rows": int(len(records)),
            "n_kept_rows": int(sum(1 for r in records if bool(r.keep))),
            "keep_threshold": float(self.config.keep_threshold),
            "min_concept_support": int(self.config.min_concept_support),
            "min_tag_support": int(self.config.min_tag_support),
            "ratio_cap": float(self.config.ratio_cap),
        }
        log_event(
            "INFO",
            "explainability.chem_ace.semantic_activity_calibration.summary",
            **summary,
        )
        return calibrated_results, records, summary

    def _compute_concept_profiles(
        self,
        *,
        concept_mol_map_train: Mapping[str, set[str]],
        ids: Sequence[str],
        y: np.ndarray,
        bitmask_ids: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for concept_id, mols in concept_mol_map_train.items():
            mol_set = {str(x) for x in mols}
            mask = np.asarray([str(mid) in mol_set for mid in ids], dtype=bool)
            out[str(concept_id)] = self._score_mask(
                mask=mask,
                y=y,
                bitmask_ids=bitmask_ids,
                support_target=int(self.config.min_concept_support),
            )
        return out

    def _compute_tag_profiles(
        self,
        *,
        tagging_results: Sequence[SemanticTaggingResult],
        concept_mol_map_train: Mapping[str, set[str]],
        ids: Sequence[str],
        y: np.ndarray,
        bitmask_ids: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        tag_to_mols: dict[str, set[str]] = {}
        for result in tagging_results:
            cid = str(result.concept_id)
            mols = concept_mol_map_train.get(cid, set())
            if len(mols) == 0:
                continue
            for tag in result.tags:
                tag_to_mols.setdefault(str(tag.tag), set()).update(str(x) for x in mols)

        out: dict[str, dict[str, Any]] = {}
        for tag, mols in tag_to_mols.items():
            mol_set = {str(x) for x in mols}
            mask = np.asarray([str(mid) in mol_set for mid in ids], dtype=bool)
            out[str(tag)] = self._score_mask(
                mask=mask,
                y=y,
                bitmask_ids=bitmask_ids,
                support_target=int(self.config.min_tag_support),
            )
        return out

    def _score_mask(
        self,
        *,
        mask: np.ndarray,
        y: np.ndarray,
        bitmask_ids: np.ndarray,
        support_target: int,
    ) -> dict[str, Any]:
        n = int(mask.sum())
        if n <= 0:
            return self._empty_profile()

        task_prev = np.mean(y, axis=0).astype(np.float64)
        task_hits = np.sum(y[mask], axis=0).astype(np.float64)
        prior = float(max(0.0, self.config.prior_strength))
        denom = float(n) + prior
        task_post = (task_hits + (task_prev * prior)) / max(1e-9, denom)
        task_ratio = task_post / np.clip(task_prev, 1e-9, None)
        task_norm = self._ratio_to_unit(task_ratio)
        task_score = self._macro_plus_min(task_norm)

        sel_task_idx = int(np.argmax(task_norm)) if task_norm.size > 0 else -1
        sel_task = self.task_cols[sel_task_idx] if sel_task_idx >= 0 else ""
        sel_task_ratio = float(task_ratio[sel_task_idx]) if sel_task_idx >= 0 else 0.0

        max_mask_id = int(max(int(np.max(bitmask_ids)), (2 ** y.shape[1]) - 1))
        total_counts = np.bincount(bitmask_ids.astype(np.int64), minlength=max_mask_id + 1).astype(np.float64)
        present_counts = np.bincount(bitmask_ids[mask].astype(np.int64), minlength=max_mask_id + 1).astype(np.float64)

        start_idx = 1 if bool(self.config.bitmask_exclude_zero) else 0
        candidate_ids = [
            i
            for i in range(start_idx, max_mask_id + 1)
            if total_counts[i] >= float(max(1, int(self.config.bitmask_min_count)))
        ]
        if len(candidate_ids) == 0:
            candidate_ids = [i for i in range(start_idx, max_mask_id + 1) if total_counts[i] > 0.0]

        bitmask_score = 0.0
        selected_bitmask = -1
        selected_bitmask_ratio = 0.0
        if len(candidate_ids) > 0:
            base_prev = total_counts[candidate_ids] / float(max(1, y.shape[0]))
            hits = present_counts[candidate_ids]
            post = (hits + (base_prev * prior)) / max(1e-9, denom)
            ratio = post / np.clip(base_prev, 1e-9, None)
            norm = self._ratio_to_unit(ratio)
            sel = int(np.argmax(norm))
            bitmask_score = float(norm[sel])
            selected_bitmask = int(candidate_ids[sel])
            selected_bitmask_ratio = float(ratio[sel])

        support_scale = min(1.0, np.sqrt(float(n) / max(1.0, float(support_target))))
        return {
            "support": int(n),
            "task_score": float(task_score * support_scale),
            "bitmask_score": float(bitmask_score * support_scale),
            "selected_task": str(sel_task),
            "selected_task_ratio": float(sel_task_ratio),
            "selected_bitmask": int(selected_bitmask),
            "selected_bitmask_ratio": float(selected_bitmask_ratio),
        }

    def _weighted_mix(self, *, task_score: float, bitmask_score: float) -> float:
        tw = float(max(0.0, self.config.task_weight))
        bw = float(max(0.0, self.config.bitmask_weight))
        den = max(1e-9, tw + bw)
        return float(((tw * float(task_score)) + (bw * float(bitmask_score))) / den)

    def _macro_plus_min(self, vals: np.ndarray) -> float:
        if vals.size <= 0:
            return 0.0
        arr = np.asarray(vals, dtype=np.float64)
        m = float(np.mean(arr))
        mn = float(np.min(arr))
        w = float(np.clip(self.config.min_w, 0.0, 1.0))
        return float((1.0 - w) * m + w * mn)

    def _ratio_to_unit(self, ratio: np.ndarray) -> np.ndarray:
        cap = float(max(1.000001, self.config.ratio_cap))
        rr = np.asarray(ratio, dtype=np.float64)
        rr = np.clip(rr, 1.0, cap)
        return (rr - 1.0) / (cap - 1.0)

    @staticmethod
    def _bitmask_ids(y: np.ndarray) -> np.ndarray:
        yy = (np.asarray(y, dtype=np.int64) > 0).astype(np.int64)
        powers = (1 << np.arange(yy.shape[1], dtype=np.int64)).reshape(1, -1)
        return np.sum(yy * powers, axis=1).astype(np.int64)

    @staticmethod
    def _empty_profile() -> dict[str, Any]:
        return {
            "support": 0,
            "task_score": 0.0,
            "bitmask_score": 0.0,
            "selected_task": "",
            "selected_task_ratio": 0.0,
            "selected_bitmask": -1,
            "selected_bitmask_ratio": 0.0,
        }


__all__ = [
    "ActivityCalibrationConfig",
    "ActivityCalibratedTagRecord",
    "ActivityAwareSemanticCalibrator",
]
