from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..types import CAVRecord, ConceptCandidate, ConceptMembership, PatchEmbeddingRecord, PatchRecord, TCAVRecord, TagAssignment
from .models import (
    CAVORM,
    ConceptMembershipORM,
    ConceptORM,
    ConceptSetORM,
    ConceptTagORM,
    ConformerORM,
    MILConceptEpochORM,
    MoleculeORM,
    PatchEmbeddingORM,
    PatchORM,
    RunORM,
    TCAVEpochORM,
    TaskORM,
)
from .session import build_engine, initialize_database, make_session_factory


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class MILConceptEpochMetric:
    """Optional MIL-specific concept metrics per epoch."""

    run_id: str
    epoch: int
    concept_id: str
    task_id: str
    attention_support: float
    witness_rate: float
    attention_entropy: float
    prevalence: float
    metadata: Mapping[str, Any]


class ChemACERepository:
    """Persistence layer for Chem-ACE metadata and artifacts."""

    def __init__(self, *, db_uri: str, artifact_dir: str):
        self.db_uri = str(db_uri)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.engine = build_engine(self.db_uri)
        initialize_database(self.engine)
        self.SessionFactory = make_session_factory(self.engine)

    def session(self) -> Session:
        """Create a new DB session."""
        return self.SessionFactory()

    def _vector_uri(self, *, group: str, vector_id: str, vector: np.ndarray) -> str:
        out_dir = self.artifact_dir / str(group)
        out_dir.mkdir(parents=True, exist_ok=True)
        uri = out_dir / f"{vector_id}.npy"
        np.save(uri, np.asarray(vector, dtype=np.float32))
        return str(uri)

    def _default_id(self, *parts: str) -> str:
        payload = "|".join(parts)
        return sha1(payload.encode("utf-8")).hexdigest()

    def create_run(self, *, run_name: str, config: Mapping[str, Any], run_id: Optional[str] = None) -> str:
        """Create run row and return run_id."""
        rid = run_id or self._default_id(
            run_name,
            _json_dumps(config),
            datetime.now(timezone.utc).isoformat(),
        )
        with self.session() as s:
            obj = RunORM(
                id=rid,
                name=str(run_name),
                config_json=_json_dumps(dict(config)),
                created_at=datetime.now(timezone.utc),
            )
            s.add(obj)
            s.commit()
        return rid

    def set_run_concept_set(self, *, run_id: str, concept_set_id: str) -> None:
        with self.session() as s:
            run = s.get(RunORM, run_id)
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            run.concept_set_id = str(concept_set_id)
            s.commit()

    def ensure_tasks(self, task_ids: Sequence[str]) -> None:
        """Insert missing tasks."""
        with self.session() as s:
            existing = {t[0] for t in s.execute(select(TaskORM.id)).all()}
            for task_id in task_ids:
                if task_id not in existing:
                    s.add(TaskORM(id=str(task_id), description=None))
            s.commit()

    def upsert_patch(self, patch: PatchRecord) -> None:
        """Persist one patch record plus parent molecule/conformer rows."""
        with self.session() as s:
            mol = s.get(MoleculeORM, patch.mol_id)
            if mol is None:
                s.add(MoleculeORM(id=str(patch.mol_id)))

            if patch.conf_id is not None:
                conformer_pk = f"{patch.mol_id}:{patch.conf_id}"
                conf = s.get(ConformerORM, conformer_pk)
                if conf is None:
                    s.add(ConformerORM(id=conformer_pk, mol_id=str(patch.mol_id), conf_id=str(patch.conf_id)))

            obj = s.get(PatchORM, patch.patch_id)
            if obj is None:
                s.add(
                    PatchORM(
                        id=str(patch.patch_id),
                        mol_id=str(patch.mol_id),
                        conf_id=(None if patch.conf_id is None else str(patch.conf_id)),
                        patch_type=str(patch.patch_type),
                        atom_indices_json=_json_dumps(list(patch.atom_indices)),
                        smarts=patch.smarts,
                        fragment_repr=patch.fragment_repr,
                        feature_metadata_json=_json_dumps(dict(patch.feature_metadata)),
                        patch_hash=str(patch.patch_hash),
                    )
                )
            s.commit()

    def upsert_patches(self, patches: Iterable[PatchRecord]) -> None:
        for patch in patches:
            self.upsert_patch(patch)

    def upsert_patch_embedding(self, rec: PatchEmbeddingRecord) -> None:
        """Persist embedding metadata (vector path should already exist)."""
        if rec.embedding_uri is None:
            raise ValueError("PatchEmbeddingRecord.embedding_uri is required for DB persistence")
        with self.session() as s:
            existing = s.execute(
                select(PatchEmbeddingORM).where(
                    PatchEmbeddingORM.patch_id == rec.patch_id,
                    PatchEmbeddingORM.layer_name == rec.layer_name,
                    PatchEmbeddingORM.strategy == rec.strategy,
                )
            ).scalar_one_or_none()
            if existing is None:
                s.add(
                    PatchEmbeddingORM(
                        patch_id=str(rec.patch_id),
                        layer_name=str(rec.layer_name),
                        strategy=str(rec.strategy),
                        embedding_uri=str(rec.embedding_uri),
                        embedding_dim=int(rec.vector.shape[-1]),
                        metadata_json=_json_dumps(dict(rec.metadata)),
                    )
                )
            else:
                existing.embedding_uri = str(rec.embedding_uri)
                existing.embedding_dim = int(rec.vector.shape[-1])
                existing.metadata_json = _json_dumps(dict(rec.metadata))
            s.commit()

    def upsert_patch_embeddings(self, recs: Iterable[PatchEmbeddingRecord]) -> None:
        """Persist embedding metadata for many records in one DB transaction."""
        with self.session() as s:
            for rec in recs:
                if rec.embedding_uri is None:
                    raise ValueError("PatchEmbeddingRecord.embedding_uri is required for DB persistence")
                existing = s.execute(
                    select(PatchEmbeddingORM).where(
                        PatchEmbeddingORM.patch_id == rec.patch_id,
                        PatchEmbeddingORM.layer_name == rec.layer_name,
                        PatchEmbeddingORM.strategy == rec.strategy,
                    )
                ).scalar_one_or_none()
                if existing is None:
                    s.add(
                        PatchEmbeddingORM(
                            patch_id=str(rec.patch_id),
                            layer_name=str(rec.layer_name),
                            strategy=str(rec.strategy),
                            embedding_uri=str(rec.embedding_uri),
                            embedding_dim=int(rec.vector.shape[-1]),
                            metadata_json=_json_dumps(dict(rec.metadata)),
                        )
                    )
                else:
                    existing.embedding_uri = str(rec.embedding_uri)
                    existing.embedding_dim = int(rec.vector.shape[-1])
                    existing.metadata_json = _json_dumps(dict(rec.metadata))
            s.commit()

    def create_concept_set_snapshot(
        self,
        *,
        run_id: str,
        layer_name: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
        concept_set_id: Optional[str] = None,
    ) -> str:
        """Create immutable concept-set snapshot row and return concept_set_id."""
        with self.session() as s:
            version = int(
                s.execute(
                    select(func.count(ConceptSetORM.id)).where(ConceptSetORM.run_id == str(run_id))
                ).scalar_one()
            ) + 1
            cs_id = concept_set_id or self._default_id(run_id, layer_name, str(version))
            s.add(
                ConceptSetORM(
                    id=str(cs_id),
                    run_id=str(run_id),
                    layer_name=str(layer_name),
                    version=version,
                    config_json=_json_dumps(dict(config)),
                    metadata_json=_json_dumps(dict(metadata)),
                    immutable=True,
                )
            )
            s.commit()
            return str(cs_id)

    def upsert_concept(self, *, concept_set_id: str, cand: ConceptCandidate) -> str:
        """Persist concept and centroid vector; concept id equals local id."""
        concept_id = str(cand.concept_local_id)
        centroid_uri = self._vector_uri(group="centroids", vector_id=concept_id, vector=cand.centroid)
        with self.session() as s:
            existing = s.get(ConceptORM, concept_id)
            if existing is None:
                s.add(
                    ConceptORM(
                        id=concept_id,
                        concept_set_id=str(concept_set_id),
                        algorithm=str(cand.algorithm),
                        support=int(cand.support),
                        coherence=float(cand.coherence),
                        centroid_uri=str(centroid_uri),
                        medoid_patch_id=str(cand.medoid_patch_id),
                        label_auto=None,
                        metadata_json=_json_dumps(dict(cand.metadata)),
                    )
                )
            s.commit()
        return concept_id

    def set_concept_label(self, *, concept_id: str, label_auto: str) -> None:
        with self.session() as s:
            concept = s.get(ConceptORM, str(concept_id))
            if concept is None:
                raise KeyError(f"Concept not found: {concept_id}")
            concept.label_auto = str(label_auto)
            s.commit()

    def upsert_concepts(self, *, concept_set_id: str, candidates: Iterable[ConceptCandidate]) -> list[str]:
        concept_ids: list[str] = []
        for cand in candidates:
            concept_ids.append(self.upsert_concept(concept_set_id=concept_set_id, cand=cand))
        return concept_ids

    def upsert_memberships(self, memberships: Iterable[ConceptMembership]) -> None:
        with self.session() as s:
            for m in memberships:
                existing = s.execute(
                    select(ConceptMembershipORM).where(
                        ConceptMembershipORM.concept_id == str(m.concept_local_id),
                        ConceptMembershipORM.patch_id == str(m.patch_id),
                    )
                ).scalar_one_or_none()
                if existing is None:
                    s.add(
                        ConceptMembershipORM(
                            concept_id=str(m.concept_local_id),
                            patch_id=str(m.patch_id),
                            membership_score=float(m.membership_score),
                            distance_to_centroid=float(m.distance_to_centroid),
                        )
                    )
                elif float(m.membership_score) > float(existing.membership_score):
                    existing.membership_score = float(m.membership_score)
                    existing.distance_to_centroid = float(m.distance_to_centroid)
            s.commit()

    def upsert_tags(self, tags: Iterable[TagAssignment]) -> None:
        with self.session() as s:
            for tag in tags:
                s.add(
                    ConceptTagORM(
                        concept_id=str(tag.concept_id),
                        tag=str(tag.tag),
                        confidence=float(tag.confidence),
                        provenance=str(tag.provenance),
                        evidence_json=_json_dumps(dict(tag.evidence_json)),
                    )
                )
            s.commit()

    def upsert_cav(self, record: CAVRecord) -> None:
        vector_id = self._default_id(record.concept_id, record.task_id, record.layer_name, str(record.seed))
        vector_uri = self._vector_uri(group="cavs", vector_id=vector_id, vector=record.cav_vector)
        with self.session() as s:
            existing = s.execute(
                select(CAVORM).where(
                    CAVORM.concept_id == str(record.concept_id),
                    CAVORM.task_id == str(record.task_id),
                    CAVORM.layer_name == str(record.layer_name),
                    CAVORM.seed == int(record.seed),
                )
            ).scalar_one_or_none()
            if existing is None:
                s.add(
                    CAVORM(
                        concept_id=str(record.concept_id),
                        task_id=str(record.task_id),
                        layer_name=str(record.layer_name),
                        seed=int(record.seed),
                        vector_uri=str(vector_uri),
                        intercept=float(record.intercept),
                        train_accuracy=float(record.train_accuracy),
                        metadata_json=_json_dumps(dict(record.metadata)),
                    )
                )
            else:
                existing.vector_uri = str(vector_uri)
                existing.intercept = float(record.intercept)
                existing.train_accuracy = float(record.train_accuracy)
                existing.metadata_json = _json_dumps(dict(record.metadata))
            s.commit()

    def upsert_tcav_epoch(self, record: TCAVRecord) -> None:
        with self.session() as s:
            existing = s.execute(
                select(TCAVEpochORM).where(
                    TCAVEpochORM.run_id == str(record.run_id),
                    TCAVEpochORM.epoch == int(record.epoch),
                    TCAVEpochORM.concept_id == str(record.concept_id),
                    TCAVEpochORM.task_id == str(record.task_id),
                    TCAVEpochORM.layer_name == str(record.layer_name),
                    TCAVEpochORM.seed == int(record.seed),
                )
            ).scalar_one_or_none()
            if existing is None:
                s.add(
                    TCAVEpochORM(
                        run_id=str(record.run_id),
                        epoch=int(record.epoch),
                        concept_id=str(record.concept_id),
                        task_id=str(record.task_id),
                        layer_name=str(record.layer_name),
                        seed=int(record.seed),
                        tcav_sign_rate=float(record.tcav_sign_rate),
                        tcav_mean_directional_derivative=float(record.tcav_mean_directional_derivative),
                        n_samples=int(record.n_samples),
                        p_value=(None if record.p_value is None else float(record.p_value)),
                        metadata_json=_json_dumps(dict(record.metadata)),
                    )
                )
            else:
                existing.tcav_sign_rate = float(record.tcav_sign_rate)
                existing.tcav_mean_directional_derivative = float(record.tcav_mean_directional_derivative)
                existing.n_samples = int(record.n_samples)
                existing.p_value = None if record.p_value is None else float(record.p_value)
                existing.metadata_json = _json_dumps(dict(record.metadata))
            s.commit()

    def upsert_mil_concept_epoch(self, metric: MILConceptEpochMetric) -> None:
        with self.session() as s:
            existing = s.execute(
                select(MILConceptEpochORM).where(
                    MILConceptEpochORM.run_id == str(metric.run_id),
                    MILConceptEpochORM.epoch == int(metric.epoch),
                    MILConceptEpochORM.concept_id == str(metric.concept_id),
                    MILConceptEpochORM.task_id == str(metric.task_id),
                )
            ).scalar_one_or_none()
            payload = _json_dumps(dict(metric.metadata))
            if existing is None:
                s.add(
                    MILConceptEpochORM(
                        run_id=str(metric.run_id),
                        epoch=int(metric.epoch),
                        concept_id=str(metric.concept_id),
                        task_id=str(metric.task_id),
                        attention_support=float(metric.attention_support),
                        witness_rate=float(metric.witness_rate),
                        attention_entropy=float(metric.attention_entropy),
                        prevalence=float(metric.prevalence),
                        metadata_json=payload,
                    )
                )
            else:
                existing.attention_support = float(metric.attention_support)
                existing.witness_rate = float(metric.witness_rate)
                existing.attention_entropy = float(metric.attention_entropy)
                existing.prevalence = float(metric.prevalence)
                existing.metadata_json = payload
            s.commit()


__all__ = ["ChemACERepository", "MILConceptEpochMetric"]
