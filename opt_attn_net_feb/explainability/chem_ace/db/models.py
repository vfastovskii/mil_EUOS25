from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative SQLAlchemy base for Chem-ACE tables."""


class RunORM(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    concept_set_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("concept_sets.id"), nullable=True)


class TaskORM(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)


class MoleculeORM(Base):
    __tablename__ = "molecules"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)


class ConformerORM(Base):
    __tablename__ = "conformers"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    mol_id: Mapped[str] = mapped_column(String(128), ForeignKey("molecules.id"), nullable=False)
    conf_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class PatchORM(Base):
    __tablename__ = "patches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mol_id: Mapped[str] = mapped_column(String(128), ForeignKey("molecules.id"), nullable=False)
    conf_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    patch_type: Mapped[str] = mapped_column(String(64), nullable=False)
    atom_indices_json: Mapped[str] = mapped_column(Text, nullable=False)
    smarts: Mapped[str | None] = mapped_column(Text, nullable=True)
    fragment_repr: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    patch_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class PatchEmbeddingORM(Base):
    __tablename__ = "patch_embeddings"
    __table_args__ = (
        UniqueConstraint("patch_id", "layer_name", "strategy", name="uq_patch_embedding"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patch_id: Mapped[str] = mapped_column(String(64), ForeignKey("patches.id"), nullable=False)
    layer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_uri: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


class ConceptSetORM(Base):
    __tablename__ = "concept_sets"
    __table_args__ = (
        UniqueConstraint("run_id", "version", name="uq_concept_set_run_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False)
    layer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ConceptORM(Base):
    __tablename__ = "concepts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    concept_set_id: Mapped[str] = mapped_column(String(64), ForeignKey("concept_sets.id"), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    support: Mapped[int] = mapped_column(Integer, nullable=False)
    coherence: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_uri: Mapped[str] = mapped_column(Text, nullable=False)
    medoid_patch_id: Mapped[str] = mapped_column(String(64), ForeignKey("patches.id"), nullable=False)
    label_auto: Mapped[str | None] = mapped_column(String(256), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


class ConceptMembershipORM(Base):
    __tablename__ = "concept_memberships"
    __table_args__ = (
        UniqueConstraint("concept_id", "patch_id", name="uq_concept_membership"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[str] = mapped_column(String(64), ForeignKey("concepts.id"), nullable=False)
    patch_id: Mapped[str] = mapped_column(String(64), ForeignKey("patches.id"), nullable=False)
    membership_score: Mapped[float] = mapped_column(Float, nullable=False)
    distance_to_centroid: Mapped[float] = mapped_column(Float, nullable=False)


class ConceptTagORM(Base):
    __tablename__ = "concept_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[str] = mapped_column(String(64), ForeignKey("concepts.id"), nullable=False)
    tag: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    provenance: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False)


class CAVORM(Base):
    __tablename__ = "cavs"
    __table_args__ = (
        UniqueConstraint("concept_id", "task_id", "layer_name", "seed", name="uq_cav"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[str] = mapped_column(String(64), ForeignKey("concepts.id"), nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), ForeignKey("tasks.id"), nullable=False)
    layer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_uri: Mapped[str] = mapped_column(Text, nullable=False)
    intercept: Mapped[float] = mapped_column(Float, nullable=False)
    train_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


class TCAVEpochORM(Base):
    __tablename__ = "tcav_epoch"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "epoch",
            "concept_id",
            "task_id",
            "layer_name",
            "seed",
            name="uq_tcav_epoch",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    concept_id: Mapped[str] = mapped_column(String(64), ForeignKey("concepts.id"), nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), ForeignKey("tasks.id"), nullable=False)
    layer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    tcav_sign_rate: Mapped[float] = mapped_column(Float, nullable=False)
    tcav_mean_directional_derivative: Mapped[float] = mapped_column(Float, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


class MILConceptEpochORM(Base):
    __tablename__ = "mil_concept_epoch"
    __table_args__ = (
        UniqueConstraint("run_id", "epoch", "concept_id", "task_id", name="uq_mil_concept_epoch"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    concept_id: Mapped[str] = mapped_column(String(64), ForeignKey("concepts.id"), nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), ForeignKey("tasks.id"), nullable=False)
    attention_support: Mapped[float] = mapped_column(Float, nullable=False)
    witness_rate: Mapped[float] = mapped_column(Float, nullable=False)
    attention_entropy: Mapped[float] = mapped_column(Float, nullable=False)
    prevalence: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


__all__ = [
    "Base",
    "CAVORM",
    "ConceptMembershipORM",
    "ConceptORM",
    "ConceptSetORM",
    "ConceptTagORM",
    "ConformerORM",
    "MILConceptEpochORM",
    "MoleculeORM",
    "PatchEmbeddingORM",
    "PatchORM",
    "RunORM",
    "TCAVEpochORM",
    "TaskORM",
]
