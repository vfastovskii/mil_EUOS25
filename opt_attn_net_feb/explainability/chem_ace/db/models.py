from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

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
    """
    Provides the base class for SQLAlchemy ORM models with declarative mapping.

    This class serves as the foundational base class for all SQLAlchemy ORM
    models in a project. It establishes a declarative mapping that allows
    developers to define database models as Python classes.

    Attributes
    ----------
    metadata : MetaData
        Associated MetaData instance allowing for schema reflection and
        database interaction in SQLAlchemy.
    """


class RunORM(Base):
    """
    Represents a database table for storing run information.

    This class is a mapped ORM model that defines the schema and behavior for the
    'runs' table in a database. It contains attributes that correspond to the
    columns of the table and their respective constraints.

    Attributes:
        id: A unique identifier for each run, used as the primary key.
        name: The name of the run, required and cannot be empty.
        created_at: The UTC timestamp indicating when the run was created,
            with a default value of the current time if not set explicitly.
        config_json: The JSON configuration associated with the run,
            stored in a text format and required.
        concept_set_id: An optional foreign key referencing the id of a
            concept set in the 'concept_sets' table.
    """
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    concept_set_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("concept_sets.id"), nullable=True)


class TaskORM(Base):
    """
    Represents a database model for tasks.

    This class defines the structure of the 'tasks' table in the database,
    including its columns and their properties. It is used to interact with
    the database ORM and perform operations related to the 'tasks' table.

    Attributes:
        id: The primary key for the task. A unique string identifier with
            a maximum length of 64 characters.
        description: An optional description of the task, stored as a string
            with a maximum length of 512 characters.
    """
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)


class MoleculeORM(Base):
    """
    Represents a Molecule ORM model for database interactions.

    This class defines the ORM representation of a molecule with database
    mapping using SQLAlchemy. It is responsible for storing the unique
    identifier of a molecule in the database.
    """
    __tablename__ = "molecules"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)


class ConformerORM(Base):
    """
    Represents a conformer in the database.

    This class defines the ORM model for conformers, which are specific 3D arrangements of
    molecules. It includes fields to store information regarding the conformer's unique
    identifier, associated molecule id, and an optional conformer identifier.

    Attributes:
        id (Mapped[str]): The primary key representing the unique identifier of the conformer.
        mol_id (Mapped[str]): The foreign key referencing the molecule associated with this conformer.
        conf_id (Mapped[Optional[str]]): An optional identifier for the specific conformer.
    """
    __tablename__ = "conformers"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    mol_id: Mapped[str] = mapped_column(String(128), ForeignKey("molecules.id"), nullable=False)
    conf_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class PatchORM(Base):
    """
    Represents a database model for storing patch information.

    This class serves as an ORM mapping for the `patches` table in a database. It is
    used to handle and manipulate patch-related data, which includes information
    such as molecule ID, patch type, atom indices, SMARTS patterns, and feature
    metadata. Each instance of this class corresponds to a single row in the table.

    Attributes:
        id: A unique identifier for the patch (primary key).
        mol_id: References the associated molecule by its ID.
        conf_id: Optionally references a specific conformer for the molecule.
        patch_type: Identifies the type of patch.
        atom_indices_json: A JSON-encoded string representing indices of atoms
            involved in the patch.
        smarts: Optionally stores a SMARTS pattern representing the patch.
        fragment_repr: Optionally stores a textual representation of the fragment.
        feature_metadata_json: A JSON-encoded string containing metadata about
            features of the patch.
        patch_hash: Stores a hash of the patch, potentially for validation or
            integrity checks.
        created_at: Timestamp indicating when the entry was created (UTC time).
    """
    __tablename__ = "patches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mol_id: Mapped[str] = mapped_column(String(128), ForeignKey("molecules.id"), nullable=False)
    conf_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    patch_type: Mapped[str] = mapped_column(String(64), nullable=False)
    atom_indices_json: Mapped[str] = mapped_column(Text, nullable=False)
    smarts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fragment_repr: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    feature_metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    patch_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class PatchEmbeddingORM(Base):
    """
    Represents the PatchEmbeddingORM model for managing patch embedding data in a database.

    This class defines a database table named 'patch_embeddings'. It is designed to store
    information related to patch embeddings, such as associated patch IDs, layer names,
    strategies, and other relevant metadata. The table includes a unique constraint to ensure
    that combinations of patch_id, layer_name, and strategy remain unique.

    Attributes:
        id: Primary key for the database table.
        patch_id: Foreign key reference to the patches table, representing the patch ID.
        layer_name: Name of the layer associated with the patch embedding.
        strategy: Strategy used for generating the patch embedding.
        embedding_uri: URI pointing to the location of the patch embedding data.
        embedding_dim: Dimensionality of the patch embedding.
        metadata_json: JSON string containing additional metadata information for the patch embedding.
    """
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
    """
    Represents a database table for storing concept sets.

    This class is used as an ORM model for a database table named 'concept_sets'. It contains
    information about concept sets, including their identifiers, run associations, configuration,
    metadata, versioning, and creation timestamps. The table enforces a unique constraint on the
    combination of 'run_id' and 'version'.

    Attributes:
        id: A string serving as the unique identifier for the concept set (primary key).
        run_id: The string identifier for the run with which the concept set is associated.
        layer_name: A string representing the name of the layer relevant to the concept set.
        version: An integer representing the version of the concept set.
        config_json: A string field storing JSON-encoded configuration data for the concept set.
        metadata_json: A string field storing JSON-encoded metadata for the concept set.
        immutable: A boolean field indicating whether the concept set is immutable.
        created_at: The date and time when this concept set record was created.
    """
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
    """
    Represents a database model for storing concept information.

    This class defines a table structure for storing details related to concepts. Each instance
    represents a concept with attributes such as its unique identifier, relations to other
    concept sets, algorithm used for generating the concept, and various other metadata.
    The data is intended to be utilized in processes that require concept aggregation,
    analysis, and querying.

    Attributes:
        id: The unique identifier of the concept.
        concept_set_id: The foreign key linking the concept to its parent concept set.
        algorithm: The algorithm used to compute the concept.
        support: The support value indicating the frequency or relevance measure of the concept.
        coherence: The coherence score representing the internal consistency of the concept.
        centroid_uri: A text representation of the URI identifying the centroid related to the concept.
        medoid_patch_id: The foreign key linking the concept to its medoid patch.
        label_auto: An optional automatically generated label for the concept.
        metadata_json: A JSON string containing additional metadata about the concept.
    """
    __tablename__ = "concepts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    concept_set_id: Mapped[str] = mapped_column(String(64), ForeignKey("concept_sets.id"), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    support: Mapped[int] = mapped_column(Integer, nullable=False)
    coherence: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_uri: Mapped[str] = mapped_column(Text, nullable=False)
    medoid_patch_id: Mapped[str] = mapped_column(String(64), ForeignKey("patches.id"), nullable=False)
    modality: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    label_auto: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


class ConceptMembershipORM(Base):
    """
    Represents the concept membership relationships within the system.

    This class is an ORM model used to define and manage the concept membership
    relationships in the database. It maps members to their respective concepts
    and patches, providing additional attributes such as membership scores and
    distances to centroids.

    Attributes:
        id: The primary key for the concept_memberships table.
        concept_id: A foreign key that references the ID of a concept.
        patch_id: A foreign key that references the ID of a patch.
        membership_score: A numeric value representing the membership score of
            the entity to the concept.
        distance_to_centroid: A numeric value representing the distance of the
            entity to the concept centroid.
    """
    __tablename__ = "concept_memberships"
    __table_args__ = (
        UniqueConstraint("concept_id", "patch_id", name="uq_concept_membership"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[str] = mapped_column(String(64), ForeignKey("concepts.id"), nullable=False)
    patch_id: Mapped[str] = mapped_column(String(64), ForeignKey("patches.id"), nullable=False)
    modality: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    membership_score: Mapped[float] = mapped_column(Float, nullable=False)
    distance_to_centroid: Mapped[float] = mapped_column(Float, nullable=False)


class ConceptTagORM(Base):
    """
    Represents the mapping of concept tags to their associated data.

    This ORM class defines the structure for the 'concept_tags' table in the database. It includes
    fields such as the unique identifier for each record, the corresponding concept ID, the tag
    associated with the concept, the confidence level of the tagging, provenance details of the
    tagging, and evidence data in JSON format.

    Attributes:
        id: A unique integer identifier for each record in the table.
        concept_id: A string representing the ID of the related concept from the 'concepts' table.
        tag: A string describing the tag associated with the concept.
        confidence: A float representing the confidence level of the tagging.
        provenance: A string indicating the provenance or source of the tag.
        evidence_json: A string containing JSON formatted evidence data related to the concept tag.
    """
    __tablename__ = "concept_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[str] = mapped_column(String(64), ForeignKey("concepts.id"), nullable=False)
    modality: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    tag: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    provenance: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False)


class CAVORM(Base):
    """
    Represents the Concept Activation Vector (CAV) in a database schema.

    This class is part of the database model and defines the structure of the
    'CAV' table, which stores information related to Concept Activation Vectors
    used in various machine learning tasks. Each CAV is uniquely identified by
    its concept, task, layer, and seed. This table is essential for managing
    and retrieving CAV-related data efficiently.

    Attributes:
        id: An integer, auto-incremented primary key representing the unique
            identifier of the CAV.
        concept_id: A string foreign key referencing the 'concepts' table
            defining the concept associated with the CAV.
        task_id: A string foreign key referencing the 'tasks' table defining
            the task associated with the CAV.
        layer_name: A string defining the name of the layer in the model
            where the CAV applies.
        seed: An integer representing the seed used for reproducibility in
            the CAV computation.
        vector_uri: A string representing the URI where the CAV vector data is
            stored.
        intercept: A float representing the intercept term in the CAV model.
        train_accuracy: A float representing the training accuracy of the CAV
            model.
        metadata_json: A string containing the metadata in JSON format related
            to the CAV.
    """
    __tablename__ = "cavs"
    __table_args__ = (
        UniqueConstraint("concept_id", "task_id", "layer_name", "seed", name="uq_cav"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[str] = mapped_column(String(64), ForeignKey("concepts.id"), nullable=False)
    concept_modality: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    task_id: Mapped[str] = mapped_column(String(64), ForeignKey("tasks.id"), nullable=False)
    layer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_uri: Mapped[str] = mapped_column(Text, nullable=False)
    intercept: Mapped[float] = mapped_column(Float, nullable=False)
    train_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


class TCAVEpochORM(Base):
    """
    Represents the TCAV epoch results stored in a database.

    This class provides the schema for the 'tcav_epoch' table, which stores the
    results of TCAV (Testing with Concept Activation Vectors) calculations for
    specific epochs, along with related metadata. Each record in the table uniquely
    identifies a TCAV calculation for a particular combination of run_id, epoch,
    concept, task, layer, and seed. It includes additional fields to store the
    results of the calculation and any metadata.
    """
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
    concept_modality: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    task_id: Mapped[str] = mapped_column(String(64), ForeignKey("tasks.id"), nullable=False)
    layer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    tcav_sign_rate: Mapped[float] = mapped_column(Float, nullable=False)
    tcav_mean_directional_derivative: Mapped[float] = mapped_column(Float, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    p_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


class MILConceptEpochORM(Base):
    """
    Represents the relationship between a Multi-Instance Learning (MIL) concept, task,
    and a particular epoch during a model run.

    This class is designed to model and store data about a specific concept's
    characteristics, such as attention, prevalence, and witness rate at a given epoch
    within a specific task and run in a machine learning workflow.

    Attributes:
        id: The primary key identifier for the record in the database.
        run_id: Identifier for the associated run. References the "runs" table.
        epoch: The epoch number within the run and task for which the record applies.
        concept_id: Identifier for the associated concept. References the "concepts" table.
        task_id: Identifier for the associated task. References the "tasks" table.
        attention_support: Measure of the attention support value for the concept during the epoch.
        witness_rate: Rate at which the concept was witnessed during the given epoch.
        attention_entropy: Entropy of the attention distribution for the concept during the epoch.
        prevalence: Prevalence rate of the concept during the epoch.
        metadata_json: JSON metadata providing additional information related to the concept
                       and epoch.
    """
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
