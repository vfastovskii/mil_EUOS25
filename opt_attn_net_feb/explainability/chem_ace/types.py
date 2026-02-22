from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

import numpy as np


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class PatchRecord:
    """Represents one molecular patch candidate used in concept discovery."""

    patch_id: str
    mol_id: str
    conf_id: Optional[str]
    patch_type: str
    atom_indices: tuple[int, ...]
    patch_hash: str
    smarts: Optional[str] = None
    fragment_repr: Optional[str] = None
    feature_metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PatchEmbeddingRecord:
    """Represents one cached embedding vector for a patch at a specific layer."""

    patch_id: str
    layer_name: str
    strategy: str
    vector: np.ndarray
    embedding_uri: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ConceptCandidate:
    """Represents one discovered concept cluster before persistence."""

    concept_local_id: str
    layer_name: str
    algorithm: str
    support: int
    coherence: float
    centroid: np.ndarray
    medoid_patch_id: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ConceptMembership:
    """Represents assignment of a patch to a discovered concept."""

    concept_local_id: str
    patch_id: str
    membership_score: float
    distance_to_centroid: float


@dataclass(frozen=True)
class TagAssignment:
    """Represents one semantic tag attached to a concept."""

    concept_id: str
    tag: str
    confidence: float
    provenance: str
    evidence_json: JsonDict


@dataclass(frozen=True)
class CAVRecord:
    """Represents one trained CAV vector and validation metadata."""

    concept_id: str
    task_id: str
    layer_name: str
    seed: int
    cav_vector: np.ndarray
    intercept: float
    train_accuracy: float
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class TCAVRecord:
    """Represents one TCAV evaluation result."""

    run_id: str
    epoch: int
    concept_id: str
    task_id: str
    layer_name: str
    seed: int
    tcav_sign_rate: float
    tcav_mean_directional_derivative: float
    n_samples: int
    p_value: Optional[float] = None
    metadata: JsonDict = field(default_factory=dict)


class PatchInputBuilder(Protocol):
    """Adapter for building model inputs from patches."""

    def build_masked_input(self, patch: PatchRecord) -> Any:
        """Build model input representing only the patch content."""

    def build_full_input(self, patch: PatchRecord) -> tuple[Any, Sequence[int]]:
        """Build full-model input and return atom indices used by the patch."""


class ModelTaskAdapter(Protocol):
    """Adapter for model-specific logic used by embedding and TCAV."""

    def forward(self, model_input: Any) -> Any:
        """Run forward pass for provided model input."""

    def get_task_scalar(self, model_output: Any, task_id: str) -> Any:
        """Return a scalar tensor-like value for one task."""


class ConceptQueryAPI(Protocol):
    """Protocol for analytics/query methods over persisted concept database."""

    def planar_conjugated_rising_tcav(self, task_id: str, last_n_epochs: int) -> Sequence[Mapping[str, Any]]:
        """Return concepts with rising TCAV and requested semantic tags."""

    def top_attention_support(self, task_id: str, epoch: int, limit: int = 20) -> Sequence[Mapping[str, Any]]:
        """Return top concepts by attention support for a task/epoch."""

    def high_tcav_low_prevalence(self, task_id: str, epoch: int, tcav_thr: float, prevalence_thr: float) -> Sequence[Mapping[str, Any]]:
        """Return rare but influential concepts."""

    def concept_collapse_indicator(self, task_id: str, epoch: int, top_k: int = 5) -> Mapping[str, Any]:
        """Return a concentration indicator of TCAV mass among top concepts."""


__all__ = [
    "CAVRecord",
    "ConceptCandidate",
    "ConceptMembership",
    "ConceptQueryAPI",
    "JsonDict",
    "ModelTaskAdapter",
    "PatchEmbeddingRecord",
    "PatchInputBuilder",
    "PatchRecord",
    "TCAVRecord",
    "TagAssignment",
]
