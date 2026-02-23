from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


JsonDict = Dict[str, Any]


class RegimeLabel(str, Enum):
    """Discrete training regime labels q(t)."""

    WARMUP = "warmup"
    FITTING = "fitting"
    STABLE = "stable_generalization"
    OVERFIT = "overfit_onset"
    REFIT = "refit"


@dataclass(frozen=True)
class PressureTensorIndex:
    """Metadata mapping tensor axes to identifiers."""

    task_ids: tuple[str, ...]
    concept_ids: tuple[str, ...]
    epochs: tuple[int, ...]


@dataclass(frozen=True)
class EpochTaskMetrics:
    """Task-level metrics at a single epoch."""

    epoch: int
    task_id: str
    attention_entropy: float
    witness_rate: float
    train_metric: float
    val_metric: float
    loss: float
    calibration_error: float
    concentration_entropy: float = 0.0
    concentration_gini: float = 0.0
    topk_mass: float = 0.0
    regime_label: RegimeLabel = RegimeLabel.FITTING
    extra: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class EpochConceptMetrics:
    """Task-concept metrics and pressure values for one epoch."""

    epoch: int
    task_id: str
    concept_id: str
    tcav: float
    tcav_smoothed: float
    delta_tcav: float
    attention_support: float
    prevalence: float
    rho: float
    drift: float
    regime_label: RegimeLabel
    regime_core: float = 0.0
    feedback: float = 0.0
    trend_loop: float = 0.0
    revert_loop: float = 0.0
    dissipation: float = 0.0
    context_term: float = 0.0


@dataclass(frozen=True)
class DynamicsDecomposition:
    """Dynamics contributions for one epoch transition."""

    regime_core: np.ndarray
    feedback: np.ndarray
    trend_loop: np.ndarray
    revert_loop: np.ndarray
    dissipation: np.ndarray
    context_term: np.ndarray
    predicted_next: np.ndarray


@dataclass(frozen=True)
class AlertRecord:
    """Machine-readable pressure alert."""

    run_id: str
    epoch: int
    code: str
    severity: str
    task_id: Optional[str]
    concept_id: Optional[str]
    score: float
    message: str
    details: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationRecord:
    """Intervention recommendation produced from alerts."""

    run_id: str
    epoch: int
    recommendation_type: str
    action_level: str
    task_id: Optional[str]
    concept_id: Optional[str]
    rationale: str
    params: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PressureRunArtifacts:
    """Paths of exported artifacts for one run."""

    tensor_npz: str
    long_csv: str
    long_parquet: Optional[str]
    metadata_json: str
    manifold_html_by_task: Mapping[str, str]
    lattice_html: str
    coupling_html: Optional[str]
    alerts_json: str
    alerts_md: str
    recommendations_json: str
    vtk_path: Optional[str]
    ricci_edges_csv: Optional[str] = None
    ricci_summary_csv: Optional[str] = None
    ricci_flow_npz: Optional[str] = None


@dataclass(frozen=True)
class ConcentrationMetrics:
    """Distribution concentration summary for concept pressure."""

    entropy: float
    gini: float
    topk_mass: float


@dataclass(frozen=True)
class RicciEdgeMetrics:
    """Per-edge Ricci diagnostics for one task and epoch."""

    epoch: int
    task_id: str
    concept_src: str
    concept_dst: str
    weight_raw: float
    curvature: float
    weight_flow: float


@dataclass(frozen=True)
class RicciTaskSummary:
    """Task-level curvature summary for one epoch."""

    epoch: int
    task_id: str
    n_nodes: int
    n_edges: int
    mean_curvature: float
    std_curvature: float
    min_curvature: float
    max_curvature: float
    negative_edge_fraction: float
    strong_negative_edge_fraction: float
    top_negative_src: Optional[str]
    top_negative_dst: Optional[str]
    top_negative_curvature: float


__all__ = [
    "AlertRecord",
    "ConcentrationMetrics",
    "DynamicsDecomposition",
    "EpochConceptMetrics",
    "EpochTaskMetrics",
    "JsonDict",
    "PressureRunArtifacts",
    "PressureTensorIndex",
    "RicciEdgeMetrics",
    "RicciTaskSummary",
    "RecommendationRecord",
    "RegimeLabel",
]
