from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for concept pressure tracker."""

    alpha: float = 0.6
    tcav_ema_beta: float = 0.8
    drift_clip: float = 5.0


@dataclass(frozen=True)
class RicciConfig:
    """Discrete graph-Ricci diagnostics and flow settings."""

    enabled: bool = True
    edge_keep_quantile: float = 0.75
    min_edge_weight: float = 0.05
    top_k_per_node: int = 4

    w_rho: float = 0.40
    w_attention: float = 0.30
    w_prevalence: float = 0.15
    w_tcav_corr: float = 0.15

    negative_curvature_threshold: float = -0.15
    strong_negative_curvature_threshold: float = -0.35

    flow_enabled: bool = True
    flow_steps: int = 8
    flow_step_size: float = 0.12
    flow_eps: float = 1e-4

    use_flow_as_concept_coupling: bool = True
    coupling_strength: float = 0.05


@dataclass(frozen=True)
class RegimeConfig:
    """Rule-based regime inference settings."""

    warmup_epochs: int = 3
    val_slope_small: float = 1e-3
    overfit_gap_threshold: float = 0.03
    entropy_drop_threshold: float = 0.05
    concentration_rise_threshold: float = 0.05


@dataclass(frozen=True)
class DynamicsConfig:
    """Discrete dynamics model settings."""

    lambda_damping: float = 0.08
    phi_state_damping: float = 0.0
    trend_coeff: float = 0.25
    revert_coeff: float = 0.20
    context_scale: float = 0.15
    regime_a: Mapping[str, float] = field(
        default_factory=lambda: {
            "warmup": 0.15,
            "fitting": 0.08,
            "stable_generalization": 0.02,
            "overfit_onset": 0.18,
            "refit": 0.06,
        }
    )
    use_cross_task_coupling: bool = True
    use_cross_concept_coupling: bool = True


@dataclass(frozen=True)
class DetectorConfig:
    """Runaway/collapse detector thresholds."""

    runaway_threshold: float = 1.0
    concentration_top_k: int = 5
    concentration_entropy_drop_alert: float = 0.10
    concentration_topk_mass_alert: float = 0.75
    blocked_concept_positive_drift: float = 0.05
    ricci_negative_edge_fraction_alert: float = 0.40
    ricci_min_curvature_alert: float = -0.45
    ricci_strong_negative_fraction_alert: float = 0.20


@dataclass(frozen=True)
class PolicyConfig:
    """Recommendation policy settings."""

    enabled: bool = True
    auto_action: bool = False


@dataclass(frozen=True)
class ExportConfig:
    """Artifact export settings."""

    output_dir: str = "lambda_vol_outputs"
    export_parquet: bool = True
    export_plotly_html: bool = True
    export_vtk: bool = False
    top_k_lattice: int = 20


@dataclass(frozen=True)
class StoreConfig:
    """Persistence store configuration."""

    db_uri: str = "sqlite:///lambda_vol.sqlite3"


@dataclass(frozen=True)
class LambdaVolConfig:
    """Top-level Lambda-Vol concept-pressure configuration."""

    run_name: str = "lambda_vol_run"
    seed: int = 0
    blocked_concepts: tuple[str, ...] = ()
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ricci: RicciConfig = field(default_factory=RicciConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    exporter: ExportConfig = field(default_factory=ExportConfig)
    store: StoreConfig = field(default_factory=StoreConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


__all__ = [
    "DetectorConfig",
    "DynamicsConfig",
    "ExportConfig",
    "LambdaVolConfig",
    "PolicyConfig",
    "RicciConfig",
    "RegimeConfig",
    "StoreConfig",
    "TrackerConfig",
]
