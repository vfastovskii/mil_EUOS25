from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import json


@dataclass(frozen=True)
class LocalSubgraphPatchConfig:
    """Configuration for atom-centered local subgraph patch generation."""

    radii: tuple[int, ...] = (1,)
    include_center_atom: bool = True


@dataclass(frozen=True)
class BRICSPatchConfig:
    """Configuration for BRICS fragment patch generation."""

    enabled: bool = True


@dataclass(frozen=True)
class MurckoPatchConfig:
    """Configuration for Murcko scaffold patch generation."""

    enabled: bool = True
    include_framework: bool = True


@dataclass(frozen=True)
class Pharm3DPatchConfig:
    """Configuration for optional 3D pharmacophore patch generation."""

    enabled: bool = True
    feature_factory_name: str = "BaseFeatures.fdef"


@dataclass(frozen=True)
class PatchGenerationConfig:
    """Bundle of all patch generator settings."""

    local_subgraph: LocalSubgraphPatchConfig = field(default_factory=LocalSubgraphPatchConfig)
    brics: BRICSPatchConfig = field(default_factory=BRICSPatchConfig)
    murcko: MurckoPatchConfig = field(default_factory=MurckoPatchConfig)
    pharm3d: Pharm3DPatchConfig = field(default_factory=Pharm3DPatchConfig)


@dataclass(frozen=True)
class EmbeddingCacheConfig:
    """Cache configuration for persisted patch embeddings."""

    cache_dir: str = "chem_ace_cache"
    format: str = "npy"
    overwrite: bool = False


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding extraction settings."""

    layer_name: str
    strategy: str = "masked_input"
    normalize: bool = True
    cache: EmbeddingCacheConfig = field(default_factory=EmbeddingCacheConfig)


@dataclass(frozen=True)
class ConceptDiscoveryConfig:
    """Concept clustering and filtering configuration."""

    # Default to a single scalable algorithm for very large embedding sets.
    algorithms: tuple[str, ...] = ("kmeans",)
    # <=1 enables automatic k selection based on embedding count.
    kmeans_k: int = 0
    # Use MiniBatchKMeans above this sample size to reduce memory pressure.
    kmeans_minibatch_over: int = 200000
    kmeans_minibatch_size: int = 4096
    hierarchical_distance_threshold: float = 1.25
    # Agglomerative clustering requires O(N^2) pairwise distances; keep guarded.
    hierarchical_max_samples: int = 25000
    hierarchical_max_pairwise_gb: float = 8.0
    hdbscan_min_cluster_size: int = 12
    # HDBSCAN can become very memory-heavy at large N.
    hdbscan_max_samples: int = 300000
    min_support: int = 8
    min_coherence: float = 0.0
    dedup_centroid_similarity_threshold: float = 0.98


@dataclass(frozen=True)
class CAVConfig:
    """CAV/TCAV evaluation settings."""

    classifier: str = "logreg"
    n_random_repeats: int = 8
    random_counterexamples_per_repeat: int = 128
    max_iter: int = 2000
    use_sign_rate: bool = True
    use_mean_directional_derivative: bool = True


@dataclass(frozen=True)
class SemanticTaggingConfig:
    """Semantic tagging and naming settings."""

    naming_rules_path: Optional[str] = None
    functional_rules_path: Optional[str] = None
    smarts_rx_rules_path: Optional[str] = None
    use_smarts_rx: bool = True
    use_openbabel_descriptors: bool = True
    use_advanced_geom_topology: bool = True
    advanced_geom_topology_max_patches: int = 3000
    advanced_geom_topology_min_atoms: int = 4
    advanced_geom_topology_max_torsion_paths: int = 96
    advanced_geom_use_convex_hull: bool = True
    advanced_geom_use_persistent_homology: bool = True
    advanced_geom_persistence_max_atoms: int = 48
    use_orca_descriptors: bool = False
    orca_descriptors_path: Optional[str] = None
    orca_conf_id_col: str = "conf_id"
    orca_mol_id_col: str = "ID"
    orca_descriptor_cols: tuple[str, ...] = ()
    orca_min_vectors_for_tagging: int = 8
    orca_z_threshold: float = 0.50
    charge_threshold_formal: int = 1
    aromatic_fraction_threshold: float = 0.35
    conjugation_size_threshold: int = 6
    planarity_rmsd_threshold: float = 0.25
    geom_min_vectors_for_tagging: int = 8
    geom_z_threshold: float = 0.50
    geom_strong_z_threshold: float = 1.00
    qm_min_vectors_for_tagging: int = 8
    qm_z_threshold: float = 0.50
    qm_strong_z_threshold: float = 1.00


@dataclass(frozen=True)
class DatabaseConfig:
    """SQL database connection configuration."""

    uri: str = "sqlite:///chem_ace.sqlite3"


@dataclass(frozen=True)
class ChemACEConfig:
    """Top-level Chem-ACE execution configuration."""

    run_name: str = "chem_ace_run"
    seed: int = 0
    output_dir: str = "chem_ace_outputs"
    cpu_workers: int = 0
    # <= 0 enables dynamic cap from target_total_patches / n_molecules.
    max_patches_per_molecule: int = 0
    # Used only when max_patches_per_molecule <= 0.
    target_total_patches: int = 1200000
    progress_log_every_molecules: int = 250
    patch_generation: PatchGenerationConfig = field(default_factory=PatchGenerationConfig)
    embedding: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig(layer_name="encoder"))
    discovery: ConceptDiscoveryConfig = field(default_factory=ConceptDiscoveryConfig)
    cav: CAVConfig = field(default_factory=CAVConfig)
    semantics: SemanticTaggingConfig = field(default_factory=SemanticTaggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config into JSON-compatible dictionary."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Persist configuration to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load_json(path: str | Path) -> Dict[str, Any]:
        """Load raw configuration mapping from JSON file."""
        return json.loads(Path(path).read_text())


__all__ = [
    "BRICSPatchConfig",
    "CAVConfig",
    "ChemACEConfig",
    "ConceptDiscoveryConfig",
    "DatabaseConfig",
    "EmbeddingCacheConfig",
    "EmbeddingConfig",
    "LocalSubgraphPatchConfig",
    "MurckoPatchConfig",
    "PatchGenerationConfig",
    "Pharm3DPatchConfig",
    "SemanticTaggingConfig",
]
