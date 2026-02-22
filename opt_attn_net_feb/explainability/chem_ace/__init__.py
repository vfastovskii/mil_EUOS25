"""Chem-ACE: automatic concept discovery + semantic tagging for molecular models.

Import submodules directly for heavy functionality, e.g.:
- explainability.chem_ace.concepts
- explainability.chem_ace.cav
- explainability.chem_ace.embedding
- explainability.chem_ace.db
"""

from .config import (
    BRICSPatchConfig,
    CAVConfig,
    ChemACEConfig,
    ConceptDiscoveryConfig,
    DatabaseConfig,
    EmbeddingCacheConfig,
    EmbeddingConfig,
    LocalSubgraphPatchConfig,
    MurckoPatchConfig,
    PatchGenerationConfig,
    Pharm3DPatchConfig,
    SemanticTaggingConfig,
)
from .types import (
    CAVRecord,
    ConceptCandidate,
    ConceptMembership,
    PatchEmbeddingRecord,
    PatchRecord,
    TCAVRecord,
    TagAssignment,
)

__all__ = [
    "BRICSPatchConfig",
    "CAVConfig",
    "CAVRecord",
    "ChemACEConfig",
    "ConceptCandidate",
    "ConceptDiscoveryConfig",
    "ConceptMembership",
    "DatabaseConfig",
    "EmbeddingCacheConfig",
    "EmbeddingConfig",
    "LocalSubgraphPatchConfig",
    "MurckoPatchConfig",
    "PatchEmbeddingRecord",
    "PatchGenerationConfig",
    "PatchRecord",
    "Pharm3DPatchConfig",
    "SemanticTaggingConfig",
    "TCAVRecord",
    "TagAssignment",
]
