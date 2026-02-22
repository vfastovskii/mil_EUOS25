from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import MurckoPatchConfig
from ..optional_deps import OptionalDependencyError, require_rdkit
from ..types import PatchRecord
from .base import PatchGenerator, make_patch_record

logger = logging.getLogger(__name__)


class MurckoPatchGenerator(PatchGenerator):
    """Generate Murcko scaffold-based patches."""

    patch_type = "murcko"

    def __init__(self, config: MurckoPatchConfig | None = None):
        self.config = config or MurckoPatchConfig()

    def generate(self, *, mol_id: str, mol: Any, conf_id: Optional[str] = None) -> list[PatchRecord]:
        if not self.config.enabled:
            return []
        try:
            require_rdkit()
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except OptionalDependencyError:
            logger.warning("RDKit not available; skipping Murcko patch generation")
            return []

        if mol is None:
            return []

        out: dict[str, PatchRecord] = {}
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return []

        match = mol.GetSubstructMatch(scaffold)
        atom_indices = sorted({int(x) for x in match})
        if atom_indices:
            patch = make_patch_record(
                mol_id=mol_id,
                conf_id=conf_id,
                patch_type=self.patch_type,
                atom_indices=atom_indices,
                smarts=Chem.MolToSmarts(scaffold),
                fragment_repr=Chem.MolToSmiles(scaffold, canonical=True),
                feature_metadata={"source": "murcko_scaffold"},
            )
            out[patch.patch_id] = patch

        if self.config.include_framework:
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            if generic is not None and generic.GetNumAtoms() > 0 and atom_indices:
                patch2 = make_patch_record(
                    mol_id=mol_id,
                    conf_id=conf_id,
                    patch_type=f"{self.patch_type}_framework",
                    atom_indices=atom_indices,
                    smarts=Chem.MolToSmarts(generic),
                    fragment_repr=Chem.MolToSmiles(generic, canonical=True),
                    feature_metadata={"source": "murcko_framework"},
                )
                out[patch2.patch_id] = patch2

        return list(out.values())


__all__ = ["MurckoPatchGenerator"]
