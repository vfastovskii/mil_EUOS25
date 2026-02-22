from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import LocalSubgraphPatchConfig
from ..optional_deps import OptionalDependencyError, require_rdkit
from ..types import PatchRecord
from .base import PatchGenerator, make_patch_record

logger = logging.getLogger(__name__)


class LocalSubgraphPatchGenerator(PatchGenerator):
    """Generate atom-centered radius-r neighborhood patches."""

    patch_type = "local_subgraph"

    def __init__(self, config: LocalSubgraphPatchConfig | None = None):
        self.config = config or LocalSubgraphPatchConfig()

    def generate(self, *, mol_id: str, mol: Any, conf_id: Optional[str] = None) -> list[PatchRecord]:
        try:
            require_rdkit()
            from rdkit import Chem
        except OptionalDependencyError:
            logger.warning("RDKit not available; skipping local subgraph patch generation")
            return []

        if mol is None:
            return []

        out: dict[str, PatchRecord] = {}
        n_atoms = int(mol.GetNumAtoms())
        for atom_idx in range(n_atoms):
            for radius in self.config.radii:
                if int(radius) < 0:
                    continue
                env_bond_ids = Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), atom_idx)
                atom_ids: set[int] = set()
                for bidx in env_bond_ids:
                    bond = mol.GetBondWithIdx(int(bidx))
                    atom_ids.add(int(bond.GetBeginAtomIdx()))
                    atom_ids.add(int(bond.GetEndAtomIdx()))
                if self.config.include_center_atom:
                    atom_ids.add(int(atom_idx))
                if not atom_ids:
                    continue

                atom_indices = sorted(atom_ids)
                try:
                    frag_smiles = Chem.MolFragmentToSmiles(
                        mol,
                        atomsToUse=atom_indices,
                        canonical=True,
                    )
                except Exception:
                    frag_smiles = None
                try:
                    smarts = Chem.MolFragmentToSmarts(mol, atomsToUse=atom_indices)
                except Exception:
                    smarts = None

                patch = make_patch_record(
                    mol_id=mol_id,
                    conf_id=conf_id,
                    patch_type=self.patch_type,
                    atom_indices=atom_indices,
                    smarts=smarts,
                    fragment_repr=frag_smiles,
                    feature_metadata={
                        "center_atom": int(atom_idx),
                        "radius": int(radius),
                    },
                )
                out[patch.patch_id] = patch
        return list(out.values())


__all__ = ["LocalSubgraphPatchGenerator"]
