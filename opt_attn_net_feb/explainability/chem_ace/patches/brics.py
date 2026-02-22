from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import BRICSPatchConfig
from ..optional_deps import OptionalDependencyError, require_rdkit
from ..types import PatchRecord
from .base import PatchGenerator, make_patch_record

logger = logging.getLogger(__name__)


class BRICSPatchGenerator(PatchGenerator):
    """Generate BRICS fragment patches with atom-index provenance."""

    patch_type = "brics"

    def __init__(self, config: BRICSPatchConfig | None = None):
        self.config = config or BRICSPatchConfig()

    def generate(self, *, mol_id: str, mol: Any, conf_id: Optional[str] = None) -> list[PatchRecord]:
        if not self.config.enabled:
            return []
        try:
            require_rdkit()
            from rdkit import Chem
            from rdkit.Chem import BRICS
        except OptionalDependencyError:
            logger.warning("RDKit not available; skipping BRICS patch generation")
            return []

        if mol is None:
            return []

        mol_work = Chem.Mol(mol)
        for atom in mol_work.GetAtoms():
            atom.SetIntProp("_orig_idx", int(atom.GetIdx()))

        try:
            fragged = BRICS.BreakBRICSBonds(mol_work)
        except Exception:
            logger.exception("BRICS bond breaking failed", extra={"mol_id": mol_id})
            return []

        out: dict[str, PatchRecord] = {}
        try:
            frags = Chem.GetMolFrags(fragged, asMols=True, sanitizeFrags=False)
        except Exception:
            logger.exception("Failed to enumerate BRICS fragments", extra={"mol_id": mol_id})
            return []

        for frag in frags:
            atom_indices = sorted(
                {
                    int(atom.GetIntProp("_orig_idx"))
                    for atom in frag.GetAtoms()
                    if atom.HasProp("_orig_idx")
                }
            )
            if not atom_indices:
                continue
            try:
                frag_smiles = Chem.MolToSmiles(frag, canonical=True)
            except Exception:
                frag_smiles = None
            try:
                smarts = Chem.MolToSmarts(frag)
            except Exception:
                smarts = None

            patch = make_patch_record(
                mol_id=mol_id,
                conf_id=conf_id,
                patch_type=self.patch_type,
                atom_indices=atom_indices,
                smarts=smarts,
                fragment_repr=frag_smiles,
                feature_metadata={"source": "BRICS"},
            )
            out[patch.patch_id] = patch
        return list(out.values())


__all__ = ["BRICSPatchGenerator"]
