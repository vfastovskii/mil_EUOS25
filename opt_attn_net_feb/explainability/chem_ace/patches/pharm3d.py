from __future__ import annotations

import logging
from pathlib import Path
import json
from typing import Any, Optional

from ..config import Pharm3DPatchConfig
from ..optional_deps import OptionalDependencyError, require_rdkit
from ..types import PatchRecord
from .base import PatchGenerator, make_patch_record

logger = logging.getLogger(__name__)


class Pharm3DPatchGenerator(PatchGenerator):
    """Generate pharmacophore-feature patches from conformer-aware RDKit features."""

    patch_type = "pharm3d"
    requires_conformer = True

    def __init__(self, config: Pharm3DPatchConfig | None = None):
        self.config = config or Pharm3DPatchConfig()

    def generate(self, *, mol_id: str, mol: Any, conf_id: Optional[str] = None) -> list[PatchRecord]:
        if not self.config.enabled:
            return []
        try:
            require_rdkit()
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures
        except OptionalDependencyError:
            logger.warning("RDKit not available; skipping Pharm3D patch generation")
            return []

        if mol is None:
            return []
        if mol.GetNumConformers() == 0:
            return []

        conf_idx = self._resolve_conf_idx(mol=mol, conf_id=conf_id)
        if conf_idx is None:
            return []
        fdef_path = Path(RDConfig.RDDataDir) / self.config.feature_factory_name
        if not fdef_path.exists():
            logger.warning("Feature factory file not found", extra={"path": str(fdef_path)})
            return []

        factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_path))
        feats = factory.GetFeaturesForMol(mol, confId=conf_idx)

        out: dict[str, PatchRecord] = {}
        for feat in feats:
            atom_ids = tuple(sorted(int(x) for x in feat.GetAtomIds()))
            if len(atom_ids) == 0:
                continue
            pos = feat.GetPos()
            md = {
                "family": str(feat.GetFamily()),
                "feature_type": str(feat.GetType()),
                "position": {
                    "x": float(pos.x),
                    "y": float(pos.y),
                    "z": float(pos.z),
                },
            }
            patch = make_patch_record(
                mol_id=mol_id,
                conf_id=conf_id,
                patch_type=self.patch_type,
                atom_indices=atom_ids,
                smarts=None,
                fragment_repr=None,
                feature_metadata=md,
            )
            out[patch.patch_id] = patch
        return list(out.values())

    @staticmethod
    def _resolve_conf_idx(*, mol: Any, conf_id: Optional[str]) -> Optional[int]:
        if conf_id is None:
            return -1

        cid = str(conf_id)
        try:
            idx = int(cid)
            mol.GetConformer(int(idx))
            return int(idx)
        except Exception:
            pass

        if mol.HasProp("_chemace_conf_id_map"):
            try:
                mapping = json.loads(mol.GetProp("_chemace_conf_id_map"))
                if cid in mapping:
                    idx = int(mapping[cid])
                    mol.GetConformer(int(idx))
                    return int(idx)
            except Exception:
                pass

        if mol.GetNumConformers() == 1:
            return 0

        return None


__all__ = ["Pharm3DPatchGenerator"]
