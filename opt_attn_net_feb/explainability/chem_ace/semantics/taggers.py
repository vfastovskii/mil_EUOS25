from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..config import SemanticTaggingConfig
from ..optional_deps import OptionalDependencyError, require_rdkit
from ..types import PatchRecord, TagAssignment
from .naming import NamingRegistry, choose_label

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticTaggingResult:
    """Semantic tags and auto label for one concept."""

    concept_id: str
    label_auto: str
    tags: list[TagAssignment]
    evidence_json: dict[str, Any]


class SemanticTagger:
    """Compute semantic descriptors and map them to chemist-readable tags."""

    def __init__(self, config: SemanticTaggingConfig):
        self.config = config
        if config.naming_rules_path:
            self.registry = NamingRegistry.from_json(config.naming_rules_path)
        else:
            self.registry = NamingRegistry.default()

    def tag_concept(
        self,
        *,
        concept_id: str,
        concept_patches: Sequence[PatchRecord],
        molecules_by_id: Mapping[str, Any],
    ) -> SemanticTaggingResult:
        descriptors = self._compute_descriptors(concept_patches=concept_patches, molecules_by_id=molecules_by_id)
        tags = []
        tags.extend(self._charge_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._conjugation_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._geometry_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._pharmacophore_tags(concept_id=concept_id, descriptors=descriptors))

        tag_names = [t.tag for t in tags]
        descriptor_rank = self._descriptor_rank(descriptors)
        label_auto = choose_label(tags=tag_names, registry=self.registry, descriptor_rank=descriptor_rank)

        return SemanticTaggingResult(
            concept_id=concept_id,
            label_auto=label_auto,
            tags=tags,
            evidence_json=descriptors,
        )

    def _compute_descriptors(
        self,
        *,
        concept_patches: Sequence[PatchRecord],
        molecules_by_id: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            require_rdkit()
            from rdkit.Chem import rdPartialCharges
            from rdkit.Chem import ChemicalFeatures
            from rdkit import RDConfig
        except OptionalDependencyError:
            return {"n_patches": int(len(concept_patches)), "rdkit_available": False}

        formal_charge_sums: list[float] = []
        gasteiger_vals: list[float] = []
        aromatic_atom_fracs: list[float] = []
        aromatic_bond_fracs: list[float] = []
        max_conj_sizes: list[int] = []
        heteroaromatic_flags: list[int] = []
        planarity_rmsd: list[float] = []
        rot_bond_counts: list[int] = []

        pharm_counts = {
            "Donor": 0,
            "Acceptor": 0,
            "Aromatic": 0,
            "PosIonizable": 0,
            "NegIonizable": 0,
        }

        factory = None
        fdef = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
        if fdef.exists():
            try:
                factory = ChemicalFeatures.BuildFeatureFactory(str(fdef))
            except Exception:
                factory = None

        for patch in concept_patches:
            mol = molecules_by_id.get(patch.mol_id)
            if mol is None:
                continue
            atom_ids = [int(i) for i in patch.atom_indices if 0 <= int(i) < int(mol.GetNumAtoms())]
            if len(atom_ids) == 0:
                continue

            # formal and partial charge descriptors
            fsum = float(sum(int(mol.GetAtomWithIdx(i).GetFormalCharge()) for i in atom_ids))
            formal_charge_sums.append(fsum)

            try:
                mol_copy = type(mol)(mol)
                rdPartialCharges.ComputeGasteigerCharges(mol_copy)
                vals = []
                for i in atom_ids:
                    atom = mol_copy.GetAtomWithIdx(i)
                    if atom.HasProp("_GasteigerCharge"):
                        v = atom.GetDoubleProp("_GasteigerCharge")
                        if np.isfinite(v):
                            vals.append(float(v))
                gasteiger_vals.extend(vals)
            except Exception:
                pass

            atoms = [mol.GetAtomWithIdx(i) for i in atom_ids]
            aromatic_atoms = sum(1 for a in atoms if a.GetIsAromatic())
            aromatic_atom_fracs.append(float(aromatic_atoms / max(1, len(atom_ids))))

            patch_bonds = []
            atom_set = set(atom_ids)
            for b in mol.GetBonds():
                a = int(b.GetBeginAtomIdx())
                c = int(b.GetEndAtomIdx())
                if a in atom_set and c in atom_set:
                    patch_bonds.append(b)
            if patch_bonds:
                aromatic_bond_fracs.append(float(sum(1 for b in patch_bonds if b.GetIsAromatic()) / len(patch_bonds)))
            else:
                aromatic_bond_fracs.append(0.0)

            conj_graph: Dict[int, set[int]] = {i: set() for i in atom_ids}
            for b in patch_bonds:
                if b.GetIsConjugated():
                    a = int(b.GetBeginAtomIdx())
                    c = int(b.GetEndAtomIdx())
                    conj_graph[a].add(c)
                    conj_graph[c].add(a)
            max_comp = self._largest_component_size(conj_graph)
            max_conj_sizes.append(int(max_comp))

            heteroaromatic = any(a.GetIsAromatic() and a.GetAtomicNum() not in {1, 6} for a in atoms)
            heteroaromatic_flags.append(1 if heteroaromatic else 0)

            if mol.GetNumConformers() > 0 and len(atom_ids) >= 3:
                conf_idx = self._resolve_conf_idx(mol=mol, conf_id=patch.conf_id)
                if conf_idx is not None:
                    conf = mol.GetConformer(int(conf_idx))
                    coords = np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in atom_ids], dtype=np.float64)
                    planarity_rmsd.append(float(self._planarity_rmsd(coords)))
                else:
                    planarity_rmsd.append(float("nan"))
            elif len(atom_ids) >= 3:
                planarity_rmsd.append(float("nan"))

            rot_bond_counts.append(
                int(
                    sum(
                        1
                        for b in patch_bonds
                        if b.GetBondTypeAsDouble() == 1.0 and (not b.IsInRing())
                    )
                )
            )

            if factory is not None:
                try:
                    for feat in factory.GetFeaturesForMol(mol):
                        atom_ids_feat = set(int(i) for i in feat.GetAtomIds())
                        if atom_ids_feat.intersection(atom_set):
                            fam = str(feat.GetFamily())
                            if fam in pharm_counts:
                                pharm_counts[fam] += 1
                except Exception:
                    pass

        return {
            "n_patches": int(len(concept_patches)),
            "rdkit_available": True,
            "formal_charge_sum_mean": float(np.nanmean(formal_charge_sums)) if formal_charge_sums else 0.0,
            "formal_charge_sum_std": float(np.nanstd(formal_charge_sums)) if formal_charge_sums else 0.0,
            "gasteiger_mean": float(np.nanmean(gasteiger_vals)) if gasteiger_vals else 0.0,
            "gasteiger_min": float(np.nanmin(gasteiger_vals)) if gasteiger_vals else 0.0,
            "gasteiger_max": float(np.nanmax(gasteiger_vals)) if gasteiger_vals else 0.0,
            "gasteiger_std": float(np.nanstd(gasteiger_vals)) if gasteiger_vals else 0.0,
            "charge_separation_proxy": float((np.nanmax(gasteiger_vals) - np.nanmin(gasteiger_vals))) if gasteiger_vals else 0.0,
            "aromatic_atom_fraction_mean": float(np.nanmean(aromatic_atom_fracs)) if aromatic_atom_fracs else 0.0,
            "aromatic_bond_fraction_mean": float(np.nanmean(aromatic_bond_fracs)) if aromatic_bond_fracs else 0.0,
            "largest_conjugated_component_mean": float(np.nanmean(max_conj_sizes)) if max_conj_sizes else 0.0,
            "heteroaromatic_presence_rate": float(np.mean(heteroaromatic_flags)) if heteroaromatic_flags else 0.0,
            "planarity_rmsd_mean": float(np.nanmean(planarity_rmsd)) if planarity_rmsd else float("nan"),
            "rotatable_bond_count_mean": float(np.nanmean(rot_bond_counts)) if rot_bond_counts else 0.0,
            "pharmacophore_counts": dict(pharm_counts),
        }

    @staticmethod
    def _largest_component_size(graph: Mapping[int, set[int]]) -> int:
        seen: set[int] = set()
        best = 0
        for node in graph.keys():
            if node in seen:
                continue
            stack = [node]
            comp = 0
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp += 1
                stack.extend(list(graph[cur]))
            best = max(best, comp)
        return int(best)

    @staticmethod
    def _planarity_rmsd(coords: np.ndarray) -> float:
        ctr = coords.mean(axis=0, keepdims=True)
        centered = coords - ctr
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        d = centered @ normal
        return float(np.sqrt(np.mean(d ** 2)))

    @staticmethod
    def _resolve_conf_idx(*, mol: Any, conf_id: Optional[str]) -> Optional[int]:
        if mol.GetNumConformers() <= 0:
            return None
        if conf_id is None:
            return 0

        key = str(conf_id)
        try:
            idx = int(key)
            mol.GetConformer(int(idx))
            return int(idx)
        except Exception:
            pass

        if mol.HasProp("_chemace_conf_id_map"):
            try:
                mapping = json.loads(mol.GetProp("_chemace_conf_id_map"))
                if key in mapping:
                    idx = int(mapping[key])
                    mol.GetConformer(int(idx))
                    return int(idx)
            except Exception:
                pass

        if mol.GetNumConformers() == 1:
            return 0
        return None

    def _charge_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        tags: list[TagAssignment] = []
        mean_fc = float(descriptors.get("formal_charge_sum_mean", 0.0))
        sep = float(descriptors.get("charge_separation_proxy", 0.0))

        if mean_fc <= -float(self.config.charge_threshold_formal):
            tags.append(self._mk_tag(concept_id, "anionic", 0.8 + min(0.2, abs(mean_fc) / 3.0), "charge_tagger", descriptors))
        elif mean_fc >= float(self.config.charge_threshold_formal):
            tags.append(self._mk_tag(concept_id, "cationic", 0.8 + min(0.2, abs(mean_fc) / 3.0), "charge_tagger", descriptors))
        elif sep >= 0.6:
            tags.append(self._mk_tag(concept_id, "zwitterionic-like", min(1.0, sep / 1.5), "charge_tagger", descriptors))
        elif abs(float(descriptors.get("gasteiger_mean", 0.0))) > 0.05 or sep > 0.25:
            tags.append(self._mk_tag(concept_id, "neutral polar", 0.65, "charge_tagger", descriptors))
        else:
            tags.append(self._mk_tag(concept_id, "neutral nonpolar", 0.65, "charge_tagger", descriptors))
        return tags

    def _conjugation_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        tags: list[TagAssignment] = []
        aro = float(descriptors.get("aromatic_atom_fraction_mean", 0.0))
        conj = float(descriptors.get("largest_conjugated_component_mean", 0.0))
        hetero = float(descriptors.get("heteroaromatic_presence_rate", 0.0))

        if aro >= float(self.config.aromatic_fraction_threshold):
            tags.append(self._mk_tag(concept_id, "aromatic pi-system", min(1.0, aro + 0.2), "conjugation_tagger", descriptors))
        if conj >= float(self.config.conjugation_size_threshold):
            tags.append(self._mk_tag(concept_id, "extended conjugation", min(1.0, conj / 10.0), "conjugation_tagger", descriptors))
        elif conj >= 2.0:
            tags.append(self._mk_tag(concept_id, "isolated unsaturation", 0.6, "conjugation_tagger", descriptors))
        else:
            tags.append(self._mk_tag(concept_id, "aliphatic", 0.6, "conjugation_tagger", descriptors))
        if hetero >= 0.2:
            tags.append(self._mk_tag(concept_id, "heteroaromatic", min(1.0, hetero + 0.3), "conjugation_tagger", descriptors))
        return tags

    def _geometry_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        tags: list[TagAssignment] = []
        rmsd = float(descriptors.get("planarity_rmsd_mean", float("nan")))
        rot = float(descriptors.get("rotatable_bond_count_mean", 0.0))

        if np.isfinite(rmsd):
            if rmsd <= float(self.config.planarity_rmsd_threshold):
                tags.append(self._mk_tag(concept_id, "planar", 0.75, "geometry_tagger", descriptors))
            elif rmsd >= 0.45:
                tags.append(self._mk_tag(concept_id, "non-planar", 0.75, "geometry_tagger", descriptors))
                tags.append(self._mk_tag(concept_id, "twisted", 0.6, "geometry_tagger", descriptors))

        if rot <= 0.5:
            tags.append(self._mk_tag(concept_id, "rigid", 0.65, "geometry_tagger", descriptors))
        if rot >= 2.0:
            tags.append(self._mk_tag(concept_id, "flexible", 0.7, "geometry_tagger", descriptors))
        return tags

    def _pharmacophore_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        tags: list[TagAssignment] = []
        counts = descriptors.get("pharmacophore_counts", {})
        d = int(counts.get("Donor", 0))
        a = int(counts.get("Acceptor", 0))
        ar = int(counts.get("Aromatic", 0))
        p = int(counts.get("PosIonizable", 0))
        n = int(counts.get("NegIonizable", 0))

        if d > 0:
            tags.append(self._mk_tag(concept_id, "HBD", min(1.0, 0.5 + 0.1 * d), "pharmacophore_tagger", descriptors))
        if a > 0:
            tags.append(self._mk_tag(concept_id, "HBA", min(1.0, 0.5 + 0.1 * a), "pharmacophore_tagger", descriptors))
        if ar > 0:
            tags.append(self._mk_tag(concept_id, "aromatic centroid", min(1.0, 0.5 + 0.1 * ar), "pharmacophore_tagger", descriptors))
        if p > 0:
            tags.append(self._mk_tag(concept_id, "cationic center", min(1.0, 0.5 + 0.1 * p), "pharmacophore_tagger", descriptors))
        if n > 0:
            tags.append(self._mk_tag(concept_id, "anionic center", min(1.0, 0.5 + 0.1 * n), "pharmacophore_tagger", descriptors))
        if d > 0 and a > 0:
            tags.append(self._mk_tag(concept_id, "HBD/HBA pair", 0.8, "pharmacophore_tagger", descriptors))
        return tags

    @staticmethod
    def _mk_tag(
        concept_id: str,
        tag: str,
        confidence: float,
        provenance: str,
        descriptors: Mapping[str, Any],
    ) -> TagAssignment:
        return TagAssignment(
            concept_id=str(concept_id),
            tag=str(tag),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            provenance=str(provenance),
            evidence_json=dict(descriptors),
        )

    @staticmethod
    def _descriptor_rank(descriptors: Mapping[str, Any]) -> list[str]:
        out: list[str] = []
        if float(descriptors.get("aromatic_atom_fraction_mean", 0.0)) > 0.35:
            out.append("aromatic")
        if float(descriptors.get("largest_conjugated_component_mean", 0.0)) >= 6:
            out.append("conjugated")
        if abs(float(descriptors.get("formal_charge_sum_mean", 0.0))) >= 1.0:
            out.append("charged")
        if float(descriptors.get("rotatable_bond_count_mean", 0.0)) >= 2.0:
            out.append("flexible")
        rmsd = float(descriptors.get("planarity_rmsd_mean", float("nan")))
        if np.isfinite(rmsd) and rmsd < 0.25:
            out.append("planar")
        if not out:
            out.append("mixed")
        return out


__all__ = ["SemanticTagger", "SemanticTaggingResult"]
