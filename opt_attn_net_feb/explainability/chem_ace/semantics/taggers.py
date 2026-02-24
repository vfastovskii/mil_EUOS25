from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ..config import SemanticTaggingConfig
from ..optional_deps import (
    OptionalDependencyError,
    require_openbabel_pybel,
    require_rdkit,
)
from ..types import PatchRecord, TagAssignment
from .naming import NamingRegistry, choose_label

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionalGroupRule:
    """One SMARTS-based functional-group tagging rule."""

    tag: str
    smarts: str = ""
    rdkit_fragment: str = ""
    min_patch_rate: float = 0.03
    confidence: float = 0.8
    provenance: str = "functional_smarts_tagger"


@dataclass(frozen=True)
class SmartsRxRule:
    """One SMARTS-RX rule describing a reactivity function."""

    tag: str
    smarts: str
    role: str = ""
    min_patch_rate: float = 0.02
    confidence: float = 0.82
    provenance: str = "smarts_rx_tagger"


@dataclass(frozen=True)
class SemanticTaggingResult:
    """Semantic tags and auto label for one concept."""

    concept_id: str
    label_auto: str
    tags: list[TagAssignment]
    evidence_json: dict[str, Any]


class SemanticTagger:
    """Compute semantic descriptors and map them to chemist-readable tags."""

    _GEOM_FAMILY_TOKENS: Mapping[str, tuple[str, ...]] = {
        "distance": ("dist", "distance", "bondlen", "bond_length", "pair_dist", "nearest"),
        "angle": ("angle", "bend", "bond_angle"),
        "dihedral": ("dihedral", "torsion", "phi", "psi", "chi"),
        "planarity": ("planar", "planarity", "out_of_plane", "oop", "flatness"),
        "shape": ("shape", "asphericity", "eccentricity", "spherocity", "globularity", "anisotropy"),
        "size": ("radius_gyration", "rg", "gyration", "span", "diameter"),
        "inertia": ("inertia", "principal", "pmi", "moment"),
        "surface_volume": ("surface", "sasa", "vsa", "psa", "volume"),
        "ring_strain": ("strain", "ring_strain", "angle_strain", "torsional_strain"),
        "hbond_geometry": ("hb", "hbond", "donor_acceptor", "d_a", "donnor", "acceptor"),
    }

    _QM_FAMILY_TOKENS: Mapping[str, tuple[str, ...]] = {
        "homo": ("homo", "ehomo", "eps_homo"),
        "lumo": ("lumo", "elumo", "eps_lumo"),
        "gap": ("gap", "homo_lumo", "bandgap", "deltae"),
        "dipole": ("dipole", "mu", "moment"),
        "polarizability": ("polariz", "alpha"),
        "hardness": ("hardness", "eta"),
        "softness": ("softness", "sigma_soft", "soft"),
        "electronegativity": ("electroneg", "chi"),
        "electrophilicity": ("electrophil", "omega"),
        "nucleophilicity": ("nucleophil",),
        "charge_transfer": ("charge_transfer", "ct", "deltaq", "charge_sep"),
        "esp": ("esp", "electrostatic", "mep"),
        "fukui": ("fukui",),
        "nbo": ("nbo",),
        "mulliken": ("mulliken",),
        "npa": ("npa",),
        "quadrupole": ("quadrupole",),
    }

    def __init__(self, config: SemanticTaggingConfig):
        self.config = config
        self.registry = self._load_naming_registry(config.naming_rules_path)
        self.functional_rules = self._load_functional_rules(config.functional_rules_path)
        self._functional_patterns = self._compile_functional_patterns(self.functional_rules)
        self._functional_fragment_functions = self._load_rdkit_fragment_functions(
            self.functional_rules
        )
        self.smarts_rx_rules = self._load_smarts_rx_rules(config.smarts_rx_rules_path)
        self._smarts_rx_patterns = self._compile_smarts_rx_patterns(self.smarts_rx_rules)
        self._openbabel_pybel = self._init_openbabel_backend()

    @staticmethod
    def _rules_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "rules"

    def _load_naming_registry(self, naming_rules_path: Optional[str]) -> NamingRegistry:
        if naming_rules_path:
            return NamingRegistry.from_json(naming_rules_path)
        return NamingRegistry.default()

    def _load_functional_rules(self, functional_rules_path: Optional[str]) -> tuple[FunctionalGroupRule, ...]:
        path = Path(functional_rules_path) if functional_rules_path else (self._rules_dir() / "default_functional_group_rules.json")
        if not path.exists():
            logger.warning("Functional-group rules file not found", extra={"path": str(path)})
            return ()

        payload = json.loads(path.read_text())
        rows = payload.get("rules", [])
        out: list[FunctionalGroupRule] = []
        for row in rows:
            tag = str(row.get("tag", "")).strip()
            smarts = str(row.get("smarts", "")).strip()
            rdkit_fragment = str(row.get("rdkit_fragment", "")).strip()
            if not tag or (not smarts and not rdkit_fragment):
                continue
            out.append(
                FunctionalGroupRule(
                    tag=tag,
                    smarts=smarts,
                    rdkit_fragment=rdkit_fragment,
                    min_patch_rate=float(row.get("min_patch_rate", 0.03)),
                    confidence=float(row.get("confidence", 0.8)),
                    provenance=str(row.get("provenance", "functional_smarts_tagger")),
                )
            )
        return tuple(out)

    def _load_smarts_rx_rules(self, smarts_rx_rules_path: Optional[str]) -> tuple[SmartsRxRule, ...]:
        if not bool(self.config.use_smarts_rx):
            return ()
        if smarts_rx_rules_path:
            path = Path(smarts_rx_rules_path)
        else:
            # Prefer new SMARTS-RX registry path; keep legacy fallback for compatibility.
            preferred = self._rules_dir() / "smartsrx.json"
            legacy = self._rules_dir() / "default_smarts_rx_rules.json"
            path = preferred if preferred.exists() else legacy
        if not path.exists():
            logger.warning("SMARTS-RX rules file not found", extra={"path": str(path)})
            return ()

        payload = json.loads(path.read_text())
        rows: Sequence[Any] = ()
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping):
            maybe_rules = payload.get("rules")
            maybe_data = payload.get("data")
            if isinstance(maybe_rules, Sequence):
                rows = maybe_rules
            elif isinstance(maybe_data, Sequence):
                rows = maybe_data
        out: list[SmartsRxRule] = []
        seen_tags: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            smarts = str(row.get("smarts", "")).strip()
            tag = str(row.get("tag", "")).strip()
            if not tag:
                for k in ("specific_type", "subcategory", "category"):
                    v = str(row.get(k, "")).strip()
                    if v:
                        tag = f"rx_{self._normalize_descriptor_name(v)}"
                        break
            if not tag or not smarts:
                continue
            role = str(row.get("role", "")).strip()
            if not role:
                cat = str(row.get("category", "")).strip()
                if cat:
                    role = self._normalize_descriptor_name(cat)
            # Keep unique tags even when external registry has repeated labels.
            if tag in seen_tags:
                seen_tags[tag] += 1
                tag = f"{tag}_{seen_tags[tag]}"
            else:
                seen_tags[tag] = 1
            out.append(
                SmartsRxRule(
                    tag=tag,
                    smarts=smarts,
                    role=role,
                    min_patch_rate=float(row.get("min_patch_rate", 0.02)),
                    confidence=float(row.get("confidence", 0.82)),
                    provenance=str(row.get("provenance", "smarts_rx_tagger")),
                )
            )
        return tuple(out)

    def _compile_functional_patterns(self, rules: Sequence[FunctionalGroupRule]) -> tuple[tuple[FunctionalGroupRule, Any], ...]:
        if not rules:
            return ()
        try:
            require_rdkit()
            from rdkit import Chem
        except OptionalDependencyError:
            return ()

        out: list[tuple[FunctionalGroupRule, Any]] = []
        for rule in rules:
            if not str(rule.smarts).strip():
                continue
            try:
                pattern = Chem.MolFromSmarts(rule.smarts)
                if pattern is None:
                    logger.warning("Invalid SMARTS rule skipped", extra={"tag": rule.tag, "smarts": rule.smarts})
                    continue
                out.append((rule, pattern))
            except Exception:
                logger.exception(
                    "Failed to compile SMARTS rule",
                    extra={"tag": rule.tag, "smarts": rule.smarts},
                )
        return tuple(out)

    def _load_rdkit_fragment_functions(
        self,
        rules: Sequence[FunctionalGroupRule],
    ) -> Mapping[str, Any]:
        frag_names = sorted(
            {str(r.rdkit_fragment).strip() for r in rules if str(r.rdkit_fragment).strip()}
        )
        if not frag_names:
            return {}
        try:
            require_rdkit()
            from rdkit.Chem import Fragments
        except OptionalDependencyError:
            logger.info("RDKit fragments backend unavailable; rdkit_fragment rules disabled")
            return {}

        out: dict[str, Any] = {}
        for name in frag_names:
            fn = getattr(Fragments, name, None)
            if callable(fn):
                out[name] = fn
            else:
                logger.warning(
                    "Unknown RDKit fragment function in rule",
                    extra={"rdkit_fragment": name},
                )
        return out

    def _compile_smarts_rx_patterns(self, rules: Sequence[SmartsRxRule]) -> tuple[tuple[SmartsRxRule, Any], ...]:
        if not rules:
            return ()
        try:
            require_rdkit()
            from rdkit import Chem
        except OptionalDependencyError:
            return ()

        out: list[tuple[SmartsRxRule, Any]] = []
        for rule in rules:
            try:
                pattern = Chem.MolFromSmarts(rule.smarts)
                if pattern is None:
                    logger.warning("Invalid SMARTS-RX rule skipped", extra={"tag": rule.tag, "smarts": rule.smarts})
                    continue
                out.append((rule, pattern))
            except Exception:
                logger.exception(
                    "Failed to compile SMARTS-RX rule",
                    extra={"tag": rule.tag, "smarts": rule.smarts},
                )
        return tuple(out)

    def _init_openbabel_backend(self) -> Any:
        if not bool(self.config.use_openbabel_descriptors):
            return None
        try:
            return require_openbabel_pybel()
        except OptionalDependencyError:
            logger.info("Open Babel pybel is unavailable; skipping Open Babel semantic descriptors")
            return None

    def tag_concept(
        self,
        *,
        concept_id: str,
        concept_patches: Sequence[PatchRecord],
        molecules_by_id: Mapping[str, Any],
        inst_by_pair: Optional[Mapping[tuple[str, str], np.ndarray]] = None,
        inst_mean_by_id: Optional[Mapping[str, np.ndarray]] = None,
        inst_geom_dim: int = 0,
        inst_qm_dim: int = 0,
        geom_feature_names: Optional[Sequence[str]] = None,
        qm_feature_names: Optional[Sequence[str]] = None,
    ) -> SemanticTaggingResult:
        descriptors = self._compute_descriptors(
            concept_patches=concept_patches,
            molecules_by_id=molecules_by_id,
            inst_by_pair=inst_by_pair,
            inst_mean_by_id=inst_mean_by_id,
            inst_geom_dim=int(inst_geom_dim),
            inst_qm_dim=int(inst_qm_dim),
            geom_feature_names=geom_feature_names,
            qm_feature_names=qm_feature_names,
        )
        tags: list[TagAssignment] = []
        tags.extend(self._functional_group_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._smarts_rx_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._charge_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._conjugation_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._geometry_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._pharmacophore_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._qm_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._openbabel_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._cross_modal_tags(concept_id=concept_id, descriptors=descriptors))
        tags = self._deduplicate_tags(tags)

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
        inst_by_pair: Optional[Mapping[tuple[str, str], np.ndarray]],
        inst_mean_by_id: Optional[Mapping[str, np.ndarray]],
        inst_geom_dim: int,
        inst_qm_dim: int,
        geom_feature_names: Optional[Sequence[str]],
        qm_feature_names: Optional[Sequence[str]],
    ) -> dict[str, Any]:
        try:
            require_rdkit()
            from rdkit.Chem import rdPartialCharges
            from rdkit.Chem import ChemicalFeatures
            from rdkit import RDConfig
            from rdkit import Chem
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
        geom_vectors: list[np.ndarray] = []
        qm_vectors: list[np.ndarray] = []

        pharm_counts = {
            "Donor": 0,
            "Acceptor": 0,
            "Aromatic": 0,
            "PosIonizable": 0,
            "NegIonizable": 0,
        }

        functional_patch_presence: Dict[str, int] = {
            str(rule.tag): 0 for rule in self.functional_rules
        }
        functional_match_hits: Dict[str, int] = {
            str(rule.tag): 0 for rule in self.functional_rules
        }
        smarts_rx_patch_presence: Dict[str, int] = {
            rule.tag: 0 for rule, _ in self._smarts_rx_patterns
        }
        smarts_rx_match_hits: Dict[str, int] = {
            rule.tag: 0 for rule, _ in self._smarts_rx_patterns
        }
        smarts_rx_role_presence: Dict[str, int] = {}

        fdef = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
        factory = None
        if fdef.exists():
            try:
                factory = ChemicalFeatures.BuildFeatureFactory(str(fdef))
            except Exception:
                factory = None

        gasteiger_cache: dict[str, np.ndarray] = {}
        pharm_cache: dict[str, Sequence[Any]] = {}
        functional_cache: dict[str, tuple[tuple[set[int], ...], ...]] = {}
        functional_fragment_cache: dict[str, dict[str, float]] = {}
        smarts_rx_cache: dict[str, tuple[tuple[set[int], ...], ...]] = {}
        mol_seen: dict[str, Any] = {}

        geom_names = self._normalize_geom_feature_names(
            geom_feature_names=geom_feature_names,
            inst_geom_dim=int(inst_geom_dim),
        )
        qm_names = self._normalize_qm_feature_names(
            qm_feature_names=qm_feature_names,
            inst_qm_dim=int(inst_qm_dim),
        )

        valid_patch_count = 0
        for patch in concept_patches:
            mol = molecules_by_id.get(patch.mol_id)
            if mol is None:
                continue
            mol_id = str(patch.mol_id)
            mol_seen[mol_id] = mol

            atom_ids = [int(i) for i in patch.atom_indices if 0 <= int(i) < int(mol.GetNumAtoms())]
            if len(atom_ids) == 0:
                continue
            valid_patch_count += 1

            fsum = float(sum(int(mol.GetAtomWithIdx(i).GetFormalCharge()) for i in atom_ids))
            formal_charge_sums.append(fsum)

            charges = gasteiger_cache.get(mol_id)
            if charges is None:
                try:
                    mol_copy = Chem.Mol(mol)
                    rdPartialCharges.ComputeGasteigerCharges(mol_copy)
                    arr = np.full((int(mol_copy.GetNumAtoms()),), np.nan, dtype=np.float64)
                    for i in range(int(mol_copy.GetNumAtoms())):
                        atom = mol_copy.GetAtomWithIdx(i)
                        if atom.HasProp("_GasteigerCharge"):
                            v = atom.GetDoubleProp("_GasteigerCharge")
                            if np.isfinite(v):
                                arr[i] = float(v)
                    charges = arr
                except Exception:
                    charges = np.full((int(mol.GetNumAtoms()),), np.nan, dtype=np.float64)
                gasteiger_cache[mol_id] = charges
            vals = [float(charges[i]) for i in atom_ids if 0 <= i < len(charges) and np.isfinite(charges[i])]
            gasteiger_vals.extend(vals)

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
                    coords = np.asarray(
                        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in atom_ids],
                        dtype=np.float64,
                    )
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
                feats = pharm_cache.get(mol_id)
                if feats is None:
                    try:
                        feats = factory.GetFeaturesForMol(mol)
                    except Exception:
                        feats = ()
                    pharm_cache[mol_id] = feats
                for feat in feats:
                    try:
                        atom_ids_feat = set(int(i) for i in feat.GetAtomIds())
                    except Exception:
                        continue
                    if atom_ids_feat.intersection(atom_set):
                        fam = str(feat.GetFamily())
                        if fam in pharm_counts:
                            pharm_counts[fam] += 1

            if self._functional_patterns:
                matches = functional_cache.get(mol_id)
                if matches is None:
                    matches_list: list[tuple[set[int], ...]] = []
                    for _, pattern in self._functional_patterns:
                        try:
                            sub_matches = mol.GetSubstructMatches(pattern, uniquify=True)
                        except Exception:
                            sub_matches = ()
                        match_sets = tuple(set(int(i) for i in m) for m in sub_matches)
                        matches_list.append(match_sets)
                    matches = tuple(matches_list)
                    functional_cache[mol_id] = matches

                for idx, (rule, _) in enumerate(self._functional_patterns):
                    overlap_hits = 0
                    for match_atoms in matches[idx]:
                        if match_atoms.intersection(atom_set):
                            overlap_hits += 1
                    if overlap_hits > 0:
                        functional_patch_presence[rule.tag] = int(functional_patch_presence.get(rule.tag, 0) + 1)
                        functional_match_hits[rule.tag] = int(functional_match_hits.get(rule.tag, 0) + overlap_hits)

            if self._functional_fragment_functions:
                frag_counts = functional_fragment_cache.get(mol_id)
                if frag_counts is None:
                    frag_counts = {}
                    for rule in self.functional_rules:
                        fn_name = str(rule.rdkit_fragment).strip()
                        if not fn_name:
                            continue
                        fn = self._functional_fragment_functions.get(fn_name)
                        if fn is None:
                            continue
                        try:
                            v = float(fn(mol))
                        except Exception:
                            v = 0.0
                        if np.isfinite(v) and v > 0.0:
                            frag_counts[str(rule.tag)] = float(v)
                    functional_fragment_cache[mol_id] = frag_counts

                if frag_counts:
                    for tag, count_v in frag_counts.items():
                        functional_patch_presence[str(tag)] = int(
                            functional_patch_presence.get(str(tag), 0) + 1
                        )
                        functional_match_hits[str(tag)] = int(
                            functional_match_hits.get(str(tag), 0)
                            + max(1, int(round(float(count_v))))
                        )

            if self._smarts_rx_patterns:
                matches_rx = smarts_rx_cache.get(mol_id)
                if matches_rx is None:
                    matches_list_rx: list[tuple[set[int], ...]] = []
                    for _, pattern in self._smarts_rx_patterns:
                        try:
                            sub_matches_rx = mol.GetSubstructMatches(pattern, uniquify=True)
                        except Exception:
                            sub_matches_rx = ()
                        match_sets_rx = tuple(set(int(i) for i in m) for m in sub_matches_rx)
                        matches_list_rx.append(match_sets_rx)
                    matches_rx = tuple(matches_list_rx)
                    smarts_rx_cache[mol_id] = matches_rx

                for idx, (rule, _) in enumerate(self._smarts_rx_patterns):
                    overlap_hits_rx = 0
                    for match_atoms_rx in matches_rx[idx]:
                        if match_atoms_rx.intersection(atom_set):
                            overlap_hits_rx += 1
                    if overlap_hits_rx > 0:
                        smarts_rx_patch_presence[rule.tag] = int(smarts_rx_patch_presence.get(rule.tag, 0) + 1)
                        smarts_rx_match_hits[rule.tag] = int(smarts_rx_match_hits.get(rule.tag, 0) + overlap_hits_rx)
                        if rule.role:
                            role_key = str(rule.role)
                            smarts_rx_role_presence[role_key] = int(smarts_rx_role_presence.get(role_key, 0) + 1)

            geom_vec, qm_vec = self._resolve_geom_qm_vectors_for_patch(
                patch=patch,
                inst_by_pair=inst_by_pair,
                inst_mean_by_id=inst_mean_by_id,
                inst_geom_dim=int(inst_geom_dim),
                inst_qm_dim=int(inst_qm_dim),
            )
            if geom_vec is not None and geom_vec.size > 0:
                geom_vectors.append(geom_vec.astype(np.float32, copy=False))
            if qm_vec is not None and qm_vec.size > 0:
                qm_vectors.append(qm_vec.astype(np.float32, copy=False))

        functional_patch_rate = {
            str(tag): float(cnt) / float(max(1, valid_patch_count))
            for tag, cnt in functional_patch_presence.items()
            if int(cnt) > 0
        }
        smarts_rx_patch_rate = {
            str(tag): float(cnt) / float(max(1, valid_patch_count))
            for tag, cnt in smarts_rx_patch_presence.items()
            if int(cnt) > 0
        }
        smarts_rx_role_rate = {
            str(role): float(cnt) / float(max(1, valid_patch_count))
            for role, cnt in smarts_rx_role_presence.items()
            if int(cnt) > 0
        }

        geom_summary = self._summarize_geom_vectors(
            geom_vectors=geom_vectors,
            geom_feature_names=geom_names,
        )
        qm_summary = self._summarize_qm_vectors(
            qm_vectors=qm_vectors,
            qm_feature_names=qm_names,
        )

        openbabel_summary = self._summarize_openbabel_descriptors(molecules_by_id=mol_seen)

        return {
            "n_patches": int(len(concept_patches)),
            "n_valid_patches": int(valid_patch_count),
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
            "functional_group_patch_rate": functional_patch_rate,
            "functional_group_match_hits": {
                str(tag): int(cnt)
                for tag, cnt in functional_match_hits.items()
                if int(cnt) > 0
            },
            "smarts_rx_patch_rate": smarts_rx_patch_rate,
            "smarts_rx_match_hits": {
                str(tag): int(cnt)
                for tag, cnt in smarts_rx_match_hits.items()
                if int(cnt) > 0
            },
            "smarts_rx_role_rate": smarts_rx_role_rate,
            "geom_summary": geom_summary,
            "qm_summary": qm_summary,
            "openbabel_descriptor_summary": openbabel_summary,
        }

    def _summarize_geom_vectors(
        self,
        *,
        geom_vectors: Sequence[np.ndarray],
        geom_feature_names: Sequence[str],
    ) -> dict[str, Any]:
        return self._summarize_vector_block(
            vectors=geom_vectors,
            feature_names=geom_feature_names,
            fallback_prefix="geom",
            family_tokens=self._GEOM_FAMILY_TOKENS,
        )

    def _summarize_qm_vectors(
        self,
        *,
        qm_vectors: Sequence[np.ndarray],
        qm_feature_names: Sequence[str],
    ) -> dict[str, Any]:
        return self._summarize_vector_block(
            vectors=qm_vectors,
            feature_names=qm_feature_names,
            fallback_prefix="qm",
            family_tokens=self._QM_FAMILY_TOKENS,
        )

    def _summarize_vector_block(
        self,
        *,
        vectors: Sequence[np.ndarray],
        feature_names: Sequence[str],
        fallback_prefix: str,
        family_tokens: Mapping[str, tuple[str, ...]],
    ) -> dict[str, Any]:
        n_vectors = int(len(vectors))
        if n_vectors <= 0:
            return {
                "n_vectors": 0,
                "n_features": 0,
                "top_features": [],
                "family_stats": {},
            }

        dim = min(int(v.shape[0]) for v in vectors if np.asarray(v).ndim == 1)
        if dim <= 0:
            return {
                "n_vectors": 0,
                "n_features": 0,
                "top_features": [],
                "family_stats": {},
            }

        mat = np.vstack([np.asarray(v, dtype=np.float32)[:dim] for v in vectors])
        feat_names = list(feature_names)
        if len(feat_names) < dim:
            feat_names = feat_names + [f"{fallback_prefix}_{i}" for i in range(len(feat_names), dim)]
        elif len(feat_names) > dim:
            feat_names = feat_names[:dim]

        abs_mean = np.mean(np.abs(mat), axis=0)
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)

        top_k = int(min(12, dim))
        top_idx = np.argsort(abs_mean)[-top_k:][::-1]
        top_features = [
            {
                "name": str(feat_names[int(i)]),
                "abs_z_mean": float(abs_mean[int(i)]),
                "z_mean": float(mean[int(i)]),
                "z_std": float(std[int(i)]),
            }
            for i in top_idx
        ]

        family_index = self._build_family_index(
            feature_names=feat_names,
            family_tokens=family_tokens,
        )
        family_stats: dict[str, dict[str, float]] = {}
        for family, idxs in family_index.items():
            if len(idxs) == 0:
                continue
            vals = mat[:, idxs]
            family_stats[str(family)] = {
                "n_features": float(len(idxs)),
                "abs_z_mean": float(np.mean(np.abs(vals))),
                "z_mean": float(np.mean(vals)),
                "z_std": float(np.std(vals)),
            }

        return {
            "n_vectors": int(n_vectors),
            "n_features": int(dim),
            "top_features": top_features,
            "family_stats": family_stats,
        }

    def _summarize_openbabel_descriptors(self, *, molecules_by_id: Mapping[str, Any]) -> dict[str, Any]:
        if self._openbabel_pybel is None:
            return {"enabled": False, "n_molecules": 0}
        try:
            require_rdkit()
            from rdkit import Chem
        except OptionalDependencyError:
            return {"enabled": False, "n_molecules": 0}

        max_eval = 512
        keys = sorted(str(k) for k in molecules_by_id.keys())[:max_eval]
        if not keys:
            return {"enabled": True, "n_molecules": 0}

        logp_vals: list[float] = []
        tpsa_vals: list[float] = []
        mr_vals: list[float] = []

        for mol_id in keys:
            mol = molecules_by_id.get(mol_id)
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
            except Exception:
                continue
            if not smi:
                continue
            try:
                obmol = self._openbabel_pybel.readstring("smi", smi)
                desc = obmol.calcdesc()
            except Exception:
                continue

            if "logP" in desc and np.isfinite(desc["logP"]):
                logp_vals.append(float(desc["logP"]))
            if "TPSA" in desc and np.isfinite(desc["TPSA"]):
                tpsa_vals.append(float(desc["TPSA"]))
            if "MR" in desc and np.isfinite(desc["MR"]):
                mr_vals.append(float(desc["MR"]))

        return {
            "enabled": True,
            "n_molecules": int(len(keys)),
            "logP_mean": float(np.mean(logp_vals)) if logp_vals else float("nan"),
            "TPSA_mean": float(np.mean(tpsa_vals)) if tpsa_vals else float("nan"),
            "MR_mean": float(np.mean(mr_vals)) if mr_vals else float("nan"),
        }

    def _functional_group_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        rates = descriptors.get("functional_group_patch_rate", {})
        if not isinstance(rates, Mapping):
            return []

        tags: list[TagAssignment] = []
        for rule in self.functional_rules:
            rate = float(rates.get(rule.tag, 0.0))
            if rate < float(rule.min_patch_rate):
                continue
            conf = max(float(rule.confidence), min(0.99, 0.55 + 0.9 * rate))
            tags.append(self._mk_tag(concept_id, rule.tag, conf, rule.provenance, descriptors))
        return tags

    def _smarts_rx_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        rates = descriptors.get("smarts_rx_patch_rate", {})
        if not isinstance(rates, Mapping):
            return []

        role_rates = descriptors.get("smarts_rx_role_rate", {})
        if not isinstance(role_rates, Mapping):
            role_rates = {}

        tags: list[TagAssignment] = []
        for rule in self.smarts_rx_rules:
            rate = float(rates.get(rule.tag, 0.0))
            if rate < float(rule.min_patch_rate):
                continue
            conf = max(float(rule.confidence), min(0.99, 0.60 + 0.9 * rate))
            tags.append(self._mk_tag(concept_id, str(rule.tag), conf, rule.provenance, descriptors))

        for role, rate in role_rates.items():
            rr = float(rate)
            if rr < 0.02:
                continue
            role_tag = str(role)
            if not role_tag.startswith("rx_role_"):
                role_tag = f"rx_role_{role_tag}"
            tags.append(
                self._mk_tag(
                    concept_id,
                    role_tag,
                    min(0.99, 0.55 + 0.8 * rr),
                    "smarts_rx_role_tagger",
                    descriptors,
                )
            )
        return tags

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

        gms = descriptors.get("geom_summary", {})
        if not isinstance(gms, Mapping):
            return tags
        n_vectors = int(gms.get("n_vectors", 0))
        if n_vectors < int(self.config.geom_min_vectors_for_tagging):
            return tags

        family_stats = gms.get("family_stats", {})
        if not isinstance(family_stats, Mapping):
            family_stats = {}

        thr = float(self.config.geom_z_threshold)
        strong_thr = float(self.config.geom_strong_z_threshold)

        def fam(name: str) -> Mapping[str, float]:
            item = family_stats.get(name, {})
            if isinstance(item, Mapping):
                return item
            return {}

        torsion_abs = float(fam("dihedral").get("abs_z_mean", 0.0))
        torsion_mean = float(fam("dihedral").get("z_mean", 0.0))
        if torsion_abs >= thr and torsion_mean >= 0.0:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "torsionally active geometry",
                    min(1.0, 0.55 + 0.22 * torsion_abs),
                    "geometry_descriptor_tagger",
                    descriptors,
                )
            )

        planar_abs = float(fam("planarity").get("abs_z_mean", 0.0))
        if planar_abs >= strong_thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "planarity-enriched geometry",
                    min(1.0, 0.55 + 0.20 * planar_abs),
                    "geometry_descriptor_tagger",
                    descriptors,
                )
            )

        shape_abs = float(fam("shape").get("abs_z_mean", 0.0))
        shape_mean = float(fam("shape").get("z_mean", 0.0))
        if shape_abs >= thr:
            if shape_mean >= 0.0:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "shape-anisotropic geometry",
                        min(1.0, 0.55 + 0.20 * shape_abs),
                        "geometry_descriptor_tagger",
                        descriptors,
                    )
                )
            else:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "shape-compact geometry",
                        min(1.0, 0.55 + 0.20 * shape_abs),
                        "geometry_descriptor_tagger",
                        descriptors,
                    )
                )

        ring_strain_abs = float(fam("ring_strain").get("abs_z_mean", 0.0))
        if ring_strain_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "ring-strained geometry",
                    min(1.0, 0.55 + 0.20 * ring_strain_abs),
                    "geometry_descriptor_tagger",
                    descriptors,
                )
            )

        surface_abs = float(fam("surface_volume").get("abs_z_mean", 0.0))
        if surface_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "surface/volume-driven geometry",
                    min(1.0, 0.55 + 0.18 * surface_abs),
                    "geometry_descriptor_tagger",
                    descriptors,
                )
            )

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

    def _qm_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        qms = descriptors.get("qm_summary", {})
        if not isinstance(qms, Mapping):
            return []
        n_vectors = int(qms.get("n_vectors", 0))
        if n_vectors < int(self.config.qm_min_vectors_for_tagging):
            return []

        family_stats = qms.get("family_stats", {})
        if not isinstance(family_stats, Mapping):
            family_stats = {}

        thr = float(self.config.qm_z_threshold)
        strong_thr = float(self.config.qm_strong_z_threshold)

        def fam(name: str) -> Mapping[str, float]:
            item = family_stats.get(name, {})
            if isinstance(item, Mapping):
                return item
            return {}

        tags: list[TagAssignment] = []

        gap_abs = float(fam("gap").get("abs_z_mean", 0.0))
        gap_mean = float(fam("gap").get("z_mean", 0.0))
        if gap_abs >= thr:
            if gap_mean >= 0.0:
                tags.append(self._mk_tag(concept_id, "large HOMO-LUMO gap", min(1.0, 0.55 + 0.25 * gap_abs), "qm_descriptor_tagger", descriptors))
            else:
                tags.append(self._mk_tag(concept_id, "small HOMO-LUMO gap", min(1.0, 0.55 + 0.25 * gap_abs), "qm_descriptor_tagger", descriptors))

        homo_abs = float(fam("homo").get("abs_z_mean", 0.0))
        homo_mean = float(fam("homo").get("z_mean", 0.0))
        if homo_abs >= strong_thr and homo_mean >= thr:
            tags.append(self._mk_tag(concept_id, "electron-rich frontier (high HOMO)", min(1.0, 0.55 + 0.2 * homo_abs), "qm_descriptor_tagger", descriptors))

        lumo_abs = float(fam("lumo").get("abs_z_mean", 0.0))
        lumo_mean = float(fam("lumo").get("z_mean", 0.0))
        if lumo_abs >= strong_thr and lumo_mean <= -thr:
            tags.append(self._mk_tag(concept_id, "electron-poor frontier (low LUMO)", min(1.0, 0.55 + 0.2 * lumo_abs), "qm_descriptor_tagger", descriptors))

        dip_abs = float(fam("dipole").get("abs_z_mean", 0.0))
        if dip_abs >= thr:
            tags.append(self._mk_tag(concept_id, "high dipole moment", min(1.0, 0.55 + 0.2 * dip_abs), "qm_descriptor_tagger", descriptors))

        pol_abs = float(fam("polarizability").get("abs_z_mean", 0.0))
        if pol_abs >= thr:
            tags.append(self._mk_tag(concept_id, "high polarizability", min(1.0, 0.55 + 0.2 * pol_abs), "qm_descriptor_tagger", descriptors))

        hard_abs = float(fam("hardness").get("abs_z_mean", 0.0))
        hard_mean = float(fam("hardness").get("z_mean", 0.0))
        soft_abs = float(fam("softness").get("abs_z_mean", 0.0))
        soft_mean = float(fam("softness").get("z_mean", 0.0))
        if hard_abs >= thr and hard_mean >= 0.0:
            tags.append(self._mk_tag(concept_id, "hard electronic profile", min(1.0, 0.55 + 0.2 * hard_abs), "qm_descriptor_tagger", descriptors))
        if soft_abs >= thr and soft_mean >= 0.0:
            tags.append(self._mk_tag(concept_id, "soft electronic profile", min(1.0, 0.55 + 0.2 * soft_abs), "qm_descriptor_tagger", descriptors))

        elec_abs = float(fam("electrophilicity").get("abs_z_mean", 0.0))
        elec_mean = float(fam("electrophilicity").get("z_mean", 0.0))
        if elec_abs >= thr and elec_mean >= 0.0:
            tags.append(self._mk_tag(concept_id, "electrophile-like electronic profile", min(1.0, 0.55 + 0.2 * elec_abs), "qm_descriptor_tagger", descriptors))

        nuc_abs = float(fam("nucleophilicity").get("abs_z_mean", 0.0))
        nuc_mean = float(fam("nucleophilicity").get("z_mean", 0.0))
        if nuc_abs >= thr and nuc_mean >= 0.0:
            tags.append(self._mk_tag(concept_id, "nucleophile-like electronic profile", min(1.0, 0.55 + 0.2 * nuc_abs), "qm_descriptor_tagger", descriptors))

        ct_abs = float(fam("charge_transfer").get("abs_z_mean", 0.0))
        if ct_abs >= thr:
            tags.append(self._mk_tag(concept_id, "charge-transfer-prone profile", min(1.0, 0.55 + 0.2 * ct_abs), "qm_descriptor_tagger", descriptors))

        fuk_abs = float(fam("fukui").get("abs_z_mean", 0.0))
        if fuk_abs >= strong_thr:
            tags.append(self._mk_tag(concept_id, "reactive frontier-density profile", min(1.0, 0.55 + 0.2 * fuk_abs), "qm_descriptor_tagger", descriptors))

        esp_abs = float(fam("esp").get("abs_z_mean", 0.0))
        if esp_abs >= strong_thr:
            tags.append(self._mk_tag(concept_id, "electrostatic potential contrast", min(1.0, 0.55 + 0.2 * esp_abs), "qm_descriptor_tagger", descriptors))

        return tags

    def _openbabel_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        ob = descriptors.get("openbabel_descriptor_summary", {})
        if not isinstance(ob, Mapping):
            return []
        if not bool(ob.get("enabled", False)):
            return []
        if int(ob.get("n_molecules", 0)) <= 0:
            return []

        tags: list[TagAssignment] = []
        logp = float(ob.get("logP_mean", float("nan")))
        tpsa = float(ob.get("TPSA_mean", float("nan")))
        mr = float(ob.get("MR_mean", float("nan")))

        if np.isfinite(logp) and logp >= 2.0:
            tags.append(self._mk_tag(concept_id, "lipophilic", min(1.0, 0.55 + 0.08 * logp), "openbabel_descriptor_tagger", descriptors))
        if np.isfinite(tpsa) and tpsa >= 75.0:
            tags.append(self._mk_tag(concept_id, "high polar surface area", min(1.0, 0.55 + 0.004 * tpsa), "openbabel_descriptor_tagger", descriptors))
        if np.isfinite(mr) and mr >= 70.0:
            tags.append(self._mk_tag(concept_id, "high refractivity", min(1.0, 0.55 + 0.003 * mr), "openbabel_descriptor_tagger", descriptors))

        return tags

    def _cross_modal_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        tags: list[TagAssignment] = []

        rx_roles = descriptors.get("smarts_rx_role_rate", {})
        if not isinstance(rx_roles, Mapping):
            rx_roles = {}
        qms = descriptors.get("qm_summary", {})
        if not isinstance(qms, Mapping):
            qms = {}
        gms = descriptors.get("geom_summary", {})
        if not isinstance(gms, Mapping):
            gms = {}

        qfam = qms.get("family_stats", {})
        if not isinstance(qfam, Mapping):
            qfam = {}
        gfam = gms.get("family_stats", {})
        if not isinstance(gfam, Mapping):
            gfam = {}

        def qfam_stat(name: str) -> Mapping[str, Any]:
            item = qfam.get(name, {})
            if isinstance(item, Mapping):
                return item
            return {}

        def gfam_stat(name: str) -> Mapping[str, Any]:
            item = gfam.get(name, {})
            if isinstance(item, Mapping):
                return item
            return {}

        thr_q = float(self.config.qm_z_threshold)
        thr_g = float(self.config.geom_z_threshold)

        rx_electrophile = float(rx_roles.get("electrophile", 0.0))
        electrophile_qm = float(qfam_stat("electrophilicity").get("abs_z_mean", 0.0))
        if rx_electrophile >= 0.02 and electrophile_qm >= thr_q:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "electrophilic reaction-center motif",
                    min(1.0, 0.58 + 0.16 * (rx_electrophile + electrophile_qm)),
                    "cross_modal_tagger",
                    descriptors,
                )
            )

        rx_nucleophile = float(rx_roles.get("nucleophile", 0.0))
        nucleophile_qm = float(qfam_stat("nucleophilicity").get("abs_z_mean", 0.0))
        if rx_nucleophile >= 0.02 and nucleophile_qm >= thr_q:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "nucleophilic reaction-center motif",
                    min(1.0, 0.58 + 0.16 * (rx_nucleophile + nucleophile_qm)),
                    "cross_modal_tagger",
                    descriptors,
                )
            )

        aromatic = float(descriptors.get("aromatic_atom_fraction_mean", 0.0))
        planar = float(gfam_stat("planarity").get("abs_z_mean", 0.0))
        gap = float(qfam_stat("gap").get("abs_z_mean", 0.0))
        if aromatic >= float(self.config.aromatic_fraction_threshold) and planar >= thr_g and gap >= thr_q:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "planar conjugated electronic motif",
                    min(1.0, 0.55 + 0.14 * (aromatic + planar + gap)),
                    "cross_modal_tagger",
                    descriptors,
                )
            )

        hbd = int((descriptors.get("pharmacophore_counts", {}) or {}).get("Donor", 0))
        hba = int((descriptors.get("pharmacophore_counts", {}) or {}).get("Acceptor", 0))
        dip = float(qfam_stat("dipole").get("abs_z_mean", 0.0))
        if hbd > 0 and hba > 0 and dip >= thr_q:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "polar donor-acceptor electronic motif",
                    min(1.0, 0.58 + 0.18 * dip),
                    "cross_modal_tagger",
                    descriptors,
                )
            )

        return tags

    @staticmethod
    def _deduplicate_tags(tags: Sequence[TagAssignment]) -> list[TagAssignment]:
        best: dict[str, TagAssignment] = {}
        for tag in tags:
            prev = best.get(tag.tag)
            if prev is None or float(tag.confidence) > float(prev.confidence):
                best[tag.tag] = tag
        return list(best.values())

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

    def _normalize_geom_feature_names(
        self,
        *,
        geom_feature_names: Optional[Sequence[str]],
        inst_geom_dim: int,
    ) -> list[str]:
        names = [str(x) for x in (geom_feature_names or ())]
        if int(inst_geom_dim) > 0:
            if len(names) > int(inst_geom_dim):
                names = names[: int(inst_geom_dim)]
            elif len(names) < int(inst_geom_dim):
                names = names + [f"geom_{i}" for i in range(len(names), int(inst_geom_dim))]
        return names

    def _normalize_qm_feature_names(self, *, qm_feature_names: Optional[Sequence[str]], inst_qm_dim: int) -> list[str]:
        names = [str(x) for x in (qm_feature_names or ())]
        if int(inst_qm_dim) > 0:
            if len(names) > int(inst_qm_dim):
                names = names[: int(inst_qm_dim)]
            elif len(names) < int(inst_qm_dim):
                names = names + [f"qm_{i}" for i in range(len(names), int(inst_qm_dim))]
        return names

    @staticmethod
    def _normalize_descriptor_name(name: str) -> str:
        s = str(name).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    def _build_family_index(
        self,
        *,
        feature_names: Sequence[str],
        family_tokens: Mapping[str, tuple[str, ...]],
    ) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {str(k): [] for k in family_tokens.keys()}
        normalized = [self._normalize_descriptor_name(n) for n in feature_names]
        for idx, name in enumerate(normalized):
            for family, tokens in family_tokens.items():
                if any(tok in name for tok in tokens):
                    out[str(family)].append(int(idx))
        return out

    def _build_qm_family_index(self, *, feature_names: Sequence[str]) -> dict[str, list[int]]:
        return self._build_family_index(
            feature_names=feature_names,
            family_tokens=self._QM_FAMILY_TOKENS,
        )

    @staticmethod
    def _resolve_geom_qm_vectors_for_patch(
        *,
        patch: PatchRecord,
        inst_by_pair: Optional[Mapping[tuple[str, str], np.ndarray]],
        inst_mean_by_id: Optional[Mapping[str, np.ndarray]],
        inst_geom_dim: int,
        inst_qm_dim: int,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        mol_id = str(patch.mol_id)

        vec = None
        if patch.conf_id is not None and inst_by_pair is not None:
            vec = inst_by_pair.get((mol_id, str(patch.conf_id)))
        if vec is None and inst_mean_by_id is not None:
            vec = inst_mean_by_id.get(mol_id)
        if vec is None:
            return None, None

        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            return None, None

        geom_out: Optional[np.ndarray] = None
        if int(inst_geom_dim) > 0:
            gdim = int(inst_geom_dim)
            if arr.size >= gdim:
                geom_out = arr[:gdim]
                if geom_out is not None and np.isfinite(geom_out).any():
                    geom_out = np.nan_to_num(geom_out, copy=False)
                else:
                    geom_out = None

        qm_out: Optional[np.ndarray] = None
        if int(inst_qm_dim) > 0:
            start = max(0, int(inst_geom_dim))
            stop = start + int(inst_qm_dim)
            if arr.size >= stop:
                qm_out = arr[start:stop]
            elif arr.size >= int(inst_qm_dim):
                qm_out = arr[: int(inst_qm_dim)]
            if qm_out is not None:
                if np.isfinite(qm_out).any():
                    qm_out = np.nan_to_num(qm_out, copy=False)
                else:
                    qm_out = None

        return geom_out, qm_out

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

    def _descriptor_rank(self, descriptors: Mapping[str, Any]) -> list[str]:
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

        fg_rates = descriptors.get("functional_group_patch_rate", {})
        if isinstance(fg_rates, Mapping):
            ordered = sorted(
                ((str(k), float(v)) for k, v in fg_rates.items()),
                key=lambda x: x[1],
                reverse=True,
            )
            out.extend([name for name, _ in ordered[:2]])

        rx_rates = descriptors.get("smarts_rx_patch_rate", {})
        if isinstance(rx_rates, Mapping):
            ordered_rx = sorted(
                ((str(k), float(v)) for k, v in rx_rates.items()),
                key=lambda x: x[1],
                reverse=True,
            )
            out.extend([name for name, _ in ordered_rx[:2]])

        qms = descriptors.get("qm_summary", {})
        if isinstance(qms, Mapping):
            fam = qms.get("family_stats", {})
            if isinstance(fam, Mapping):
                ordered_fam = sorted(
                    (
                        (str(k), float(v.get("abs_z_mean", 0.0)))
                        for k, v in fam.items()
                        if isinstance(v, Mapping)
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
                out.extend([name for name, score in ordered_fam[:2] if score >= float(self.config.qm_z_threshold)])

        gms = descriptors.get("geom_summary", {})
        if isinstance(gms, Mapping):
            gfam = gms.get("family_stats", {})
            if isinstance(gfam, Mapping):
                ordered_gfam = sorted(
                    (
                        (str(k), float(v.get("abs_z_mean", 0.0)))
                        for k, v in gfam.items()
                        if isinstance(v, Mapping)
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
                out.extend([name for name, score in ordered_gfam[:2] if score >= float(self.config.geom_z_threshold)])

        if not out:
            out.append("mixed")
        return out


__all__ = [
    "SemanticTagger",
    "SemanticTaggingResult",
    "FunctionalGroupRule",
    "SmartsRxRule",
]
