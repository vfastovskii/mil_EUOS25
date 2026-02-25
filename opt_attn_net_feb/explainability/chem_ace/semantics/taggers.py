from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import SemanticTaggingConfig
from ..optional_deps import (
    OptionalDependencyError,
    has_ripser,
    has_scipy,
    require_openbabel_pybel,
    require_ripser,
    require_rdkit,
    require_scipy,
)
from ..types import PatchRecord, TagAssignment
from .naming import NamingRegistry, choose_label
from ....utils.progress import log_event

logger = logging.getLogger(__name__)


_VDW_RADII: Mapping[int, float] = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}


@dataclass(frozen=True)
class AdvancedGeometryTopologyConfig:
    """Controls optional geometry/topology descriptor extraction for semantic tagging."""

    enabled: bool = True
    max_patches: int = 3000
    min_atoms: int = 4
    max_torsion_paths: int = 96
    use_convex_hull: bool = True
    use_persistent_homology: bool = True
    persistence_max_atoms: int = 48


def select_patch_ids_for_advanced(
    *,
    patch_ids: Sequence[str],
    max_patches: int,
) -> set[str]:
    """Deterministically downsample patch IDs for advanced geometry computations."""
    cap = int(max(0, max_patches))
    if cap <= 0 or len(patch_ids) <= cap:
        return {str(x) for x in patch_ids}

    ranked: list[tuple[str, str]] = []
    for pid in patch_ids:
        p = str(pid)
        h = sha1(f"chemace_adv_geom_v1|{p}".encode("utf-8")).hexdigest()
        ranked.append((h, p))
    ranked.sort(key=lambda x: x[0])
    return {p for _, p in ranked[:cap]}


def compute_patch_advanced_metrics(
    *,
    coords: np.ndarray,
    atom_numbers: Sequence[int],
    adjacency: Mapping[int, Sequence[int]],
    config: AdvancedGeometryTopologyConfig,
) -> dict[str, float]:
    """Compute advanced geometry/topology metrics for one patch."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return {}
    n_atoms = int(arr.shape[0])
    if n_atoms < int(max(3, config.min_atoms)):
        return {}

    centered = arr - np.mean(arr, axis=0, keepdims=True)
    sq_norm = np.sum(np.square(centered), axis=1)
    radius_gyration = float(np.sqrt(np.mean(sq_norm)))

    cov = np.matmul(centered.T, centered) / float(max(1, n_atoms))
    eig = np.linalg.eigvalsh(cov)
    eig = np.asarray(np.maximum(eig, 0.0), dtype=np.float64)
    eig.sort()
    lam3, lam2, lam1 = float(eig[0]), float(eig[1]), float(eig[2])
    denom = float(max(1e-8, lam1 + lam2 + lam3))
    shape_anisotropy = float((lam1 - lam3) / max(1e-8, lam1))
    shape_asphericity = float(lam1 - 0.5 * (lam2 + lam3))
    shape_acylindricity = float(lam2 - lam3)
    shape_planarity_index = float(lam3 / max(1e-8, lam1))
    shape_compactness = float((lam1 * lam2 * lam3) ** (1.0 / 3.0) / max(1e-8, denom / 3.0))

    edges = int(sum(len(v) for v in adjacency.values()) // 2)
    components = int(_count_components(adjacency=adjacency, n_nodes=n_atoms))
    cycle_rank = float(max(0, edges - n_atoms + components))

    torsion_angles = _sample_dihedral_angles(
        coords=arr,
        adjacency=adjacency,
        max_paths=int(max(1, config.max_torsion_paths)),
    )
    torsion_entropy = float(0.0)
    torsion_abs_mean = float(0.0)
    if torsion_angles.size > 0:
        torsion_abs_mean = float(np.mean(np.abs(torsion_angles)) / math.pi)
        hist, _ = np.histogram(torsion_angles, bins=12, range=(-math.pi, math.pi), density=False)
        probs = hist.astype(np.float64)
        probs = probs / float(max(1.0, float(np.sum(probs))))
        nz = probs[probs > 0.0]
        if nz.size > 0:
            torsion_entropy = float(-np.sum(nz * np.log(nz)) / math.log(12.0))

    hull_volume = float("nan")
    hull_area = float("nan")
    hull_surface_volume_ratio = float("nan")
    cavity_void_fraction = float("nan")
    packing_fraction = float("nan")
    point_density = float("nan")
    if bool(config.use_convex_hull) and n_atoms >= 4 and has_scipy():
        try:
            scipy_mod = require_scipy()
            hull = scipy_mod.spatial.ConvexHull(arr)
            hull_volume = float(hull.volume)
            hull_area = float(hull.area)
            if np.isfinite(hull_volume) and hull_volume > 1e-8:
                hull_surface_volume_ratio = float(hull_area / hull_volume)
                point_density = float(n_atoms / hull_volume)
                vdw_vol = float(_vdw_volume(atom_numbers))
                packing_fraction = float(np.clip(vdw_vol / hull_volume, 0.0, 1.0))
                cavity_void_fraction = float(np.clip(1.0 - packing_fraction, 0.0, 1.0))
        except Exception:
            pass

    h1_count = float(cycle_rank)
    h1_persistence_sum = float(cycle_rank)
    h1_persistence_max = float(cycle_rank)
    if (
        bool(config.use_persistent_homology)
        and has_ripser()
        and n_atoms <= int(max(4, config.persistence_max_atoms))
    ):
        try:
            ripser_mod = require_ripser()
            dmat = _pairwise_distances(arr)
            out = ripser_mod.ripser(dmat, maxdim=1, distance_matrix=True)
            dgms = out.get("dgms", [])
            h1 = np.asarray(dgms[1], dtype=np.float64) if len(dgms) > 1 else np.zeros((0, 2), dtype=np.float64)
            if h1.size > 0:
                birth = h1[:, 0]
                death = h1[:, 1]
                finite = np.isfinite(birth) & np.isfinite(death) & (death > birth)
                life = death[finite] - birth[finite]
                if life.size > 0:
                    h1_count = float(life.size)
                    h1_persistence_sum = float(np.sum(life))
                    h1_persistence_max = float(np.max(life))
                else:
                    h1_count = 0.0
                    h1_persistence_sum = 0.0
                    h1_persistence_max = 0.0
            else:
                h1_count = 0.0
                h1_persistence_sum = 0.0
                h1_persistence_max = 0.0
        except Exception:
            pass

    return {
        "n_atoms": float(n_atoms),
        "radius_gyration": float(radius_gyration),
        "shape_anisotropy": float(shape_anisotropy),
        "shape_asphericity": float(shape_asphericity),
        "shape_acylindricity": float(shape_acylindricity),
        "shape_planarity_index": float(shape_planarity_index),
        "shape_compactness": float(shape_compactness),
        "cycle_rank": float(cycle_rank),
        "torsion_entropy_norm": float(torsion_entropy),
        "torsion_abs_mean": float(torsion_abs_mean),
        "hull_volume": float(hull_volume),
        "hull_area": float(hull_area),
        "hull_surface_volume_ratio": float(hull_surface_volume_ratio),
        "cavity_void_fraction": float(cavity_void_fraction),
        "packing_fraction": float(packing_fraction),
        "point_density": float(point_density),
        "h1_count": float(h1_count),
        "h1_persistence_sum": float(h1_persistence_sum),
        "h1_persistence_max": float(h1_persistence_max),
    }


def summarize_advanced_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Aggregate advanced geometry/topology metrics into a JSON-friendly summary."""
    if len(rows) == 0:
        return {
            "enabled": True,
            "n_patches_evaluated": 0,
            "metrics": {},
        }

    keys: set[str] = set()
    for row in rows:
        for k in row.keys():
            keys.add(str(k))

    metrics: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        vals = [float(row.get(key, float("nan"))) for row in rows]
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        q = np.quantile(arr, [0.1, 0.5, 0.9])
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = float(std / (abs(mean) + 1e-8))
        metrics[str(key)] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "min": float(np.min(arr)),
            "q10": float(q[0]),
            "median": float(q[1]),
            "q90": float(q[2]),
            "max": float(np.max(arr)),
        }

    return {
        "enabled": True,
        "n_patches_evaluated": int(len(rows)),
        "metrics": metrics,
    }


def _count_components(*, adjacency: Mapping[int, Sequence[int]], n_nodes: int) -> int:
    seen: set[int] = set()
    components = 0
    for node in range(int(n_nodes)):
        if node in seen:
            continue
        components += 1
        stack = [int(node)]
        while stack:
            cur = int(stack.pop())
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in adjacency.get(cur, ()):
                ni = int(nxt)
                if ni not in seen:
                    stack.append(ni)
    return int(components)


def _sample_dihedral_angles(
    *,
    coords: np.ndarray,
    adjacency: Mapping[int, Sequence[int]],
    max_paths: int,
) -> np.ndarray:
    n = int(coords.shape[0])
    if n < 4:
        return np.zeros((0,), dtype=np.float64)

    seen_paths: set[tuple[int, int, int, int]] = set()
    angles: list[float] = []

    for j in range(n):
        nbr_j = [int(x) for x in adjacency.get(j, ())]
        for k in nbr_j:
            if j >= k:
                continue
            left = [i for i in nbr_j if int(i) != int(k)]
            right = [l for l in adjacency.get(k, ()) if int(l) != int(j)]
            for i in left:
                for l in right:
                    ii, jj, kk, ll = int(i), int(j), int(k), int(l)
                    if ii == ll:
                        continue
                    p = (ii, jj, kk, ll)
                    pr = (ll, kk, jj, ii)
                    key = p if p <= pr else pr
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    angle = _dihedral(coords[ii], coords[jj], coords[kk], coords[ll])
                    if np.isfinite(angle):
                        angles.append(float(angle))
                    if len(angles) >= int(max_paths):
                        return np.asarray(angles, dtype=np.float64)
    return np.asarray(angles, dtype=np.float64)


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    n1n = np.linalg.norm(n1)
    n2n = np.linalg.norm(n2)
    b1n = np.linalg.norm(b1)
    if n1n <= 1e-12 or n2n <= 1e-12 or b1n <= 1e-12:
        return float("nan")

    n1 = n1 / n1n
    n2 = n2 / n2n
    m1 = np.cross(n1, b1 / b1n)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    sq = np.sum(np.square(diff), axis=-1)
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq).astype(np.float64, copy=False)


def _vdw_volume(atom_numbers: Sequence[int]) -> float:
    vol = 0.0
    for z in atom_numbers:
        r = float(_VDW_RADII.get(int(z), 1.70))
        vol += (4.0 / 3.0) * math.pi * (r ** 3)
    return float(vol)


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
        "distance": (
            "dist",
            "distance",
            "bondlen",
            "bond_length",
            "pair_dist",
            "nearest",
            "euclid",
            "radius",
        ),
        "angle": ("angle", "bend", "bond_angle", "valence_angle"),
        "dihedral": ("dihedral", "torsion", "phi", "psi", "chi", "omega", "tor"),
        "planarity": ("planar", "planarity", "out_of_plane", "oop", "flatness", "pbf"),
        "shape": (
            "shape",
            "asphericity",
            "eccentricity",
            "spherocity",
            "globularity",
            "anisotropy",
            "inertialshape",
            "npr",
        ),
        "size": ("radius_gyration", "rgyr", "gyration", "span", "diameter", "extent"),
        "inertia": ("inertia", "principal", "pmi", "moment", "rotconst", "rotational"),
        "surface_volume": ("surface", "sasa", "vsa", "psa", "volume", "molarvolume", "labuteasa"),
        "ring_strain": ("strain", "ring_strain", "angle_strain", "torsional_strain", "ring_tension"),
        "hbond_geometry": ("hb", "hbond", "donor_acceptor", "d_a", "dha", "dha_angle"),
        "global_3d_fingerprint": ("rdf", "morse", "whim", "getaway", "autocorr3d", "moran3d", "geary3d"),
    }

    _QM_FAMILY_TOKENS: Mapping[str, tuple[str, ...]] = {
        "homo": ("homo", "ehomo", "eps_homo"),
        "lumo": ("lumo", "elumo", "eps_lumo"),
        "gap": ("gap", "homo_lumo", "bandgap", "deltae"),
        "chemical_potential": ("mu_ev", "chemical_potential"),
        "dipole": ("dipole_d", "dipole", "dipolemoment"),
        "polarizability": ("polariz", "alpha"),
        "hardness": ("hardness", "eta"),
        "softness": ("softness", "sigma_soft", "soft"),
        "electronegativity": ("electroneg", "chi"),
        "electrophilicity": ("electrophil", "omega"),
        "nucleophilicity": ("nucleophil",),
        "charge_transfer": ("charge_transfer", "ct", "deltaq", "charge_sep"),
        "esp": ("esp", "electrostatic", "mep"),
        "fukui": ("fukui", "fplus", "fminus"),
        "fukui_plus": ("fplus",),
        "fukui_minus": ("fminus",),
        "ionization_potential": ("vip", "ip", "ionization_potential"),
        "electron_affinity": ("vea", "ea", "electron_affinity"),
        "bond_order": ("bo_sum", "bo_max", "bo_mean", "bond_order", "bo_"),
        "bond_order_conjugation": ("bo_conj", "conj_bo"),
        "atomic_charge_distribution": (
            "q_min",
            "q_max",
            "q_mean",
            "q_std",
            "q_abs_sum",
            "q_range",
            "q_pos_top3_mean",
            "q_neg_top3_mean",
        ),
        "charge_spread_distance": ("q_abs_r_mean", "q_abs_r2_rms", "d_pos_neg"),
        "nbo": ("nbo",),
        "mulliken": ("mulliken",),
        "npa": ("npa",),
        "quadrupole": ("quadrupole", "quad_norm", "quad_trace"),
    }

    _ORCA_FAMILY_TOKENS: Mapping[str, tuple[str, ...]] = {
        "homo": ("homo", "ehomo"),
        "lumo": ("lumo", "elumo"),
        "gap": ("gap", "homo_lumo", "deltae"),
        "excitation_energy": ("exc", "s1", "t1", "vertical_exc", "transition_energy"),
        "oscillator_strength": ("fosc", "osc", "oscillator", "f_"),
        "transition_dipole": ("transition_dipole", "tdm", "mu_trans"),
        "singlet_triplet_gap": ("s1_t1", "delta_st", "singlet_triplet"),
        "spin_orbit": ("soc", "spin_orbit"),
        "charge_transfer_excited": ("ct_exc", "charge_transfer_excited", "excited_ct", "nto"),
        "reorganization_energy": ("reorg", "lambda_reorg"),
        "radiative_rate": ("kr", "radiative_rate"),
        "nonradiative_rate": ("knr", "nonradiative_rate"),
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
        self._advanced_geom_cfg = AdvancedGeometryTopologyConfig(
            enabled=bool(self.config.use_advanced_geom_topology),
            max_patches=int(self.config.advanced_geom_topology_max_patches),
            min_atoms=int(self.config.advanced_geom_topology_min_atoms),
            max_torsion_paths=int(self.config.advanced_geom_topology_max_torsion_paths),
            use_convex_hull=bool(self.config.advanced_geom_use_convex_hull),
            use_persistent_homology=bool(self.config.advanced_geom_use_persistent_homology),
            persistence_max_atoms=int(self.config.advanced_geom_persistence_max_atoms),
        )
        (
            self._orca_vectors_by_conf_id,
            self._orca_vectors_by_mol_id,
            self._orca_feature_names,
            self._orca_enabled,
        ) = self._load_orca_descriptor_index()
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
            log_event(
                "INFO",
                "explainability.chem_ace.tag_concepts.openbabel",
                enabled=False,
                reason="disabled_by_config",
            )
            return None
        try:
            backend = require_openbabel_pybel()
            backend_name = getattr(backend, "__name__", backend.__class__.__name__)
            log_event(
                "INFO",
                "explainability.chem_ace.tag_concepts.openbabel",
                enabled=True,
                backend=str(backend_name),
            )
            return backend
        except OptionalDependencyError as exc:
            log_event(
                "WARN",
                "explainability.chem_ace.tag_concepts.openbabel",
                enabled=False,
                reason="optional_dependency_missing",
                error=repr(exc),
            )
            logger.info("Open Babel pybel is unavailable; skipping Open Babel semantic descriptors")
            return None

    def _load_orca_descriptor_index(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[str, ...], bool]:
        """
        Load optional external ORCA descriptors and build conf/molecule lookup maps.

        Expected table columns:
        - conf identifier (default: `conf_id`)
        - molecule identifier (default: `ID`)
        - numeric ORCA descriptor columns (auto-inferred or explicitly listed)
        """
        if not bool(self.config.use_orca_descriptors):
            log_event(
                "INFO",
                "explainability.chem_ace.tag_concepts.orca",
                enabled=False,
                reason="disabled_by_config",
            )
            return {}, {}, (), False

        path_raw = self.config.orca_descriptors_path
        if path_raw is None or not str(path_raw).strip():
            log_event(
                "WARN",
                "explainability.chem_ace.tag_concepts.orca",
                enabled=False,
                reason="missing_orca_descriptors_path",
            )
            return {}, {}, (), False

        path = Path(str(path_raw))
        if not path.exists():
            log_event(
                "WARN",
                "explainability.chem_ace.tag_concepts.orca",
                enabled=False,
                reason="orca_descriptors_path_not_found",
                path=str(path),
            )
            return {}, {}, (), False

        try:
            suffix = path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                df = pd.read_parquet(path)
            elif suffix in {".json"}:
                df = pd.read_json(path)
            else:
                df = pd.read_csv(path)
        except Exception as exc:
            log_event(
                "WARN",
                "explainability.chem_ace.tag_concepts.orca",
                enabled=False,
                reason="read_failed",
                path=str(path),
                error=repr(exc),
            )
            return {}, {}, (), False

        if df.empty:
            log_event(
                "WARN",
                "explainability.chem_ace.tag_concepts.orca",
                enabled=False,
                reason="empty_table",
                path=str(path),
            )
            return {}, {}, (), False

        conf_col = str(self.config.orca_conf_id_col)
        mol_col = str(self.config.orca_mol_id_col)
        explicit_cols = [str(x) for x in self.config.orca_descriptor_cols if str(x).strip()]

        work = df.copy()
        for col in (conf_col, mol_col):
            if col in work.columns:
                work[col] = work[col].astype(str)
                work.loc[work[col].str.lower().isin(["", "nan", "none"]), col] = ""

        if len(explicit_cols) > 0:
            descriptor_cols = [c for c in explicit_cols if c in work.columns]
        else:
            descriptor_cols = []
            for col in work.columns:
                if col in {conf_col, mol_col}:
                    continue
                try:
                    series = pd.to_numeric(work[col], errors="coerce")
                except Exception:
                    continue
                if int(series.notna().sum()) <= 0:
                    continue
                descriptor_cols.append(str(col))

        if len(descriptor_cols) == 0:
            log_event(
                "WARN",
                "explainability.chem_ace.tag_concepts.orca",
                enabled=False,
                reason="no_numeric_descriptor_columns",
                path=str(path),
            )
            return {}, {}, (), False

        mat = np.asarray(
            work[descriptor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
        med = np.nanmedian(mat, axis=0)
        q25 = np.nanpercentile(mat, 25.0, axis=0)
        q75 = np.nanpercentile(mat, 75.0, axis=0)
        iqr = q75 - q25
        iqr = np.where(np.isfinite(iqr) & (np.abs(iqr) > 1e-12), iqr, 1.0)
        z = (mat - med.reshape(1, -1)) / iqr.reshape(1, -1)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        by_conf: dict[str, np.ndarray] = {}
        by_mol_lists: dict[str, list[np.ndarray]] = {}
        for i in range(int(z.shape[0])):
            vec = np.asarray(z[i], dtype=np.float32)
            conf_id = ""
            mol_id = ""
            if conf_col in work.columns:
                conf_id = str(work.iloc[i][conf_col]).strip()
            if mol_col in work.columns:
                mol_id = str(work.iloc[i][mol_col]).strip()
            if conf_id:
                by_conf.setdefault(conf_id, vec)
            if mol_id:
                if mol_id not in by_mol_lists:
                    by_mol_lists[mol_id] = []
                by_mol_lists[mol_id].append(vec)

        by_mol: dict[str, np.ndarray] = {}
        for mol_id, vectors in by_mol_lists.items():
            if len(vectors) == 0:
                continue
            by_mol[str(mol_id)] = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)

        log_event(
            "INFO",
            "explainability.chem_ace.tag_concepts.orca",
            enabled=True,
            path=str(path),
            n_rows=int(z.shape[0]),
            n_descriptor_cols=int(len(descriptor_cols)),
            n_conf_indexed=int(len(by_conf)),
            n_mol_indexed=int(len(by_mol)),
        )
        return by_conf, by_mol, tuple(str(c) for c in descriptor_cols), True

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
        tag_t0 = time.perf_counter()
        log_event(
            "START",
            "explainability.chem_ace.tag_concepts.tag_one",
            concept_id=str(concept_id),
            n_patches=int(len(concept_patches)),
        )
        descriptors = self._compute_descriptors(
            concept_id=str(concept_id),
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
        tags.extend(self._advanced_geometry_topology_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._pharmacophore_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._qm_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._orca_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._openbabel_tags(concept_id=concept_id, descriptors=descriptors))
        tags.extend(self._cross_modal_tags(concept_id=concept_id, descriptors=descriptors))
        tags = self._deduplicate_tags(tags)

        tag_names = [t.tag for t in tags]
        descriptor_rank = self._descriptor_rank(descriptors)
        label_auto = choose_label(tags=tag_names, registry=self.registry, descriptor_rank=descriptor_rank)
        log_event(
            "DONE",
            "explainability.chem_ace.tag_concepts.tag_one",
            concept_id=str(concept_id),
            n_tags=int(len(tags)),
            label_auto=str(label_auto),
            elapsed_s=f"{(time.perf_counter() - tag_t0):.2f}",
        )

        return SemanticTaggingResult(
            concept_id=concept_id,
            label_auto=label_auto,
            tags=tags,
            evidence_json=descriptors,
        )

    def _compute_descriptors(
        self,
        *,
        concept_id: str,
        concept_patches: Sequence[PatchRecord],
        molecules_by_id: Mapping[str, Any],
        inst_by_pair: Optional[Mapping[tuple[str, str], np.ndarray]],
        inst_mean_by_id: Optional[Mapping[str, np.ndarray]],
        inst_geom_dim: int,
        inst_qm_dim: int,
        geom_feature_names: Optional[Sequence[str]],
        qm_feature_names: Optional[Sequence[str]],
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        n_patches = int(len(concept_patches))
        log_event(
            "START",
            "explainability.chem_ace.tag_concepts.compute_descriptors",
            concept_id=str(concept_id),
            n_patches=n_patches,
            functional_rules=int(len(self.functional_rules)),
            smarts_rx_rules=int(len(self.smarts_rx_rules)),
            openbabel_enabled=bool(self._openbabel_pybel is not None),
        )
        try:
            require_rdkit()
            from rdkit.Chem import rdPartialCharges
            from rdkit.Chem import ChemicalFeatures
            from rdkit import RDConfig
            from rdkit import Chem
        except OptionalDependencyError as exc:
            log_event(
                "WARN",
                "explainability.chem_ace.tag_concepts.compute_descriptors",
                concept_id=str(concept_id),
                rdkit_available=False,
                error=repr(exc),
            )
            return {"n_patches": n_patches, "rdkit_available": False}

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
        orca_vectors: list[np.ndarray] = []
        advanced_geom_rows: list[dict[str, float]] = []
        advanced_selected_patch_ids = select_patch_ids_for_advanced(
            patch_ids=[str(p.patch_id) for p in concept_patches],
            max_patches=int(self._advanced_geom_cfg.max_patches),
        )

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
        gasteiger_mol_calcs = 0
        functional_overlap_hits_total = 0
        smarts_rx_overlap_hits_total = 0

        geom_names = self._normalize_geom_feature_names(
            geom_feature_names=geom_feature_names,
            inst_geom_dim=int(inst_geom_dim),
        )
        qm_names = self._normalize_qm_feature_names(
            qm_feature_names=qm_feature_names,
            inst_qm_dim=int(inst_qm_dim),
        )

        valid_patch_count = 0
        progress_every = max(1, n_patches // 5) if n_patches > 0 else 1
        emit_progress = n_patches >= 250
        for done, patch in enumerate(concept_patches, start=1):
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
                gasteiger_mol_calcs += 1
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

            conf_idx_patch: Optional[int] = None
            coords_patch: Optional[np.ndarray] = None
            if mol.GetNumConformers() > 0 and len(atom_ids) >= 3:
                conf_idx_patch = self._resolve_conf_idx(mol=mol, conf_id=patch.conf_id)
                if conf_idx_patch is not None:
                    conf = mol.GetConformer(int(conf_idx_patch))
                    coords_patch = np.asarray(
                        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in atom_ids],
                        dtype=np.float64,
                    )
                    planarity_rmsd.append(float(self._planarity_rmsd(coords_patch)))
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
                        functional_overlap_hits_total += int(overlap_hits)
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
                        smarts_rx_overlap_hits_total += int(overlap_hits_rx)
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
            orca_vec = self._resolve_orca_vector_for_patch(patch=patch)
            if orca_vec is not None and orca_vec.size > 0:
                orca_vectors.append(orca_vec.astype(np.float32, copy=False))

            if (
                bool(self._advanced_geom_cfg.enabled)
                and (str(patch.patch_id) in advanced_selected_patch_ids)
                and (coords_patch is not None)
                and (coords_patch.shape[0] >= int(max(3, self._advanced_geom_cfg.min_atoms)))
            ):
                local_adj = self._build_local_patch_adjacency(
                    atom_indices=atom_ids,
                    patch_bonds=patch_bonds,
                )
                atom_numbers = [int(a.GetAtomicNum()) for a in atoms]
                adv = compute_patch_advanced_metrics(
                    coords=coords_patch,
                    atom_numbers=atom_numbers,
                    adjacency=local_adj,
                    config=self._advanced_geom_cfg,
                )
                if len(adv) > 0:
                    advanced_geom_rows.append(adv)
            if emit_progress and ((done % progress_every) == 0 or done == n_patches):
                log_event(
                    "PROGRESS",
                    "explainability.chem_ace.tag_concepts.compute_descriptors.loop",
                    concept_id=str(concept_id),
                    done=int(done),
                    total=n_patches,
                    pct=f"{(100.0 * done / max(1, n_patches)):.1f}",
                    valid_patches=int(valid_patch_count),
                    advanced_samples=int(len(advanced_geom_rows)),
                )

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
        orca_summary = self._summarize_orca_vectors(
            orca_vectors=orca_vectors,
        )
        advanced_geom_topology_summary = summarize_advanced_metrics(advanced_geom_rows)
        log_event(
            "INFO",
            "explainability.chem_ace.tag_concepts.compute_descriptors.summary",
            concept_id=str(concept_id),
            n_valid_patches=int(valid_patch_count),
            n_unique_molecules=int(len(mol_seen)),
            gasteiger_molecules=int(gasteiger_mol_calcs),
            smarts_overlap_hits=int(functional_overlap_hits_total),
            smarts_rx_overlap_hits=int(smarts_rx_overlap_hits_total),
            geom_vectors=int(len(geom_vectors)),
            qm_vectors=int(len(qm_vectors)),
            orca_vectors=int(len(orca_vectors)),
            advanced_geom_patches=int(len(advanced_geom_rows)),
        )
        if self._openbabel_pybel is not None:
            log_event(
                "START",
                "explainability.chem_ace.tag_concepts.openbabel_summary",
                concept_id=str(concept_id),
                n_molecules=int(len(mol_seen)),
            )
        openbabel_summary = self._summarize_openbabel_descriptors(molecules_by_id=mol_seen)
        if self._openbabel_pybel is not None:
            log_event(
                "DONE",
                "explainability.chem_ace.tag_concepts.openbabel_summary",
                concept_id=str(concept_id),
                n_molecules=int(openbabel_summary.get("n_molecules", 0)),
                enabled=bool(openbabel_summary.get("enabled", False)),
            )

        out = {
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
            "orca_summary": orca_summary,
            "advanced_geometry_topology": advanced_geom_topology_summary,
            "openbabel_descriptor_summary": openbabel_summary,
        }
        log_event(
            "DONE",
            "explainability.chem_ace.tag_concepts.compute_descriptors",
            concept_id=str(concept_id),
            n_valid_patches=int(valid_patch_count),
            elapsed_s=f"{(time.perf_counter() - t0):.2f}",
        )
        return out

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

    def _summarize_orca_vectors(
        self,
        *,
        orca_vectors: Sequence[np.ndarray],
    ) -> dict[str, Any]:
        return self._summarize_vector_block(
            vectors=orca_vectors,
            feature_names=self._orca_feature_names,
            fallback_prefix="orca",
            family_tokens=self._ORCA_FAMILY_TOKENS,
        )

    def _resolve_orca_vector_for_patch(self, *, patch: PatchRecord) -> Optional[np.ndarray]:
        if not bool(self._orca_enabled):
            return None
        if patch.conf_id is not None:
            vec = self._orca_vectors_by_conf_id.get(str(patch.conf_id))
            if vec is not None:
                return np.asarray(vec, dtype=np.float32).reshape(-1)
        vec = self._orca_vectors_by_mol_id.get(str(patch.mol_id))
        if vec is not None:
            return np.asarray(vec, dtype=np.float32).reshape(-1)
        return None

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
        matched_idxs: set[int] = set()
        for idxs in family_index.values():
            matched_idxs.update(int(i) for i in idxs)

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

        unmatched_idx = [i for i in range(dim) if i not in matched_idxs]
        unmatched_top_idx = sorted(unmatched_idx, key=lambda i: float(abs_mean[int(i)]), reverse=True)[: min(12, len(unmatched_idx))]
        unmatched_top_features = [
            {
                "name": str(feat_names[int(i)]),
                "abs_z_mean": float(abs_mean[int(i)]),
                "z_mean": float(mean[int(i)]),
                "z_std": float(std[int(i)]),
            }
            for i in unmatched_top_idx
        ]

        return {
            "n_vectors": int(n_vectors),
            "n_features": int(dim),
            "top_features": top_features,
            "family_stats": family_stats,
            "n_family_matched_features": int(len(matched_idxs)),
            "n_unmatched_features": int(max(0, dim - len(matched_idxs))),
            "family_coverage": float(len(matched_idxs) / max(1, dim)),
            "top_unmatched_features": unmatched_top_features,
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

        size_abs = float(fam("size").get("abs_z_mean", 0.0))
        size_mean = float(fam("size").get("z_mean", 0.0))
        if size_abs >= thr:
            if size_mean >= 0.0:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "expanded conformer envelope",
                        min(1.0, 0.55 + 0.20 * size_abs),
                        "geometry_descriptor_tagger",
                        descriptors,
                    )
                )
            else:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "compact conformer envelope",
                        min(1.0, 0.55 + 0.20 * size_abs),
                        "geometry_descriptor_tagger",
                        descriptors,
                    )
                )

        inertia_abs = float(fam("inertia").get("abs_z_mean", 0.0))
        if inertia_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "anisotropic inertia profile",
                    min(1.0, 0.55 + 0.18 * inertia_abs),
                    "geometry_descriptor_tagger",
                    descriptors,
                )
            )

        hbond_geom_abs = float(fam("hbond_geometry").get("abs_z_mean", 0.0))
        if hbond_geom_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "hydrogen-bond geometry pattern",
                    min(1.0, 0.55 + 0.18 * hbond_geom_abs),
                    "geometry_descriptor_tagger",
                    descriptors,
                )
            )

        global_3d_abs = float(fam("global_3d_fingerprint").get("abs_z_mean", 0.0))
        if global_3d_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "rich 3D field signature",
                    min(1.0, 0.55 + 0.18 * global_3d_abs),
                    "geometry_descriptor_tagger",
                    descriptors,
                )
            )

        return tags

    def _advanced_geometry_topology_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        ag = descriptors.get("advanced_geometry_topology", {})
        if not isinstance(ag, Mapping):
            return []
        if int(ag.get("n_patches_evaluated", 0)) <= 0:
            return []

        metrics = ag.get("metrics", {})
        if not isinstance(metrics, Mapping):
            return []

        def m(name: str, key: str = "mean") -> float:
            row = metrics.get(name, {})
            if isinstance(row, Mapping):
                return float(row.get(key, 0.0))
            return 0.0

        tags: list[TagAssignment] = []
        anis = m("shape_anisotropy")
        torsion_ent = m("torsion_entropy_norm")
        cycle_rank = m("cycle_rank")
        h1_sum = m("h1_persistence_sum")
        void_frac = m("cavity_void_fraction")
        sv_ratio = m("hull_surface_volume_ratio")
        packing = m("packing_fraction")
        rg_cv = m("radius_gyration", "cv")

        if anis >= 0.55:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "anisotropic 3D shape envelope",
                    min(1.0, 0.55 + 0.25 * anis),
                    "advanced_geometry_topology_tagger",
                    descriptors,
                )
            )
        if torsion_ent >= 0.45 or rg_cv >= 0.45:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "torsionally diverse conformer ensemble",
                    min(1.0, 0.55 + 0.22 * max(torsion_ent, rg_cv)),
                    "advanced_geometry_topology_tagger",
                    descriptors,
                )
            )
        if cycle_rank >= 1.0 or h1_sum >= 0.8:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "loop-rich topology",
                    min(1.0, 0.55 + 0.20 * max(cycle_rank, h1_sum)),
                    "advanced_geometry_topology_tagger",
                    descriptors,
                )
            )
        if np.isfinite(void_frac) and void_frac >= 0.15:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "cavity-prone geometry",
                    min(1.0, 0.55 + 0.30 * void_frac),
                    "advanced_geometry_topology_tagger",
                    descriptors,
                )
            )
        if np.isfinite(sv_ratio) and sv_ratio >= 2.4:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "high-curvature molecular surface",
                    min(1.0, 0.55 + 0.05 * (sv_ratio - 2.4)),
                    "advanced_geometry_topology_tagger",
                    descriptors,
                )
            )
        if np.isfinite(packing) and packing >= 0.82:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "densely packed local geometry",
                    min(1.0, 0.55 + 0.30 * packing),
                    "advanced_geometry_topology_tagger",
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

        mu_abs = float(fam("chemical_potential").get("abs_z_mean", 0.0))
        mu_mean = float(fam("chemical_potential").get("z_mean", 0.0))
        if mu_abs >= thr:
            if mu_mean >= 0.0:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "elevated chemical potential profile",
                        min(1.0, 0.55 + 0.18 * mu_abs),
                        "qm_descriptor_tagger",
                        descriptors,
                    )
                )
            else:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "reduced chemical potential profile",
                        min(1.0, 0.55 + 0.18 * mu_abs),
                        "qm_descriptor_tagger",
                        descriptors,
                    )
                )

        ip_abs = float(fam("ionization_potential").get("abs_z_mean", 0.0))
        ip_mean = float(fam("ionization_potential").get("z_mean", 0.0))
        if ip_abs >= thr:
            if ip_mean >= 0.0:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "high ionization-potential profile",
                        min(1.0, 0.55 + 0.2 * ip_abs),
                        "qm_descriptor_tagger",
                        descriptors,
                    )
                )
            else:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "low ionization-potential profile",
                        min(1.0, 0.55 + 0.2 * ip_abs),
                        "qm_descriptor_tagger",
                        descriptors,
                    )
                )

        ea_abs = float(fam("electron_affinity").get("abs_z_mean", 0.0))
        ea_mean = float(fam("electron_affinity").get("z_mean", 0.0))
        if ea_abs >= thr:
            if ea_mean >= 0.0:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "high electron-affinity profile",
                        min(1.0, 0.55 + 0.2 * ea_abs),
                        "qm_descriptor_tagger",
                        descriptors,
                    )
                )
            else:
                tags.append(
                    self._mk_tag(
                        concept_id,
                        "low electron-affinity profile",
                        min(1.0, 0.55 + 0.2 * ea_abs),
                        "qm_descriptor_tagger",
                        descriptors,
                    )
                )

        bo_abs = float(fam("bond_order").get("abs_z_mean", 0.0))
        bo_mean = float(fam("bond_order").get("z_mean", 0.0))
        if bo_abs >= thr and bo_mean >= 0.0:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "bond-order rigidification profile",
                    min(1.0, 0.55 + 0.2 * bo_abs),
                    "qm_descriptor_tagger",
                    descriptors,
                )
            )

        bo_conj_abs = float(fam("bond_order_conjugation").get("abs_z_mean", 0.0))
        bo_conj_mean = float(fam("bond_order_conjugation").get("z_mean", 0.0))
        if bo_conj_abs >= thr and bo_conj_mean >= 0.0:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "conjugated bond-order network",
                    min(1.0, 0.55 + 0.2 * bo_conj_abs),
                    "qm_descriptor_tagger",
                    descriptors,
                )
            )

        charge_abs = float(fam("atomic_charge_distribution").get("abs_z_mean", 0.0))
        if charge_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "polarized atomic-charge landscape",
                    min(1.0, 0.55 + 0.18 * charge_abs),
                    "qm_descriptor_tagger",
                    descriptors,
                )
            )

        spread_abs = float(fam("charge_spread_distance").get("abs_z_mean", 0.0))
        spread_mean = float(fam("charge_spread_distance").get("z_mean", 0.0))
        if spread_abs >= thr and spread_mean >= 0.0:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "long-range charge-separation profile",
                    min(1.0, 0.55 + 0.18 * spread_abs),
                    "qm_descriptor_tagger",
                    descriptors,
                )
            )

        fplus_abs = float(fam("fukui_plus").get("abs_z_mean", 0.0))
        if fplus_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "nucleophilic hotspot profile (Fukui+)",
                    min(1.0, 0.55 + 0.2 * fplus_abs),
                    "qm_descriptor_tagger",
                    descriptors,
                )
            )

        fminus_abs = float(fam("fukui_minus").get("abs_z_mean", 0.0))
        if fminus_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "electrophilic hotspot profile (Fukui-)",
                    min(1.0, 0.55 + 0.2 * fminus_abs),
                    "qm_descriptor_tagger",
                    descriptors,
                )
            )

        quad_abs = float(fam("quadrupole").get("abs_z_mean", 0.0))
        if quad_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "anisotropic quadrupole field",
                    min(1.0, 0.55 + 0.18 * quad_abs),
                    "qm_descriptor_tagger",
                    descriptors,
                )
            )

        return tags

    def _orca_tags(self, *, concept_id: str, descriptors: Mapping[str, Any]) -> list[TagAssignment]:
        osum = descriptors.get("orca_summary", {})
        if not isinstance(osum, Mapping):
            return []
        n_vectors = int(osum.get("n_vectors", 0))
        if n_vectors < int(self.config.orca_min_vectors_for_tagging):
            return []

        family_stats = osum.get("family_stats", {})
        if not isinstance(family_stats, Mapping):
            family_stats = {}
        thr = float(self.config.orca_z_threshold)

        def fam(name: str) -> Mapping[str, Any]:
            item = family_stats.get(name, {})
            if isinstance(item, Mapping):
                return item
            return {}

        tags: list[TagAssignment] = []
        fosc_abs = float(fam("oscillator_strength").get("abs_z_mean", 0.0))
        fosc_mean = float(fam("oscillator_strength").get("z_mean", 0.0))
        eexc_abs = float(fam("excitation_energy").get("abs_z_mean", 0.0))
        eexc_mean = float(fam("excitation_energy").get("z_mean", 0.0))
        st_abs = float(fam("singlet_triplet_gap").get("abs_z_mean", 0.0))
        soc_abs = float(fam("spin_orbit").get("abs_z_mean", 0.0))
        ctex_abs = float(fam("charge_transfer_excited").get("abs_z_mean", 0.0))
        kr_abs = float(fam("radiative_rate").get("abs_z_mean", 0.0))
        knr_abs = float(fam("nonradiative_rate").get("abs_z_mean", 0.0))

        if fosc_abs >= thr and fosc_mean >= 0.0:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "bright excited-state manifold",
                    min(1.0, 0.55 + 0.22 * fosc_abs),
                    "orca_descriptor_tagger",
                    descriptors,
                )
            )
        if eexc_abs >= thr and eexc_mean <= -thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "low-energy excitation profile",
                    min(1.0, 0.55 + 0.20 * eexc_abs),
                    "orca_descriptor_tagger",
                    descriptors,
                )
            )
        if ctex_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "excited-state charge-transfer motif",
                    min(1.0, 0.55 + 0.20 * ctex_abs),
                    "orca_descriptor_tagger",
                    descriptors,
                )
            )
        if st_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "singlet-triplet split electronic manifold",
                    min(1.0, 0.55 + 0.18 * st_abs),
                    "orca_descriptor_tagger",
                    descriptors,
                )
            )
        if soc_abs >= thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "spin-orbit coupled transition channel",
                    min(1.0, 0.55 + 0.18 * soc_abs),
                    "orca_descriptor_tagger",
                    descriptors,
                )
            )
        if kr_abs >= thr and knr_abs < thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "radiative-decay-favored excited state",
                    min(1.0, 0.55 + 0.16 * kr_abs),
                    "orca_descriptor_tagger",
                    descriptors,
                )
            )
        if knr_abs >= thr and kr_abs < thr:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "nonradiative-decay-favored excited state",
                    min(1.0, 0.55 + 0.16 * knr_abs),
                    "orca_descriptor_tagger",
                    descriptors,
                )
            )
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
        orca = descriptors.get("orca_summary", {})
        if not isinstance(orca, Mapping):
            orca = {}

        qfam = qms.get("family_stats", {})
        if not isinstance(qfam, Mapping):
            qfam = {}
        gfam = gms.get("family_stats", {})
        if not isinstance(gfam, Mapping):
            gfam = {}
        ofam = orca.get("family_stats", {})
        if not isinstance(ofam, Mapping):
            ofam = {}

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

        def ofam_stat(name: str) -> Mapping[str, Any]:
            item = ofam.get(name, {})
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

        # Heuristic photophysics proxy tags for transmittance / fluorescence behavior.
        # These are intentionally labeled as "proxy" because they do not use explicit excited-state observables.
        gap_mean = float(qfam_stat("gap").get("z_mean", 0.0))
        gap_abs = float(qfam_stat("gap").get("abs_z_mean", 0.0))
        hard_mean = float(qfam_stat("hardness").get("z_mean", 0.0))
        dip_abs = float(qfam_stat("dipole").get("abs_z_mean", 0.0))
        ct_abs = float(qfam_stat("charge_transfer").get("abs_z_mean", 0.0))
        bo_conj_abs = float(qfam_stat("bond_order_conjugation").get("abs_z_mean", 0.0))
        fplus_abs = float(qfam_stat("fukui_plus").get("abs_z_mean", 0.0))
        fminus_abs = float(qfam_stat("fukui_minus").get("abs_z_mean", 0.0))

        if gap_abs >= thr_q and gap_mean >= thr_q and hard_mean >= 0.0 and ct_abs < thr_q and dip_abs < max(thr_q, 0.9):
            tags.append(
                self._mk_tag(
                    concept_id,
                    "transmittance-favored photophysics proxy",
                    min(1.0, 0.55 + 0.12 * (gap_abs + max(0.0, gap_mean) + max(0.0, hard_mean))),
                    "photophysics_proxy_tagger",
                    descriptors,
                )
            )

        if bo_conj_abs >= thr_q and planar >= thr_g and (ct_abs >= thr_q or dip_abs >= thr_q):
            tags.append(
                self._mk_tag(
                    concept_id,
                    "fluorescence-favored photophysics proxy",
                    min(1.0, 0.55 + 0.10 * (bo_conj_abs + planar + max(ct_abs, dip_abs))),
                    "photophysics_proxy_tagger",
                    descriptors,
                )
            )

        if gap_mean <= -thr_q and (ct_abs >= thr_q or fminus_abs >= thr_q):
            tags.append(
                self._mk_tag(
                    concept_id,
                    "red-shifted absorption proxy",
                    min(1.0, 0.55 + 0.10 * (abs(gap_mean) + max(ct_abs, fminus_abs))),
                    "photophysics_proxy_tagger",
                    descriptors,
                )
            )

        if gap_mean >= thr_q and fplus_abs < thr_q and fminus_abs < thr_q:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "blue-shifted transparency proxy",
                    min(1.0, 0.55 + 0.10 * (gap_mean + gap_abs)),
                    "photophysics_proxy_tagger",
                    descriptors,
                )
            )

        # Cross-link explicit excited-state ORCA semantics with structure/geometry.
        fosc_abs = float(ofam_stat("oscillator_strength").get("abs_z_mean", 0.0))
        ex_abs = float(ofam_stat("excitation_energy").get("abs_z_mean", 0.0))
        ex_mean = float(ofam_stat("excitation_energy").get("z_mean", 0.0))
        ctex_abs = float(ofam_stat("charge_transfer_excited").get("abs_z_mean", 0.0))
        st_abs = float(ofam_stat("singlet_triplet_gap").get("abs_z_mean", 0.0))
        soc_abs = float(ofam_stat("spin_orbit").get("abs_z_mean", 0.0))
        if aromatic >= float(self.config.aromatic_fraction_threshold) and planar >= thr_g and fosc_abs >= float(self.config.orca_z_threshold):
            tags.append(
                self._mk_tag(
                    concept_id,
                    "planar aromatic bright-state motif",
                    min(1.0, 0.55 + 0.12 * (aromatic + planar + fosc_abs)),
                    "cross_modal_orca_tagger",
                    descriptors,
                )
            )
        if ctex_abs >= float(self.config.orca_z_threshold) and dip_abs >= thr_q:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "charge-transfer dipolar excited-state motif",
                    min(1.0, 0.55 + 0.12 * (ctex_abs + dip_abs)),
                    "cross_modal_orca_tagger",
                    descriptors,
                )
            )
        if ex_abs >= float(self.config.orca_z_threshold) and ex_mean <= -float(self.config.orca_z_threshold) and bo_conj_abs >= thr_q:
            tags.append(
                self._mk_tag(
                    concept_id,
                    "conjugation-driven low-energy excitation motif",
                    min(1.0, 0.55 + 0.10 * (ex_abs + bo_conj_abs)),
                    "cross_modal_orca_tagger",
                    descriptors,
                )
            )
        if st_abs >= float(self.config.orca_z_threshold) and soc_abs >= float(self.config.orca_z_threshold):
            tags.append(
                self._mk_tag(
                    concept_id,
                    "spin-mixed singlet-triplet manifold",
                    min(1.0, 0.55 + 0.10 * (st_abs + soc_abs)),
                    "cross_modal_orca_tagger",
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
    def _build_local_patch_adjacency(
        *,
        atom_indices: Sequence[int],
        patch_bonds: Sequence[Any],
    ) -> dict[int, list[int]]:
        """
        Build local 0..(n-1) adjacency for a patch from molecule-level atom indices.
        """
        local_map = {int(g): int(i) for i, g in enumerate(atom_indices)}
        out: dict[int, set[int]] = {int(i): set() for i in range(len(atom_indices))}
        for b in patch_bonds:
            try:
                u = int(local_map[int(b.GetBeginAtomIdx())])
                v = int(local_map[int(b.GetEndAtomIdx())])
            except Exception:
                continue
            if u == v:
                continue
            out[u].add(v)
            out[v].add(u)
        return {int(k): sorted(int(x) for x in vals) for k, vals in out.items()}

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

        ag = descriptors.get("advanced_geometry_topology", {})
        if isinstance(ag, Mapping):
            metrics = ag.get("metrics", {})
            if isinstance(metrics, Mapping):
                anis = metrics.get("shape_anisotropy", {})
                if isinstance(anis, Mapping) and float(anis.get("mean", 0.0)) >= 0.55:
                    out.append("anisotropic_3d_shape")
                cavity = metrics.get("cavity_void_fraction", {})
                if isinstance(cavity, Mapping) and float(cavity.get("mean", 0.0)) >= 0.15:
                    out.append("cavity_prone")
                topo = metrics.get("h1_persistence_sum", {})
                if isinstance(topo, Mapping) and float(topo.get("mean", 0.0)) >= 0.8:
                    out.append("loop_rich_topology")

        osum = descriptors.get("orca_summary", {})
        if isinstance(osum, Mapping):
            ofam = osum.get("family_stats", {})
            if isinstance(ofam, Mapping):
                ordered_ofam = sorted(
                    (
                        (str(k), float(v.get("abs_z_mean", 0.0)))
                        for k, v in ofam.items()
                        if isinstance(v, Mapping)
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
                out.extend([name for name, score in ordered_ofam[:2] if score >= float(self.config.orca_z_threshold)])

        if not out:
            out.append("mixed")
        return out


__all__ = [
    "SemanticTagger",
    "SemanticTaggingResult",
    "FunctionalGroupRule",
    "SmartsRxRule",
]
