#!/usr/bin/env python3
"""
Run ORCA quantum calculations for conformers stored in an SDF and export descriptors.

This script is designed to produce a table directly consumable by Chem-ACE:
  - one row per conformer
  - stable keys: `conf_id` and `ID` (molecule id)
  - numeric descriptor columns suitable for semantic tagging

Typical use:
  python3 orca_from_sdf.py \
    --sdf /path/conformers.sdf \
    --out_csv /path/orca_descriptors.csv \
    --work_dir /path/orca_work \
    --orca_bin /path/to/orca \
    --jobs 8 \
    --orca_nprocs 3 \
    --orca_maxcore_mb 4000 \
    --run_triplets \
    --nroots_singlet 24 \
    --nroots_triplet 24
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha1
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("orca_from_sdf")


# Baseline descriptor names from docs/quantum_descriptors_list.pdf
# (produced by calc_quantum_descriptors.py). We can drop ORCA columns that
# overlap these to avoid duplicating already-available QM features.
_KNOWN_QM_BASELINE_NAMES = {
    "energy_hartree",
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "mu_ev",
    "eta_ev",
    "softness_1_per_ev",
    "chi_ev",
    "omega_ev",
    "dipole_d",
    "quad_norm_au",
    "quad_trace_au",
    "bo_sum",
    "bo_max",
    "bo_mean_bonds",
    "bo_conj_sum",
    "bo_conj_mean",
    "q_min",
    "q_max",
    "q_mean",
    "q_std",
    "q_abs_sum",
    "q_range",
    "q_pos_top3_mean",
    "q_neg_top3_mean",
    "q_abs_r_mean",
    "q_abs_r2_rms",
    "d_pos_neg",
    "vip_ev",
    "vea_ev",
    "fplus_max",
    "fplus_sum_pos",
    "fplus_top3_mean",
    "fminus_max",
    "fminus_sum_pos",
    "fminus_top3_mean",
}

# Alias groups map common naming variants to one canonical token.
_DEDUP_ALIAS_GROUPS = [
    {"homo_ev", "homo"},
    {"lumo_ev", "lumo"},
    {"gap_ev", "homo_lumo_gap_ev", "homo_lumo_gap"},
    {"dipole_d", "dipole_debye", "dipole"},
    {"energy_hartree", "energy"},
    {"q_pos_top3_mean", "q_pos_top_3_mean"},
    {"q_neg_top3_mean", "q_neg_top_3_mean"},
    {"softness_1_per_ev", "softness"},
    {"vip_ev", "ip_ev"},
    {"vea_ev", "ea_ev"},
]


@dataclass(frozen=True)
class JobSpec:
    """One conformer-level ORCA workload."""

    conf_id: str
    mol_id: str
    sdf_index: int
    natoms: int
    charge: int
    multiplicity: int
    work_dir: str
    singlet_inp: str
    singlet_log: str
    triplet_inp: Optional[str] = None
    triplet_log: Optional[str] = None


@dataclass(frozen=True)
class RunConfig:
    """ORCA execution and parser settings."""

    orca_bin: str
    timeout_s: int
    fosc_bright_threshold: float
    max_state_columns: int
    resume: bool
    force_rerun: bool
    keep_failed_logs: bool


@dataclass(frozen=True)
class OrcaRunResult:
    """One ORCA execution result."""

    return_code: int
    elapsed_s: float
    terminated_normally: bool
    status: str


_RX_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"
_RE_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+(" + _RX_FLOAT + r")")
_RE_DIPOLE_MAG_DEBYE = re.compile(r"Magnitude\s*\(Debye\)\s*:\s*(" + _RX_FLOAT + r")", re.IGNORECASE)
_RE_DIPOLE_VEC = re.compile(
    r"Total Dipole Moment\s*:\s*(" + _RX_FLOAT + r")\s+(" + _RX_FLOAT + r")\s+(" + _RX_FLOAT + r")",
    re.IGNORECASE,
)
_RE_ORBITAL_LINE = re.compile(
    r"^\s*\d+\s+(" + _RX_FLOAT + r")\s+(" + _RX_FLOAT + r")\s+(" + _RX_FLOAT + r")\s*$",
    re.MULTILINE,
)
_RE_STATE_LINE = re.compile(
    r"STATE\s+(\d+)\s*:.*?(" + _RX_FLOAT + r")\s+eV\s+(" + _RX_FLOAT + r")\s+nm(?:.*?f\s*=\s*(" + _RX_FLOAT + r"))?",
    re.IGNORECASE,
)
_RE_STATE_FALLBACK = re.compile(
    r"^\s*(\d+)\s+(" + _RX_FLOAT + r")\s+eV\s+(" + _RX_FLOAT + r")\s+nm(?:.*?f\s*=\s*(" + _RX_FLOAT + r"))?",
    re.IGNORECASE | re.MULTILINE,
)
_RE_MULLIKEN = re.compile(r"^\s*\d+\s+[A-Za-z]{1,3}\s*:\s*(" + _RX_FLOAT + r")", re.MULTILINE)
_RE_RADIATIVE = re.compile(r"radiative\s+rate[^:=]*[:=]\s*(" + _RX_FLOAT + r")", re.IGNORECASE)
_RE_NONRADIATIVE = re.compile(r"(?:non[-\s]?radiative|internal conversion)\s+rate[^:=]*[:=]\s*(" + _RX_FLOAT + r")", re.IGNORECASE)
_RE_SOC = re.compile(r"spin[-\s]?orbit[^:=]*[:=]\s*(" + _RX_FLOAT + r")", re.IGNORECASE)
_RE_CT = re.compile(r"(?:charge[-\s]?transfer|ct(?:\s+character)?)\s*[:=]\s*(" + _RX_FLOAT + r")", re.IGNORECASE)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if bool(verbose) else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _safe_text(value: Any) -> str:
    return str(value).strip()


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    return float(v) if math.isfinite(v) else float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _slug(text: str, max_len: int = 32) -> str:
    t = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text).strip())
    t = re.sub(r"_+", "_", t).strip("._-")
    if len(t) == 0:
        t = "item"
    return t[:max_len]


def _get_prop(mol: Any, name: str) -> str:
    n = str(name)
    if n == "_Name":
        try:
            return _safe_text(mol.GetProp("_Name")) if mol.HasProp("_Name") else ""
        except Exception:
            return ""
    try:
        if mol.HasProp(n):
            return _safe_text(mol.GetProp(n))
    except Exception:
        return ""
    return ""


def _build_xyz_block(mol: Any) -> Optional[Tuple[List[str], int]]:
    if mol is None:
        return None
    if int(mol.GetNumConformers()) <= 0:
        return None
    conf = mol.GetConformer(0)
    lines: List[str] = []
    natoms = int(mol.GetNumAtoms())
    for idx in range(natoms):
        atom = mol.GetAtomWithIdx(int(idx))
        pos = conf.GetAtomPosition(int(idx))
        lines.append(f"{atom.GetSymbol():<3s} {pos.x: .10f} {pos.y: .10f} {pos.z: .10f}")
    return lines, natoms


def _determine_charge(mol: Any, charge_prop: str, default_charge: int) -> int:
    val = _get_prop(mol, charge_prop) if charge_prop else ""
    if len(val) > 0:
        return _to_int(val, default_charge)
    try:
        return int(sum(int(mol.GetAtomWithIdx(i).GetFormalCharge()) for i in range(int(mol.GetNumAtoms()))))
    except Exception:
        return int(default_charge)


def _determine_multiplicity(mol: Any, multiplicity_prop: str, default_multiplicity: int) -> int:
    val = _get_prop(mol, multiplicity_prop) if multiplicity_prop else ""
    if len(val) > 0:
        return max(1, _to_int(val, default_multiplicity))
    return max(1, int(default_multiplicity))


def _norm_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    # Normalize unit/case variants.
    s = s.replace("per_ev", "_per_ev")
    s = s.replace("_ev", "_ev")
    for group in _DEDUP_ALIAS_GROUPS:
        if s in group:
            return sorted(group)[0]
    return s


def _load_existing_descriptor_names(
    *,
    existing_qm_csv: Optional[str],
    existing_nonfeature_cols: Sequence[str],
) -> set[str]:
    out: set[str] = set()
    if existing_qm_csv is None or len(str(existing_qm_csv).strip()) == 0:
        return out
    path = Path(str(existing_qm_csv))
    if not path.exists():
        LOGGER.warning("existing_qm_csv not found, skipping CSV dedupe: %s", str(path))
        return out
    try:
        df0 = pd.read_csv(path, nrows=1)
    except Exception as exc:
        LOGGER.warning("Failed reading existing_qm_csv header: %s error=%r", str(path), exc)
        return out
    skip = {str(x).strip() for x in existing_nonfeature_cols if len(str(x).strip()) > 0}
    for col in df0.columns:
        c = str(col)
        if c in skip:
            continue
        out.add(_norm_name(c))
    return out


def _build_orca_input(
    *,
    keywords: str,
    nprocs: int,
    maxcore_mb: int,
    scf_max_iter: int,
    xyz_lines: Sequence[str],
    charge: int,
    multiplicity: int,
    nroots: int,
    tda: bool,
    triplets: bool,
    extra_block: str,
) -> str:
    parts: List[str] = []
    parts.append(f"! {keywords}".strip())
    parts.append("%pal")
    parts.append(f"  nprocs {int(max(1, nprocs))}")
    parts.append("end")
    parts.append(f"%maxcore {int(max(1, maxcore_mb))}")
    parts.append("%scf")
    parts.append(f"  MaxIter {int(max(50, scf_max_iter))}")
    parts.append("end")
    parts.append("%output")
    parts.append("  Print[P_Mayer] 1")
    parts.append("  Print[P_AtCharges_M] 1")
    parts.append("  Print[P_AtCharges_H] 1")
    parts.append("end")
    if int(nroots) > 0:
        parts.append("%tddft")
        parts.append(f"  nroots {int(nroots)}")
        parts.append(f"  tda {'true' if bool(tda) else 'false'}")
        if bool(triplets):
            parts.append("  triplets true")
        parts.append("end")
    if len(extra_block.strip()) > 0:
        parts.append(extra_block.rstrip("\n"))
    parts.append(f"* xyz {int(charge)} {int(max(1, multiplicity))}")
    parts.extend(list(xyz_lines))
    parts.append("*")
    parts.append("")
    return "\n".join(parts)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tail_contains(path: Path, marker: str, nbytes: int = 250000) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > nbytes:
            fh.seek(-nbytes, 2)
        data = fh.read()
    return marker in data.decode("utf-8", errors="ignore")


def _run_orca(*, inp: Path, out_log: Path, config: RunConfig) -> OrcaRunResult:
    out_log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    status = "ok"
    rc = -1
    try:
        with out_log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [str(config.orca_bin), str(inp)],
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=(None if int(config.timeout_s) <= 0 else int(config.timeout_s)),
            )
        rc = int(proc.returncode)
    except FileNotFoundError:
        status = "orca_not_found"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception:
        status = "exec_error"

    elapsed = float(time.perf_counter() - t0)
    normal = _tail_contains(out_log, "ORCA TERMINATED NORMALLY")
    if status == "ok":
        if (rc != 0) or (not normal):
            status = "failed"
    return OrcaRunResult(return_code=int(rc), elapsed_s=elapsed, terminated_normally=bool(normal), status=str(status))


def _first_match_float(rx: re.Pattern[str], text: str) -> float:
    m = rx.search(text)
    if m is None:
        return float("nan")
    return _to_float(m.group(1))


def _last_match_float(rx: re.Pattern[str], text: str) -> float:
    matches = list(rx.finditer(text))
    if len(matches) == 0:
        return float("nan")
    return _to_float(matches[-1].group(1))


def _parse_orbital_energies(text: str) -> Tuple[float, float, float]:
    rows: List[Tuple[float, float]] = []
    for m in _RE_ORBITAL_LINE.finditer(text):
        occ = _to_float(m.group(1))
        ev = _to_float(m.group(3))
        if math.isfinite(occ) and math.isfinite(ev):
            rows.append((occ, ev))
    if len(rows) == 0:
        return float("nan"), float("nan"), float("nan")

    occupied = [ev for occ, ev in rows if occ > 1e-3]
    virtual = [ev for occ, ev in rows if occ <= 1e-3]
    if len(occupied) == 0 or len(virtual) == 0:
        return float("nan"), float("nan"), float("nan")

    homo = float(max(occupied))
    lumo = float(min(virtual))
    gap = float(lumo - homo)
    return homo, lumo, gap


def _parse_dipole(text: str) -> Tuple[float, float, float, float]:
    mag = _last_match_float(_RE_DIPOLE_MAG_DEBYE, text)
    m = _RE_DIPOLE_VEC.search(text)
    if m is None:
        return float("nan"), float("nan"), float("nan"), float(mag)
    x = _to_float(m.group(1))
    y = _to_float(m.group(2))
    z = _to_float(m.group(3))
    if not math.isfinite(mag):
        if all(math.isfinite(v) for v in (x, y, z)):
            mag = float(math.sqrt(x * x + y * y + z * z))
    return x, y, z, float(mag)


def _parse_excited_states(text: str) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    seen: set[Tuple[int, int]] = set()

    def _add_match(m: re.Match[str]) -> None:
        state_idx = _to_int(m.group(1), 0)
        e_ev = _to_float(m.group(2))
        nm = _to_float(m.group(3))
        fosc = _to_float(m.group(4)) if (m.lastindex is not None and m.lastindex >= 4 and m.group(4) is not None) else float("nan")
        if (not math.isfinite(e_ev)) or (not math.isfinite(nm)):
            return
        key = (int(state_idx), int(round(1e6 * e_ev)))
        if key in seen:
            return
        seen.add(key)
        out.append({"state": float(state_idx), "ev": float(e_ev), "nm": float(nm), "fosc": float(fosc)})

    for m in _RE_STATE_LINE.finditer(text):
        _add_match(m)
    if len(out) == 0:
        for m in _RE_STATE_FALLBACK.finditer(text):
            _add_match(m)

    out.sort(key=lambda r: float(r["ev"]))
    return out


def _parse_mulliken_charge_summary(text: str) -> Dict[str, float]:
    vals = [_to_float(m.group(1)) for m in _RE_MULLIKEN.finditer(text)]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) == 0:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    pos = np.sort(arr[arr > 0.0])[::-1]
    neg = np.sort(arr[arr < 0.0])
    return {
        "q_min": float(np.min(arr)),
        "q_max": float(np.max(arr)),
        "q_mean": float(np.mean(arr)),
        "q_std": float(np.std(arr)),
        "q_abs_sum": float(np.sum(np.abs(arr))),
        "q_range": float(np.max(arr) - np.min(arr)),
        "q_pos_top3_mean": float(np.mean(pos[:3])) if pos.size > 0 else 0.0,
        "q_neg_top3_mean": float(np.mean(neg[:3])) if neg.size > 0 else 0.0,
    }


def _summarize_states(
    *,
    states: Sequence[Mapping[str, float]],
    fosc_threshold: float,
    max_state_columns: int,
    prefix: str,
) -> Dict[str, float]:
    if len(states) == 0:
        return {}
    ev = np.asarray([_to_float(s.get("ev")) for s in states], dtype=np.float64)
    nm = np.asarray([_to_float(s.get("nm")) for s in states], dtype=np.float64)
    fosc = np.asarray([_to_float(s.get("fosc")) for s in states], dtype=np.float64)
    finite_f = np.where(np.isfinite(fosc), fosc, 0.0)

    out: Dict[str, float] = {
        f"{prefix}_n_states": float(len(states)),
        f"{prefix}_min_ev": float(np.min(ev)),
        f"{prefix}_max_ev": float(np.max(ev)),
        f"{prefix}_mean_ev": float(np.mean(ev)),
        f"{prefix}_s1_ev": float(ev[0]),
        f"{prefix}_s1_nm": float(nm[0]),
    }

    if np.isfinite(fosc[0]):
        out[f"fosc_s1"] = float(fosc[0])

    out[f"fosc_max"] = float(np.max(finite_f))
    out[f"fosc_mean"] = float(np.mean(finite_f))
    out[f"fosc_sum"] = float(np.sum(finite_f))
    if float(np.sum(finite_f)) > 1e-12:
        out[f"{prefix}_weighted_by_fosc_ev"] = float(np.sum(ev * finite_f) / np.sum(finite_f))

    bright_idx = np.where(finite_f >= float(fosc_threshold))[0]
    out[f"{prefix}_bright_count"] = float(bright_idx.size)
    if bright_idx.size > 0:
        i0 = int(bright_idx[np.argmin(ev[bright_idx])])
        out[f"{prefix}_bright_min_ev"] = float(ev[i0])
        out[f"{prefix}_bright_min_nm"] = float(nm[i0])
        out["lambda_onset_nm"] = float(np.max(nm[bright_idx]))

    imax = int(np.argmax(finite_f))
    out["lambda_max_nm"] = float(nm[imax])
    out["lambda_max_fosc"] = float(finite_f[imax])

    # Bands for optics-oriented explainability.
    def _sum_band(lo_nm: float, hi_nm: float) -> float:
        mask = (nm >= float(lo_nm)) & (nm < float(hi_nm))
        if not bool(np.any(mask)):
            return 0.0
        return float(np.sum(finite_f[mask]))

    out["fosc_uv_200_400"] = _sum_band(200.0, 400.0)
    out["fosc_vis_400_700"] = _sum_band(400.0, 700.0)
    out["fosc_nir_700_1200"] = _sum_band(700.0, 1200.0)
    out["fosc_blue_400_500"] = _sum_band(400.0, 500.0)
    out["fosc_green_500_600"] = _sum_band(500.0, 600.0)
    out["fosc_red_600_700"] = _sum_band(600.0, 700.0)

    # Keep first N state-wise values as explicit columns for richer downstream analysis.
    k = int(max(0, max_state_columns))
    if k > 0:
        top_n = min(k, len(states))
        for i in range(top_n):
            idx = i + 1
            state = states[i]
            out[f"{prefix}_state_{idx:02d}_ev"] = _to_float(state.get("ev"))
            out[f"{prefix}_state_{idx:02d}_nm"] = _to_float(state.get("nm"))
            out[f"fosc_state_{idx:02d}"] = _to_float(state.get("fosc"))
    return out


def _parse_orca_output(path: Path, fosc_bright_threshold: float, max_state_columns: int) -> Dict[str, float]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    out: Dict[str, float] = {}
    out["energy_hartree"] = _last_match_float(_RE_ENERGY, text)

    dip_x, dip_y, dip_z, dip_mag = _parse_dipole(text)
    out["dipole_x_debye"] = float(dip_x)
    out["dipole_y_debye"] = float(dip_y)
    out["dipole_z_debye"] = float(dip_z)
    out["dipole_debye"] = float(dip_mag)

    homo, lumo, gap = _parse_orbital_energies(text)
    out["homo_ev"] = float(homo)
    out["lumo_ev"] = float(lumo)
    out["homo_lumo_gap_ev"] = float(gap)

    out.update(_parse_mulliken_charge_summary(text))

    states = _parse_excited_states(text)
    out.update(
        _summarize_states(
            states=states,
            fosc_threshold=float(fosc_bright_threshold),
            max_state_columns=int(max_state_columns),
            prefix="exc",
        )
    )

    out["kr_s1"] = _first_match_float(_RE_RADIATIVE, text)
    out["knr_s1"] = _first_match_float(_RE_NONRADIATIVE, text)
    out["soc_cm1"] = _first_match_float(_RE_SOC, text)
    out["ct_exc_index"] = _first_match_float(_RE_CT, text)
    return out


def _merge_triplet_info(
    *,
    row: Dict[str, Any],
    triplet_desc: Mapping[str, float],
) -> None:
    if len(triplet_desc) == 0:
        return
    # Parse first triplet state as T1 if available.
    t1_ev = _to_float(triplet_desc.get("exc_s1_ev"))
    t1_nm = _to_float(triplet_desc.get("exc_s1_nm"))
    if math.isfinite(t1_ev):
        row["triplet_t1_ev"] = float(t1_ev)
    if math.isfinite(t1_nm):
        row["triplet_t1_nm"] = float(t1_nm)

    s1_ev = _to_float(row.get("exc_s1_ev"))
    if math.isfinite(s1_ev) and math.isfinite(t1_ev):
        row["s1_t1_gap_ev"] = float(s1_ev - t1_ev)

    # Keep triplet aggregates with clear prefix.
    for key, value in triplet_desc.items():
        if str(key).startswith("exc_"):
            row[f"triplet_{key}"] = value


def _run_one_job(job: JobSpec, run_cfg: RunConfig) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "conf_id": str(job.conf_id),
        "ID": str(job.mol_id),
        "sdf_index": int(job.sdf_index),
        "natoms": int(job.natoms),
        "charge": int(job.charge),
        "multiplicity": int(job.multiplicity),
        "orca_singlet_log": str(job.singlet_log),
        "orca_triplet_log": (None if job.triplet_log is None else str(job.triplet_log)),
    }

    singlet_log = Path(job.singlet_log)
    run_singlet = True
    if bool(run_cfg.resume) and (not bool(run_cfg.force_rerun)) and singlet_log.exists():
        run_singlet = not _tail_contains(singlet_log, "ORCA TERMINATED NORMALLY")

    if run_singlet:
        singlet_res = _run_orca(inp=Path(job.singlet_inp), out_log=singlet_log, config=run_cfg)
    else:
        singlet_res = OrcaRunResult(
            return_code=0,
            elapsed_s=0.0,
            terminated_normally=bool(_tail_contains(singlet_log, "ORCA TERMINATED NORMALLY")),
            status="resumed",
        )

    row["singlet_status"] = str(singlet_res.status)
    row["singlet_return_code"] = int(singlet_res.return_code)
    row["singlet_elapsed_s"] = float(singlet_res.elapsed_s)
    row["singlet_terminated_normally"] = int(bool(singlet_res.terminated_normally))

    singlet_desc = _parse_orca_output(
        singlet_log,
        run_cfg.fosc_bright_threshold,
        run_cfg.max_state_columns,
    )
    row.update(singlet_desc)

    triplet_ok = True
    if job.triplet_inp is not None and job.triplet_log is not None:
        triplet_log = Path(job.triplet_log)
        run_triplet = True
        if bool(run_cfg.resume) and (not bool(run_cfg.force_rerun)) and triplet_log.exists():
            run_triplet = not _tail_contains(triplet_log, "ORCA TERMINATED NORMALLY")

        if run_triplet:
            triplet_res = _run_orca(inp=Path(job.triplet_inp), out_log=triplet_log, config=run_cfg)
        else:
            triplet_res = OrcaRunResult(
                return_code=0,
                elapsed_s=0.0,
                terminated_normally=bool(_tail_contains(triplet_log, "ORCA TERMINATED NORMALLY")),
                status="resumed",
            )

        row["triplet_status"] = str(triplet_res.status)
        row["triplet_return_code"] = int(triplet_res.return_code)
        row["triplet_elapsed_s"] = float(triplet_res.elapsed_s)
        row["triplet_terminated_normally"] = int(bool(triplet_res.terminated_normally))
        triplet_ok = bool(triplet_res.terminated_normally) or ("resumed" == str(triplet_res.status))

        triplet_desc = _parse_orca_output(
            triplet_log,
            run_cfg.fosc_bright_threshold,
            run_cfg.max_state_columns,
        )
        _merge_triplet_info(row=row, triplet_desc=triplet_desc)

    singlet_ok = bool(singlet_res.terminated_normally) or ("resumed" == str(singlet_res.status))
    if singlet_ok and triplet_ok:
        row["job_status"] = "ok"
    elif singlet_ok and (not triplet_ok):
        row["job_status"] = "ok_singlet_triplet_failed"
    else:
        row["job_status"] = "failed"

    if (row["job_status"] != "ok") and (not bool(run_cfg.keep_failed_logs)):
        # Keep artifacts by default in production; optional cleanup only if explicitly requested.
        pass

    return row


def _load_extra_block(path: Optional[str]) -> str:
    if path is None:
        return ""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"extra block file not found: {p}")
    return p.read_text(encoding="utf-8")


def _prepare_jobs(
    *,
    sdf_path: Path,
    work_dir: Path,
    conf_id_prop: str,
    mol_id_prop: str,
    charge_prop: str,
    multiplicity_prop: str,
    default_charge: int,
    default_multiplicity: int,
    orca_keywords: str,
    orca_nprocs: int,
    orca_maxcore_mb: int,
    scf_max_iter: int,
    nroots_singlet: int,
    nroots_triplet: int,
    use_tda: bool,
    run_triplets: bool,
    extra_block: str,
    max_confs: int,
) -> List[JobSpec]:
    try:
        from rdkit import Chem
    except Exception as exc:
        raise RuntimeError("RDKit is required to parse SDF and build ORCA inputs.") from exc

    jobs: List[JobSpec] = []
    failed = 0
    empty_xyz = 0
    missing_conf = 0

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    if supplier is None:
        raise RuntimeError(f"failed to read SDF: {sdf_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    for idx, mol in enumerate(supplier):
        if int(max_confs) > 0 and len(jobs) >= int(max_confs):
            break
        if mol is None:
            failed += 1
            continue

        xyz = _build_xyz_block(mol)
        if xyz is None:
            empty_xyz += 1
            continue
        xyz_lines, natoms = xyz

        conf_id = _get_prop(mol, conf_id_prop)
        if len(conf_id) == 0:
            conf_id = _get_prop(mol, "_Name")
        if len(conf_id) == 0:
            conf_id = f"conf_{idx:08d}"
            missing_conf += 1

        mol_id = _get_prop(mol, mol_id_prop) if len(str(mol_id_prop).strip()) > 0 else ""
        if len(mol_id) == 0:
            mol_id = _get_prop(mol, "ID")
        if len(mol_id) == 0:
            mol_id = conf_id

        charge = _determine_charge(mol, charge_prop=charge_prop, default_charge=default_charge)
        multiplicity = _determine_multiplicity(
            mol,
            multiplicity_prop=multiplicity_prop,
            default_multiplicity=default_multiplicity,
        )

        seed = f"{conf_id}|{mol_id}|{idx}"
        digest = sha1(seed.encode("utf-8")).hexdigest()[:10]
        local_dir = work_dir / f"{idx:08d}_{_slug(conf_id)}_{digest}"
        local_dir.mkdir(parents=True, exist_ok=True)

        singlet_inp = local_dir / "singlet.inp"
        singlet_log = local_dir / "singlet.log"
        singlet_text = _build_orca_input(
            keywords=str(orca_keywords),
            nprocs=int(orca_nprocs),
            maxcore_mb=int(orca_maxcore_mb),
            scf_max_iter=int(scf_max_iter),
            xyz_lines=xyz_lines,
            charge=int(charge),
            multiplicity=int(multiplicity),
            nroots=int(max(0, nroots_singlet)),
            tda=bool(use_tda),
            triplets=False,
            extra_block=extra_block,
        )
        _write_text(singlet_inp, singlet_text)

        triplet_inp: Optional[Path] = None
        triplet_log: Optional[Path] = None
        if bool(run_triplets):
            triplet_inp = local_dir / "triplet.inp"
            triplet_log = local_dir / "triplet.log"
            triplet_text = _build_orca_input(
                keywords=str(orca_keywords),
                nprocs=int(orca_nprocs),
                maxcore_mb=int(orca_maxcore_mb),
                scf_max_iter=int(scf_max_iter),
                xyz_lines=xyz_lines,
                charge=int(charge),
                multiplicity=int(multiplicity),
                nroots=int(max(0, nroots_triplet)),
                tda=bool(use_tda),
                triplets=True,
                extra_block=extra_block,
            )
            _write_text(triplet_inp, triplet_text)

        jobs.append(
            JobSpec(
                conf_id=str(conf_id),
                mol_id=str(mol_id),
                sdf_index=int(idx),
                natoms=int(natoms),
                charge=int(charge),
                multiplicity=int(multiplicity),
                work_dir=str(local_dir),
                singlet_inp=str(singlet_inp),
                singlet_log=str(singlet_log),
                triplet_inp=(None if triplet_inp is None else str(triplet_inp)),
                triplet_log=(None if triplet_log is None else str(triplet_log)),
            )
        )

    LOGGER.info(
        "Prepared ORCA jobs: total=%d parse_failures=%d no_conformer=%d missing_conf_id=%d",
        len(jobs),
        int(failed),
        int(empty_xyz),
        int(missing_conf),
    )
    return jobs


def _normalize_descriptor_table(df: pd.DataFrame) -> pd.DataFrame:
    # Keep metadata first, then deterministic descriptor order.
    meta_cols = [
        "conf_id",
        "ID",
        "sdf_index",
        "natoms",
        "charge",
        "multiplicity",
        "job_status",
        "singlet_status",
        "singlet_return_code",
        "singlet_elapsed_s",
        "singlet_terminated_normally",
        "triplet_status",
        "triplet_return_code",
        "triplet_elapsed_s",
        "triplet_terminated_normally",
        "orca_singlet_log",
        "orca_triplet_log",
    ]
    keep_meta = [c for c in meta_cols if c in df.columns]
    descriptor_cols = sorted([c for c in df.columns if c not in set(keep_meta)])
    return df[keep_meta + descriptor_cols]


def _apply_descriptor_dedupe(
    *,
    df: pd.DataFrame,
    use_known_baseline_dedupe: bool,
    existing_qm_names: set[str],
) -> Tuple[pd.DataFrame, List[str]]:
    meta_cols = {
        "conf_id",
        "ID",
        "sdf_index",
        "natoms",
        "charge",
        "multiplicity",
        "job_status",
        "singlet_status",
        "singlet_return_code",
        "singlet_elapsed_s",
        "singlet_terminated_normally",
        "triplet_status",
        "triplet_return_code",
        "triplet_elapsed_s",
        "triplet_terminated_normally",
        "orca_singlet_log",
        "orca_triplet_log",
    }
    baseline_norm = {_norm_name(x) for x in _KNOWN_QM_BASELINE_NAMES}
    dropped: List[str] = []
    keep_cols: List[str] = []

    for col in df.columns:
        if col in meta_cols:
            keep_cols.append(col)
            continue
        cn = _norm_name(col)
        drop = False
        if bool(use_known_baseline_dedupe) and (cn in baseline_norm):
            drop = True
        if (not drop) and (cn in existing_qm_names):
            drop = True
        if drop:
            dropped.append(str(col))
        else:
            keep_cols.append(col)

    # Keep deterministic order; _normalize_descriptor_table will do final sorting.
    return df[keep_cols].copy(), dropped


def _write_summary(
    *,
    summary_path: Path,
    args: argparse.Namespace,
    n_jobs: int,
    n_ok: int,
    elapsed_s: float,
) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_jobs": int(n_jobs),
        "n_ok": int(n_ok),
        "ok_rate": float(n_ok / max(1, n_jobs)),
        "elapsed_s": float(elapsed_s),
        "args": vars(args),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_all_jobs(
    *,
    jobs: Sequence[JobSpec],
    run_cfg: RunConfig,
    max_workers: int,
    log_every: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = int(len(jobs))
    if total == 0:
        return rows

    done = 0
    t0 = time.perf_counter()
    if int(max_workers) <= 1:
        for job in jobs:
            rows.append(_run_one_job(job, run_cfg))
            done += 1
            if (done % int(max(1, log_every)) == 0) or (done == total):
                dt = max(1e-9, time.perf_counter() - t0)
                rate = done / dt
                eta = (total - done) / max(1e-9, rate)
                LOGGER.info(
                    "ORCA progress: done=%d/%d (%.1f%%) conf_per_s=%.2f eta_s=%.1f",
                    done,
                    total,
                    100.0 * done / max(1, total),
                    rate,
                    eta,
                )
        return rows

    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futures = [ex.submit(_run_one_job, job, run_cfg) for job in jobs]
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if (done % int(max(1, log_every)) == 0) or (done == total):
                dt = max(1e-9, time.perf_counter() - t0)
                rate = done / dt
                eta = (total - done) / max(1e-9, rate)
                LOGGER.info(
                    "ORCA progress: done=%d/%d (%.1f%%) conf_per_s=%.2f eta_s=%.1f",
                    done,
                    total,
                    100.0 * done / max(1, total),
                    rate,
                    eta,
                )
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run ORCA for conformers from SDF and export descriptor CSV for Chem-ACE."
    )
    ap.add_argument("--sdf", required=True, help="Input SDF with conformers (one conformer per record).")
    ap.add_argument("--out_csv", required=True, help="Output descriptor table CSV.")
    ap.add_argument("--out_summary_json", default=None, help="Optional summary JSON path.")
    ap.add_argument("--work_dir", required=True, help="Working directory for ORCA inputs/logs.")
    ap.add_argument("--orca_bin", default="orca", help="ORCA binary path (or executable in PATH).")
    ap.add_argument("--jobs", type=int, default=1, help="Number of concurrent conformer jobs.")
    ap.add_argument("--orca_nprocs", type=int, default=1, help="ORCA %%pal nprocs per conformer job.")
    ap.add_argument("--orca_maxcore_mb", type=int, default=3000, help="ORCA %%maxcore in MB.")
    ap.add_argument("--timeout_s", type=int, default=0, help="Per-ORCA-run timeout (0 means no timeout).")

    ap.add_argument("--conf_id_prop", default="_Name", help="SDF property used as conformer id.")
    ap.add_argument("--mol_id_prop", default="ID", help="SDF property used as molecule id.")
    ap.add_argument("--charge_prop", default="charge", help="SDF property used for total charge.")
    ap.add_argument("--multiplicity_prop", default="multiplicity", help="SDF property used for spin multiplicity.")
    ap.add_argument("--default_charge", type=int, default=0)
    ap.add_argument("--default_multiplicity", type=int, default=1)
    ap.add_argument("--max_confs", type=int, default=0, help="Optional cap on number of conformers to run.")

    ap.add_argument(
        "--orca_keywords",
        default="wB97X-D4 def2-TZVP def2/J RIJCOSX TightSCF",
        help="ORCA ! line keywords.",
    )
    ap.add_argument("--scf_max_iter", type=int, default=500)
    ap.add_argument("--nroots_singlet", type=int, default=24, help="TDDFT singlet roots.")
    ap.add_argument("--run_triplets", action="store_true", help="Run additional triplet TDDFT calculation.")
    ap.add_argument("--nroots_triplet", type=int, default=24, help="TDDFT triplet roots.")
    ap.add_argument("--use_tda", action="store_true", help="Use TDA in TDDFT blocks.")
    ap.add_argument("--extra_block_file", default=None, help="Optional text file appended to ORCA input.")
    ap.add_argument("--fosc_bright_threshold", type=float, default=0.05)
    ap.add_argument(
        "--max_state_columns",
        type=int,
        default=24,
        help="How many low-energy excited states to export as explicit columns.",
    )
    ap.add_argument(
        "--drop_known_qm_duplicates",
        action="store_true",
        help="Drop descriptors that overlap baseline QM list from docs/quantum_descriptors_list.pdf.",
    )
    ap.add_argument(
        "--no_drop_known_qm_duplicates",
        dest="drop_known_qm_duplicates",
        action="store_false",
    )
    ap.set_defaults(drop_known_qm_duplicates=True)
    ap.add_argument(
        "--existing_qm_csv",
        default=None,
        help="Optional existing QM feature CSV; overlapping descriptor names will be dropped.",
    )
    ap.add_argument(
        "--existing_nonfeature_cols",
        nargs="+",
        default=["record_index", "ID", "conf_id", "status", "error", "split"],
        help="Columns in existing_qm_csv to ignore when building dedupe name set.",
    )

    ap.add_argument("--resume", action="store_true", help="Reuse completed logs if present.")
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.set_defaults(resume=True)
    ap.add_argument("--force_rerun", action="store_true", help="Ignore resume and rerun all jobs.")
    ap.add_argument("--keep_failed_logs", action="store_true", help="Keep failed logs (default: keep).")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    _configure_logging(verbose=bool(args.verbose))

    sdf_path = Path(str(args.sdf)).resolve()
    out_csv = Path(str(args.out_csv)).resolve()
    out_summary = (
        Path(str(args.out_summary_json)).resolve()
        if args.out_summary_json is not None
        else out_csv.with_suffix(".summary.json")
    )
    work_dir = Path(str(args.work_dir)).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF not found: {sdf_path}")

    extra_block = _load_extra_block(args.extra_block_file)

    LOGGER.info("Preparing ORCA jobs from SDF: %s", str(sdf_path))
    jobs = _prepare_jobs(
        sdf_path=sdf_path,
        work_dir=work_dir,
        conf_id_prop=str(args.conf_id_prop),
        mol_id_prop=str(args.mol_id_prop),
        charge_prop=str(args.charge_prop),
        multiplicity_prop=str(args.multiplicity_prop),
        default_charge=int(args.default_charge),
        default_multiplicity=int(args.default_multiplicity),
        orca_keywords=str(args.orca_keywords),
        orca_nprocs=int(args.orca_nprocs),
        orca_maxcore_mb=int(args.orca_maxcore_mb),
        scf_max_iter=int(args.scf_max_iter),
        nroots_singlet=int(args.nroots_singlet),
        nroots_triplet=int(args.nroots_triplet),
        use_tda=bool(args.use_tda),
        run_triplets=bool(args.run_triplets),
        extra_block=str(extra_block),
        max_confs=int(args.max_confs),
    )

    if len(jobs) == 0:
        LOGGER.warning("No valid conformers found in SDF. Writing empty CSV.")
        pd.DataFrame(columns=["conf_id", "ID"]).to_csv(out_csv, index=False)
        _write_summary(summary_path=out_summary, args=args, n_jobs=0, n_ok=0, elapsed_s=0.0)
        return

    run_cfg = RunConfig(
        orca_bin=str(args.orca_bin),
        timeout_s=int(args.timeout_s),
        fosc_bright_threshold=float(args.fosc_bright_threshold),
        max_state_columns=int(args.max_state_columns),
        resume=bool(args.resume),
        force_rerun=bool(args.force_rerun),
        keep_failed_logs=bool(args.keep_failed_logs),
    )

    LOGGER.info(
        "Running ORCA: jobs=%d workers=%d orca_nprocs=%d maxcore_mb=%d run_triplets=%s",
        len(jobs),
        int(max(1, args.jobs)),
        int(max(1, args.orca_nprocs)),
        int(max(1, args.orca_maxcore_mb)),
        str(bool(args.run_triplets)).lower(),
    )
    t0 = time.perf_counter()
    rows = _run_all_jobs(
        jobs=jobs,
        run_cfg=run_cfg,
        max_workers=int(max(1, args.jobs)),
        log_every=int(max(1, args.log_every)),
    )
    elapsed_s = float(time.perf_counter() - t0)

    df = pd.DataFrame(rows)
    if "conf_id" not in df.columns:
        df["conf_id"] = ""
    if "ID" not in df.columns:
        df["ID"] = ""

    existing_qm_names = _load_existing_descriptor_names(
        existing_qm_csv=args.existing_qm_csv,
        existing_nonfeature_cols=tuple(str(x) for x in args.existing_nonfeature_cols),
    )
    if bool(args.drop_known_qm_duplicates) or len(existing_qm_names) > 0:
        before_cols = list(df.columns)
        df, dropped_cols = _apply_descriptor_dedupe(
            df=df,
            use_known_baseline_dedupe=bool(args.drop_known_qm_duplicates),
            existing_qm_names=existing_qm_names,
        )
        LOGGER.info(
            "Descriptor dedupe: dropped=%d kept=%d (before=%d, baseline_dedupe=%s, existing_qm_names=%d)",
            int(len(dropped_cols)),
            int(len(df.columns)),
            int(len(before_cols)),
            str(bool(args.drop_known_qm_duplicates)).lower(),
            int(len(existing_qm_names)),
        )
        if len(dropped_cols) > 0:
            dedupe_path = out_csv.with_suffix(".dedupe_dropped.json")
            dedupe_payload = {
                "dropped_columns": [str(x) for x in sorted(dropped_cols)],
                "n_dropped": int(len(dropped_cols)),
                "used_known_qm_baseline": bool(args.drop_known_qm_duplicates),
                "existing_qm_csv": (None if args.existing_qm_csv is None else str(args.existing_qm_csv)),
                "n_existing_qm_names": int(len(existing_qm_names)),
            }
            dedupe_path.write_text(json.dumps(dedupe_payload, indent=2), encoding="utf-8")
            LOGGER.info("Wrote dedupe report: %s", str(dedupe_path))

    df = _normalize_descriptor_table(df)
    df.to_csv(out_csv, index=False)

    n_ok = int((df.get("job_status", pd.Series(dtype=str)).astype(str) == "ok").sum())
    _write_summary(
        summary_path=out_summary,
        args=args,
        n_jobs=int(len(df)),
        n_ok=n_ok,
        elapsed_s=elapsed_s,
    )

    LOGGER.info(
        "Done: rows=%d ok=%d failed=%d out_csv=%s elapsed_s=%.2f",
        len(df),
        n_ok,
        int(len(df) - n_ok),
        str(out_csv),
        elapsed_s,
    )


if __name__ == "__main__":
    main()
