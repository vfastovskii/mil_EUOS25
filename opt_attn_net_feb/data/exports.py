from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None  # type: ignore

try:
    from ..utils.constants import TASK_COLS
except Exception:  # pragma: no cover
    from utils.constants import TASK_COLS


def _identity_no_grad():
    def deco(func):
        return func
    return deco


_NO_GRAD = (torch.no_grad if torch is not None else _identity_no_grad)


def _normalize_conf_id(value: Any) -> str:
    """Normalize conf_id string for stable joins across CSV/object dtype conversions."""
    s = str(value).strip()
    if (not s) or (s.lower() in {"nan", "none"}):
        return ""
    if s.endswith(".0"):
        try:
            fv = float(s)
            iv = int(fv)
            if abs(fv - float(iv)) <= 1e-12:
                return str(iv)
        except Exception:
            pass
    return s


def _add_signature_attention_metrics(
    *,
    df: pd.DataFrame,
    signature_cols: Sequence[str],
    task_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Add per-task attention mass/rank diagnostics for conformer signature groups.

    For each signature column and task, computes within-molecule:
      - signature attention mass (sum over conformers sharing the signature)
      - rank of that signature by attention mass
      - top-signature indicator
    """
    out = df.copy()
    if out.empty:
        return out
    if "ID" not in out.columns:
        return out

    for sig_col in signature_cols:
        if sig_col not in out.columns:
            continue
        sig = out[sig_col].astype(str).str.strip()
        valid_mask = sig.ne("") & sig.ne("nan") & sig.ne("None")
        if not bool(valid_mask.any()):
            continue

        for task in task_cols:
            geom_col = f"attn_geom_{str(task)}"
            qm_col = f"attn_qm_{str(task)}"
            have_geom = geom_col in out.columns
            have_qm = qm_col in out.columns
            if (not have_geom) and (not have_qm):
                continue

            cols = ["ID", sig_col]
            if have_geom:
                cols.append(geom_col)
            if have_qm:
                cols.append(qm_col)
            work = out.loc[valid_mask, cols].copy()
            if have_geom and have_qm:
                work["_attn_eff"] = np.maximum(
                    work[geom_col].to_numpy(dtype=np.float64),
                    work[qm_col].to_numpy(dtype=np.float64),
                )
            elif have_geom:
                work["_attn_eff"] = work[geom_col].to_numpy(dtype=np.float64)
            else:
                work["_attn_eff"] = work[qm_col].to_numpy(dtype=np.float64)
            work["_sig_mass"] = (
                work.groupby(["ID", sig_col], sort=False)["_attn_eff"]
                .transform("sum")
                .astype(np.float64)
            )
            work["_sig_rank"] = (
                work.groupby("ID", sort=False)["_sig_mass"]
                .rank(method="dense", ascending=False)
                .astype(np.float64)
            )

            mass_col = f"{sig_col}_mass_{str(task)}"
            rank_col = f"{sig_col}_rank_{str(task)}"
            top_col = f"{sig_col}_top_{str(task)}"
            out[mass_col] = 0.0
            out[rank_col] = np.nan
            out[top_col] = 0

            out.loc[valid_mask, mass_col] = work["_sig_mass"].to_numpy(dtype=np.float64)
            out.loc[valid_mask, rank_col] = work["_sig_rank"].to_numpy(dtype=np.float64)
            out.loc[valid_mask, top_col] = ((work["_sig_rank"] <= 1.0) & (work["_sig_mass"] > 0.0)).astype(np.int32).to_numpy()
    return out


@_NO_GRAD()
def export_leaderboard_attention(
    model: Any,
    dl_lb_export: Any,
    device: torch.device,
    out_path: Path,
    pred_thresholds: List[float] | None = None,
    conf_signature_map: Mapping[str, str] | None = None,
    conf_signature_alt_map: Mapping[str, str] | None = None,
    true_labels_by_id: Mapping[str, Sequence[float]] | None = None,
) -> Path:
    """
    Exports attention weights for leaderboard evaluation to a specified output path.

    This function processes model predictions and per-task attention weights, normalizes
    attention per task over valid conformers, and writes one row per conformer:

    - ID
    - conf_id
    - 4 endpoint predictions (probabilities from logits)
    - 4 endpoint binary labels (thresholded probabilities)
    - 4 endpoint true binary labels (if `true_labels_by_id` provided)
    - 8 attention weights (per endpoint and per 3D modality):
      - `attn_geom_<task>`
      - `attn_qm_<task>`
    - optional fusion gate weights (per endpoint and modality):
      - `fusion_gate_2d_<task>`
      - `fusion_gate_3d_geom_<task>`
      - `fusion_gate_3d_qm_<task>`

    Parameters:
        model: Any
            The PyTorch model that will generate predictions including attention weights.

        dl_lb_export: Any
            Data loader containing the input data for evaluation (e.g., molecule data,
            conformations, etc.).

        device: torch.device
            The device (CPU or GPU) to which the model and input data will be moved during
            processing.

        out_path: Path
            The file path where the attention data will be written. File format is inferred
            from the extension (.parquet or .csv).

        pred_thresholds: List[float] | None
            Optional probability thresholds per task for binary labels. If not provided,
            0.5 is used for all tasks.
        conf_signature_map: Mapping[str, str] | None
            Optional mapping conf_id -> pmapper signature hash.
        conf_signature_alt_map: Mapping[str, str] | None
            Optional mapping conf_id -> alternate/coarser pmapper signature hash.
        true_labels_by_id: Mapping[str, Sequence[float]] | None
            Optional mapping `ID -> [task0..task3]` with true binary labels.
            When provided, exports `true_label_<task>` columns.

    Raises:
        RuntimeError:
            Raised if attention weights are not returned by the model when expected.

    Notes:
        - The function operates without calculating gradients due to the use of @torch.no_grad().
        - Attention weights are normalized to ensure they sum to 1 (with safety checks for
          invalid values).

    Returns:
        Path:
            Actual written output path. If parquet writing is requested but parquet
            engine is unavailable, this will be a CSV fallback path.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for export_leaderboard_attention")

    model.eval()
    model.to(device)

    if pred_thresholds is None:
        pred_thresholds_arr = np.full((len(TASK_COLS),), 0.5, dtype=np.float64)
    else:
        if len(pred_thresholds) != len(TASK_COLS):
            raise ValueError(
                f"pred_thresholds length mismatch: got {len(pred_thresholds)}, "
                f"expected {len(TASK_COLS)}"
            )
        pred_thresholds_arr = np.asarray(pred_thresholds, dtype=np.float64)
        if not np.all(np.isfinite(pred_thresholds_arr)):
            raise ValueError("pred_thresholds must be finite numbers")
        if np.any((pred_thresholds_arr < 0.0) | (pred_thresholds_arr > 1.0)):
            raise ValueError("pred_thresholds must be within [0, 1]")

    rows: List[Dict[str, Any]] = []

    for mol_ids, conf_pad, x2d, x3d, kpm in dl_lb_export:
        x2d = x2d.to(device, non_blocking=True)
        x3d = x3d.to(device, non_blocking=True)
        kpm = kpm.to(device, non_blocking=True)

        logits, abs_out, fluo_out, attn = model(
            x2d,
            x3d,
            kpm,
            return_attn=True,
            return_attn_modalities=True,
        )
        if not isinstance(attn, dict):
            raise RuntimeError("Expected modality attention dict when return_attn_modalities=True")
        attn_geom_t = attn.get("attn_geom")
        attn_qm_t = attn.get("attn_qm")
        modality_gates_t = attn.get("modality_gates")
        modality_order = tuple(str(x) for x in (attn.get("modality_order") or ()))
        if (attn_geom_t is None) and (attn_qm_t is None):
            raise RuntimeError("No modality attention returned; expected attn_geom and/or attn_qm")

        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        probs_np = torch.sigmoid(logits).detach().cpu().numpy()  # [B,4]
        attn_geom_np = (
            None if attn_geom_t is None else attn_geom_t.detach().cpu().numpy()
        )  # [B,4,N]
        attn_qm_np = (
            None if attn_qm_t is None else attn_qm_t.detach().cpu().numpy()
        )  # [B,4,N]
        modality_gates_np = (
            None if modality_gates_t is None else modality_gates_t.detach().cpu().numpy()
        )  # [B,4,M]
        ref = attn_geom_np if attn_geom_np is not None else attn_qm_np
        if ref is None:
            raise RuntimeError("Internal error: missing reference attention tensor")
        kpm_np = kpm.detach().cpu().numpy().astype(bool)
        B, T, N = ref.shape
        if T != len(TASK_COLS):
            raise RuntimeError(f"Unexpected attention task count: {T}, expected {len(TASK_COLS)}")
        if (attn_geom_np is not None) and (attn_geom_np.shape != ref.shape):
            raise RuntimeError("attn_geom shape mismatch")
        if (attn_qm_np is not None) and (attn_qm_np.shape != ref.shape):
            raise RuntimeError("attn_qm shape mismatch")

        for b in range(B):
            mid = str(mol_ids[b])
            valid = ~kpm_np[b]
            L = int(valid.sum())
            if L <= 0:
                continue
            confs = [str(x) for x in conf_pad[b, :L].tolist()]

            # Normalize modality attention for each task over valid conformers.
            attn_geom_norm = np.zeros((T, L), dtype=np.float64)
            attn_qm_norm = np.zeros((T, L), dtype=np.float64)
            for t in range(T):
                if attn_geom_np is not None:
                    wg = attn_geom_np[b, t, :L].astype(np.float64)
                    sg = float(wg.sum())
                    if not np.isfinite(sg) or sg <= 0:
                        wg[:] = 0.0
                    else:
                        wg /= sg
                else:
                    wg = np.zeros((L,), dtype=np.float64)
                if attn_qm_np is not None:
                    wq = attn_qm_np[b, t, :L].astype(np.float64)
                    sq = float(wq.sum())
                    if not np.isfinite(sq) or sq <= 0:
                        wq[:] = 0.0
                    else:
                        wq /= sq
                else:
                    wq = np.zeros((L,), dtype=np.float64)
                attn_geom_norm[t, :] = wg
                attn_qm_norm[t, :] = wq

            pred_cols = {
                f"pred_{TASK_COLS[t]}": float(probs_np[b, t])
                for t in range(T)
            }
            pred_label_cols = {
                f"pred_label_{TASK_COLS[t]}": int(float(probs_np[b, t]) >= float(pred_thresholds_arr[t]))
                for t in range(T)
            }
            true_label_cols: Dict[str, Any] = {}
            if true_labels_by_id is not None:
                y_true_raw = true_labels_by_id.get(mid)
                if y_true_raw is None:
                    true_label_cols = {
                        f"true_label_{TASK_COLS[t]}": np.nan
                        for t in range(T)
                    }
                else:
                    y_true_vec = np.asarray(y_true_raw, dtype=np.float64).reshape(-1)
                    if y_true_vec.shape[0] != T:
                        raise ValueError(
                            f"true_labels_by_id[{mid!r}] has length {int(y_true_vec.shape[0])}, expected {int(T)}"
                        )
                    true_label_cols = {
                        f"true_label_{TASK_COLS[t]}": int(float(y_true_vec[t]) > 0.5)
                        for t in range(T)
                    }
            for i in range(L):
                row: Dict[str, Any] = {
                    "ID": mid,
                    "conf_id": confs[i],
                    **pred_cols,
                    **pred_label_cols,
                    **true_label_cols,
                }
                if (
                    modality_gates_np is not None
                    and len(modality_order) == int(modality_gates_np.shape[2])
                ):
                    for t in range(T):
                        for mi, modality_name in enumerate(modality_order):
                            row[f"fusion_gate_{str(modality_name)}_{TASK_COLS[t]}"] = float(
                                modality_gates_np[b, t, mi]
                            )
                if conf_signature_map is not None:
                    key = _normalize_conf_id(confs[i])
                    row["pmapper_sig_md5"] = str(
                        conf_signature_map.get(key, conf_signature_map.get(str(confs[i]), ""))
                    )
                if conf_signature_alt_map is not None:
                    key = _normalize_conf_id(confs[i])
                    row["pmapper_sig_md5_alt"] = str(
                        conf_signature_alt_map.get(key, conf_signature_alt_map.get(str(confs[i]), ""))
                    )
                for t in range(T):
                    row[f"attn_geom_{TASK_COLS[t]}"] = float(attn_geom_norm[t, i])
                    row[f"attn_qm_{TASK_COLS[t]}"] = float(attn_qm_norm[t, i])
                rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    signature_cols: list[str] = []
    if "pmapper_sig_md5" in df_out.columns:
        signature_cols.append("pmapper_sig_md5")
    if "pmapper_sig_md5_alt" in df_out.columns:
        signature_cols.append("pmapper_sig_md5_alt")
    if len(signature_cols) > 0:
        df_out = _add_signature_attention_metrics(
            df=df_out,
            signature_cols=tuple(signature_cols),
            task_cols=tuple(TASK_COLS),
        )
    if out_path.suffix.lower() in [".parquet", ".pq"]:
        try:
            df_out.to_parquet(out_path, index=False)
        except ImportError as exc:
            # Fallback for environments without optional parquet dependencies.
            fallback_csv = out_path.with_suffix(".csv")
            df_out.to_csv(fallback_csv, index=False)
            print(
                "[ATTN][WARN] parquet engine unavailable; wrote CSV instead: "
                f"{fallback_csv} ({exc})"
            )
            out_path = fallback_csv
    else:
        df_out.to_csv(out_path, index=False)

    print(f"[ATTN] wrote {len(df_out)} rows -> {out_path}")
    return Path(out_path)


def _load_prediction_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception:
            # Export may have fallen back from parquet to CSV.
            csv_fallback = path.with_suffix(".csv")
            if csv_fallback.exists():
                return pd.read_csv(csv_fallback)
            raise
    return pd.read_csv(path)


def _compute_ricci_bridge_scores(
    *,
    ricci_edges_csv: str | None,
    task_cols: Sequence[str],
) -> dict[tuple[str, str], float]:
    if ricci_edges_csv is None:
        return {}
    path = Path(ricci_edges_csv)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    required = {"epoch", "task_id", "concept_src", "concept_dst", "curvature", "weight_flow"}
    if df.empty or (not required.issubset(df.columns)):
        return {}

    out: dict[tuple[str, str], float] = {}
    for task in task_cols:
        dft = df[df["task_id"].astype(str) == str(task)]
        if dft.empty:
            continue
        last_epoch = int(dft["epoch"].max())
        dft = dft[dft["epoch"].astype(int) == last_epoch]
        if dft.empty:
            continue

        sums: dict[str, float] = {}
        counts: dict[str, float] = {}
        for row in dft.itertuples(index=False):
            neg = max(0.0, -float(row.curvature))
            wt = max(0.0, float(row.weight_flow))
            score = neg * max(wt, 1e-6)
            src = str(row.concept_src)
            dst = str(row.concept_dst)
            sums[src] = sums.get(src, 0.0) + score
            sums[dst] = sums.get(dst, 0.0) + score
            counts[src] = counts.get(src, 0.0) + 1.0
            counts[dst] = counts.get(dst, 0.0) + 1.0

        if not sums:
            continue
        mean_scores = {k: (sums[k] / max(1.0, counts.get(k, 1.0))) for k in sums}
        mx = max(mean_scores.values()) if mean_scores else 0.0
        if mx <= 1e-12:
            continue
        for concept_id, val in mean_scores.items():
            out[(str(task), str(concept_id))] = float(val / mx)

    return out


def _compute_tcav_last_scores(
    *,
    lambda_vol_long_csv: str | None,
    task_cols: Sequence[str],
) -> dict[tuple[str, str], float]:
    """
    Load last-epoch TCAV scores from Lambda-Vol long CSV and normalize per task.

    Returns mapping:
      (task_id, concept_id) -> normalized tcav in [0, 1] (positive part only)
    """
    if lambda_vol_long_csv is None:
        return {}
    path = Path(lambda_vol_long_csv)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    required = {"epoch", "task_id", "concept_id"}
    if df.empty or (not required.issubset(df.columns)):
        return {}

    score_col = "tcav_smoothed" if "tcav_smoothed" in df.columns else ("tcav" if "tcav" in df.columns else None)
    if score_col is None:
        return {}

    out: dict[tuple[str, str], float] = {}
    for task in task_cols:
        dft = df[df["task_id"].astype(str) == str(task)]
        if dft.empty:
            continue
        last_epoch = int(dft["epoch"].max())
        dft = dft[dft["epoch"].astype(int) == last_epoch]
        if dft.empty:
            continue

        vals: dict[str, float] = {}
        for row in dft.itertuples(index=False):
            cid = str(getattr(row, "concept_id"))
            val = float(getattr(row, score_col, 0.0))
            vals[cid] = max(0.0, val)

        if not vals:
            continue
        mx = max(vals.values())
        if mx <= 1e-12:
            continue
        for cid, val in vals.items():
            out[(str(task), str(cid))] = float(val / mx)
    return out


def _infer_modality_from_tag_text(tag: str) -> str:
    t = str(tag).strip().lower()
    if len(t) == 0:
        return "2d"
    qm_tokens = (
        "homo",
        "lumo",
        "gap",
        "electrophil",
        "nucleophil",
        "dipole",
        "polariz",
        "electrostatic",
        "fukui",
        "hard electronic",
        "soft electronic",
        "charge-transfer",
        "quantum",
        "frontier",
        "excited-state",
        "oscillator",
        "singlet-triplet",
        "spin-orbit",
        "radiative",
        "nonradiative",
        "orca",
    )
    geom_tokens = (
        "planar",
        "non-planar",
        "twisted",
        "rigid",
        "flexible",
        "geometry",
        "torsion",
        "ring-strained",
        "shape",
        "surface/volume",
        "topology",
        "cavity",
        "inertia",
        "conformer",
        "3d",
    )
    if any(tok in t for tok in qm_tokens):
        return "quantum"
    if any(tok in t for tok in geom_tokens):
        return "geometry"
    return "2d"


def _concept_modality_bundle(metadata: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, list[str]]]:
    weights_raw = metadata.get("modality_weights", {})
    tags_by_modality = metadata.get("tags_by_modality", {})
    out_weights = {"2d": 0.0, "geometry": 0.0, "quantum": 0.0}
    out_tags = {"2d": [], "geometry": [], "quantum": []}

    if isinstance(weights_raw, Mapping):
        for k in out_weights:
            try:
                out_weights[k] = float(weights_raw.get(k, 0.0))
            except Exception:
                out_weights[k] = 0.0

    if isinstance(tags_by_modality, Mapping):
        for k in out_tags:
            v = tags_by_modality.get(k, [])
            if isinstance(v, (list, tuple)):
                out_tags[k] = [str(x) for x in v if str(x).strip()]

    if sum(out_weights.values()) <= 1e-12:
        # Fallback: infer weak modality weights from plain tags.
        tags = metadata.get("tags", [])
        if isinstance(tags, (list, tuple)):
            for tag in tags:
                m = _infer_modality_from_tag_text(str(tag))
                out_weights[m] += 1.0
                out_tags[m].append(str(tag))
    if sum(out_weights.values()) <= 1e-12:
        out_weights["2d"] = 1.0

    total = float(sum(max(0.0, v) for v in out_weights.values()))
    if total > 0.0:
        out_weights = {k: float(max(0.0, v) / total) for k, v in out_weights.items()}

    for k in out_tags:
        out_tags[k] = sorted(set([str(x) for x in out_tags[k] if str(x).strip()]))
    return out_weights, out_tags


def _build_concept_index_maps(
    *,
    concept_ids: Sequence[str],
    concept_mol_map: Mapping[str, set[str]],
    concept_conf_map: Mapping[str, set[tuple[str, str]]],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]]]:
    mol_to_concepts: dict[str, set[str]] = {}
    conf_to_concepts: dict[tuple[str, str], set[str]] = {}

    for cid in concept_ids:
        concept_id = str(cid)
        for mol_id in concept_mol_map.get(concept_id, set()):
            mid = str(mol_id)
            if mid not in mol_to_concepts:
                mol_to_concepts[mid] = set()
            mol_to_concepts[mid].add(concept_id)

        for mol_id, conf_id in concept_conf_map.get(concept_id, set()):
            key = (str(mol_id), _normalize_conf_id(conf_id))
            if key not in conf_to_concepts:
                conf_to_concepts[key] = set()
            conf_to_concepts[key].add(concept_id)

    return mol_to_concepts, conf_to_concepts


def _concept_phrase(
    *,
    concept_id: str,
    metadata: Mapping[str, Any],
    is_conf_level: bool,
    bridge_score: float,
    bridge_threshold: float,
    modality: str | None = None,
) -> str:
    label = str(metadata.get("label_auto") or concept_id)
    tags = _select_display_tags(metadata=metadata, modality=modality, max_tags=3)
    tags_txt = ", ".join([str(x) for x in tags]) if len(tags) > 0 else "no-tags"
    level = "conf" if is_conf_level else "mol"
    phrase = f"{label} [{tags_txt}] ({level})"
    if float(bridge_score) >= float(bridge_threshold):
        phrase += " bridge-like"
    return phrase


def _load_a_priori_tags_map(a_priori_tags_csv: str | None) -> dict[str, list[str]]:
    if a_priori_tags_csv is None:
        return {}
    path = Path(a_priori_tags_csv)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}

    id_col = None
    for c in ("ID", "id", "mol_id"):
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        return {}

    def _split_tags(value: Any) -> list[str]:
        txt = str(value).strip()
        if len(txt) == 0 or txt.lower() in {"nan", "none"}:
            return []
        return [str(x).strip() for x in txt.split(";") if str(x).strip()]

    out: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        mid = str(getattr(row, id_col, "")).strip()
        if len(mid) == 0:
            continue
        tags: list[str] = []
        for col in ("a_priori_tags", "a_priori_functional_tags", "a_priori_smartsrx_tags"):
            if col not in df.columns:
                continue
            tags.extend(_split_tags(getattr(row, col, "")))
        if len(tags) == 0:
            continue
        out[mid] = sorted(set(tags))
    return out


def _load_strict_mixer_scores(
    strict_scores_csv: str | None,
) -> tuple[dict[tuple[str, str, str, str], float], dict[tuple[str, str], float]]:
    """
    Load strict mixer-space concept scores.

    Returns:
      - row-level map: (task, mol_id, conf_id, concept_id) -> strict score
      - global map:    (task, concept_id) -> mean strict score
    """
    if strict_scores_csv is None:
        return {}, {}
    path = Path(str(strict_scores_csv))
    if not path.exists():
        return {}, {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, {}
    if df.empty:
        return {}, {}
    need = {"ID", "conf_id", "task", "concept_id", "strict_cosine"}
    if not need.issubset(set(df.columns)):
        return {}, {}

    row_map: dict[tuple[str, str, str, str], float] = {}
    for row in df.itertuples(index=False):
        t = str(getattr(row, "task", "")).strip()
        mid = str(getattr(row, "ID", "")).strip()
        conf = _normalize_conf_id(getattr(row, "conf_id", ""))
        cid = str(getattr(row, "concept_id", "")).strip()
        if not t or not mid or not cid:
            continue
        try:
            score = float(getattr(row, "strict_cosine", 0.0))
        except Exception:
            score = 0.0
        if not np.isfinite(score):
            continue
        key = (t, mid, conf, cid)
        prev = row_map.get(key)
        row_map[key] = float(score) if prev is None else float(max(prev, float(score)))

    if len(row_map) == 0:
        return {}, {}

    agg_sum: dict[tuple[str, str], float] = {}
    agg_cnt: dict[tuple[str, str], int] = {}
    for (t, _mid, _conf, cid), val in row_map.items():
        k = (t, cid)
        agg_sum[k] = float(agg_sum.get(k, 0.0) + float(val))
        agg_cnt[k] = int(agg_cnt.get(k, 0) + 1)
    global_map = {
        k: float(agg_sum[k] / float(max(1, agg_cnt[k])))
        for k in agg_sum.keys()
    }
    return row_map, global_map


def _is_generic_tag(tag: str) -> bool:
    t = str(tag).strip().lower()
    if len(t) == 0:
        return True
    generic_exact = {
        "hbd",
        "hba",
        "hbd/hba pair",
        "aromatic pi-system",
        "planar",
        "non-planar",
        "twisted",
        "rigid",
        "flexible",
        "aliphatic",
        "neutral polar",
        "neutral nonpolar",
        "cationic center",
        "anionic center",
        "geometry_signal_present",
        "quantum_signal_present",
        "no-tags",
    }
    if t in generic_exact:
        return True
    if t.startswith("rx_role_"):
        return True
    if t.endswith("_signal_present"):
        return True
    return False


def _select_display_tags(
    *,
    metadata: Mapping[str, Any],
    modality: str | None,
    max_tags: int,
) -> list[str]:
    desired_mod = None if modality is None else str(modality)
    details = metadata.get("tag_details", [])
    picked: list[str] = []
    if isinstance(details, list):
        rows: list[tuple[float, str]] = []
        for rec in details:
            if not isinstance(rec, Mapping):
                continue
            tag = str(rec.get("tag", "")).strip()
            if len(tag) == 0:
                continue
            if desired_mod is not None:
                mod = str(rec.get("modality", "")).strip()
                if mod != desired_mod:
                    continue
            conf = float(rec.get("confidence", 0.0))
            # Boost specific tags and downweight broad placeholders.
            if _is_generic_tag(tag):
                conf *= 0.35
            else:
                conf *= 1.25
            rows.append((conf, tag))
        rows.sort(key=lambda x: x[0], reverse=True)
        for _, tag in rows:
            if tag not in picked:
                picked.append(tag)
            if len(picked) >= int(max_tags):
                break

    if len(picked) < int(max_tags):
        fallback_tags: list[str] = []
        if desired_mod is None:
            raw = metadata.get("tags", [])
            if isinstance(raw, (list, tuple)):
                fallback_tags = [str(x) for x in raw if str(x).strip()]
        else:
            _w, tags_by_modality = _concept_modality_bundle(metadata)
            raw = tags_by_modality.get(desired_mod, [])
            if isinstance(raw, (list, tuple)):
                fallback_tags = [str(x) for x in raw if str(x).strip()]
            if len(fallback_tags) == 0:
                raw_all = metadata.get("tags", [])
                if isinstance(raw_all, (list, tuple)):
                    fallback_tags = [str(x) for x in raw_all if str(x).strip()]

        # Prefer specific fallback tags first.
        fallback_tags = sorted(
            set(fallback_tags),
            key=lambda x: (int(_is_generic_tag(str(x))), -len(str(x))),
        )
        for tag in fallback_tags:
            if tag not in picked:
                picked.append(str(tag))
            if len(picked) >= int(max_tags):
                break

    return picked[: max(1, int(max_tags))]


def export_prediction_text_explanations(
    *,
    pred_table_path: Path,
    out_path: Path,
    concept_ids: Sequence[str],
    concept_metadata: Mapping[str, Mapping[str, Any]],
    concept_mol_map: Mapping[str, set[str]],
    concept_conf_map: Mapping[str, set[tuple[str, str]]],
    task_cols: Sequence[str] = TASK_COLS,
    ricci_edges_csv: str | None = None,
    lambda_vol_long_csv: str | None = None,
    a_priori_tags_csv: str | None = None,
    strict_scores_csv: str | None = None,
    tcav_weight: float = 0.35,
    strict_weight: float = 0.35,
    top_k: int = 3,
    bridge_threshold: float = 0.20,
) -> Path:
    """
    Attach concept-aware textual explanations to per-row prediction exports.

    The function links prediction rows (ID, conf_id) with Chem-ACE concept
    semantics and optional Lambda-Vol signals (Ricci bridge + last-epoch TCAV).
    It writes a CSV with additional columns:

    - `top_concepts_<task>`
    - `top_concept_labels_<task>`
    - `prediction_explanation_<task>`
    - modality-specific columns per task:
      - `top_concepts_2d_<task>`, `prediction_explanation_2d_<task>`
      - `top_concepts_geom_<task>`, `prediction_explanation_geom_<task>`
      - `top_concepts_qm_<task>`, `prediction_explanation_qm_<task>`
    - `prediction_explanation` (multi-task joined text)
    """
    df = _load_prediction_table(Path(pred_table_path)).copy()
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    required = {"ID", "conf_id"}
    for task in task_cols:
        required.add(f"pred_{task}")
        required.add(f"attn_geom_{task}")
        required.add(f"attn_qm_{task}")
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Prediction table missing required columns: {missing}")

    bridge_scores = _compute_ricci_bridge_scores(ricci_edges_csv=ricci_edges_csv, task_cols=task_cols)
    tcav_scores = _compute_tcav_last_scores(lambda_vol_long_csv=lambda_vol_long_csv, task_cols=task_cols)
    a_priori_tags_map = _load_a_priori_tags_map(a_priori_tags_csv)
    strict_row_scores, strict_global_scores = _load_strict_mixer_scores(strict_scores_csv)
    mol_to_concepts, conf_to_concepts = _build_concept_index_maps(
        concept_ids=concept_ids,
        concept_mol_map=concept_mol_map,
        concept_conf_map=concept_conf_map,
    )

    concept_set = {str(x) for x in concept_ids}
    kk = max(1, int(top_k))
    tcav_weight = float(max(0.0, tcav_weight))
    strict_weight = float(max(0.0, strict_weight))
    scope_molecules = set([str(x) for x in df["ID"].astype(str).tolist()])
    scope_molecules.update([str(x) for x in mol_to_concepts.keys()])
    scope_n = float(max(1, len(scope_molecules)))
    concept_coverage = {
        str(cid): float(concept_metadata.get(str(cid), {}).get("n_molecules_total", 0.0))
        for cid in concept_set
    }

    per_task_expl_cols: dict[str, list[str]] = {str(t): [] for t in task_cols}
    per_task_top_ids: dict[str, list[str]] = {str(t): [] for t in task_cols}
    per_task_top_labels: dict[str, list[str]] = {str(t): [] for t in task_cols}
    per_task_assigned_tags: dict[str, list[str]] = {str(t): [] for t in task_cols}
    modality_map = (("2d", "2d"), ("geometry", "geom"), ("quantum", "qm"))
    per_task_mod_top_ids: dict[str, dict[str, list[str]]] = {
        str(t): {str(col): [] for _, col in modality_map}
        for t in task_cols
    }
    per_task_mod_top_labels: dict[str, dict[str, list[str]]] = {
        str(t): {str(col): [] for _, col in modality_map}
        for t in task_cols
    }
    per_task_mod_expl: dict[str, dict[str, list[str]]] = {
        str(t): {str(col): [] for _, col in modality_map}
        for t in task_cols
    }
    joined_explanations: list[str] = []
    assigned_semantic_tags: list[str] = []

    for row in df.itertuples(index=False):
        mol_id = str(getattr(row, "ID"))
        conf_id = str(getattr(row, "conf_id"))
        conf_id_norm = _normalize_conf_id(conf_id)
        conf_hits = conf_to_concepts.get((mol_id, conf_id_norm), set())
        mol_hits = mol_to_concepts.get(mol_id, set())
        active = sorted((conf_hits | mol_hits) & concept_set)

        row_join_parts: list[str] = []
        for task in task_cols:
            t = str(task)
            pred = float(getattr(row, f"pred_{t}"))
            label_col = f"pred_label_{t}"
            label = int(getattr(row, label_col)) if label_col in df.columns else int(pred >= 0.5)
            attn_geom = float(getattr(row, f"attn_geom_{t}"))
            attn_qm = float(getattr(row, f"attn_qm_{t}"))
            attn = float(max(attn_geom, attn_qm))

            scored: list[tuple[float, str, bool, float, float, float, dict[str, float], dict[str, list[str]]]] = []
            for cid in active:
                conf_level = cid in conf_hits
                bridge = float(bridge_scores.get((t, cid), 0.0))
                tcav = float(tcav_scores.get((t, cid), 0.0))
                strict_val = float(
                    strict_row_scores.get((t, mol_id, conf_id_norm, cid), strict_global_scores.get((t, cid), 0.0))
                )
                strict_val = float(np.clip(strict_val, -1.0, 1.0))
                support = float(concept_metadata.get(cid, {}).get("support", 1.0))
                coverage = float(concept_coverage.get(str(cid), 0.0))
                if coverage <= 0.0:
                    coverage = max(1.0, support)
                modality_weights, modality_tags = _concept_modality_bundle(concept_metadata.get(cid, {}))
                support_gain = 1.0 + min(0.10, 0.02 * float(np.log1p(max(support, 0.0))))
                rarity_idf = float(math.log1p((scope_n + 1.0) / (1.0 + coverage)))
                rarity_gain = float(np.clip(0.75 + 0.35 * rarity_idf, 0.75, 1.40))
                strict_gain = float(np.clip(1.0 + strict_weight * strict_val, 0.25, 2.50))
                base = (
                    attn
                    * (1.15 if conf_level else 0.85)
                    * support_gain
                    * rarity_gain
                    * (1.0 + 0.35 * bridge)
                    * (1.0 + tcav_weight * max(0.0, tcav))
                    * strict_gain
                )
                scored.append(
                    (float(base), cid, conf_level, bridge, tcav, strict_val, modality_weights, modality_tags)
                )

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:kk]

            top_ids = [x[1] for x in top]
            top_labels = [str(concept_metadata.get(x[1], {}).get("label_auto") or x[1]) for x in top]

            per_task_top_ids[t].append("|".join(top_ids))
            per_task_top_labels[t].append("|".join(top_labels))

            task_tags: list[str] = []
            if len(top) > 0:
                for _, cid, _is_conf_level, _bridge, _tcav, _strict_val, _mw, _mt in top:
                    task_tags.extend(
                        _select_display_tags(
                            metadata=concept_metadata.get(cid, {}),
                            modality=None,
                            max_tags=2,
                        )
                    )
            if len(task_tags) == 0:
                task_tags = [str(x) for x in a_priori_tags_map.get(mol_id, [])]
            task_tags = sorted(set([str(x) for x in task_tags if str(x).strip()]))
            per_task_assigned_tags[t].append("|".join(task_tags[:8]))

            if len(top) == 0:
                prior_txt = ""
                if len(task_tags) > 0:
                    prior_txt = " A-priori tags: " + ", ".join(task_tags[:4]) + "."
                expl = (
                    f"{t}: p={pred:.3f} (label={label}). "
                    "No matched Chem-ACE concept on this conformer; prediction uses global representation."
                    + prior_txt
                )
            else:
                phrases = [
                    _concept_phrase(
                        concept_id=cid,
                        metadata=concept_metadata.get(cid, {}),
                        is_conf_level=is_conf_level,
                        bridge_score=bridge,
                        bridge_threshold=bridge_threshold,
                    )
                    for _, cid, is_conf_level, bridge, _, _, _, _ in top
                ]
                has_bridge = any(float(bridge) >= float(bridge_threshold) for _, _, _, bridge, _, _, _, _ in top)
                mean_tcav_top = float(np.mean([float(x[4]) for x in top])) if len(top) > 0 else 0.0
                mean_strict_top = float(np.mean([float(x[5]) for x in top])) if len(top) > 0 else 0.0
                expl = f"{t}: p={pred:.3f} (label={label}). Key concepts: " + "; ".join(phrases) + "."
                if has_bridge:
                    expl += " Ricci note: bridge-like concept channel detected."
                if mean_tcav_top > 0.0:
                    expl += f" TCAV support={mean_tcav_top:.2f}."
                if abs(mean_strict_top) > 1e-6:
                    expl += f" Strict mixer alignment={mean_strict_top:.2f}."

            # Optional conformer pharmacophore-signature diagnostics (from pmapper).
            sig_main = str(getattr(row, "pmapper_sig_md5", "")).strip()
            sig_mass_col = f"pmapper_sig_md5_mass_{t}"
            sig_rank_col = f"pmapper_sig_md5_rank_{t}"
            if sig_main and (sig_mass_col in df.columns):
                try:
                    sig_mass = float(getattr(row, sig_mass_col))
                except Exception:
                    sig_mass = 0.0
                if sig_rank_col in df.columns:
                    try:
                        sig_rank_raw = getattr(row, sig_rank_col)
                        sig_rank = int(round(float(sig_rank_raw))) if np.isfinite(float(sig_rank_raw)) else None
                    except Exception:
                        sig_rank = None
                else:
                    sig_rank = None
                sig_short = str(sig_main)[:12]
                if (sig_rank is not None) and (sig_rank <= 1) and (sig_mass >= 0.35):
                    expl += (
                        f" Pharmacophore-signature note: dominant conformer signature "
                        f"{sig_short} (attention mass={sig_mass:.2f})."
                    )
                elif sig_mass > 0.0:
                    rank_txt = "" if sig_rank is None else f", rank={int(sig_rank)}"
                    expl += (
                        f" Pharmacophore-signature mass={sig_mass:.2f}{rank_txt} "
                        f"(sig={sig_short})."
                    )

            per_task_expl_cols[t].append(expl)
            row_join_parts.append(expl)

            for modality_key, modality_col in modality_map:
                modal_scored: list[tuple[float, str, bool, float, float]] = []
                for base, cid, is_conf_level, bridge, tcav, _strict_val, modality_weights, _modality_tags in scored:
                    w_mod = float(modality_weights.get(modality_key, 0.0))
                    if w_mod <= 0.0:
                        continue
                    modal_scored.append(
                        (float(base * w_mod), cid, is_conf_level, bridge, tcav)
                    )

                modal_scored.sort(key=lambda x: x[0], reverse=True)
                modal_top = modal_scored[:kk]
                modal_top_ids = [x[1] for x in modal_top]
                modal_top_labels = [
                    str(concept_metadata.get(x[1], {}).get("label_auto") or x[1])
                    for x in modal_top
                ]
                per_task_mod_top_ids[t][modality_col].append("|".join(modal_top_ids))
                per_task_mod_top_labels[t][modality_col].append("|".join(modal_top_labels))

                if len(modal_top) == 0:
                    if modality_key == "2d":
                        m_name = "2D SMARTS"
                    elif modality_key == "geometry":
                        m_name = "3D geometry"
                    else:
                        m_name = "3D quantum"
                    m_expl = (
                        f"{t}/{m_name}: no matched concept for this conformer-row."
                    )
                else:
                    m_phrases = [
                        _concept_phrase(
                            concept_id=cid,
                            metadata=concept_metadata.get(cid, {}),
                            is_conf_level=is_conf_level,
                            bridge_score=bridge,
                            bridge_threshold=bridge_threshold,
                            modality=modality_key,
                        )
                        for _, cid, is_conf_level, bridge, _ in modal_top
                    ]
                    m_mean_tcav = float(np.mean([float(x[4]) for x in modal_top])) if len(modal_top) > 0 else 0.0
                    if modality_key == "2d":
                        m_name = "2D SMARTS"
                    elif modality_key == "geometry":
                        m_name = "3D geometry"
                    else:
                        m_name = "3D quantum"
                    m_expl = f"{t}/{m_name}: " + "; ".join(m_phrases) + "."
                    if m_mean_tcav > 0.0:
                        m_expl += f" TCAV={m_mean_tcav:.2f}."
                per_task_mod_expl[t][modality_col].append(m_expl)

            modal_join = [
                per_task_mod_expl[t]["2d"][-1],
                per_task_mod_expl[t]["geom"][-1],
                per_task_mod_expl[t]["qm"][-1],
            ]
            row_join_parts.append(" ".join(modal_join))

        joined_explanations.append(" | ".join(row_join_parts))
        row_tags: list[str] = []
        for t in task_cols:
            row_tags.extend(
                [str(x) for x in str(per_task_assigned_tags[str(t)][-1]).split("|") if str(x).strip()]
            )
        if len(row_tags) == 0:
            row_tags = [str(x) for x in a_priori_tags_map.get(mol_id, [])]
        assigned_semantic_tags.append("|".join(sorted(set(row_tags))[:16]))

    for task in task_cols:
        t = str(task)
        df[f"top_concepts_{t}"] = per_task_top_ids[t]
        df[f"top_concept_labels_{t}"] = per_task_top_labels[t]
        df[f"assigned_semantic_tags_{t}"] = per_task_assigned_tags[t]
        df[f"prediction_explanation_{t}"] = per_task_expl_cols[t]
        df[f"top_concepts_2d_{t}"] = per_task_mod_top_ids[t]["2d"]
        df[f"top_concept_labels_2d_{t}"] = per_task_mod_top_labels[t]["2d"]
        df[f"prediction_explanation_2d_{t}"] = per_task_mod_expl[t]["2d"]
        df[f"top_concepts_geom_{t}"] = per_task_mod_top_ids[t]["geom"]
        df[f"top_concept_labels_geom_{t}"] = per_task_mod_top_labels[t]["geom"]
        df[f"prediction_explanation_geom_{t}"] = per_task_mod_expl[t]["geom"]
        df[f"top_concepts_qm_{t}"] = per_task_mod_top_ids[t]["qm"]
        df[f"top_concept_labels_qm_{t}"] = per_task_mod_top_labels[t]["qm"]
        df[f"prediction_explanation_qm_{t}"] = per_task_mod_expl[t]["qm"]
    df["assigned_semantic_tags"] = assigned_semantic_tags
    df["prediction_explanation"] = joined_explanations

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[EXPLAIN] wrote {len(df)} rows -> {out_path}")
    return out_path
