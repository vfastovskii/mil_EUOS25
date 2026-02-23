from __future__ import annotations

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


@_NO_GRAD()
def export_leaderboard_attention(
    model: Any,
    dl_lb_export: Any,
    device: torch.device,
    out_path: Path,
    pred_thresholds: List[float] | None = None,
) -> Path:
    """
    Exports attention weights for leaderboard evaluation to a specified output path.

    This function processes model predictions and per-task attention weights, normalizes
    attention per task over valid conformers, and writes one row per conformer:

    - ID
    - conf_id
    - 4 endpoint predictions (probabilities from logits)
    - 4 endpoint binary labels (thresholded probabilities)
    - 4 attention weights (one per endpoint)

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

        logits, abs_out, fluo_out, attn = model(x2d, x3d, kpm, return_attn=True)
        if attn is None:
            raise RuntimeError("Attention not returned; expected attn when return_attn=True")

        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        probs_np = torch.sigmoid(logits).detach().cpu().numpy()  # [B,4]
        attn_np = attn.detach().cpu().numpy()          # [B,4,N]
        kpm_np = kpm.detach().cpu().numpy().astype(bool)
        B, T, N = attn_np.shape
        if T != len(TASK_COLS):
            raise RuntimeError(f"Unexpected attention task count: {T}, expected {len(TASK_COLS)}")

        for b in range(B):
            mid = str(mol_ids[b])
            valid = ~kpm_np[b]
            L = int(valid.sum())
            if L <= 0:
                continue
            confs = [str(x) for x in conf_pad[b, :L].tolist()]

            # Normalize attention for each task over valid conformers.
            attn_norm = np.zeros((T, L), dtype=np.float64)
            for t in range(T):
                w = attn_np[b, t, :L].astype(np.float64)
                s = float(w.sum())
                if not np.isfinite(s) or s <= 0:
                    w[:] = 1.0 / float(L)
                else:
                    w /= s  # enforce sum-to-1 (safety)
                attn_norm[t, :] = w

            pred_cols = {
                f"pred_{TASK_COLS[t]}": float(probs_np[b, t])
                for t in range(T)
            }
            pred_label_cols = {
                f"pred_label_{TASK_COLS[t]}": int(float(probs_np[b, t]) >= float(pred_thresholds_arr[t]))
                for t in range(T)
            }
            for i in range(L):
                row: Dict[str, Any] = {
                    "ID": mid,
                    "conf_id": confs[i],
                    **pred_cols,
                    **pred_label_cols,
                }
                for t in range(T):
                    row[f"attn_{TASK_COLS[t]}"] = float(attn_norm[t, i])
                rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
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
            key = (str(mol_id), str(conf_id))
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
) -> str:
    label = str(metadata.get("label_auto") or concept_id)
    tags = metadata.get("tags", [])
    if isinstance(tags, (list, tuple)):
        tags_txt = ", ".join([str(x) for x in tags[:2]]) if len(tags) > 0 else "no-tags"
    else:
        tags_txt = "no-tags"
    level = "conf" if is_conf_level else "mol"
    phrase = f"{label} [{tags_txt}] ({level})"
    if float(bridge_score) >= float(bridge_threshold):
        phrase += " bridge-like"
    return phrase


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
    top_k: int = 3,
    bridge_threshold: float = 0.20,
) -> Path:
    """
    Attach concept-aware textual explanations to per-row prediction exports.

    The function links prediction rows (ID, conf_id) with Chem-ACE concept
    semantics and optional Ricci bridge scores from Lambda-Vol artifacts.
    It writes a CSV with additional columns:

    - `top_concepts_<task>`
    - `top_concept_labels_<task>`
    - `prediction_explanation_<task>`
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
        required.add(f"attn_{task}")
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Prediction table missing required columns: {missing}")

    bridge_scores = _compute_ricci_bridge_scores(ricci_edges_csv=ricci_edges_csv, task_cols=task_cols)
    mol_to_concepts, conf_to_concepts = _build_concept_index_maps(
        concept_ids=concept_ids,
        concept_mol_map=concept_mol_map,
        concept_conf_map=concept_conf_map,
    )

    concept_set = {str(x) for x in concept_ids}
    kk = max(1, int(top_k))

    per_task_expl_cols: dict[str, list[str]] = {str(t): [] for t in task_cols}
    per_task_top_ids: dict[str, list[str]] = {str(t): [] for t in task_cols}
    per_task_top_labels: dict[str, list[str]] = {str(t): [] for t in task_cols}
    joined_explanations: list[str] = []

    for row in df.itertuples(index=False):
        mol_id = str(getattr(row, "ID"))
        conf_id = str(getattr(row, "conf_id"))
        conf_hits = conf_to_concepts.get((mol_id, conf_id), set())
        mol_hits = mol_to_concepts.get(mol_id, set())
        active = sorted((conf_hits | mol_hits) & concept_set)

        row_join_parts: list[str] = []
        for task in task_cols:
            t = str(task)
            pred = float(getattr(row, f"pred_{t}"))
            label_col = f"pred_label_{t}"
            label = int(getattr(row, label_col)) if label_col in df.columns else int(pred >= 0.5)
            attn = float(getattr(row, f"attn_{t}"))

            scored: list[tuple[float, str, bool, float]] = []
            for cid in active:
                conf_level = cid in conf_hits
                bridge = float(bridge_scores.get((t, cid), 0.0))
                support = float(concept_metadata.get(cid, {}).get("support", 1.0))
                support_gain = 1.0 + min(0.25, 0.05 * float(np.log1p(max(support, 0.0))))
                base = attn * (1.15 if conf_level else 0.85) * support_gain * (1.0 + 0.35 * bridge)
                scored.append((float(base), cid, conf_level, bridge))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:kk]

            top_ids = [x[1] for x in top]
            top_labels = [str(concept_metadata.get(x[1], {}).get("label_auto") or x[1]) for x in top]

            per_task_top_ids[t].append("|".join(top_ids))
            per_task_top_labels[t].append("|".join(top_labels))

            if len(top) == 0:
                expl = (
                    f"{t}: p={pred:.3f} (label={label}). "
                    "No matched Chem-ACE concept on this conformer; prediction uses global representation."
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
                    for _, cid, is_conf_level, bridge in top
                ]
                has_bridge = any(float(bridge) >= float(bridge_threshold) for _, _, _, bridge in top)
                expl = f"{t}: p={pred:.3f} (label={label}). Key concepts: " + "; ".join(phrases) + "."
                if has_bridge:
                    expl += " Ricci note: bridge-like concept channel detected."

            per_task_expl_cols[t].append(expl)
            row_join_parts.append(expl)

        joined_explanations.append(" | ".join(row_join_parts))

    for task in task_cols:
        t = str(task)
        df[f"top_concepts_{t}"] = per_task_top_ids[t]
        df[f"top_concept_labels_{t}"] = per_task_top_labels[t]
        df[f"prediction_explanation_{t}"] = per_task_expl_cols[t]
    df["prediction_explanation"] = joined_explanations

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[EXPLAIN] wrote {len(df)} rows -> {out_path}")
    return out_path
