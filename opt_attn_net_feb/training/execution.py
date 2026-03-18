from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import gc
import json
import shutil

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import Trial

from ..callbacks.concept_rl import ConceptRLControllerCallback, ConceptRLPolicyConfig
from ..data.collate import collate_export, collate_train
from ..data.datasets import MILExportDataset, MILTrainDataset
from ..data.exports import export_leaderboard_attention, export_prediction_text_explanations
from ..explainability.chem_ace.embedding.hooks import LayerActivationHook
from ..utils.constants import TASK_COLS
from ..utils.data_io import align_by_id
from ..utils.ops import (
    apply_standardizer,
    build_bitmask_group_definition,
    build_aux_targets_and_masks,
    build_aux_weights,
    build_sampler_diagnostics_df,
    build_task_weights,
    coerce_binary_labels,
    fit_standardizer,
    make_balanced_batch_sampler,
    make_weighted_sampler,
    pos_weight_per_task,
    set_all_seeds,
)
from ..utils.progress import log_event, log_step
from .builders import DataLoaderBuilder, LoaderConfig, MILModelBuilder
from .configs import HPOConfig
from .explainability_runtime import (
    FinalExplainabilityConfig,
    build_positive_concept_targets_with_report,
    build_lambda_vol_callback,
    make_monitor_loader,
    prepare_chem_ace_bundle,
)
from .loss_config import compute_gamma, compute_lam, compute_posw_clips
from .search_space import search_space
from .trainer import LightningTrainerConfig, LightningTrainerFactory, ModelEvaluator


def _normalize_conf_id(value: Any) -> str:
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


def _load_conformer_signature_maps(
    *,
    signature_csv: str | None,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Load conformer signature maps from Chem-ACE pmapper export.

    Returns:
      (conf_id -> primary_signature, conf_id -> alternate_signature)
    """
    if signature_csv is None:
        return {}, {}
    path = Path(str(signature_csv))
    if not path.exists():
        return {}, {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, {}
    if df.empty or ("conf_id" not in df.columns):
        return {}, {}

    sig_map: Dict[str, str] = {}
    sig_map_alt: Dict[str, str] = {}
    has_main = "pmapper_sig_md5" in df.columns
    has_alt = "pmapper_sig_md5_alt" in df.columns
    for row in df.itertuples(index=False):
        conf_id = _normalize_conf_id(getattr(row, "conf_id", ""))
        if not conf_id:
            continue
        if has_main:
            sig = str(getattr(row, "pmapper_sig_md5", "")).strip()
            if sig:
                sig_map[conf_id] = sig
        if has_alt:
            sig_alt = str(getattr(row, "pmapper_sig_md5_alt", "")).strip()
            if sig_alt:
                sig_map_alt[conf_id] = sig_alt
    return sig_map, sig_map_alt


def _read_table_auto(path: Path) -> pd.DataFrame:
    suffix = str(path.suffix).lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception:
            fallback = path.with_suffix(".csv")
            if fallback.exists():
                return pd.read_csv(fallback)
            raise
    return pd.read_csv(path)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if (na <= 1e-12) or (nb <= 1e-12):
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _task_vec(z: np.ndarray, task_idx: int) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim >= 2:
        ti = int(np.clip(int(task_idx), 0, max(0, int(arr.shape[0]) - 1)))
        return np.asarray(arr[ti], dtype=np.float32).reshape(-1)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _run_strict_mixer_rerank(
    *,
    model: torch.nn.Module,
    device: torch.device,
    layer_name: str,
    out_dir: Path,
    attention_table_path: Path,
    concept_ids: Sequence[str],
    concept_metadata: Mapping[str, Mapping[str, Any]],
    concept_mol_map: Mapping[str, set[str]],
    concept_conf_map: Mapping[str, set[tuple[str, str]]],
    ids_2d_file: Sequence[str],
    X2d_file: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    id2pos: Mapping[str, int],
    conf_sorted: np.ndarray,
    Xinst_sorted: np.ndarray,
    top_rows_per_task: int,
    batch_size: int,
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Strict re-embedding pass in trained mixer space.

    Scope:
      - concept medoids (from train-discovered concepts)
      - top-attention leaderboard conformers (per task)

    Output:
      - row-level strict scores CSV (ID, conf_id, task, concept_id, strict_cosine)
      - concept-level summary CSV
      - JSON summary
    """
    if not attention_table_path.exists():
        return None, None, None
    df = _read_table_auto(attention_table_path)
    if df.empty:
        return None, None, None
    need_cols = {"ID", "conf_id"}
    for t in TASK_COLS:
        need_cols.add(f"attn_geom_{str(t)}")
        need_cols.add(f"attn_qm_{str(t)}")
    if not need_cols.issubset(set(df.columns)):
        return None, None, None

    # Build leaderboard top-attention subset per task.
    top_k = int(top_rows_per_task)
    top_rows_by_task: Dict[int, List[tuple[str, str, float]]] = {}
    for ti, task in enumerate(TASK_COLS):
        attn_geom_col = f"attn_geom_{str(task)}"
        attn_qm_col = f"attn_qm_{str(task)}"
        sub = df.loc[:, ["ID", "conf_id", attn_geom_col, attn_qm_col]].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan)
        sub["_attn_rank"] = np.maximum(
            sub[attn_geom_col].fillna(0.0).to_numpy(dtype=np.float64),
            sub[attn_qm_col].fillna(0.0).to_numpy(dtype=np.float64),
        )
        sub = sub.dropna(subset=["_attn_rank"])
        if top_k > 0:
            sub = sub.nlargest(int(top_k), columns="_attn_rank")
        rows: List[tuple[str, str, float]] = []
        for row in sub.itertuples(index=False):
            mid = str(getattr(row, "ID", "")).strip()
            conf = _normalize_conf_id(getattr(row, "conf_id", ""))
            if (not mid) or (not conf):
                continue
            try:
                attn_val = float(getattr(row, "_attn_rank"))
            except Exception:
                attn_val = 0.0
            if not np.isfinite(attn_val):
                continue
            rows.append((mid, conf, float(attn_val)))
        top_rows_by_task[int(ti)] = rows

    concept_set = {str(x) for x in concept_ids}
    if len(concept_set) == 0:
        return None, None, None

    # Build concept membership indexes with normalized conformer IDs.
    mol_to_concepts: Dict[str, set[str]] = {}
    conf_to_concepts: Dict[tuple[str, str], set[str]] = {}
    for cid, mols in concept_mol_map.items():
        c = str(cid)
        if c not in concept_set:
            continue
        for mol_id in mols:
            mid = str(mol_id)
            if not mid:
                continue
            if mid not in mol_to_concepts:
                mol_to_concepts[mid] = set()
            mol_to_concepts[mid].add(c)
    for cid, pairs in concept_conf_map.items():
        c = str(cid)
        if c not in concept_set:
            continue
        for mol_id, conf_id in pairs:
            key = (str(mol_id), _normalize_conf_id(conf_id))
            if (not key[0]) or (not key[1]):
                continue
            if key not in conf_to_concepts:
                conf_to_concepts[key] = set()
            conf_to_concepts[key].add(c)

    # Resolve medoid references from concept metadata.
    medoid_query_by_cid: Dict[str, tuple[str, str]] = {}
    for cid in sorted(concept_set):
        md = concept_metadata.get(str(cid), {})
        mid = str(md.get("medoid_mol_id", "")).strip()
        conf = _normalize_conf_id(md.get("medoid_conf_id", ""))
        if not mid:
            continue
        medoid_query_by_cid[str(cid)] = (mid, conf)
    if len(medoid_query_by_cid) == 0:
        return None, None, None

    query_keys: set[tuple[str, str]] = set()
    for rows in top_rows_by_task.values():
        for mid, conf, _attn in rows:
            query_keys.add((str(mid), str(conf)))
    query_keys.update(set(medoid_query_by_cid.values()))
    if len(query_keys) == 0:
        return None, None, None

    id2d_idx = {str(mid): int(i) for i, mid in enumerate(ids_2d_file)}
    starts_arr = np.asarray(starts)
    counts_arr = np.asarray(counts)
    conf_arr = np.asarray(conf_sorted)
    x2d_all = np.asarray(X2d_file, dtype=np.float32)
    xinst_all = np.asarray(Xinst_sorted, dtype=np.float32)

    mol_cache: Dict[str, tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]] = {}
    resolved: Dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def _load_mol(mid: str) -> Optional[tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]]:
        if mid in mol_cache:
            return mol_cache[mid]
        if mid not in id2d_idx or mid not in id2pos:
            return None
        p = int(id2pos[mid])
        s = int(starts_arr[p])
        c = int(counts_arr[p])
        if c <= 0:
            return None
        bag = xinst_all[s : s + c]
        if bag.ndim != 2 or int(bag.shape[0]) == 0:
            return None
        conf_slice = conf_arr[s : s + c]
        conf_norm = [_normalize_conf_id(x) for x in conf_slice.tolist()]
        conf_idx: Dict[str, int] = {}
        for j, cf in enumerate(conf_norm):
            if cf and cf not in conf_idx:
                conf_idx[cf] = int(j)
        x2d_vec = np.asarray(x2d_all[int(id2d_idx[mid])], dtype=np.float32).reshape(-1)
        cached = (x2d_vec, np.asarray(bag, dtype=np.float32), conf_norm, conf_idx)
        mol_cache[mid] = cached
        return cached

    for mid, conf in sorted(query_keys):
        mol_data = _load_mol(str(mid))
        if mol_data is None:
            continue
        x2d_vec, bag, _conf_norm, conf_idx = mol_data
        if conf and conf in conf_idx:
            inst_vec = np.asarray(bag[int(conf_idx[conf])], dtype=np.float32).reshape(-1)
        else:
            inst_vec = np.asarray(np.mean(bag, axis=0), dtype=np.float32).reshape(-1)
        resolved[(str(mid), str(conf))] = (x2d_vec, inst_vec)

    if len(resolved) == 0:
        return None, None, None

    batch = max(1, int(batch_size))
    query_list: List[tuple[str, str, np.ndarray, np.ndarray]] = [
        (mid, conf, x2d, inst) for (mid, conf), (x2d, inst) in resolved.items()
    ]
    embeddings: Dict[tuple[str, str], np.ndarray] = {}

    model_mode = bool(model.training)
    model.eval()
    with torch.no_grad():
        with LayerActivationHook(model, str(layer_name)) as hook:
            for i in range(0, len(query_list), batch):
                chunk = query_list[i : i + batch]
                x2d_np = np.stack([x[2] for x in chunk], axis=0).astype(np.float32)
                x3d_np = np.stack([x[3] for x in chunk], axis=0).astype(np.float32)[:, None, :]
                kpm_np = np.zeros((x2d_np.shape[0], 1), dtype=bool)
                x2d_t = torch.from_numpy(x2d_np).to(device=device, non_blocking=True)
                x3d_t = torch.from_numpy(x3d_np).to(device=device, non_blocking=True)
                kpm_t = torch.from_numpy(kpm_np).to(device=device, non_blocking=True)
                _ = model(x2d_t, x3d_t, kpm_t, return_attn=False)
                act = hook.last_activation
                if act is None:
                    continue
                act_np = np.asarray(act.detach().cpu().numpy(), dtype=np.float32)
                for j, (mid, conf, _x2d, _inst) in enumerate(chunk):
                    embeddings[(str(mid), str(conf))] = np.asarray(act_np[j], dtype=np.float32)
    if model_mode:
        model.train()

    if len(embeddings) == 0:
        return None, None, None

    rows_out: List[Dict[str, Any]] = []
    agg_sum: Dict[tuple[str, str], float] = {}
    agg_cnt: Dict[tuple[str, str], int] = {}
    agg_max: Dict[tuple[str, str], float] = {}

    for ti, task in enumerate(TASK_COLS):
        rows_task = top_rows_by_task.get(int(ti), [])
        task_name = str(task)
        for mid, conf, attn_val in rows_task:
            z_row = embeddings.get((str(mid), str(conf)))
            if z_row is None:
                continue
            active = set(mol_to_concepts.get(str(mid), set()))
            active.update(conf_to_concepts.get((str(mid), str(conf)), set()))
            active = active & concept_set
            if len(active) == 0:
                continue
            row_vec = _task_vec(z_row, int(ti))
            for cid in sorted(active):
                med_key = medoid_query_by_cid.get(str(cid))
                if med_key is None:
                    continue
                z_med = embeddings.get((str(med_key[0]), str(med_key[1])))
                if z_med is None:
                    continue
                med_vec = _task_vec(z_med, int(ti))
                strict = _cosine_similarity(row_vec, med_vec)
                rows_out.append(
                    {
                        "ID": str(mid),
                        "conf_id": str(conf),
                        "task": task_name,
                        "task_idx": int(ti),
                        "concept_id": str(cid),
                        "strict_cosine": float(strict),
                        "attn": float(attn_val),
                    }
                )
                k = (task_name, str(cid))
                agg_sum[k] = float(agg_sum.get(k, 0.0) + float(strict))
                agg_cnt[k] = int(agg_cnt.get(k, 0) + 1)
                agg_max[k] = float(max(float(agg_max.get(k, -1.0)), float(strict)))

    if len(rows_out) == 0:
        return None, None, None

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / "chem_ace_strict_mixer_scores.csv"
    pd.DataFrame(rows_out).to_csv(rows_csv, index=False)

    summary_rows: List[Dict[str, Any]] = []
    for (task_name, cid), cnt in sorted(agg_cnt.items()):
        summary_rows.append(
            {
                "task": str(task_name),
                "concept_id": str(cid),
                "n_rows": int(cnt),
                "mean_strict_cosine": float(agg_sum[(task_name, cid)] / float(max(1, cnt))),
                "max_strict_cosine": float(agg_max[(task_name, cid)]),
            }
        )
    concept_csv = out_dir / "chem_ace_strict_mixer_concepts.csv"
    pd.DataFrame(summary_rows).to_csv(concept_csv, index=False)

    summary = {
        "layer_name": str(layer_name),
        "top_rows_per_task": int(top_k),
        "batch_size": int(batch),
        "n_queries_requested": int(len(query_keys)),
        "n_queries_embedded": int(len(embeddings)),
        "n_task_rows_scored": int(len(rows_out)),
        "n_task_concepts_scored": int(len(summary_rows)),
        "paths": {
            "strict_scores_csv": str(rows_csv),
            "strict_concepts_csv": str(concept_csv),
        },
    }
    summary_json = out_dir / "chem_ace_strict_mixer_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    return rows_csv, concept_csv, summary_json


def _export_concept_rl_active_assignments(
    *,
    out_csv_path: Path,
    out_summary_path: Path,
    ids_train: Sequence[str],
    y_cls_train: np.ndarray,
    concept_targets: dict[int, tuple[str, ...]],
    concept_mol_map: Dict[str, set[str]],
    concept_conf_map: Dict[str, set[tuple[str, str]]],
) -> tuple[str, str]:
    """Export per-active train sample target-concept assignments used by Concept-RL."""
    ids = [str(x) for x in ids_train]
    y = np.asarray(y_cls_train, dtype=np.float32)
    if y.ndim != 2 or y.shape[0] != len(ids):
        raise ValueError(
            f"Invalid y_cls_train shape for assignment export: shape={tuple(y.shape)} n_ids={len(ids)}"
        )

    conf_hits: Dict[tuple[str, str], int] = {}
    for cid, pairs in concept_conf_map.items():
        c = str(cid)
        for mol_id, _conf_id in pairs:
            key = (c, str(mol_id))
            conf_hits[key] = int(conf_hits.get(key, 0) + 1)

    rows: List[Dict[str, Any]] = []
    summary_tasks: Dict[str, Any] = {}

    for ti, task_name in enumerate(TASK_COLS):
        pos_idx = np.where(y[:, ti] > 0.5)[0]
        target_ids = [str(x) for x in concept_targets.get(int(ti), tuple())]
        hit_pos = 0
        total_target_hits = 0
        total_conf_hits = 0

        for i in pos_idx.tolist():
            mol_id = ids[int(i)]
            hit_concepts: List[str] = []
            conf_hits_for_mol = 0

            for cid in target_ids:
                if mol_id in concept_mol_map.get(cid, set()):
                    hit_concepts.append(str(cid))
                    conf_hits_for_mol += int(conf_hits.get((str(cid), mol_id), 0))

            has_hit = len(hit_concepts) > 0
            if has_hit:
                hit_pos += 1
            total_target_hits += int(len(hit_concepts))
            total_conf_hits += int(conf_hits_for_mol)

            rows.append(
                {
                    "task_idx": int(ti),
                    "task_name": str(task_name),
                    "mol_id": str(mol_id),
                    "is_active": 1,
                    "target_concepts": "|".join(hit_concepts),
                    "n_target_concepts_hit": int(len(hit_concepts)),
                    "n_target_conf_hits": int(conf_hits_for_mol),
                    "has_target_hit": int(1 if has_hit else 0),
                }
            )

        n_pos = int(len(pos_idx))
        summary_tasks[str(task_name)] = {
            "n_active": int(n_pos),
            "n_active_with_target_hit": int(hit_pos),
            "active_hit_coverage": float(hit_pos / float(max(1, n_pos))),
            "n_target_concepts": int(len(target_ids)),
            "avg_target_concepts_hit_per_active": float(total_target_hits / float(max(1, n_pos))),
            "avg_target_conf_hits_per_active": float(total_conf_hits / float(max(1, n_pos))),
            "target_concepts": [str(x) for x in target_ids],
        }

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    summary_payload = {
        "n_train_ids": int(len(ids)),
        "n_rows": int(len(df)),
        "tasks": summary_tasks,
    }
    out_summary_path.write_text(json.dumps(summary_payload, indent=2))
    return str(out_csv_path), str(out_summary_path)


def _resolve_device(accelerator: str) -> torch.device:
    """
    Resolves the device to be used for computation based on the availability of CUDA
    and the specified accelerator type.

    Parameters:
    accelerator (str): The desired accelerator type. It can be 'gpu', 'cuda', or other
                       strings indicating the accelerator preference.

    Returns:
    torch.device: The resolved computation device, either 'cuda' if available and
                  requested, or 'cpu' otherwise.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() and str(accelerator) in ("gpu", "cuda") else "cpu"
    )


@dataclass(frozen=True)
class TrainerSystemConfig:
    """
    A configuration class for defining system parameters for a trainer.

    This class provides a structured way to define and store configuration settings
    for training systems. The configuration settings include values for controlling
    training duration, hardware utilization, and computational precision.

    Attributes:
        max_epochs: int
            The maximum number of epochs allowed for training.
        patience: int
            The number of epochs to wait for improvement before applying early stopping.
        accelerator: str
            The type of hardware accelerator to use for training, such as 'cpu' or 'gpu'.
        devices: int
            The number of devices available for distributed training.
        precision: str
            The precision to use during training, typically 'float32', 'bfloat16', etc.
    """
    max_epochs: int
    patience: int
    accelerator: str
    devices: int
    precision: str


@dataclass(frozen=True)
class CVRunConfig:
    """
    Configuration for a cross-validation run.

    Encapsulates the settings required for executing a single run of a
    cross-validation process, including random seed for reproducibility,
    training system configurations, data loading specifications, and the
    root path for storing checkpoints.

    Attributes:
    seed: Random seed for ensuring reproducibility of experiments. Type: int.
    trainer: Configuration for the training system. Type: TrainerSystemConfig.
    loader: Configuration for the data loader. Type: LoaderConfig.
    ckpt_root: Path to the root directory for storing checkpoints. Type: Path.
    """
    seed: int
    trainer: TrainerSystemConfig
    loader: LoaderConfig
    ckpt_root: Path
    run_tag: str = "mil"


@dataclass(frozen=True)
class StudyConfig:
    """
    Represents configuration for a study.

    This class defines the parameters required to configure and execute a study. It
    includes details such as the output directory, study name, number of trials,
    direction of optimization, random seed, and pruning configuration. The dataclass
    is immutable to ensure the configuration cannot be altered once initialized.

    Attributes:
    outdir (Path): The output directory where study results will be stored.
    study_name (str): The name of the study.
    n_trials (int): The number of trials to execute in the study.
    seed (int): The random seed for reproducibility.
    direction (str, optional): The optimization direction, either "maximize" or
    "minimize". Defaults to "maximize".
    pruner_kind (str, optional): Pruner type, either "percentile" or "median".
        Defaults to "percentile".
    pruner_warmup_steps (int, optional): The number of warmup steps before pruning
        trials. Defaults to 8.
    pruner_startup_trials (int, optional): Number of full trials to run before
        enabling pruning decisions. Defaults to 10.
    pruner_percentile (float, optional): Percentile threshold used by
        PercentilePruner. Lower values are less aggressive. Defaults to 25.0.
    """
    outdir: Path
    study_name: str
    n_trials: int
    seed: int
    direction: str = "maximize"
    pruner_kind: str = "percentile"
    pruner_warmup_steps: int = 8
    pruner_startup_trials: int = 10
    pruner_percentile: float = 25.0


@dataclass(frozen=True)
class FinalTrainConfig:
    """
    Configuration data class for the final training setup.

    Represents the configuration needed for initiating the final training process,
    including parameters for seeding, trainer configuration, data loader setup, and
    optional attention output management.
    """
    seed: int
    trainer: TrainerSystemConfig
    loader: LoaderConfig
    attn_out: str | None = None
    explainability: FinalExplainabilityConfig | None = None


@dataclass(frozen=True)
class MILCVData:
    """
    MILCVData serves as a container for data related to Multiple Instance Learning (MIL)
    with cross-validation support. The class aggregates data components including features,
    labels, weights, IDs, fold information, and metadata for handling MIL-specific datasets.

    The class is designed to hold and organize scaled features, classification and regression
    labels, sample weights, and additional attributes required for MIL analysis. It also
    includes fold-specific information and mappings that facilitate the corresponding
    cross-validation workflows.

    Attributes:
        X2d_scaled (np.ndarray): Scaled 2D feature array.
        y_cls (np.ndarray): Classification labels.
        w_cls (np.ndarray): Weights associated with classification labels.
        y_abs (np.ndarray): Absolute regression labels.
        m_abs (np.ndarray): Mask for absolute regression.
        w_abs (np.ndarray): Weights associated with absolute regression labels.
        y_fluo (np.ndarray): Fluorescence-related regression labels.
        m_fluo (np.ndarray): Mask for fluorescence-related regression.
        w_fluo (np.ndarray): Weights associated with fluorescence-related regression labels.
        ids (List[str]): Instance or observation identifiers.
        folds_info (Sequence[Tuple[np.ndarray, np.ndarray, int]]): Cross-validation fold
            information consisting of training indices, validation indices, and the fold
            integer identifier.
        starts (np.ndarray): Start indices for MIL bag-level data.
        counts (np.ndarray): Instance counts corresponding to MIL bags.
        id2pos (Dict[str, int]): Mapping from instance/ID to positional index.
        Xinst_sorted (np.ndarray): Instance-level feature array sorted for efficient
            processing.
    """
    X2d_scaled: np.ndarray
    y_cls: np.ndarray
    w_cls: np.ndarray
    y_abs: np.ndarray
    m_abs: np.ndarray
    w_abs: np.ndarray
    y_fluo: np.ndarray
    m_fluo: np.ndarray
    w_fluo: np.ndarray
    ids: List[str]
    folds_info: Sequence[Tuple[np.ndarray, np.ndarray, int]]
    starts: np.ndarray
    counts: np.ndarray
    id2pos: Dict[str, int]
    Xinst_sorted: np.ndarray
    inst_geom_dim: int = -1
    inst_qm_dim: int = -1


@dataclass(frozen=True)
class MILFinalData:
    """
    Represents a container for finalized multi-instance learning (MIL) input data.

    This class is designed to organize, store, and maintain information about
    processed data required for multi-instance learning tasks. The attributes
    define the structure of input data and metadata, which are critical for
    handling and processing MIL-specific datasets.

    Attributes:
        df_full (pd.DataFrame): Full dataset DataFrame, including all data points.
        id_col (str): Column name in `df_full` representing unique IDs for
            instances or bags.
        split_col (str): Column name in `df_full` defining data splits (e.g.,
            train/test/validation).
        leaderboard_split (str): Identifier for a specific split designated for
            leaderboard or evaluation purposes.
        X2d_file_ids (List[str]): List of file IDs corresponding to 2D features,
            intended for MIL-specific feature representation.
        X2d_file (np.ndarray): 2D features in the form of a NumPy array, typically
            used for MIL representation.
        starts (np.ndarray): Array of start indices for instances or bags within
            the dataset, mapping to their corresponding positions.
        counts (np.ndarray): Array of counts or lengths indicating how many
            instances are associated with each bag or unique ID.
        id2pos (Dict[str, int]): Dictionary mapping unique IDs to their positions
            within the dataset for fast access and lookup.
        Xinst_sorted (np.ndarray): NumPy array of sorted instance-level features,
            ensuring organized MIL feature representation.
        conf_sorted (np.ndarray): NumPy array of sorted confidences, aligning with
            `Xinst_sorted` for consistent ordering.
    """
    df_full: pd.DataFrame
    id_col: str
    split_col: str
    leaderboard_split: str
    X2d_file_ids: List[str]
    X2d_file: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    id2pos: Dict[str, int]
    Xinst_sorted: np.ndarray
    conf_sorted: np.ndarray
    inst_geom_dim: int = -1
    inst_qm_dim: int = -1
    inst_geom_cols: tuple[str, ...] = ()
    inst_qm_cols: tuple[str, ...] = ()
    starts_raw: np.ndarray | None = None
    counts_raw: np.ndarray | None = None
    id2pos_raw: Dict[str, int] | None = None
    Xinst_sorted_raw: np.ndarray | None = None
    conf_sorted_raw: np.ndarray | None = None
    inst_geom_dim_raw: int = -1
    inst_qm_dim_raw: int = -1
    inst_geom_cols_raw: tuple[str, ...] = ()
    inst_qm_cols_raw: tuple[str, ...] = ()


def drop_ids_without_bags(
    *,
    ids: List[str],
    X2d: np.ndarray,
    df_part: pd.DataFrame,
    id2pos: Dict[str, int],
    id_col: str,
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    Filters out IDs that do not have corresponding bags (positions) in the id2pos mapping.

    This function takes a list of IDs, a 2D array, a DataFrame, and a mapping of IDs to
    positions. It filters the input data to retain only those entries in the list of IDs
    that are present in the id2pos dictionary. If all IDs are present in the id2pos
    mapping, the input data is returned as-is.

    Parameters:
    ids: List[str]
        A list of IDs to be filtered.
    X2d: np.ndarray
        A 2D numpy array corresponding to the provided IDs. The array will be filtered
        based on the IDs that have corresponding positions in the id2pos mapping.
    df_part: pd.DataFrame
        A DataFrame containing data corresponding to the IDs. Rows in the DataFrame
        will be filtered based on the IDs that have corresponding positions in id2pos.
    id2pos: Dict[str, int]
        A dictionary mapping IDs to positions. Only IDs present in this dictionary will
        be retained in the filtered output.
    id_col: str
        The name of the column in the DataFrame that contains the IDs.

    Returns:
    Tuple[List[str], np.ndarray, pd.DataFrame]
        A tuple containing the filtered list of IDs, the filtered 2D numpy array, and
        the filtered DataFrame. Rows and elements in the outputs correspond only to
        IDs that exist in the id2pos mapping.
    """
    mask = np.array([(i in id2pos) for i in ids], dtype=bool)
    if mask.all():
        return ids, X2d, df_part
    df2 = df_part.loc[mask].reset_index(drop=True)
    ids2 = df2[id_col].astype(str).tolist()
    return ids2, X2d[mask], df2


class MILFoldTrainer:
    """
    Manages training over multiple Instance Learning (MIL) data folds.

    This class orchestrates the MIL training process for a given fold of data
    evaluation under a trial-specific hyperparameter optimization (HPO) configuration
    and cross-validation setup. The main goal is to tune the model using relevant
    parameters and obtain optimized scores while evaluating for each data fold.
    The fold-specific results and configurations can subsequently be used for
    model comparison and selection.

    Attributes:
        trial: Specific trial being executed during HPO search.
        hpo_config: Configuration instance containing hyperparameters for the training.
        data: MILCVData instance encapsulating training data and metadata.
        run_config: Configuration for the runtime setup, including training parameters.

    Methods:
        run_fold:
            Executes the training and evaluation for a specific fold of the data.
    """

    def __init__(
        self,
        *,
        trial: Trial,
        hpo_config: HPOConfig,
        data: MILCVData,
        run_config: CVRunConfig,
    ):
        self.trial = trial
        self.hpo_config = hpo_config
        self.data = data
        self.run_config = run_config
        self.loader_builder = DataLoaderBuilder(run_config.loader)
        self.eval_device = _resolve_device(run_config.trainer.accelerator)

    def run_fold(
        self,
        *,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        fold_id: int,
    ) -> Tuple[float, Dict[str, Any]]:
        cfg = self.hpo_config
        run_tag = self._run_tag()
        log_event(
            "START",
            "hpo.fold.run",
            trial=int(self.trial.number),
            fold=int(fold_id),
            n_train=int(len(train_idx)),
            n_val=int(len(val_idx)),
            run_tag=str(run_tag),
        )

        set_all_seeds(int(self.run_config.seed) + 5000 * int(fold_id) + int(self.trial.number))

        log_event("INFO", "hpo.fold.compute_loss_weighting", trial=int(self.trial.number), fold=int(fold_id))
        lam = compute_lam(cfg.loss, y_train=self.data.y_cls[train_idx])
        posw = pos_weight_per_task(
            self.data.y_cls[train_idx],
            clip=compute_posw_clips(cfg.loss),
        )
        gamma_t = compute_gamma(cfg.loss)

        log_event("INFO", "hpo.fold.standardize_aux_targets", trial=int(self.trial.number), fold=int(fold_id))
        mu_abs, sd_abs = fit_standardizer(self.data.y_abs, self.data.m_abs, train_idx)
        mu_f, sd_f = fit_standardizer(self.data.y_fluo, self.data.m_fluo, train_idx)
        y_abs_sc = apply_standardizer(self.data.y_abs, mu_abs, sd_abs)
        y_fluo_sc = apply_standardizer(self.data.y_fluo, mu_f, sd_f)

        w_cls_tr = np.asarray(self.data.w_cls[train_idx], dtype=np.float32).copy()

        ids_tr = [self.data.ids[i] for i in train_idx]
        ids_va = [self.data.ids[i] for i in val_idx]

        log_event(
            "INFO",
            "hpo.fold.build_datasets",
            trial=int(self.trial.number),
            fold=int(fold_id),
            n_train_ids=int(len(ids_tr)),
            n_val_ids=int(len(ids_va)),
            run_tag=str(run_tag),
        )
        ds_tr = MILTrainDataset(
            ids_tr,
            self.data.X2d_scaled[train_idx],
            self.data.y_cls[train_idx],
            w_cls_tr,
            y_abs_sc[train_idx],
            self.data.m_abs[train_idx],
            self.data.w_abs[train_idx],
            y_fluo_sc[train_idx],
            self.data.m_fluo[train_idx],
            self.data.w_fluo[train_idx],
            self.data.starts,
            self.data.counts,
            self.data.id2pos,
            self.data.Xinst_sorted,
            max_instances=0,
            seed=int(self.run_config.seed) + int(fold_id),
        )
        ds_va = MILTrainDataset(
            ids_va,
            self.data.X2d_scaled[val_idx],
            self.data.y_cls[val_idx],
            self.data.w_cls[val_idx],
            y_abs_sc[val_idx],
            self.data.m_abs[val_idx],
            self.data.w_abs[val_idx],
            y_fluo_sc[val_idx],
            self.data.m_fluo[val_idx],
            self.data.w_fluo[val_idx],
            self.data.starts,
            self.data.counts,
            self.data.id2pos,
            self.data.Xinst_sorted,
            max_instances=0,
            seed=int(self.run_config.seed) + 999 + int(fold_id),
        )

        log_event(
            "INFO",
            "hpo.fold.build_dataloaders",
            trial=int(self.trial.number),
            fold=int(fold_id),
            batch_size=int(cfg.runtime.batch_size),
            balanced_sampler=bool(cfg.sampler.use_balanced_batch_sampler),
            run_tag=str(run_tag),
        )
        if bool(cfg.sampler.use_balanced_batch_sampler):
            batch_sampler = make_balanced_batch_sampler(
                self.data.y_cls[train_idx],
                batch_size=int(cfg.runtime.batch_size),
                rare_mult=float(cfg.sampler.rare_oversample_mult),
                rare_target_prev=float(cfg.sampler.rare_target_prev),
                sample_weight_cap=float(cfg.sampler.sample_weight_cap),
                batch_pos_fraction=float(cfg.sampler.batch_pos_fraction),
                min_pos_per_batch=int(cfg.sampler.min_pos_per_batch),
                rare_prev_thr=cfg.sampler.rare_prev_thr,
                seed=int(self.run_config.seed) + 1000 * int(fold_id) + int(self.trial.number),
            )
            dl_tr = self.loader_builder.train_loader(
                ds_tr,
                batch_size=int(cfg.runtime.batch_size),
                batch_sampler=batch_sampler,
                collate_fn=collate_train,
            )
            sampler = batch_sampler
        else:
            sampler = make_weighted_sampler(
                self.data.y_cls[train_idx],
                rare_mult=float(cfg.sampler.rare_oversample_mult),
                rare_target_prev=float(cfg.sampler.rare_target_prev),
                sample_weight_cap=float(cfg.sampler.sample_weight_cap),
                rare_prev_thr=cfg.sampler.rare_prev_thr,
            )
            dl_tr = self.loader_builder.train_loader(
                ds_tr,
                batch_size=int(cfg.runtime.batch_size),
                sampler=sampler,
                collate_fn=collate_train,
            )
        sampler_diag_dir = self.run_config.ckpt_root.parent / "sampler_diagnostics"
        sampler_diag_dir.mkdir(parents=True, exist_ok=True)
        sampler_diag_path = sampler_diag_dir / (
            f"hpo_trial{int(self.trial.number):05d}_fold{int(fold_id):02d}_sampler_diagnostics.csv"
        )
        sampler_diag_df = build_sampler_diagnostics_df(
            y=self.data.y_cls[train_idx],
            batch_size=int(cfg.runtime.batch_size),
            use_balanced_batch_sampler=bool(cfg.sampler.use_balanced_batch_sampler),
            rare_mult=float(cfg.sampler.rare_oversample_mult),
            rare_target_prev=float(cfg.sampler.rare_target_prev),
            sample_weight_cap=float(cfg.sampler.sample_weight_cap),
            batch_pos_fraction=float(cfg.sampler.batch_pos_fraction),
            min_pos_per_batch=int(cfg.sampler.min_pos_per_batch),
            rare_prev_thr=cfg.sampler.rare_prev_thr,
            seed=int(self.run_config.seed) + 1000 * int(fold_id) + int(self.trial.number),
        )
        sampler_diag_df.to_csv(sampler_diag_path, index=False)
        _diag_pick = sampler_diag_df.set_index(["section", "metric"])["value"].to_dict()
        log_event(
            "INFO",
            "hpo.fold.sampler_diagnostics",
            trial=int(self.trial.number),
            fold=int(fold_id),
            path=str(sampler_diag_path),
            sampler_mode=str("balanced_batch" if bool(cfg.sampler.use_balanced_batch_sampler) else "weighted_sampler"),
            mean_pos_per_batch=f"{float(_diag_pick.get(('summary', 'mean_pos_per_batch'), float('nan'))):.3f}",
            duplicate_rate=f"{float(_diag_pick.get(('summary', 'duplicate_rate'), float('nan'))):.6f}",
        )
        dl_va = self.loader_builder.eval_loader(
            ds_va,
            batch_size=min(128, int(cfg.runtime.batch_size)),
            collate_fn=collate_train,
        )

        bitmask_group_top_ids, bitmask_group_class_weight = build_bitmask_group_definition(
            self.data.y_cls[train_idx],
            top_k=int(cfg.loss.bitmask_group_top_k),
            class_weight_alpha=float(cfg.loss.bitmask_group_weight_alpha),
            class_weight_cap=float(cfg.loss.bitmask_group_weight_cap),
        )

        log_event(
            "INFO",
            "hpo.fold.build_model",
            trial=int(self.trial.number),
            fold=int(fold_id),
            run_tag=str(run_tag),
        )
        model = MILModelBuilder.build(
            config=cfg,
            mol_dim=int(self.data.X2d_scaled.shape[1]),
            inst_dim=int(self.data.Xinst_sorted.shape[1]),
            inst_geom_dim=int(self.data.inst_geom_dim),
            inst_qm_dim=int(self.data.inst_qm_dim),
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
            bitmask_group_top_ids=bitmask_group_top_ids,
            bitmask_group_class_weight=bitmask_group_class_weight,
        )

        fold_ckpt_dir = self.run_config.ckpt_root / f"mil_trial{self.trial.number}_fold{fold_id}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer_cfg = LightningTrainerConfig(
            max_epochs=int(self.run_config.trainer.max_epochs),
            patience=int(self.run_config.trainer.patience),
            accelerator=str(self.run_config.trainer.accelerator),
            devices=int(self.run_config.trainer.devices),
            precision=str(self.run_config.trainer.precision),
            accumulate_grad_batches=int(cfg.runtime.accumulate_grad_batches),
            # Save fold-best checkpoint so each trial can persist best-epoch params.
            save_checkpoint=True,
            save_weights_only=True,
        )
        trainer, ckpt_cb = LightningTrainerFactory(trainer_cfg).build(
            ckpt_dir=str(fold_ckpt_dir),
            trial=self.trial,
        )
        with log_step("hpo.fold.fit", trial=int(self.trial.number), fold=int(fold_id), run_tag=str(run_tag)):
            trainer.fit(model, dl_tr, dl_va)

        epochs_trained = int(trainer.current_epoch) + 1

        best_epoch = None
        best_ckpt_path: str | None = None
        if ckpt_cb is not None:
            best_path = ckpt_cb.best_model_path
            if best_path and Path(best_path).exists():
                ckpt = torch.load(best_path, map_location="cpu")
                best_epoch = int(ckpt.get("epoch", -1))
                model.load_state_dict(ckpt["state_dict"], strict=True)
                # Retain fold-best checkpoint under a stable trial artifact path.
                retained_dir = self.run_config.ckpt_root.parent / "hpo_trial_fold_best_ckpts"
                retained_dir.mkdir(parents=True, exist_ok=True)
                retained_path = retained_dir / (
                    f"mil_trial{int(self.trial.number)}_fold{int(fold_id)}_best_epoch{int(best_epoch)}.ckpt"
                )
                shutil.copy2(best_path, retained_path)
                best_ckpt_path = str(retained_path)
                log_event(
                    "INFO",
                    "hpo.fold.best_ckpt_retained",
                    trial=int(self.trial.number),
                    fold=int(fold_id),
                    best_epoch=int(best_epoch),
                    path=str(retained_path),
                )

        evaluator = ModelEvaluator(device=self.eval_device)
        with log_step("hpo.fold.eval", trial=int(self.trial.number), fold=int(fold_id), run_tag=str(run_tag)):
            best_macro, best_aps, best_macro_auc, best_aucs = evaluator.eval_best_epoch(model, dl_va)
        best_min = float(np.min(best_aps))

        objective_mode = str(cfg.objective.mode)
        min_w = float(cfg.objective.min_w)
        if objective_mode in {"macro_pr_auc", "macro_ap"}:
            fold_score = float(best_macro)
        elif objective_mode == "macro_plus_min":
            fold_score = float((1.0 - min_w) * float(best_macro) + min_w * best_min)
        else:
            raise ValueError(f"Unsupported objective_mode={objective_mode}")

        banner = str(run_tag).strip().replace("_", "-").upper()
        if not banner:
            banner = "MIL"
        print(
            f"[{banner}] trial={self.trial.number} fold={fold_id} trained_epochs={epochs_trained} "
            f"best_epoch={best_epoch} best_macro_pr_auc={best_macro:.6f} min_pr_auc={best_min:.6f} "
            f"best_macro_roc_auc={best_macro_auc:.6f} pr_aucs={best_aps} roc_aucs={best_aucs} "
            f"score={fold_score:.6f} mode={cfg.objective.mode} min_w={min_w:.2f}"
        )
        log_event(
            "INFO",
            "hpo.fold.metrics",
            trial=int(self.trial.number),
            fold=int(fold_id),
            score=f"{fold_score:.6f}",
            macro_pr_auc=f"{best_macro:.6f}",
            macro_roc_auc=f"{best_macro_auc:.6f}",
            min_pr_auc=f"{best_min:.6f}",
            pr_auc_t0=f"{float(best_aps[0]):.6f}",
            pr_auc_t1=f"{float(best_aps[1]):.6f}",
            pr_auc_t2=f"{float(best_aps[2]):.6f}",
            pr_auc_t3=f"{float(best_aps[3]):.6f}",
            roc_auc_t0=f"{float(best_aucs[0]):.6f}",
            roc_auc_t1=f"{float(best_aucs[1]):.6f}",
            roc_auc_t2=f"{float(best_aucs[2]):.6f}",
            roc_auc_t3=f"{float(best_aucs[3]):.6f}",
            run_tag=str(run_tag),
        )

        detail = {
            "trained_epochs": epochs_trained,
            "best_epoch": best_epoch,
            "best_ckpt_path": best_ckpt_path,
            "objective_macro_ap": float(best_macro),
            "macro_ap_best_epoch": float(best_macro),
            "macro_pr_auc_best_epoch": float(best_macro),
            "macro_auc_best_epoch": float(best_macro_auc),
            "min_ap_best_epoch": best_min,
            "min_pr_auc_best_epoch": best_min,
            "score": fold_score,
            "objective_mode": str(cfg.objective.mode),
            "min_w": min_w,
            "ap_task0": float(best_aps[0]),
            "ap_task1": float(best_aps[1]),
            "ap_task2": float(best_aps[2]),
            "ap_task3": float(best_aps[3]),
            "pr_auc_task0": float(best_aps[0]),
            "pr_auc_task1": float(best_aps[1]),
            "pr_auc_task2": float(best_aps[2]),
            "pr_auc_task3": float(best_aps[3]),
            "auc_task0": float(best_aucs[0]),
            "auc_task1": float(best_aucs[1]),
            "auc_task2": float(best_aucs[2]),
            "auc_task3": float(best_aucs[3]),
            "accumulate_grad_batches": int(cfg.runtime.accumulate_grad_batches),
        }

        try:
            shutil.rmtree(fold_ckpt_dir, ignore_errors=True)
        except Exception:
            pass

        del trainer, model, dl_tr, dl_va, ds_tr, ds_va, sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        log_event(
            "DONE",
            "hpo.fold.run",
            trial=int(self.trial.number),
            fold=int(fold_id),
            score=f"{fold_score:.6f}",
            run_tag=str(run_tag),
        )
        return fold_score, detail

    def _run_tag(self) -> str:
        raw = str(getattr(self.run_config, "run_tag", "") or "").strip()
        if raw and raw.lower() != "mil":
            return raw
        has_2d = int(self.data.X2d_scaled.shape[1]) > 0
        has_3d = int(max(0, self.data.inst_geom_dim) + max(0, self.data.inst_qm_dim)) > 0
        if has_2d and has_3d:
            return "mt_2d3d"
        if has_2d:
            return "mt_2d"
        if has_3d:
            return "mt_3d"
        return "mil"


def _persist_trial_best_epoch_artifacts(
    *,
    outdir: Path,
    trial: Trial,
    params: Mapping[str, Any],
    fold_detail: Mapping[str, Any],
    mean_score: float,
) -> str | None:
    """
    Persist per-trial best-epoch artifact:
      - selected best fold by fold score,
      - checkpoint path for that fold best epoch,
      - trial params used to produce it.
    """
    if len(fold_detail) == 0:
        return None

    best_fold_id: str | None = None
    best_fold_score = float("-inf")
    best_fold_payload: Mapping[str, Any] | None = None
    for fold_id, payload in fold_detail.items():
        score = float(payload.get("score", float("-inf")))
        if score > best_fold_score:
            best_fold_score = float(score)
            best_fold_id = str(fold_id)
            best_fold_payload = payload

    if best_fold_id is None or best_fold_payload is None:
        return None

    save_dir = Path(outdir) / "hpo_trial_best_epoch_params"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"trial_{int(trial.number):05d}_best_epoch_params.json"

    payload = {
        "trial_number": int(trial.number),
        "objective_value_cv": float(mean_score),
        "best_fold_id": str(best_fold_id),
        "best_fold_objective_value": float(best_fold_score),
        "objective_mode": str(best_fold_payload.get("objective_mode", "unknown")),
        "best_epoch": best_fold_payload.get("best_epoch"),
        "best_ckpt_path": best_fold_payload.get("best_ckpt_path"),
        "params": dict(params),
        "fold_detail": dict(fold_detail),
    }
    out_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda x: (
                x.item() if isinstance(x, np.generic) else (x.tolist() if isinstance(x, np.ndarray) else str(x))
            ),
        )
    )
    return str(out_path)


class MILCrossValidator:
    """
    Provides functionality for multi-instance learning cross-validation.

    This class is designed to handle the cross-validation process for multi-instance
    learning (MIL) tasks. It takes in a dataset and configuration details, and
    performs the necessary evaluation by dividing the dataset into folds, training
    and validating on these folds, and computing performance scores. It integrates
    hyperparameter optimization (HPO) and mode selection for evaluating different
    configurations.

    Attributes:
        data (MILCVData): The dataset and cross-validation folding information.
        run_config (CVRunConfig): Configuration details for running the cross-validation.
    """

    def __init__(self, *, data: MILCVData, run_config: CVRunConfig):
        self.data = data
        self.run_config = run_config

    def evaluate_trial(self, trial: Trial) -> float:
        log_event("START", "hpo.trial.evaluate", trial=int(trial.number))
        params = search_space(trial)
        cfg = HPOConfig.from_params(params)
        if str(cfg.objective.mode) not in {"macro_pr_auc", "macro_ap", "macro_plus_min"}:
            raise ValueError(
                f"Unsupported objective_mode={cfg.objective.mode}; "
                "allowed values are macro_pr_auc, macro_ap, macro_plus_min."
            )

        fold_runner = MILFoldTrainer(
            trial=trial,
            hpo_config=cfg,
            data=self.data,
            run_config=self.run_config,
        )

        scores: List[float] = []
        fold_detail: Dict[str, Any] = {}
        for step, (tr, va, fold_id) in enumerate(self.data.folds_info):
            log_event(
                "INFO",
                "hpo.trial.fold_start",
                trial=int(trial.number),
                cv_step=int(step),
                fold=int(fold_id),
            )
            _fold_score, detail = fold_runner.run_fold(
                train_idx=np.asarray(tr, dtype=np.int64),
                val_idx=np.asarray(va, dtype=np.int64),
                fold_id=int(fold_id),
            )
            objective_mode = str(cfg.objective.mode)
            if objective_mode in {"macro_pr_auc", "macro_ap"}:
                trial_score = float(detail.get("macro_pr_auc_best_epoch", detail.get("macro_ap_best_epoch", 0.0)))
            else:
                trial_score = float(_fold_score)
            scores.append(float(trial_score))
            fold_detail[str(fold_id)] = detail

            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                log_event(
                    "WARN",
                    "hpo.trial.pruned",
                    trial=int(trial.number),
                    cv_step=int(step),
                    mean_score=f"{float(np.mean(scores)):.6f}",
                )
                raise optuna.TrialPruned()

        mean_score = float(np.mean(scores))
        trial_artifact_json = _persist_trial_best_epoch_artifacts(
            outdir=Path(self.run_config.ckpt_root).parent,
            trial=trial,
            params=params,
            fold_detail=fold_detail,
            mean_score=float(mean_score),
        )
        trial.set_user_attr("fold_detail", fold_detail)
        if trial_artifact_json is not None:
            trial.set_user_attr("best_epoch_params_json", str(trial_artifact_json))
            log_event(
                "INFO",
                "hpo.trial.best_epoch_artifact_saved",
                trial=int(trial.number),
                path=str(trial_artifact_json),
            )
        log_event(
            "DONE",
            "hpo.trial.evaluate",
            trial=int(trial.number),
            mean_score=f"{mean_score:.6f}",
        )
        return mean_score


class StudyArtifactsWriter:
    """
    Provides methods for saving study artifacts and metrics.

    This class defines static methods to save Optuna study artifacts such as trials
    information and best parameters in structured formats, along with capabilities
    to save fold-specific metrics in a consistent manner.
    """
    @staticmethod
    def save_study_artifacts(*, outdir: Path, study: optuna.Study, prefix: str) -> None:
        df_trials = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "user_attrs")
        )
        df_trials.to_csv(outdir / f"{prefix}_trials.csv", index=False)
        best = dict(study.best_params)
        best["best_value_macro_ap_cv"] = float(study.best_value)
        (outdir / f"{prefix}_best_params.json").write_text(json.dumps(best, indent=2))

    @staticmethod
    def save_best_fold_metrics(
        *,
        outdir: Path,
        prefix: str,
        fold_metrics: Dict[str, Any],
    ) -> None:
        (outdir / f"{prefix}_best_fold_metrics.json").write_text(
            json.dumps(fold_metrics, indent=2)
        )


class MILStudyRunner:
    """
    MILStudyRunner is a class for managing and executing hyperparameter optimization
    (HPO) studies using the Optuna framework.

    This class is designed to facilitate the creation and execution of machine
    learning cross-validation studies. It handles study storage, sampler and pruner
    configuration, and manages the optimization trials and results storage. Users can
    apply this to systematically tune hyperparameters for machine learning models.

    Attributes:
        config (StudyConfig): Configuration for the study.
        cross_validator (MILCrossValidator): Object that evaluates individual trials.
    """

    def __init__(self, *, config: StudyConfig, cross_validator: MILCrossValidator):
        self.config = config
        self.cross_validator = cross_validator

    def _make_storage(self) -> str:
        return f"sqlite:///{(self.config.outdir / f'{self.config.study_name}.sqlite3').as_posix()}"

    def run(self) -> optuna.Study:
        with log_step(
            "hpo.study.run",
            study_name=str(self.config.study_name),
            n_trials=int(self.config.n_trials),
            pruner=str(self.config.pruner_kind),
        ):
            self.config.outdir.mkdir(parents=True, exist_ok=True)
            sampler = optuna.samplers.TPESampler(seed=int(self.config.seed))
            if str(self.config.pruner_kind).lower() == "median":
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=int(self.config.pruner_startup_trials),
                    n_warmup_steps=int(self.config.pruner_warmup_steps),
                )
            else:
                # Default: less aggressive than median pruning for sparse multitask AP.
                pruner = optuna.pruners.PercentilePruner(
                    percentile=float(self.config.pruner_percentile),
                    n_startup_trials=int(self.config.pruner_startup_trials),
                    n_warmup_steps=int(self.config.pruner_warmup_steps),
                )
            study = optuna.create_study(
                direction=str(self.config.direction),
                sampler=sampler,
                pruner=pruner,
                study_name=str(self.config.study_name),
                storage=self._make_storage(),
                load_if_exists=True,
            )
            log_event("INFO", "hpo.study.optimize.start")
            study.optimize(
                self.cross_validator.evaluate_trial,
                n_trials=int(self.config.n_trials),
                gc_after_trial=True,
                catch=(RuntimeError, ValueError, FloatingPointError),
            )
            log_event("INFO", "hpo.study.optimize.done", best_value=f"{float(study.best_value):.6f}")
            StudyArtifactsWriter.save_study_artifacts(
                outdir=self.config.outdir,
                study=study,
                prefix=self.config.study_name,
            )
            StudyArtifactsWriter.save_best_fold_metrics(
                outdir=self.config.outdir,
                prefix=self.config.study_name,
                fold_metrics=study.best_trial.user_attrs.get("fold_detail", {}),
            )
            print(f"[HPO] best macro AP (CV mean) = {study.best_value:.6f}")
            return study


class MILFinalTrainer:
    """
    Represents a MIL (Multiple Instance Learning) final trainer.

    This class is responsible for configuring, constructing datasets, initializing models,
    and managing the training process for MIL tasks. It uses the provided configurations
    and data to execute a training pipeline, including data preparation, loader creation,
    model building, and trainer setup. The class handles auxiliary target creation,
    standardization, and sampling strategies to improve the robustness of training.
    It ensures resources such as data loaders, model instances, and trainer configurations
    are optimally utilized for evaluation and comparison between training and leaderboard
    datasets.

    Attributes:
        config (FinalTrainConfig): Configuration object containing parameters for training,
            including dataset loader settings, trainer behavior, and other necessary options.
        loader_builder (DataLoaderBuilder): Builder instance used to create train and
            evaluation loaders based on the specified data configuration.
        eval_device: The device resolved for evaluation tasks, based on trainer configuration.

    Methods:
        run(outdir: Path, best_params: Dict[str, Any], data: MILFinalData) -> None:
            Executes the full training pipeline. Prepares datasets, applies auxiliary data
            transformations, builds data loaders, initializes the model, and trains it using
            a configured trainer. Saves the best model checkpoint for further evaluation.
    """

    def __init__(self, *, config: FinalTrainConfig):
        self.config = config
        self.loader_builder = DataLoaderBuilder(config.loader)
        self.eval_device = _resolve_device(config.trainer.accelerator)

    def run(self, *, outdir: Path, best_params: Dict[str, Any], data: MILFinalData) -> None:
        log_event(
            "START",
            "final.run",
            outdir=str(outdir),
            leaderboard_split=str(data.leaderboard_split),
            n_best_params=int(len(best_params)),
        )
        cfg = HPOConfig.from_params(
            best_params,
            fallback_lambda_power=1.0,
            fallback_lam_floor=0.25,
            fallback_lam_ceil=6.0,
            fallback_pos_weight_clip=50.0,
        )

        log_event("INFO", "final.prepare_splits")
        df_tr = data.df_full[data.df_full[data.split_col] == "train"].copy().reset_index(drop=True)
        df_lb = (
            data.df_full[data.df_full[data.split_col] == data.leaderboard_split]
            .copy()
            .reset_index(drop=True)
        )
        if len(df_lb) == 0:
            raise ValueError(f"No rows with split == '{data.leaderboard_split}'")
        log_event("INFO", "final.split_counts", n_train=int(len(df_tr)), n_lb=int(len(df_lb)))

        ids_tr = df_tr[data.id_col].astype(str).tolist()
        ids_lb = df_lb[data.id_col].astype(str).tolist()

        log_event("INFO", "final.align_2d_features")
        X2d_tr = align_by_id(data.X2d_file_ids, data.X2d_file, ids_tr)
        X2d_lb = align_by_id(data.X2d_file_ids, data.X2d_file, ids_lb)

        log_event("INFO", "final.drop_ids_without_bags")
        ids_tr, X2d_tr, df_tr = drop_ids_without_bags(
            ids=ids_tr,
            X2d=X2d_tr,
            df_part=df_tr,
            id2pos=data.id2pos,
            id_col=data.id_col,
        )
        ids_lb, X2d_lb, df_lb = drop_ids_without_bags(
            ids=ids_lb,
            X2d=X2d_lb,
            df_part=df_lb,
            id2pos=data.id2pos,
            id_col=data.id_col,
        )
        log_event("INFO", "final.ids_after_drop", n_train=int(len(ids_tr)), n_lb=int(len(ids_lb)))

        log_event("INFO", "final.build_targets_and_weights")
        y_tr = coerce_binary_labels(df_tr)
        w_tr = build_task_weights(df_tr)
        y_abs_tr, m_abs_tr, y_fluo_tr, m_fluo_tr = build_aux_targets_and_masks(df_tr)
        w_abs_tr, w_fluo_tr = build_aux_weights(df_tr)

        y_lb = coerce_binary_labels(df_lb)
        w_lb = build_task_weights(df_lb)
        y_abs_lb, m_abs_lb, y_fluo_lb, m_fluo_lb = build_aux_targets_and_masks(df_lb)
        w_abs_lb, w_fluo_lb = build_aux_weights(df_lb)

        log_event("INFO", "final.standardize_aux_targets")
        tr_idx = np.arange(len(df_tr), dtype=np.int64)
        mu_abs, sd_abs = fit_standardizer(y_abs_tr, m_abs_tr, tr_idx)
        mu_f, sd_f = fit_standardizer(y_fluo_tr, m_fluo_tr, tr_idx)
        y_abs_tr_sc = apply_standardizer(y_abs_tr, mu_abs, sd_abs)
        y_abs_lb_sc = apply_standardizer(y_abs_lb, mu_abs, sd_abs)
        y_fluo_tr_sc = apply_standardizer(y_fluo_tr, mu_f, sd_f)
        y_fluo_lb_sc = apply_standardizer(y_fluo_lb, mu_f, sd_f)

        log_event("INFO", "final.compute_loss_weighting")
        lam = compute_lam(
            cfg.loss,
            y_train=y_tr,
            fallback_lambda_power=1.0,
            fallback_lam_floor=0.25,
            fallback_lam_ceil=6.0,
        )
        posw = pos_weight_per_task(y_tr, clip=compute_posw_clips(cfg.loss, fallback_clip=50.0))
        gamma_t = compute_gamma(cfg.loss)

        bitmask_group_top_ids, bitmask_group_class_weight = build_bitmask_group_definition(
            y_tr,
            top_k=int(cfg.loss.bitmask_group_top_k),
            class_weight_alpha=float(cfg.loss.bitmask_group_weight_alpha),
            class_weight_cap=float(cfg.loss.bitmask_group_weight_cap),
        )

        log_event(
            "INFO",
            "final.explainability_flags",
            run_chem_ace=bool(self.config.explainability.run_chem_ace if self.config.explainability else False),
            run_lambda_vol=bool(self.config.explainability.run_lambda_vol if self.config.explainability else False),
            run_concept_rl=bool(self.config.explainability.run_concept_rl if self.config.explainability else False),
        )
        explain_cfg = self.config.explainability
        if (
            explain_cfg is not None
            and (bool(explain_cfg.run_lambda_vol) or bool(explain_cfg.run_concept_rl))
            and not bool(explain_cfg.run_chem_ace)
        ):
            explain_cfg = replace(explain_cfg, run_chem_ace=True)

        chem_bundle = None
        if explain_cfg is not None and bool(explain_cfg.run_chem_ace):
            log_event("INFO", "final.prepare_chem_ace_bundle.start")
            # Anti-leakage guard: build Chem-ACE concepts only from train IDs.
            ids_scope = sorted(set(ids_tr))
            log_event(
                "INFO",
                "final.prepare_chem_ace_bundle.scope",
                n_train_ids=int(len(ids_tr)),
                n_leaderboard_ids=int(len(ids_lb)),
                n_scope_ids=int(len(ids_scope)),
                scope_splits="train",
                anti_leakage="enabled",
            )
            chem_bundle = prepare_chem_ace_bundle(
                config=explain_cfg,
                outdir=outdir,
                seed=int(self.config.seed),
                df_full=data.df_full,
                id_col=data.id_col,
                ids_scope=ids_scope,
                ids_infer_scope=ids_lb,
                ids_2d_file=data.X2d_file_ids,
                X2d_file=data.X2d_file,
                starts=data.starts,
                counts=data.counts,
                id2pos=data.id2pos,
                conf_sorted=data.conf_sorted,
                Xinst_sorted=data.Xinst_sorted,
                inst_geom_dim=int(data.inst_geom_dim),
                inst_qm_dim=int(data.inst_qm_dim),
                inst_geom_cols=tuple(str(x) for x in data.inst_geom_cols),
                inst_qm_cols=tuple(str(x) for x in data.inst_qm_cols),
                starts_raw=data.starts_raw,
                counts_raw=data.counts_raw,
                id2pos_raw=(None if data.id2pos_raw is None else dict(data.id2pos_raw)),
                conf_sorted_raw=data.conf_sorted_raw,
                Xinst_sorted_raw=data.Xinst_sorted_raw,
                inst_geom_dim_raw=int(data.inst_geom_dim_raw),
                inst_qm_dim_raw=int(data.inst_qm_dim_raw),
                inst_geom_cols_raw=tuple(str(x) for x in data.inst_geom_cols_raw),
                inst_qm_cols_raw=tuple(str(x) for x in data.inst_qm_cols_raw),
            )
            log_event(
                "INFO",
                "final.prepare_chem_ace_bundle.done",
                n_concepts=int(len(chem_bundle.concept_ids)),
            )

        concept_rl_targets: dict[int, tuple[str, ...]] = {}
        concept_rl_target_report: dict[str, Any] | None = None
        if explain_cfg is not None and bool(explain_cfg.run_concept_rl) and chem_bundle is not None:
            log_event("INFO", "final.concept_rl.build_targets")
            concept_rl_targets, concept_rl_target_report = build_positive_concept_targets_with_report(
                config=explain_cfg,
                ids_train=ids_tr,
                y_cls_train=y_tr,
                chem_bundle=chem_bundle,
            )
            if concept_rl_target_report is not None:
                log_event(
                    "INFO",
                    "final.concept_rl.target_source",
                    concept_source_train=str(
                        concept_rl_target_report.get(
                            "concept_source_train",
                            getattr(chem_bundle, "train_membership_source", "unknown"),
                        )
                    ),
                    n_concepts=int(concept_rl_target_report.get("n_concepts", len(chem_bundle.concept_ids))),
                )

        rl_active = bool(
            explain_cfg is not None
            and bool(explain_cfg.run_concept_rl)
            and (chem_bundle is not None)
            and any(len(v) > 0 for v in concept_rl_targets.values())
        )
        if bool(explain_cfg is not None and bool(explain_cfg.run_concept_rl) and not rl_active):
            print("[FINAL][CONCEPT-RL] requested but disabled (no target concepts available).")
        log_event("INFO", "final.concept_rl.status", rl_active=bool(rl_active))

        final_dir = outdir / "final_best_train_vs_leaderboard"
        final_dir.mkdir(parents=True, exist_ok=True)
        log_event("INFO", "final.output_dir_ready", final_dir=str(final_dir))
        target_report_path: Path | None = None
        target_assignment_csv_path: Path | None = None
        target_assignment_summary_path: Path | None = None
        if concept_rl_target_report is not None:
            target_report_path = final_dir / "concept_rl_target_selection.json"
            target_report_path.write_text(json.dumps(concept_rl_target_report, indent=2))
            log_event("INFO", "final.concept_rl.target_report", path=str(target_report_path))

        log_event("INFO", "final.build_datasets")
        ds_tr = MILTrainDataset(
            ids_tr,
            X2d_tr,
            y_tr,
            w_tr,
            y_abs_tr_sc,
            m_abs_tr,
            w_abs_tr,
            y_fluo_tr_sc,
            m_fluo_tr,
            w_fluo_tr,
            data.starts,
            data.counts,
            data.id2pos,
            data.Xinst_sorted,
            conf_sorted=(data.conf_sorted if rl_active else None),
            max_instances=0,
            seed=int(self.config.seed),
            include_metadata=bool(rl_active),
        )
        ds_lb = MILTrainDataset(
            ids_lb,
            X2d_lb,
            y_lb,
            w_lb,
            y_abs_lb_sc,
            m_abs_lb,
            w_abs_lb,
            y_fluo_lb_sc,
            m_fluo_lb,
            w_fluo_lb,
            data.starts,
            data.counts,
            data.id2pos,
            data.Xinst_sorted,
            max_instances=0,
            seed=int(self.config.seed) + 999,
        )

        log_event(
            "INFO",
            "final.build_dataloaders",
            batch_size=int(cfg.runtime.batch_size),
            balanced_sampler=bool(cfg.sampler.use_balanced_batch_sampler),
        )
        if bool(cfg.sampler.use_balanced_batch_sampler):
            sampler_tr = make_balanced_batch_sampler(
                y_tr,
                batch_size=int(cfg.runtime.batch_size),
                rare_mult=float(cfg.sampler.rare_oversample_mult),
                rare_target_prev=float(cfg.sampler.rare_target_prev),
                sample_weight_cap=float(cfg.sampler.sample_weight_cap),
                batch_pos_fraction=float(cfg.sampler.batch_pos_fraction),
                min_pos_per_batch=int(cfg.sampler.min_pos_per_batch),
                rare_prev_thr=cfg.sampler.rare_prev_thr,
                seed=int(self.config.seed) + 4242,
            )
            dl_tr = self.loader_builder.train_loader(
                ds_tr,
                batch_size=int(cfg.runtime.batch_size),
                batch_sampler=sampler_tr,
                collate_fn=collate_train,
            )
        else:
            sampler_tr = make_weighted_sampler(
                y_tr,
                rare_mult=float(cfg.sampler.rare_oversample_mult),
                rare_target_prev=float(cfg.sampler.rare_target_prev),
                sample_weight_cap=float(cfg.sampler.sample_weight_cap),
                rare_prev_thr=cfg.sampler.rare_prev_thr,
            )
            dl_tr = self.loader_builder.train_loader(
                ds_tr,
                batch_size=int(cfg.runtime.batch_size),
                sampler=sampler_tr,
                collate_fn=collate_train,
            )
        sampler_diag_path = final_dir / "sampler_diagnostics.csv"
        sampler_diag_df = build_sampler_diagnostics_df(
            y=y_tr,
            batch_size=int(cfg.runtime.batch_size),
            use_balanced_batch_sampler=bool(cfg.sampler.use_balanced_batch_sampler),
            rare_mult=float(cfg.sampler.rare_oversample_mult),
            rare_target_prev=float(cfg.sampler.rare_target_prev),
            sample_weight_cap=float(cfg.sampler.sample_weight_cap),
            batch_pos_fraction=float(cfg.sampler.batch_pos_fraction),
            min_pos_per_batch=int(cfg.sampler.min_pos_per_batch),
            rare_prev_thr=cfg.sampler.rare_prev_thr,
            seed=int(self.config.seed) + 4242,
        )
        sampler_diag_df.to_csv(sampler_diag_path, index=False)
        _diag_pick = sampler_diag_df.set_index(["section", "metric"])["value"].to_dict()
        log_event(
            "INFO",
            "final.sampler_diagnostics",
            path=str(sampler_diag_path),
            sampler_mode=str("balanced_batch" if bool(cfg.sampler.use_balanced_batch_sampler) else "weighted_sampler"),
            mean_pos_per_batch=f"{float(_diag_pick.get(('summary', 'mean_pos_per_batch'), float('nan'))):.3f}",
            duplicate_rate=f"{float(_diag_pick.get(('summary', 'duplicate_rate'), float('nan'))):.6f}",
        )
        dl_val = self.loader_builder.eval_loader(
            ds_lb,
            batch_size=min(128, int(cfg.runtime.batch_size)),
            collate_fn=collate_train,
        )

        lambda_vol_cb = None
        if explain_cfg is not None and bool(explain_cfg.run_lambda_vol) and (chem_bundle is not None):
            log_event("INFO", "final.lambda_vol.build_callback.start")
            monitor_loader = make_monitor_loader(
                ids=ids_lb,
                x2d=X2d_lb,
                starts=data.starts,
                counts=data.counts,
                id2pos=data.id2pos,
                xinst_sorted=data.Xinst_sorted,
                conf_sorted=data.conf_sorted,
                batch_size=min(64, int(cfg.runtime.batch_size)),
                seed=int(self.config.seed) + 707,
                loader_cfg=self.config.loader,
            )
            lambda_vol_cb = build_lambda_vol_callback(
                config=explain_cfg,
                outdir=outdir,
                seed=int(self.config.seed),
                monitor_loader=monitor_loader,
                chem_bundle=chem_bundle,
            )
            log_event("INFO", "final.lambda_vol.build_callback.done")

        log_event("INFO", "final.build_model")
        model = MILModelBuilder.build(
            config=cfg,
            mol_dim=int(X2d_tr.shape[1]),
            inst_dim=int(data.Xinst_sorted.shape[1]),
            inst_geom_dim=int(data.inst_geom_dim),
            inst_qm_dim=int(data.inst_qm_dim),
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
            bitmask_group_top_ids=bitmask_group_top_ids,
            bitmask_group_class_weight=bitmask_group_class_weight,
        )

        if rl_active and explain_cfg is not None and chem_bundle is not None:
            train_set = {str(x) for x in ids_tr}
            target_concepts = {
                str(cid)
                for vals in concept_rl_targets.values()
                for cid in vals
            }
            rl_concept_mol_map: Dict[str, set[str]] = {}
            rl_concept_conf_map: Dict[str, set[tuple[str, str]]] = {}
            for cid in sorted(target_concepts):
                mols = {
                    str(mid) for mid in chem_bundle.concept_mol_map.get(str(cid), set())
                    if str(mid) in train_set
                }
                confs = {
                    (str(mid), str(conf))
                    for (mid, conf) in chem_bundle.concept_conf_map.get(str(cid), set())
                    if str(mid) in train_set
                }
                if len(mols) > 0:
                    rl_concept_mol_map[str(cid)] = mols
                if len(confs) > 0:
                    rl_concept_conf_map[str(cid)] = confs

            target_assignment_csv_path = final_dir / "concept_rl_active_assignments.csv"
            target_assignment_summary_path = final_dir / "concept_rl_active_assignments_summary.json"
            _export_concept_rl_active_assignments(
                out_csv_path=target_assignment_csv_path,
                out_summary_path=target_assignment_summary_path,
                ids_train=ids_tr,
                y_cls_train=y_tr,
                concept_targets=concept_rl_targets,
                concept_mol_map=rl_concept_mol_map,
                concept_conf_map=rl_concept_conf_map,
            )
            log_event(
                "INFO",
                "final.concept_rl.active_assignments",
                csv_path=str(target_assignment_csv_path),
                summary_path=str(target_assignment_summary_path),
                n_target_concepts=int(len(target_concepts)),
                n_target_concepts_with_mols=int(len(rl_concept_mol_map)),
                n_target_concepts_with_confs=int(len(rl_concept_conf_map)),
            )

            model.configure_rl_concept_guidance(
                task_target_concepts=concept_rl_targets,
                concept_conf_map=rl_concept_conf_map,
                concept_mol_map=rl_concept_mol_map,
                init_scale=float(explain_cfg.concept_rl_init_scale),
                max_scale=float(explain_cfg.concept_rl_max_scale),
                negative_penalty=float(explain_cfg.concept_rl_negative_penalty),
            )
            print(
                "[FINAL][CONCEPT-RL] enabled "
                f"(targets_per_task={[len(v) for _, v in sorted(concept_rl_targets.items())]})."
            )
            log_event(
                "INFO",
                "final.concept_rl.enabled",
                targets_per_task=[len(v) for _, v in sorted(concept_rl_targets.items())],
            )

        rl_cb = None
        rl_policy_path = None
        if rl_active and explain_cfg is not None:
            rl_policy_path = final_dir / "concept_rl_policy_history.json"
            rl_cb = ConceptRLControllerCallback(
                config=ConceptRLPolicyConfig(
                    init_mean=float(explain_cfg.concept_rl_init_scale),
                    sigma=float(explain_cfg.concept_rl_policy_sigma),
                    learning_rate=float(explain_cfg.concept_rl_policy_lr),
                    max_scale=float(explain_cfg.concept_rl_max_scale),
                    reward_alignment_w=float(explain_cfg.concept_rl_reward_alignment_w),
                    reward_min_ap_w=float(explain_cfg.concept_rl_reward_min_ap_w),
                    baseline_momentum=float(explain_cfg.concept_rl_baseline_momentum),
                    reward_key="val_macro_ap",
                    reward_min_key="val_min_ap",
                    alignment_key="train_concept_alignment",
                ),
                out_json_path=str(rl_policy_path),
            )

        extra_callbacks = []
        if lambda_vol_cb is not None:
            extra_callbacks.append(lambda_vol_cb)
        if rl_cb is not None:
            extra_callbacks.append(rl_cb)

        trainer_cfg = LightningTrainerConfig(
            max_epochs=int(self.config.trainer.max_epochs),
            patience=int(self.config.trainer.patience),
            accelerator=str(self.config.trainer.accelerator),
            devices=int(self.config.trainer.devices),
            precision=str(self.config.trainer.precision),
            accumulate_grad_batches=int(cfg.runtime.accumulate_grad_batches),
            # Keep a single best checkpoint for the final run only.
            save_checkpoint=True,
            save_weights_only=True,
        )
        trainer, ckpt_cb = LightningTrainerFactory(trainer_cfg).build(
            ckpt_dir=str(final_dir),
            trial=None,
            extra_callbacks=(extra_callbacks if extra_callbacks else None),
        )
        with log_step("final.trainer.fit"):
            trainer.fit(model, dl_tr, dl_val)

        best_epoch = None
        best_ckpt_path = None
        if ckpt_cb is not None:
            best_path = ckpt_cb.best_model_path
            if best_path and Path(best_path).exists():
                ckpt = torch.load(best_path, map_location="cpu")
                best_epoch = int(ckpt.get("epoch", -1))
                best_ckpt_path = str(best_path)
                model.load_state_dict(ckpt["state_dict"], strict=True)
                print(f"[FINAL] loaded best ckpt: {best_path}")
                log_event("INFO", "final.best_ckpt_loaded", path=str(best_path), best_epoch=best_epoch)

        evaluator = ModelEvaluator(device=self.eval_device)
        log_event("INFO", "final.eval.start")
        macro_ap_lb, aps_lb, macro_auc_lb, aucs_lb = evaluator.eval_best_epoch(model, dl_val)
        eval_json = {
            "macro_ap": float(macro_ap_lb),
            "macro_pr_auc": float(macro_ap_lb),
            "macro_auc": float(macro_auc_lb),
            "ap_task0": float(aps_lb[0]),
            "ap_task1": float(aps_lb[1]),
            "ap_task2": float(aps_lb[2]),
            "ap_task3": float(aps_lb[3]),
            "pr_auc_task0": float(aps_lb[0]),
            "pr_auc_task1": float(aps_lb[1]),
            "pr_auc_task2": float(aps_lb[2]),
            "pr_auc_task3": float(aps_lb[3]),
            "auc_task0": float(aucs_lb[0]),
            "auc_task1": float(aucs_lb[1]),
            "auc_task2": float(aucs_lb[2]),
            "auc_task3": float(aucs_lb[3]),
            "best_epoch": best_epoch,
            "best_ckpt_path": best_ckpt_path,
        }
        (final_dir / "leaderboard_eval.json").write_text(json.dumps(eval_json, indent=2))
        print(
            f"[FINAL] leaderboard eval: macro_pr_auc={macro_ap_lb:.6f} macro_roc_auc={macro_auc_lb:.6f} "
            f"pr_aucs={aps_lb} roc_aucs={aucs_lb}"
        )
        log_event(
            "INFO",
            "final.eval.metrics",
            macro_pr_auc=f"{macro_ap_lb:.6f}",
            macro_roc_auc=f"{macro_auc_lb:.6f}",
            best_epoch=best_epoch,
        )

        pd.DataFrame(
            {
                "task": list(TASK_COLS),
                "auc": [float(x) for x in aucs_lb],
            }
        ).to_csv(final_dir / "leaderboard_auc_per_task.csv", index=False)

        export_ds = MILExportDataset(
            ids_lb,
            X2d_lb,
            starts=data.starts,
            counts=data.counts,
            id2pos=data.id2pos,
            Xinst_sorted=data.Xinst_sorted,
            conf_sorted=data.conf_sorted,
            max_instances=0,
            seed=int(self.config.seed) + 123,
        )
        export_dl = self.loader_builder.eval_loader(
            export_ds,
            batch_size=min(64, int(cfg.runtime.batch_size)),
            collate_fn=collate_export,
        )
        log_event("INFO", "final.export.attention.start")
        out_path = Path(self.config.attn_out) if self.config.attn_out else (outdir / "leaderboard_attn.csv")
        true_labels_by_id: Dict[str, np.ndarray] = {
            str(ids_lb[i]): np.asarray(y_lb[i], dtype=np.float32).reshape(-1)
            for i in range(len(ids_lb))
        }
        conf_sig_map: Dict[str, str] = {}
        conf_sig_map_alt: Dict[str, str] = {}
        if chem_bundle is not None:
            conf_sig_map, conf_sig_map_alt = _load_conformer_signature_maps(
                signature_csv=getattr(chem_bundle, "conformer_signature_csv", None),
            )
            log_event(
                "INFO",
                "final.export.attention.pmapper_signatures",
                n_signatures=int(len(conf_sig_map)),
                n_signatures_alt=int(len(conf_sig_map_alt)),
                source_csv=(None if getattr(chem_bundle, "conformer_signature_csv", None) is None else str(chem_bundle.conformer_signature_csv)),
            )
        written_attn_path = export_leaderboard_attention(
            model,
            export_dl,
            device=self.eval_device,
            out_path=out_path,
            conf_signature_map=(conf_sig_map if len(conf_sig_map) > 0 else None),
            conf_signature_alt_map=(conf_sig_map_alt if len(conf_sig_map_alt) > 0 else None),
            true_labels_by_id=true_labels_by_id,
        )
        log_event("INFO", "final.export.attention.done", path=str(written_attn_path))

        strict_scores_csv: Path | None = None
        strict_concepts_csv: Path | None = None
        strict_summary_json: Path | None = None
        if (
            chem_bundle is not None
            and explain_cfg is not None
            and bool(explain_cfg.chem_ace_strict_rerank)
        ):
            log_event(
                "INFO",
                "final.export.strict_mixer_rerank.start",
                layer_name=str(explain_cfg.chem_ace_strict_rerank_layer_name),
                top_rows_per_task=int(explain_cfg.chem_ace_strict_rerank_top_rows_per_task),
                batch_size=int(explain_cfg.chem_ace_strict_rerank_batch_size),
            )
            strict_scores_csv, strict_concepts_csv, strict_summary_json = _run_strict_mixer_rerank(
                model=model,
                device=self.eval_device,
                layer_name=str(explain_cfg.chem_ace_strict_rerank_layer_name),
                out_dir=final_dir,
                attention_table_path=written_attn_path,
                concept_ids=chem_bundle.concept_ids,
                concept_metadata=chem_bundle.concept_metadata,
                concept_mol_map=chem_bundle.concept_mol_map,
                concept_conf_map=chem_bundle.concept_conf_map,
                ids_2d_file=data.X2d_file_ids,
                X2d_file=data.X2d_file,
                starts=data.starts,
                counts=data.counts,
                id2pos=data.id2pos,
                conf_sorted=data.conf_sorted,
                Xinst_sorted=data.Xinst_sorted,
                top_rows_per_task=int(explain_cfg.chem_ace_strict_rerank_top_rows_per_task),
                batch_size=int(explain_cfg.chem_ace_strict_rerank_batch_size),
            )
            log_event(
                "INFO",
                "final.export.strict_mixer_rerank.done",
                strict_scores_csv=(None if strict_scores_csv is None else str(strict_scores_csv)),
                strict_concepts_csv=(None if strict_concepts_csv is None else str(strict_concepts_csv)),
                strict_summary_json=(None if strict_summary_json is None else str(strict_summary_json)),
            )

        lv_art = getattr(lambda_vol_cb, "last_artifacts", None) if lambda_vol_cb is not None else None

        explained_pred_path: Path | None = None
        if chem_bundle is not None:
            log_event("INFO", "final.export.text_explanations.start")
            explained_pred_path = written_attn_path.with_name(f"{written_attn_path.stem}_explained.csv")
            explained_pred_path = export_prediction_text_explanations(
                pred_table_path=written_attn_path,
                out_path=explained_pred_path,
                concept_ids=chem_bundle.concept_ids,
                concept_metadata=chem_bundle.concept_metadata,
                concept_mol_map=chem_bundle.concept_mol_map,
                concept_conf_map=chem_bundle.concept_conf_map,
                task_cols=TASK_COLS,
                ricci_edges_csv=(None if lv_art is None else lv_art.ricci_edges_csv),
                lambda_vol_long_csv=(None if lv_art is None else lv_art.long_csv),
                a_priori_tags_csv=(
                    chem_bundle.a_priori_tags_infer_csv
                    if chem_bundle.a_priori_tags_infer_csv is not None
                    else chem_bundle.a_priori_tags_csv
                ),
                strict_scores_csv=(
                    None if strict_scores_csv is None else str(strict_scores_csv)
                ),
                strict_weight=(
                    0.0
                    if explain_cfg is None
                    else float(explain_cfg.chem_ace_strict_rerank_weight)
                ),
                top_k=3,
                bridge_threshold=0.20,
            )
            log_event("INFO", "final.export.text_explanations.done", path=str(explained_pred_path))

        explainability_payload: dict[str, Any] = {}
        if chem_bundle is not None:
            explainability_payload["chem_ace"] = {
                "output_dir": str(chem_bundle.output_dir),
                "db_uri": str(chem_bundle.db_uri),
                "run_id": str(chem_bundle.run_id),
                "concept_set_id": str(chem_bundle.concept_set_id),
                "train_membership_source": str(chem_bundle.train_membership_source),
                "n_concepts": int(len(chem_bundle.concept_ids)),
                "concept_ids": [str(x) for x in chem_bundle.concept_ids],
                "a_priori_tags_csv": (
                    None if chem_bundle.a_priori_tags_csv is None else str(chem_bundle.a_priori_tags_csv)
                ),
                "a_priori_vs_concepts_csv": (
                    None
                    if chem_bundle.a_priori_vs_concepts_csv is None
                    else str(chem_bundle.a_priori_vs_concepts_csv)
                ),
                "a_priori_tags_infer_csv": (
                    None
                    if chem_bundle.a_priori_tags_infer_csv is None
                    else str(chem_bundle.a_priori_tags_infer_csv)
                ),
                "a_priori_vs_concepts_infer_csv": (
                    None
                    if chem_bundle.a_priori_vs_concepts_infer_csv is None
                    else str(chem_bundle.a_priori_vs_concepts_infer_csv)
                ),
                "activity_calibrated_tags_csv": (
                    None
                    if chem_bundle.activity_calibrated_tags_csv is None
                    else str(chem_bundle.activity_calibrated_tags_csv)
                ),
                "activity_calibration_summary_json": (
                    None
                    if chem_bundle.activity_calibration_summary_json is None
                    else str(chem_bundle.activity_calibration_summary_json)
                ),
                "conformer_signature_csv": (
                    None
                    if getattr(chem_bundle, "conformer_signature_csv", None) is None
                    else str(chem_bundle.conformer_signature_csv)
                ),
                "conformer_signature_summary_json": (
                    None
                    if getattr(chem_bundle, "conformer_signature_summary_json", None) is None
                    else str(chem_bundle.conformer_signature_summary_json)
                ),
                "prediction_explanations_csv": (
                    None if explained_pred_path is None else str(explained_pred_path)
                ),
                "strict_mixer_scores_csv": (
                    None if strict_scores_csv is None else str(strict_scores_csv)
                ),
                "strict_mixer_concepts_csv": (
                    None if strict_concepts_csv is None else str(strict_concepts_csv)
                ),
                "strict_mixer_summary_json": (
                    None if strict_summary_json is None else str(strict_summary_json)
                ),
            }
        if lambda_vol_cb is not None:
            if lv_art is not None:
                explainability_payload["lambda_vol"] = {
                    "tensor_npz": str(lv_art.tensor_npz),
                    "long_csv": str(lv_art.long_csv),
                    "metadata_json": str(lv_art.metadata_json),
                    "lattice_html": str(lv_art.lattice_html),
                    "alerts_json": str(lv_art.alerts_json),
                    "recommendations_json": str(lv_art.recommendations_json),
                    "manifold_html_by_task": dict(lv_art.manifold_html_by_task),
                    "ricci_edges_csv": lv_art.ricci_edges_csv,
                    "ricci_summary_csv": lv_art.ricci_summary_csv,
                    "ricci_flow_npz": lv_art.ricci_flow_npz,
                }
        if concept_rl_target_report is not None:
            explainability_payload["concept_rl"] = {
                "target_selection_report_json": (
                    None if target_report_path is None else str(target_report_path)
                ),
                "active_assignments_csv": (
                    None if target_assignment_csv_path is None else str(target_assignment_csv_path)
                ),
                "active_assignments_summary_json": (
                    None
                    if target_assignment_summary_path is None
                    else str(target_assignment_summary_path)
                ),
                "target_concepts_by_task": {
                    str(int(k)): [str(x) for x in v] for k, v in concept_rl_targets.items()
                },
            }
        if (
            rl_cb is not None
            and rl_policy_path is not None
            and rl_policy_path.exists()
            and ("concept_rl" in explainability_payload)
        ):
            explainability_payload["concept_rl"]["policy_history_json"] = str(rl_policy_path)

        if explainability_payload:
            (final_dir / "explainability_artifacts.json").write_text(
                json.dumps(explainability_payload, indent=2)
            )
            log_event(
                "INFO",
                "final.export.explainability_payload",
                path=str(final_dir / "explainability_artifacts.json"),
            )
        log_event("DONE", "final.run", final_dir=str(final_dir))


__all__ = [
    "TrainerSystemConfig",
    "CVRunConfig",
    "StudyConfig",
    "FinalTrainConfig",
    "FinalExplainabilityConfig",
    "MILCVData",
    "MILFinalData",
    "MILFoldTrainer",
    "MILCrossValidator",
    "MILStudyRunner",
    "MILFinalTrainer",
    "StudyArtifactsWriter",
    "drop_ids_without_bags",
]
