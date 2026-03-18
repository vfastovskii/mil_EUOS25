from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import gc
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
try:
    from scipy.stats import t as _student_t_dist
except Exception:  # pragma: no cover - scipy may be unavailable in minimal envs
    _student_t_dist = None

from ..data.collate import collate_export, collate_train
from ..data.datasets import MILExportDataset, MILTrainDataset
from ..data.exports import export_attention_dataset_summary, export_leaderboard_attention
from ..training.builders import DataLoaderBuilder, LoaderConfig, MILModelBuilder
from ..training.configs import HPOConfig
from ..training.execution import (
    CVRunConfig,
    MILCVData,
    MILFoldTrainer,
    StudyConfig,
    TrainerSystemConfig,
    _resolve_device,
)
from ..training.loss_config import compute_gamma, compute_lam, compute_posw_clips
from ..training.model_artifacts import save_mil_model_artifact
from ..training.search_space import search_space
from ..training.trainer import LightningTrainerConfig, LightningTrainerFactory, ModelEvaluator
from ..utils.constants import TASK_COLS
from ..utils.data_io import align_by_id, load_2d, load_labels
from ..utils.instances import build_instance_index, load_and_merge_instances
from ..utils.metrics import ap_per_task, roc_auc_per_task
from ..utils.ops import (
    apply_standardizer,
    bitmask_ids,
    build_aux_targets_and_masks,
    build_aux_weights,
    build_bitmask_group_definition,
    build_sampler_diagnostics_df,
    build_task_weights,
    coerce_binary_labels,
    fit_standardizer,
    fold_indices,
    make_balanced_batch_sampler,
    make_weighted_sampler,
    maybe_set_torch_fast_flags,
    pos_weight_per_task,
    set_all_seeds,
)
from ..utils.progress import log_event, log_step

FAMILY_CHOICES: tuple[str, ...] = ("catboost_st", "mt_2d", "mt_2d3d", "mt_3d")


def _mean_sd_ci95(values: np.ndarray) -> tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0, mean, mean, 0.0
    sd = float(np.std(arr, ddof=1))
    df = int(arr.size - 1)
    if _student_t_dist is not None:
        crit = float(_student_t_dist.ppf(0.975, df=df))
    else:  # Fallback to normal approximation if scipy is unavailable.
        crit = 1.96
    half = float(crit * sd / math.sqrt(float(arr.size)))
    return mean, sd, mean - half, mean + half, half


def _maybe_mirror_best_params(path: Path) -> None:
    """
    Mirror best-params artifact to persistent storage immediately if configured.

    Set env var `FAMILY_SUITE_BEST_PARAMS_MIRROR_DIR` to enable.
    """
    mirror_dir = str(os.environ.get("FAMILY_SUITE_BEST_PARAMS_MIRROR_DIR", "")).strip()
    if not mirror_dir:
        return
    src = Path(path)
    try:
        dst_dir = Path(mirror_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_bytes(src.read_bytes())
        tmp.replace(dst)
        log_event("INFO", "family.hpo.best_params.mirrored", src=str(src), dst=str(dst))
    except Exception as exc:
        log_event("WARN", "family.hpo.best_params.mirror_failed", src=str(src), error=repr(exc))


@dataclass(frozen=True)
class FamilySuiteConfig:
    labels: str
    feat2d_scaled: str
    feat2d_raw: str | None
    feat3d_scaled: str
    feat3d_qm_scaled: str
    study_dir: str
    id_col: str
    conf_col: str
    split_col: str
    fold_col: str
    use_splits: tuple[str, ...]
    leaderboard_split: str
    max_epochs: int
    patience: int
    trials: int
    seed: int
    nn_accelerator: str
    nn_devices: int
    precision: str
    num_workers: int
    cpu_workers: int
    pin_memory: bool
    run_hpo: bool
    hpo_only: bool
    pruner_warmup_steps: int
    model_families: tuple[str, ...]
    calibration_method: str
    skip_family_explainability: bool
    skip_family_calibration: bool
    skip_family_blending: bool
    best_params_dir: str | None
    family_best_params_json: str | None
    catboost_task_params_jsons: tuple[str, ...]
    catboost_final_iterations_jsons: tuple[str, ...]
    mt_2d_params_json: str | None
    mt_2d3d_params_json: str | None
    mt_3d_params_json: str | None
    mt_2d_final_epochs_json: str | None
    mt_2d3d_final_epochs_json: str | None
    mt_3d_final_epochs_json: str | None
    catboost_hpo_parallel_tasks: int
    blend_seed_ensemble_size: int
    blend_seed_step: int


@dataclass(frozen=True)
class _MILFamilyData:
    family: str
    ids_train: List[str]
    ids_lb: List[str]
    X2d_train: np.ndarray
    X2d_lb: np.ndarray
    y_cls_train: np.ndarray
    y_cls_lb: np.ndarray
    w_cls_train: np.ndarray
    w_cls_lb: np.ndarray
    y_abs_train: np.ndarray
    m_abs_train: np.ndarray
    w_abs_train: np.ndarray
    y_abs_lb: np.ndarray
    m_abs_lb: np.ndarray
    w_abs_lb: np.ndarray
    y_fluo_train: np.ndarray
    m_fluo_train: np.ndarray
    w_fluo_train: np.ndarray
    y_fluo_lb: np.ndarray
    m_fluo_lb: np.ndarray
    w_fluo_lb: np.ndarray
    folds_info: Sequence[Tuple[np.ndarray, np.ndarray, int]]
    starts: np.ndarray
    counts: np.ndarray
    id2pos: Dict[str, int]
    Xinst_sorted: np.ndarray
    conf_sorted: np.ndarray
    inst_geom_dim: int
    inst_qm_dim: int


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0), 1e-6, 1 - 1e-6)


def _metric_sample_weight_for_task(w: np.ndarray | None, task_idx: int) -> np.ndarray | None:
    """Apply the same per-task weighting policy used by reported metrics."""
    if w is None or int(task_idx) not in (0, 1):
        return None
    sw = np.asarray(w, dtype=np.float64).reshape(-1)
    sw = np.nan_to_num(sw, nan=0.0, posinf=0.0, neginf=0.0)
    sw = np.clip(sw, 0.0, np.inf)
    if float(np.sum(sw)) <= 0.0:
        return None
    return sw


def _df_by_ids(df: pd.DataFrame, *, id_col: str, ids: Sequence[str]) -> pd.DataFrame:
    # Keep first row for duplicate IDs to preserve one-label-per-ID contract.
    d = df.drop_duplicates(subset=[id_col], keep="first").copy()
    d[id_col] = d[id_col].astype(str)
    pos = {str(x): i for i, x in enumerate(d[id_col].tolist())}
    idx = [pos[str(i)] for i in ids if str(i) in pos]
    return d.iloc[idx].reset_index(drop=True)


def _metric_table(
    *,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    w_cls: np.ndarray,
) -> pd.DataFrame:
    aps = ap_per_task(y_true, p_pred, w_cls=w_cls, weighted_tasks=(0, 1))
    aucs = roc_auc_per_task(y_true, p_pred, w_cls=w_cls, weighted_tasks=(0, 1))
    rows: List[Dict[str, Any]] = []
    nlls: List[float] = []
    briers: List[float] = []
    for t, task in enumerate(TASK_COLS):
        yt = y_true[:, t].astype(int)
        pt = _clip_prob(p_pred[:, t])
        sw = _metric_sample_weight_for_task(w_cls[:, t] if w_cls is not None else None, t)
        try:
            nll = float(log_loss(yt, pt, sample_weight=sw, labels=[0, 1]))
        except Exception:
            nll = float("nan")
        try:
            brier = float(brier_score_loss(yt, pt, sample_weight=sw))
        except Exception:
            brier = float("nan")
        nlls.append(nll)
        briers.append(brier)
        rows.append(
            {
                "task": str(task),
                "pr_auc": float(aps[t]),
                "roc_auc": float(aucs[t]),
                "nll": nll,
                "brier": brier,
            }
        )
    rows.append(
        {
            "task": "macro",
            "pr_auc": float(np.mean(aps)),
            "roc_auc": float(np.mean(aucs)),
            "nll": float(np.nanmean(nlls)),
            "brier": float(np.nanmean(briers)),
        }
    )
    return pd.DataFrame(rows)


def _save_family_best_params(*, outdir: Path, family: str, best_params: Mapping[str, Any], best_value: float) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{family}.json"
    payload = {
        "family": str(family),
        "best_params": dict(best_params),
        "best_value_macro_ap_cv": float(best_value),
    }
    path.write_text(json.dumps(payload, indent=2))
    _maybe_mirror_best_params(path)
    return path


def _save_catboost_task_best_params(
    *,
    outdir: Path,
    task_idx: int,
    best_params: Mapping[str, Any],
    best_value: float,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"catboost_st_t{int(task_idx)}.json"
    payload = {
        "family": "catboost_st",
        "task_idx": int(task_idx),
        "task_name": str(TASK_COLS[int(task_idx)]),
        "best_params": dict(best_params),
        "best_value_pr_auc_cv": float(best_value),
    }
    path.write_text(json.dumps(payload, indent=2))
    _maybe_mirror_best_params(path)
    return path


def _save_family_final_epochs(
    *,
    outdir: Path,
    family: str,
    fold_best_epochs: Sequence[int],
    selected_epochs: int,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"final_epochs_{family}.json"
    payload = {
        "family": str(family),
        "selection_scope": "cv_train_only",
        "selection_rule": "round(mean(best_epoch_plus_1))",
        "fold_best_epochs": [int(x) for x in fold_best_epochs],
        "selected_epochs": int(selected_epochs),
    }
    path.write_text(json.dumps(payload, indent=2))
    _maybe_mirror_best_params(path)
    return path


def _save_catboost_final_iterations(
    *,
    outdir: Path,
    task_idx: int,
    fold_best_iterations: Sequence[int],
    selected_iterations: int,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"final_iterations_catboost_t{int(task_idx)}.json"
    payload = {
        "family": "catboost_st",
        "task_idx": int(task_idx),
        "task_name": str(TASK_COLS[int(task_idx)]),
        "selection_scope": "cv_train_only",
        "selection_rule": "round(mean(best_iteration_plus_1))",
        "fold_best_iterations": [int(x) for x in fold_best_iterations],
        "selected_iterations": int(selected_iterations),
    }
    path.write_text(json.dumps(payload, indent=2))
    _maybe_mirror_best_params(path)
    return path


def _load_catboost_task_best_params(*, outdir: Path, task_idx: int) -> Dict[str, Any]:
    path = outdir / f"catboost_st_t{int(task_idx)}.json"
    if not path.exists():
        raise FileNotFoundError(f"CatBoost best params file not found for task {task_idx}: {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("best_params"), dict):
        return dict(payload["best_params"])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(
        f"Invalid CatBoost best params JSON for task {task_idx}: expected object, got {type(payload).__name__}"
    )


def _load_family_best_params(*, outdir: Path, family: str) -> Dict[str, Any]:
    path = outdir / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"Best params file not found for family '{family}': {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("best_params"), dict):
        return dict(payload["best_params"])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Invalid best params JSON for '{family}': expected object, got {type(payload).__name__}")


def _load_family_overrides_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid family_best_params_json: expected object at top-level, got {type(payload).__name__}")
    fam = payload.get("family")
    if isinstance(fam, str) and isinstance(payload.get("best_params"), dict):
        return {str(fam): dict(payload["best_params"])}
    return dict(payload)


def _load_single_family_override_json(*, path: Path, family: str) -> Any:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid params JSON for family '{family}': expected object, got {type(payload).__name__}"
        )
    payload_family = payload.get("family")
    if isinstance(payload_family, str) and str(payload_family) != str(family):
        raise ValueError(
            f"Family mismatch in {path}: expected '{family}', got '{payload_family}'"
        )
    value: Any = payload.get("best_params") if isinstance(payload.get("best_params"), dict) else payload
    if not isinstance(value, dict):
        raise ValueError(f"Invalid params JSON for family '{family}' in {path}: best_params must be a dict.")
    # Remove metadata keys if a bare object with metadata was provided.
    if "best_params" not in payload:
        value = {k: v for k, v in value.items() if k not in {"family", "best_value_macro_ap_cv"}}
    return _normalize_family_params(family=family, value=value)


def _load_family_selected_epochs_json(*, path: Path, family: str) -> int:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid final-epochs JSON for family '{family}': expected object, got {type(payload).__name__}"
        )
    payload_family = payload.get("family")
    if isinstance(payload_family, str) and str(payload_family) != str(family):
        raise ValueError(
            f"Family mismatch in {path}: expected '{family}', got '{payload_family}'"
        )

    def _coerce_pos_int(value: Any, *, field: str, min_value: int = 1) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field} in {path}: bool is not allowed.")
        try:
            iv = int(value)
        except Exception as exc:
            raise ValueError(f"Invalid {field} in {path}: expected integer-like value, got {value!r}") from exc
        if iv < int(min_value):
            raise ValueError(f"Invalid {field} in {path}: value must be >= {int(min_value)}, got {iv}.")
        return iv

    if "selected_epochs" in payload:
        return _coerce_pos_int(payload.get("selected_epochs"), field="selected_epochs")
    if "target_epochs" in payload:
        return _coerce_pos_int(payload.get("target_epochs"), field="target_epochs")
    if "best_epoch" in payload:
        return int(_coerce_pos_int(payload.get("best_epoch"), field="best_epoch", min_value=0) + 1)
    raise ValueError(
        f"Invalid final-epochs JSON for family '{family}' in {path}: "
        "missing one of ['selected_epochs', 'target_epochs', 'best_epoch']."
    )


def _normalize_catboost_params(value: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("best_params"), dict):
        value = value["best_params"]
    if not isinstance(value, dict):
        raise ValueError("catboost_st params must be a dict")
    def _with_default_iterations(d: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(d)
        # If iterations is not provided, use CatBoost max-iterations default for this pipeline.
        if out.get("iterations", None) is None:
            out["iterations"] = 4000
        return out

    if all(k in value for k in ("task_0", "task_1", "task_2", "task_3")):
        return {f"task_{int(t)}": _with_default_iterations(value[f"task_{int(t)}"]) for t in range(4)}
    # Backward-compatible shared-params fallback.
    return {f"task_{int(t)}": _with_default_iterations(value) for t in range(4)}


def _resolve_catboost_iterations(params: Mapping[str, Any], *, default_iterations: int = 4000) -> int:
    raw = params.get("iterations", None)
    if raw is None:
        return int(default_iterations)
    try:
        it = int(raw)
    except Exception:
        return int(default_iterations)
    return int(max(1, it))


def _normalize_family_params(*, family: str, value: Any) -> Any:
    if family == "catboost_st":
        return _normalize_catboost_params(value)
    if isinstance(value, dict) and isinstance(value.get("best_params"), dict):
        return dict(value["best_params"])
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"Params for family '{family}' must be a dict.")


def _load_catboost_task_overrides(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw_path in paths:
        p = Path(str(raw_path))
        if not p.exists():
            raise FileNotFoundError(f"CatBoost task params file not found: {p}")
        payload = json.loads(p.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid CatBoost task params JSON {p}: expected object.")
        if isinstance(payload.get("best_params"), dict):
            task_idx = payload.get("task_idx", None)
            best_params = payload.get("best_params")
        else:
            # Allow bare format: {"task_idx": i, ...params...}
            task_idx = payload.get("task_idx", None)
            best_params = {k: v for k, v in payload.items() if k != "task_idx"}
        if task_idx is None:
            raise ValueError(f"CatBoost task params JSON {p} is missing 'task_idx'.")
        t = int(task_idx)
        if t < 0 or t > 3:
            raise ValueError(f"CatBoost task_idx in {p} must be in [0,3], got {t}.")
        key = f"task_{t}"
        if key in out:
            raise ValueError(f"Duplicate CatBoost task_idx={t} across provided JSON files.")
        if not isinstance(best_params, dict):
            raise ValueError(f"CatBoost task params JSON {p} has invalid 'best_params'.")
        out[key] = dict(best_params)
    missing = [f"task_{t}" for t in range(4) if f"task_{t}" not in out]
    if missing:
        raise ValueError(
            "catboost_task_params_jsons must provide all 4 tasks. "
            f"Missing: {missing}"
        )
    return out


def _load_catboost_task_iterations_overrides(paths: Sequence[str]) -> Dict[int, int]:
    out: Dict[int, int] = {}

    def _coerce_positive_int(value: Any, *, field: str, src: Path) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field} in {src}: bool is not allowed.")
        try:
            iv = int(value)
        except Exception as exc:
            raise ValueError(
                f"Invalid {field} in {src}: expected integer-like value, got {value!r}."
            ) from exc
        if iv < 1:
            raise ValueError(f"Invalid {field} in {src}: expected >= 1, got {iv}.")
        return iv

    for raw_path in paths:
        p = Path(str(raw_path))
        if not p.exists():
            raise FileNotFoundError(f"CatBoost final-iterations JSON not found: {p}")
        payload = json.loads(p.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid CatBoost final-iterations JSON {p}: expected object.")
        fam = payload.get("family")
        if isinstance(fam, str) and str(fam) != "catboost_st":
            raise ValueError(f"Family mismatch in {p}: expected 'catboost_st', got '{fam}'.")
        task_idx = payload.get("task_idx", None)
        if task_idx is None:
            raise ValueError(f"CatBoost final-iterations JSON {p} is missing 'task_idx'.")
        t = int(task_idx)
        if t < 0 or t > 3:
            raise ValueError(f"CatBoost task_idx in {p} must be in [0,3], got {t}.")
        if t in out:
            raise ValueError(f"Duplicate CatBoost task_idx={t} across final-iterations JSON files.")
        if "selected_iterations" in payload:
            sel = _coerce_positive_int(payload["selected_iterations"], field="selected_iterations", src=p)
        elif "target_iterations" in payload:
            sel = _coerce_positive_int(payload["target_iterations"], field="target_iterations", src=p)
        elif "iterations" in payload:
            sel = _coerce_positive_int(payload["iterations"], field="iterations", src=p)
        else:
            raise ValueError(
                f"Invalid CatBoost final-iterations JSON {p}: "
                "missing one of ['selected_iterations', 'target_iterations', 'iterations']."
            )
        out[int(t)] = int(sel)

    missing = [int(t) for t in range(4) if int(t) not in out]
    if missing:
        raise ValueError(
            "catboost_final_iterations_jsons must provide all 4 tasks. "
            f"Missing task_idx: {missing}"
        )
    return out


def _mt_2d_fixed_inactive_params() -> Dict[str, Any]:
    # 2D-only family: keep 3D/attention branch knobs fixed and explicit.
    return {
        "inst_hidden": 256,
        "inst_layers": 3,
        "inst_dropout": 0.05,
        "attn_heads": 8,
        "attn_dropout": 0.05,
        "inst_embedder_name": "mlp_v3_3d",
        "aggregator_name": "task_attention_pool",
    }


def _search_space_mt_2d(trial: optuna.Trial) -> Dict[str, Any]:
    # Family-specific MIL search space for 2D-only model.
    # Excludes inactive 3D/attention suggestions to keep trials meaningful.
    p: Dict[str, Any] = {
        "mol_hidden": trial.suggest_categorical("mol_hidden", [128, 256, 512]),
        "mol_layers": trial.suggest_int("mol_layers", 2, 5),
        "mol_dropout": trial.suggest_float("mol_dropout", 0.01, 0.25),
        "proj_dim": trial.suggest_categorical("proj_dim", [256, 512]),
        "mixer_hidden": trial.suggest_categorical("mixer_hidden", [128, 256, 512]),
        "mixer_layers": trial.suggest_int("mixer_layers", 2, 5),
        "mixer_dropout": trial.suggest_float("mixer_dropout", 0.01, 0.2),
        "mol_embedder_name": trial.suggest_categorical("mol_embedder_name", ["mlp_v3_2d"]),
        "predictor_name": trial.suggest_categorical("predictor_name", ["mlp_v3"]),
        "head_num_layers": trial.suggest_int("head_num_layers", 2, 4),
        "head_dropout": trial.suggest_float("head_dropout", 0.01, 0.2),
        "head_fc2_gain_non_last": trial.suggest_float("head_fc2_gain_non_last", 1e-3, 1e-2),
        "activation": trial.suggest_categorical("activation", ["GELU", "ReLU", "LeakyReLU"]),
        "lr": trial.suggest_float("lr", 8e-5, 8e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 3e-6, 3e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "posw_clip_t0": trial.suggest_float("posw_clip_t0", 12.0, 28.0, log=True),
        "posw_clip_t1": trial.suggest_float("posw_clip_t1", 35.0, 90.0, log=True),
        "posw_clip_t2": trial.suggest_float("posw_clip_t2", 3.0, 10.0, log=True),
        "posw_clip_t3": trial.suggest_float("posw_clip_t3", 90.0, 220.0, log=True),
        "gamma_t0": trial.suggest_float("gamma_t0", 0.5, 2.0),
        "gamma_t1": trial.suggest_float("gamma_t1", 1.0, 3.0),
        "gamma_t2": trial.suggest_float("gamma_t2", 0.0, 1.5),
        "gamma_t3": trial.suggest_float("gamma_t3", 1.5, 4.0),
        "rare_oversample_mult": trial.suggest_float("rare_oversample_mult", 2.0, 10.0),
        "rare_target_prev": trial.suggest_float("rare_target_prev", 0.06, 0.12),
        "sample_weight_cap": trial.suggest_float("sample_weight_cap", 6.0, 9.0),
        "lam_t0": trial.suggest_float("lam_t0", 0.6, 1.6, log=True),
        "lam_t1": trial.suggest_float("lam_t1", 1.0, 2.4, log=True),
        "lam_t2": trial.suggest_float("lam_t2", 0.25, 0.9, log=True),
        "lam_t3": trial.suggest_float("lam_t3", 1.8, 3.5, log=True),
        "lam_floor": trial.suggest_float("lam_floor", 0.35, 0.85),
        "lam_ceil": trial.suggest_float("lam_ceil", 1.30, 2.20),
        "lambda_aux_abs": trial.suggest_float("lambda_aux_abs", 0.05, 0.3),
        "lambda_aux_fluo": trial.suggest_float("lambda_aux_fluo", 0.05, 0.3),
        "lambda_aux_bitmask": trial.suggest_float("lambda_aux_bitmask", 0.02, 0.1),
        "reg_loss_type": trial.suggest_categorical("reg_loss_type", ["mse"]),
        "min_w": 0.40,
        "accumulate_grad_batches": trial.suggest_categorical("accumulate_grad_batches", [8, 16]),
        "head_stochastic_depth": trial.suggest_float("head_stochastic_depth", 0.0, 0.1),
    }
    p.update(_mt_2d_fixed_inactive_params())
    return p


def _mil_search_space_for_family(*, trial: optuna.Trial, family: str) -> Dict[str, Any]:
    if str(family) == "mt_2d":
        return _search_space_mt_2d(trial)
    if str(family) in {"mt_2d3d", "mt_3d"}:
        # 3D-bearing families are the most memory intensive; keep per-step batch conservative.
        return search_space(trial, batch_choices=(128, 256, 512))
    return search_space(trial, batch_choices=(256, 512, 1024))


class _MILMacroCrossValidator:
    def __init__(self, *, data: MILCVData, run_config: CVRunConfig, family: str):
        self.data = data
        self.run_config = run_config
        self.family = str(family)

    def evaluate_trial(self, trial: optuna.Trial) -> float:
        log_event("START", "family.hpo.trial.evaluate", trial=int(trial.number), family=str(self.family))
        params = _mil_search_space_for_family(trial=trial, family=str(self.family))
        # Keep fixed objective weighting across MIL-family trials.
        params["min_w"] = 0.4
        cfg = HPOConfig.from_params(params)

        fold_runner = MILFoldTrainer(
            trial=trial,
            hpo_config=cfg,
            data=self.data,
            run_config=self.run_config,
        )
        fold_scores: List[float] = []
        fold_detail: Dict[str, Any] = {}
        for step, (tr, va, fold_id) in enumerate(self.data.folds_info):
            try:
                fold_score, detail = fold_runner.run_fold(
                    train_idx=np.asarray(tr, dtype=np.int64),
                    val_idx=np.asarray(va, dtype=np.int64),
                    fold_id=int(fold_id),
                )
            except torch.cuda.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                log_event(
                    "WARN",
                    "family.hpo.trial.pruned_oom",
                    trial=int(trial.number),
                    family=str(self.family),
                    fold=int(fold_id),
                    error=repr(exc),
                )
                raise optuna.TrialPruned(
                    f"OOM in family={self.family} trial={int(trial.number)} fold={int(fold_id)}"
                ) from exc
            except RuntimeError as exc:
                msg = str(exc).lower()
                if ("out of memory" in msg) and ("cuda" in msg):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    log_event(
                        "WARN",
                        "family.hpo.trial.pruned_oom",
                        trial=int(trial.number),
                        family=str(self.family),
                        fold=int(fold_id),
                        error=repr(exc),
                    )
                    raise optuna.TrialPruned(
                        f"OOM in family={self.family} trial={int(trial.number)} fold={int(fold_id)}"
                    ) from exc
                raise
            objective_mode = str(cfg.objective.mode)
            if objective_mode in {"macro_pr_auc", "macro_ap"}:
                trial_score = float(detail.get("macro_pr_auc_best_epoch", detail.get("macro_ap_best_epoch", 0.0)))
            else:
                trial_score = float(fold_score)
            fold_scores.append(float(trial_score))
            detail["objective_value"] = float(trial_score)
            fold_detail[str(fold_id)] = detail
            trial.report(float(np.mean(fold_scores)), step=int(step))
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_detail", fold_detail)
        log_event(
            "DONE",
            "family.hpo.trial.evaluate",
            trial=int(trial.number),
            family=str(self.family),
            mean_score=f"{mean_score:.6f}",
        )
        return mean_score


def _catboost_search_space(trial: optuna.Trial, *, task_idx: int) -> Dict[str, Any]:
    # Prevalence-aware clipping ranges for scale_pos_weight:
    # t0~5.6%, t1~1.5%, t2~16.7%, t3~0.24%.
    if int(task_idx) == 0:
        posw_lo, posw_hi = 10.0, 70.0
    elif int(task_idx) == 1:
        posw_lo, posw_hi = 90.0, 250.0
    elif int(task_idx) == 2:
        posw_lo, posw_hi = 10.0, 70.0
    else:  # t3
        posw_lo, posw_hi = 90.0, 300.0
    p = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 80),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        "pos_weight_clip": trial.suggest_float("pos_weight_clip", posw_lo, posw_hi, log=True),
    }
    if p["bootstrap_type"] == "Bayesian":
        p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    else:
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    return p


def _catboost_common_params(*, params: Mapping[str, Any], seed: int, threads: int, scale_pos_weight: float) -> Dict[str, Any]:
    cb = dict(
        iterations=int(_resolve_catboost_iterations(params)),
        loss_function="Logloss",
        eval_metric="PRAUC",
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        min_data_in_leaf=int(params["min_data_in_leaf"]),
        random_strength=float(params["random_strength"]),
        rsm=float(params["rsm"]),
        bootstrap_type=str(params["bootstrap_type"]),
        random_seed=int(seed),
        scale_pos_weight=float(scale_pos_weight),
        verbose=False,
        task_type="CPU",
        thread_count=int(max(1, threads)),
        allow_writing_files=False,
    )
    if "bagging_temperature" in params:
        cb["bagging_temperature"] = float(params["bagging_temperature"])
    if "subsample" in params:
        cb["subsample"] = float(params["subsample"])
    return cb


def _calibrate_task(
    y: np.ndarray,
    p: np.ndarray,
    method: str,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    yb = np.asarray(y, dtype=int).reshape(-1)
    pp = _clip_prob(p).reshape(-1)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if sw is not None:
        if int(sw.shape[0]) != int(yb.shape[0]):
            raise ValueError(
                f"sample_weight length mismatch: got {int(sw.shape[0])}, expected {int(yb.shape[0])}"
            )
        sw = np.nan_to_num(sw, nan=0.0, posinf=0.0, neginf=0.0)
        sw = np.clip(sw, 0.0, np.inf)
        if float(np.sum(sw)) <= 0.0:
            sw = None
    if int(np.unique(yb).size) < 2:
        return pp.astype(np.float32), {"kind": "identity_single_class"}
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        out = ir.fit_transform(pp, yb, sample_weight=sw)
        return np.asarray(out, dtype=np.float32), {
            "kind": "isotonic",
            "x_thresholds": [float(x) for x in ir.X_thresholds_.tolist()],
            "y_thresholds": [float(x) for x in ir.y_thresholds_.tolist()],
        }
    if method == "temperature":
        logits = np.log(pp / (1.0 - pp))
        best_t = 1.0
        best_nll = float("inf")
        for t in np.exp(np.linspace(np.log(0.25), np.log(4.0), 96)):
            q = 1.0 / (1.0 + np.exp(-logits / float(t)))
            q = _clip_prob(q)
            try:
                nll = float(log_loss(yb, q, labels=[0, 1], sample_weight=sw))
            except Exception:
                continue
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        out = 1.0 / (1.0 + np.exp(-logits / best_t))
        return np.asarray(_clip_prob(out), dtype=np.float32), {"kind": "temperature", "temperature": float(best_t)}
    # default platt (score-space logistic calibration):
    # fit y ~ sigmoid(A * score + B), where score is the model score.
    # Here score=probability output (predict_proba[:,1]) because all families expose probabilities.
    x = pp.reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    try:
        lr.fit(x, yb, sample_weight=sw)
    except Exception:
        # Fallback to identity if calibration fit is unstable for this task.
        return np.asarray(pp, dtype=np.float32), {"kind": "identity_fit_failed"}
    out = lr.predict_proba(x)[:, 1]
    return np.asarray(_clip_prob(out), dtype=np.float32), {
        "kind": "platt",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
        "input_space": "raw_score_prob",
    }


def _apply_calibration_task(p: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    pp = _clip_prob(np.asarray(p, dtype=np.float64).reshape(-1))
    kind = str(params.get("kind", "identity"))
    if kind in {"identity_single_class", "identity", "identity_fit_failed"}:
        return np.asarray(pp, dtype=np.float32)
    if kind == "isotonic":
        xs = np.asarray(params.get("x_thresholds", []), dtype=np.float64).reshape(-1)
        ys = np.asarray(params.get("y_thresholds", []), dtype=np.float64).reshape(-1)
        if int(xs.size) < 2 or int(ys.size) < 2 or int(xs.size) != int(ys.size):
            return np.asarray(pp, dtype=np.float32)
        out = np.interp(pp, xs, ys, left=float(ys[0]), right=float(ys[-1]))
        return np.asarray(_clip_prob(out), dtype=np.float32)
    logits = np.log(pp / (1.0 - pp))
    if kind == "temperature":
        t = float(params.get("temperature", 1.0))
        t = max(t, 1e-6)
        out = 1.0 / (1.0 + np.exp(-logits / t))
        return np.asarray(_clip_prob(out), dtype=np.float32)
    if kind == "platt":
        coef = float(params.get("coef", 1.0))
        intercept = float(params.get("intercept", 0.0))
        # Backward-compatibility:
        # old calibration JSONs used logit(prob) as score; new ones use raw probability score.
        input_space = str(params.get("input_space", "logit_prob"))
        score = pp if input_space in {"raw_score_prob", "raw_score", "raw_prob", "prob"} else logits
        out = 1.0 / (1.0 + np.exp(-(coef * score + intercept)))
        return np.asarray(_clip_prob(out), dtype=np.float32)
    return np.asarray(pp, dtype=np.float32)


def _apply_blend_task(*, x: np.ndarray, task_cfg: Mapping[str, Any], fams: Sequence[str]) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    kind = str(task_cfg.get("kind", "convex_blending"))
    if kind == "logistic_stacking":
        weights = task_cfg.get("weights", {})
        intercept = float(task_cfg.get("intercept", 0.0))
        w = np.asarray([float(weights.get(f, 0.0)) for f in fams], dtype=np.float64).reshape(1, -1)
        logits = (x_arr * w).sum(axis=1) + intercept
        out = 1.0 / (1.0 + np.exp(-logits))
        return np.asarray(_clip_prob(out), dtype=np.float32)
    if kind == "convex_blending":
        weights = task_cfg.get("weights", {})
        w = np.asarray([float(weights.get(f, 0.0)) for f in fams], dtype=np.float64)
        if float(np.sum(w)) <= 0.0:
            w = np.ones(len(fams), dtype=np.float64)
        w = np.clip(w, 0.0, np.inf)
        w = w / float(max(np.sum(w), 1e-12))
        out = np.dot(x_arr, w.reshape(-1, 1)).reshape(-1)
        return np.asarray(_clip_prob(out), dtype=np.float32)
    weights = task_cfg.get("weights", {})
    w = np.asarray([float(weights.get(f, 0.0)) for f in fams], dtype=np.float64)
    if float(np.sum(np.abs(w))) <= 0.0:
        w = np.ones(len(fams), dtype=np.float64)
    w = w / float(np.sum(w))
    out = np.dot(x_arr, w.reshape(-1, 1)).reshape(-1)
    return np.asarray(_clip_prob(out), dtype=np.float32)


def _require_calibrated_prob_columns(*, df: pd.DataFrame, family: str, scope: str) -> None:
    missing = [f"p_cal_t{int(t)}" for t in range(4) if f"p_cal_t{int(t)}" not in df.columns]
    if missing:
        raise RuntimeError(
            f"Blending requires calibrated probabilities, missing columns for family={family} scope={scope}: {missing}"
        )


def _fit_blend_task(
    *,
    x: np.ndarray,
    y: np.ndarray,
    fams: Sequence[str],
    sample_weight: np.ndarray | None,
) -> tuple[Dict[str, Any], np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if sw is not None:
        if int(sw.shape[0]) != int(y_arr.shape[0]):
            raise ValueError(
                f"blend sample_weight length mismatch: got {int(sw.shape[0])}, expected {int(y_arr.shape[0])}"
            )
        sw = np.nan_to_num(sw, nan=0.0, posinf=0.0, neginf=0.0)
        sw = np.clip(sw, 0.0, np.inf)
        if float(np.sum(sw)) <= 0.0:
            sw = None

    if int(np.unique(y_arr).size) < 2:
        w_uni = np.ones(len(fams), dtype=np.float64) / float(max(1, len(fams)))
        p_task = np.clip(np.dot(x_arr, w_uni), 1e-6, 1 - 1e-6)
        return (
            {
                "kind": "uniform_single_class",
                "weights": {f: float(w_uni[i]) for i, f in enumerate(fams)},
                "intercept": 0.0,
            },
            p_task,
        )

    # Constrained convex blending:
    #   p = sum_i w_i * p_i, with w_i >= 0 and sum_i w_i = 1.
    # Objective here is direct weighted ROC-AUC maximization on train OOF.
    # We use deterministic randomized search + local refinement on the simplex.
    n_models = int(x_arr.shape[1])
    sw_local = np.ones((x_arr.shape[0],), dtype=np.float64) if sw is None else sw
    sw_sum = float(np.sum(sw_local))
    if sw_sum <= 0.0:
        sw_local = np.ones((x_arr.shape[0],), dtype=np.float64)
        sw_sum = float(np.sum(sw_local))

    def _normalize_simplex(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        vv = np.clip(np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf)
        s = float(np.sum(vv))
        if not np.isfinite(s) or s <= 0.0:
            return np.ones((n_models,), dtype=np.float64) / float(max(1, n_models))
        return vv / s

    def _eval_weights(w_vec: np.ndarray) -> tuple[float, float, np.ndarray]:
        wv = _normalize_simplex(w_vec)
        p_vec = _clip_prob(np.dot(x_arr, wv.reshape(-1, 1)).reshape(-1))
        try:
            auc = float(roc_auc_score(y_arr, p_vec, sample_weight=sw_local))
        except Exception:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y_arr, p_vec, sample_weight=sw_local))
        except Exception:
            ap = float("nan")
        return auc, ap, p_vec

    seed_basis = int(
        (int(x_arr.shape[0]) * 17 + int(x_arr.shape[1]) * 97 + int(np.sum(y_arr)) * 131) % (2**31 - 1)
    )
    rng = np.random.RandomState(seed_basis)

    candidates: List[np.ndarray] = []
    candidates.append(np.ones((n_models,), dtype=np.float64) / float(max(1, n_models)))  # uniform
    for i in range(n_models):  # one-hot
        v = np.zeros((n_models,), dtype=np.float64)
        v[i] = 1.0
        candidates.append(v)
    # Pairwise mixes are cheap and improve robustness when one model dominates.
    if n_models >= 2:
        alphas = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                for a in alphas.tolist():
                    v = np.zeros((n_models,), dtype=np.float64)
                    v[i] = float(a)
                    v[j] = float(1.0 - a)
                    candidates.append(v)
    # Random simplex samples for broader search.
    n_random = int(max(1024, 256 * n_models))
    for _ in range(n_random):
        candidates.append(rng.dirichlet(np.ones((n_models,), dtype=np.float64)))

    best_w = candidates[0]
    best_auc = float("-inf")
    best_ap = float("-inf")
    best_p = _clip_prob(np.dot(x_arr, _normalize_simplex(best_w).reshape(-1, 1)).reshape(-1))
    n_eval = 0
    for cand in candidates:
        auc, ap, p_vec = _eval_weights(cand)
        n_eval += 1
        if (np.isfinite(auc) and (auc > best_auc + 1e-12)) or (
            np.isfinite(auc) and abs(auc - best_auc) <= 1e-12 and np.isfinite(ap) and ap > best_ap
        ):
            best_auc = float(auc)
            best_ap = float(ap) if np.isfinite(ap) else float(best_ap)
            best_w = _normalize_simplex(cand)
            best_p = p_vec

    # Local refinement around best candidate.
    for sigma, n_steps in ((0.10, 320), (0.05, 320), (0.02, 240), (0.01, 160)):
        for _ in range(int(n_steps)):
            cand = _normalize_simplex(best_w + rng.normal(loc=0.0, scale=float(sigma), size=n_models))
            auc, ap, p_vec = _eval_weights(cand)
            n_eval += 1
            if (np.isfinite(auc) and (auc > best_auc + 1e-12)) or (
                np.isfinite(auc) and abs(auc - best_auc) <= 1e-12 and np.isfinite(ap) and ap > best_ap
            ):
                best_auc = float(auc)
                best_ap = float(ap) if np.isfinite(ap) else float(best_ap)
                best_w = cand
                best_p = p_vec

    p_task = np.asarray(best_p, dtype=np.float64)
    cfg = {
        "kind": "convex_blending",
        "weights": {f: float(best_w[i]) for i, f in enumerate(fams)},
        "constraint": "simplex_non_negative_sum1",
        "optimizer": "simplex_random_local_search",
        "objective": "roc_auc",
        "n_eval": int(max(1, n_eval)),
        "train_roc_auc": float(best_auc) if np.isfinite(best_auc) else float("nan"),
        "train_pr_auc_tiebreak": float(best_ap) if np.isfinite(best_ap) else float("nan"),
    }
    return cfg, p_task


def _log_oof_bitmask_coverage(*, family: str, df_oof: pd.DataFrame) -> None:
    y = np.stack([df_oof[f"y_t{t}"].to_numpy(dtype=np.int64) for t in range(4)], axis=1)
    folds = df_oof["fold_id"].to_numpy(dtype=np.int64)
    bm = bitmask_ids(y)
    all_masks = set([int(x) for x in np.unique(bm).tolist()])
    for fold_id in sorted([int(x) for x in np.unique(folds).tolist() if int(x) >= 0]):
        m = folds == int(fold_id)
        if int(np.sum(m)) <= 0:
            continue
        fold_masks = set([int(x) for x in np.unique(bm[m]).tolist()])
        missing = sorted(list(all_masks - fold_masks))
        cov = float(len(fold_masks) / float(max(1, len(all_masks))))
        pos = y[m].mean(axis=0)
        log_event(
            "INFO",
            "family.cv.oof.bitmask_coverage.fold",
            family=str(family),
            fold=int(fold_id),
            n_rows=int(np.sum(m)),
            n_masks_total=int(len(all_masks)),
            n_masks_fold=int(len(fold_masks)),
            coverage=f"{cov:.3f}",
            pos_t0=f"{float(pos[0]):.6f}",
            pos_t1=f"{float(pos[1]):.6f}",
            pos_t2=f"{float(pos[2]):.6f}",
            pos_t3=f"{float(pos[3]):.6f}",
        )
        if len(missing) > 0:
            log_event(
                "WARN",
                "family.cv.oof.bitmask_coverage.missing",
                family=str(family),
                fold=int(fold_id),
                n_missing_masks=int(len(missing)),
                missing_masks=",".join([str(int(x)) for x in missing[:16]]),
            )


def _make_internal_val_indices(n: int, seed: int) -> np.ndarray:
    n_int = int(max(0, n))
    if n_int <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_int <= 4096:
        return np.arange(n_int, dtype=np.int64)
    k = int(min(max(1024, int(round(0.2 * n_int))), 8192))
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(n_int, size=int(k), replace=False)
    return np.sort(np.asarray(idx, dtype=np.int64))


def _run_mil_final_train_and_predict(
    *,
    cfg: FamilySuiteConfig,
    family_data: _MILFamilyData,
    best_params: Mapping[str, Any],
    outdir: Path,
    write_outputs: bool = True,
    write_explainability_outputs: bool = True,
    output_prefix: str = "leaderboard",
    fixed_train_epochs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    set_all_seeds(int(cfg.seed))
    hpo_cfg = HPOConfig.from_params(
        dict(best_params),
        fallback_lambda_power=1.0,
        fallback_lam_floor=0.25,
        fallback_lam_ceil=6.0,
        fallback_pos_weight_clip=50.0,
    )
    lam = compute_lam(hpo_cfg.loss, y_train=family_data.y_cls_train)
    posw = pos_weight_per_task(
        family_data.y_cls_train,
        clip=compute_posw_clips(hpo_cfg.loss),
    )
    gamma_t = compute_gamma(hpo_cfg.loss)

    train_idx_all = np.arange(len(family_data.y_abs_train), dtype=np.int64)
    mu_abs, sd_abs = fit_standardizer(
        family_data.y_abs_train,
        family_data.m_abs_train,
        train_idx_all,
    )
    mu_f, sd_f = fit_standardizer(
        family_data.y_fluo_train,
        family_data.m_fluo_train,
        train_idx_all,
    )
    y_abs_tr_sc = apply_standardizer(family_data.y_abs_train, mu_abs, sd_abs)
    y_abs_lb_sc = apply_standardizer(family_data.y_abs_lb, mu_abs, sd_abs)
    y_fluo_tr_sc = apply_standardizer(family_data.y_fluo_train, mu_f, sd_f)
    y_fluo_lb_sc = apply_standardizer(family_data.y_fluo_lb, mu_f, sd_f)

    w_cls_tr = np.asarray(family_data.w_cls_train, dtype=np.float32).copy()

    ds_tr = MILTrainDataset(
        family_data.ids_train,
        family_data.X2d_train,
        family_data.y_cls_train,
        w_cls_tr,
        y_abs_tr_sc,
        family_data.m_abs_train,
        family_data.w_abs_train,
        y_fluo_tr_sc,
        family_data.m_fluo_train,
        family_data.w_fluo_train,
        family_data.starts,
        family_data.counts,
        family_data.id2pos,
        family_data.Xinst_sorted,
        max_instances=0,
        seed=int(cfg.seed) + 7,
    )
    val_internal_idx = _make_internal_val_indices(len(family_data.ids_train), seed=int(cfg.seed) + 27183)
    ds_val_internal = MILTrainDataset(
        [family_data.ids_train[int(i)] for i in val_internal_idx.tolist()],
        family_data.X2d_train[val_internal_idx],
        family_data.y_cls_train[val_internal_idx],
        family_data.w_cls_train[val_internal_idx],
        y_abs_tr_sc[val_internal_idx],
        family_data.m_abs_train[val_internal_idx],
        family_data.w_abs_train[val_internal_idx],
        y_fluo_tr_sc[val_internal_idx],
        family_data.m_fluo_train[val_internal_idx],
        family_data.w_fluo_train[val_internal_idx],
        family_data.starts,
        family_data.counts,
        family_data.id2pos,
        family_data.Xinst_sorted,
        max_instances=0,
        seed=int(cfg.seed) + 11,
    )
    ds_eval = MILTrainDataset(
        family_data.ids_lb,
        family_data.X2d_lb,
        family_data.y_cls_lb,
        family_data.w_cls_lb,
        y_abs_lb_sc,
        family_data.m_abs_lb,
        family_data.w_abs_lb,
        y_fluo_lb_sc,
        family_data.m_fluo_lb,
        family_data.w_fluo_lb,
        family_data.starts,
        family_data.counts,
        family_data.id2pos,
        family_data.Xinst_sorted,
        max_instances=0,
        seed=int(cfg.seed) + 17,
    )

    loader_builder = DataLoaderBuilder(
        LoaderConfig(num_workers=int(cfg.num_workers), pin_memory=bool(cfg.pin_memory and torch.cuda.is_available()))
    )
    if bool(hpo_cfg.sampler.use_balanced_batch_sampler):
        batch_sampler = make_balanced_batch_sampler(
            family_data.y_cls_train,
            batch_size=int(hpo_cfg.runtime.batch_size),
            rare_mult=float(hpo_cfg.sampler.rare_oversample_mult),
            rare_target_prev=float(hpo_cfg.sampler.rare_target_prev),
            sample_weight_cap=float(hpo_cfg.sampler.sample_weight_cap),
            batch_pos_fraction=float(hpo_cfg.sampler.batch_pos_fraction),
            min_pos_per_batch=int(hpo_cfg.sampler.min_pos_per_batch),
            rare_prev_thr=hpo_cfg.sampler.rare_prev_thr,
            seed=int(cfg.seed) + 1000,
        )
        dl_tr = loader_builder.train_loader(
            ds_tr,
            batch_size=int(hpo_cfg.runtime.batch_size),
            batch_sampler=batch_sampler,
            collate_fn=collate_train,
        )
    else:
        sampler = make_weighted_sampler(
            family_data.y_cls_train,
            rare_mult=float(hpo_cfg.sampler.rare_oversample_mult),
            rare_target_prev=float(hpo_cfg.sampler.rare_target_prev),
            sample_weight_cap=float(hpo_cfg.sampler.sample_weight_cap),
            rare_prev_thr=hpo_cfg.sampler.rare_prev_thr,
        )
        dl_tr = loader_builder.train_loader(
            ds_tr,
            batch_size=int(hpo_cfg.runtime.batch_size),
            sampler=sampler,
            collate_fn=collate_train,
        )
    dl_val_internal = loader_builder.eval_loader(
        ds_val_internal,
        batch_size=min(128, int(hpo_cfg.runtime.batch_size)),
        collate_fn=collate_train,
    )
    dl_eval = loader_builder.eval_loader(
        ds_eval,
        batch_size=min(128, int(hpo_cfg.runtime.batch_size)),
        collate_fn=collate_train,
    )

    bitmask_group_top_ids, bitmask_group_class_weight = build_bitmask_group_definition(
        family_data.y_cls_train,
        top_k=int(hpo_cfg.loss.bitmask_group_top_k),
        class_weight_alpha=float(hpo_cfg.loss.bitmask_group_weight_alpha),
        class_weight_cap=float(hpo_cfg.loss.bitmask_group_weight_cap),
    )
    model = MILModelBuilder.build(
        config=hpo_cfg,
        mol_dim=int(family_data.X2d_train.shape[1]),
        inst_dim=int(family_data.Xinst_sorted.shape[1]),
        inst_geom_dim=int(family_data.inst_geom_dim),
        inst_qm_dim=int(family_data.inst_qm_dim),
        pos_weight=posw,
        gamma=gamma_t,
        lam=lam,
        bitmask_group_top_ids=bitmask_group_top_ids,
        bitmask_group_class_weight=bitmask_group_class_weight,
    )

    family_dir = outdir / family_data.family
    family_dir.mkdir(parents=True, exist_ok=True)
    sampler_diag_path = family_dir / f"{output_prefix}_sampler_diagnostics.csv"
    sampler_diag_df = build_sampler_diagnostics_df(
        y=family_data.y_cls_train,
        batch_size=int(hpo_cfg.runtime.batch_size),
        use_balanced_batch_sampler=bool(hpo_cfg.sampler.use_balanced_batch_sampler),
        rare_mult=float(hpo_cfg.sampler.rare_oversample_mult),
        rare_target_prev=float(hpo_cfg.sampler.rare_target_prev),
        sample_weight_cap=float(hpo_cfg.sampler.sample_weight_cap),
        batch_pos_fraction=float(hpo_cfg.sampler.batch_pos_fraction),
        min_pos_per_batch=int(hpo_cfg.sampler.min_pos_per_batch),
        rare_prev_thr=hpo_cfg.sampler.rare_prev_thr,
        seed=int(cfg.seed) + 1000,
    )
    sampler_diag_df.to_csv(sampler_diag_path, index=False)
    _diag_pick = sampler_diag_df.set_index(["section", "metric"])["value"].to_dict()
    log_event(
        "INFO",
        "family.final.mil.sampler_diagnostics",
        family=str(family_data.family),
        output_prefix=str(output_prefix),
        path=str(sampler_diag_path),
        sampler_mode=str("balanced_batch" if bool(hpo_cfg.sampler.use_balanced_batch_sampler) else "weighted_sampler"),
        mean_pos_per_batch=f"{float(_diag_pick.get(('summary', 'mean_pos_per_batch'), float('nan'))):.3f}",
        duplicate_rate=f"{float(_diag_pick.get(('summary', 'duplicate_rate'), float('nan'))):.6f}",
    )
    use_fixed_epochs = fixed_train_epochs is not None
    target_epochs = int(fixed_train_epochs) if use_fixed_epochs else int(cfg.max_epochs)
    target_patience = int(max(int(cfg.patience), target_epochs + 5)) if use_fixed_epochs else int(cfg.patience)
    trainer_cfg = LightningTrainerConfig(
        max_epochs=int(target_epochs),
        patience=int(target_patience),
        accelerator=str(cfg.nn_accelerator),
        devices=int(cfg.nn_devices),
        precision=str(cfg.precision),
        accumulate_grad_batches=int(hpo_cfg.runtime.accumulate_grad_batches),
        save_checkpoint=bool(not use_fixed_epochs),
        save_weights_only=True,
    )
    trainer, ckpt_cb = LightningTrainerFactory(trainer_cfg).build(
        ckpt_dir=str(family_dir),
        trial=None,
    )
    trainer.fit(model, dl_tr, dl_val_internal)
    epochs_trained = int(trainer.current_epoch) + 1
    best_epoch: int | None = None
    if ckpt_cb is not None:
        best_path = ckpt_cb.best_model_path
        if best_path and Path(best_path).exists():
            ckpt = torch.load(best_path, map_location="cpu")
            best_epoch = int(ckpt.get("epoch", -1))
            model.load_state_dict(ckpt["state_dict"], strict=True)
    if best_epoch is None:
        best_epoch = max(0, int(epochs_trained) - 1)

    eval_device = _resolve_device(cfg.nn_accelerator)
    evaluator = ModelEvaluator(device=eval_device)
    macro_ap, aps, macro_auc, aucs = evaluator.eval_best_epoch(model, dl_eval)
    eval_json = {
        "macro_pr_auc": float(macro_ap),
        "macro_roc_auc": float(macro_auc),
        "pr_aucs": [float(x) for x in aps],
        "roc_aucs": [float(x) for x in aucs],
    }
    if bool(write_outputs):
        (family_dir / f"{output_prefix}_eval.json").write_text(json.dumps(eval_json, indent=2))
        model_artifact_path = save_mil_model_artifact(
            path=family_dir / f"{output_prefix}_model.pt",
            model=model,
            best_params=best_params,
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
            train_info={
                "use_fixed_epochs": bool(use_fixed_epochs),
                "target_epochs": int(target_epochs),
                "target_patience": int(target_patience),
                "epochs_trained": int(epochs_trained),
                "best_epoch": int(best_epoch),
            },
            metadata={
                "family": str(family_data.family),
                "output_prefix": str(output_prefix),
                "mol_dim": int(family_data.X2d_train.shape[1]),
                "inst_dim": int(family_data.Xinst_sorted.shape[1]),
                "inst_geom_dim": int(family_data.inst_geom_dim),
                "inst_qm_dim": int(family_data.inst_qm_dim),
                "task_cols": [str(x) for x in TASK_COLS],
            },
        )
        log_event(
            "INFO",
            "family.final.mil.model_saved",
            family=str(family_data.family),
            output_prefix=str(output_prefix),
            path=str(model_artifact_path),
        )

    model.eval()
    model.to(eval_device)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in dl_eval:
            x2d, x3d, kpm, *_ = batch
            x2d = x2d.to(eval_device, non_blocking=True)
            x3d = x3d.to(eval_device, non_blocking=True)
            kpm = kpm.to(eval_device, non_blocking=True)
            logits, _, _ = model(x2d, x3d, kpm, return_attn=False)
            p = torch.sigmoid(torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)).detach().cpu().numpy()
            preds.append(p.astype(np.float32))
    P = np.concatenate(preds, axis=0)
    Y = np.asarray(family_data.y_cls_lb, dtype=np.int64)
    W = np.asarray(family_data.w_cls_lb, dtype=np.float32)

    df_pred = pd.DataFrame({"ID": [str(x) for x in family_data.ids_lb]})
    for t in range(4):
        df_pred[f"p_t{t}"] = P[:, t]
        df_pred[f"y_t{t}"] = Y[:, t]
        df_pred[f"w_t{t}"] = W[:, t]
        df_pred[f"pred_t{t}"] = (P[:, t] >= 0.5).astype(int)
    metrics = _metric_table(y_true=Y, p_pred=P, w_cls=W)
    if bool(write_outputs):
        metrics.to_csv(outdir / f"{output_prefix}_metrics_{family_data.family}.csv", index=False)
        df_pred.to_csv(outdir / f"{output_prefix}_preds_{family_data.family}.csv", index=False)
        if bool(write_explainability_outputs) and (int(family_data.inst_geom_dim) > 0 or int(family_data.inst_qm_dim) > 0):
            export_ds = MILExportDataset(
                ids=[str(x) for x in family_data.ids_lb],
                X2d=np.asarray(family_data.X2d_lb, dtype=np.float32),
                starts=np.asarray(family_data.starts, dtype=np.int64),
                counts=np.asarray(family_data.counts, dtype=np.int64),
                id2pos={str(k): int(v) for k, v in family_data.id2pos.items()},
                Xinst_sorted=np.asarray(family_data.Xinst_sorted, dtype=np.float32),
                conf_sorted=np.asarray(family_data.conf_sorted),
                max_instances=0,
                seed=int(cfg.seed) + 123,
            )
            export_dl = loader_builder.eval_loader(
                export_ds,
                batch_size=min(64, int(hpo_cfg.runtime.batch_size)),
                collate_fn=collate_export,
            )
            true_labels_by_id = {
                str(family_data.ids_lb[i]): np.asarray(Y[i], dtype=np.float32).reshape(-1)
                for i in range(len(family_data.ids_lb))
            }
            attn_out_path = outdir / f"{output_prefix}_attention.csv"
            written_attn_path = export_leaderboard_attention(
                model=model,
                dl_lb_export=export_dl,
                device=eval_device,
                out_path=attn_out_path,
                true_labels_by_id=true_labels_by_id,
            )
            summary_paths = export_attention_dataset_summary(
                pred_table_path=written_attn_path,
            )
            log_event(
                "INFO",
                "family.final.mil.modality_export.done",
                family=str(family_data.family),
                path=str(written_attn_path),
                fusion_gate_summary_path=str(summary_paths["fusion_gate_summary"]),
                top_conformers_path=str(summary_paths["top_conformers"]),
            )

    train_info = {
        "family": str(family_data.family),
        "use_fixed_epochs": bool(use_fixed_epochs),
        "target_epochs": int(target_epochs),
        "target_patience": int(target_patience),
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch),
    }
    if bool(write_outputs):
        simple_model_path = family_dir / f"{output_prefix}_model.pth"
        model = model.to("cpu")
        torch.save(model, simple_model_path)
        log_event(
            "INFO",
            "family.final.mil.model_pickle_saved",
            family=str(family_data.family),
            output_prefix=str(output_prefix),
            path=str(simple_model_path),
        )
    del trainer, model, dl_tr, dl_val_internal, dl_eval, ds_tr, ds_val_internal, ds_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return df_pred, metrics, train_info


def _slice_mil_family_data_for_split(
    *,
    family_data: _MILFamilyData,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> _MILFamilyData:
    tr = np.asarray(train_idx, dtype=np.int64)
    ev = np.asarray(eval_idx, dtype=np.int64)
    return _MILFamilyData(
        family=str(family_data.family),
        ids_train=[str(family_data.ids_train[int(i)]) for i in tr.tolist()],
        ids_lb=[str(family_data.ids_train[int(i)]) for i in ev.tolist()],
        X2d_train=np.asarray(family_data.X2d_train[tr], dtype=np.float32),
        X2d_lb=np.asarray(family_data.X2d_train[ev], dtype=np.float32),
        y_cls_train=np.asarray(family_data.y_cls_train[tr], dtype=np.int64),
        y_cls_lb=np.asarray(family_data.y_cls_train[ev], dtype=np.int64),
        w_cls_train=np.asarray(family_data.w_cls_train[tr], dtype=np.float32),
        w_cls_lb=np.asarray(family_data.w_cls_train[ev], dtype=np.float32),
        y_abs_train=np.asarray(family_data.y_abs_train[tr], dtype=np.float32),
        m_abs_train=np.asarray(family_data.m_abs_train[tr], dtype=bool),
        w_abs_train=np.asarray(family_data.w_abs_train[tr], dtype=np.float32),
        y_abs_lb=np.asarray(family_data.y_abs_train[ev], dtype=np.float32),
        m_abs_lb=np.asarray(family_data.m_abs_train[ev], dtype=bool),
        w_abs_lb=np.asarray(family_data.w_abs_train[ev], dtype=np.float32),
        y_fluo_train=np.asarray(family_data.y_fluo_train[tr], dtype=np.float32),
        m_fluo_train=np.asarray(family_data.m_fluo_train[tr], dtype=bool),
        w_fluo_train=np.asarray(family_data.w_fluo_train[tr], dtype=np.float32),
        y_fluo_lb=np.asarray(family_data.y_fluo_train[ev], dtype=np.float32),
        m_fluo_lb=np.asarray(family_data.m_fluo_train[ev], dtype=bool),
        w_fluo_lb=np.asarray(family_data.w_fluo_train[ev], dtype=np.float32),
        folds_info=tuple(),
        starts=np.asarray(family_data.starts, dtype=np.int64),
        counts=np.asarray(family_data.counts, dtype=np.int64),
        id2pos={str(k): int(v) for k, v in family_data.id2pos.items()},
        Xinst_sorted=np.asarray(family_data.Xinst_sorted, dtype=np.float32),
        conf_sorted=np.asarray(family_data.conf_sorted),
        inst_geom_dim=int(family_data.inst_geom_dim),
        inst_qm_dim=int(family_data.inst_qm_dim),
    )


def _run_mil_oof_predictions(
    *,
    cfg: FamilySuiteConfig,
    family: str,
    family_data: _MILFamilyData,
    best_params: Mapping[str, Any],
    outdir: Path,
    fixed_fold_train_epochs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    n = int(len(family_data.ids_train))
    if n <= 0:
        raise ValueError(f"No train rows for family={family}")
    oof = np.full((n, 4), np.nan, dtype=np.float32)
    fold_ids = np.full((n,), -1, dtype=np.int64)
    fold_metric_rows: List[Dict[str, Any]] = []
    fold_best_epochs: List[int] = []
    cv_tmp_dir = outdir / "_cv_oof_tmp" / str(family)
    cv_tmp_dir.mkdir(parents=True, exist_ok=True)
    for fold_step, (tr, va, fold_id) in enumerate(family_data.folds_info):
        tr_idx = np.asarray(tr, dtype=np.int64)
        va_idx = np.asarray(va, dtype=np.int64)
        fold_ids[va_idx] = int(fold_id)
        log_event(
            "START",
            "family.cv.mil.fold",
            family=str(family),
            fold=int(fold_id),
            cv_step=int(fold_step),
            n_train=int(tr_idx.size),
            n_val=int(va_idx.size),
        )
        fold_data = _slice_mil_family_data_for_split(
            family_data=family_data,
            train_idx=tr_idx,
            eval_idx=va_idx,
        )
        df_fold, _, train_info = _run_mil_final_train_and_predict(
            cfg=cfg,
            family_data=fold_data,
            best_params=best_params,
            outdir=cv_tmp_dir,
            write_outputs=False,
            output_prefix=f"cv_fold{int(fold_id)}",
            fixed_train_epochs=(None if fixed_fold_train_epochs is None else int(fixed_fold_train_epochs)),
        )
        fold_best_epoch = int(train_info.get("best_epoch", max(0, int(train_info.get("epochs_trained", 1)) - 1)))
        fold_best_epochs.append(int(fold_best_epoch))
        log_event(
            "INFO",
            "family.cv.mil.fold.epoch",
            family=str(family),
            fold=int(fold_id),
            best_epoch=int(fold_best_epoch),
            epochs_trained=int(train_info.get("epochs_trained", 0)),
        )
        p_fold = np.stack([df_fold[f"p_t{t}"].to_numpy(dtype=np.float32) for t in range(4)], axis=1)
        oof[va_idx, :] = p_fold
        fold_metrics = _metric_table(
            y_true=np.asarray(family_data.y_cls_train[va_idx], dtype=np.int64),
            p_pred=np.asarray(p_fold, dtype=np.float64),
            w_cls=np.asarray(family_data.w_cls_train[va_idx], dtype=np.float32),
        )
        for row in fold_metrics.to_dict(orient="records"):
            row["fold"] = int(fold_id)
            row["family"] = str(family)
            fold_metric_rows.append(row)
        macro_row = fold_metrics[fold_metrics["task"] == "macro"]
        if len(macro_row) == 1:
            by_task = fold_metrics[fold_metrics["task"] != "macro"].set_index("task")

            def _metric(task_name: str, metric_name: str) -> str:
                if task_name in by_task.index:
                    try:
                        return f"{float(by_task.loc[task_name, metric_name]):.6f}"
                    except Exception:
                        return "nan"
                return "nan"

            log_event(
                "DONE",
                "family.cv.mil.fold",
                family=str(family),
                fold=int(fold_id),
                macro_pr_auc=f"{float(macro_row.iloc[0]['pr_auc']):.6f}",
                macro_roc_auc=f"{float(macro_row.iloc[0]['roc_auc']):.6f}",
                pr_auc_t0=_metric(str(TASK_COLS[0]), "pr_auc"),
                pr_auc_t1=_metric(str(TASK_COLS[1]), "pr_auc"),
                pr_auc_t2=_metric(str(TASK_COLS[2]), "pr_auc"),
                pr_auc_t3=_metric(str(TASK_COLS[3]), "pr_auc"),
                roc_auc_t0=_metric(str(TASK_COLS[0]), "roc_auc"),
                roc_auc_t1=_metric(str(TASK_COLS[1]), "roc_auc"),
                roc_auc_t2=_metric(str(TASK_COLS[2]), "roc_auc"),
                roc_auc_t3=_metric(str(TASK_COLS[3]), "roc_auc"),
            )
        else:
            log_event("DONE", "family.cv.mil.fold", family=str(family), fold=int(fold_id))
    if np.isnan(oof).any():
        missing = int(np.isnan(oof).sum())
        raise RuntimeError(f"OOF MIL predictions are incomplete for family={family}; missing={missing}")
    df_oof = pd.DataFrame({"ID": [str(x) for x in family_data.ids_train], "fold_id": fold_ids.astype(int)})
    for t in range(4):
        df_oof[f"p_t{t}"] = oof[:, t].astype(np.float32)
        df_oof[f"y_t{t}"] = np.asarray(family_data.y_cls_train[:, t], dtype=np.int64)
        df_oof[f"w_t{t}"] = np.asarray(family_data.w_cls_train[:, t], dtype=np.float32)
        df_oof[f"pred_t{t}"] = (df_oof[f"p_t{t}"] >= 0.5).astype(int)
    df_fold_metrics = pd.DataFrame(fold_metric_rows)
    if fixed_fold_train_epochs is not None:
        plus_one = [int(x) + 1 for x in fold_best_epochs]
        selected_epochs = int(max(1, int(fixed_fold_train_epochs)))
        selection_source = "override_fixed_epochs"
        selection_stat = "override_fixed"
        selected_from_plus_one: float | str = "override"
    else:
        plus_one = [int(x) + 1 for x in fold_best_epochs]
        selected_from_plus_one = (
            float(np.mean(np.asarray(plus_one, dtype=np.float64)))
            if plus_one
            else float(max(1, int(cfg.max_epochs)))
        )
        selected_epochs = int(max(1, int(round(float(selected_from_plus_one)))))
        selection_source = "cv_mean_best_epoch"
        selection_stat = "mean"
    summary = {
        "family": str(family),
        "fold_best_epochs": [int(x) for x in fold_best_epochs],
        "fold_best_plus_one": [int(x) for x in plus_one],
        "selected_epochs": int(selected_epochs),
        "selection_source": str(selection_source),
        "selection_stat": str(selection_stat),
    }
    log_event(
        "INFO",
        "family.cv.mil.epoch_selection",
        family=str(family),
        fold_best_epochs=",".join([str(int(x)) for x in fold_best_epochs]),
        selection_values=",".join([str(int(x)) for x in plus_one]),
        selected_epochs=int(selected_epochs),
        selection_stat=str(selection_stat),
        selected_from_fold_best_plus_one=(
            f"{float(selected_from_plus_one):.6f}"
            if isinstance(selected_from_plus_one, (float, int))
            else str(selected_from_plus_one)
        ),
        selection_source=str(selection_source),
    )
    return df_oof, df_fold_metrics, summary


def _run_catboost_oof_predictions(
    *,
    cfg: FamilySuiteConfig,
    df_train: pd.DataFrame,
    ids_2d_cat_file: Sequence[str],
    X2d_cat_file: np.ndarray,
    task_best_params: Mapping[str, Mapping[str, Any]],
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    from catboost import CatBoostClassifier

    ids_tr = df_train[cfg.id_col].astype(str).tolist()
    X2d_tr = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_tr)
    y_tr = coerce_binary_labels(df_train)
    w_tr = build_task_weights(df_train)
    folds = sorted(df_train[cfg.fold_col].dropna().astype(int).unique().tolist())
    folds_info = fold_indices(df_train, cfg.fold_col, folds)
    oof = np.full((len(ids_tr), 4), np.nan, dtype=np.float32)
    fold_ids = np.full((len(ids_tr),), -1, dtype=np.int64)
    fold_metric_rows: List[Dict[str, Any]] = []
    best_iters_by_task: Dict[int, List[int]] = {int(t): [] for t in range(4)}
    cap_hits_by_task: Dict[int, int] = {int(t): 0 for t in range(4)}
    folds_by_task: Dict[int, int] = {int(t): 0 for t in range(4)}
    for fold_step, (tr, va, fold_id) in enumerate(folds_info):
        tr_idx = np.asarray(tr, dtype=np.int64)
        va_idx = np.asarray(va, dtype=np.int64)
        fold_ids[va_idx] = int(fold_id)
        p_fold = np.zeros((len(va_idx), 4), dtype=np.float32)
        for t in range(4):
            key = f"task_{int(t)}"
            if key not in task_best_params:
                raise KeyError(f"Missing CatBoost params for {key}")
            p = dict(task_best_params[key])
            iter_cap = int(_resolve_catboost_iterations(p))
            pos = float(y_tr[tr_idx, t].sum())
            neg = float(len(tr_idx) - pos)
            spw = min(neg / max(pos, 1.0), float(p.get("pos_weight_clip", 100.0)))
            cb = CatBoostClassifier(
                **_catboost_common_params(
                    params=p,
                    seed=int(cfg.seed) + 97 * int(t) + 7919 * int(fold_step),
                    threads=max(1, int(cfg.cpu_workers)),
                    scale_pos_weight=float(spw),
                )
            )
            sw = (w_tr[tr_idx, t] if t in (0, 1) else None)
            cb.fit(
                X2d_tr[tr_idx],
                y_tr[tr_idx, t],
                sample_weight=sw,
                eval_set=(X2d_tr[va_idx], y_tr[va_idx, t]),
                use_best_model=True,
                early_stopping_rounds=200,
                verbose=False,
            )
            bi = int(cb.get_best_iteration())
            if bi < 0:
                bi = int(cb.tree_count_) - 1
            best_iter_1based = int(max(1, int(max(0, bi)) + 1))
            best_iters_by_task[int(t)].append(int(max(0, bi)))
            folds_by_task[int(t)] = int(folds_by_task[int(t)] + 1)
            if int(best_iter_1based) >= int(iter_cap):
                cap_hits_by_task[int(t)] = int(cap_hits_by_task[int(t)] + 1)
            p_va = cb.predict_proba(X2d_tr[va_idx])[:, 1]
            p_fold[:, t] = np.asarray(_clip_prob(p_va), dtype=np.float32)
        oof[va_idx, :] = p_fold
        fold_metrics = _metric_table(
            y_true=np.asarray(y_tr[va_idx], dtype=np.int64),
            p_pred=np.asarray(p_fold, dtype=np.float64),
            w_cls=np.asarray(w_tr[va_idx], dtype=np.float32),
        )
        for row in fold_metrics.to_dict(orient="records"):
            row["fold"] = int(fold_id)
            row["family"] = str(family)
            fold_metric_rows.append(row)
        macro_row = fold_metrics[fold_metrics["task"] == "macro"]
        if len(macro_row) == 1:
            by_task = fold_metrics[fold_metrics["task"] != "macro"].set_index("task")

            def _metric(task_name: str, metric_name: str) -> str:
                if task_name in by_task.index:
                    try:
                        return f"{float(by_task.loc[task_name, metric_name]):.6f}"
                    except Exception:
                        return "nan"
                return "nan"

            log_event(
                "INFO",
                "family.cv.catboost.fold",
                family=str(family),
                fold=int(fold_id),
                macro_pr_auc=f"{float(macro_row.iloc[0]['pr_auc']):.6f}",
                macro_roc_auc=f"{float(macro_row.iloc[0]['roc_auc']):.6f}",
                pr_auc_t0=_metric(str(TASK_COLS[0]), "pr_auc"),
                pr_auc_t1=_metric(str(TASK_COLS[1]), "pr_auc"),
                pr_auc_t2=_metric(str(TASK_COLS[2]), "pr_auc"),
                pr_auc_t3=_metric(str(TASK_COLS[3]), "pr_auc"),
                roc_auc_t0=_metric(str(TASK_COLS[0]), "roc_auc"),
                roc_auc_t1=_metric(str(TASK_COLS[1]), "roc_auc"),
                roc_auc_t2=_metric(str(TASK_COLS[2]), "roc_auc"),
                roc_auc_t3=_metric(str(TASK_COLS[3]), "roc_auc"),
            )
    if np.isnan(oof).any():
        missing = int(np.isnan(oof).sum())
        raise RuntimeError(f"OOF CatBoost predictions are incomplete; missing={missing}")
    df_oof = pd.DataFrame({"ID": [str(x) for x in ids_tr], "fold_id": fold_ids.astype(int)})
    for t in range(4):
        df_oof[f"p_t{t}"] = oof[:, t].astype(np.float32)
        df_oof[f"y_t{t}"] = np.asarray(y_tr[:, t], dtype=np.int64)
        df_oof[f"w_t{t}"] = np.asarray(w_tr[:, t], dtype=np.float32)
        df_oof[f"pred_t{t}"] = (df_oof[f"p_t{t}"] >= 0.5).astype(int)
    selected_iterations: Dict[int, int] = {}
    for t in range(4):
        plus_one = [int(x) + 1 for x in best_iters_by_task.get(int(t), [])]
        raw_selected_mean = (
            float(np.mean(np.asarray(plus_one, dtype=np.float64)))
            if plus_one
            else 4000.0
        )
        selected_iterations[int(t)] = int(max(1, int(round(float(raw_selected_mean)))))
        n_folds_t = int(folds_by_task.get(int(t), 0))
        cap_hits_t = int(cap_hits_by_task.get(int(t), 0))
        cap_hit_rate = float(cap_hits_t / float(max(1, n_folds_t)))
        log_event(
            "INFO",
            "family.cv.catboost.iter_selection",
            task_idx=int(t),
            fold_best_iterations=",".join([str(int(x)) for x in best_iters_by_task.get(int(t), [])]),
            selection_values=",".join([str(int(x)) for x in plus_one]),
            selected_iterations=int(selected_iterations[int(t)]),
            selection_stat="mean",
            selected_from_fold_best_plus_one=f"{float(raw_selected_mean):.6f}",
            cap_hits=int(cap_hits_t),
            n_folds=int(n_folds_t),
            cap_hit_rate=f"{cap_hit_rate:.3f}",
        )
        if cap_hit_rate >= 0.5:
            log_event(
                "WARN",
                "family.cv.catboost.iter_selection.cap_hit_high",
                task_idx=int(t),
                cap_hits=int(cap_hits_t),
                n_folds=int(n_folds_t),
                cap_hit_rate=f"{cap_hit_rate:.3f}",
                recommendation="Consider increasing CatBoost iterations cap above 4000 for this task.",
            )
    summary = {
        "family": str(family),
        "fold_best_iterations": {str(int(k)): [int(x) for x in v] for k, v in best_iters_by_task.items()},
        "selected_iterations": {str(int(k)): int(v) for k, v in selected_iterations.items()},
        "cap_hits_by_task": {str(int(k)): int(v) for k, v in cap_hits_by_task.items()},
        "n_folds_by_task": {str(int(k)): int(v) for k, v in folds_by_task.items()},
    }
    return df_oof, pd.DataFrame(fold_metric_rows), summary


def _prepare_family_inputs(
    cfg: FamilySuiteConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray, List[str], np.ndarray, Dict[str, _MILFamilyData]]:
    df_full = load_labels(cfg.labels, id_col=cfg.id_col)
    df_full[cfg.id_col] = df_full[cfg.id_col].astype(str)
    df_full[cfg.split_col] = df_full[cfg.split_col].astype(str)

    df_train = df_full[df_full[cfg.split_col].isin(set(cfg.use_splits))].copy().reset_index(drop=True)
    df_lb = df_full[df_full[cfg.split_col] == str(cfg.leaderboard_split)].copy().reset_index(drop=True)
    if len(df_train) == 0:
        raise ValueError(f"No rows in labels match use_splits={cfg.use_splits}")
    if len(df_lb) == 0:
        raise ValueError(f"No rows with split == '{cfg.leaderboard_split}'")

    ids_2d_file, X2d_file = load_2d(cfg.feat2d_scaled, id_col=cfg.id_col)
    if cfg.feat2d_raw:
        ids_2d_cat_file, X2d_cat_file = load_2d(cfg.feat2d_raw, id_col=cfg.id_col)
        cat_source = "raw"
    else:
        ids_2d_cat_file, X2d_cat_file = ids_2d_file, X2d_file
        cat_source = "scaled_fallback"
    ids_train_all = df_train[cfg.id_col].astype(str).tolist()
    ids_lb_all = df_lb[cfg.id_col].astype(str).tolist()
    _ = align_by_id(ids_2d_file, X2d_file, ids_train_all)
    _ = align_by_id(ids_2d_file, X2d_file, ids_lb_all)
    _ = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_train_all)
    _ = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_lb_all)
    log_event(
        "INFO",
        "family_suite.prepare_data.features_2d",
        mil_source="scaled",
        catboost_source=cat_source,
    )

    allowed = set(ids_train_all) | set(ids_lb_all)
    ids_conf, conf_ids, Xinst, inst_meta = load_and_merge_instances(
        cfg.feat3d_scaled,
        cfg.feat3d_qm_scaled,
        allowed_ids=allowed,
        id_col=cfg.id_col,
        conf_col=cfg.conf_col,
        return_meta=True,
    )
    _, starts, counts, id2pos, Xinst_sorted, conf_sorted = build_instance_index(ids_conf, conf_ids, Xinst)

    def _build_dense_instance_index_with_fallback(
        *,
        ids_order: Sequence[str],
        starts_base: np.ndarray,
        counts_base: np.ndarray,
        id2pos_base: Mapping[str, int],
        Xinst_base: np.ndarray,
        conf_base: np.ndarray,
        inst_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray, np.ndarray, int]:
        starts_new = np.zeros((len(ids_order),), dtype=np.int64)
        counts_new = np.zeros((len(ids_order),), dtype=np.int64)
        id2pos_new: Dict[str, int] = {}
        x_chunks: List[np.ndarray] = []
        c_chunks: List[np.ndarray] = []
        cursor = 0
        n_missing = 0
        for i, mol_id_raw in enumerate(ids_order):
            mol_id = str(mol_id_raw)
            starts_new[i] = int(cursor)
            id2pos_new[mol_id] = int(i)
            p = id2pos_base.get(mol_id)
            if p is None:
                bag_x = np.zeros((1, int(inst_dim)), dtype=np.float32)
                bag_c = np.asarray(["dummy"], dtype=object)
                n_missing += 1
            else:
                s = int(starts_base[int(p)])
                c = int(counts_base[int(p)])
                if c <= 0:
                    bag_x = np.zeros((1, int(inst_dim)), dtype=np.float32)
                    bag_c = np.asarray(["dummy"], dtype=object)
                    n_missing += 1
                else:
                    bag_x = np.asarray(Xinst_base[s : s + c], dtype=np.float32)
                    bag_c = np.asarray(conf_base[s : s + c], dtype=object)
            counts_new[i] = int(bag_x.shape[0])
            cursor += int(bag_x.shape[0])
            x_chunks.append(bag_x)
            c_chunks.append(bag_c)
        if x_chunks:
            Xinst_new = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
            conf_new = np.concatenate(c_chunks, axis=0)
        else:
            Xinst_new = np.zeros((0, int(inst_dim)), dtype=np.float32)
            conf_new = np.zeros((0,), dtype=object)
        return starts_new, counts_new, id2pos_new, Xinst_new, conf_new, int(n_missing)

    union_ids_all = list(dict.fromkeys([str(x) for x in (ids_train_all + ids_lb_all)]))
    starts_dense, counts_dense, id2pos_dense, Xinst_dense, conf_dense, n_missing_3d = _build_dense_instance_index_with_fallback(
        ids_order=union_ids_all,
        starts_base=np.asarray(starts, dtype=np.int64),
        counts_base=np.asarray(counts, dtype=np.int64),
        id2pos_base={str(k): int(v) for k, v in id2pos.items()},
        Xinst_base=np.asarray(Xinst_sorted, dtype=np.float32),
        conf_base=np.asarray(conf_sorted),
        inst_dim=int(inst_meta["inst_dim"]),
    )
    log_event(
        "INFO",
        "family_suite.prepare_data.instance_fallback",
        n_ids_total=int(len(union_ids_all)),
        n_ids_missing_3d=int(n_missing_3d),
        n_rows_dense=int(Xinst_dense.shape[0]),
        inst_dim=int(inst_meta["inst_dim"]),
    )

    def _make_mil_family(family: str) -> _MILFamilyData:
        if family == "mt_2d":
            ids_tr = ids_train_all
            ids_lb = ids_lb_all
            X2d_tr = align_by_id(ids_2d_file, X2d_file, ids_tr)
            X2d_lb = align_by_id(ids_2d_file, X2d_file, ids_lb)
            union_ids = list(dict.fromkeys([str(x) for x in (ids_tr + ids_lb)]))
            starts_loc = np.arange(len(union_ids), dtype=np.int64)
            counts_loc = np.ones(len(union_ids), dtype=np.int64)
            id2pos_loc = {str(mid): int(i) for i, mid in enumerate(union_ids)}
            Xinst_loc = np.zeros((len(union_ids), 1), dtype=np.float32)
            conf_loc = np.array(["dummy"] * len(union_ids), dtype=object)
            geom_dim = 0
            qm_dim = 0
        elif family == "mt_3d":
            ids_tr = ids_train_all
            ids_lb = ids_lb_all
            X2d_tr = np.zeros((len(ids_tr), 0), dtype=np.float32)
            X2d_lb = np.zeros((len(ids_lb), 0), dtype=np.float32)
            starts_loc, counts_loc, id2pos_loc, Xinst_loc, conf_loc = (
                starts_dense,
                counts_dense,
                id2pos_dense,
                Xinst_dense,
                conf_dense,
            )
            geom_dim = int(inst_meta["geom_dim"])
            qm_dim = int(inst_meta["qm_dim"])
        else:  # mt_2d3d
            ids_tr = ids_train_all
            ids_lb = ids_lb_all
            X2d_tr = align_by_id(ids_2d_file, X2d_file, ids_tr)
            X2d_lb = align_by_id(ids_2d_file, X2d_file, ids_lb)
            starts_loc, counts_loc, id2pos_loc, Xinst_loc, conf_loc = (
                starts_dense,
                counts_dense,
                id2pos_dense,
                Xinst_dense,
                conf_dense,
            )
            geom_dim = int(inst_meta["geom_dim"])
            qm_dim = int(inst_meta["qm_dim"])

        df_tr = _df_by_ids(df_train, id_col=cfg.id_col, ids=ids_tr)
        df_lb_f = _df_by_ids(df_lb, id_col=cfg.id_col, ids=ids_lb)
        folds = sorted(df_tr[cfg.fold_col].dropna().astype(int).unique().tolist())
        folds_info = fold_indices(df_tr, cfg.fold_col, folds)

        y_cls_tr = coerce_binary_labels(df_tr)
        y_cls_lb = coerce_binary_labels(df_lb_f)
        w_cls_tr = build_task_weights(df_tr)
        w_cls_lb = build_task_weights(df_lb_f)
        y_abs_tr, m_abs_tr, y_fluo_tr, m_fluo_tr = build_aux_targets_and_masks(df_tr)
        y_abs_lb, m_abs_lb, y_fluo_lb, m_fluo_lb = build_aux_targets_and_masks(df_lb_f)
        w_abs_tr, w_fluo_tr = build_aux_weights(df_tr)
        w_abs_lb, w_fluo_lb = build_aux_weights(df_lb_f)

        return _MILFamilyData(
            family=str(family),
            ids_train=[str(x) for x in ids_tr],
            ids_lb=[str(x) for x in ids_lb],
            X2d_train=np.asarray(X2d_tr, dtype=np.float32),
            X2d_lb=np.asarray(X2d_lb, dtype=np.float32),
            y_cls_train=np.asarray(y_cls_tr, dtype=np.int64),
            y_cls_lb=np.asarray(y_cls_lb, dtype=np.int64),
            w_cls_train=np.asarray(w_cls_tr, dtype=np.float32),
            w_cls_lb=np.asarray(w_cls_lb, dtype=np.float32),
            y_abs_train=np.asarray(y_abs_tr, dtype=np.float32),
            m_abs_train=np.asarray(m_abs_tr, dtype=bool),
            w_abs_train=np.asarray(w_abs_tr, dtype=np.float32),
            y_abs_lb=np.asarray(y_abs_lb, dtype=np.float32),
            m_abs_lb=np.asarray(m_abs_lb, dtype=bool),
            w_abs_lb=np.asarray(w_abs_lb, dtype=np.float32),
            y_fluo_train=np.asarray(y_fluo_tr, dtype=np.float32),
            m_fluo_train=np.asarray(m_fluo_tr, dtype=bool),
            w_fluo_train=np.asarray(w_fluo_tr, dtype=np.float32),
            y_fluo_lb=np.asarray(y_fluo_lb, dtype=np.float32),
            m_fluo_lb=np.asarray(m_fluo_lb, dtype=bool),
            w_fluo_lb=np.asarray(w_fluo_lb, dtype=np.float32),
            folds_info=folds_info,
            starts=np.asarray(starts_loc, dtype=np.int64),
            counts=np.asarray(counts_loc, dtype=np.int64),
            id2pos={str(k): int(v) for k, v in id2pos_loc.items()},
            Xinst_sorted=np.asarray(Xinst_loc, dtype=np.float32),
            conf_sorted=np.asarray(conf_loc),
            inst_geom_dim=int(geom_dim),
            inst_qm_dim=int(qm_dim),
        )

    fam_data = {f: _make_mil_family(f) for f in ("mt_2d", "mt_2d3d", "mt_3d")}
    return df_train, df_lb, ids_2d_file, X2d_file, ids_2d_cat_file, X2d_cat_file, fam_data


def run_family_suite(args: Any) -> None:
    cfg = FamilySuiteConfig(
        labels=str(args.labels),
        feat2d_scaled=str(args.feat2d_scaled),
        feat2d_raw=(None if getattr(args, "feat2d_raw", None) in (None, "") else str(args.feat2d_raw)),
        feat3d_scaled=str(args.feat3d_scaled),
        feat3d_qm_scaled=str(args.feat3d_qm_scaled),
        study_dir=str(args.study_dir),
        id_col=str(args.id_col),
        conf_col=str(args.conf_col),
        split_col=str(args.split_col),
        fold_col=str(args.fold_col),
        use_splits=tuple(str(x) for x in args.use_splits),
        leaderboard_split=str(args.leaderboard_split),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        trials=int(args.trials),
        seed=int(args.seed),
        nn_accelerator=str(args.nn_accelerator),
        nn_devices=int(args.nn_devices),
        precision=str(args.precision),
        num_workers=int(args.num_workers),
        cpu_workers=int(args.cpu_workers),
        pin_memory=bool(args.pin_memory),
        run_hpo=bool(args.run_hpo),
        hpo_only=bool(args.hpo_only),
        pruner_warmup_steps=int(args.pruner_warmup_steps),
        model_families=tuple(str(x) for x in (args.model_families or FAMILY_CHOICES)),
        calibration_method=str(args.calibration_method),
        skip_family_explainability=bool(getattr(args, "skip_family_explainability", False)),
        skip_family_calibration=bool(getattr(args, "skip_family_calibration", False)),
        skip_family_blending=bool(getattr(args, "skip_family_blending", False)),
        best_params_dir=(None if args.best_params_dir is None else str(args.best_params_dir)),
        family_best_params_json=(
            None if getattr(args, "family_best_params_json", None) in (None, "") else str(args.family_best_params_json)
        ),
        catboost_task_params_jsons=tuple(str(x) for x in (getattr(args, "catboost_task_params_jsons", None) or [])),
        catboost_final_iterations_jsons=tuple(
            str(x) for x in (getattr(args, "catboost_final_iterations_jsons", None) or [])
        ),
        mt_2d_params_json=(
            None if getattr(args, "mt_2d_params_json", None) in (None, "") else str(args.mt_2d_params_json)
        ),
        mt_2d3d_params_json=(
            None if getattr(args, "mt_2d3d_params_json", None) in (None, "") else str(args.mt_2d3d_params_json)
        ),
        mt_3d_params_json=(
            None if getattr(args, "mt_3d_params_json", None) in (None, "") else str(args.mt_3d_params_json)
        ),
        mt_2d_final_epochs_json=(
            None
            if getattr(args, "mt_2d_final_epochs_json", None) in (None, "")
            else str(args.mt_2d_final_epochs_json)
        ),
        mt_2d3d_final_epochs_json=(
            None
            if getattr(args, "mt_2d3d_final_epochs_json", None) in (None, "")
            else str(args.mt_2d3d_final_epochs_json)
        ),
        mt_3d_final_epochs_json=(
            None
            if getattr(args, "mt_3d_final_epochs_json", None) in (None, "")
            else str(args.mt_3d_final_epochs_json)
        ),
        catboost_hpo_parallel_tasks=int(getattr(args, "catboost_hpo_parallel_tasks", 1)),
        blend_seed_ensemble_size=int(getattr(args, "blend_seed_ensemble_size", 1)),
        blend_seed_step=int(getattr(args, "blend_seed_step", 1000)),
    )
    invalid = [x for x in cfg.model_families if x not in FAMILY_CHOICES]
    if invalid:
        raise ValueError(f"Unknown model family names: {invalid}. Allowed: {list(FAMILY_CHOICES)}")

    set_all_seeds(int(cfg.seed))
    maybe_set_torch_fast_flags()
    outdir = Path(cfg.study_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_params_dir = Path(cfg.best_params_dir) if cfg.best_params_dir else (outdir / "best_params")
    best_params_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = outdir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    override_params: Dict[str, Any] = {}
    if cfg.family_best_params_json:
        override_path = Path(cfg.family_best_params_json)
        if not override_path.exists():
            raise FileNotFoundError(f"family_best_params_json not found: {override_path}")
        override_params = _load_family_overrides_json(override_path)
        log_event(
            "INFO",
            "family_suite.overrides.loaded",
            path=str(override_path),
            families=",".join(sorted([str(k) for k in override_params.keys()])),
        )
    if len(cfg.catboost_task_params_jsons) > 0:
        catboost_override = _load_catboost_task_overrides(cfg.catboost_task_params_jsons)
        override_params["catboost_st"] = catboost_override
        log_event(
            "INFO",
            "family_suite.catboost_task_overrides.loaded",
            n_files=int(len(cfg.catboost_task_params_jsons)),
            tasks=",".join(sorted(catboost_override.keys())),
        )
    catboost_selected_iterations_overrides: Dict[int, int] = {}
    if len(cfg.catboost_final_iterations_jsons) > 0:
        catboost_selected_iterations_overrides = _load_catboost_task_iterations_overrides(
            cfg.catboost_final_iterations_jsons
        )
        log_event(
            "INFO",
            "family_suite.catboost_final_iterations_overrides.loaded",
            n_files=int(len(cfg.catboost_final_iterations_jsons)),
            selected_iterations=",".join(
                [f"t{int(t)}:{int(catboost_selected_iterations_overrides[int(t)])}" for t in range(4)]
            ),
        )
    per_family_override_paths: Dict[str, str] = {
        "mt_2d": str(cfg.mt_2d_params_json) if cfg.mt_2d_params_json else "",
        "mt_2d3d": str(cfg.mt_2d3d_params_json) if cfg.mt_2d3d_params_json else "",
        "mt_3d": str(cfg.mt_3d_params_json) if cfg.mt_3d_params_json else "",
    }
    for fam, raw_path in per_family_override_paths.items():
        if not raw_path:
            continue
        p = Path(raw_path)
        if not p.exists():
            raise FileNotFoundError(f"{fam} params JSON not found: {p}")
        if fam in override_params:
            log_event(
                "INFO",
                "family_suite.family_override.replaced_existing",
                family=str(fam),
                previous_source="family_best_params_json",
                new_source="family_specific_flag",
            )
        override_params[fam] = _load_single_family_override_json(path=p, family=fam)
        log_event(
            "INFO",
            "family_suite.family_override.loaded",
            family=str(fam),
            path=str(p),
        )
    epoch_override_paths: Dict[str, str] = {
        "mt_2d": str(cfg.mt_2d_final_epochs_json) if cfg.mt_2d_final_epochs_json else "",
        "mt_2d3d": str(cfg.mt_2d3d_final_epochs_json) if cfg.mt_2d3d_final_epochs_json else "",
        "mt_3d": str(cfg.mt_3d_final_epochs_json) if cfg.mt_3d_final_epochs_json else "",
    }
    mil_selected_epochs_overrides: Dict[str, int] = {}
    for fam, raw_path in epoch_override_paths.items():
        if not raw_path:
            continue
        p = Path(raw_path)
        if not p.exists():
            raise FileNotFoundError(f"{fam} final-epochs JSON not found: {p}")
        sel_epochs = _load_family_selected_epochs_json(path=p, family=fam)
        mil_selected_epochs_overrides[str(fam)] = int(sel_epochs)
        log_event(
            "INFO",
            "family_suite.family_epochs_override.loaded",
            family=str(fam),
            path=str(p),
            selected_epochs=int(sel_epochs),
            source="family_specific_flag",
        )
    for fam in ("mt_2d", "mt_2d3d", "mt_3d"):
        if fam in mil_selected_epochs_overrides:
            continue
        auto_path = best_params_dir / f"final_epochs_{fam}.json"
        if not auto_path.exists():
            continue
        try:
            sel_epochs = _load_family_selected_epochs_json(path=auto_path, family=fam)
            mil_selected_epochs_overrides[str(fam)] = int(sel_epochs)
            log_event(
                "INFO",
                "family_suite.family_epochs_override.loaded",
                family=str(fam),
                path=str(auto_path),
                selected_epochs=int(sel_epochs),
                source="best_params_dir_auto",
            )
        except Exception as exc:
            log_event(
                "WARN",
                "family_suite.family_epochs_override.auto_load_failed",
                family=str(fam),
                path=str(auto_path),
                error=repr(exc),
            )

    with log_step("family_suite.prepare_data", families=list(cfg.model_families)):
        df_train, df_lb, ids_2d_file, X2d_file, ids_2d_cat_file, X2d_cat_file, mil_families = _prepare_family_inputs(
            cfg
        )

    best_params: Dict[str, Any] = {}
    if cfg.run_hpo:
        for family in cfg.model_families:
            if family in override_params:
                best_params[family] = _normalize_family_params(family=family, value=override_params[family])
                log_event("INFO", "family.hpo.skipped_with_override", family=str(family))
                if family == "catboost_st":
                    for t in range(4):
                        saved_path = _save_catboost_task_best_params(
                            outdir=best_params_dir,
                            task_idx=int(t),
                            best_params=best_params[family][f"task_{int(t)}"],
                            best_value=0.0,
                        )
                        log_event(
                            "INFO",
                            "family.hpo.catboost.best_params.saved",
                            task_idx=int(t),
                            path=str(saved_path),
                            best_value="override",
                        )
                summary_path = _save_family_best_params(
                    outdir=best_params_dir,
                    family=family,
                    best_params=best_params[family],
                    best_value=0.0,
                )
                log_event("INFO", "family.hpo.best_params.saved", family=str(family), path=str(summary_path))
                continue
            if family == "catboost_st":
                try:
                    from catboost import CatBoostClassifier  # noqa: F401
                except Exception as exc:
                    raise RuntimeError("catboost is required for family 'catboost_st'.") from exc
                folds = sorted(df_train[cfg.fold_col].dropna().astype(int).unique().tolist())
                folds_info = fold_indices(df_train, cfg.fold_col, folds)
                ids_tr = df_train[cfg.id_col].astype(str).tolist()
                X2d_tr = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_tr)
                y_tr = coerce_binary_labels(df_train)
                w_tr = build_task_weights(df_train)
                catboost_params_by_task: Dict[str, Dict[str, Any]] = {}
                best_values_by_task: Dict[str, float] = {}
                task_parallel = int(max(1, min(4, int(cfg.catboost_hpo_parallel_tasks))))
                threads_per_task = int(max(1, int(cfg.cpu_workers) // task_parallel))
                log_event(
                    "INFO",
                    "family.hpo.catboost.parallel_config",
                    task_parallel=int(task_parallel),
                    cpu_workers=int(cfg.cpu_workers),
                    threads_per_task=int(threads_per_task),
                )

                def _run_catboost_task_hpo(task_idx: int) -> tuple[str, Dict[str, Any], float]:
                    study_name = f"catboost_st_t{int(task_idx)}"

                    def objective(trial: optuna.Trial, task_idx_inner: int = int(task_idx)) -> float:
                        from catboost import CatBoostClassifier

                        p = _catboost_search_space(trial, task_idx=int(task_idx_inner))
                        fold_pr_scores: List[float] = []
                        fold_roc_scores: List[float] = []
                        for fold_step, (tr, va, fold_id) in enumerate(folds_info):
                            pos = float(y_tr[tr, task_idx_inner].sum())
                            neg = float(len(tr) - pos)
                            spw = min(neg / max(pos, 1.0), float(p["pos_weight_clip"]))
                            cb = CatBoostClassifier(
                                **_catboost_common_params(
                                    params=p,
                                    seed=int(cfg.seed) + 97 * int(task_idx_inner) + 7919 * int(trial.number),
                                    threads=int(threads_per_task),
                                    scale_pos_weight=float(spw),
                                )
                            )
                            sw = (w_tr[tr, task_idx_inner] if task_idx_inner in (0, 1) else None)
                            cb.fit(
                                X2d_tr[tr],
                                y_tr[tr, task_idx_inner],
                                sample_weight=sw,
                                eval_set=(X2d_tr[va], y_tr[va, task_idx_inner]),
                                use_best_model=True,
                                early_stopping_rounds=200,
                                verbose=False,
                            )
                            p_va = cb.predict_proba(X2d_tr[va])[:, 1]
                            p_va = np.clip(np.nan_to_num(p_va, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
                            sw_va = (w_tr[va, task_idx_inner] if task_idx_inner in (0, 1) else None)
                            try:
                                ap_t = float(average_precision_score(y_tr[va, task_idx_inner], p_va, sample_weight=sw_va))
                            except ValueError:
                                ap_t = 0.0
                            try:
                                roc_t = float(roc_auc_score(y_tr[va, task_idx_inner], p_va, sample_weight=sw_va))
                            except ValueError:
                                roc_t = 0.5
                            fold_pr_scores.append(ap_t)
                            fold_roc_scores.append(roc_t)
                            trial.report(float(np.mean(fold_pr_scores)), step=int(fold_step))
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                            log_event(
                                "INFO",
                                "family.hpo.catboost.trial.fold",
                                task_idx=int(task_idx_inner),
                                trial=int(trial.number),
                                fold_id=int(fold_id),
                                pr_auc=f"{ap_t:.6f}",
                                roc_auc=f"{roc_t:.6f}",
                            )
                        mean_pr = float(np.mean(fold_pr_scores)) if fold_pr_scores else 0.0
                        mean_roc = float(np.mean(fold_roc_scores)) if fold_roc_scores else 0.5
                        trial.set_user_attr("cv_pr_auc", mean_pr)
                        trial.set_user_attr("cv_roc_auc", mean_roc)
                        log_event(
                            "INFO",
                            "family.hpo.catboost.trial.done",
                            task_idx=int(task_idx_inner),
                            trial=int(trial.number),
                            cv_pr_auc=f"{mean_pr:.6f}",
                            cv_roc_auc=f"{mean_roc:.6f}",
                        )
                        return mean_pr

                    storage = f"sqlite:///{(outdir / f'{study_name}.sqlite3').as_posix()}"
                    study = optuna.create_study(
                        study_name=study_name,
                        direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=int(cfg.seed) + 113 * int(task_idx)),
                        pruner=optuna.pruners.PercentilePruner(
                            percentile=25.0,
                            n_startup_trials=10,
                            n_warmup_steps=int(cfg.pruner_warmup_steps),
                        ),
                        storage=storage,
                        load_if_exists=True,
                    )
                    with log_step("family.hpo.catboost.optimize", task_idx=int(task_idx), n_trials=int(cfg.trials)):
                        study.optimize(objective, n_trials=int(cfg.trials), gc_after_trial=True)
                    pd.DataFrame(
                        study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
                    ).to_csv(outdir / f"{study_name}_trials.csv", index=False)

                    bp_t = dict(study.best_params)
                    saved_path = _save_catboost_task_best_params(
                        outdir=best_params_dir,
                        task_idx=int(task_idx),
                        best_params=bp_t,
                        best_value=float(study.best_value),
                    )
                    log_event(
                        "INFO",
                        "family.hpo.catboost.best_params.saved",
                        task_idx=int(task_idx),
                        path=str(saved_path),
                        best_value=f"{float(study.best_value):.6f}",
                    )
                    return f"task_{int(task_idx)}", bp_t, float(study.best_value)

                if int(task_parallel) <= 1:
                    for t in range(4):
                        key, bp_t, best_v = _run_catboost_task_hpo(int(t))
                        catboost_params_by_task[key] = bp_t
                        best_values_by_task[key] = float(best_v)
                else:
                    with ThreadPoolExecutor(max_workers=int(task_parallel)) as pool:
                        futures = [pool.submit(_run_catboost_task_hpo, int(t)) for t in range(4)]
                        for fut in as_completed(futures):
                            key, bp_t, best_v = fut.result()
                            catboost_params_by_task[key] = bp_t
                            best_values_by_task[key] = float(best_v)
                summary_path = _save_family_best_params(
                    outdir=best_params_dir,
                    family=family,
                    best_params=catboost_params_by_task,
                    best_value=float(np.mean([v for v in best_values_by_task.values()] or [0.0])),
                )
                log_event(
                    "INFO",
                    "family.hpo.best_params.saved",
                    family=str(family),
                    path=str(summary_path),
                )
                best_params[family] = catboost_params_by_task
                continue

            data = mil_families[family]
            cv_data = MILCVData(
                X2d_scaled=data.X2d_train,
                y_cls=data.y_cls_train,
                w_cls=data.w_cls_train,
                y_abs=data.y_abs_train,
                m_abs=data.m_abs_train,
                w_abs=data.w_abs_train,
                y_fluo=data.y_fluo_train,
                m_fluo=data.m_fluo_train,
                w_fluo=data.w_fluo_train,
                ids=data.ids_train,
                folds_info=data.folds_info,
                starts=data.starts,
                counts=data.counts,
                id2pos=data.id2pos,
                Xinst_sorted=data.Xinst_sorted,
                inst_geom_dim=int(data.inst_geom_dim),
                inst_qm_dim=int(data.inst_qm_dim),
            )
            run_cfg = CVRunConfig(
                seed=int(cfg.seed),
                trainer=TrainerSystemConfig(
                    max_epochs=int(cfg.max_epochs),
                    patience=int(cfg.patience),
                    accelerator=str(cfg.nn_accelerator),
                    devices=int(cfg.nn_devices),
                    precision=str(cfg.precision),
                ),
                loader=LoaderConfig(
                    num_workers=int(cfg.num_workers),
                    pin_memory=bool(cfg.pin_memory and torch.cuda.is_available()),
                ),
                ckpt_root=(outdir / f"_tmp_best_ckpts_{family}"),
                run_tag=str(family),
            )
            run_cfg.ckpt_root.mkdir(parents=True, exist_ok=True)
            cv = _MILMacroCrossValidator(data=cv_data, run_config=run_cfg, family=str(family))
            study_name = f"{family}_mil"
            storage = f"sqlite:///{(outdir / f'{study_name}.sqlite3').as_posix()}"
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=int(cfg.seed)),
                pruner=optuna.pruners.PercentilePruner(
                    percentile=25.0,
                    n_startup_trials=10,
                    n_warmup_steps=int(cfg.pruner_warmup_steps),
                ),
                storage=storage,
                load_if_exists=True,
            )
            with log_step("family.hpo.mil.optimize", family=family, n_trials=int(cfg.trials)):
                study.optimize(cv.evaluate_trial, n_trials=int(cfg.trials), gc_after_trial=True)
            pd.DataFrame(
                study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
            ).to_csv(outdir / f"{study_name}_trials.csv", index=False)
            bp = dict(study.best_params)
            bp["min_w"] = 0.4
            if str(family) == "mt_2d":
                bp.update(_mt_2d_fixed_inactive_params())
            best_params[family] = bp
            _save_family_best_params(
                outdir=best_params_dir,
                family=family,
                best_params=bp,
                best_value=float(study.best_value),
            )
            log_event("INFO", "family.hpo.best_params.saved", family=str(family), path=str(best_params_dir / f"{family}.json"))
    else:
        for family in cfg.model_families:
            if family in override_params:
                best_params[family] = _normalize_family_params(family=family, value=override_params[family])
                log_event("INFO", "family.best_params.override_used", family=str(family))
                continue
            if family == "catboost_st":
                try:
                    best_params[family] = {
                        f"task_{int(t)}": _load_catboost_task_best_params(outdir=best_params_dir, task_idx=int(t))
                        for t in range(4)
                    }
                except FileNotFoundError:
                    # Backward compatibility with older shared CatBoost params format.
                    shared = _load_family_best_params(outdir=best_params_dir, family=family)
                    best_params[family] = _normalize_catboost_params(shared)
                    log_event(
                        "WARN",
                        "family.best_params.catboost.shared_fallback",
                        path=str(best_params_dir / f"{family}.json"),
                    )
            else:
                best_params[family] = _load_family_best_params(outdir=best_params_dir, family=family)

    if cfg.hpo_only:
        log_event("INFO", "family_suite.hpo_only.completed", n_families=int(len(cfg.model_families)))
        return

    seed_ensemble_size = int(max(1, int(cfg.blend_seed_ensemble_size)))
    seed_step = int(max(1, int(cfg.blend_seed_step)))
    model_specs: List[Tuple[str, str, int, int]] = []
    for family in cfg.model_families:
        for rep_idx in range(seed_ensemble_size):
            rep_seed = int(cfg.seed) + int(rep_idx) * int(seed_step)
            model_key = str(family) if int(seed_ensemble_size) == 1 else f"{str(family)}__seed{int(rep_seed)}"
            model_specs.append((str(model_key), str(family), int(rep_seed), int(rep_idx)))
    log_event(
        "INFO",
        "family_suite.seed_ensemble",
        ensemble_size=int(seed_ensemble_size),
        seed_step=int(seed_step),
        n_models_total=int(len(model_specs)),
        models=",".join([str(k) for (k, _, _, _) in model_specs]),
    )

    # Strict no-leak calibration/blending fit scope: train OOF only.
    cv_oof_tables: Dict[str, pd.DataFrame] = {}
    cv_fold_metrics_by_model: Dict[str, pd.DataFrame] = {}
    mil_selected_epochs: Dict[str, int] = {}
    catboost_selected_iterations: Dict[str, Dict[int, int]] = {}
    for model_key, family, rep_seed, rep_idx in model_specs:
        cfg_rep = replace(cfg, seed=int(rep_seed))
        with log_step(
            "family.cv.run",
            family=str(family),
            model=str(model_key),
            replica_idx=int(rep_idx),
            seed=int(rep_seed),
        ):
            if family == "catboost_st":
                df_oof, df_fold_metrics, sel_summary = _run_catboost_oof_predictions(
                    cfg=cfg_rep,
                    df_train=df_train,
                    ids_2d_cat_file=ids_2d_cat_file,
                    X2d_cat_file=X2d_cat_file,
                    task_best_params=dict(best_params[family]),
                    family=str(model_key),
                )
                sel_map_raw = sel_summary.get("selected_iterations", {})
                sel_map: Dict[int, int] = {
                    int(t): int(sel_map_raw.get(str(int(t)), 4000))
                    for t in range(4)
                }
                if len(catboost_selected_iterations_overrides) > 0:
                    for t in range(4):
                        cv_sel = int(sel_map[int(t)])
                        override_sel = int(catboost_selected_iterations_overrides[int(t)])
                        sel_map[int(t)] = int(override_sel)
                        log_event(
                            "INFO",
                            "family.cv.catboost.iter_selection.override_applied",
                            model=str(model_key),
                            task_idx=int(t),
                            cv_selected_iterations=int(cv_sel),
                            override_selected_iterations=int(override_sel),
                            source="catboost_final_iterations_jsons",
                        )
                catboost_selected_iterations[str(model_key)] = dict(sel_map)
                if int(rep_idx) == 0:
                    for t in range(4):
                        sel_iter = int(sel_map[int(t)])
                        save_path = _save_catboost_final_iterations(
                            outdir=best_params_dir,
                            task_idx=int(t),
                            fold_best_iterations=sel_summary.get("fold_best_iterations", {}).get(str(int(t)), []),
                            selected_iterations=int(sel_iter),
                        )
                        log_event(
                            "INFO",
                            "family.cv.catboost.iterations.saved",
                            model=str(model_key),
                            task_idx=int(t),
                            selected_iterations=int(sel_iter),
                            path=str(save_path),
                        )
            else:
                df_oof, df_fold_metrics, sel_summary = _run_mil_oof_predictions(
                    cfg=cfg_rep,
                    family=str(model_key),
                    family_data=mil_families[family],
                    best_params=best_params[family],
                    outdir=outdir,
                    fixed_fold_train_epochs=mil_selected_epochs_overrides.get(str(family)),
                )
                sel_epochs = int(sel_summary.get("selected_epochs", max(1, int(cfg.max_epochs))))
                mil_selected_epochs[str(model_key)] = int(sel_epochs)
                if int(rep_idx) == 0:
                    save_path = _save_family_final_epochs(
                        outdir=best_params_dir,
                        family=str(family),
                        fold_best_epochs=sel_summary.get("fold_best_epochs", []),
                        selected_epochs=int(sel_epochs),
                    )
                    log_event(
                        "INFO",
                        "family.cv.mil.final_epochs.saved",
                        family=str(family),
                        model=str(model_key),
                        selected_epochs=int(sel_epochs),
                        selection_source=str(sel_summary.get("selection_source", "cv_mean_best_epoch")),
                        path=str(save_path),
                    )
        cv_oof_tables[str(model_key)] = df_oof
        _log_oof_bitmask_coverage(family=str(model_key), df_oof=df_oof)
        df_oof.to_csv(outdir / f"train_oof_preds_{model_key}.csv", index=False)
        cv_fold_metrics_by_model[str(model_key)] = (
            df_fold_metrics.copy() if isinstance(df_fold_metrics, pd.DataFrame) else pd.DataFrame()
        )
        if len(df_fold_metrics) > 0:
            df_fold_metrics.to_csv(outdir / f"cv_fold_metrics_{model_key}.csv", index=False)
        y_oof = np.stack([df_oof[f"y_t{t}"].to_numpy(dtype=np.int64) for t in range(4)], axis=1)
        p_oof = np.stack([df_oof[f"p_t{t}"].to_numpy(dtype=np.float64) for t in range(4)], axis=1)
        w_oof = np.stack(
            [
                df_oof[f"w_t{t}"].to_numpy(dtype=np.float32)
                if f"w_t{t}" in df_oof.columns
                else np.ones(len(df_oof), dtype=np.float32)
                for t in range(4)
            ],
            axis=1,
        )
        _metric_table(y_true=y_oof, p_pred=p_oof, w_cls=w_oof).to_csv(
            outdir / f"cv_oof_metrics_{model_key}.csv", index=False
        )

    # Enforce identical OOF ID/fold mapping across families.
    fam_keys = list(cv_oof_tables.keys())
    if len(fam_keys) >= 2:
        ref_family = str(fam_keys[0])
        ref_map = (
            cv_oof_tables[ref_family][["ID", "fold_id"]]
            .copy()
            .assign(ID=lambda d: d["ID"].astype(str), fold_id=lambda d: d["fold_id"].astype(int))
            .set_index("ID")["fold_id"]
        )
        for fam in fam_keys[1:]:
            cur_map = (
                cv_oof_tables[str(fam)][["ID", "fold_id"]]
                .copy()
                .assign(ID=lambda d: d["ID"].astype(str), fold_id=lambda d: d["fold_id"].astype(int))
                .set_index("ID")["fold_id"]
            )
            if int(ref_map.shape[0]) != int(cur_map.shape[0]) or set(ref_map.index) != set(cur_map.index):
                raise RuntimeError(
                    f"OOF ID set mismatch across families: ref={ref_family} vs {fam} "
                    f"(n_ref={int(ref_map.shape[0])}, n_cur={int(cur_map.shape[0])})"
                )
            aligned = cur_map.loc[ref_map.index]
            mismatches = int(np.sum((aligned.to_numpy(dtype=np.int64) != ref_map.to_numpy(dtype=np.int64))))
            log_event(
                "INFO",
                "family.cv.oof.alignment",
                ref_family=str(ref_family),
                family=str(fam),
                n_ids=int(ref_map.shape[0]),
                fold_mismatches=int(mismatches),
            )
            if mismatches > 0:
                raise RuntimeError(
                    f"OOF fold mismatch across families: ref={ref_family} vs {fam}, mismatches={int(mismatches)}"
                )

    calibration_fit_method = "identity" if cfg.skip_family_calibration else str(cfg.calibration_method)
    calibrated_oof: Dict[str, pd.DataFrame] = {}
    calib_params_by_family: Dict[str, Dict[str, Any]] = {}
    for family, df_oof in cv_oof_tables.items():
        with log_step(
            "family.calibration.fit",
            family=str(family),
            scope="train_oof",
            method=str(calibration_fit_method),
            skipped=bool(cfg.skip_family_calibration),
            n_rows=int(len(df_oof)),
        ):
            arr = df_oof.copy()
            y = np.stack([arr[f"y_t{t}"].to_numpy(dtype=np.int64) for t in range(4)], axis=1)
            p = np.stack([arr[f"p_t{t}"].to_numpy(dtype=np.float64) for t in range(4)], axis=1)
            p_cal = np.zeros_like(p, dtype=np.float32)
            params_by_task: Dict[str, Any] = {}
            for t in range(4):
                raw_sw_t = arr[f"w_t{t}"].to_numpy(dtype=np.float64) if f"w_t{t}" in arr.columns else None
                sw_t = _metric_sample_weight_for_task(raw_sw_t, t)
                pos_rate = float(np.mean(y[:, t])) if int(y.shape[0]) > 0 else 0.0
                sw_eff = (
                    np.asarray(sw_t, dtype=np.float64)
                    if sw_t is not None
                    else np.ones((int(y.shape[0]),), dtype=np.float64)
                )
                sw_sum = float(np.sum(sw_eff))
                sw_pos = float(np.sum(sw_eff * np.asarray(y[:, t], dtype=np.float64)))
                pos_rate_weighted = float(sw_pos / sw_sum) if sw_sum > 0.0 else float("nan")
                if cfg.skip_family_calibration:
                    p_cal[:, t] = np.asarray(p[:, t], dtype=np.float32)
                    t_params = {"kind": "identity", "skip_reason": "skip_family_calibration"}
                else:
                    p_cal[:, t], t_params = _calibrate_task(
                        y[:, t],
                        p[:, t],
                        cfg.calibration_method,
                        sample_weight=sw_t,
                    )
                params_by_task[str(t)] = t_params
                arr[f"p_cal_t{t}"] = p_cal[:, t]
                log_event(
                    "INFO",
                    "family.calibration.fit.task",
                    family=str(family),
                    scope="train_oof",
                    task_idx=int(t),
                    kind=str(t_params.get("kind", "unknown")),
                    skipped=bool(cfg.skip_family_calibration),
                    pos_rate=f"{pos_rate:.6f}",
                    pos_rate_weighted=f"{pos_rate_weighted:.6f}",
                    weighting_policy="metrics_aligned" if sw_t is not None else "unweighted",
                    sample_weight_sum=f"{sw_sum:.3f}",
                    n_rows=int(y.shape[0]),
                )
            cal_pred_path = outdir / f"train_oof_preds_{family}_calibrated.csv"
            arr.to_csv(cal_pred_path, index=False)
            calibrated_oof[family] = arr
            calib_params_by_family[family] = params_by_task
            cal_json_path = cal_dir / f"{family}_calib.json"
            cal_json_path.write_text(
                json.dumps(
                    {
                        "family": str(family),
                        "fit_scope": "train_oof",
                        "method": str(calibration_fit_method),
                        "skipped": bool(cfg.skip_family_calibration),
                        "task_params": params_by_task,
                    },
                    indent=2,
                )
            )
            w = np.stack(
                [
                    arr[f"w_t{t}"].to_numpy(dtype=np.float32)
                    if f"w_t{t}" in arr.columns
                    else np.ones(len(arr), dtype=np.float32)
                    for t in range(4)
                ],
                axis=1,
            )
            cal_metric_path = outdir / f"cv_oof_metrics_{family}_calibrated.csv"
            cal_metrics = _metric_table(y_true=y, p_pred=p_cal, w_cls=w)
            cal_metrics.to_csv(cal_metric_path, index=False)
            macro_row = cal_metrics[cal_metrics["task"] == "macro"]
            if len(macro_row) == 1:
                log_event(
                    "INFO",
                    "family.calibration.fit.summary",
                    family=str(family),
                    scope="train_oof",
                    skipped=bool(cfg.skip_family_calibration),
                    macro_pr_auc=f"{float(macro_row.iloc[0]['pr_auc']):.6f}",
                    macro_roc_auc=f"{float(macro_row.iloc[0]['roc_auc']):.6f}",
                    preds_path=str(cal_pred_path),
                    metrics_path=str(cal_metric_path),
                    calib_json_path=str(cal_json_path),
                )

    blend_weights: Dict[str, Any] | None = None
    blend_oof_metrics: pd.DataFrame | None = None
    skip_blend_reason = ""
    if cfg.skip_family_blending:
        skip_blend_reason = "skip_family_blending_flag"
    elif int(len(calibrated_oof)) <= 1:
        skip_blend_reason = "single_model"

    if skip_blend_reason:
        log_event(
            "INFO",
            "family.blend.fit.skipped",
            scope="train_oof",
            reason=str(skip_blend_reason),
            n_families=int(len(calibrated_oof)),
        )
    else:
        with log_step("family.blend.fit", scope="train_oof"):
            fams = list(calibrated_oof.keys())
            if len(fams) == 0:
                raise RuntimeError("No family predictions available for blending.")
            common_ids = set(calibrated_oof[fams[0]]["ID"].astype(str).tolist())
            for f in fams[1:]:
                common_ids &= set(calibrated_oof[f]["ID"].astype(str).tolist())
            common_ids = sorted(common_ids)
            if len(common_ids) == 0:
                raise RuntimeError("No common train OOF IDs across selected families for blending.")
            log_event(
                "INFO",
                "family.blend.fit.scope",
                scope="train_oof",
                n_families=int(len(fams)),
                families=",".join([str(x) for x in fams]),
                n_common_ids=int(len(common_ids)),
            )

            y_bl: Optional[np.ndarray] = None
            w_bl: Optional[np.ndarray] = None
            fold_bl: Optional[np.ndarray] = None
            x_by_family_oof: Dict[str, np.ndarray] = {}
            for f in fams:
                d = calibrated_oof[f].copy()
                d["ID"] = d["ID"].astype(str)
                d = d.set_index("ID").loc[common_ids].reset_index(drop=False)
                _require_calibrated_prob_columns(df=d, family=str(f), scope="train_oof")
                x_by_family_oof[f] = np.stack([d[f"p_cal_t{t}"].to_numpy(dtype=np.float64) for t in range(4)], axis=1)
                log_event(
                    "INFO",
                    "family.blend.fit.family_input",
                    scope="train_oof",
                    family=str(f),
                    input_source="calibrated_probabilities",
                    n_rows=int(len(d)),
                )
                if y_bl is None:
                    y_bl = np.stack([d[f"y_t{t}"].to_numpy(dtype=np.int64) for t in range(4)], axis=1)
                    w_bl = np.stack(
                        [
                            d[f"w_t{t}"].to_numpy(dtype=np.float32)
                            if f"w_t{t}" in d.columns
                            else np.ones(len(d), dtype=np.float32)
                            for t in range(4)
                        ],
                        axis=1,
                    )
                    if "fold_id" in d.columns:
                        fold_bl = d["fold_id"].to_numpy(dtype=np.int64)
            assert y_bl is not None
            assert w_bl is not None

            blend_pred_oof = np.zeros_like(y_bl, dtype=np.float64)
            blend_weights = {
                "fit_scope": "train_oof",
                "families": fams,
                "input_source": "calibrated_probabilities",
                "calibration_method": str(calibration_fit_method),
                "blend_method": "convex_blending_simplex",
                "weighting_policy": "metrics_aligned",
                "tasks": {},
            }
            for t in range(4):
                x_task = np.column_stack([x_by_family_oof[f][:, t] for f in fams]).astype(np.float64)
                y_task = y_bl[:, t].astype(int)
                blend_sw = _metric_sample_weight_for_task(w_bl[:, t], t)
                task_cfg, p_task = _fit_blend_task(
                    x=x_task,
                    y=y_task,
                    fams=fams,
                    sample_weight=blend_sw,
                )
                blend_pred_oof[:, t] = np.asarray(p_task, dtype=np.float64)
                blend_weights["tasks"][str(t)] = dict(task_cfg)
                w_vec = np.asarray(list(task_cfg.get("weights", {}).values()), dtype=np.float64)
                w_vec = w_vec if w_vec.size > 0 else np.asarray([1.0], dtype=np.float64)
                w_sum = float(np.sum(np.clip(w_vec, 0.0, np.inf)))
                w_norm = np.clip(w_vec, 0.0, np.inf) / float(max(w_sum, 1e-12))
                w_entropy = -float(np.sum(w_norm * np.log(np.clip(w_norm, 1e-12, 1.0))))
                log_event(
                    "INFO",
                    "family.blend.fit.task",
                    scope="train_oof",
                    task_idx=int(t),
                    kind=str(task_cfg.get("kind", "unknown")),
                    n_rows=int(y_task.shape[0]),
                    pos_rate=f"{float(np.mean(y_task)):.6f}",
                    constraint=str(task_cfg.get("constraint", "")),
                    optimizer=str(task_cfg.get("optimizer", "")),
                    objective=str(task_cfg.get("objective", "roc_auc")),
                    n_eval=int(task_cfg.get("n_eval", 0)),
                    train_roc_auc=f"{float(task_cfg.get('train_roc_auc', np.nan)):.6f}",
                    train_pr_auc_tiebreak=f"{float(task_cfg.get('train_pr_auc_tiebreak', np.nan)):.6f}",
                    weight_max=f"{float(np.max(w_norm)):.6f}",
                    weight_entropy=f"{float(w_entropy):.6f}",
                    weight_l1=f"{float(np.sum(np.abs(np.asarray(list(task_cfg.get('weights', {}).values()), dtype=np.float64)))):.6f}",
                    weighting_policy="metrics_aligned" if blend_sw is not None else "unweighted",
                    sample_weight_sum=f"{float(np.sum(blend_sw)):.3f}" if blend_sw is not None else f"{float(y_task.shape[0]):.3f}",
                )
            blend_weights_path = outdir / "blend_weights.json"
            blend_weights_path.write_text(json.dumps(blend_weights, indent=2))

            blend_oof_df = pd.DataFrame({"ID": common_ids})
            if fold_bl is not None:
                blend_oof_df["fold_id"] = fold_bl.astype(int)
            for t in range(4):
                blend_oof_df[f"p_blend_t{t}"] = blend_pred_oof[:, t]
                blend_oof_df[f"y_t{t}"] = y_bl[:, t]
                blend_oof_df[f"pred_t{t}"] = (blend_pred_oof[:, t] >= 0.5).astype(int)
            blend_oof_pred_path = outdir / "train_oof_preds_blend.csv"
            blend_oof_df.to_csv(blend_oof_pred_path, index=False)
            blend_oof_metric_path = outdir / "cv_oof_metrics_blend.csv"
            blend_oof_metrics = _metric_table(y_true=y_bl, p_pred=blend_pred_oof, w_cls=w_bl)
            blend_oof_metrics.to_csv(blend_oof_metric_path, index=False)
            if fold_bl is not None:
                fold_metric_rows_blend: List[Dict[str, Any]] = []
                for fold_id in sorted([int(x) for x in np.unique(fold_bl).tolist() if int(x) >= 0]):
                    m = np.asarray(fold_bl == int(fold_id))
                    if int(np.sum(m)) <= 0:
                        continue
                    mt = _metric_table(y_true=y_bl[m], p_pred=blend_pred_oof[m], w_cls=w_bl[m])
                    for row in mt.to_dict(orient="records"):
                        row["fold"] = int(fold_id)
                        row["family"] = "blend"
                        fold_metric_rows_blend.append(row)
                if len(fold_metric_rows_blend) > 0:
                    pd.DataFrame(fold_metric_rows_blend).to_csv(outdir / "cv_fold_metrics_blend.csv", index=False)
            macro_row = blend_oof_metrics[blend_oof_metrics["task"] == "macro"]
            if len(macro_row) == 1:
                log_event(
                    "INFO",
                    "family.blend.fit.summary",
                    scope="train_oof",
                    macro_pr_auc=f"{float(macro_row.iloc[0]['pr_auc']):.6f}",
                    macro_roc_auc=f"{float(macro_row.iloc[0]['roc_auc']):.6f}",
                    preds_path=str(blend_oof_pred_path),
                    metrics_path=str(blend_oof_metric_path),
                    weights_path=str(blend_weights_path),
                )

    pred_tables: Dict[str, pd.DataFrame] = {}
    leaderboard_metrics_by_family: Dict[str, pd.DataFrame] = {}
    for model_key, family, rep_seed, rep_idx in model_specs:
        cfg_rep = replace(cfg, seed=int(rep_seed))
        if family == "catboost_st":
            from catboost import CatBoostClassifier

            ids_tr = df_train[cfg.id_col].astype(str).tolist()
            ids_lb = df_lb[cfg.id_col].astype(str).tolist()
            X2d_tr = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_tr)
            X2d_lb = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_lb)
            y_tr = coerce_binary_labels(df_train)
            y_lb = coerce_binary_labels(df_lb)
            w_tr = build_task_weights(df_train)
            w_lb = build_task_weights(df_lb)
            catboost_task_params = dict(best_params[family])
            pred = np.zeros((len(ids_lb), 4), dtype=np.float64)
            selected_iter_map = catboost_selected_iterations.get(str(model_key), {})
            family_model_dir = outdir / str(family)
            family_model_dir.mkdir(parents=True, exist_ok=True)
            for t in range(4):
                if f"task_{int(t)}" not in catboost_task_params:
                    raise KeyError(
                        f"Missing CatBoost params for task_{t}. "
                        "Expected task-wise JSONs: catboost_st_t0..catboost_st_t3."
                    )
                p = dict(catboost_task_params[f"task_{int(t)}"])
                pos = float(y_tr[:, t].sum())
                neg = float(len(y_tr) - pos)
                spw = min(neg / max(pos, 1.0), float(p.get("pos_weight_clip", 100.0)))
                target_iterations = int(selected_iter_map.get(int(t), int(_resolve_catboost_iterations(p))))
                cb_kwargs = _catboost_common_params(
                    params=p,
                    seed=int(cfg_rep.seed) + 97 * int(t),
                    threads=max(1, int(cfg_rep.cpu_workers)),
                    scale_pos_weight=float(spw),
                )
                cb_kwargs["iterations"] = int(target_iterations)
                cb = CatBoostClassifier(
                    **cb_kwargs
                )
                sw = (w_tr[:, t] if t in (0, 1) else None)
                cb.fit(
                    X2d_tr,
                    y_tr[:, t],
                    sample_weight=sw,
                    use_best_model=False,
                    verbose=False,
                )
                cb_model_path = family_model_dir / f"leaderboard_{model_key}_task{int(t)}.cbm"
                cb.save_model(str(cb_model_path))
                pred[:, t] = cb.predict_proba(X2d_lb)[:, 1]
                log_event(
                    "INFO",
                    "family.final.catboost.task",
                    family=str(family),
                    model=str(model_key),
                    replica_idx=int(rep_idx),
                    seed=int(rep_seed),
                    task_idx=int(t),
                    selected_iterations=int(target_iterations),
                    scale_pos_weight=f"{float(spw):.6f}",
                    model_path=str(cb_model_path),
                )
            df_pred = pd.DataFrame({"ID": [str(x) for x in ids_lb]})
            for t in range(4):
                df_pred[f"p_t{t}"] = pred[:, t]
                df_pred[f"y_t{t}"] = y_lb[:, t]
                df_pred[f"w_t{t}"] = w_lb[:, t]
                df_pred[f"pred_t{t}"] = (pred[:, t] >= 0.5).astype(int)
            df_pred.to_csv(outdir / f"leaderboard_preds_{model_key}.csv", index=False)
            metrics_family = _metric_table(y_true=y_lb, p_pred=pred, w_cls=w_lb)
            metrics_family.to_csv(outdir / f"leaderboard_metrics_{model_key}.csv", index=False)
            leaderboard_metrics_by_family[str(model_key)] = metrics_family
            pred_tables[str(model_key)] = df_pred
            continue

        selected_epochs = int(
            mil_selected_epochs.get(str(model_key), mil_selected_epochs.get(str(family), int(cfg.max_epochs)))
        )
        log_event(
            "INFO",
            "family.final.mil.epoch_plan",
            family=str(family),
            model=str(model_key),
            replica_idx=int(rep_idx),
            seed=int(rep_seed),
            selected_epochs=int(selected_epochs),
        )
        df_pred, metrics_family, train_info = _run_mil_final_train_and_predict(
            cfg=cfg_rep,
            family_data=mil_families[family],
            best_params=best_params[family],
            outdir=outdir,
            write_outputs=True,
            write_explainability_outputs=bool(not cfg.skip_family_explainability),
            output_prefix=f"leaderboard_{model_key}",
            fixed_train_epochs=int(selected_epochs),
        )
        leaderboard_metrics_by_family[str(model_key)] = metrics_family
        metrics_family.to_csv(outdir / f"leaderboard_metrics_{model_key}.csv", index=False)
        df_pred.to_csv(outdir / f"leaderboard_preds_{model_key}.csv", index=False)
        log_event(
            "INFO",
            "family.final.mil.train_done",
            family=str(family),
            model=str(model_key),
            replica_idx=int(rep_idx),
            seed=int(rep_seed),
            target_epochs=int(train_info.get("target_epochs", selected_epochs)),
            epochs_trained=int(train_info.get("epochs_trained", 0)),
            best_epoch=int(train_info.get("best_epoch", 0)),
            use_fixed_epochs=bool(train_info.get("use_fixed_epochs", True)),
        )
        pred_tables[str(model_key)] = df_pred

    # Apply train-OOF fitted calibrators on leaderboard predictions.
    calibrated: Dict[str, pd.DataFrame] = {}
    for family, dfp in pred_tables.items():
        with log_step(
            "family.calibration.apply",
            family=str(family),
            scope="leaderboard",
            method=str(calibration_fit_method),
            skipped=bool(cfg.skip_family_calibration),
            n_rows=int(len(dfp)),
        ):
            arr = dfp.copy()
            p_cal = np.zeros((len(arr), 4), dtype=np.float32)
            params_by_task = calib_params_by_family.get(str(family), {})
            for t in range(4):
                t_params = params_by_task.get(str(t), {"kind": "identity"})
                p_cal[:, t] = _apply_calibration_task(
                    arr[f"p_t{t}"].to_numpy(dtype=np.float64),
                    t_params,
                )
                arr[f"p_cal_t{t}"] = p_cal[:, t]
                log_event(
                    "INFO",
                    "family.calibration.apply.task",
                    family=str(family),
                    scope="leaderboard",
                    task_idx=int(t),
                    kind=str(t_params.get("kind", "identity")),
                    n_rows=int(len(arr)),
                )
            cal_lb_pred_path = outdir / f"leaderboard_preds_{family}_calibrated.csv"
            arr.to_csv(cal_lb_pred_path, index=False)
            calibrated[family] = arr
            y = np.stack([arr[f"y_t{t}"].to_numpy(dtype=np.int64) for t in range(4)], axis=1)
            w_lb = np.stack(
                [arr[f"w_t{t}"].to_numpy(dtype=np.float32) if f"w_t{t}" in arr.columns else np.ones(len(arr), dtype=np.float32) for t in range(4)],
                axis=1,
            )
            cal_lb_metric_path = outdir / f"leaderboard_metrics_{family}_calibrated.csv"
            cal_lb_metrics = _metric_table(
                y_true=y,
                p_pred=p_cal,
                w_cls=w_lb,
            )
            cal_lb_metrics.to_csv(cal_lb_metric_path, index=False)
            macro_row = cal_lb_metrics[cal_lb_metrics["task"] == "macro"]
            if len(macro_row) == 1:
                log_event(
                    "INFO",
                    "family.calibration.apply.summary",
                    family=str(family),
                    scope="leaderboard",
                    skipped=bool(cfg.skip_family_calibration),
                    macro_pr_auc=f"{float(macro_row.iloc[0]['pr_auc']):.6f}",
                    macro_roc_auc=f"{float(macro_row.iloc[0]['roc_auc']):.6f}",
                    preds_path=str(cal_lb_pred_path),
                    metrics_path=str(cal_lb_metric_path),
                )

    blend_lb_metrics: pd.DataFrame | None = None
    if blend_weights is None:
        log_event(
            "INFO",
            "family.blend.apply.skipped",
            scope="leaderboard",
            reason=str(skip_blend_reason or "blend_not_fitted"),
            n_families=int(len(calibrated)),
        )
    else:
        with log_step("family.blend.apply", scope="leaderboard"):
            fams_lb = [str(x) for x in blend_weights.get("families", []) if str(x) in calibrated]
            if len(fams_lb) == 0:
                raise RuntimeError("No calibrated families available for leaderboard blending.")
            common_ids = set(calibrated[fams_lb[0]]["ID"].astype(str).tolist())
            for f in fams_lb[1:]:
                common_ids &= set(calibrated[f]["ID"].astype(str).tolist())
            common_ids = sorted(common_ids)
            if len(common_ids) == 0:
                raise RuntimeError("No common leaderboard IDs across selected families for blending.")
            log_event(
                "INFO",
                "family.blend.apply.scope",
                scope="leaderboard",
                n_families=int(len(fams_lb)),
                families=",".join([str(x) for x in fams_lb]),
                n_common_ids=int(len(common_ids)),
            )

            y_bl = None
            w_bl = None
            x_by_family: Dict[str, np.ndarray] = {}
            for f in fams_lb:
                d = calibrated[f].copy()
                d["ID"] = d["ID"].astype(str)
                d = d.set_index("ID").loc[common_ids].reset_index(drop=False)
                _require_calibrated_prob_columns(df=d, family=str(f), scope="leaderboard")
                x_by_family[f] = np.stack([d[f"p_cal_t{t}"].to_numpy(dtype=np.float64) for t in range(4)], axis=1)
                log_event(
                    "INFO",
                    "family.blend.apply.family_input",
                    scope="leaderboard",
                    family=str(f),
                    input_source="calibrated_probabilities",
                    n_rows=int(len(d)),
                )
                if y_bl is None:
                    y_bl = np.stack([d[f"y_t{t}"].to_numpy(dtype=np.int64) for t in range(4)], axis=1)
                    w_bl = np.stack(
                        [
                            d[f"w_t{t}"].to_numpy(dtype=np.float32)
                            if f"w_t{t}" in d.columns
                            else np.ones(len(d), dtype=np.float32)
                            for t in range(4)
                        ],
                        axis=1,
                    )
            assert y_bl is not None
            assert w_bl is not None

            blend_pred = np.zeros_like(y_bl, dtype=np.float64)
            for t in range(4):
                X_task = np.column_stack([x_by_family[f][:, t] for f in fams_lb]).astype(np.float64)
                task_cfg = blend_weights.get("tasks", {}).get(
                    str(t),
                    {"kind": "convex_blending", "weights": {f: 1.0 / float(max(1, len(fams_lb))) for f in fams_lb}},
                )
                blend_pred[:, t] = _apply_blend_task(x=X_task, task_cfg=task_cfg, fams=fams_lb)
                log_event(
                    "INFO",
                    "family.blend.apply.task",
                    scope="leaderboard",
                    task_idx=int(t),
                    kind=str(task_cfg.get("kind", "convex_blending")),
                    n_rows=int(X_task.shape[0]),
                )

            blend_df = pd.DataFrame({"ID": common_ids})
            for t in range(4):
                blend_df[f"p_blend_t{t}"] = blend_pred[:, t]
                blend_df[f"y_t{t}"] = y_bl[:, t]
                blend_df[f"pred_t{t}"] = (blend_pred[:, t] >= 0.5).astype(int)
            blend_lb_pred_path = outdir / "leaderboard_preds_blend.csv"
            blend_df.to_csv(blend_lb_pred_path, index=False)
            (outdir / "blend_weights.json").write_text(json.dumps(blend_weights, indent=2))

            blend_lb_metric_path = outdir / "leaderboard_metrics_blend.csv"
            blend_lb_metrics = _metric_table(
                y_true=y_bl,
                p_pred=blend_pred,
                w_cls=w_bl,
            )
            blend_lb_metrics.to_csv(blend_lb_metric_path, index=False)
            macro_row = blend_lb_metrics[blend_lb_metrics["task"] == "macro"]
            if len(macro_row) == 1:
                log_event(
                    "INFO",
                    "family.blend.apply.summary",
                    scope="leaderboard",
                    macro_pr_auc=f"{float(macro_row.iloc[0]['pr_auc']):.6f}",
                    macro_roc_auc=f"{float(macro_row.iloc[0]['roc_auc']):.6f}",
                    preds_path=str(blend_lb_pred_path),
                    metrics_path=str(blend_lb_metric_path),
                )

    # Summary table for direct single-model vs blend comparison on leaderboard.
    blend_by_task = (
        blend_lb_metrics.set_index("task")[["pr_auc", "roc_auc"]]
        if blend_lb_metrics is not None
        else None
    )
    rows_summary: List[Dict[str, Any]] = []
    for model_name, dfm in sorted(leaderboard_metrics_by_family.items(), key=lambda kv: str(kv[0])):
        if dfm is None or len(dfm) == 0:
            continue
        fam_by_task = dfm.set_index("task")[["pr_auc", "roc_auc"]]
        for task_name in [str(t) for t in TASK_COLS]:
            if task_name not in fam_by_task.index:
                continue
            row_summary: Dict[str, Any] = {
                "task": str(task_name),
                "model": str(model_name),
                "pr_auc_cv_mean": float("nan"),
                "pr_auc_cv_sd": float("nan"),
                "pr_auc_cv_ci95_low": float("nan"),
                "pr_auc_cv_ci95_high": float("nan"),
                "pr_auc_cv_ci95_halfwidth": float("nan"),
                "roc_auc_cv_mean": float("nan"),
                "roc_auc_cv_sd": float("nan"),
                "roc_auc_cv_ci95_low": float("nan"),
                "roc_auc_cv_ci95_high": float("nan"),
                "roc_auc_cv_ci95_halfwidth": float("nan"),
                "pr_auc_lbrd": float(fam_by_task.loc[task_name, "pr_auc"]),
                "roc_auc_lbrd": float(fam_by_task.loc[task_name, "roc_auc"]),
                "pr_auc_lbd_blend": float("nan"),
                "roc_auc_lbd_blend": float("nan"),
                "pr_auc_lbrd_blend": float("nan"),
                "roc_auc_lbrd_blend": float("nan"),
            }
            if blend_by_task is not None and task_name in blend_by_task.index:
                row_summary["pr_auc_lbd_blend"] = float(blend_by_task.loc[task_name, "pr_auc"])
                row_summary["roc_auc_lbd_blend"] = float(blend_by_task.loc[task_name, "roc_auc"])
                row_summary["pr_auc_lbrd_blend"] = float(blend_by_task.loc[task_name, "pr_auc"])
                row_summary["roc_auc_lbrd_blend"] = float(blend_by_task.loc[task_name, "roc_auc"])
            rows_summary.append(row_summary)
            dff = cv_fold_metrics_by_model.get(str(model_name))
            if isinstance(dff, pd.DataFrame) and len(dff) > 0:
                cur = dff[dff["task"].astype(str) == str(task_name)].copy()
                if len(cur) > 0:
                    pr_vals = pd.to_numeric(cur["pr_auc"], errors="coerce").to_numpy(dtype=np.float64)
                    roc_vals = pd.to_numeric(cur["roc_auc"], errors="coerce").to_numpy(dtype=np.float64)
                    pr_vals = pr_vals[np.isfinite(pr_vals)]
                    roc_vals = roc_vals[np.isfinite(roc_vals)]
                    if pr_vals.size > 0:
                        mean, sd, lo, hi, half = _mean_sd_ci95(pr_vals)
                        rows_summary[-1]["pr_auc_cv_mean"] = mean
                        rows_summary[-1]["pr_auc_cv_sd"] = sd
                        rows_summary[-1]["pr_auc_cv_ci95_low"] = lo
                        rows_summary[-1]["pr_auc_cv_ci95_high"] = hi
                        rows_summary[-1]["pr_auc_cv_ci95_halfwidth"] = half
                    if roc_vals.size > 0:
                        mean, sd, lo, hi, half = _mean_sd_ci95(roc_vals)
                        rows_summary[-1]["roc_auc_cv_mean"] = mean
                        rows_summary[-1]["roc_auc_cv_sd"] = sd
                        rows_summary[-1]["roc_auc_cv_ci95_low"] = lo
                        rows_summary[-1]["roc_auc_cv_ci95_high"] = hi
                        rows_summary[-1]["roc_auc_cv_ci95_halfwidth"] = half
    if len(rows_summary) > 0:
        results_path = outdir / "results.csv"
        pd.DataFrame(rows_summary).to_csv(results_path, index=False)
        log_event(
            "INFO",
            "family.results.summary_written",
            path=str(results_path),
            n_rows=int(len(rows_summary)),
            n_models=int(len(leaderboard_metrics_by_family)),
            n_tasks=int(len(TASK_COLS)),
            blend_available=bool(blend_lb_metrics is not None),
        )
    log_event(
        "DONE",
        "family_suite.run",
        outdir=str(outdir),
        n_families=int(len(cfg.model_families)),
        n_models_total=int(len(model_specs)),
    )


__all__ = ["run_family_suite", "FAMILY_CHOICES"]
