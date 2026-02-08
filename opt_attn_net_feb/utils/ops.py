from __future__ import annotations

from typing import List, Optional, Set, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler

from .constants import (
    TASK_COLS,
    AUX_ABS_COLS,
    AUX_FLUO_BASE_COLS,
    WEIGHT_COLS,
)


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def maybe_set_torch_fast_flags():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def infer_feature_cols(df: pd.DataFrame, nonfeat: Set[str]) -> List[str]:
    return sorted([c for c in df.columns if c not in nonfeat])


def coerce_binary_labels(df: pd.DataFrame) -> np.ndarray:
    y = df[TASK_COLS].fillna(0).astype(int).to_numpy()
    return (y > 0).astype(np.int64)


def build_task_weights(df_lab: pd.DataFrame) -> np.ndarray:
    W = np.ones((len(df_lab), 4), dtype=np.float32)
    for t in range(4):
        col = WEIGHT_COLS[t]
        if col in df_lab.columns:
            W[:, t] = df_lab[col].astype(float).fillna(1.0).to_numpy(dtype=np.float32)
    return np.clip(W, 0.0, np.inf)


def get_float_col_or_nan(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full((len(df),), np.nan, dtype=np.float32)
    return df[col].astype(float).to_numpy(dtype=np.float32)


def build_aux_targets_and_masks(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_abs = np.stack([get_float_col_or_nan(df, c) for c in AUX_ABS_COLS], axis=1).astype(np.float32)
    m_abs = np.isfinite(y_abs)

    y_fbase = np.stack([get_float_col_or_nan(df, c) for c in AUX_FLUO_BASE_COLS], axis=1).astype(np.float32)
    m_fbase = np.isfinite(y_fbase)

    y_fluo4 = np.concatenate([y_fbase, y_fbase], axis=1).astype(np.float32)  # (n,4)
    m_fluo4 = np.concatenate([m_fbase, m_fbase], axis=1)
    return y_abs, m_abs, y_fluo4, m_fluo4


def build_aux_weights(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    w340 = df["sample_weight_340"].astype(float).fillna(1.0).to_numpy(dtype=np.float32) if "sample_weight_340" in df.columns else np.ones(len(df), dtype=np.float32)
    w450 = df["sample_weight_450"].astype(float).fillna(1.0).to_numpy(dtype=np.float32) if "sample_weight_450" in df.columns else np.ones(len(df), dtype=np.float32)
    wad  = df["w_ad"].astype(float).fillna(1.0).to_numpy(dtype=np.float32) if "w_ad" in df.columns else np.ones(len(df), dtype=np.float32)

    w_abs = np.stack([w340, w450], axis=1).astype(np.float32)
    w_fluo4 = np.repeat(wad.reshape(-1, 1), 4, axis=1).astype(np.float32)
    return np.clip(w_abs, 0.0, np.inf), np.clip(w_fluo4, 0.0, np.inf)


def pos_weight_per_task(y: np.ndarray, clip) -> torch.Tensor:
    """
    Compute per-task positive class weights with optional per-task clipping.

    Args:
        y: Binary labels array of shape (N, T).
        clip: Either a single float applied to all tasks, or a sequence/array of
              floats of length T providing per-task clips.
    Returns:
        torch.Tensor of shape (T,) with per-task pos_weight values.
    """
    pos = y.sum(axis=0).astype(np.float64)
    neg = (y.shape[0] - pos).astype(np.float64)
    ratio = neg / np.maximum(pos, 1.0)

    # Apply clipping: support scalar or per-task sequence
    try:
        is_scalar = np.isscalar(clip)
    except Exception:
        is_scalar = False
    if is_scalar:
        c = float(clip)
        ratio = np.minimum(ratio, c)
    else:
        c = np.asarray(clip, dtype=np.float64).reshape(-1)
        if c.size == 1:
            c = np.repeat(c, ratio.size)
        if c.size != ratio.size:
            raise ValueError(f"clip length {c.size} does not match number of tasks {ratio.size}")
        ratio = np.minimum(ratio, c)
    return torch.tensor(ratio, dtype=torch.float32)


def lambda_from_prevalence(y: np.ndarray, power: float) -> np.ndarray:
    p = y.mean(axis=0) + 1e-12
    lam = (1.0 / p) ** power
    lam = lam / lam.mean()
    return lam.astype(np.float32)


def make_weighted_sampler(
    y: np.ndarray,
    rare_mult: float,
    rare_prev_thr: float = 0.02,
    sample_weight_cap: float = 10.0,
) -> WeightedRandomSampler:
    """
    Build a data-driven oversampling sampler.

    - Compute per-task prevalence p_t = mean(y[:, t]).
    - Mark tasks rare if p_t < rare_prev_thr.
    - Per-sample weight: 1 + rare_mult * I(sample is positive for ANY rare task).
    - Clamp weights to [1, sample_weight_cap] to avoid extreme oversampling.

    Args:
        y: Binary labels array of shape (N, T).
        rare_mult: Oversampling multiplier applied to positives of rare tasks.
        rare_prev_thr: Prevalence threshold to define rare tasks.
        sample_weight_cap: Upper cap for per-sample weight.
    """
    y = (y > 0).astype(np.int64)
    if y.ndim != 2 or y.shape[1] == 0:
        w = np.ones((y.shape[0],), dtype=np.float64)
    else:
        prev = y.mean(axis=0)
        rare_tasks = prev < float(rare_prev_thr)
        if rare_tasks.any():
            rare_pos = (y[:, rare_tasks].sum(axis=1) > 0).astype(np.float64)
            w = 1.0 + float(rare_mult) * rare_pos
        else:
            w = np.ones((y.shape[0],), dtype=np.float64)
    cap = max(1.0, float(sample_weight_cap))
    w = np.clip(w, 1.0, cap)
    w_t = torch.tensor(w, dtype=torch.double)
    return WeightedRandomSampler(weights=w_t, num_samples=len(w_t), replacement=True)


def fit_standardizer(y: np.ndarray, m: np.ndarray, tr_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C = y.shape[1]
    mu = np.zeros((C,), dtype=np.float32)
    sd = np.ones((C,), dtype=np.float32)
    for c in range(C):
        vals = y[tr_idx, c][m[tr_idx, c]]
        if vals.size > 0:
            mu[c] = float(vals.mean())
            s = float(vals.std())
            sd[c] = float(s if s > 1e-6 else 1.0)
        else:
            mu[c] = 0.0
            sd[c] = 1.0
    return mu, sd


def apply_standardizer(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((y - mu.reshape(1, -1)) / sd.reshape(1, -1)).astype(np.float32)


def fold_indices(df: pd.DataFrame, fold_col: str, folds: List[int]) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    out = []
    arr = df[fold_col].astype(int).to_numpy()
    for f in folds:
        va = np.where(arr == f)[0]
        tr = np.where(arr != f)[0]
        if len(va) == 0:
            raise ValueError(f"Fold {f} has 0 samples.")
        out.append((tr, va, f))
    return out
