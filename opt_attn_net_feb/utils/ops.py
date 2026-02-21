from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Sampler, WeightedRandomSampler

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
    rare_target_prev: float = 0.10,
    sample_weight_cap: float = 10.0,
    rare_prev_thr: Optional[float] = None,
) -> WeightedRandomSampler:
    """
    Build a multitask oversampling sampler.

    - Compute per-task prevalence p_t = mean(y[:, t]).
    - Preferred mode (default): task rarity is a smooth deficiency score based on
      `rare_target_prev`: deficiency_t = clip((target - p_t) / target, 0, 1).
      Per-sample rarity = max_t(y_it * deficiency_t).
      Weight: 1 + rare_mult * rarity_i.
    - Legacy fallback mode: if `rare_prev_thr` is provided, use hard thresholding
      (rare if p_t < rare_prev_thr) and binary per-sample rarity.
    - Clamp weights to [1, sample_weight_cap] to avoid extreme oversampling.

    Args:
        y: Binary labels array of shape (N, T).
        rare_mult: Oversampling multiplier applied to positives of rare tasks.
        rare_target_prev: Target prevalence used to compute smooth rarity in auto mode.
        sample_weight_cap: Upper cap for per-sample weight.
        rare_prev_thr: Optional legacy hard threshold. If provided, overrides auto mode.
    """
    y = (y > 0).astype(np.int64)
    if y.ndim != 2 or y.shape[1] == 0:
        w = np.ones((y.shape[0],), dtype=np.float64)
    else:
        prev = y.mean(axis=0).astype(np.float64)

        if rare_prev_thr is not None:
            # Backward-compatible behavior for older configs.
            severity = (prev < float(rare_prev_thr)).astype(np.float64)
        else:
            target = float(np.clip(float(rare_target_prev), 1e-6, 1.0))
            severity = np.clip((target - prev) / target, 0.0, 1.0)

        if np.any(severity > 0.0):
            sample_rarity = np.max(
                y.astype(np.float64) * severity.reshape(1, -1),
                axis=1,
            )
            w = 1.0 + float(rare_mult) * sample_rarity
        else:
            w = np.ones((y.shape[0],), dtype=np.float64)
    cap = max(1.0, float(sample_weight_cap))
    w = np.clip(w, 1.0, cap)
    w_t = torch.tensor(w, dtype=torch.double)
    return WeightedRandomSampler(weights=w_t, num_samples=len(w_t), replacement=True)


class MultitaskBalancedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that enforces a positive quota per batch for multitask training.

    Positives are sampled with rarity-aware weights, negatives uniformly.
    """

    def __init__(
        self,
        *,
        y: np.ndarray,
        batch_size: int,
        rare_mult: float,
        rare_target_prev: float = 0.10,
        sample_weight_cap: float = 10.0,
        batch_pos_fraction: float = 0.35,
        min_pos_per_batch: int = 1,
        rare_prev_thr: Optional[float] = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        yb = (np.asarray(y) > 0).astype(np.int64)
        if yb.ndim != 2:
            raise ValueError(f"Expected y to have shape [N,T], got shape {yb.shape}")
        if yb.shape[0] == 0:
            raise ValueError("Cannot build batch sampler with empty dataset.")

        self.y = yb
        self.batch_size = int(max(1, batch_size))
        self.rare_mult = float(rare_mult)
        self.rare_target_prev = float(rare_target_prev)
        self.sample_weight_cap = float(sample_weight_cap)
        self.batch_pos_fraction = float(np.clip(batch_pos_fraction, 0.0, 1.0))
        self.min_pos_per_batch = int(max(0, min_pos_per_batch))
        self.rare_prev_thr = rare_prev_thr
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        n = int(self.y.shape[0])
        self._num_batches = (
            n // self.batch_size if self.drop_last else int(math.ceil(n / self.batch_size))
        )
        self._all_idx = np.arange(n, dtype=np.int64)

        any_pos = self.y.sum(axis=1) > 0
        self._pos_idx = np.flatnonzero(any_pos)
        self._neg_idx = np.flatnonzero(~any_pos)

        if self._pos_idx.size > 0:
            prev = self.y.mean(axis=0).astype(np.float64)
            if self.rare_prev_thr is not None:
                severity = (prev < float(self.rare_prev_thr)).astype(np.float64)
            else:
                target = float(np.clip(self.rare_target_prev, 1e-6, 1.0))
                severity = np.clip((target - prev) / target, 0.0, 1.0)
            sample_rarity = np.max(self.y.astype(np.float64) * severity.reshape(1, -1), axis=1)
            pos_w = 1.0 + self.rare_mult * sample_rarity[self._pos_idx]
            cap = max(1.0, self.sample_weight_cap)
            pos_w = np.clip(pos_w, 1.0, cap)
            den = float(pos_w.sum())
            if den > 0.0:
                self._pos_prob = pos_w / den
            else:
                self._pos_prob = np.full((self._pos_idx.size,), 1.0 / float(self._pos_idx.size))
        else:
            self._pos_prob = None

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        has_pos = self._pos_idx.size > 0
        has_neg = self._neg_idx.size > 0

        for _ in range(self._num_batches):
            if not has_pos or not has_neg:
                batch = rng.choice(self._all_idx, size=self.batch_size, replace=True)
                yield batch.tolist()
                continue

            target_pos = int(round(self.batch_size * self.batch_pos_fraction))
            n_pos = max(self.min_pos_per_batch, target_pos)
            min_pos = 1 if has_pos else 0
            max_pos = self.batch_size - 1 if has_neg else self.batch_size
            if max_pos < min_pos:
                max_pos = min_pos
            n_pos = int(np.clip(n_pos, min_pos, max_pos))
            n_neg = self.batch_size - n_pos

            pos_draw = rng.choice(self._pos_idx, size=n_pos, replace=True, p=self._pos_prob)
            neg_draw = rng.choice(self._neg_idx, size=n_neg, replace=True)
            batch = np.concatenate([pos_draw, neg_draw], axis=0)
            rng.shuffle(batch)
            yield batch.tolist()


def make_balanced_batch_sampler(
    y: np.ndarray,
    *,
    batch_size: int,
    rare_mult: float,
    rare_target_prev: float = 0.10,
    sample_weight_cap: float = 10.0,
    batch_pos_fraction: float = 0.35,
    min_pos_per_batch: int = 1,
    rare_prev_thr: Optional[float] = None,
    seed: int = 0,
) -> MultitaskBalancedBatchSampler:
    return MultitaskBalancedBatchSampler(
        y=y,
        batch_size=int(batch_size),
        rare_mult=float(rare_mult),
        rare_target_prev=float(rare_target_prev),
        sample_weight_cap=float(sample_weight_cap),
        batch_pos_fraction=float(batch_pos_fraction),
        min_pos_per_batch=int(min_pos_per_batch),
        rare_prev_thr=rare_prev_thr,
        seed=int(seed),
        drop_last=False,
    )


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
