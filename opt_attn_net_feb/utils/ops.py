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


def bitmask_ids(y: np.ndarray) -> np.ndarray:
    """Encode multitask binary labels into integer bitmasks."""
    yb = (np.asarray(y) > 0).astype(np.int64)
    if yb.ndim != 2:
        raise ValueError(f"Expected y to have shape [N,T], got shape {yb.shape}")
    bits = (1 << np.arange(yb.shape[1], dtype=np.int64)).reshape(1, -1)
    return (yb * bits).sum(axis=1).astype(np.int64)


def make_bitmask_sample_weights(
    y: np.ndarray,
    *,
    alpha: float = 0.5,
    cap: float = 3.0,
) -> np.ndarray:
    """
    Build per-sample weights based on bitmask-frequency rarity.

    Weight for sample i with bitmask m_i:
      w_i = clip((median_nonzero_count / count(m_i)) ** alpha, 1, cap)
    """
    yb = (np.asarray(y) > 0).astype(np.int64)
    if yb.ndim != 2:
        raise ValueError(f"Expected y to have shape [N,T], got shape {yb.shape}")
    if yb.shape[0] == 0:
        return np.ones((0,), dtype=np.float32)

    m = bitmask_ids(yb)
    max_mask = int(1 << yb.shape[1])
    cnt = np.bincount(m, minlength=max_mask).astype(np.float64)
    nonzero = cnt[cnt > 0]
    if nonzero.size == 0:
        return np.ones((yb.shape[0],), dtype=np.float32)

    ref = float(np.median(nonzero))
    per_sample_cnt = cnt[m]
    w = (ref / np.maximum(per_sample_cnt, 1.0)) ** max(0.0, float(alpha))
    return np.clip(w, 1.0, max(1.0, float(cap))).astype(np.float32)


def build_bitmask_group_definition(
    y_train: np.ndarray,
    *,
    top_k: int = 6,
    class_weight_alpha: float = 0.5,
    class_weight_cap: float = 5.0,
) -> Tuple[List[int], np.ndarray]:
    """
    Build bitmask grouping (top frequent masks + other) from train fold only.

    Returns:
        top_mask_ids: list of selected frequent mask IDs.
        class_weight: np.ndarray of shape [len(top_mask_ids)+1] for CE weighting.
    """
    yb = (np.asarray(y_train) > 0).astype(np.int64)
    if yb.ndim != 2:
        raise ValueError(f"Expected y_train to have shape [N,T], got shape {yb.shape}")
    if yb.shape[0] == 0:
        return [], np.ones((1,), dtype=np.float32)

    mask_ids = bitmask_ids(yb)
    n_masks = int(1 << yb.shape[1])
    counts = np.bincount(mask_ids, minlength=n_masks).astype(np.float64)

    k = max(0, int(top_k))
    if k == 0:
        top_ids: List[int] = []
    else:
        ranked = [int(i) for i in np.argsort(-counts).tolist() if counts[int(i)] > 0.0]
        top_ids = ranked[:k]

    other_idx = len(top_ids)
    mask_to_group = np.full((n_masks,), other_idx, dtype=np.int64)
    for g, mid in enumerate(top_ids):
        mask_to_group[int(mid)] = int(g)
    groups = mask_to_group[mask_ids]
    group_counts = np.bincount(groups, minlength=other_idx + 1).astype(np.float64)

    nonzero = group_counts[group_counts > 0.0]
    if nonzero.size == 0:
        class_weight = np.ones((other_idx + 1,), dtype=np.float32)
    else:
        ref = float(np.median(nonzero))
        cw = (ref / np.maximum(group_counts, 1.0)) ** max(0.0, float(class_weight_alpha))
        class_weight = np.clip(
            cw,
            1.0,
            max(1.0, float(class_weight_cap)),
        ).astype(np.float32)

    return top_ids, class_weight


def _task_rarity_severity(
    y: np.ndarray,
    *,
    rare_target_prev: float,
    rare_prev_thr: Optional[float],
) -> np.ndarray:
    prev = y.mean(axis=0).astype(np.float64)
    if rare_prev_thr is not None:
        return (prev < float(rare_prev_thr)).astype(np.float64)
    target = float(np.clip(float(rare_target_prev), 1e-6, 1.0))
    return np.clip((target - prev) / target, 0.0, 1.0)


def _sample_rarity(y: np.ndarray, severity: np.ndarray) -> np.ndarray:
    if y.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.max(y.astype(np.float64) * severity.reshape(1, -1), axis=1)


def _normalize_probs(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return w
    den = float(w.sum())
    if den <= 0.0 or not np.isfinite(den):
        return np.full((w.size,), 1.0 / float(w.size), dtype=np.float64)
    return w / den


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
        severity = _task_rarity_severity(
            y,
            rare_target_prev=float(rare_target_prev),
            rare_prev_thr=rare_prev_thr,
        )
        if np.any(severity > 0.0):
            sample_rarity = _sample_rarity(y, severity)
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
        enforce_bitmask_quota: bool = True,
        quota_t450_per_256: int = 4,
        quota_fgt480_per_256: int = 1,
        quota_multi_per_256: int = 8,
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
        self.enforce_bitmask_quota = bool(enforce_bitmask_quota)
        self.quota_t450_per_256 = int(max(0, quota_t450_per_256))
        self.quota_fgt480_per_256 = int(max(0, quota_fgt480_per_256))
        self.quota_multi_per_256 = int(max(0, quota_multi_per_256))
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
        self._t450_idx = np.flatnonzero(self.y[:, 1] > 0) if self.y.shape[1] > 1 else np.array([], dtype=np.int64)
        self._fgt480_idx = np.flatnonzero(self.y[:, 3] > 0) if self.y.shape[1] > 3 else np.array([], dtype=np.int64)
        self._multi_idx = np.flatnonzero(self.y.sum(axis=1) >= 2)

        if self._pos_idx.size > 0:
            severity = _task_rarity_severity(
                self.y,
                rare_target_prev=self.rare_target_prev,
                rare_prev_thr=self.rare_prev_thr,
            )
            sample_rarity = _sample_rarity(self.y, severity)
            all_w = 1.0 + self.rare_mult * sample_rarity
            cap = max(1.0, self.sample_weight_cap)
            self._all_w = np.clip(all_w, 1.0, cap)
            self._pos_prob = _normalize_probs(self._all_w[self._pos_idx])
            self._t450_prob = _normalize_probs(self._all_w[self._t450_idx]) if self._t450_idx.size > 0 else np.array([], dtype=np.float64)
            self._fgt480_prob = _normalize_probs(self._all_w[self._fgt480_idx]) if self._fgt480_idx.size > 0 else np.array([], dtype=np.float64)
            self._multi_prob = _normalize_probs(self._all_w[self._multi_idx]) if self._multi_idx.size > 0 else np.array([], dtype=np.float64)
        else:
            self._pos_prob = None
            self._t450_prob = np.array([], dtype=np.float64)
            self._fgt480_prob = np.array([], dtype=np.float64)
            self._multi_prob = np.array([], dtype=np.float64)

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

            if self.enforce_bitmask_quota and n_pos > 0:
                remaining = n_pos
                pos_chunks: List[np.ndarray] = []

                def _scaled_quota(per_256: int) -> int:
                    if per_256 <= 0:
                        return 0
                    q = int(round(float(self.batch_size) * (float(per_256) / 256.0)))
                    return max(1, q)

                quota_specs = [
                    # Prioritize rarest endpoint first.
                    (self._fgt480_idx, self._fgt480_prob, _scaled_quota(self.quota_fgt480_per_256)),
                    (self._t450_idx, self._t450_prob, _scaled_quota(self.quota_t450_per_256)),
                    (self._multi_idx, self._multi_prob, _scaled_quota(self.quota_multi_per_256)),
                ]

                for pool_idx, pool_prob, q in quota_specs:
                    if remaining <= 0 or q <= 0 or pool_idx.size == 0:
                        continue
                    take = min(q, remaining)
                    if take > 0:
                        pos_chunks.append(rng.choice(pool_idx, size=take, replace=True, p=pool_prob))
                        remaining -= take

                if remaining > 0:
                    pos_chunks.append(
                        rng.choice(self._pos_idx, size=remaining, replace=True, p=self._pos_prob)
                    )
                pos_draw = np.concatenate(pos_chunks, axis=0) if len(pos_chunks) > 0 else np.array([], dtype=np.int64)
            else:
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
    enforce_bitmask_quota: bool = True,
    quota_t450_per_256: int = 4,
    quota_fgt480_per_256: int = 1,
    quota_multi_per_256: int = 8,
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
        enforce_bitmask_quota=bool(enforce_bitmask_quota),
        quota_t450_per_256=int(quota_t450_per_256),
        quota_fgt480_per_256=int(quota_fgt480_per_256),
        quota_multi_per_256=int(quota_multi_per_256),
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
