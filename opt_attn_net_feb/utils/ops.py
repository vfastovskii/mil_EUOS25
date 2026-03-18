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

# Scale factors for raw per-row sample-weight columns.
# Requested calibration: 0.50 -> 0.95 for transmittance tasks.
SAMPLE_WEIGHT_COL_SCALE: Dict[str, float] = {
    "sample_weight_340": 1.9,
    "sample_weight_450": 1.9,
}


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
        col = WEIGHT_COLS.get(t)
        if col and col in df_lab.columns:
            scale = float(SAMPLE_WEIGHT_COL_SCALE.get(col, 1.0))
            W[:, t] = (
                df_lab[col]
                .astype(float)
                .fillna(1.0)
                .to_numpy(dtype=np.float32)
                * np.float32(scale)
            )
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
    w340_scale = np.float32(SAMPLE_WEIGHT_COL_SCALE.get("sample_weight_340", 1.0))
    w450_scale = np.float32(SAMPLE_WEIGHT_COL_SCALE.get("sample_weight_450", 1.0))
    w340 = (
        df["sample_weight_340"].astype(float).fillna(1.0).to_numpy(dtype=np.float32) * w340_scale
        if "sample_weight_340" in df.columns
        else np.ones(len(df), dtype=np.float32)
    )
    w450 = (
        df["sample_weight_450"].astype(float).fillna(1.0).to_numpy(dtype=np.float32) * w450_scale
        if "sample_weight_450" in df.columns
        else np.ones(len(df), dtype=np.float32)
    )
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
        self._all_w = np.ones((n,), dtype=np.float64)

        any_pos = self.y.sum(axis=1) > 0
        self._pos_idx = np.flatnonzero(any_pos)
        self._neg_idx = np.flatnonzero(~any_pos)

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

    def __len__(self) -> int:
        return self._num_batches

    @staticmethod
    def _used_to_array(used: Set[int]) -> np.ndarray:
        if len(used) == 0:
            return np.empty((0,), dtype=np.int64)
        return np.fromiter((int(x) for x in used), dtype=np.int64, count=len(used))

    def _available_unique(self, pool_idx: np.ndarray, used: Set[int]) -> np.ndarray:
        if pool_idx.size == 0:
            return np.empty((0,), dtype=np.int64)
        used_arr = self._used_to_array(used)
        if used_arr.size == 0:
            return np.asarray(pool_idx, dtype=np.int64)
        mask = ~np.isin(pool_idx, used_arr, assume_unique=False)
        return np.asarray(pool_idx[mask], dtype=np.int64)

    def _draw_unique(
        self,
        *,
        rng: np.random.Generator,
        pool_idx: np.ndarray,
        take: int,
        used: Set[int],
        weighted: bool,
    ) -> np.ndarray:
        need = int(max(0, take))
        if need <= 0 or pool_idx.size == 0:
            return np.empty((0,), dtype=np.int64)
        avail = self._available_unique(pool_idx, used)
        if avail.size == 0:
            return np.empty((0,), dtype=np.int64)
        k = int(min(need, int(avail.size)))
        if k <= 0:
            return np.empty((0,), dtype=np.int64)
        if k == int(avail.size):
            draw = np.array(avail, copy=True)
            rng.shuffle(draw)
            return draw.astype(np.int64)
        probs = None
        if weighted:
            probs = _normalize_probs(self._all_w[avail])
        draw = rng.choice(avail, size=k, replace=False, p=probs)
        return np.asarray(draw, dtype=np.int64)

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

            used = set()
            pos_draw = self._draw_unique(
                rng=rng,
                pool_idx=self._pos_idx,
                take=n_pos,
                used=used,
                weighted=True,
            )
            used.update(int(x) for x in pos_draw.tolist())
            neg_draw = self._draw_unique(
                rng=rng,
                pool_idx=self._neg_idx,
                take=(self.batch_size - int(pos_draw.size)),
                used=used,
                weighted=False,
            )
            used.update(int(x) for x in neg_draw.tolist())
            fill_needed = int(self.batch_size - int(pos_draw.size) - int(neg_draw.size))
            if fill_needed > 0:
                fill_draw = self._draw_unique(
                    rng=rng,
                    pool_idx=self._all_idx,
                    take=fill_needed,
                    used=used,
                    weighted=False,
                )
                used.update(int(x) for x in fill_draw.tolist())
            else:
                fill_draw = np.empty((0,), dtype=np.int64)
            still_needed = int(self.batch_size - int(pos_draw.size) - int(neg_draw.size) - int(fill_draw.size))
            if still_needed > 0:
                # Extremely small datasets may not have enough unique rows to fill a full batch.
                refill = np.asarray(rng.choice(self._all_idx, size=still_needed, replace=True), dtype=np.int64)
            else:
                refill = np.empty((0,), dtype=np.int64)
            batch = np.concatenate([pos_draw, neg_draw, fill_draw, refill], axis=0)
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


def build_sampler_diagnostics_df(
    *,
    y: np.ndarray,
    batch_size: int,
    use_balanced_batch_sampler: bool,
    rare_mult: float,
    rare_target_prev: float = 0.10,
    sample_weight_cap: float = 10.0,
    batch_pos_fraction: float = 0.35,
    min_pos_per_batch: int = 1,
    rare_prev_thr: Optional[float] = None,
    seed: int = 0,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a compact long-form sampler diagnostics table.

    The report summarizes:
    - mean positive rows per batch
    - duplicate rate within sampled batches
    - unique-row coverage
    - per-task sampled prevalence and exposure lift
    """
    yb = (np.asarray(y) > 0).astype(np.int64)
    if yb.ndim != 2:
        raise ValueError(f"Expected y to have shape [N,T], got shape {yb.shape}")

    batch_size_int = int(max(1, batch_size))
    sampler_mode = "balanced_batch" if bool(use_balanced_batch_sampler) else "weighted_sampler"

    if bool(use_balanced_batch_sampler):
        sampler = make_balanced_batch_sampler(
            yb,
            batch_size=batch_size_int,
            rare_mult=float(rare_mult),
            rare_target_prev=float(rare_target_prev),
            sample_weight_cap=float(sample_weight_cap),
            batch_pos_fraction=float(batch_pos_fraction),
            min_pos_per_batch=int(min_pos_per_batch),
            rare_prev_thr=rare_prev_thr,
            seed=int(seed),
        )
        total_batches = int(len(sampler))
        n_batches = total_batches if max_batches is None else int(min(total_batches, max(0, int(max_batches))))
        batches = [np.asarray(batch, dtype=np.int64) for _, batch in zip(range(n_batches), iter(sampler))]
    else:
        sampler = make_weighted_sampler(
            yb,
            rare_mult=float(rare_mult),
            rare_target_prev=float(rare_target_prev),
            sample_weight_cap=float(sample_weight_cap),
            rare_prev_thr=rare_prev_thr,
        )
        draws = np.fromiter(iter(sampler), dtype=np.int64, count=int(len(sampler)))
        total_batches = int(math.ceil(float(draws.size) / float(batch_size_int))) if draws.size > 0 else 0
        n_batches = total_batches if max_batches is None else int(min(total_batches, max(0, int(max_batches))))
        batches = [
            np.asarray(draws[i * batch_size_int:(i + 1) * batch_size_int], dtype=np.int64)
            for i in range(int(n_batches))
        ]

    batch_sizes = np.asarray([int(len(b)) for b in batches], dtype=np.int64)
    duplicate_counts = np.asarray(
        [int(max(0, len(b) - len(np.unique(b)))) for b in batches],
        dtype=np.int64,
    )
    unique_counts = batch_sizes - duplicate_counts

    any_pos = yb.sum(axis=1) > 0
    batch_pos_counts = np.asarray(
        [int(np.sum(any_pos[b])) for b in batches],
        dtype=np.int64,
    ) if len(batches) > 0 else np.zeros((0,), dtype=np.int64)
    batch_neg_counts = batch_sizes - batch_pos_counts

    sampled_idx = np.concatenate(batches, axis=0) if len(batches) > 0 else np.empty((0,), dtype=np.int64)
    sampled_unique_idx = np.unique(sampled_idx) if sampled_idx.size > 0 else np.empty((0,), dtype=np.int64)

    rows: List[Dict[str, object]] = []
    base_meta: Dict[str, object] = {
        "sampler_mode": str(sampler_mode),
        "batch_size_target": int(batch_size_int),
        "n_dataset_rows": int(yb.shape[0]),
        "n_batches_analyzed": int(len(batches)),
        "rare_mult": float(rare_mult),
        "rare_target_prev": float(rare_target_prev),
        "sample_weight_cap": float(sample_weight_cap),
        "batch_pos_fraction": float(batch_pos_fraction),
        "min_pos_per_batch": int(min_pos_per_batch),
        "seed": int(seed),
    }

    def _append_row(section: str, metric: str, value: float, *, task_idx: Optional[int] = None) -> None:
        rows.append(
            {
                **base_meta,
                "section": str(section),
                "metric": str(metric),
                "task_idx": ("" if task_idx is None else int(task_idx)),
                "task": ("" if task_idx is None else str(TASK_COLS[int(task_idx)])),
                "value": float(value) if np.isfinite(value) else float("nan"),
            }
        )

    total_sampled_rows = int(sampled_idx.size)
    total_duplicates = int(duplicate_counts.sum()) if duplicate_counts.size > 0 else 0
    duplicate_rate = (
        float(total_duplicates / float(total_sampled_rows))
        if total_sampled_rows > 0 else 0.0
    )
    _append_row("summary", "dataset_any_positive_prevalence", float(any_pos.mean()) if any_pos.size > 0 else float("nan"))
    _append_row("summary", "sampled_any_positive_prevalence", float(any_pos[sampled_idx].mean()) if sampled_idx.size > 0 else float("nan"))
    _append_row("summary", "mean_pos_per_batch", float(batch_pos_counts.mean()) if batch_pos_counts.size > 0 else float("nan"))
    _append_row("summary", "mean_neg_per_batch", float(batch_neg_counts.mean()) if batch_neg_counts.size > 0 else float("nan"))
    _append_row("summary", "mean_unique_per_batch", float(unique_counts.mean()) if unique_counts.size > 0 else float("nan"))
    _append_row("summary", "duplicate_rate", float(duplicate_rate))
    _append_row(
        "summary",
        "unique_row_coverage_rate",
        float(sampled_unique_idx.size / float(max(1, yb.shape[0]))),
    )
    _append_row("summary", "sampled_rows_total", float(total_sampled_rows))

    for t in range(int(yb.shape[1])):
        dataset_prev = float(yb[:, t].mean()) if yb.shape[0] > 0 else float("nan")
        sampled_prev = float(yb[sampled_idx, t].mean()) if sampled_idx.size > 0 else float("nan")
        exposure_lift = (
            float(sampled_prev / dataset_prev)
            if np.isfinite(dataset_prev) and dataset_prev > 0.0 and np.isfinite(sampled_prev)
            else float("nan")
        )
        task_pos_idx = np.flatnonzero(yb[:, t] > 0)
        sampled_task_pos = sampled_idx[yb[sampled_idx, t] > 0] if sampled_idx.size > 0 else np.empty((0,), dtype=np.int64)
        unique_task_pos = np.unique(sampled_task_pos) if sampled_task_pos.size > 0 else np.empty((0,), dtype=np.int64)
        unique_positive_coverage = (
            float(unique_task_pos.size / float(task_pos_idx.size))
            if task_pos_idx.size > 0 else float("nan")
        )
        per_batch_pos = np.asarray([int(yb[b, t].sum()) for b in batches], dtype=np.int64) if len(batches) > 0 else np.zeros((0,), dtype=np.int64)
        batch_hit_rate = (
            float(np.mean(per_batch_pos > 0))
            if per_batch_pos.size > 0 else float("nan")
        )
        draws_per_positive = (
            float(sampled_task_pos.size / float(task_pos_idx.size))
            if task_pos_idx.size > 0 else float("nan")
        )
        _append_row("task_exposure", "dataset_prevalence", dataset_prev, task_idx=t)
        _append_row("task_exposure", "sampled_prevalence", sampled_prev, task_idx=t)
        _append_row("task_exposure", "exposure_lift", exposure_lift, task_idx=t)
        _append_row("task_exposure", "mean_positive_rows_per_batch", float(per_batch_pos.mean()) if per_batch_pos.size > 0 else float("nan"), task_idx=t)
        _append_row("task_exposure", "batch_hit_rate", batch_hit_rate, task_idx=t)
        _append_row("task_exposure", "unique_positive_coverage_rate", unique_positive_coverage, task_idx=t)
        _append_row("task_exposure", "positive_draws_per_positive_sample", draws_per_positive, task_idx=t)

    return pd.DataFrame(rows)


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
