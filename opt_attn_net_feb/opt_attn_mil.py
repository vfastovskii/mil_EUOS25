#!/usr/bin/env python3
from __future__ import annotations

# DEPRECATION GUARD: This monolithic module is legacy and must not be imported.
# The codebase has been refactored into modular packages under opt_attn_net_feb/.
# Importing this file will raise to prevent accidental use. Migrate to:
#   - opt_attn_net_feb.models.mil_task_attn_mixer_with_aux
#   - opt_attn_net_feb.data (datasets, collate, exports)
#   - opt_attn_net_feb.train (trainer, hpo)
#   - opt_attn_net_feb.utils (ops, metrics, data_io, constants)
raise ImportError(
    "opt_attn_net_feb.opt_attn_mil is deprecated and disabled. Use the modular API under opt_attn_net_feb/."
)

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import optuna
from optuna.trial import Trial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from torch.cuda.amp import autocast

try:
    from pytorch_lightning.callbacks import Callback
except Exception:
    from lightning.pytorch.callbacks import Callback  # fallback


# =========================
# CONFIG
# =========================
TASK_COLS = [
    "Transmittance_340",
    "Transmittance_450",
    "Fluorescence_340_450",
    "Fluorescence_more_than_480",
]

AUX_ABS_COLS = [
    "Transmittance_340_quantitative",
    "Transmittance_450_quantitative",
]

AUX_FLUO_BASE_COLS = ["wl_pred_nm", "qy_pred"]

WEIGHT_COLS = {
    0: "sample_weight_340",
    1: "sample_weight_450",
    2: "w_ad",
    3: "w_ad",
}

NONFEAT_2D = {"ID", "curated_SMILES", "split"}
NONFEAT_3D = {"ID", "conf_id", "smiles", "split"}
NONFEAT_QM = {"record_index", "ID", "conf_id", "status", "error", "split"}


# =========================
# OPTUNA PRUNING CALLBACK
# =========================
class OptunaPruningCallbackLocal(Callback):
    def __init__(self, trial: Trial, monitor: str = "val_macro_ap"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor, None)
        if current is None:
            return
        current_val = float(current.detach().cpu().item()) if torch.is_tensor(current) else float(current)
        self.trial.report(current_val, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {trainer.current_epoch}")


# =========================
# UTILS
# =========================
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


def infer_feature_cols(df: pd.DataFrame, nonfeat: set) -> List[str]:
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


def pos_weight_per_task(y: np.ndarray, clip: float) -> torch.Tensor:
    pos = y.sum(axis=0).astype(np.float64)
    neg = (y.shape[0] - pos).astype(np.float64)
    ratio = neg / np.maximum(pos, 1.0)
    ratio = np.minimum(ratio, clip)
    return torch.tensor(ratio, dtype=torch.float32)


def lambda_from_prevalence(y: np.ndarray, power: float) -> np.ndarray:
    p = y.mean(axis=0) + 1e-12
    lam = (1.0 / p) ** power
    lam = lam / lam.mean()
    return lam.astype(np.float32)


def make_weighted_sampler(y: np.ndarray, rare_mult: float) -> WeightedRandomSampler:
    y = y.astype(int)
    rare_pos = ((y[:, 1] == 1) | (y[:, 3] == 1)).astype(np.float64)
    w = 1.0 + float(rare_mult) * rare_pos
    w = torch.tensor(w, dtype=torch.double)
    return WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)


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


# =========================
# METRICS
# =========================
def ap_per_task(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    w_cls: Optional[np.ndarray] = None,
    weighted_tasks: Tuple[int, ...] = (0, 1),
) -> List[float]:
    y_true = y_true.astype(int)
    p_pred = np.nan_to_num(p_pred, nan=0.0, posinf=1.0, neginf=0.0)
    p_pred = np.clip(p_pred, 0.0, 1.0)

    out: List[float] = []
    for t in range(4):
        if y_true[:, t].sum() == 0:
            out.append(0.0)
            continue
        sw = None
        if w_cls is not None and t in weighted_tasks:
            sw = np.asarray(w_cls[:, t], dtype=float)
        try:
            out.append(float(average_precision_score(y_true[:, t], p_pred[:, t], sample_weight=sw)))
        except ValueError:
            out.append(0.0)
    return out


# =========================
# IO
# =========================
def load_labels(labels_csv: str, id_col="ID") -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if id_col not in df.columns:
        raise ValueError(f"labels missing {id_col}")
    df[id_col] = df[id_col].astype(str)
    return df


def load_2d(feat2d_csv: str, id_col="ID") -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(feat2d_csv)
    if id_col not in df.columns:
        raise ValueError(f"2d features missing {id_col}")
    df[id_col] = df[id_col].astype(str)

    # Logging: uniqueness and duplicates in 2D scaled features
    n_rows = len(df)
    n_unique_ids = df[id_col].nunique(dropna=False)
    dups_mask = df.duplicated([id_col], keep=False)
    if dups_mask.any():
        dup_counts = df.groupby(id_col, dropna=False).size()
        dup_counts = dup_counts[dup_counts > 1].sort_values(ascending=False)
        total_extra_rows = int((dup_counts - 1).sum())
        examples = ", ".join([f"{str(i)}×{int(dup_counts.loc[i])}" for i in dup_counts.index[:10]])
        print(f"[DATA-2D] rows={n_rows} unique_ids={n_unique_ids} duplicated_ids={len(dup_counts)} total_extra_rows={total_extra_rows}")
        if examples:
            print(f"[DATA-2D] duplicate ID examples (up to 10): {examples}")
    else:
        print(f"[DATA-2D] rows={n_rows} unique_ids={n_unique_ids} duplicated_ids=0")

    feat_cols = infer_feature_cols(df, NONFEAT_2D)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    ids = df[id_col].astype(str).tolist()
    return ids, X


def align_by_id(ids_file: List[str], X: np.ndarray, ids_target: List[str]) -> np.ndarray:
    id2row = {str(i): r for r, i in enumerate(ids_file)}
    miss = [i for i in ids_target if str(i) not in id2row]
    if miss:
        raise ValueError(f"[2D] Missing {len(miss)} IDs (first 10): {miss[:10]}")
    rows = np.array([id2row[str(i)] for i in ids_target], dtype=np.int64)
    return X[rows]


def load_and_merge_instances(
    geom_csv: str,
    qm_csv: str,
    allowed_ids: Optional[Set[str]],
    id_col="ID",
    conf_col="conf_id",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dg = pd.read_csv(geom_csv)
    dq = pd.read_csv(qm_csv)

    for d, name in [(dg, "3d_scaled"), (dq, "3d_qm_scaled")]:
        for c in [id_col, conf_col]:
            if c not in d.columns:
                raise ValueError(f"{name} missing column {c}")

    dg[id_col] = dg[id_col].astype(str)
    dg[conf_col] = dg[conf_col].astype(str)
    dq[id_col] = dq[id_col].astype(str)
    dq[conf_col] = dq[conf_col].astype(str)

    if allowed_ids is not None:
        dg = dg[dg[id_col].isin(allowed_ids)].copy()
        dq = dq[dq[id_col].isin(allowed_ids)].copy()

    if "error" in dq.columns:
        dq = dq[dq["error"].isna() | (dq["error"].astype(str).str.len() == 0)].copy()
    if "status" in dq.columns:
        s = dq["status"].astype(str).str.lower()
        ok = s.isin(["ok", "success", "0", "1", "true"]) | dq["status"].isna()
        dq = dq[ok].copy()

    # Logging: uniqueness and duplicates in 3D geometry and quantum tables
    # Geometry
    n_rows_g = len(dg)
    n_unique_ids_g = dg[id_col].nunique(dropna=False)
    n_unique_pairs_g = dg[[id_col, conf_col]].drop_duplicates().shape[0]
    grp_g = dg.groupby([id_col, conf_col], dropna=False).size()
    dup_g = grp_g[grp_g > 1].sort_values(ascending=False)
    if len(dup_g) > 0:
        total_extra_rows_g = int((dup_g - 1).sum())
        dup_ids_g = pd.Index([k[0] for k in dup_g.index]).nunique()
        ex_g = ", ".join([f"{str(k[0])}|{str(k[1])}×{int(v)}" for k, v in dup_g.head(10).items()])
        print(f"[DATA-3D-GEOM] rows={n_rows_g} unique_ids={n_unique_ids_g} unique_pairs={n_unique_pairs_g} duplicated_pairs={len(dup_g)} duplicated_ids={dup_ids_g} total_extra_rows={total_extra_rows_g}")
        if ex_g:
            print(f"[DATA-3D-GEOM] duplicate (ID,conf) examples (up to 10): {ex_g}")
    else:
        print(f"[DATA-3D-GEOM] rows={n_rows_g} unique_ids={n_unique_ids_g} unique_pairs={n_unique_pairs_g} duplicated_pairs=0")

    # Quantum
    n_rows_q = len(dq)
    n_unique_ids_q = dq[id_col].nunique(dropna=False)
    n_unique_pairs_q = dq[[id_col, conf_col]].drop_duplicates().shape[0]
    grp_q = dq.groupby([id_col, conf_col], dropna=False).size()
    dup_q = grp_q[grp_q > 1].sort_values(ascending=False)
    if len(dup_q) > 0:
        total_extra_rows_q = int((dup_q - 1).sum())
        dup_ids_q = pd.Index([k[0] for k in dup_q.index]).nunique()
        ex_q = ", ".join([f"{str(k[0])}|{str(k[1])}×{int(v)}" for k, v in dup_q.head(10).items()])
        print(f"[DATA-3D-QM] rows={n_rows_q} unique_ids={n_unique_ids_q} unique_pairs={n_unique_pairs_q} duplicated_pairs={len(dup_q)} duplicated_ids={dup_ids_q} total_extra_rows={total_extra_rows_q}")
        if ex_q:
            print(f"[DATA-3D-QM] duplicate (ID,conf) examples (up to 10): {ex_q}")
    else:
        print(f"[DATA-3D-QM] rows={n_rows_q} unique_ids={n_unique_ids_q} unique_pairs={n_unique_pairs_q} duplicated_pairs=0")

    g_cols = infer_feature_cols(dg, NONFEAT_3D)
    q_cols = infer_feature_cols(dq, NONFEAT_QM)

    # Deduplicate potential repeated (ID, conf_id) keys by averaging features
    dg_sub = dg[[id_col, conf_col] + g_cols].copy()
    dq_sub = dq[[id_col, conf_col] + q_cols].copy()

    if dg_sub.duplicated([id_col, conf_col]).any():
        dg_sub = dg_sub.groupby([id_col, conf_col], as_index=False)[g_cols].mean()
    if dq_sub.duplicated([id_col, conf_col]).any():
        dq_sub = dq_sub.groupby([id_col, conf_col], as_index=False)[q_cols].mean()

    m = dg_sub.merge(
        dq_sub,
        on=[id_col, conf_col],
        how="inner",
        validate="one_to_one",
    )

    ids_conf = m[id_col].astype(str).to_numpy()
    conf_ids = m[conf_col].astype(str).to_numpy()
    Xg = m[g_cols].to_numpy(dtype=np.float32)
    Xq = m[q_cols].to_numpy(dtype=np.float32)
    X = np.hstack([Xg, Xq]).astype(np.float32)
    return ids_conf, conf_ids, X


def build_instance_index(
    ids_conf: np.ndarray,
    conf_ids: np.ndarray,
    Xinst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], np.ndarray, np.ndarray]:
    # stable sort by ID to pack each bag contiguously
    order = np.argsort(ids_conf, kind="mergesort")
    ids_sorted = ids_conf[order]
    X_sorted = Xinst[order]
    conf_sorted = conf_ids[order]

    uniq, starts, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    id2pos = {str(u): i for i, u in enumerate(uniq)}
    return uniq, starts.astype(np.int64), counts.astype(np.int64), id2pos, X_sorted, conf_sorted


# =========================
# LOSSES
# =========================
class MultiTaskFocal(nn.Module):
    """
    Weighted focal BCE per task:
      - w: per-sample per-task weights
      - returns per-task scalar losses (shape [4])
    """
    def __init__(self, pos_weight: torch.Tensor, gamma: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("gamma", gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = (1 - pt).pow(self.gamma.view(1, -1))
        loss = focal * bce

        w = torch.clamp(w, min=0.0)
        num = (loss * w).sum(dim=0)
        den = w.sum(dim=0).clamp(min=1e-6)
        return num / den


def reg_loss_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    # Ensure boolean mask and avoid computing loss on invalid targets (prevents NaNs)
    mask_b = mask.bool()
    mask_f = mask_b.float()
    w_eff = torch.clamp(w, min=0.0) * mask_f

    target_safe = torch.where(mask_b, target, pred.detach())

    if loss_type == "smoothl1":
        per = F.smooth_l1_loss(pred, target_safe, reduction="none")
    elif loss_type == "mse":
        per = (pred - target_safe).pow(2)
    else:
        raise ValueError(f"Unknown reg loss type: {loss_type}")

    per = per * mask_f
    num = (per * w_eff).sum(dim=0)
    den = w_eff.sum(dim=0).clamp(min=1e-6)
    return (num / den).mean()


# =========================
# MODEL
# =========================
def make_mlp(in_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Module:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(int(n_layers)):
        layers += [
            nn.Linear(d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        ]
        d = hidden_dim
    return nn.Sequential(*layers)


class TaskAttentionPool(nn.Module):
    """
    Multi-query attention pooling:
      - queries: [T=4] learned vectors
      - returns:
        pooled: [B, T, D]
        attn:   [B, T, N] (masked PAD=0 and renormalized to sum 1 over N)
    """
    def __init__(self, dim: int, n_heads: int, dropout: float, n_tasks: int = 4):
        super().__init__()
        dim = int(dim)
        n_heads = int(n_heads)
        n_tasks = int(n_tasks)
        if dim % n_heads != 0:
            raise ValueError(f"TaskAttentionPool: dim={dim} must be divisible by n_heads={n_heads}")
        self.n_tasks = n_tasks
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.q = nn.Parameter(torch.randn(1, n_tasks, dim) * 0.02)

    def forward(
        self,
        tokens: torch.Tensor,           # [B, N, D]
        key_padding_mask: torch.Tensor, # [B, N] True=PAD
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = tokens.shape[0]
        q = self.q.expand(B, -1, -1)  # [B, T, D]

        if not return_attn:
            out, _ = self.mha(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
            return out, None

        # Try to get per-head weights (torch>=2); fallback to averaged weights.
        try:
            out, attn = self.mha(
                q, tokens, tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # [B, H, T, N]
            )
            attn = attn.mean(dim=1)  # -> [B, T, N]
        except TypeError:
            out, attn = self.mha(
                q, tokens, tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,  # [B, T, N]
            )

        # Mask PAD to 0 and renormalize so sum over instances == 1 for each (B,T)
        pad = key_padding_mask.unsqueeze(1).expand(-1, attn.shape[1], -1)  # [B,T,N]
        attn = attn.masked_fill(pad, 0.0)
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        attn = attn / denom
        return out, attn


class MILTaskAttnMixerWithAux(pl.LightningModule):
    """
    - 2D MLP encoder -> e2d
    - 3D MLP encoder -> tokens
    - task-specific attention pooling -> pooled per task + attn maps
    - project 2D and pooled-3D to same dim, concat, mixer -> z_task
    - cls logits from task-specific z_task
    - aux heads from mean(z_task)
    """
    def __init__(
        self,
        mol_dim: int,
        inst_dim: int,
        mol_hidden: int,
        mol_layers: int,
        mol_dropout: float,
        inst_hidden: int,
        inst_layers: int,
        inst_dropout: float,
        proj_dim: int,
        attn_heads: int,
        attn_dropout: float,
        mixer_hidden: int,
        mixer_layers: int,
        mixer_dropout: float,
        lr: float,
        weight_decay: float,
        pos_weight: torch.Tensor,
        gamma: torch.Tensor,
        lam: np.ndarray,
        lambda_aux_abs: float,
        lambda_aux_fluo: float,
        reg_loss_type: str,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight", "gamma", "lam"])

        self.mol_enc = make_mlp(int(mol_dim), int(mol_hidden), int(mol_layers), float(mol_dropout))
        self.inst_enc = make_mlp(int(inst_dim), int(inst_hidden), int(inst_layers), float(inst_dropout))

        self.attn_pool = TaskAttentionPool(dim=int(inst_hidden), n_heads=int(attn_heads), dropout=float(attn_dropout), n_tasks=4)

        self.proj2d = nn.Sequential(nn.Linear(int(mol_hidden), int(proj_dim)), nn.LayerNorm(int(proj_dim)))
        self.proj3d = nn.Sequential(nn.Linear(int(inst_hidden), int(proj_dim)), nn.LayerNorm(int(proj_dim)))

        self.mixer = make_mlp(int(2 * proj_dim), int(mixer_hidden), int(mixer_layers), float(mixer_dropout))

        self.cls_heads = nn.ModuleList([nn.Linear(int(mixer_hidden), 1) for _ in range(4)])
        self.abs_heads = nn.ModuleList([nn.Linear(int(mixer_hidden), 1) for _ in range(2)])
        self.fluo_heads = nn.ModuleList([nn.Linear(int(mixer_hidden), 1) for _ in range(4)])

        self.cls_loss = MultiTaskFocal(pos_weight=pos_weight, gamma=gamma)
        self.register_buffer("lam", torch.tensor(lam, dtype=torch.float32))

        self.lambda_aux_abs = float(lambda_aux_abs)
        self.lambda_aux_fluo = float(lambda_aux_fluo)
        self.reg_loss_type = str(reg_loss_type)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self._val_p: List[np.ndarray] = []
        self._val_y: List[np.ndarray] = []
        self._val_w: List[np.ndarray] = []

    def forward(
        self,
        x2d: torch.Tensor,              # [B,F2]
        x3d_pad: torch.Tensor,          # [B,N,F3]
        key_padding_mask: torch.Tensor, # [B,N] True=PAD
        return_attn: bool = False,
    ):
        B, N, F3 = x3d_pad.shape
        tok = self.inst_enc(x3d_pad.reshape(B * N, F3)).reshape(B, N, -1)  # [B,N,inst_hidden]
        tok = F.layer_norm(tok, (tok.shape[-1],))

        pooled_tasks, attn = self.attn_pool(tok, key_padding_mask=key_padding_mask, return_attn=return_attn)
        # pooled_tasks: [B,4,inst_hidden]; attn: [B,4,N] if return_attn else None

        e2d = self.proj2d(self.mol_enc(x2d))  # [B,proj]
        e2d_rep = e2d.unsqueeze(1).expand(-1, 4, -1)  # [B,4,proj]

        e3d = self.proj3d(pooled_tasks.reshape(B * 4, -1)).reshape(B, 4, -1)  # [B,4,proj]

        mix_in = torch.cat([e2d_rep, e3d], dim=2).reshape(B * 4, -1)  # [B*4,2*proj]
        z_tasks = self.mixer(mix_in).reshape(B, 4, -1)  # [B,4,mixer_hidden]

        logits = torch.cat([self.cls_heads[t](z_tasks[:, t, :]) for t in range(4)], dim=1)  # [B,4]

        z_aux = z_tasks.detach().mean(dim=1)  # [B,mixer_hidden]
        abs_out = torch.cat([hd(z_aux) for hd in self.abs_heads], dim=1)   # [B,2]
        fluo_out = torch.cat([hd(z_aux) for hd in self.fluo_heads], dim=1) # [B,4]

        if return_attn:
            return logits, abs_out, fluo_out, attn
        return logits, abs_out, fluo_out

    def training_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo = batch
        logits, abs_out, fluo_out = self(x2d, x3d, kpm, return_attn=False)

        with autocast(enabled=False):
            per_task = self.cls_loss(logits.float(), y_cls.float(), w_cls.float())
            loss_cls = (per_task * self.lam.float()).mean()

            loss_abs = reg_loss_weighted(abs_out.float(), y_abs.float(), m_abs, w_abs.float(), self.reg_loss_type)
            loss_fluo = reg_loss_weighted(fluo_out.float(), y_fluo.float(), m_fluo, w_fluo.float(), self.reg_loss_type)

            loss = loss_cls + self.lambda_aux_abs * loss_abs + self.lambda_aux_fluo * loss_fluo

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=int(y_cls.shape[0]))
        return loss

    def on_validation_epoch_start(self):
        self._val_p, self._val_y, self._val_w = [], [], []

    def validation_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, *_ = batch
        logits, _, _ = self(x2d, x3d, kpm, return_attn=False)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = y_cls.detach().cpu().numpy().astype(int)
        w = w_cls.detach().cpu().numpy().astype(np.float32)

        self._val_p.append(p)
        self._val_y.append(y)
        self._val_w.append(w)

    def on_validation_epoch_end(self):
        if not self._val_p:
            return
        P = np.concatenate(self._val_p, axis=0)
        Y = np.concatenate(self._val_y, axis=0).astype(int)
        W = np.concatenate(self._val_w, axis=0).astype(np.float32)

        aps = ap_per_task(Y, P, w_cls=W, weighted_tasks=(0, 1))
        mac = float(np.mean(aps))

        for t in range(4):
            self.log(f"val_ap_{t}", float(aps[t]), prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_macro_ap", float(mac), prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# =========================
# DATASETS + COLLATE
# =========================
class MILTrainDataset(Dataset):
    """One item = one molecule (bag)."""
    def __init__(
        self,
        ids: List[str],
        X2d: np.ndarray,
        y_cls: np.ndarray, w_cls: np.ndarray,
        y_abs: np.ndarray, m_abs: np.ndarray, w_abs: np.ndarray,
        y_fluo: np.ndarray, m_fluo: np.ndarray, w_fluo: np.ndarray,
        starts: np.ndarray, counts: np.ndarray, id2pos: Dict[str, int], Xinst_sorted: np.ndarray,
        max_instances: int,
        seed: int,
    ):
        self.ids = [str(x) for x in ids]
        self.X2d = np.asarray(X2d, dtype=np.float32)

        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        self.w_cls = torch.tensor(w_cls, dtype=torch.float32)
        self.y_abs = torch.tensor(y_abs, dtype=torch.float32)
        self.m_abs = torch.tensor(m_abs, dtype=torch.bool)
        self.w_abs = torch.tensor(w_abs, dtype=torch.float32)
        self.y_fluo = torch.tensor(y_fluo, dtype=torch.float32)
        self.m_fluo = torch.tensor(m_fluo, dtype=torch.bool)
        self.w_fluo = torch.tensor(w_fluo, dtype=torch.float32)

        self.starts = starts
        self.counts = counts
        self.id2pos = id2pos
        self.Xinst = Xinst_sorted

        self.max_instances = int(max_instances)
        self.rng = np.random.default_rng(seed)

        if len(self.ids) != self.X2d.shape[0]:
            raise ValueError(f"MILTrainDataset: len(ids)={len(self.ids)} != X2d rows={self.X2d.shape[0]}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        mol_id = self.ids[i]
        p = self.id2pos[mol_id]
        s = int(self.starts[p])
        c = int(self.counts[p])
        bag = self.Xinst[s:s + c]

        if self.max_instances > 0 and bag.shape[0] > self.max_instances:
            idx = self.rng.choice(bag.shape[0], size=self.max_instances, replace=False)
            bag = bag[idx]

        return (
            self.X2d[i],                   # np float32 [F2]
            bag,                           # np float32 [Ni,F3]
            self.y_cls[i], self.w_cls[i],  # torch [4]
            self.y_abs[i], self.m_abs[i], self.w_abs[i],          # [2]
            self.y_fluo[i], self.m_fluo[i], self.w_fluo[i],       # [4]
        )


def collate_train(batch):
    B = len(batch)
    x2d_np = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
    bags_np = [b[1] for b in batch]
    lens = np.array([int(x.shape[0]) for x in bags_np], dtype=np.int64)
    max_len = int(lens.max())

    F3 = int(bags_np[0].shape[1])
    x3d_pad = np.zeros((B, max_len, F3), dtype=np.float32)
    for i, bag in enumerate(bags_np):
        x3d_pad[i, : bag.shape[0], :] = bag

    # key_padding_mask True=PAD
    kpm = np.ones((B, max_len), dtype=bool)
    for i, L in enumerate(lens):
        kpm[i, :L] = False

    x2d = torch.from_numpy(x2d_np).float()
    x3d = torch.from_numpy(x3d_pad).float()
    kpm_t = torch.from_numpy(kpm)

    y_cls = torch.stack([b[2] for b in batch], dim=0)
    w_cls = torch.stack([b[3] for b in batch], dim=0)

    y_abs = torch.stack([b[4] for b in batch], dim=0)
    m_abs = torch.stack([b[5] for b in batch], dim=0)
    w_abs = torch.stack([b[6] for b in batch], dim=0)

    y_fluo = torch.stack([b[7] for b in batch], dim=0)
    m_fluo = torch.stack([b[8] for b in batch], dim=0)
    w_fluo = torch.stack([b[9] for b in batch], dim=0)

    return x2d, x3d, kpm_t, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo


class MILExportDataset(Dataset):
    """Leaderboard export dataset: returns mol_id, conf_ids list, plus tensors inputs."""
    def __init__(
        self,
        ids: List[str],
        X2d: np.ndarray,
        starts: np.ndarray, counts: np.ndarray, id2pos: Dict[str, int],
        Xinst_sorted: np.ndarray,
        conf_sorted: np.ndarray,
        max_instances: int = 0,
        seed: int = 0,
    ):
        self.ids = [str(x) for x in ids]
        self.X2d = np.asarray(X2d, dtype=np.float32)
        self.starts = starts
        self.counts = counts
        self.id2pos = id2pos
        self.Xinst = Xinst_sorted
        self.conf_sorted = conf_sorted.astype(str)

        self.max_instances = int(max_instances)
        self.rng = np.random.default_rng(seed)

        if len(self.ids) != self.X2d.shape[0]:
            raise ValueError(f"MILExportDataset: len(ids)={len(self.ids)} != X2d rows={self.X2d.shape[0]}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        mol_id = self.ids[i]
        p = self.id2pos[mol_id]
        s = int(self.starts[p])
        c = int(self.counts[p])
        bag = self.Xinst[s:s + c]
        confs = self.conf_sorted[s:s + c]

        if self.max_instances > 0 and bag.shape[0] > self.max_instances:
            idx = self.rng.choice(bag.shape[0], size=self.max_instances, replace=False)
            bag = bag[idx]
            confs = confs[idx]

        return mol_id, confs.tolist(), self.X2d[i], bag


def collate_export(batch):
    # returns mol_ids(list[str]), conf_pad(object array), x2d, x3d, kpm
    B = len(batch)
    mol_ids = [b[0] for b in batch]
    conf_lists = [b[1] for b in batch]
    x2d_np = np.stack([b[2] for b in batch], axis=0).astype(np.float32)
    bags_np = [b[3] for b in batch]

    lens = np.array([len(x) for x in conf_lists], dtype=np.int64)
    max_len = int(lens.max())
    F3 = int(bags_np[0].shape[1])

    x3d_pad = np.zeros((B, max_len, F3), dtype=np.float32)
    conf_pad = np.empty((B, max_len), dtype=object)
    conf_pad[:] = ""

    for i in range(B):
        L = int(lens[i])
        x3d_pad[i, :L, :] = bags_np[i]
        conf_pad[i, :L] = conf_lists[i]

    kpm = np.ones((B, max_len), dtype=bool)
    for i, L in enumerate(lens):
        kpm[i, :int(L)] = False

    return (
        mol_ids,
        conf_pad,
        torch.from_numpy(x2d_np).float(),
        torch.from_numpy(x3d_pad).float(),
        torch.from_numpy(kpm),
    )


# =========================
# TRAINER
# =========================
def make_trainer_gpu(
    max_epochs: int,
    patience: int,
    accelerator: str,
    devices: int,
    precision: str,
    accumulate_grad_batches: int,
    ckpt_dir: str,
    trial: Optional[Trial] = None,
) -> Tuple[pl.Trainer, pl.callbacks.ModelCheckpoint]:
    es = pl.callbacks.EarlyStopping(monitor="val_macro_ap", mode="max", patience=int(patience))
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_macro_ap",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

    callbacks = [es, ckpt]
    if trial is not None:
        callbacks.append(OptunaPruningCallbackLocal(trial, monitor="val_macro_ap"))

    trainer = pl.Trainer(
        max_epochs=int(max_epochs),
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        deterministic=True,
        gradient_clip_val=0.0,
        accumulate_grad_batches=int(accumulate_grad_batches),
    )
    return trainer, ckpt


@torch.no_grad()
def eval_best_epoch(model: MILTaskAttnMixerWithAux, dl_va: DataLoader, device: torch.device) -> Tuple[float, List[float]]:
    model.eval()
    model.to(device)
    ps, ys, ws = [], [], []
    for batch in dl_va:
        x2d, x3d, kpm, y_cls, w_cls, *_ = batch
        x2d = x2d.to(device, non_blocking=True)
        x3d = x3d.to(device, non_blocking=True)
        kpm = kpm.to(device, non_blocking=True)
        logits, _, _ = model(x2d, x3d, kpm, return_attn=False)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        ps.append(p)
        ys.append(y_cls.detach().cpu().numpy())
        ws.append(w_cls.detach().cpu().numpy())
    P = np.concatenate(ps, axis=0)
    Y = np.concatenate(ys, axis=0).astype(int)
    W = np.concatenate(ws, axis=0).astype(np.float32)
    aps = ap_per_task(Y, P, w_cls=W, weighted_tasks=(0, 1))
    return float(np.mean(aps)), aps


@torch.no_grad()
def export_leaderboard_attention(
    model: MILTaskAttnMixerWithAux,
    dl_lb_export: DataLoader,
    device: torch.device,
    out_path: Path,
):
    """
    Write columns: ID, conf_id, task, attn_weight
    attn_weight sums to 1 over conformers for each (ID, task).
    """
    model.eval()
    model.to(device)

    rows: List[Dict[str, Any]] = []

    for mol_ids, conf_pad, x2d, x3d, kpm in dl_lb_export:
        x2d = x2d.to(device, non_blocking=True)
        x3d = x3d.to(device, non_blocking=True)
        kpm = kpm.to(device, non_blocking=True)

        logits, abs_out, fluo_out, attn = model(x2d, x3d, kpm, return_attn=True)
        if attn is None:
            raise RuntimeError("Attention not returned; expected attn when return_attn=True")

        attn_np = attn.detach().cpu().numpy()          # [B,4,N]
        kpm_np = kpm.detach().cpu().numpy().astype(bool)
        B, T, N = attn_np.shape

        for b in range(B):
            mid = str(mol_ids[b])
            valid = ~kpm_np[b]
            L = int(valid.sum())
            if L <= 0:
                continue
            confs = [str(x) for x in conf_pad[b, :L].tolist()]

            for t in range(T):
                w = attn_np[b, t, :L].astype(np.float64)
                s = float(w.sum())
                if not np.isfinite(s) or s <= 0:
                    w[:] = 1.0 / float(L)
                else:
                    w /= s  # enforce sum-to-1 (safety)
                task_name = TASK_COLS[t]
                for i in range(L):
                    rows.append({"ID": mid, "conf_id": confs[i], "task": task_name, "attn_weight": float(w[i])})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    if out_path.suffix.lower() in [".parquet", ".pq"]:
        df_out.to_parquet(out_path, index=False)
    else:
        df_out.to_csv(out_path, index=False)

    print(f"[ATTN] wrote {len(df_out)} rows -> {out_path}")


# =========================
# OPTUNA SEARCH SPACE
# =========================
def search_space(trial: Trial) -> Dict[str, Any]:
    p = {
        "mol_hidden": trial.suggest_categorical("mol_hidden", [256, 512, 1024]),
        "mol_layers": trial.suggest_int("mol_layers", 2, 5),
        "mol_dropout": trial.suggest_float("mol_dropout", 0.0, 0.2),

        "inst_hidden": trial.suggest_categorical("inst_hidden", [256, 512, 1024]),
        "inst_layers": trial.suggest_int("inst_layers", 1, 3),
        "inst_dropout": trial.suggest_float("inst_dropout", 0.0, 0.15),

        "proj_dim": trial.suggest_categorical("proj_dim", [256, 512]),

        "attn_heads": trial.suggest_categorical("attn_heads", [4, 8]),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.2),

        "mixer_hidden": trial.suggest_categorical("mixer_hidden", [256, 512, 1024]),
        "mixer_layers": trial.suggest_int("mixer_layers", 1, 3),
        "mixer_dropout": trial.suggest_float("mixer_dropout", 0.0, 0.3),

        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),

        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "pos_weight_clip": trial.suggest_float("pos_weight_clip", 30.0, 200.0, log=True),
        "focal_gamma_rare": trial.suggest_float("focal_gamma_rare", 0.0, 3.0),
        "rare_oversample_mult": trial.suggest_float("rare_oversample_mult", 0.0, 50.0),

        "lambda_power": trial.suggest_float("lambda_power", 0.0, 1.5),
        "lambda_aux_abs": trial.suggest_float("lambda_aux_abs", 0.05, 0.4),
        "lambda_aux_fluo": trial.suggest_float("lambda_aux_fluo", 0.05, 0.4),
        "reg_loss_type": trial.suggest_categorical("reg_loss_type", ["smoothl1", "mse"]),

        "accumulate_grad_batches": trial.suggest_categorical("accumulate_grad_batches", [4, 8]),
    }
    if int(p["inst_hidden"]) % int(p["attn_heads"]) != 0:
        raise optuna.TrialPruned("inst_hidden must be divisible by attn_heads")
    return p


# =========================
# OBJECTIVE (CV)
# =========================
def objective_mil_cv(
    trial: Trial,
    X2d_scaled: np.ndarray,
    y_cls: np.ndarray,
    w_cls: np.ndarray,
    y_abs: np.ndarray, m_abs: np.ndarray, w_abs: np.ndarray,
    y_fluo: np.ndarray, m_fluo: np.ndarray, w_fluo: np.ndarray,
    ids: List[str],
    folds_info,
    starts: np.ndarray, counts: np.ndarray, id2pos: Dict[str, int], Xinst_sorted: np.ndarray,
    seed: int,
    max_epochs: int,
    patience: int,
    accelerator: str,
    devices: int,
    num_workers: int,
    pin_memory: bool,
    precision: str,
    ckpt_root: Path,
) -> float:
    p = search_space(trial)
    max_instances = 0

    scores: List[float] = []
    fold_detail: Dict[str, Any] = {}

    device = torch.device("cuda" if torch.cuda.is_available() and accelerator in ("gpu", "cuda") else "cpu")

    for step, (tr, va, f) in enumerate(folds_info):
        set_all_seeds(seed + 5000 * f + trial.number)

        lam = lambda_from_prevalence(y_cls[tr], power=float(p["lambda_power"]))
        posw = pos_weight_per_task(y_cls[tr], clip=float(p["pos_weight_clip"]))

        prev = y_cls[tr].mean(axis=0)
        gamma = np.array([float(p["focal_gamma_rare"]) if prev[i] < 0.02 else 0.0 for i in range(4)], dtype=np.float32)
        gamma_t = torch.tensor(gamma, dtype=torch.float32)

        # aux standardization fit on TRAIN indices only
        mu_abs, sd_abs = fit_standardizer(y_abs, m_abs, tr)
        mu_f, sd_f = fit_standardizer(y_fluo, m_fluo, tr)
        y_abs_sc = apply_standardizer(y_abs, mu_abs, sd_abs)
        y_fluo_sc = apply_standardizer(y_fluo, mu_f, sd_f)

        ids_tr = [ids[i] for i in tr]
        ids_va = [ids[i] for i in va]

        ds_tr = MILTrainDataset(
            ids_tr, X2d_scaled[tr],
            y_cls[tr], w_cls[tr],
            y_abs_sc[tr], m_abs[tr], w_abs[tr],
            y_fluo_sc[tr], m_fluo[tr], w_fluo[tr],
            starts, counts, id2pos, Xinst_sorted,
            max_instances=max_instances, seed=seed + f,
        )
        ds_va = MILTrainDataset(
            ids_va, X2d_scaled[va],
            y_cls[va], w_cls[va],
            y_abs_sc[va], m_abs[va], w_abs[va],
            y_fluo_sc[va], m_fluo[va], w_fluo[va],
            starts, counts, id2pos, Xinst_sorted,
            max_instances=max_instances, seed=seed + 999 + f,
        )

        sampler = make_weighted_sampler(y_cls[tr], rare_mult=float(p["rare_oversample_mult"]))

        nw = int(num_workers)
        dl_kw = dict(
            num_workers=nw,
            pin_memory=pin_memory,
            persistent_workers=(nw > 0),
        )
        if nw > 0:
            dl_kw["prefetch_factor"] = 2

        dl_tr = DataLoader(ds_tr, batch_size=int(p["batch_size"]), sampler=sampler, collate_fn=collate_train, **dl_kw)
        dl_va = DataLoader(ds_va, batch_size=min(128, int(p["batch_size"])), shuffle=False, collate_fn=collate_train, **dl_kw)

        model = MILTaskAttnMixerWithAux(
            mol_dim=int(X2d_scaled.shape[1]),
            inst_dim=int(Xinst_sorted.shape[1]),
            mol_hidden=int(p["mol_hidden"]),
            mol_layers=int(p["mol_layers"]),
            mol_dropout=float(p["mol_dropout"]),
            inst_hidden=int(p["inst_hidden"]),
            inst_layers=int(p["inst_layers"]),
            inst_dropout=float(p["inst_dropout"]),
            proj_dim=int(p["proj_dim"]),
            attn_heads=int(p["attn_heads"]),
            attn_dropout=float(p["attn_dropout"]),
            mixer_hidden=int(p["mixer_hidden"]),
            mixer_layers=int(p["mixer_layers"]),
            mixer_dropout=float(p["mixer_dropout"]),
            lr=float(p["lr"]),
            weight_decay=float(p["weight_decay"]),
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
            lambda_aux_abs=float(p["lambda_aux_abs"]),
            lambda_aux_fluo=float(p["lambda_aux_fluo"]),
            reg_loss_type=str(p["reg_loss_type"]),
        )

        fold_ckpt_dir = ckpt_root / f"mil_trial{trial.number}_fold{f}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer, ckpt_cb = make_trainer_gpu(
            max_epochs=max_epochs,
            patience=patience,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            accumulate_grad_batches=int(p["accumulate_grad_batches"]),
            ckpt_dir=str(fold_ckpt_dir),
            trial=trial,
        )
        trainer.fit(model, dl_tr, dl_va)

        epochs_trained = int(trainer.current_epoch) + 1

        best_path = ckpt_cb.best_model_path
        best_epoch = None
        if best_path and Path(best_path).exists():
            ckpt = torch.load(best_path, map_location="cpu")
            best_epoch = int(ckpt.get("epoch", -1))
            model.load_state_dict(ckpt["state_dict"], strict=True)

        best_macro, best_aps = eval_best_epoch(model, dl_va, device=device)

        print(
            f"[MIL-TASK-ATTN] trial={trial.number} fold={f} trained_epochs={epochs_trained} best_epoch={best_epoch} "
            f"best_macro_ap={best_macro:.6f} aps={best_aps}"
        )

        scores.append(float(best_macro))
        fold_detail[str(f)] = {
            "trained_epochs": epochs_trained,
            "best_epoch": best_epoch,
            "macro_ap_best_epoch": float(best_macro),
            "ap_task0": float(best_aps[0]),
            "ap_task1": float(best_aps[1]),
            "ap_task2": float(best_aps[2]),
            "ap_task3": float(best_aps[3]),
            "accumulate_grad_batches": int(p["accumulate_grad_batches"]),
        }

        try:
            shutil.rmtree(fold_ckpt_dir, ignore_errors=True)
        except Exception:
            pass

        del trainer, model, dl_tr, dl_va, ds_tr, ds_va, sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        trial.report(float(np.mean(scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("fold_detail", fold_detail)
    return float(np.mean(scores))


def save_study_artifacts(outdir: Path, study: optuna.Study, prefix: str):
    df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df_trials.to_csv(outdir / f"{prefix}_trials.csv", index=False)
    best = dict(study.best_params)
    best["best_value_macro_ap_cv"] = float(study.best_value)
    (outdir / f"{prefix}_best_params.json").write_text(json.dumps(best, indent=2))


def save_best_fold_metrics(outdir: Path, prefix: str, fold_metrics: Dict[str, Any]):
    (outdir / f"{prefix}_best_fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))


# =========================
# FINAL TRAIN ON TRAIN, VALIDATE ON LEADERBOARD, EXPORT ATTN
# =========================
def drop_ids_without_bags(
    ids: List[str],
    X2d: np.ndarray,
    df_part: pd.DataFrame,
    id2pos: Dict[str, int],
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    mask = np.array([(i in id2pos) for i in ids], dtype=bool)
    if mask.all():
        return ids, X2d, df_part
    df2 = df_part.loc[mask].reset_index(drop=True)
    ids2 = df2["ID"].astype(str).tolist()
    return ids2, X2d[mask], df2


def train_best_and_export(
    outdir: Path,
    df_full: pd.DataFrame,
    best_params: Dict[str, Any],
    args,
    X2d_file_ids: List[str],
    X2d_file: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    id2pos: Dict[str, int],
    Xinst_sorted: np.ndarray,
    conf_sorted: np.ndarray,
    num_workers: int,
    pin_memory: bool,
):
    df_tr = df_full[df_full[args.split_col] == "train"].copy().reset_index(drop=True)
    df_lb = df_full[df_full[args.split_col] == args.leaderboard_split].copy().reset_index(drop=True)
    if len(df_lb) == 0:
        raise ValueError(f"No rows with split == '{args.leaderboard_split}'")

    ids_tr = df_tr[args.id_col].astype(str).tolist()
    ids_lb = df_lb[args.id_col].astype(str).tolist()

    X2d_tr = align_by_id(X2d_file_ids, X2d_file, ids_tr)
    X2d_lb = align_by_id(X2d_file_ids, X2d_file, ids_lb)

    # drop IDs without bags
    ids_tr, X2d_tr, df_tr = drop_ids_without_bags(ids_tr, X2d_tr, df_tr.rename(columns={args.id_col: "ID"}), id2pos)
    ids_lb, X2d_lb, df_lb = drop_ids_without_bags(ids_lb, X2d_lb, df_lb.rename(columns={args.id_col: "ID"}), id2pos)

    # targets/weights
    y_tr = coerce_binary_labels(df_tr)
    w_tr = build_task_weights(df_tr)
    y_abs_tr, m_abs_tr, y_fluo_tr, m_fluo_tr = build_aux_targets_and_masks(df_tr)
    w_abs_tr, w_fluo_tr = build_aux_weights(df_tr)

    y_lb = coerce_binary_labels(df_lb)
    w_lb = build_task_weights(df_lb)
    y_abs_lb, m_abs_lb, y_fluo_lb, m_fluo_lb = build_aux_targets_and_masks(df_lb)
    w_abs_lb, w_fluo_lb = build_aux_weights(df_lb)

    # aux standardization fit on train
    tr_idx = np.arange(len(df_tr), dtype=np.int64)
    mu_abs, sd_abs = fit_standardizer(y_abs_tr, m_abs_tr, tr_idx)
    mu_f, sd_f = fit_standardizer(y_fluo_tr, m_fluo_tr, tr_idx)
    y_abs_tr_sc = apply_standardizer(y_abs_tr, mu_abs, sd_abs)
    y_abs_lb_sc = apply_standardizer(y_abs_lb, mu_abs, sd_abs)
    y_fluo_tr_sc = apply_standardizer(y_fluo_tr, mu_f, sd_f)
    y_fluo_lb_sc = apply_standardizer(y_fluo_lb, mu_f, sd_f)

    # loss weights
    lam = lambda_from_prevalence(y_tr, power=float(best_params["lambda_power"]))
    posw = pos_weight_per_task(y_tr, clip=float(best_params["pos_weight_clip"]))
    prev = y_tr.mean(axis=0)
    gamma = np.array([float(best_params["focal_gamma_rare"]) if prev[i] < 0.02 else 0.0 for i in range(4)], dtype=np.float32)
    gamma_t = torch.tensor(gamma, dtype=torch.float32)

    # datasets + loaders
    ds_tr = MILTrainDataset(
        ids_tr, X2d_tr,
        y_tr, w_tr,
        y_abs_tr_sc, m_abs_tr, w_abs_tr,
        y_fluo_tr_sc, m_fluo_tr, w_fluo_tr,
        starts, counts, id2pos, Xinst_sorted,
        max_instances=0, seed=int(args.seed),
    )
    ds_lb = MILTrainDataset(
        ids_lb, X2d_lb,
        y_lb, w_lb,
        y_abs_lb_sc, m_abs_lb, w_abs_lb,
        y_fluo_lb_sc, m_fluo_lb, w_fluo_lb,
        starts, counts, id2pos, Xinst_sorted,
        max_instances=0, seed=int(args.seed) + 999,
    )

    sampler_tr = make_weighted_sampler(y_tr, rare_mult=float(best_params["rare_oversample_mult"]))

    nw = int(num_workers)
    dl_kw = dict(
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=(nw > 0),
    )
    if nw > 0:
        dl_kw["prefetch_factor"] = 2

    dl_tr = DataLoader(ds_tr, batch_size=int(best_params["batch_size"]), sampler=sampler_tr, collate_fn=collate_train, **dl_kw)
    dl_val = DataLoader(ds_lb, batch_size=min(128, int(best_params["batch_size"])), shuffle=False, collate_fn=collate_train, **dl_kw)

    model = MILTaskAttnMixerWithAux(
        mol_dim=int(X2d_tr.shape[1]),
        inst_dim=int(Xinst_sorted.shape[1]),
        mol_hidden=int(best_params["mol_hidden"]),
        mol_layers=int(best_params["mol_layers"]),
        mol_dropout=float(best_params["mol_dropout"]),
        inst_hidden=int(best_params["inst_hidden"]),
        inst_layers=int(best_params["inst_layers"]),
        inst_dropout=float(best_params["inst_dropout"]),
        proj_dim=int(best_params["proj_dim"]),
        attn_heads=int(best_params["attn_heads"]),
        attn_dropout=float(best_params["attn_dropout"]),
        mixer_hidden=int(best_params["mixer_hidden"]),
        mixer_layers=int(best_params["mixer_layers"]),
        mixer_dropout=float(best_params["mixer_dropout"]),
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        pos_weight=posw,
        gamma=gamma_t,
        lam=lam,
        lambda_aux_abs=float(best_params["lambda_aux_abs"]),
        lambda_aux_fluo=float(best_params["lambda_aux_fluo"]),
        reg_loss_type=str(best_params["reg_loss_type"]),
    )

    final_dir = outdir / "final_best_train_vs_leaderboard"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer, ckpt_cb = make_trainer_gpu(
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        accelerator=str(args.nn_accelerator),
        devices=int(args.nn_devices),
        precision=str(args.precision),
        accumulate_grad_batches=int(best_params["accumulate_grad_batches"]),
        ckpt_dir=str(final_dir),
        trial=None,
    )
    trainer.fit(model, dl_tr, dl_val)

    best_path = ckpt_cb.best_model_path
    if best_path and Path(best_path).exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[FINAL] loaded best ckpt: {best_path}")

    # Evaluate best checkpoint on leaderboard (validation) split
    dev_eval = torch.device("cuda" if torch.cuda.is_available() and str(args.nn_accelerator) in ("gpu", "cuda") else "cpu")
    macro_ap_lb, aps_lb = eval_best_epoch(model, dl_val, device=dev_eval)
    eval_json = {
        "macro_ap": float(macro_ap_lb),
        "ap_task0": float(aps_lb[0]),
        "ap_task1": float(aps_lb[1]),
        "ap_task2": float(aps_lb[2]),
        "ap_task3": float(aps_lb[3]),
    }
    (final_dir / "leaderboard_eval.json").write_text(json.dumps(eval_json, indent=2))
    print(f"[FINAL] leaderboard eval: macro_ap={macro_ap_lb:.6f} aps={aps_lb}")

    # export attention weights on leaderboard
    export_ds = MILExportDataset(
        ids_lb,
        X2d_lb,
        starts=starts, counts=counts, id2pos=id2pos,
        Xinst_sorted=Xinst_sorted,
        conf_sorted=conf_sorted,
        max_instances=0,
        seed=int(args.seed) + 123,
    )
    export_dl = DataLoader(export_ds, batch_size=min(64, int(best_params["batch_size"])), shuffle=False, collate_fn=collate_export, **dl_kw)

    dev = torch.device("cuda" if torch.cuda.is_available() and str(args.nn_accelerator) in ("gpu", "cuda") else "cpu")
    out_path = Path(args.attn_out) if args.attn_out else (outdir / "leaderboard_attn.parquet")
    export_leaderboard_attention(model, export_dl, device=dev, out_path=out_path)


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--labels", required=True)
    ap.add_argument("--feat2d_scaled", required=True)
    ap.add_argument("--feat3d_scaled", required=True)
    ap.add_argument("--feat3d_qm_scaled", required=True)

    ap.add_argument("--study_dir", required=True)

    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--conf_col", default="conf_id")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--fold_col", default="cv_fold")

    ap.add_argument("--use_splits", nargs="+", default=["train"])  # for HPO subset
    ap.add_argument("--folds", nargs="+", type=int, default=None)

    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--trials", type=int, default=50)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--nn_accelerator", default="gpu")
    ap.add_argument("--nn_devices", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument("--pin_memory", action="store_true")

    # final export
    ap.add_argument("--export_leaderboard_attn", action="store_true")
    ap.add_argument("--leaderboard_split", default="leaderboard")
    ap.add_argument("--attn_out", default=None)

    args = ap.parse_args()

    set_all_seeds(int(args.seed))
    maybe_set_torch_fast_flags()

    outdir = Path(args.study_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "nn_accelerator": str(args.nn_accelerator),
        "nn_devices": int(args.nn_devices),
        "precision": str(args.precision),
        "patience": int(args.patience),
        "argv": " ".join([str(x) for x in os.sys.argv]),
        "weight_cols": WEIGHT_COLS,
        "model": "MILTaskAttnMixerWithAux (task-specific attention queries)",
    }
    (outdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    # num_workers auto
    if int(args.num_workers) < 0:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
        if cpus <= 0:
            cpus = os.cpu_count() or 0
        num_workers = max(2, min(23, cpus - 2)) if cpus >= 4 else 0
    else:
        num_workers = int(args.num_workers)

    pin_memory = bool(args.pin_memory) and torch.cuda.is_available()

    # Load full labels (needed for final export)
    df_full = load_labels(args.labels, id_col=args.id_col)
    df_full[args.id_col] = df_full[args.id_col].astype(str)
    df_full[args.split_col] = df_full[args.split_col].astype(str)

    # HPO subset
    df_hpo = df_full[df_full[args.split_col].isin(args.use_splits)].copy().reset_index(drop=True)
    if len(df_hpo) == 0:
        raise ValueError(f"No rows in labels match use_splits={args.use_splits}")

    ids_hpo = df_hpo[args.id_col].astype(str).tolist()

    # 2D features
    ids_2d_file, X2d_file = load_2d(args.feat2d_scaled, id_col=args.id_col)
    X2d_hpo = align_by_id(ids_2d_file, X2d_file, ids_hpo)

    y_cls = coerce_binary_labels(df_hpo)
    w_cls = build_task_weights(df_hpo)
    y_abs, m_abs, y_fluo, m_fluo = build_aux_targets_and_masks(df_hpo)
    w_abs, w_fluo = build_aux_weights(df_hpo)

    # folds
    if args.folds is None:
        folds = sorted(df_hpo[args.fold_col].dropna().astype(int).unique().tolist())
    else:
        folds = list(map(int, args.folds))
    folds_info = fold_indices(df_hpo, args.fold_col, folds)

    # instances (for HPO IDs)
    allowed_hpo = set(ids_hpo)
    ids_conf_hpo, conf_ids_hpo, Xinst_hpo = load_and_merge_instances(
        args.feat3d_scaled,
        args.feat3d_qm_scaled,
        allowed_ids=allowed_hpo,
        id_col=args.id_col,
        conf_col=args.conf_col,
    )
    _, starts_hpo, counts_hpo, id2pos_hpo, Xinst_sorted_hpo, conf_sorted_hpo = build_instance_index(ids_conf_hpo, conf_ids_hpo, Xinst_hpo)

    # drop molecules that ended up with 0 conformers
    have_bag_mask = np.array([(i in id2pos_hpo) for i in ids_hpo], dtype=bool)
    if not have_bag_mask.all():
        missing = int((~have_bag_mask).sum())
        ex = [ids_hpo[i] for i in np.where(~have_bag_mask)[0][:10]]
        print(f"[WARN] Dropping {missing} HPO IDs with 0 conformers after merge. Examples: {ex}")

        df_hpo = df_hpo.loc[have_bag_mask].reset_index(drop=True)
        ids_hpo = df_hpo[args.id_col].astype(str).tolist()

        X2d_hpo = X2d_hpo[have_bag_mask]
        y_cls = y_cls[have_bag_mask]
        w_cls = w_cls[have_bag_mask]
        y_abs = y_abs[have_bag_mask]
        m_abs = m_abs[have_bag_mask]
        y_fluo = y_fluo[have_bag_mask]
        m_fluo = m_fluo[have_bag_mask]
        w_abs = w_abs[have_bag_mask]
        w_fluo = w_fluo[have_bag_mask]

        folds = sorted(df_hpo[args.fold_col].dropna().astype(int).unique().tolist())
        folds_info = fold_indices(df_hpo, args.fold_col, folds)

    print(f"[DATA-HPO] n_ids={len(ids_hpo)} | X2d_dim={X2d_hpo.shape[1]}")
    print(f"[DATA-HPO] n_conf={Xinst_sorted_hpo.shape[0]} | inst_dim={Xinst_sorted_hpo.shape[1]}")
    print(f"[DATALOADER] num_workers={num_workers} pin_memory={pin_memory} precision={args.precision}")

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)

    def make_storage(name: str) -> str:
        return f"sqlite:///{(outdir / f'{name}.sqlite3').as_posix()}"

    ckpt_root = outdir / "_tmp_best_ckpts"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    study_name = "mil_task_attn_mixer_aux_gpu"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=make_storage(study_name),
        load_if_exists=True,
    )
    study.optimize(
        lambda tr: objective_mil_cv(
            tr,
            X2d_scaled=X2d_hpo,
            y_cls=y_cls, w_cls=w_cls,
            y_abs=y_abs, m_abs=m_abs, w_abs=w_abs,
            y_fluo=y_fluo, m_fluo=m_fluo, w_fluo=w_fluo,
            ids=ids_hpo,
            folds_info=folds_info,
            starts=starts_hpo, counts=counts_hpo, id2pos=id2pos_hpo, Xinst_sorted=Xinst_sorted_hpo,
            seed=int(args.seed),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            accelerator=str(args.nn_accelerator),
            devices=int(args.nn_devices),
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            precision=str(args.precision),
            ckpt_root=ckpt_root,
        ),
        n_trials=int(args.trials),
        gc_after_trial=True,
        catch=(RuntimeError, ValueError, FloatingPointError),
    )

    save_study_artifacts(outdir, study, prefix=study_name)
    save_best_fold_metrics(outdir, study_name, study.best_trial.user_attrs.get("fold_detail", {}))
    print(f"[HPO] best macro AP (CV mean) = {study.best_value:.6f}")

    # final export step needs instance index for union(train+leaderboard) not just HPO IDs
    if args.export_leaderboard_attn:
        allowed_final = set(df_full[df_full[args.split_col].isin(["train", args.leaderboard_split])][args.id_col].astype(str).tolist())
        ids_conf_all, conf_ids_all, Xinst_all = load_and_merge_instances(
            args.feat3d_scaled,
            args.feat3d_qm_scaled,
            allowed_ids=allowed_final,
            id_col=args.id_col,
            conf_col=args.conf_col,
        )
        _, starts_all, counts_all, id2pos_all, Xinst_sorted_all, conf_sorted_all = build_instance_index(ids_conf_all, conf_ids_all, Xinst_all)

        train_best_and_export(
            outdir=outdir,
            df_full=df_full.rename(columns={args.id_col: "ID"}),
            best_params=dict(study.best_params),
            args=args,
            X2d_file_ids=ids_2d_file,
            X2d_file=X2d_file,
            starts=starts_all,
            counts=counts_all,
            id2pos=id2pos_all,
            Xinst_sorted=Xinst_sorted_all,
            conf_sorted=conf_sorted_all,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # cleanup
    try:
        shutil.rmtree(ckpt_root, ignore_errors=True)
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
