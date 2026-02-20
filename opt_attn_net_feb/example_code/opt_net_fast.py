#!/usr/bin/env python3
from __future__ import annotations

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
    from lightning.pytorch.callbacks import Callback  # fallback if you switch namespaces


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

# Classification weights per task (used in BOTH MLP and MIL classification loss)
WEIGHT_COLS = {
    0: "sample_weight_340",
    1: "sample_weight_450",
    2: "w_ad",
    3: "w_ad",
}

NONFEAT_2D = {"ID", "curated_SMILES", "split"}
NONFEAT_3D = {"ID", "conf_id", "smiles", "split"}
NONFEAT_QM = {"record_index", "ID", "conf_id", "status", "error", "split"}


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

        if torch.is_tensor(current):
            current_val = float(current.detach().cpu().item())
        else:
            current_val = float(current)

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


def coerce_binary_labels(df: pd.DataFrame) -> np.ndarray:
    y = df[TASK_COLS].fillna(0).astype(int).to_numpy()
    return (y > 0).astype(np.int64)


def build_task_weights(df_lab: pd.DataFrame) -> np.ndarray:
    """
    Per-sample per-task weights used in BOTH MLP and MIL classification loss.
      - task0: sample_weight_340
      - task1: sample_weight_450
      - task2/3: w_ad (as in your current pipeline)
    """
    W = np.ones((len(df_lab), 4), dtype=np.float32)
    for t in range(4):
        col = WEIGHT_COLS[t]
        if col in df_lab.columns:
            w = df_lab[col].astype(float).fillna(1.0).to_numpy()
            W[:, t] = w.astype(np.float32)
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

    # keep your original behavior: duplicate (wl,qy) for both fluo tasks => (n,4)
    y_fluo4 = np.concatenate([y_fbase, y_fbase], axis=1).astype(np.float32)
    m_fluo4 = np.concatenate([m_fbase, m_fbase], axis=1)
    return y_abs, m_abs, y_fluo4, m_fluo4


def build_aux_weights(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    w340 = (
        df["sample_weight_340"].astype(float).fillna(1.0).to_numpy(dtype=np.float32)
        if "sample_weight_340" in df.columns else np.ones(len(df), dtype=np.float32)
    )
    w450 = (
        df["sample_weight_450"].astype(float).fillna(1.0).to_numpy(dtype=np.float32)
        if "sample_weight_450" in df.columns else np.ones(len(df), dtype=np.float32)
    )
    wad = (
        df["w_ad"].astype(float).fillna(1.0).to_numpy(dtype=np.float32)
        if "w_ad" in df.columns else np.ones(len(df), dtype=np.float32)
    )

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
    """
    Keep your original oversampling heuristic.
    """
    y = y.astype(int)
    rare_pos = ((y[:, 1] == 1) | (y[:, 3] == 1)).astype(np.float64)
    w = 1.0 + rare_mult * rare_pos
    w = torch.tensor(w, dtype=torch.double)
    return WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)


def infer_feature_cols(df: pd.DataFrame, nonfeat: set) -> List[str]:
    return sorted([c for c in df.columns if c not in nonfeat])


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
# METRICS (Weighted PR-AUC for tasks 0/1)
# =========================
def ap_per_task(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    w_cls: Optional[np.ndarray] = None,
    weighted_tasks: Tuple[int, ...] = (0, 1),
) -> List[float]:
    """
    Per-task Average Precision (PR-AUC).
    If w_cls is given: tasks in weighted_tasks use sample_weight=w_cls[:,t].
    """
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


def macro_ap_weighted(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    w_cls: Optional[np.ndarray] = None,
) -> float:
    aps = ap_per_task(y_true, p_pred, w_cls=w_cls, weighted_tasks=(0, 1))
    m = float(np.mean(aps))
    return 0.0 if not np.isfinite(m) else m


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
    feat_cols = infer_feature_cols(df, NONFEAT_2D)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    ids = df[id_col].astype(str).tolist()
    return ids, X


def align_by_id(ids_file: List[str], X: np.ndarray, ids_target: List[str]) -> np.ndarray:
    id2row = {str(i): r for r, i in enumerate(ids_file)}
    miss = [i for i in ids_target if str(i) not in id2row]
    if miss:
        raise ValueError(f"[2D SCALED] Missing {len(miss)} IDs (first 10): {miss[:10]}")
    rows = np.array([id2row[str(i)] for i in ids_target], dtype=np.int64)
    return X[rows]


def load_and_merge_instances(
    geom_csv: str,
    qm_csv: str,
    allowed_ids: Optional[Set[str]],
    id_col="ID",
    conf_col="conf_id",
) -> Tuple[np.ndarray, np.ndarray]:
    dg = pd.read_csv(geom_csv)
    dq = pd.read_csv(qm_csv)

    for d, name in [(dg, "3d_scaled"), (dq, "3d_quantum_scaled")]:
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

    g_cols = infer_feature_cols(dg, NONFEAT_3D)
    q_cols = infer_feature_cols(dq, NONFEAT_QM)

    m = dg[[id_col, conf_col] + g_cols].merge(
        dq[[id_col, conf_col] + q_cols],
        on=[id_col, conf_col],
        how="inner",
        validate="one_to_one",
    )

    ids_conf = m[id_col].astype(str).to_numpy()
    Xg = m[g_cols].to_numpy(dtype=np.float32)
    Xq = m[q_cols].to_numpy(dtype=np.float32)
    X309 = np.hstack([Xg, Xq]).astype(np.float32)
    return ids_conf, X309


def build_instance_matrix_1042(
    ids_conf: np.ndarray,
    X309: np.ndarray,
    ids_2d_aligned: List[str],
    X2d_scaled_aligned: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
    id2row_2d = {str(i): r for r, i in enumerate(ids_2d_aligned)}
    rows = np.array([id2row_2d.get(str(i), -1) for i in ids_conf], dtype=np.int64)
    keep = rows >= 0
    if not np.all(keep):
        ids_conf = ids_conf[keep]
        X309 = X309[keep]
        rows = rows[keep]

    X2d_rep = X2d_scaled_aligned[rows]
    X1042 = np.hstack([X2d_rep, X309]).astype(np.float32)

    order = np.argsort(ids_conf, kind="mergesort")
    ids_sorted = ids_conf[order]
    X_sorted = X1042[order]

    uniq, starts, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    id2pos = {str(u): i for i, u in enumerate(uniq)}
    return uniq, starts.astype(np.int64), counts.astype(np.int64), id2pos, X_sorted


# =========================
# LOSSES / MODELS
# =========================
class MultiTaskFocal(nn.Module):
    """
    Uses w as a per-sample per-task weight, and returns a weighted mean per task:
        sum_i w_i * loss_i / sum_i w_i
    So 0.5 downweights contribution exactly as desired.
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
    mask_f = mask.float()
    w_eff = torch.clamp(w, min=0.0) * mask_f

    if loss_type == "smoothl1":
        per = F.smooth_l1_loss(pred, target, reduction="none")
    elif loss_type == "mse":
        per = (pred - target).pow(2)
    else:
        raise ValueError(f"Unknown reg loss type: {loss_type}")

    per = per * mask_f
    num = (per * w_eff).sum(dim=0)
    den = w_eff.sum(dim=0).clamp(min=1e-6)
    return (num / den).mean()


def make_mlp(in_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Module:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [
            nn.Linear(d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        d = hidden_dim
    return nn.Sequential(*layers)


class MultiTaskMLPWithAux(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
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

        self.trunk = make_mlp(in_dim, hidden_dim, n_layers, dropout)

        self.cls_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(4)])
        self.abs_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(2)])
        self.fluo_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(4)])

        self.cls_loss = MultiTaskFocal(pos_weight=pos_weight, gamma=gamma)
        self.register_buffer("lam", torch.tensor(lam, dtype=torch.float32))

        self.lambda_aux_abs = float(lambda_aux_abs)
        self.lambda_aux_fluo = float(lambda_aux_fluo)
        self.reg_loss_type = str(reg_loss_type)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        # accumulate val preds/targets to compute AP once per epoch (fast + correct)
        self._val_p: List[np.ndarray] = []
        self._val_y: List[np.ndarray] = []
        self._val_w: List[np.ndarray] = []

    def forward(self, x):
        h = self.trunk(x)
        logits = torch.cat([hd(h) for hd in self.cls_heads], dim=1)
        abs_out = torch.cat([hd(h) for hd in self.abs_heads], dim=1)
        fluo_out = torch.cat([hd(h) for hd in self.fluo_heads], dim=1)
        return logits, abs_out, fluo_out

    def training_step(self, batch, batch_idx):
        x, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo = batch
        logits, abs_out, fluo_out = self(x)

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
        x, y_cls, w_cls, *_ = batch
        logits, _, _ = self(x)

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


class MILMeanPoolPackedWithAux(pl.LightningModule):
    def __init__(
        self,
        inst_dim: int,
        inst_hidden: int,
        inst_layers: int,
        inst_dropout: float,
        trunk_hidden: int,
        trunk_layers: int,
        trunk_dropout: float,
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

        self.inst_embed = make_mlp(inst_dim, inst_hidden, inst_layers, inst_dropout)
        self.inst_proj = nn.Linear(inst_hidden, inst_hidden)
        self.trunk = make_mlp(inst_hidden, trunk_hidden, trunk_layers, trunk_dropout)

        self.cls_heads = nn.ModuleList([nn.Linear(trunk_hidden, 1) for _ in range(4)])
        self.abs_heads = nn.ModuleList([nn.Linear(trunk_hidden, 1) for _ in range(2)])
        self.fluo_heads = nn.ModuleList([nn.Linear(trunk_hidden, 1) for _ in range(4)])

        self.cls_loss = MultiTaskFocal(pos_weight=pos_weight, gamma=gamma)
        self.register_buffer("lam", torch.tensor(lam, dtype=torch.float32))

        self.lambda_aux_abs = float(lambda_aux_abs)
        self.lambda_aux_fluo = float(lambda_aux_fluo)
        self.reg_loss_type = str(reg_loss_type)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        # accumulate val preds/targets to compute AP once per epoch (fast + correct)
        self._val_p: List[np.ndarray] = []
        self._val_y: List[np.ndarray] = []
        self._val_w: List[np.ndarray] = []

    def forward(self, x_all: torch.Tensor, bag_idx: torch.Tensor, B: int):
        h = self.inst_proj(self.inst_embed(x_all))
        Hdim = h.shape[1]

        pooled_sum = torch.zeros((B, Hdim), device=h.device, dtype=h.dtype)
        pooled_sum.index_add_(0, bag_idx, h)

        ones = torch.ones((h.shape[0], 1), device=h.device, dtype=h.dtype)
        cnt = torch.zeros((B, 1), device=h.device, dtype=h.dtype)
        cnt.index_add_(0, bag_idx, ones)

        pooled = pooled_sum / cnt.clamp(min=1.0)
        z = self.trunk(pooled)

        logits = torch.cat([hd(z) for hd in self.cls_heads], dim=1)
        abs_out = torch.cat([hd(z) for hd in self.abs_heads], dim=1)
        fluo_out = torch.cat([hd(z) for hd in self.fluo_heads], dim=1)
        return logits, abs_out, fluo_out

    def training_step(self, batch, batch_idx):
        x_all, bag_idx, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo = batch
        B = int(y_cls.shape[0])
        logits, abs_out, fluo_out = self(x_all, bag_idx, B)

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
        x_all, bag_idx, y_cls, w_cls, *_ = batch
        B = int(y_cls.shape[0])
        logits, _, _ = self(x_all, bag_idx, B)

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
# DATASETS
# =========================
class TabularAuxDataset(Dataset):
    def __init__(self, X, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        self.w_cls = torch.tensor(w_cls, dtype=torch.float32)
        self.y_abs = torch.tensor(y_abs, dtype=torch.float32)
        self.m_abs = torch.tensor(m_abs, dtype=torch.bool)
        self.w_abs = torch.tensor(w_abs, dtype=torch.float32)
        self.y_fluo = torch.tensor(y_fluo, dtype=torch.float32)
        self.m_fluo = torch.tensor(m_fluo, dtype=torch.bool)
        self.w_fluo = torch.tensor(w_fluo, dtype=torch.float32)

    def __len__(self):  # noqa: D401
        return self.X.shape[0]

    def __getitem__(self, i):
        return (
            self.X[i],
            self.y_cls[i], self.w_cls[i],
            self.y_abs[i], self.m_abs[i], self.w_abs[i],
            self.y_fluo[i], self.m_fluo[i], self.w_fluo[i],
        )


class MILPackedDataset(Dataset):
    def __init__(
        self,
        ids,
        y_cls, w_cls,
        y_abs, m_abs, w_abs,
        y_fluo, m_fluo, w_fluo,
        starts, counts, id2pos, Xinst_sorted,
        max_instances, seed,
    ):
        self.ids = [str(x) for x in ids]
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
            bag,
            self.y_cls[i], self.w_cls[i],
            self.y_abs[i], self.m_abs[i], self.w_abs[i],
            self.y_fluo[i], self.m_fluo[i], self.w_fluo[i],
        )


def mil_collate_packed(batch):
    B = len(batch)
    bags_np = [b[0] for b in batch]
    lens = [int(x.shape[0]) for x in bags_np]

    x_all = torch.from_numpy(np.concatenate(bags_np, axis=0)).float()
    bag_idx = torch.repeat_interleave(
        torch.arange(B, dtype=torch.long),
        torch.tensor(lens, dtype=torch.long),
    )

    y_cls = torch.stack([b[1] for b in batch], dim=0)
    w_cls = torch.stack([b[2] for b in batch], dim=0)

    y_abs = torch.stack([b[3] for b in batch], dim=0)
    m_abs = torch.stack([b[4] for b in batch], dim=0)
    w_abs = torch.stack([b[5] for b in batch], dim=0)

    y_fluo = torch.stack([b[6] for b in batch], dim=0)
    m_fluo = torch.stack([b[7] for b in batch], dim=0)
    w_fluo = torch.stack([b[8] for b in batch], dim=0)

    return x_all, bag_idx, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo


# =========================
# OPTUNA SEARCH SPACE
# =========================
def search_space_nn_common(trial: Trial) -> Dict[str, Any]:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 1024]),
        "n_layers": trial.suggest_int("n_layers", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
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


# =========================
# TRAINER + BEST-CKPT EVAL
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
def eval_mlp_best_epoch(model: MultiTaskMLPWithAux, dl_va: DataLoader, device: torch.device) -> Tuple[float, List[float]]:
    model.eval()
    model.to(device)
    ps, ys, ws = [], [], []
    for batch in dl_va:
        x, y_cls, w_cls, *_ = batch
        x = x.to(device, non_blocking=True)
        logits, _, _ = model(x)
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
def eval_mil_best_epoch(model: MILMeanPoolPackedWithAux, dl_va: DataLoader, device: torch.device) -> Tuple[float, List[float]]:
    model.eval()
    model.to(device)
    ps, ys, ws = [], [], []
    for batch in dl_va:
        x_all, bag_idx, y_cls, w_cls, *_ = batch
        x_all = x_all.to(device, non_blocking=True)
        bag_idx = bag_idx.to(device, non_blocking=True)
        B = int(y_cls.shape[0])
        logits, _, _ = model(x_all, bag_idx, B)
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


def save_study_artifacts(outdir: Path, study: optuna.Study, prefix: str):
    df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df_trials.to_csv(outdir / f"{prefix}_trials.csv", index=False)
    best = dict(study.best_params)
    best["best_value_macro_ap_cv"] = float(study.best_value)
    (outdir / f"{prefix}_best_params.json").write_text(json.dumps(best, indent=2))


def save_best_fold_metrics(outdir: Path, prefix: str, fold_metrics: Dict[str, Any]):
    (outdir / f"{prefix}_best_fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))


# =========================
# OBJECTIVES
# =========================
def objective_mlp_cv(
    trial: Trial,
    X2d_scaled: np.ndarray,
    y_cls: np.ndarray,
    w_cls: np.ndarray,
    y_abs: np.ndarray, m_abs: np.ndarray, w_abs: np.ndarray,
    y_fluo: np.ndarray, m_fluo: np.ndarray, w_fluo: np.ndarray,
    folds_info,
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
    p = search_space_nn_common(trial)
    scores: List[float] = []
    fold_detail: Dict[str, Any] = {}

    device = torch.device("cuda" if torch.cuda.is_available() and accelerator in ("gpu", "cuda") else "cpu")

    for step, (tr, va, f) in enumerate(folds_info):
        set_all_seeds(seed + 1000 * f + trial.number)

        lam = lambda_from_prevalence(y_cls[tr], power=p["lambda_power"])
        posw = pos_weight_per_task(y_cls[tr], clip=p["pos_weight_clip"])

        prev = y_cls[tr].mean(axis=0)
        gamma = np.array([p["focal_gamma_rare"] if prev[i] < 0.02 else 0.0 for i in range(4)], dtype=np.float32)
        gamma_t = torch.tensor(gamma, dtype=torch.float32)

        mu_abs, sd_abs = fit_standardizer(y_abs, m_abs, tr)
        mu_f, sd_f = fit_standardizer(y_fluo, m_fluo, tr)
        y_abs_sc = apply_standardizer(y_abs, mu_abs, sd_abs)
        y_fluo_sc = apply_standardizer(y_fluo, mu_f, sd_f)

        ds_tr = TabularAuxDataset(
            X2d_scaled[tr], y_cls[tr], w_cls[tr],
            y_abs_sc[tr], m_abs[tr], w_abs[tr],
            y_fluo_sc[tr], m_fluo[tr], w_fluo[tr],
        )
        ds_va = TabularAuxDataset(
            X2d_scaled[va], y_cls[va], w_cls[va],
            y_abs_sc[va], m_abs[va], w_abs[va],
            y_fluo_sc[va], m_fluo[va], w_fluo[va],
        )

        sampler = make_weighted_sampler(y_cls[tr], rare_mult=p["rare_oversample_mult"])
        nw = int(num_workers)

        dl_kw = dict(
            num_workers=nw,
            pin_memory=pin_memory,
            persistent_workers=(nw > 0),
        )
        if nw > 0:
            dl_kw["prefetch_factor"] = 2

        dl_tr = DataLoader(ds_tr, batch_size=int(p["batch_size"]), sampler=sampler, **dl_kw)
        dl_va = DataLoader(ds_va, batch_size=1024, shuffle=False, **dl_kw)

        model = MultiTaskMLPWithAux(
            in_dim=int(X2d_scaled.shape[1]),
            hidden_dim=int(p["hidden_dim"]),
            n_layers=int(p["n_layers"]),
            dropout=float(p["dropout"]),
            lr=float(p["lr"]),
            weight_decay=float(p["weight_decay"]),
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
            lambda_aux_abs=float(p["lambda_aux_abs"]),
            lambda_aux_fluo=float(p["lambda_aux_fluo"]),
            reg_loss_type=str(p["reg_loss_type"]),
        )

        fold_ckpt_dir = ckpt_root / f"mlp_trial{trial.number}_fold{f}"
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

        best_macro, best_aps = eval_mlp_best_epoch(model, dl_va, device=device)

        print(
            f"[MLP] trial={trial.number} fold={f} trained_epochs={epochs_trained} best_epoch={best_epoch} "
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
            # sanity check: are 0/1 weights actually < 1 sometimes?
            "mean_w_task0_train": float(np.mean(w_cls[tr, 0])),
            "mean_w_task1_train": float(np.mean(w_cls[tr, 1])),
            "frac_w_task0_lt1_train": float(np.mean((w_cls[tr, 0] < 0.999).astype(float))),
            "frac_w_task1_lt1_train": float(np.mean((w_cls[tr, 1] < 0.999).astype(float))),
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


def objective_mil_cv(
    trial: Trial,
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
    p = search_space_nn_common(trial)

    # MIL-only params
    inst_hidden = trial.suggest_categorical("inst_hidden", [256, 512, 1024])
    inst_layers = trial.suggest_int("inst_layers", 1, 3)
    inst_dropout = trial.suggest_float("inst_dropout", 0.0, 0.1)
    trunk_hidden = trial.suggest_categorical("trunk_hidden", [256, 512, 1024])
    trunk_layers = trial.suggest_int("trunk_layers", 1, 3)
    trunk_dropout = trial.suggest_float("trunk_dropout", 0.0, 0.3)
    max_instances = 0  # keep all

    scores: List[float] = []
    fold_detail: Dict[str, Any] = {}

    device = torch.device("cuda" if torch.cuda.is_available() and accelerator in ("gpu", "cuda") else "cpu")

    for step, (tr, va, f) in enumerate(folds_info):
        set_all_seeds(seed + 5000 * f + trial.number)

        lam = lambda_from_prevalence(y_cls[tr], power=p["lambda_power"])
        posw = pos_weight_per_task(y_cls[tr], clip=p["pos_weight_clip"])

        prev = y_cls[tr].mean(axis=0)
        gamma = np.array([p["focal_gamma_rare"] if prev[i] < 0.02 else 0.0 for i in range(4)], dtype=np.float32)
        gamma_t = torch.tensor(gamma, dtype=torch.float32)

        mu_abs, sd_abs = fit_standardizer(y_abs, m_abs, tr)
        mu_f, sd_f = fit_standardizer(y_fluo, m_fluo, tr)
        y_abs_sc = apply_standardizer(y_abs, mu_abs, sd_abs)
        y_fluo_sc = apply_standardizer(y_fluo, mu_f, sd_f)

        ids_tr = [ids[i] for i in tr]
        ids_va = [ids[i] for i in va]

        ds_tr = MILPackedDataset(
            ids_tr, y_cls[tr], w_cls[tr],
            y_abs_sc[tr], m_abs[tr], w_abs[tr],
            y_fluo_sc[tr], m_fluo[tr], w_fluo[tr],
            starts, counts, id2pos, Xinst_sorted,
            max_instances=int(max_instances), seed=seed + f,
        )
        ds_va = MILPackedDataset(
            ids_va, y_cls[va], w_cls[va],
            y_abs_sc[va], m_abs[va], w_abs[va],
            y_fluo_sc[va], m_fluo[va], w_fluo[va],
            starts, counts, id2pos, Xinst_sorted,
            max_instances=int(max_instances), seed=seed + 999 + f,
        )

        sampler = make_weighted_sampler(y_cls[tr], rare_mult=p["rare_oversample_mult"])
        nw = int(num_workers)

        dl_kw = dict(
            num_workers=nw,
            pin_memory=pin_memory,
            persistent_workers=(nw > 0),
        )
        if nw > 0:
            dl_kw["prefetch_factor"] = 2

        dl_tr = DataLoader(
            ds_tr,
            batch_size=int(p["batch_size"]),
            sampler=sampler,
            collate_fn=mil_collate_packed,
            **dl_kw,
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=min(128, int(p["batch_size"])),
            shuffle=False,
            collate_fn=mil_collate_packed,
            **dl_kw,
        )

        model = MILMeanPoolPackedWithAux(
            inst_dim=int(Xinst_sorted.shape[1]),
            inst_hidden=int(inst_hidden),
            inst_layers=int(inst_layers),
            inst_dropout=float(inst_dropout),
            trunk_hidden=int(trunk_hidden),
            trunk_layers=int(trunk_layers),
            trunk_dropout=float(trunk_dropout),
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

        best_macro, best_aps = eval_mil_best_epoch(model, dl_va, device=device)

        print(
            f"[MIL] trial={trial.number} fold={f} trained_epochs={epochs_trained} best_epoch={best_epoch} "
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
            "mean_w_task0_train": float(np.mean(w_cls[tr, 0])),
            "mean_w_task1_train": float(np.mean(w_cls[tr, 1])),
            "frac_w_task0_lt1_train": float(np.mean((w_cls[tr, 0] < 0.999).astype(float))),
            "frac_w_task1_lt1_train": float(np.mean((w_cls[tr, 1] < 0.999).astype(float))),
            "mil_params": {
                "inst_hidden": int(inst_hidden),
                "inst_layers": int(inst_layers),
                "inst_dropout": float(inst_dropout),
                "trunk_hidden": int(trunk_hidden),
                "trunk_layers": int(trunk_layers),
                "trunk_dropout": float(trunk_dropout),
            },
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


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--labels", required=True)
    ap.add_argument("--feat2d_scaled", required=True)

    ap.add_argument("--feat3d_scaled", default=None)
    ap.add_argument("--feat3d_qm_scaled", default=None)

    ap.add_argument("--study_dir", required=True)

    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--conf_col", default="conf_id")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--fold_col", default="cv_fold")
    ap.add_argument("--use_splits", nargs="+", default=["train"])
    ap.add_argument("--folds", nargs="+", type=int, default=None)

    ap.add_argument("--do_mlp", action="store_true")
    ap.add_argument("--do_mil", action="store_true")

    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--trials_mlp", type=int, default=50)
    ap.add_argument("--trials_mil", type=int, default=50)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--nn_accelerator", default="gpu")
    ap.add_argument("--nn_devices", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable pin_memory in DataLoaders (can trigger EMFILE on some clusters). Default: off.",
    )

    args = ap.parse_args()

    if not args.do_mlp and not args.do_mil:
        args.do_mlp = True
        args.do_mil = True

    if args.do_mil:
        if not args.feat3d_scaled or not args.feat3d_qm_scaled:
            raise ValueError("--do_mil requires --feat3d_scaled and --feat3d_qm_scaled")

    set_all_seeds(args.seed)
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
    }
    (outdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    # num_workers: -1 => auto (SLURM aware). Otherwise, use user value.
    if args.num_workers < 0:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
        if cpus <= 0:
            cpus = os.cpu_count() or 0
        # good default for single-process training: keep a couple cores for the main process
        num_workers = max(2, min(23, cpus - 2)) if cpus >= 4 else 0
    else:
        num_workers = int(args.num_workers)

    pin_memory = bool(args.pin_memory) and torch.cuda.is_available()

    df = load_labels(args.labels, id_col=args.id_col)
    df[args.split_col] = df[args.split_col].astype(str)
    df = df[df[args.split_col].isin(args.use_splits)].copy().reset_index(drop=True)

    ids_all = df[args.id_col].astype(str).tolist()

    ids_2d, X2d_file = load_2d(args.feat2d_scaled, id_col=args.id_col)
    X2d_scaled = align_by_id(ids_2d, X2d_file, ids_all)

    y_cls = coerce_binary_labels(df)
    w_cls = build_task_weights(df)
    y_abs, m_abs, y_fluo, m_fluo = build_aux_targets_and_masks(df)
    w_abs, w_fluo = build_aux_weights(df)

    if args.folds is None:
        folds = sorted(df[args.fold_col].dropna().astype(int).unique().tolist())
    else:
        folds = args.folds
    folds_info = fold_indices(df, args.fold_col, folds)

    print(f"[DATA-2D] n_ids={len(ids_all)} | X2d_dim={X2d_scaled.shape[1]}")
    print(f"[DATALOADER] num_workers={num_workers} pin_memory={pin_memory} precision={args.precision}")
    print(f"[EARLYSTOP] patience={int(args.patience)} (validation checks)")
    print(f"[WEIGHTS] task0={WEIGHT_COLS[0]} task1={WEIGHT_COLS[1]} task2={WEIGHT_COLS[2]} task3={WEIGHT_COLS[3]}")
    print(f"[WEIGHTS] global means: w0={float(np.mean(w_cls[:,0])):.3f} w1={float(np.mean(w_cls[:,1])):.3f}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)

    def make_storage(name: str) -> str:
        return f"sqlite:///{(outdir / f'{name}.sqlite3').as_posix()}"

    ckpt_root = outdir / "_tmp_best_ckpts"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # --- MIL path ---
    if args.do_mil:
        allowed_ids = set(ids_all)
        ids_conf, X309 = load_and_merge_instances(
            args.feat3d_scaled,
            args.feat3d_qm_scaled,
            allowed_ids=allowed_ids,
            id_col=args.id_col,
            conf_col=args.conf_col,
        )

        _, starts, counts, id2pos, Xinst_sorted = build_instance_matrix_1042(
            ids_conf=ids_conf,
            X309=X309,
            ids_2d_aligned=ids_all,
            X2d_scaled_aligned=X2d_scaled,
        )
        if Xinst_sorted.shape[1] != 1042:
            raise RuntimeError(f"[MIL] instance dim expected 1042, got {Xinst_sorted.shape[1]}")

        have_bag_mask = np.array([(_id in id2pos) for _id in ids_all], dtype=bool)
        if not have_bag_mask.all():
            missing = int((~have_bag_mask).sum())
            ex = [ids_all[i] for i in np.where(~have_bag_mask)[0][:10]]
            print(f"[WARN] Dropping {missing} IDs with 0 conformers after merge. Examples: {ex}")

            df = df.loc[have_bag_mask].reset_index(drop=True)
            ids_all = df[args.id_col].astype(str).tolist()

            X2d_scaled = X2d_scaled[have_bag_mask]
            y_cls = y_cls[have_bag_mask]
            w_cls = w_cls[have_bag_mask]
            y_abs, m_abs, y_fluo, m_fluo = (
                y_abs[have_bag_mask],
                m_abs[have_bag_mask],
                y_fluo[have_bag_mask],
                m_fluo[have_bag_mask],
            )
            w_abs, w_fluo = w_abs[have_bag_mask], w_fluo[have_bag_mask]

            folds = sorted(df[args.fold_col].dropna().astype(int).unique().tolist())
            folds_info = fold_indices(df, args.fold_col, folds)

        print(f"[DATA-MIL] n_ids={len(ids_all)} | n_conf={Xinst_sorted.shape[0]} | inst_dim={Xinst_sorted.shape[1]}")
        print("\n[HPO] MIL Packed MeanPool + Aux on 1042D instances (GPU) ...")

        study_name = "mil_packed_1042_aux_gpu"
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
                y_cls=y_cls, w_cls=w_cls,
                y_abs=y_abs, m_abs=m_abs, w_abs=w_abs,
                y_fluo=y_fluo, m_fluo=m_fluo, w_fluo=w_fluo,
                ids=ids_all,
                folds_info=folds_info,
                starts=starts, counts=counts, id2pos=id2pos, Xinst_sorted=Xinst_sorted,
                seed=args.seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                accelerator=args.nn_accelerator,
                devices=args.nn_devices,
                num_workers=num_workers,
                pin_memory=pin_memory,
                precision=args.precision,
                ckpt_root=ckpt_root,
            ),
            n_trials=args.trials_mil,
            gc_after_trial=True,
            catch=(RuntimeError, ValueError, FloatingPointError),
        )
        save_study_artifacts(outdir, study, prefix=study_name)
        save_best_fold_metrics(outdir, study_name, study.best_trial.user_attrs.get("fold_detail", {}))
        print(f"[HPO] MIL best macro AP (CV-mean, best-epoch eval) = {study.best_value:.5f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # --- MLP path ---
    if args.do_mlp:
        print("\n[HPO] 2D MLP+Aux on 2D SCALED (GPU) ...")
        study_name = "mlp_scaled_aux_gpu"
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=make_storage(study_name),
            load_if_exists=True,
        )
        study.optimize(
            lambda tr: objective_mlp_cv(
                tr,
                X2d_scaled=X2d_scaled,
                y_cls=y_cls, w_cls=w_cls,
                y_abs=y_abs, m_abs=m_abs, w_abs=w_abs,
                y_fluo=y_fluo, m_fluo=m_fluo, w_fluo=w_fluo,
                folds_info=folds_info,
                seed=args.seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                accelerator=args.nn_accelerator,
                devices=args.nn_devices,
                num_workers=num_workers,
                pin_memory=pin_memory,
                precision=args.precision,
                ckpt_root=ckpt_root,
            ),
            n_trials=args.trials_mlp,
            gc_after_trial=True,
            catch=(RuntimeError, ValueError, FloatingPointError),
        )
        save_study_artifacts(outdir, study, prefix=study_name)
        save_best_fold_metrics(outdir, study_name, study.best_trial.user_attrs.get("fold_detail", {}))
        print(f"[HPO] MLP best macro AP (CV-mean, best-epoch eval) = {study.best_value:.5f}")

    # final cleanup
    try:
        shutil.rmtree(ckpt_root, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
