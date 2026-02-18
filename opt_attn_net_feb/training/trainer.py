from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

try:
    from pytorch_lightning.callbacks import Callback
except Exception:  # pragma: no cover
    from lightning.pytorch.callbacks import Callback  # type: ignore

from ..callbacks.optuna_pruning import OptunaPruningCallbackLocal
from ..utils.metrics import ap_per_task


def make_trainer_gpu(
    max_epochs: int,
    patience: int,
    accelerator: str,
    devices: int,
    precision: str,
    accumulate_grad_batches: int,
    ckpt_dir: str,
    trial: Optional["optuna.trial.Trial"] = None,
) -> Tuple[pl.Trainer, pl.callbacks.ModelCheckpoint]:
    """Construct a deterministic Lightning trainer with standard callbacks.

    - EarlyStopping and ModelCheckpoint on val_macro_ap
    - Optional Optuna pruning callback when a trial is provided
    """
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

    callbacks: List[Callback] = [ es, ckpt ]
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
def eval_best_epoch(model, dl_va: DataLoader, device: torch.device) -> Tuple[float, List[float]]:
    """Evaluate a trained model on a validation loader and return macro AP and per-task APs."""
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


__all__ = ["make_trainer_gpu", "eval_best_epoch"]
