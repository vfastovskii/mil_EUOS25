from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

try:
    from pytorch_lightning.callbacks import Callback
except Exception:  # pragma: no cover
    from lightning.pytorch.callbacks import Callback  # type: ignore

from ..callbacks.optuna_pruning import OptunaPruningCallbackLocal
from ..utils.metrics import ap_per_task, roc_auc_per_task


@dataclass(frozen=True)
class LightningTrainerConfig:
    """
    Configuration class for LightningTrainer.

    This class serves as a data container for configuring a LightningTrainer.
    It provides essential properties related to model training, such as the
    number of epochs, early stopping patience, hardware configuration,
    gradient accumulation, and checkpoint settings.

    Attributes:
        max_epochs: int
            The maximum number of training epochs to run.
        patience: int
            The patience value for early stopping in terms of epochs.
        accelerator: str
            Backend to use for training, e.g., 'cpu', 'gpu', 'tpu'.
        devices: int
            Number of devices to use for training.
        precision: str
            Floating-point precision type, e.g., '32', '16', or 'bf16'.
        accumulate_grad_batches: int
            Number of batches before performing a gradient update.
        save_checkpoint: bool
            Whether to save model checkpoint files during training. Defaults to True.
        save_weights_only: bool
            Whether to save only model weights when saving the checkpoint. Defaults to True.
    """
    max_epochs: int
    patience: int
    accelerator: str
    devices: int
    precision: str
    accumulate_grad_batches: int
    save_checkpoint: bool = True
    save_weights_only: bool = True


class LightningTrainerFactory:
    """
    Factory class responsible for creating and configuring a PyTorch Lightning Trainer instance
    and optional ModelCheckpoint callback.

    This class provides a method to construct a PyTorch Lightning Trainer with specific configurations
    determined by the provided LightningTrainerConfig. It also includes functionality for adding
    callbacks such as early stopping, checkpointing, and Optuna pruning for hyperparameter optimization.

    Attributes:
        config (LightningTrainerConfig): Configuration object that holds all necessary trainer
            parameters such as training epochs, checkpoint settings, and acceleration options.
    """

    def __init__(self, config: LightningTrainerConfig):
        self.config = config

    def build(
        self,
        *,
        ckpt_dir: str,
        trial: Optional["optuna.trial.Trial"] = None,
    ) -> Tuple[pl.Trainer, Optional[pl.callbacks.ModelCheckpoint]]:
        es = pl.callbacks.EarlyStopping(
            monitor="val_macro_ap",
            mode="max",
            patience=int(self.config.patience),
        )
        callbacks: List[Callback] = [es]
        ckpt: Optional[pl.callbacks.ModelCheckpoint] = None
        if bool(self.config.save_checkpoint):
            ckpt = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best",
                monitor="val_macro_ap",
                mode="max",
                save_top_k=1,
                save_last=False,
                auto_insert_metric_name=False,
                save_weights_only=bool(self.config.save_weights_only),
            )
            callbacks.append(ckpt)
        if trial is not None:
            callbacks.append(OptunaPruningCallbackLocal(trial, monitor="val_macro_ap"))

        trainer = pl.Trainer(
            max_epochs=int(self.config.max_epochs),
            logger=False,
            enable_checkpointing=bool(self.config.save_checkpoint),
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=callbacks,
            accelerator=str(self.config.accelerator),
            devices=int(self.config.devices),
            precision=str(self.config.precision),
            deterministic=True,
            gradient_clip_val=0.0,
            accumulate_grad_batches=int(self.config.accumulate_grad_batches),
        )
        return trainer, ckpt


class ModelEvaluator:
    """
    The ModelEvaluator class is responsible for evaluating machine learning models by calculating performance
    metrics on validation datasets. Its primary purpose is to streamline the process of evaluating a model's
    effectiveness across various tasks, using provided validation data.

    This utility class operates entirely on a specified computational device and works in a no-grad computation
    context to optimize performance during evaluation. Its main method processes batches of validation data,
    applies the model, and aggregates predictions and labels to compute task-based metrics.

    Attributes:
        device (torch.device): The device on which the evaluation computations will be performed. This could be a CPU
            or a GPU, depending on the system configuration and model requirements.

    Methods:
        eval_best_epoch (model, dl_va): Evaluates the provided model on the given validation data loader and returns
            the computed average precision per task along with its mean.
    """

    def __init__(self, *, device: torch.device):
        self.device = device

    @torch.no_grad()
    def eval_best_epoch(self, model, dl_va: DataLoader) -> Tuple[float, List[float], float, List[float]]:
        model.eval()
        model.to(self.device)
        ps, ys, ws = [], [], []
        for batch in dl_va:
            x2d, x3d, kpm, y_cls, w_cls, *_ = batch
            x2d = x2d.to(self.device, non_blocking=True)
            x3d = x3d.to(self.device, non_blocking=True)
            kpm = kpm.to(self.device, non_blocking=True)
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
        aucs = roc_auc_per_task(Y, P, w_cls=W, weighted_tasks=(0, 1))
        return float(np.mean(aps)), aps, float(np.mean(aucs)), aucs


def make_trainer_gpu(
    max_epochs: int,
    patience: int,
    accelerator: str,
    devices: int,
    precision: str,
    accumulate_grad_batches: int,
    ckpt_dir: str,
    trial: Optional["optuna.trial.Trial"] = None,
    save_checkpoint: bool = True,
    save_weights_only: bool = True,
) -> Tuple[pl.Trainer, Optional[pl.callbacks.ModelCheckpoint]]:
    """
    Creates and returns a PyTorch Lightning Trainer instance configured for GPU
    training, as well as an optional checkpoint callback. The trainer is configured
    based on the provided parameters using `LightningTrainerConfig` and
    `LightningTrainerFactory`.

    Parameters:
    max_epochs: int
        The maximum number of epochs to train the model.
    patience: int
        The number of epochs to wait for improvement before early stopping.
    accelerator: str
        The type of accelerator to use for training (e.g., "gpu").
    devices: int
        The number of devices to use for training.
    precision: str
        The precision to use during training (e.g., "16", "32").
    accumulate_grad_batches: int
        The number of steps to accumulate gradients before performing a backward
        pass.
    ckpt_dir: str
        The directory where the model checkpoints will be saved.
    trial: Optional[optuna.trial.Trial]
        An optional Optuna trial instance to perform hyperparameter optimization.
    save_checkpoint: bool
        Whether to save checkpoints during training.
    save_weights_only: bool
        Whether to save only the model weights, excluding the optimizer state.

    Returns:
    Tuple[pl.Trainer, Optional[pl.callbacks.ModelCheckpoint]]
        A tuple containing the initialized PyTorch Lightning Trainer and an
        optional checkpoint callback.

    Raises:
        This function does not raise explicit exceptions but may propagate exceptions
        from `LightningTrainerConfig` or `LightningTrainerFactory`.
    """
    cfg = LightningTrainerConfig(
        max_epochs=int(max_epochs),
        patience=int(patience),
        accelerator=str(accelerator),
        devices=int(devices),
        precision=str(precision),
        accumulate_grad_batches=int(accumulate_grad_batches),
        save_checkpoint=bool(save_checkpoint),
        save_weights_only=bool(save_weights_only),
    )
    return LightningTrainerFactory(cfg).build(ckpt_dir=ckpt_dir, trial=trial)


@torch.no_grad()
def eval_best_epoch(
    model,
    dl_va: DataLoader,
    device: torch.device,
) -> Tuple[float, List[float], float, List[float]]:
    """
    Evaluates the best epoch for a given model on a validation dataset without computing gradients.

    This method uses a ModelEvaluator instance to evaluate the model's performance
    on a given DataLoader for the validation set. The evaluation is carried out in
    a no-grad context to improve memory efficiency and speed during inference. The
    resulting loss and additional metrics are returned.

    Arguments:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dl_va (DataLoader): The DataLoader for the validation dataset.
        device (torch.device): The device on which the evaluation will be performed.

    Returns:
        Tuple[float, List[float]]: A tuple containing the average loss (float) and a list
        of additional metric scores (List[float]).
    """
    return ModelEvaluator(device=device).eval_best_epoch(model, dl_va)


__all__ = [
    "LightningTrainerConfig",
    "LightningTrainerFactory",
    "ModelEvaluator",
    "make_trainer_gpu",
    "eval_best_epoch",
]
