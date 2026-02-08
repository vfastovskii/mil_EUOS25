from __future__ import annotations

from typing import Optional

import torch
try:
    from pytorch_lightning.callbacks import Callback
except Exception:  # pragma: no cover
    from lightning.pytorch.callbacks import Callback  # type: ignore

import optuna
from optuna.trial import Trial


class OptunaPruningCallbackLocal(Callback):
    """Optuna pruning callback compatible with Lightning Trainer.

    Monitors a metric and reports it to the Optuna trial. If the trial should be
    pruned according to Optuna's logic, raises optuna.TrialPruned.
    """

    def __init__(self, trial: Trial, monitor: str = "val_macro_ap"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: D401
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor, None)
        if current is None:
            return
        current_val = float(current.detach().cpu().item()) if torch.is_tensor(current) else float(current)
        self.trial.report(current_val, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {trainer.current_epoch}")
