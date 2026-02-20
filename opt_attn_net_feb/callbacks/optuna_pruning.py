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
    """
    Callback for integrating Optuna's pruning feature with PyTorch Lightning.

    This class serves as a wrapper that implements a callback which periodically
    monitors a specified validation metric during training and reports the value
    to an Optuna trial. If the monitored metric suggests that the trial should
    be pruned, the callback raises a `TrialPruned` exception to terminate the
    current trial early.

    Attributes:
        trial (Trial): An Optuna Trial instance used to report metrics and decide
            pruning.
        monitor (str): The name of the metric to monitor for pruning signals.
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
