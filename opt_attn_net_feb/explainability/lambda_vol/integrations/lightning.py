from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Optional, Protocol

import pandas as pd

from ..monitor import EpochStepResult, LambdaVolMonitor

try:
    from pytorch_lightning.callbacks import Callback
except Exception:  # pragma: no cover
    try:
        from lightning.pytorch.callbacks import Callback  # type: ignore
    except Exception:  # pragma: no cover
        Callback = object  # type: ignore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LightningEpochFrames:
    """Epoch frames collected from a Lightning training run."""

    tcav_df: pd.DataFrame
    concept_attention_df: Optional[pd.DataFrame] = None
    task_attention_df: Optional[pd.DataFrame] = None
    task_metrics_df: Optional[pd.DataFrame] = None
    context_covariates: Optional[Mapping[str, float]] = None


class LightningFrameProvider(Protocol):
    """Collects Lambda-Vol DataFrames from Lightning trainer/module state."""

    def collect_epoch_frames(
        self,
        *,
        trainer: Any,
        pl_module: Any,
        epoch: int,
    ) -> LightningEpochFrames:
        """Return epoch frames for Lambda-Vol monitoring."""


@dataclass
class LambdaVolLightningCallback(Callback):
    """PyTorch Lightning callback integrating Lambda-Vol monitoring at epoch end."""

    monitor: LambdaVolMonitor
    frame_provider: LightningFrameProvider
    export_on_fit_end: bool = True

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: D401
        epoch = int(getattr(trainer, "current_epoch", 0))
        frames = self.frame_provider.collect_epoch_frames(
            trainer=trainer,
            pl_module=pl_module,
            epoch=epoch,
        )
        result: EpochStepResult = self.monitor.step_from_frames(
            epoch=epoch,
            tcav_df=frames.tcav_df,
            concept_attention_df=frames.concept_attention_df,
            task_attention_df=frames.task_attention_df,
            task_metrics_df=frames.task_metrics_df,
            context_covariates=frames.context_covariates,
        )
        logger.info(
            "Lambda-Vol callback epoch end",
            extra={
                "epoch": result.epoch,
                "regime": result.regime_label.value,
                "n_alerts": result.n_alerts,
            },
        )

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: D401
        if bool(self.export_on_fit_end):
            self.monitor.finalize()


__all__ = ["LambdaVolLightningCallback", "LightningEpochFrames", "LightningFrameProvider"]
