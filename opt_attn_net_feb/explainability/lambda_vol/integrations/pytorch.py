from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd

from ..monitor import EpochStepResult, LambdaVolMonitor


@dataclass
class LambdaVolPyTorchAdapter:
    """Lightweight adapter for plain PyTorch epoch-end monitoring."""

    monitor: LambdaVolMonitor

    def on_epoch_end(
        self,
        *,
        epoch: int,
        tcav_df: pd.DataFrame,
        concept_attention_df: Optional[pd.DataFrame] = None,
        task_attention_df: Optional[pd.DataFrame] = None,
        task_metrics_df: Optional[pd.DataFrame] = None,
        context_covariates: Optional[Mapping[str, float]] = None,
    ) -> EpochStepResult:
        """Call this at the end of each epoch in a custom PyTorch loop."""
        return self.monitor.step_from_frames(
            epoch=int(epoch),
            tcav_df=tcav_df,
            concept_attention_df=concept_attention_df,
            task_attention_df=task_attention_df,
            task_metrics_df=task_metrics_df,
            context_covariates=context_covariates,
        )


__all__ = ["LambdaVolPyTorchAdapter"]
