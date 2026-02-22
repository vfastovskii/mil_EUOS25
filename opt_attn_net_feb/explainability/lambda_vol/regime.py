from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from .config import RegimeConfig
from .types import RegimeLabel

logger = logging.getLogger(__name__)


class RegimeClassifier(Protocol):
    """Optional classifier-based regime inference interface."""

    def predict_regime(
        self,
        *,
        epoch: int,
        context: Mapping[str, float],
        history_task_df: pd.DataFrame,
    ) -> RegimeLabel:
        """Predict regime label for the current epoch."""


@dataclass(frozen=True)
class RuleBasedRegimeInferer:
    """Rule-based regime labeling q(t)."""

    config: RegimeConfig

    def infer(
        self,
        *,
        epoch: int,
        task_df_history: pd.DataFrame,
        current_task_df: pd.DataFrame,
        last_regime: Optional[RegimeLabel],
    ) -> RegimeLabel:
        """Infer global epoch regime from history and current task metrics."""
        ep = int(epoch)
        if ep < int(self.config.warmup_epochs):
            return RegimeLabel.WARMUP

        merged = self._merge_history(task_df_history=task_df_history, current_task_df=current_task_df)
        if merged.empty:
            return RegimeLabel.FITTING

        val_slope = self._series_slope(merged, "val_metric")
        loss_slope = self._series_slope(merged, "loss")
        entropy_slope = self._series_slope(merged, "attention_entropy")
        topk_slope = self._series_slope(merged, "topk_mass") if "topk_mass" in merged.columns else 0.0

        last = merged.iloc[-1]
        gap = float(last["train_metric"] - last["val_metric"])

        if (val_slope < -float(self.config.val_slope_small)) and (gap > float(self.config.overfit_gap_threshold)):
            return RegimeLabel.OVERFIT

        stable_val = abs(val_slope) <= float(self.config.val_slope_small)
        stable_loss = abs(loss_slope) <= float(self.config.val_slope_small)
        low_gap = gap <= float(self.config.overfit_gap_threshold) * 0.5
        if stable_val and stable_loss and low_gap:
            return RegimeLabel.STABLE

        if last_regime == RegimeLabel.OVERFIT and val_slope > float(self.config.val_slope_small):
            return RegimeLabel.REFIT

        if entropy_slope < -float(self.config.entropy_drop_threshold) and topk_slope > float(
            self.config.concentration_rise_threshold
        ):
            return RegimeLabel.OVERFIT

        return RegimeLabel.FITTING

    @staticmethod
    def _merge_history(task_df_history: pd.DataFrame, current_task_df: pd.DataFrame) -> pd.DataFrame:
        req = {"epoch", "task_id", "train_metric", "val_metric", "loss", "attention_entropy"}
        if not req.issubset(set(current_task_df.columns)):
            raise ValueError(f"current_task_df missing required columns: {sorted(req)}")

        if task_df_history is None or task_df_history.empty:
            merged = current_task_df.copy()
        else:
            merged = pd.concat([task_df_history.copy(), current_task_df.copy()], ignore_index=True)

        # average per epoch across tasks to get global regime signal
        agg_cols = [c for c in ["train_metric", "val_metric", "loss", "attention_entropy", "topk_mass"] if c in merged.columns]
        out = merged.groupby("epoch", as_index=False)[agg_cols].mean().sort_values("epoch")
        return out

    @staticmethod
    def _series_slope(df: pd.DataFrame, col: str, window: int = 4) -> float:
        if col not in df.columns:
            return 0.0
        sub = df[["epoch", col]].dropna().tail(int(max(2, window)))
        if len(sub) < 2:
            return 0.0
        x = sub["epoch"].to_numpy(dtype=np.float64)
        y = sub[col].to_numpy(dtype=np.float64)
        return float(np.polyfit(x, y, deg=1)[0])


@dataclass(frozen=True)
class HybridRegimeInferer:
    """Regime inferer combining classifier (optional) and rule fallback."""

    rules: RuleBasedRegimeInferer
    classifier: Optional[RegimeClassifier] = None

    def infer(
        self,
        *,
        epoch: int,
        task_df_history: pd.DataFrame,
        current_task_df: pd.DataFrame,
        context: Mapping[str, float],
        last_regime: Optional[RegimeLabel],
    ) -> RegimeLabel:
        if self.classifier is not None:
            try:
                merged = self.rules._merge_history(task_df_history=task_df_history, current_task_df=current_task_df)
                pred = self.classifier.predict_regime(
                    epoch=int(epoch),
                    context=context,
                    history_task_df=merged,
                )
                return RegimeLabel(pred)
            except Exception:
                logger.exception("Classifier regime inference failed; falling back to rules")

        return self.rules.infer(
            epoch=int(epoch),
            task_df_history=task_df_history,
            current_task_df=current_task_df,
            last_regime=last_regime,
        )


__all__ = ["HybridRegimeInferer", "RegimeClassifier", "RuleBasedRegimeInferer"]
