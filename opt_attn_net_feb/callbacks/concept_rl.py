from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from pytorch_lightning.callbacks import Callback
except Exception:  # pragma: no cover
    try:
        from lightning.pytorch.callbacks import Callback  # type: ignore
    except Exception:  # pragma: no cover
        Callback = object  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptRLPolicyConfig:
    """Policy-gradient control configuration for concept-guidance scale."""

    init_mean: float = 0.02
    sigma: float = 0.02
    learning_rate: float = 0.05
    max_scale: float = 0.20
    reward_alignment_w: float = 0.25
    baseline_momentum: float = 0.90
    reward_key: str = "val_macro_ap"
    alignment_key: str = "train_concept_alignment"


@dataclass(frozen=True)
class ConceptRLEpochRecord:
    epoch: int
    action_scale: float
    policy_mean_before: float
    policy_mean_after: float
    reward_model: float
    reward_alignment: float
    reward_total: float
    baseline: float
    advantage: float


@dataclass
class ConceptRLControllerCallback(Callback):
    """REINFORCE-style controller for model concept-guidance strength."""

    config: ConceptRLPolicyConfig
    out_json_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.policy_mean = float(np.clip(self.config.init_mean, 0.0, self.config.max_scale))
        self.baseline: Optional[float] = None
        self.last_action: float = self.policy_mean
        self._epoch_records: list[ConceptRLEpochRecord] = []

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:  # noqa: D401
        if not getattr(pl_module, "rl_enabled", False):
            return
        sigma = float(max(1e-8, self.config.sigma))
        action = float(np.random.normal(loc=self.policy_mean, scale=sigma))
        action = float(np.clip(action, 0.0, self.config.max_scale))
        self.last_action = action
        if hasattr(pl_module, "set_rl_guidance_scale"):
            pl_module.set_rl_guidance_scale(action)
        logger.info(
            "Concept RL sampled action",
            extra={
                "epoch": int(getattr(trainer, "current_epoch", 0)),
                "action_scale": action,
                "policy_mean": self.policy_mean,
            },
        )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: D401
        if not getattr(pl_module, "rl_enabled", False):
            return
        metrics = getattr(trainer, "callback_metrics", {})
        reward_model = self._metric_float(metrics.get(self.config.reward_key, 0.0))
        reward_align = self._metric_float(metrics.get(self.config.alignment_key, 0.0))
        reward_total = float(reward_model + float(self.config.reward_alignment_w) * reward_align)

        if self.baseline is None:
            self.baseline = reward_total
        advantage = float(reward_total - float(self.baseline))
        self.baseline = float(
            float(self.config.baseline_momentum) * float(self.baseline)
            + (1.0 - float(self.config.baseline_momentum)) * reward_total
        )

        sigma = float(max(1e-8, self.config.sigma))
        mu_before = float(self.policy_mean)
        grad_log_prob = (float(self.last_action) - mu_before) / float(sigma ** 2)
        self.policy_mean = float(
            np.clip(
                mu_before + float(self.config.learning_rate) * advantage * grad_log_prob,
                0.0,
                float(self.config.max_scale),
            )
        )

        rec = ConceptRLEpochRecord(
            epoch=int(getattr(trainer, "current_epoch", 0)),
            action_scale=float(self.last_action),
            policy_mean_before=mu_before,
            policy_mean_after=float(self.policy_mean),
            reward_model=float(reward_model),
            reward_alignment=float(reward_align),
            reward_total=float(reward_total),
            baseline=float(self.baseline),
            advantage=float(advantage),
        )
        self._epoch_records.append(rec)

        if hasattr(pl_module, "log"):
            pl_module.log("rl_action_scale", float(self.last_action), on_step=False, on_epoch=True)
            pl_module.log("rl_policy_mean", float(self.policy_mean), on_step=False, on_epoch=True)
            pl_module.log("rl_reward_total", float(reward_total), on_step=False, on_epoch=True)
            pl_module.log("rl_advantage", float(advantage), on_step=False, on_epoch=True)

        logger.info(
            "Concept RL update",
            extra={
                "epoch": rec.epoch,
                "action": rec.action_scale,
                "mu_before": rec.policy_mean_before,
                "mu_after": rec.policy_mean_after,
                "reward_total": rec.reward_total,
                "advantage": rec.advantage,
            },
        )

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:  # noqa: D401
        if self.out_json_path is None:
            return
        p = Path(self.out_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "history": [asdict(x) for x in self._epoch_records],
        }
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info("Wrote concept RL policy history", extra={"path": str(p)})

    @staticmethod
    def _metric_float(v: Any) -> float:
        try:
            if hasattr(v, "detach"):
                return float(v.detach().cpu().item())
            return float(v)
        except Exception:
            return 0.0


__all__ = ["ConceptRLPolicyConfig", "ConceptRLEpochRecord", "ConceptRLControllerCallback"]

