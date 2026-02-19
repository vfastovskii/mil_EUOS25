from __future__ import annotations

from typing import Any, Mapping, List

import numpy as np
import torch

from .configs import LossWeightingConfig
from ..utils.ops import lambda_from_prevalence


def _as_loss_config(
    params: LossWeightingConfig | Mapping[str, Any],
    *,
    fallback_lambda_power: float = 1.0,
    fallback_lam_floor: float = 0.25,
    fallback_lam_ceil: float = 3.5,
    fallback_pos_weight_clip: float = 50.0,
) -> LossWeightingConfig:
    if isinstance(params, LossWeightingConfig):
        return params
    return LossWeightingConfig.from_params(
        params,
        fallback_lambda_power=fallback_lambda_power,
        fallback_lam_floor=fallback_lam_floor,
        fallback_lam_ceil=fallback_lam_ceil,
        fallback_pos_weight_clip=fallback_pos_weight_clip,
    )


def compute_lam(
    params: LossWeightingConfig | Mapping[str, Any],
    *,
    y_train: np.ndarray,
    fallback_lambda_power: float = 1.0,
    fallback_lam_floor: float = 0.25,
    fallback_lam_ceil: float = 3.5,
) -> np.ndarray:
    cfg = _as_loss_config(
        params,
        fallback_lambda_power=fallback_lambda_power,
        fallback_lam_floor=fallback_lam_floor,
        fallback_lam_ceil=fallback_lam_ceil,
    )
    lam_per_task = cfg.per_task_lam()
    if lam_per_task is not None:
        lam_vec = np.array(
            list(lam_per_task),
            dtype=np.float32,
        )
        lam = lam_vec / max(float(lam_vec.mean()), 1e-12)
        lam_floor = float(cfg.lam_floor)
        lam_ceil = float(cfg.lam_ceil)
        lam = np.clip(lam, lam_floor, lam_ceil)
        return lam / max(float(lam.mean()), 1e-12)
    return lambda_from_prevalence(y_train, power=float(cfg.lambda_power))


def compute_posw_clips(
    params: LossWeightingConfig | Mapping[str, Any],
    *,
    fallback_clip: float = 50.0,
) -> List[float] | float:
    cfg = _as_loss_config(params, fallback_pos_weight_clip=fallback_clip)
    posw_clips = cfg.per_task_posw_clips()
    if posw_clips is not None:
        return [float(v) for v in posw_clips]
    return float(cfg.pos_weight_clip)


def compute_gamma(params: LossWeightingConfig | Mapping[str, Any]) -> torch.Tensor:
    cfg = _as_loss_config(params)
    gamma = np.array(
        [
            float(cfg.gamma_t0),
            float(cfg.gamma_t1),
            float(cfg.gamma_t2),
            float(cfg.gamma_t3),
        ],
        dtype=np.float32,
    )
    return torch.tensor(gamma, dtype=torch.float32)


__all__ = [
    "compute_lam",
    "compute_posw_clips",
    "compute_gamma",
]
