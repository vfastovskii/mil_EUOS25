from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from ..utils.ops import lambda_from_prevalence


def compute_lam(
    params: Dict[str, Any],
    *,
    y_train: np.ndarray,
    fallback_lambda_power: float = 1.0,
    fallback_lam_floor: float = 0.25,
    fallback_lam_ceil: float = 3.5,
) -> np.ndarray:
    if all(k in params for k in ("lam_t0", "lam_t1", "lam_t2", "lam_t3")):
        lam_vec = np.array(
            [
                float(params["lam_t0"]),
                float(params["lam_t1"]),
                float(params["lam_t2"]),
                float(params["lam_t3"]),
            ],
            dtype=np.float32,
        )
        lam = lam_vec / max(float(lam_vec.mean()), 1e-12)
        lam_floor = float(params.get("lam_floor", fallback_lam_floor))
        lam_ceil = float(params.get("lam_ceil", fallback_lam_ceil))
        lam = np.clip(lam, lam_floor, lam_ceil)
        return lam / max(float(lam.mean()), 1e-12)
    return lambda_from_prevalence(y_train, power=float(params.get("lambda_power", fallback_lambda_power)))


def compute_posw_clips(params: Dict[str, Any], *, fallback_clip: float = 50.0) -> List[float] | float:
    if all(k in params for k in ("posw_clip_t0", "posw_clip_t1", "posw_clip_t2", "posw_clip_t3")):
        return [
            float(params["posw_clip_t0"]),
            float(params["posw_clip_t1"]),
            float(params["posw_clip_t2"]),
            float(params["posw_clip_t3"]),
        ]
    return float(params.get("pos_weight_clip", fallback_clip))


def compute_gamma(params: Dict[str, Any]) -> torch.Tensor:
    gamma = np.array(
        [
            float(params["gamma_t0"]),
            float(params["gamma_t1"]),
            float(params["gamma_t2"]),
            float(params["gamma_t3"]),
        ],
        dtype=np.float32,
    )
    return torch.tensor(gamma, dtype=torch.float32)


__all__ = [
    "compute_lam",
    "compute_posw_clips",
    "compute_gamma",
]
