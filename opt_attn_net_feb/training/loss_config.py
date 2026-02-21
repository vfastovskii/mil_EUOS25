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
    """
    Converts input parameters into a LossWeightingConfig instance.

    This function takes either a `LossWeightingConfig` instance or a mapping of parameters
    and converts it into a `LossWeightingConfig` instance. If a mapping is provided, the
    function utilizes the given fallback values for specific parameters to handle
    uncertainties or defaults during the conversion.

    Parameters:
        params (LossWeightingConfig | Mapping[str, Any]): Input configuration, which can
            be either an instance of `LossWeightingConfig` or a dictionary containing
            configuration parameters.
        fallback_lambda_power (float): A fallback value for the lambda_power parameter.
            Default is 1.0.
        fallback_lam_floor (float): A fallback lower bound for the lambda parameter.
            Default is 0.25.
        fallback_lam_ceil (float): A fallback upper bound for the lambda parameter.
            Default is 3.5.
        fallback_pos_weight_clip (float): A fallback maximum value for the
            positional weights. Default is 50.0.

    Returns:
        LossWeightingConfig: The resulting `LossWeightingConfig` instance created either
            from the input `params` or a mapping with fallback values.
    """
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
    """
    Computes the lambda values for loss weighting per task.

    This function determines lambda values either from a provided configuration or based
    on the prevalence of class labels within the training data. It ensures that lambda
    values for each task are normalized, and optionally applies floor and ceiling constraints
    to limit the range of values. The computed lambda is useful in scenarios requiring
    task-specific adjustments to loss functions during training.

    Parameters:
    params: LossWeightingConfig | Mapping[str, Any]
        Contains the loss weighting configuration. If a mapping is provided, values such
        as lambda power, floor, and ceiling constraints are extracted from it. A
        `LossWeightingConfig` object can be passed directly as well.

    y_train: np.ndarray
        An array containing the training labels. Used when deriving the lambda values
        based on label prevalence across the training data.

    fallback_lambda_power: float, optional
        Default value to use for the lambda power in case it is not specified in the
        provided configuration. Defaults to 1.0.

    fallback_lam_floor: float, optional
        Default value to use for the lambda floor constraint if it is not explicitly
        provided in the configuration. Defaults to 0.25.

    fallback_lam_ceil: float, optional
        Default value to use for the lambda ceiling constraint if it is not explicitly
        provided in the configuration. Defaults to 3.5.

    Returns:
    np.ndarray
        An array of computed lambda values for each task. These values are normalized and
        optionally constrained by the provided floor and ceiling settings.
    """
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
    """
    Computes the positive weight clips for a given loss weighting configuration.

    This function accepts a loss weighting configuration or a mapping, computes the
    positive weight clips per task if available, and provides the corresponding
    float values as a list. If no per-task clips are found, it returns the fallback
    positive weight clip as a single float. The fallback clip value can be adjusted
    via a parameter.

    Args:
        params: A LossWeightingConfig object or a dictionary-like mapping containing
            the configuration for loss weighting.
        fallback_clip: The float value to use as the fallback positive weight clip
            if none is provided in the parameters. Defaults to 50.0.

    Returns:
        A list of float values representing the positive weight clips per task if
        applicable, or a single float value for the positive weight clip.
    """
    cfg = _as_loss_config(params, fallback_pos_weight_clip=fallback_clip)
    posw_clips = cfg.per_task_posw_clips()
    if posw_clips is not None:
        return [float(v) for v in posw_clips]
    return float(cfg.pos_weight_clip)


def compute_gamma(params: LossWeightingConfig | Mapping[str, Any]) -> torch.Tensor:
    """
    Compute the gamma values and return as a tensor.

    The function calculates the gamma values based on the provided configuration
    parameters and converts them into a PyTorch tensor. The configuration is
    initially parsed into a specific format before extracting the gamma values.

    Args:
        params (LossWeightingConfig | Mapping[str, Any]): Configuration containing
            gamma parameters (`gamma_t0`, `gamma_t1`, `gamma_t2`, `gamma_t3`)
            needed for computation.

    Returns:
        torch.Tensor: A tensor containing the computed gamma values.
    """
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
