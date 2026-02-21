from __future__ import annotations

from typing import Any, Dict

import optuna
from optuna.trial import Trial


def search_space(trial: Trial) -> Dict[str, Any]:
    """
    Generates a dictionary defining a hyperparameter search space for optimization.

    This function specifies a comprehensive set of hyperparameters to be tuned for
    training a machine learning model using an optimization library, such as Optuna.
    These parameters include network architecture settings (e.g., layer dimensions,
    dropout rates, activation functions), optimization parameters (e.g., learning
    rate, weight decay), regularization settings, and other training-specific
    configurations. It also applies specific constraints to ensure validity,
    such as aligning certain parameter relationships, e.g., divisibility
    requirements. The returned dictionary is tailored for dimensionality
    restrictions, extensible component selection, and specific tasks.

    Parameters:
    trial : Trial
        The trial object representing the current parameter configuration being
        tested within the search space.

    Raises:
    optuna.TrialPruned
        If specific constraints between parameters are not satisfied, such as the
        `inst_hidden` size not being divisible by the number of attention heads.

    Returns:
    Dict[str, Any]
        A dictionary containing the suggested values for each parameter within the
        specified hyperparameter search space.
    """
    p = {"mol_hidden": trial.suggest_categorical("mol_hidden", [128, 256]),
         "mol_layers": trial.suggest_int("mol_layers", 2, 5),
         "mol_dropout": trial.suggest_float("mol_dropout", 0.10, 0.25),
         "inst_hidden": trial.suggest_categorical("inst_hidden", [128, 256]),
         "inst_layers": trial.suggest_int("inst_layers", 3, 5),
         "inst_dropout": trial.suggest_float("inst_dropout", 0.05, 0.15),
         "proj_dim": trial.suggest_categorical("proj_dim", [256, 512]),
         "attn_heads": trial.suggest_categorical("attn_heads", [8, 16, 32]),
         "attn_dropout": trial.suggest_float("attn_dropout", 0.05, 0.2),
         "mixer_hidden": trial.suggest_categorical("mixer_hidden", [128, 256]),
         "mixer_layers": trial.suggest_int("mixer_layers", 3, 5),
         "mixer_dropout": trial.suggest_float("mixer_dropout", 0.05, 0.2),
         "mol_embedder_name": trial.suggest_categorical("mol_embedder_name", ["mlp_v3_2d"]),
         "inst_embedder_name": trial.suggest_categorical("inst_embedder_name", ["mlp_v3_3d"]),
         "aggregator_name": trial.suggest_categorical("aggregator_name", ["task_attention_pool"]),
         "predictor_name": trial.suggest_categorical("predictor_name", ["mlp_v3"]),
         "head_num_layers": trial.suggest_categorical("head_num_layers", [2, 3, 4, 6]),
         "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.2),
         "head_fc2_gain_non_last": trial.suggest_categorical("head_fc2_gain_non_last", [1e-3, 3e-3, 1e-2]),
         "activation": trial.suggest_categorical("activation", ["GELU", "SiLU", "Mish", "ReLU", "LeakyReLU"]),
         "lr": trial.suggest_float("lr", 8e-5, 8e-4, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 3e-6, 3e-4, log=True),
         "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),

         # Task mapping
         # t0 -> Transmittance_340
         # t1 -> Transmittance_450
         # t2 -> Fluorescence_340_450
         # t3 -> Fluorescence_more_than_480

         # POsitive rate per task
         # T340   : 0.05637 +/- 0.00010  (min 0.05625, max 0.05649)
         # T450   : 0.01473 +/- 0.00009  (min 0.01462, max 0.01483)
         # F340450: 0.16686 +/- 0.00006  (min 0.16680, max 0.16695)
         # Fgt480 : 0.00238 +/- 0.00004  (min 0.00235, max 0.00244)

         "posw_clip_t0": trial.suggest_float("posw_clip_t0", 12.0, 28.0, log=True),
         "posw_clip_t1": trial.suggest_float("posw_clip_t1", 35.0, 90.0, log=True),
         "posw_clip_t2": trial.suggest_float("posw_clip_t2", 3.0, 10.0, log=True),
         "posw_clip_t3": trial.suggest_float("posw_clip_t3", 90.0, 220.0, log=True),
         "gamma_t0": trial.suggest_float("gamma_t0", 0.5, 2.0), "gamma_t1": trial.suggest_float("gamma_t1", 1.0, 3.0),
         "gamma_t2": trial.suggest_float("gamma_t2", 0.0, 1.5), "gamma_t3": trial.suggest_float("gamma_t3", 1.5, 4.0),
         "rare_oversample_mult": trial.suggest_float("rare_oversample_mult", 2.0, 10.0),
         "rare_target_prev": trial.suggest_float("rare_target_prev", 0.06, 0.12),
         "sample_weight_cap": trial.suggest_float("sample_weight_cap", 6.0, 9.0),
         "lam_t0": trial.suggest_float("lam_t0", 0.6, 1.6, log=True),
         "lam_t1": trial.suggest_float("lam_t1", 1.0, 2.4, log=True),
         "lam_t2": trial.suggest_float("lam_t2", 0.25, 0.9, log=True),
         "lam_t3": trial.suggest_float("lam_t3", 1.8, 3.5, log=True),

        "lam_floor": trial.suggest_float("lam_floor", 0.35, 0.85),
        "lam_ceil": trial.suggest_float("lam_ceil", 1.30, 2.20),
        "lambda_aux_abs": trial.suggest_float("lambda_aux_abs", 0.05, 0.5),
        "lambda_aux_fluo": trial.suggest_float("lambda_aux_fluo", 0.05, 0.5),
        "lambda_aux_bitmask": trial.suggest_float("lambda_aux_bitmask", 0.02, 0.08),
        "reg_loss_type": trial.suggest_categorical("reg_loss_type", ["mse"]),
         "min_w": trial.suggest_float("min_w", 0.1, 0.6),
         "accumulate_grad_batches": trial.suggest_categorical("accumulate_grad_batches", [8, 16]),
         "head_stochastic_depth": trial.suggest_float("head_stochastic_depth", 0.0, 0.1)}

    if int(p["inst_hidden"]) % int(p["attn_heads"]) != 0:
        raise optuna.TrialPruned("inst_hidden must be divisible by attn_heads")
    return p


__all__ = ["search_space"]
