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
    p = {
        # Dimension-capped search: keep effective layer widths <= 1024.
        # (With V3 gated FFNs, internal width is ~4x hidden_dim.)
        "mol_hidden": trial.suggest_categorical("mol_hidden", [128, 256]),
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
        # Component selection by name (extensible registries)
        "mol_embedder_name": trial.suggest_categorical("mol_embedder_name", ["mlp_v3_2d"]),
        "inst_embedder_name": trial.suggest_categorical("inst_embedder_name", ["mlp_v3_3d"]),
        "aggregator_name": trial.suggest_categorical("aggregator_name", ["task_attention_pool"]),
        "predictor_name": trial.suggest_categorical("predictor_name", ["mlp_v3"]),
        # Shared predictor-head architecture knobs (applied to all heads)
        "head_num_layers": trial.suggest_categorical("head_num_layers", [2, 3, 4, 6]),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.2),
        "head_fc2_gain_non_last": trial.suggest_categorical("head_fc2_gain_non_last", [1e-3, 3e-3, 1e-2]),
        # Activation choice for encoders and mixer
        "activation": trial.suggest_categorical("activation", ["GELU", "SiLU", "Mish", "ReLU", "LeakyReLU"]),
        "lr": trial.suggest_float("lr", 8e-5, 8e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 3e-6, 3e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "posw_clip_t0": trial.suggest_float("posw_clip_t0", 10.0, 200.0, log=True),
        "posw_clip_t1": trial.suggest_float("posw_clip_t1", 10.0, 200.0, log=True),
        "posw_clip_t2": trial.suggest_float("posw_clip_t2", 10.0, 200.0, log=True),
        "posw_clip_t3": trial.suggest_float("posw_clip_t3", 10.0, 200.0, log=True),
        "gamma_t0": trial.suggest_float("gamma_t0", 0.0, 4.0),
        "gamma_t1": trial.suggest_float("gamma_t1", 0.0, 4.0),
        "gamma_t2": trial.suggest_float("gamma_t2", 0.0, 4.0),
        "gamma_t3": trial.suggest_float("gamma_t3", 0.0, 4.0),
        "rare_oversample_mult": trial.suggest_float("rare_oversample_mult", 0.0, 20.0),
        "rare_target_prev": trial.suggest_float("rare_target_prev", 0.03, 0.30),
        "sample_weight_cap": trial.suggest_float("sample_weight_cap", 5.0, 10.0),
        # Prior lambda_power was low; keep per-task lambda search close to neutral.
        "lam_t0": trial.suggest_float("lam_t0", 0.25, 3.0, log=True),
        "lam_t1": trial.suggest_float("lam_t1", 0.25, 3.0, log=True),
        "lam_t2": trial.suggest_float("lam_t2", 0.25, 3.0, log=True),
        "lam_t3": trial.suggest_float("lam_t3", 0.25, 3.0, log=True),
        "lam_floor": trial.suggest_float("lam_floor", 0.25, 1.0),
        "lam_ceil": trial.suggest_float("lam_ceil", 1.0, 3.5),
        "lambda_aux_abs": trial.suggest_float("lambda_aux_abs", 0.05, 0.5),
        "lambda_aux_fluo": trial.suggest_float("lambda_aux_fluo", 0.05, 0.5),
        "reg_loss_type": trial.suggest_categorical("reg_loss_type", ["mse"]),
        # HPO objective controls (fixed to macro_plus_min to avoid neglecting weaker tasks)
        "min_w": trial.suggest_float("min_w", 0.1, 0.6),
        "accumulate_grad_batches": trial.suggest_categorical("accumulate_grad_batches", [8, 16]),
    }
    p["head_stochastic_depth"] = trial.suggest_float("head_stochastic_depth", 0.0, 0.1)
    if int(p["inst_hidden"]) % int(p["attn_heads"]) != 0:
        raise optuna.TrialPruned("inst_hidden must be divisible by attn_heads")
    return p


__all__ = ["search_space"]
