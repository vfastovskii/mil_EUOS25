from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BackboneConfig:
    """
    Configuration class for the Backbone model.

    This class provides a structured and immutable configuration for defining
    the hyperparameters and settings of the Backbone model. It is designed
    to manage the model's layer dimensions, dropout rates, activation
    functions, embedder names, and other components required during
    initialization or training of the model.

    Attributes:
        mol_hidden: Integer specifying the hidden size for molecular layers.
        mol_layers: Integer specifying the number of layers for molecular
            embeddings.
        mol_dropout: Float specifying the dropout rate for molecular layers.
        inst_hidden: Integer specifying the hidden size for instance layers.
        inst_layers: Integer specifying the number of layers for instance
            embeddings.
        inst_dropout: Float specifying the dropout rate for instance layers.
        proj_dim: Integer specifying the dimensionality of the projection
            layer.
        attn_heads: Integer specifying the number of attention heads.
        attn_dropout: Float specifying the dropout rate for attention layers.
        mixer_hidden: Integer specifying the hidden size for the mixer layers.
        mixer_layers: Integer specifying the number of mixer layers.
        mixer_dropout: Float specifying the dropout rate for mixer layers.
        activation: String specifying the activation function used throughout
            the model.
        mol_embedder_name: String specifying the name of the embedding method
            used for molecular inputs.
        inst_embedder_name: String specifying the name of the embedding method
            used for instance inputs.
        aggregator_name: String specifying the name of the aggregation method
            used in the model.
        predictor_name: String specifying the name of the predictor method
            used in the output layers.
    """
    mol_hidden: int = 1024
    mol_layers: int = 2
    mol_dropout: float = 0.10
    inst_hidden: int = 256
    inst_layers: int = 3
    inst_dropout: float = 0.05
    proj_dim: int = 512
    attn_heads: int = 8
    attn_dropout: float = 0.05
    mixer_hidden: int = 512
    mixer_layers: int = 3
    mixer_dropout: float = 0.05
    activation: str = "GELU"
    mol_embedder_name: str = "mlp_v3_2d"
    inst_embedder_name: str = "mlp_v3_3d"
    aggregator_name: str = "task_attention_pool"
    predictor_name: str = "mlp_v3"


@dataclass(frozen=True)
class HeadConfig:
    """
    Configuration for the head of a model.

    This class provides various parameters that define the structure and behavior
    of the model's head. It uses a dataclass to ensure immutability and simplifies
    the creation and management of configuration objects.

    Attributes:
        num_layers: Number of layers in the head.
        dropout: Dropout probability for the layers.
        stochastic_depth: Probability of stochastic depth in the layers.
        fc2_gain_non_last: Gain value applied to the second fully-connected
            layer in non-last layers.
    """
    num_layers: int = 2
    dropout: float = 0.1
    stochastic_depth: float = 0.1
    fc2_gain_non_last: float = 1e-2


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Representation of an optimization configuration for machine learning models.

    This class encapsulates parameters for configuring the optimization process,
    including learning rate (`lr`) and weight decay. It is used to store these
    parameters in an immutable way to ensure consistency throughout the training
    pipeline.
    """
    lr: float = 8e-5
    weight_decay: float = 3e-6


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Encapsulates immutable runtime configuration settings.

    This dataclass is used to store various configuration settings
    associated with the runtime of an application or model training,
    such as batch size and gradient accumulation steps. The frozen
    property ensures the immutability of its instances after creation.

    Attributes:
        batch_size: The size of each batch to process during training or
            runtime.
        accumulate_grad_batches: Number of gradient accumulation steps.
    """
    batch_size: int = 128
    accumulate_grad_batches: int = 8


@dataclass(frozen=True)
class SamplerConfig:
    """
    Represents configuration settings for a sampler.

    This data class provides parameters to manage sampler configurations,
    focusing on controlling the weights and thresholds for sampling data.
    It is designed to be immutable due to the `frozen` attribute.
    """
    rare_oversample_mult: float = 0.0
    rare_target_prev: float = 0.10
    rare_prev_thr: float | None = None
    sample_weight_cap: float = 10.0
    use_balanced_batch_sampler: bool = True
    batch_pos_fraction: float = 0.35
    min_pos_per_batch: int = 1
    enforce_bitmask_quota: bool = True
    quota_t450_per_256: int = 4
    quota_fgt480_per_256: int = 1
    quota_multi_per_256: int = 8
    use_bitmask_loss_weight: bool = True
    bitmask_weight_alpha: float = 0.5
    bitmask_weight_cap: float = 3.0


@dataclass(frozen=True)
class LossWeightingConfig:
    """
    Configuration used for loss weighting in a model.

    This dataclass contains parameters for adjusting the weightings of
    various tasks and related configuration options. It includes bounds
    for lambda coefficients, clipping values for positive weights,
    and auxiliary loss components. The class can also generate specific
    configuration tuples that simplify access to parameter groupings.

    Attributes:
        lam_t0: Weighting coefficient for task 0. Defaults to None.
        lam_t1: Weighting coefficient for task 1. Defaults to None.
        lam_t2: Weighting coefficient for task 2. Defaults to None.
        lam_t3: Weighting coefficient for task 3. Defaults to None.
        lam_floor: Minimum bounding value for lambda coefficients. Defaults to 0.25.
        lam_ceil: Maximum bounding value for lambda coefficients. Defaults to 3.5.
        lambda_power: Exponent applied to lambda calculations. Defaults to 1.0.
        posw_clip_t0: Clip threshold for task 0 positive weights. Defaults to None.
        posw_clip_t1: Clip threshold for task 1 positive weights. Defaults to None.
        posw_clip_t2: Clip threshold for task 2 positive weights. Defaults to None.
        posw_clip_t3: Clip threshold for task 3 positive weights. Defaults to None.
        pos_weight_clip: Global positive weight clipping value. Defaults to 50.0.
        gamma_t0: Parameter representing a gamma adjustment for task 0. Defaults to 0.0.
        gamma_t1: Parameter representing a gamma adjustment for task 1. Defaults to 0.0.
        gamma_t2: Parameter representing a gamma adjustment for task 2. Defaults to 0.0.
        gamma_t3: Parameter representing a gamma adjustment for task 3. Defaults to 0.0.
        lambda_aux_abs: Auxiliary absolute loss weighting. Defaults to 0.05.
        lambda_aux_fluo: Auxiliary fluorescence loss weighting. Defaults to 0.05.
        reg_loss_type: Type of regression loss to be used. Defaults to 'mse'.

    Methods:
        per_task_lam:
            Retrieves a tuple of lambda weights for each task if all tasks have
            defined values. Returns None if any task lambda is undefined.

        per_task_posw_clips:
            Retrieves a tuple of positive weight clip thresholds for each task
            if all tasks have defined values. Returns None if any task threshold is undefined.

        from_params:
            A factory method for creating a LossWeightingConfig instance from
            a dictionary of parameters. Supports default fallback values for
            missing attributes.

    Raises:
        ValueError: Raised by the methods if inconsistent parameter values
        are calculated or invalid configurations are provided by input mapping.
    """
    lam_t0: float | None = None
    lam_t1: float | None = None
    lam_t2: float | None = None
    lam_t3: float | None = None
    lam_floor: float = 0.25
    lam_ceil: float = 3.5
    lambda_power: float = 1.0
    posw_clip_t0: float | None = None
    posw_clip_t1: float | None = None
    posw_clip_t2: float | None = None
    posw_clip_t3: float | None = None
    pos_weight_clip: float = 50.0
    gamma_t0: float = 0.0
    gamma_t1: float = 0.0
    gamma_t2: float = 0.0
    gamma_t3: float = 0.0
    lambda_aux_abs: float = 0.05
    lambda_aux_fluo: float = 0.05
    lambda_aux_bitmask: float = 0.05
    bitmask_group_top_k: int = 6
    bitmask_group_weight_alpha: float = 0.5
    bitmask_group_weight_cap: float = 5.0
    reg_loss_type: str = "mse"

    def per_task_lam(self) -> tuple[float, float, float, float] | None:
        vals = (self.lam_t0, self.lam_t1, self.lam_t2, self.lam_t3)
        if all(v is not None for v in vals):
            return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
        return None

    def per_task_posw_clips(self) -> tuple[float, float, float, float] | None:
        vals = (self.posw_clip_t0, self.posw_clip_t1, self.posw_clip_t2, self.posw_clip_t3)
        if all(v is not None for v in vals):
            return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
        return None

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any],
        *,
        fallback_lambda_power: float = 1.0,
        fallback_lam_floor: float = 0.25,
        fallback_lam_ceil: float = 3.5,
        fallback_pos_weight_clip: float = 50.0,
    ) -> LossWeightingConfig:
        return cls(
            lam_t0=(float(params["lam_t0"]) if "lam_t0" in params else None),
            lam_t1=(float(params["lam_t1"]) if "lam_t1" in params else None),
            lam_t2=(float(params["lam_t2"]) if "lam_t2" in params else None),
            lam_t3=(float(params["lam_t3"]) if "lam_t3" in params else None),
            lam_floor=float(params.get("lam_floor", fallback_lam_floor)),
            lam_ceil=float(params.get("lam_ceil", fallback_lam_ceil)),
            lambda_power=float(params.get("lambda_power", fallback_lambda_power)),
            posw_clip_t0=(float(params["posw_clip_t0"]) if "posw_clip_t0" in params else None),
            posw_clip_t1=(float(params["posw_clip_t1"]) if "posw_clip_t1" in params else None),
            posw_clip_t2=(float(params["posw_clip_t2"]) if "posw_clip_t2" in params else None),
            posw_clip_t3=(float(params["posw_clip_t3"]) if "posw_clip_t3" in params else None),
            pos_weight_clip=float(params.get("pos_weight_clip", fallback_pos_weight_clip)),
            gamma_t0=float(params.get("gamma_t0", 0.0)),
            gamma_t1=float(params.get("gamma_t1", 0.0)),
            gamma_t2=float(params.get("gamma_t2", 0.0)),
            gamma_t3=float(params.get("gamma_t3", 0.0)),
            lambda_aux_abs=float(params.get("lambda_aux_abs", 0.05)),
            lambda_aux_fluo=float(params.get("lambda_aux_fluo", 0.05)),
            lambda_aux_bitmask=float(params.get("lambda_aux_bitmask", 0.05)),
            bitmask_group_top_k=int(params.get("bitmask_group_top_k", 6)),
            bitmask_group_weight_alpha=float(params.get("bitmask_group_weight_alpha", 0.5)),
            bitmask_group_weight_cap=float(params.get("bitmask_group_weight_cap", 5.0)),
            reg_loss_type=str(params.get("reg_loss_type", "mse")),
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    """
    Data class for configuring objectives.

    Represents configuration parameters for setting up objectives. Provides options
    to define the objective calculation mode and associated weights.

    Attributes:
        mode: Mode for calculating objectives, with a default value of
              "macro_plus_min".
        min_w: Minimum weight for objectives, with a default value of 0.30.
    """
    mode: str = "macro_plus_min"
    min_w: float = 0.30


@dataclass(frozen=True)
class HPOConfig:
    """
    Represents the configuration for hyperparameter optimization (HPO).

    The HPOConfig class is designed to encapsulate all the necessary configurations
    required for conducting hyperparameter optimization in a machine learning
    or related workflow. It gathers configurations for the model's backbone, heads,
    optimization, runtime, sampler settings, loss weighting, and objective strategies.
    This structure aids in maintaining consistency, modularity, and clarity for
    managing extensive parameter profiles, enabling robust and reproducible
    experimental setups.

    Attributes:
        backbone (BackboneConfig): Configuration of the model's backbone, including
                                   hidden layers, dropout rates, attention heads,
                                   and embedder names.
        heads (HeadConfig): Configuration of the model's head layers, such as number
                            of layers, dropout, stochastic depth, and gain factors.
        optimization (OptimizationConfig): Settings for optimization, covering
                                           learning rate and weight decay.
        runtime (RuntimeConfig): Runtime configurations including batch size and
                                 gradient accumulation.
        sampler (SamplerConfig): Configuration related to data sampling, including
                                  oversampling multiplier and sample weight caps.
        loss (LossWeightingConfig): Settings for managing and weighting loss values.
        objective (ObjectiveConfig): Objectives for optimization, specifying the
                                     optimization mode and its constraints.

    Methods:
        from_params: Class method to initialize an HPOConfig object from a given
                     set of parameters. It allows for flexible fallback values
                     and validates specific configurations.
    """
    backbone: BackboneConfig
    heads: HeadConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    loss: LossWeightingConfig
    objective: ObjectiveConfig

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any],
        *,
        fallback_lambda_power: float = 1.0,
        fallback_lam_floor: float = 0.25,
        fallback_lam_ceil: float = 3.5,
        fallback_pos_weight_clip: float = 50.0,
        fallback_min_w: float = 0.30,
    ) -> HPOConfig:
        head_num_layers = int(params.get("head_num_layers", 2))
        head_stochastic_depth = float(
            params.get("head_stochastic_depth", 0.0 if head_num_layers == 1 else 0.1)
        )

        backbone = BackboneConfig(
            mol_hidden=int(params.get("mol_hidden", 1024)),
            mol_layers=int(params.get("mol_layers", 2)),
            mol_dropout=float(params.get("mol_dropout", 0.10)),
            inst_hidden=int(params.get("inst_hidden", 256)),
            inst_layers=int(params.get("inst_layers", 3)),
            inst_dropout=float(params.get("inst_dropout", 0.05)),
            proj_dim=int(params.get("proj_dim", 512)),
            attn_heads=int(params.get("attn_heads", 8)),
            attn_dropout=float(params.get("attn_dropout", 0.05)),
            mixer_hidden=int(params.get("mixer_hidden", 512)),
            mixer_layers=int(params.get("mixer_layers", 3)),
            mixer_dropout=float(params.get("mixer_dropout", 0.05)),
            activation=str(params.get("activation", "GELU")),
            mol_embedder_name=str(params.get("mol_embedder_name", "mlp_v3_2d")),
            inst_embedder_name=str(params.get("inst_embedder_name", "mlp_v3_3d")),
            aggregator_name=str(params.get("aggregator_name", "task_attention_pool")),
            predictor_name=str(params.get("predictor_name", "mlp_v3")),
        )

        heads = HeadConfig(
            num_layers=head_num_layers,
            dropout=float(params.get("head_dropout", 0.1)),
            stochastic_depth=head_stochastic_depth,
            fc2_gain_non_last=float(params.get("head_fc2_gain_non_last", 1e-2)),
        )

        optimization = OptimizationConfig(
            lr=float(params.get("lr", 8e-5)),
            weight_decay=float(params.get("weight_decay", 3e-6)),
        )

        runtime = RuntimeConfig(
            batch_size=int(params.get("batch_size", 128)),
            accumulate_grad_batches=int(params.get("accumulate_grad_batches", 8)),
        )

        sampler = SamplerConfig(
            rare_oversample_mult=float(params.get("rare_oversample_mult", 0.0)),
            rare_target_prev=float(params.get("rare_target_prev", 0.10)),
            rare_prev_thr=(
                float(params["rare_prev_thr"]) if "rare_prev_thr" in params else None
            ),
            sample_weight_cap=float(params.get("sample_weight_cap", 10.0)),
            use_balanced_batch_sampler=bool(params.get("use_balanced_batch_sampler", True)),
            batch_pos_fraction=float(params.get("batch_pos_fraction", 0.35)),
            min_pos_per_batch=int(params.get("min_pos_per_batch", 1)),
            enforce_bitmask_quota=bool(params.get("enforce_bitmask_quota", True)),
            quota_t450_per_256=int(params.get("quota_t450_per_256", 4)),
            quota_fgt480_per_256=int(params.get("quota_fgt480_per_256", 1)),
            quota_multi_per_256=int(params.get("quota_multi_per_256", 8)),
            use_bitmask_loss_weight=bool(params.get("use_bitmask_loss_weight", True)),
            bitmask_weight_alpha=float(params.get("bitmask_weight_alpha", 0.5)),
            bitmask_weight_cap=float(params.get("bitmask_weight_cap", 3.0)),
        )

        loss = LossWeightingConfig.from_params(
            params,
            fallback_lambda_power=fallback_lambda_power,
            fallback_lam_floor=fallback_lam_floor,
            fallback_lam_ceil=fallback_lam_ceil,
            fallback_pos_weight_clip=fallback_pos_weight_clip,
        )

        objective = ObjectiveConfig(
            # Fixed objective policy: optimize mean AP while preventing weak-task neglect.
            mode="macro_plus_min",
            min_w=float(params.get("min_w", fallback_min_w)),
        )

        if backbone.inst_hidden % max(1, backbone.attn_heads) != 0:
            raise ValueError(
                f"Invalid config: inst_hidden={backbone.inst_hidden} must be divisible by attn_heads={backbone.attn_heads}"
            )

        return cls(
            backbone=backbone,
            heads=heads,
            optimization=optimization,
            runtime=runtime,
            sampler=sampler,
            loss=loss,
            objective=objective,
        )


__all__ = [
    "BackboneConfig",
    "HeadConfig",
    "OptimizationConfig",
    "RuntimeConfig",
    "SamplerConfig",
    "LossWeightingConfig",
    "ObjectiveConfig",
    "HPOConfig",
]
