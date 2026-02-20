from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BackboneConfig:
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
    num_layers: int = 2
    dropout: float = 0.1
    stochastic_depth: float = 0.1
    fc2_gain_non_last: float = 1e-2


@dataclass(frozen=True)
class OptimizationConfig:
    lr: float = 8e-5
    weight_decay: float = 3e-6


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int = 128
    accumulate_grad_batches: int = 8


@dataclass(frozen=True)
class SamplerConfig:
    rare_oversample_mult: float = 0.0
    rare_prev_thr: float = 0.02
    sample_weight_cap: float = 10.0


@dataclass(frozen=True)
class LossWeightingConfig:
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
            reg_loss_type=str(params.get("reg_loss_type", "mse")),
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: str = "macro_plus_min"
    min_w: float = 0.30


@dataclass(frozen=True)
class HPOConfig:
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
            rare_prev_thr=float(params.get("rare_prev_thr", 0.02)),
            sample_weight_cap=float(params.get("sample_weight_cap", 10.0)),
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
