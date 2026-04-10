from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MILBackboneConfig:
    mol_dim: int
    inst_dim: int
    inst_geom_dim: int
    inst_qm_dim: int
    mol_hidden: int
    mol_layers: int
    mol_dropout: float
    inst_hidden: int
    inst_layers: int
    inst_dropout: float
    proj_dim: int
    attn_heads: int
    attn_dropout: float
    mixer_hidden: int
    mixer_layers: int
    mixer_dropout: float
    activation: str = "GELU"
    mol_embedder_name: str = "mlp_v3_2d"
    inst_embedder_name: str = "mlp_v3_3d"
    aggregator_name: str = "task_attention_pool"
    aggregator_kwargs: Optional[Dict[str, Any]] = None
    fusion_use_task_2d_adapter: bool = True
    fusion_use_modality_gates: bool = True
    fusion_use_modality_interaction: bool = True
    fusion_gate_hidden: Optional[int] = None
    fusion_interaction_heads: int = 4


@dataclass(frozen=True)
class MILPredictorConfig:
    predictor_name: str = "mlp_v3"
    num_layers: int = 2
    dropout: float = 0.1
    stochastic_depth: float = 0.1
    fc2_gain_non_last: float = 1e-2


@dataclass(frozen=True)
class MILOptimizationConfig:
    lr: float = 8e-5
    weight_decay: float = 3e-6
    lr_scale_2d: float = 1.0
    lr_scale_3d: float = 1.0
    lr_scale_fusion: float = 1.0
    lr_scale_heads: float = 1.0
    weight_decay_scale_2d: float = 1.0
    weight_decay_scale_3d: float = 1.0
    weight_decay_scale_fusion: float = 1.0
    weight_decay_scale_heads: float = 1.0
    stage_2d_only_epochs: int = 0
    stage_3d_only_epochs: int = 0
    multitask_gradient_mode: str = "pcgrad_shared"
    log_task_gradient_diagnostics: bool = True


@dataclass(frozen=True)
class MILLossConfig:
    lambda_aux_abs: float = 0.05
    lambda_aux_fluo: float = 0.05
    lambda_aux_bitmask: float = 0.05
    lambda_contrastive_cross_modal: float = 0.05
    lambda_contrastive_3d_consistency: float = 0.05
    lambda_contrastive_supervised: float = 0.05
    contrastive_proj_dim: int = 64
    contrastive_temperature: float = 0.10
    consistency_view_keep_rate: float = 0.70
    cross_modal_include_geom_qm: bool = True
    learnable_task_uncertainty: bool = True
    task_uncertainty_init_log_var: float = 0.0
    task_uncertainty_reg: float = 0.5
    reg_loss_type: str = "mse"
    bitmask_group_top_ids: Optional[List[int]] = None
    bitmask_group_class_weight: Optional[List[float]] = None


@dataclass(frozen=True)
class MILModelConfig:
    backbone: MILBackboneConfig
    predictor: MILPredictorConfig
    optimization: MILOptimizationConfig
    loss: MILLossConfig
    objective_mode: str = "macro_plus_min"
    objective_min_w: float = 0.40


__all__ = [
    "MILBackboneConfig",
    "MILPredictorConfig",
    "MILOptimizationConfig",
    "MILLossConfig",
    "MILModelConfig",
]
