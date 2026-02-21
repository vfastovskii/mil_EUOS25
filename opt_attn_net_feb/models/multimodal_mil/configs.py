from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MILBackboneConfig:
    mol_dim: int
    inst_dim: int
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


@dataclass(frozen=True)
class MILLossConfig:
    lambda_aux_abs: float = 0.05
    lambda_aux_fluo: float = 0.05
    lambda_aux_bitmask: float = 0.05
    reg_loss_type: str = "mse"
    bitmask_group_top_ids: Optional[List[int]] = None
    bitmask_group_class_weight: Optional[List[float]] = None


@dataclass(frozen=True)
class MILModelConfig:
    backbone: MILBackboneConfig
    predictor: MILPredictorConfig
    optimization: MILOptimizationConfig
    loss: MILLossConfig


__all__ = [
    "MILBackboneConfig",
    "MILPredictorConfig",
    "MILOptimizationConfig",
    "MILLossConfig",
    "MILModelConfig",
]
