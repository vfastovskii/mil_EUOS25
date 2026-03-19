from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.multimodal_mil import MILTaskAttnMixerWithAux
from ..models.multimodal_mil.configs import (
    MILBackboneConfig,
    MILModelConfig,
    MILOptimizationConfig,
    MILLossConfig,
    MILPredictorConfig,
)
from .configs import HPOConfig


@dataclass(frozen=True)
class LoaderConfig:
    """
    Configuration class for a data loader.

    This class defines the configuration parameters for a data loader, including the number of
    workers and whether to enable pinned memory for data transfer.

    Attributes:
        num_workers: Number of worker threads to use for loading data.
        pin_memory: Boolean indicating whether to enable pinned memory for faster host-to-device
        data transfer.
    """
    num_workers: int
    pin_memory: bool

class MILModelBuilder:
    """
    Class responsible for building Multiple Instance Learning (MIL) models using a task-specific
    Attention Mixer architecture with auxiliary configurations.

    This class provides functionality to construct an MIL model based on the specified
    hyperparameter optimization (HPO) configurations. The assembled model is suitable
    for learning tasks involving molecular and instance-level embeddings, along with
    auxiliary predictors and various optimization settings.

    It supports static model configuration generation, leveraging input parameters and
    predefined structural setups from the provided HPO configurations.

    Static Methods:
        build: Constructs an MIL model based on input configurations, dimensions, and
               task-specific settings.

    """

    @staticmethod
    def build(
        *,
        config: HPOConfig,
        mol_dim: int,
        inst_dim: int,
        inst_geom_dim: int,
        inst_qm_dim: int,
        pos_weight: torch.Tensor,
        gamma: torch.Tensor,
        lam: np.ndarray,
        bitmask_group_top_ids: list[int] | None = None,
        bitmask_group_class_weight: np.ndarray | None = None,
    ) -> MILTaskAttnMixerWithAux:
        b = config.backbone
        h = config.heads
        opt = config.optimization
        loss = config.loss
        model_cfg = MILModelConfig(
            backbone=MILBackboneConfig(
                mol_dim=int(mol_dim),
                inst_dim=int(inst_dim),
                inst_geom_dim=int(inst_geom_dim),
                inst_qm_dim=int(inst_qm_dim),
                mol_hidden=int(b.mol_hidden),
                mol_layers=int(b.mol_layers),
                mol_dropout=float(b.mol_dropout),
                inst_hidden=int(b.inst_hidden),
                inst_layers=int(b.inst_layers),
                inst_dropout=float(b.inst_dropout),
                proj_dim=int(b.proj_dim),
                attn_heads=int(b.attn_heads),
                attn_dropout=float(b.attn_dropout),
                mixer_hidden=int(b.mixer_hidden),
                mixer_layers=int(b.mixer_layers),
                mixer_dropout=float(b.mixer_dropout),
                activation=str(b.activation),
                mol_embedder_name=str(b.mol_embedder_name),
                inst_embedder_name=str(b.inst_embedder_name),
                aggregator_name=str(b.aggregator_name),
                fusion_use_task_2d_adapter=bool(b.fusion_use_task_2d_adapter),
                fusion_use_modality_gates=bool(b.fusion_use_modality_gates),
                fusion_use_modality_interaction=bool(b.fusion_use_modality_interaction),
                fusion_gate_hidden=(
                    None if b.fusion_gate_hidden is None else int(b.fusion_gate_hidden)
                ),
                fusion_interaction_heads=int(b.fusion_interaction_heads),
            ),
            predictor=MILPredictorConfig(
                predictor_name=str(b.predictor_name),
                num_layers=int(h.num_layers),
                dropout=float(h.dropout),
                stochastic_depth=float(h.stochastic_depth),
                fc2_gain_non_last=float(h.fc2_gain_non_last),
            ),
            optimization=MILOptimizationConfig(
                lr=float(opt.lr),
                weight_decay=float(opt.weight_decay),
                lr_scale_2d=float(opt.lr_scale_2d),
                lr_scale_3d=float(opt.lr_scale_3d),
                lr_scale_fusion=float(opt.lr_scale_fusion),
                lr_scale_heads=float(opt.lr_scale_heads),
                weight_decay_scale_2d=float(opt.weight_decay_scale_2d),
                weight_decay_scale_3d=float(opt.weight_decay_scale_3d),
                weight_decay_scale_fusion=float(opt.weight_decay_scale_fusion),
                weight_decay_scale_heads=float(opt.weight_decay_scale_heads),
                stage_2d_only_epochs=int(opt.stage_2d_only_epochs),
                stage_3d_only_epochs=int(opt.stage_3d_only_epochs),
            ),
            loss=MILLossConfig(
                lambda_aux_abs=float(loss.lambda_aux_abs),
                lambda_aux_fluo=float(loss.lambda_aux_fluo),
                lambda_aux_bitmask=float(loss.lambda_aux_bitmask),
                lambda_contrastive_cross_modal=float(loss.lambda_contrastive_cross_modal),
                lambda_contrastive_3d_consistency=float(loss.lambda_contrastive_3d_consistency),
                lambda_contrastive_supervised=float(loss.lambda_contrastive_supervised),
                contrastive_proj_dim=int(loss.contrastive_proj_dim),
                contrastive_temperature=float(loss.contrastive_temperature),
                consistency_view_keep_rate=float(loss.consistency_view_keep_rate),
                cross_modal_include_geom_qm=bool(loss.cross_modal_include_geom_qm),
                reg_loss_type=str(loss.reg_loss_type),
                bitmask_group_top_ids=(
                    [int(x) for x in bitmask_group_top_ids]
                    if bitmask_group_top_ids is not None
                    else None
                ),
                bitmask_group_class_weight=(
                    [float(x) for x in bitmask_group_class_weight.tolist()]
                    if bitmask_group_class_weight is not None
                    else None
                ),
            ),
            objective_mode=str(config.objective.mode),
            objective_min_w=float(config.objective.min_w),
        )
        return MILTaskAttnMixerWithAux.from_config(
            config=model_cfg,
            pos_weight=pos_weight,
            gamma=gamma,
            lam=lam,
        )


class DataLoaderBuilder:
    """
    A builder class for creating DataLoader instances.

    This class simplifies the process of configuring and creating DataLoader
    instances for training and evaluation datasets. It uses the provided
    configuration to set up optimal DataLoader parameters and provides methods
    to create DataLoaders for training and evaluation purposes.
    """

    def __init__(self, cfg: LoaderConfig):
        self.cfg = cfg

    def _loader_kwargs(self) -> Dict[str, Any]:
        nw = int(self.cfg.num_workers)
        kw: Dict[str, Any] = {
            "num_workers": nw,
            "pin_memory": bool(self.cfg.pin_memory),
            "persistent_workers": (nw > 0),
        }
        if nw > 0:
            kw["prefetch_factor"] = 2
        return kw

    def train_loader(
        self,
        dataset,
        *,
        batch_size: int,
        sampler=None,
        batch_sampler=None,
        collate_fn=None,
    ):
        kw = self._loader_kwargs()
        if batch_sampler is not None:
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **kw,
            )
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate_fn,
            **kw,
        )

    def eval_loader(self, dataset, *, batch_size: int, collate_fn):
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )


__all__ = [
    "LoaderConfig",
    "MILModelBuilder",
    "DataLoaderBuilder",
]
