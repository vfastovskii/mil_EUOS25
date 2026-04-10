from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ...losses.multi_task_focal import MultiTaskFocal
from ...utils.metrics import ap_per_task
from ...utils.progress import log_event
from .aggregators import build_aggregator
from .configs import MILModelConfig
from .constants import NUM_ABS_HEADS, NUM_FLUO_HEADS, NUM_TASKS
from .embedder_mlp_v3_base import build_mlp_v3_embedder
from .embedders import build_2d_embedder, build_3d_embedder
from .head_mlp_v3 import MLPPredictorV3Like
from .head_utils import apply_shared_heads, apply_task_heads, make_projection
from .predictors import build_predictor_heads
from .training import compute_training_losses


def _make_activation_module(name: str) -> nn.Module:
    n = str(name).strip().lower()
    if n == "gelu":
        return nn.GELU()
    if n in {"relu", "leakyrelu", "leaky_relu"}:
        return nn.ReLU()
    return nn.SiLU()


class MILTaskAttnMixerWithAux(pl.LightningModule):
    """
    - 2D embedder -> e2d (no aggregator)
    - 3D geometry embedder -> tokens -> geometry aggregator
    - 3D quantum embedder -> tokens -> quantum aggregator
    - project 2D/3D-geom/3D-qm to same dim
    - make 2D task-aware before fusion
    - apply explicit per-task modality gates
    - run a tiny modality interaction block
    - flatten modality summaries and pass through mixer -> z_task
    - cls logits from task-specific z_task
    - aux heads from mean(z_task)
    """

    @staticmethod
    def _build_head_group(
        *,
        predictor_name: str,
        in_dim: int,
        count: int,
        activation: str,
        num_layers: int,
        dropout: float,
        stochastic_depth: float,
        fc2_gain_non_last: float,
    ) -> nn.ModuleList:
        return build_predictor_heads(
            name=str(predictor_name),
            in_dim=int(in_dim),
            count=int(count),
            activation=str(activation),
            num_layers=int(num_layers),
            dropout=float(dropout),
            stochastic_depth=float(stochastic_depth),
            fc2_gain_non_last=float(fc2_gain_non_last),
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: MILModelConfig,
        pos_weight: torch.Tensor,
        gamma: torch.Tensor,
        lam: np.ndarray,
    ) -> MILTaskAttnMixerWithAux:
        b = config.backbone
        h = config.predictor
        opt = config.optimization
        loss = config.loss
        return cls(
            mol_dim=int(b.mol_dim),
            inst_dim=int(b.inst_dim),
            inst_geom_dim=int(b.inst_geom_dim),
            inst_qm_dim=int(b.inst_qm_dim),
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
            multitask_gradient_mode=str(opt.multitask_gradient_mode),
            log_task_gradient_diagnostics=bool(opt.log_task_gradient_diagnostics),
            pos_weight=pos_weight,
            gamma=gamma,
            lam=lam,
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
            learnable_task_uncertainty=bool(loss.learnable_task_uncertainty),
            task_uncertainty_init_log_var=float(loss.task_uncertainty_init_log_var),
            task_uncertainty_reg=float(loss.task_uncertainty_reg),
            reg_loss_type=str(loss.reg_loss_type),
            bitmask_group_top_ids=(
                [int(x) for x in (loss.bitmask_group_top_ids or [])]
            ),
            bitmask_group_class_weight=(
                [float(x) for x in (loss.bitmask_group_class_weight or [])]
            ),
            activation=str(b.activation),
            mol_embedder_name=str(b.mol_embedder_name),
            inst_embedder_name=str(b.inst_embedder_name),
            aggregator_name=str(b.aggregator_name),
            aggregator_kwargs=b.aggregator_kwargs,
            fusion_use_task_2d_adapter=bool(b.fusion_use_task_2d_adapter),
            fusion_use_modality_gates=bool(b.fusion_use_modality_gates),
            fusion_use_modality_interaction=bool(b.fusion_use_modality_interaction),
            fusion_gate_hidden=(
                None if b.fusion_gate_hidden is None else int(b.fusion_gate_hidden)
            ),
            fusion_interaction_heads=int(b.fusion_interaction_heads),
            predictor_name=str(h.predictor_name),
            head_num_layers=int(h.num_layers),
            head_dropout=float(h.dropout),
            head_stochastic_depth=float(h.stochastic_depth),
            head_fc2_gain_non_last=float(h.fc2_gain_non_last),
            objective_mode=str(config.objective_mode),
            objective_min_w=float(config.objective_min_w),
        )

    def __init__(
        self,
        mol_dim: int,
        inst_dim: int,
        inst_geom_dim: int,
        inst_qm_dim: int,
        mol_hidden: int,
        mol_layers: int,
        mol_dropout: float,
        inst_hidden: int,
        inst_layers: int,
        inst_dropout: float,
        proj_dim: int,
        attn_heads: int,
        attn_dropout: float,
        mixer_hidden: int,
        mixer_layers: int,
        mixer_dropout: float,
        lr: float,
        weight_decay: float,
        lr_scale_2d: float,
        lr_scale_3d: float,
        lr_scale_fusion: float,
        lr_scale_heads: float,
        weight_decay_scale_2d: float,
        weight_decay_scale_3d: float,
        weight_decay_scale_fusion: float,
        weight_decay_scale_heads: float,
        stage_2d_only_epochs: int,
        stage_3d_only_epochs: int,
        multitask_gradient_mode: str,
        log_task_gradient_diagnostics: bool,
        pos_weight: torch.Tensor,
        gamma: torch.Tensor,
        lam: np.ndarray,
        lambda_aux_abs: float,
        lambda_aux_fluo: float,
        lambda_aux_bitmask: float,
        reg_loss_type: str,
        lambda_contrastive_cross_modal: float = 0.05,
        lambda_contrastive_3d_consistency: float = 0.05,
        lambda_contrastive_supervised: float = 0.05,
        contrastive_proj_dim: int = 64,
        contrastive_temperature: float = 0.10,
        consistency_view_keep_rate: float = 0.70,
        cross_modal_include_geom_qm: bool = True,
        learnable_task_uncertainty: bool = True,
        task_uncertainty_init_log_var: float = 0.0,
        task_uncertainty_reg: float = 0.5,
        bitmask_group_top_ids: Optional[List[int]] = None,
        bitmask_group_class_weight: Optional[List[float]] = None,
        activation: str = "GELU",
        mol_embedder_name: str = "mlp_v3_2d",
        inst_embedder_name: str = "mlp_v3_3d",
        aggregator_name: str = "task_attention_pool",
        aggregator_kwargs: Optional[Dict[str, Any]] = None,
        fusion_use_task_2d_adapter: bool = True,
        fusion_use_modality_gates: bool = True,
        fusion_use_modality_interaction: bool = True,
        fusion_gate_hidden: Optional[int] = None,
        fusion_interaction_heads: int = 4,
        predictor_name: str = "mlp_v3",
        head_num_layers: int = 2,
        head_dropout: float = 0.1,
        head_stochastic_depth: float = 0.1,
        head_fc2_gain_non_last: float = 1e-2,
        objective_mode: str = "macro_plus_min",
        objective_min_w: float = 0.40,
        **legacy_kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight", "gamma", "lam"])

        self.mol_dim = int(mol_dim)
        self.inst_dim = int(inst_dim)
        self.inst_geom_dim = int(inst_geom_dim)
        self.inst_qm_dim = int(inst_qm_dim)
        self.inst_hidden = int(inst_hidden)
        self.proj_dim = int(proj_dim)
        self.mixer_type = "mlp_v3"
        _ = legacy_kwargs
        self.objective_mode = str(objective_mode)
        self.objective_min_w = float(objective_min_w)
        self.multitask_gradient_mode = str(multitask_gradient_mode).strip().lower()
        if self.multitask_gradient_mode not in {"none", "pcgrad_shared"}:
            raise ValueError(
                f"Unsupported multitask_gradient_mode={multitask_gradient_mode!r}. "
                "Expected one of {'none', 'pcgrad_shared'}."
            )
        self.log_task_gradient_diagnostics = bool(log_task_gradient_diagnostics)
        self.learnable_task_uncertainty = bool(learnable_task_uncertainty)
        self.task_uncertainty_reg = float(max(0.0, task_uncertainty_reg))
        if self.learnable_task_uncertainty:
            self.task_loss_log_vars = nn.Parameter(
                torch.full((NUM_TASKS,), float(task_uncertainty_init_log_var), dtype=torch.float32)
            )
        else:
            self.register_parameter("task_loss_log_vars", None)
        self.automatic_optimization = bool(self.multitask_gradient_mode == "none")
        self.manual_accumulate_grad_batches = 1
        self.lr_group_scales = {
            "2d": float(max(1e-4, lr_scale_2d)),
            "3d": float(max(1e-4, lr_scale_3d)),
            "fusion": float(max(1e-4, lr_scale_fusion)),
            "heads": float(max(1e-4, lr_scale_heads)),
        }
        self.weight_decay_group_scales = {
            "2d": float(max(0.0, weight_decay_scale_2d)),
            "3d": float(max(0.0, weight_decay_scale_3d)),
            "fusion": float(max(0.0, weight_decay_scale_fusion)),
            "heads": float(max(0.0, weight_decay_scale_heads)),
        }
        self.stage_2d_only_epochs = int(max(0, stage_2d_only_epochs))
        self.stage_3d_only_epochs = int(max(0, stage_3d_only_epochs))
        self._stagewise_active = False
        self._optimizer_group_names: Tuple[str, ...] = tuple()
        self._group_2d_prefixes = (
            "mol_enc.",
            "mol_post_embed_norm.",
            "proj2d.",
            "task_2d_adapter.",
            "task_2d_post_norm.",
        )
        self._group_3d_prefixes = (
            "inst_geom_enc.",
            "inst_qm_enc.",
            "inst_geom_post_embed_norm.",
            "inst_qm_post_embed_norm.",
            "attn_pool_geom.",
            "attn_pool_qm.",
            "agg_geom_post_norm.",
            "agg_qm_post_norm.",
            "proj3d_geom.",
            "proj3d_qm.",
        )
        self._group_head_prefixes = (
            "cls_heads.",
            "abs_heads.",
            "fluo_heads.",
            "bitmask_head.",
            "contrastive_heads.",
            "task_contrastive_head.",
        )
        self._last_stage_trainability: Optional[Tuple[bool, bool]] = None
        use_2d = self.mol_dim > 0
        use_geom = self.inst_geom_dim > 0
        use_qm = self.inst_qm_dim > 0
        if not (use_2d or use_geom or use_qm):
            raise ValueError(
                "At least one modality must be enabled: "
                f"mol_dim={self.mol_dim} inst_geom_dim={self.inst_geom_dim} inst_qm_dim={self.inst_qm_dim}"
            )

        self.mol_enc = (
            None
            if not use_2d
            else build_2d_embedder(
                name=str(mol_embedder_name),
                input_dim=int(mol_dim),
                hidden_dim=int(mol_hidden),
                layers=int(mol_layers),
                dropout=float(mol_dropout),
                activation=str(activation),
            )
        )
        self.inst_geom_enc = (
            None
            if self.inst_geom_dim <= 0
            else build_3d_embedder(
                name=str(inst_embedder_name),
                input_dim=int(self.inst_geom_dim),
                hidden_dim=int(inst_hidden),
                layers=int(inst_layers),
                dropout=float(inst_dropout),
                activation=str(activation),
            )
        )
        self.inst_qm_enc = (
            None
            if self.inst_qm_dim <= 0
            else build_3d_embedder(
                name=str(inst_embedder_name),
                input_dim=int(self.inst_qm_dim),
                hidden_dim=int(inst_hidden),
                layers=int(inst_layers),
                dropout=float(inst_dropout),
                activation=str(activation),
            )
        )
        self.mol_post_embed_norm = (
            None if self.mol_enc is None else nn.LayerNorm(int(mol_hidden))
        )
        self.inst_geom_post_embed_norm = nn.LayerNorm(int(inst_hidden))
        self.inst_qm_post_embed_norm = nn.LayerNorm(int(inst_hidden))

        agg_kwargs = dict(aggregator_kwargs or {})
        overlap = {"dim", "n_heads", "dropout", "n_tasks"}.intersection(agg_kwargs.keys())
        if overlap:
            raise ValueError(f"aggregator_kwargs cannot override reserved keys: {sorted(overlap)}")
        self.attn_pool_geom = (
            None
            if self.inst_geom_enc is None
            else build_aggregator(
                name=str(aggregator_name),
                dim=int(inst_hidden),
                n_heads=int(attn_heads),
                dropout=float(attn_dropout),
                n_tasks=NUM_TASKS,
                **agg_kwargs,
            )
        )
        self.attn_pool_qm = (
            None
            if self.inst_qm_enc is None
            else build_aggregator(
                name=str(aggregator_name),
                dim=int(inst_hidden),
                n_heads=int(attn_heads),
                dropout=float(attn_dropout),
                n_tasks=NUM_TASKS,
                **agg_kwargs,
            )
        )
        self.agg_geom_post_norm = (
            None if self.attn_pool_geom is None else nn.LayerNorm(int(inst_hidden))
        )
        self.agg_qm_post_norm = (
            None if self.attn_pool_qm is None else nn.LayerNorm(int(inst_hidden))
        )

        self.proj2d = (
            None if self.mol_enc is None else make_projection(int(mol_hidden), int(proj_dim))
        )
        self.proj3d_geom = (
            None if self.attn_pool_geom is None else make_projection(int(inst_hidden), int(proj_dim))
        )
        self.proj3d_qm = (
            None if self.attn_pool_qm is None else make_projection(int(inst_hidden), int(proj_dim))
        )
        self.active_modalities = tuple(
            x for x, enabled in (
                ("2d", self.proj2d is not None),
                ("3d_geom", self.proj3d_geom is not None),
                ("3d_qm", self.proj3d_qm is not None),
            ) if enabled
        )
        self.modality_name_to_index = {
            str(name): int(i) for i, name in enumerate(self.active_modalities)
        }
        n_modalities = int(len(self.active_modalities))
        act_mod = _make_activation_module(str(activation))
        self.fusion_use_task_2d_adapter = bool(fusion_use_task_2d_adapter and self.proj2d is not None)
        self.fusion_use_modality_gates = bool(fusion_use_modality_gates)
        self.fusion_use_modality_interaction = bool(fusion_use_modality_interaction and n_modalities > 1)
        gate_hidden = (
            int(fusion_gate_hidden)
            if fusion_gate_hidden is not None
            else int(max(32, min(self.proj_dim, max(1, self.proj_dim // 2))))
        )
        self.task_2d_tokens = (
            None
            if not self.fusion_use_task_2d_adapter
            else nn.Parameter(torch.randn(NUM_TASKS, self.proj_dim) * 0.02)
        )
        self.task_2d_adapter = (
            None
            if not self.fusion_use_task_2d_adapter
            else nn.Sequential(
                nn.Linear(2 * self.proj_dim, self.proj_dim),
                _make_activation_module(str(activation)),
                nn.Dropout(float(mixer_dropout)),
                nn.Linear(self.proj_dim, self.proj_dim),
            )
        )
        self.task_2d_post_norm = (
            None if not self.fusion_use_task_2d_adapter else nn.LayerNorm(self.proj_dim)
        )
        self.modality_type_embeddings = nn.Parameter(
            torch.randn(max(1, n_modalities), self.proj_dim) * 0.02
        )
        self.modality_gate_net = nn.Sequential(
            nn.Linear(self.proj_dim, gate_hidden),
            act_mod,
            nn.Dropout(float(mixer_dropout)),
            nn.Linear(gate_hidden, 1),
        )
        interaction_heads = int(max(1, fusion_interaction_heads))
        if self.proj_dim % interaction_heads != 0:
            interaction_heads = 1
        self.fusion_interaction_heads = int(interaction_heads)
        self.modality_interaction_attn = (
            None
            if not self.fusion_use_modality_interaction
            else nn.MultiheadAttention(
                embed_dim=self.proj_dim,
                num_heads=self.fusion_interaction_heads,
                dropout=float(mixer_dropout),
                batch_first=True,
                bias=True,
            )
        )
        self.modality_interaction_dropout = (
            nn.Identity()
            if not self.fusion_use_modality_interaction
            else nn.Dropout(float(mixer_dropout))
        )
        self.modality_interaction_ln1 = (
            None if not self.fusion_use_modality_interaction else nn.LayerNorm(self.proj_dim)
        )
        self.modality_interaction_ffn = (
            None
            if not self.fusion_use_modality_interaction
            else nn.Sequential(
                nn.Linear(self.proj_dim, 2 * self.proj_dim),
                _make_activation_module(str(activation)),
                nn.Dropout(float(mixer_dropout)),
                nn.Linear(2 * self.proj_dim, self.proj_dim),
            )
        )
        self.modality_interaction_ln2 = (
            None if not self.fusion_use_modality_interaction else nn.LayerNorm(self.proj_dim)
        )
        self.lambda_contrastive_cross_modal = float(max(0.0, lambda_contrastive_cross_modal))
        self.lambda_contrastive_3d_consistency = float(max(0.0, lambda_contrastive_3d_consistency))
        self.lambda_contrastive_supervised = float(max(0.0, lambda_contrastive_supervised))
        self.contrastive_proj_dim = int(max(8, contrastive_proj_dim))
        self.contrastive_temperature = float(max(1e-4, contrastive_temperature))
        self.consistency_view_keep_rate = float(np.clip(consistency_view_keep_rate, 0.25, 1.0))
        self.cross_modal_include_geom_qm = bool(cross_modal_include_geom_qm)
        self.contrastive_heads = nn.ModuleDict(
            {
                str(name): nn.Sequential(
                    nn.Linear(self.proj_dim, self.proj_dim),
                    _make_activation_module(str(activation)),
                    nn.Dropout(float(mixer_dropout)),
                    nn.Linear(self.proj_dim, self.contrastive_proj_dim),
                )
                for name in self.active_modalities
            }
        )
        self.task_contrastive_head = nn.Sequential(
            nn.Linear(int(mixer_hidden), int(mixer_hidden)),
            _make_activation_module(str(activation)),
            nn.Dropout(float(mixer_dropout)),
            nn.Linear(int(mixer_hidden), self.contrastive_proj_dim),
        )
        mixer_in_dim = int(max(1, len(self.active_modalities)) * int(proj_dim))

        self.mixer = build_mlp_v3_embedder(
            input_dim=mixer_in_dim,
            hidden_dim=int(mixer_hidden),
            layers=int(mixer_layers),
            dropout=float(mixer_dropout),
            activation=str(activation),
        )
        self.mixer_post_norm = nn.LayerNorm(int(mixer_hidden))

        self.cls_heads = self._build_head_group(
            predictor_name=str(predictor_name),
            in_dim=int(mixer_hidden),
            count=NUM_TASKS,
            activation=str(activation),
            num_layers=int(head_num_layers),
            dropout=float(head_dropout),
            stochastic_depth=float(head_stochastic_depth),
            fc2_gain_non_last=float(head_fc2_gain_non_last),
        )
        self.abs_heads = self._build_head_group(
            predictor_name=str(predictor_name),
            in_dim=int(mixer_hidden),
            count=NUM_ABS_HEADS,
            activation=str(activation),
            num_layers=int(head_num_layers),
            dropout=float(head_dropout),
            stochastic_depth=float(head_stochastic_depth),
            fc2_gain_non_last=float(head_fc2_gain_non_last),
        )
        self.fluo_heads = self._build_head_group(
            predictor_name=str(predictor_name),
            in_dim=int(mixer_hidden),
            count=NUM_FLUO_HEADS,
            activation=str(activation),
            num_layers=int(head_num_layers),
            dropout=float(head_dropout),
            stochastic_depth=float(head_stochastic_depth),
            fc2_gain_non_last=float(head_fc2_gain_non_last),
        )

        self.lambda_aux_bitmask = float(lambda_aux_bitmask)
        top_ids = [int(x) for x in (bitmask_group_top_ids or [])]
        n_masks = int(1 << NUM_TASKS)
        top_ids = [m for m in top_ids if 0 <= m < n_masks]
        # Keep unique order and reserve one "other" group.
        seen = set()
        top_ids = [m for m in top_ids if not (m in seen or seen.add(m))]
        top_ids = top_ids[: max(0, n_masks - 1)]
        self.bitmask_group_top_ids = top_ids
        self.bitmask_num_groups = int(len(top_ids) + 1)

        if self.lambda_aux_bitmask > 0.0 and self.bitmask_num_groups >= 2:
            self.bitmask_head = MLPPredictorV3Like(
                input_dim=int(mixer_hidden),
                hidden_dim=int(mixer_hidden),
                num_layers=int(head_num_layers),
                expansion=2.0,
                activation=str(activation),
                use_glu=True,
                dropout=float(head_dropout),
                stochastic_depth=float(head_stochastic_depth),
                use_layernorm=True,
                pre_layer_norm=True,
                output_dim=int(self.bitmask_num_groups),
                input_layernorm=True,
                final_layernorm=False,
                res_scale_init=0.1,
                inner_multiple=64,
                head_dropout=0.0,
                output_bias=None,
                fc2_gain_non_last=float(head_fc2_gain_non_last),
                proj_gain=0.5,
            )
            mask_to_group = torch.full(
                (n_masks,),
                fill_value=int(self.bitmask_num_groups - 1),
                dtype=torch.long,
            )
            for g, mid in enumerate(top_ids):
                mask_to_group[int(mid)] = int(g)
            self.register_buffer("bitmask_mask_to_group", mask_to_group)

            cw = (
                torch.tensor([float(x) for x in bitmask_group_class_weight], dtype=torch.float32)
                if bitmask_group_class_weight is not None
                else torch.ones((self.bitmask_num_groups,), dtype=torch.float32)
            )
            if cw.numel() != self.bitmask_num_groups:
                cw = torch.ones((self.bitmask_num_groups,), dtype=torch.float32)
            self.register_buffer("bitmask_group_class_weight", cw)
        else:
            self.bitmask_head = None
            self.register_buffer(
                "bitmask_mask_to_group",
                torch.zeros((n_masks,), dtype=torch.long),
            )
            self.register_buffer("bitmask_group_class_weight", torch.ones((1,), dtype=torch.float32))

        self.cls_loss = MultiTaskFocal(pos_weight=pos_weight, gamma=gamma)
        self.register_buffer("lam", torch.tensor(lam, dtype=torch.float32))

        self.lambda_aux_abs = float(lambda_aux_abs)
        self.lambda_aux_fluo = float(lambda_aux_fluo)
        self.reg_loss_type = str(reg_loss_type)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self._val_p: List[np.ndarray] = []
        self._val_y: List[np.ndarray] = []
        self._val_w: List[np.ndarray] = []
        self._train_real_3d_with: int = 0
        self._train_real_3d_without: int = 0
        self._val_real_3d_with: int = 0
        self._val_real_3d_without: int = 0

        # Optional concept-guidance RL controls (configured externally for final runs).
        self.rl_enabled: bool = False
        self.rl_guidance_scale: float = 0.0
        self.rl_guidance_scale_max: float = 0.2
        self.rl_negative_penalty: float = 0.15
        self.rl_task_target_concepts: list[tuple[str, ...]] = [tuple() for _ in range(NUM_TASKS)]
        self.rl_target_conf_pairs_by_task: list[set[tuple[str, str]]] = [set() for _ in range(NUM_TASKS)]
        self.rl_target_mols_by_task: list[set[str]] = [set() for _ in range(NUM_TASKS)]

    def _parameter_group_name(self, param_name: str) -> str:
        if param_name == "task_loss_log_vars":
            return "heads"
        if param_name == "task_2d_tokens":
            return "2d"
        if param_name.startswith(self._group_2d_prefixes):
            return "2d"
        if param_name.startswith(self._group_3d_prefixes):
            return "3d"
        if param_name.startswith(self._group_head_prefixes):
            return "heads"
        return "fusion"

    def _stage_mode_for_epoch(self, epoch_idx: int) -> str:
        epoch = int(max(0, epoch_idx))
        has_2d = bool("2d" in self.active_modalities)
        has_3d = bool(("3d_geom" in self.active_modalities) or ("3d_qm" in self.active_modalities))
        if has_2d and has_3d and epoch < int(self.stage_2d_only_epochs):
            return "2d_only"
        if has_2d and has_3d and epoch < int(self.stage_2d_only_epochs + self.stage_3d_only_epochs):
            return "3d_only"
        return "joint"

    def _enabled_modalities_for_current_stage(self) -> Set[str]:
        if not bool(self._stagewise_active):
            return set(self.active_modalities)
        mode = self._stage_mode_for_epoch(int(getattr(self, "current_epoch", 0)))
        if mode == "2d_only":
            return {"2d"} & set(self.active_modalities)
        if mode == "3d_only":
            return {"3d_geom", "3d_qm"} & set(self.active_modalities)
        return set(self.active_modalities)

    def _apply_stage_trainability(self) -> None:
        if not bool(self._stagewise_active):
            enable_2d, enable_3d = True, True
        else:
            mode = self._stage_mode_for_epoch(int(getattr(self, "current_epoch", 0)))
            enable_2d = (mode != "3d_only")
            enable_3d = (mode != "2d_only")
        state = (bool(enable_2d), bool(enable_3d))
        if self._last_stage_trainability == state:
            return
        for name, param in self.named_parameters():
            group = self._parameter_group_name(str(name))
            if group == "2d":
                param.requires_grad = bool(enable_2d)
            elif group == "3d":
                param.requires_grad = bool(enable_3d)
            else:
                param.requires_grad = True
        self._last_stage_trainability = state

    def _named_parameters_by_group(self) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
        groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
            "2d": [],
            "3d": [],
            "fusion": [],
            "heads": [],
        }
        for name, param in self.named_parameters():
            groups[self._parameter_group_name(str(name))].append((str(name), param))
        return groups

    def get_training_debug_summary(self) -> Dict[str, Any]:
        return {
            "mol_dim": int(self.mol_dim),
            "inst_dim": int(self.inst_dim),
            "inst_geom_dim": int(self.inst_geom_dim),
            "inst_qm_dim": int(self.inst_qm_dim),
            "inst_hidden": int(self.inst_hidden),
            "proj_dim": int(self.proj_dim),
            "active_modalities": ",".join(str(x) for x in self.active_modalities) if len(self.active_modalities) > 0 else "none",
            "n_active_modalities": int(len(self.active_modalities)),
            "fusion_task_2d_adapter": bool(self.fusion_use_task_2d_adapter),
            "fusion_modality_gates": bool(self.fusion_use_modality_gates),
            "fusion_modality_interaction": bool(self.fusion_use_modality_interaction),
            "objective_mode": str(self.objective_mode),
            "objective_min_w": float(self.objective_min_w),
            "mixer_type": str(self.mixer_type),
            "multitask_gradient_mode": str(self.multitask_gradient_mode),
            "log_task_gradient_diagnostics": bool(self.log_task_gradient_diagnostics),
            "learnable_task_uncertainty": bool(self.learnable_task_uncertainty),
            "task_uncertainty_reg": float(self.task_uncertainty_reg),
            "lambda_aux_abs": float(self.lambda_aux_abs),
            "lambda_aux_fluo": float(self.lambda_aux_fluo),
            "lambda_aux_bitmask": float(self.lambda_aux_bitmask),
            "lambda_contrastive_cross_modal": float(self.lambda_contrastive_cross_modal),
            "lambda_contrastive_3d_consistency": float(self.lambda_contrastive_3d_consistency),
            "lambda_contrastive_supervised": float(self.lambda_contrastive_supervised),
            "contrastive_proj_dim": int(self.contrastive_proj_dim),
            "contrastive_temperature": float(self.contrastive_temperature),
            "consistency_view_keep_rate": float(self.consistency_view_keep_rate),
            "cross_modal_include_geom_qm": bool(self.cross_modal_include_geom_qm),
            "stage_2d_only_epochs": int(self.stage_2d_only_epochs),
            "stage_3d_only_epochs": int(self.stage_3d_only_epochs),
            "base_lr": float(self.lr),
            "base_weight_decay": float(self.weight_decay),
            "lr_scale_2d": float(self.lr_group_scales["2d"]),
            "lr_scale_3d": float(self.lr_group_scales["3d"]),
            "lr_scale_fusion": float(self.lr_group_scales["fusion"]),
            "lr_scale_heads": float(self.lr_group_scales["heads"]),
            "weight_decay_scale_2d": float(self.weight_decay_group_scales["2d"]),
            "weight_decay_scale_3d": float(self.weight_decay_group_scales["3d"]),
            "weight_decay_scale_fusion": float(self.weight_decay_group_scales["fusion"]),
            "weight_decay_scale_heads": float(self.weight_decay_group_scales["heads"]),
        }

    def get_optimizer_group_summaries(self) -> List[Dict[str, Any]]:
        groups = self._named_parameters_by_group()
        summaries: List[Dict[str, Any]] = []
        for group_name in ("2d", "3d", "fusion", "heads"):
            named_params = groups[group_name]
            if len(named_params) == 0:
                continue
            n_params = int(sum(int(param.numel()) for _, param in named_params))
            n_trainable_params = int(sum(int(param.numel()) for _, param in named_params if bool(param.requires_grad)))
            sample_param_names = ",".join([str(name) for name, _ in named_params[:3]])
            summaries.append(
                {
                    "group": str(group_name),
                    "lr": float(self.lr) * float(self.lr_group_scales[group_name]),
                    "weight_decay": float(self.weight_decay) * float(self.weight_decay_group_scales[group_name]),
                    "n_tensors": int(len(named_params)),
                    "n_trainable_tensors": int(sum(1 for _, param in named_params if bool(param.requires_grad))),
                    "n_params": int(n_params),
                    "n_trainable_params": int(n_trainable_params),
                    "sample_params": str(sample_param_names),
                }
            )
        return summaries

    def _current_optimizer_group_values(self) -> Dict[str, Dict[str, float]]:
        values: Dict[str, Dict[str, float]] = {}
        trainer = getattr(self, "trainer", None)
        opt_list = getattr(trainer, "optimizers", None) if trainer is not None else None
        if opt_list:
            opt = opt_list[0]
            for idx, group_name in enumerate(self._optimizer_group_names):
                if idx >= len(opt.param_groups):
                    continue
                group_cfg = opt.param_groups[idx]
                values[str(group_name)] = {
                    "lr": float(group_cfg.get("lr", self.lr)),
                    "weight_decay": float(group_cfg.get("weight_decay", self.weight_decay)),
                }
        if len(values) == 0:
            for summary in self.get_optimizer_group_summaries():
                values[str(summary["group"])] = {
                    "lr": float(summary["lr"]),
                    "weight_decay": float(summary["weight_decay"]),
                }
        return values

    def _current_stage_log_fields(self) -> Dict[str, Any]:
        mode = self._stage_mode_for_epoch(int(getattr(self, "current_epoch", 0)))
        enabled_modalities = sorted(str(x) for x in self._enabled_modalities_for_current_stage())
        groups = self._named_parameters_by_group()
        opt_values = self._current_optimizer_group_values()
        fields: Dict[str, Any] = {
            "epoch": int(getattr(self, "current_epoch", 0)),
            "stage_mode": str(mode),
            "enabled_modalities": ",".join(enabled_modalities) if len(enabled_modalities) > 0 else "none",
            "active_modalities": ",".join(str(x) for x in self.active_modalities) if len(self.active_modalities) > 0 else "none",
        }
        for group_name in ("2d", "3d", "fusion", "heads"):
            named_params = groups[group_name]
            if len(named_params) == 0:
                continue
            n_trainable_tensors = int(sum(1 for _, param in named_params if bool(param.requires_grad)))
            n_trainable_params = int(sum(int(param.numel()) for _, param in named_params if bool(param.requires_grad)))
            fields[f"group_{group_name}_trainable"] = bool(n_trainable_tensors > 0)
            fields[f"group_{group_name}_n_tensors"] = int(len(named_params))
            fields[f"group_{group_name}_n_trainable_tensors"] = int(n_trainable_tensors)
            fields[f"group_{group_name}_n_trainable_params"] = int(n_trainable_params)
            if group_name in opt_values:
                fields[f"group_{group_name}_lr"] = float(opt_values[group_name]["lr"])
                fields[f"group_{group_name}_weight_decay"] = float(opt_values[group_name]["weight_decay"])
        return fields

    def _shared_training_parameters(self) -> List[torch.nn.Parameter]:
        shared_groups = {"2d", "3d", "fusion"}
        params: List[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            if not bool(param.requires_grad):
                continue
            if self._parameter_group_name(str(name)) in shared_groups:
                params.append(param)
        return params

    @staticmethod
    def _clone_grad_list(
        grads: Sequence[Optional[torch.Tensor]],
    ) -> List[Optional[torch.Tensor]]:
        return [None if g is None else g.detach().clone() for g in grads]

    @staticmethod
    def _grad_device(
        grads: Sequence[Optional[torch.Tensor]],
    ) -> torch.device:
        for grad in grads:
            if grad is not None:
                return grad.device
        return torch.device("cpu")

    @classmethod
    def _grad_dot(
        cls,
        grads_a: Sequence[Optional[torch.Tensor]],
        grads_b: Sequence[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        terms: List[torch.Tensor] = []
        for ga, gb in zip(grads_a, grads_b):
            if ga is None or gb is None:
                continue
            terms.append((ga.float() * gb.float()).sum())
        if len(terms) <= 0:
            device = cls._grad_device(grads_a)
            if str(device) == "cpu":
                device = cls._grad_device(grads_b)
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(terms).sum()

    @classmethod
    def _grad_norm(
        cls,
        grads: Sequence[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        return torch.sqrt(cls._grad_dot(grads, grads).clamp_min(0.0))

    @classmethod
    def _grad_cosine(
        cls,
        grads_a: Sequence[Optional[torch.Tensor]],
        grads_b: Sequence[Optional[torch.Tensor]],
        eps: float = 1e-12,
    ) -> torch.Tensor:
        den = cls._grad_norm(grads_a) * cls._grad_norm(grads_b)
        if float(den.detach().item()) <= float(eps):
            device = den.device
            if str(device) == "cpu":
                device = cls._grad_device(grads_a)
                if str(device) == "cpu":
                    device = cls._grad_device(grads_b)
            return torch.zeros((), dtype=torch.float32, device=device)
        return cls._grad_dot(grads_a, grads_b) / den.clamp_min(float(eps))

    @classmethod
    def _sum_grad_lists(
        cls,
        grad_lists: Sequence[Sequence[Optional[torch.Tensor]]],
    ) -> List[Optional[torch.Tensor]]:
        if len(grad_lists) <= 0:
            return []
        summed = cls._clone_grad_list(grad_lists[0])
        for extra in grad_lists[1:]:
            for idx, g in enumerate(extra):
                if g is None:
                    continue
                if summed[idx] is None:
                    summed[idx] = g.detach().clone()
                else:
                    summed[idx] = summed[idx] + g.detach()
        return summed

    @classmethod
    def _subtract_grad_lists(
        cls,
        grads_a: Sequence[Optional[torch.Tensor]],
        grads_b: Sequence[Optional[torch.Tensor]],
    ) -> List[Optional[torch.Tensor]]:
        out = cls._clone_grad_list(grads_a)
        for idx, gb in enumerate(grads_b):
            if gb is None:
                continue
            if out[idx] is None:
                out[idx] = -gb.detach().clone()
            else:
                out[idx] = out[idx] - gb.detach()
        return out

    def _pcgrad_merge(
        self,
        task_grads: Sequence[Sequence[Optional[torch.Tensor]]],
    ) -> List[Optional[torch.Tensor]]:
        if len(task_grads) <= 0:
            return []
        projected = [self._clone_grad_list(grads) for grads in task_grads]
        n_tasks = int(len(task_grads))
        if n_tasks <= 1:
            return projected[0]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(self, "global_step", 0)) + 17)
        for task_idx in range(n_tasks):
            order = torch.randperm(n_tasks, generator=generator).tolist()
            for other_idx in order:
                if int(other_idx) == int(task_idx):
                    continue
                dot = self._grad_dot(projected[task_idx], task_grads[int(other_idx)])
                if float(dot.detach().item()) >= 0.0:
                    continue
                denom = self._grad_dot(task_grads[int(other_idx)], task_grads[int(other_idx)]).clamp_min(1e-12)
                scale = (dot / denom).detach()
                for grad_pos, g_other in enumerate(task_grads[int(other_idx)]):
                    if projected[task_idx][grad_pos] is None or g_other is None:
                        continue
                    projected[task_idx][grad_pos] = projected[task_idx][grad_pos] - scale * g_other.detach()
        return self._sum_grad_lists(projected)

    def _log_task_gradient_diagnostics(
        self,
        *,
        task_grads: Sequence[Sequence[Optional[torch.Tensor]]],
        batch_size: int,
    ) -> None:
        if (not bool(self.log_task_gradient_diagnostics)) or len(task_grads) < 2:
            return
        cos_vals: List[torch.Tensor] = []
        for i in range(len(task_grads)):
            for j in range(i + 1, len(task_grads)):
                cos_ij = self._grad_cosine(task_grads[i], task_grads[j])
                cos_vals.append(cos_ij)
                self.log(
                    f"train_task_grad_cos_t{i}_t{j}",
                    cos_ij,
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(batch_size),
                )
        if len(cos_vals) <= 0:
            return
        target_device = cos_vals[0].device
        cos_stack = torch.stack([c.float().to(device=target_device) for c in cos_vals])
        self.log("train_task_grad_cos_mean", cos_stack.mean(), on_step=False, on_epoch=True, batch_size=int(batch_size))
        self.log("train_task_grad_cos_min", cos_stack.min(), on_step=False, on_epoch=True, batch_size=int(batch_size))
        self.log(
            "train_task_grad_conflict_rate",
            (cos_stack < 0.0).float().mean(),
            on_step=False,
            on_epoch=True,
            batch_size=int(batch_size),
        )

    def _gradient_accumulation_steps(self) -> int:
        if not bool(self.automatic_optimization):
            return int(max(1, getattr(self, "manual_accumulate_grad_batches", 1)))
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 1
        return int(max(1, getattr(trainer, "accumulate_grad_batches", 1)))

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        accum = int(self._gradient_accumulation_steps())
        if ((int(batch_idx) + 1) % accum) == 0:
            return True
        trainer = getattr(self, "trainer", None)
        num_batches = None if trainer is None else getattr(trainer, "num_training_batches", None)
        if num_batches is None:
            return True
        try:
            return int(batch_idx) + 1 >= int(num_batches)
        except Exception:
            return True

    def _accumulate_real_3d_counts(
        self,
        *,
        split: str,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> None:
        real_valid = self._non_dummy_valid_instance_mask(
            x3d_pad=x3d_pad,
            key_padding_mask=key_padding_mask,
        )
        n_with = int(real_valid.any(dim=1).sum().detach().item())
        batch_size = int(x3d_pad.shape[0])
        n_without = int(max(0, batch_size - n_with))
        if str(split) == "train":
            self._train_real_3d_with += int(n_with)
            self._train_real_3d_without += int(n_without)
        elif str(split) == "val":
            self._val_real_3d_with += int(n_with)
            self._val_real_3d_without += int(n_without)
        else:
            raise ValueError(f"Unsupported split={split}")

    def _log_real_3d_epoch_summary(self, *, split: str) -> None:
        if str(split) == "train":
            n_with = int(self._train_real_3d_with)
            n_without = int(self._train_real_3d_without)
            event_name = "mil.train.epoch_real3d"
        elif str(split) == "val":
            n_with = int(self._val_real_3d_with)
            n_without = int(self._val_real_3d_without)
            event_name = "mil.val.epoch_real3d"
        else:
            raise ValueError(f"Unsupported split={split}")
        total = int(n_with + n_without)
        if total <= 0:
            return
        log_event(
            "INFO",
            event_name,
            epoch=int(getattr(self, "current_epoch", 0)),
            stage_mode=str(self._stage_mode_for_epoch(int(getattr(self, "current_epoch", 0)))),
            n_samples_with_real_3d=int(n_with),
            n_samples_without_real_3d=int(n_without),
            frac_with_real_3d=float(n_with / float(total)),
        )

    def on_fit_start(self) -> None:
        self._stagewise_active = True
        self._apply_stage_trainability()
        log_event("INFO", "mil.fit.training_setup", **self.get_training_debug_summary())

    def on_train_epoch_start(self) -> None:
        self._apply_stage_trainability()
        self._train_real_3d_with = 0
        self._train_real_3d_without = 0
        log_event("INFO", "mil.train.epoch_start", **self._current_stage_log_fields())

    def on_train_epoch_end(self) -> None:
        self._log_real_3d_epoch_summary(split="train")

    def on_fit_end(self) -> None:
        self._stagewise_active = False
        self._last_stage_trainability = None
        self._apply_stage_trainability()
        log_event("INFO", "mil.fit.stage_reset", stagewise_active=bool(self._stagewise_active))

    def configure_rl_concept_guidance(
        self,
        *,
        task_target_concepts: Dict[int, Sequence[str]],
        concept_conf_map: Dict[str, Set[Tuple[str, str]]],
        concept_mol_map: Dict[str, Set[str]],
        init_scale: float = 0.02,
        max_scale: float = 0.20,
        negative_penalty: float = 0.15,
    ) -> None:
        """Attach concept-target guidance maps used by the RL controller callback."""
        self.rl_task_target_concepts = [tuple() for _ in range(NUM_TASKS)]
        self.rl_target_conf_pairs_by_task = [set() for _ in range(NUM_TASKS)]
        self.rl_target_mols_by_task = [set() for _ in range(NUM_TASKS)]

        any_target = False
        for ti in range(NUM_TASKS):
            concept_ids = [str(x) for x in task_target_concepts.get(int(ti), [])]
            self.rl_task_target_concepts[ti] = tuple(concept_ids)
            if concept_ids:
                any_target = True
            conf_pairs = self.rl_target_conf_pairs_by_task[ti]
            mols = self.rl_target_mols_by_task[ti]
            for cid in concept_ids:
                for mol_id, conf_id in concept_conf_map.get(str(cid), set()):
                    conf_pairs.add((str(mol_id), str(conf_id)))
                for mol_id in concept_mol_map.get(str(cid), set()):
                    mols.add(str(mol_id))

        has_conf_target = bool(any(len(x) > 0 for x in self.rl_target_conf_pairs_by_task))
        self.rl_enabled = bool(any_target and has_conf_target)
        self.rl_guidance_scale_max = float(max(0.0, max_scale))
        self.rl_negative_penalty = float(max(0.0, negative_penalty))
        self.set_rl_guidance_scale(float(init_scale))

    def set_rl_guidance_scale(self, value: float) -> None:
        """Update concept-guidance strength used in training loss."""
        v = float(value)
        if not np.isfinite(v):
            return
        self.rl_guidance_scale = float(np.clip(v, 0.0, max(self.rl_guidance_scale_max, 0.0)))

    def _concept_alignment_score(
        self,
        *,
        y_cls: torch.Tensor,
        attn: Any,
        key_padding_mask: torch.Tensor,
        mol_ids: Sequence[str],
        conf_pad: np.ndarray,
    ) -> torch.Tensor:
        """Average attention mass on task-target concepts for positive labels."""
        if (not self.rl_enabled) or attn is None:
            return torch.zeros((), dtype=y_cls.dtype, device=y_cls.device)

        attn_tensor = self._resolve_alignment_attention(attn)
        if attn_tensor is None:
            return torch.zeros((), dtype=y_cls.dtype, device=y_cls.device)

        total = torch.zeros((), dtype=attn_tensor.dtype, device=attn_tensor.device)
        denom = 0
        bsz = int(y_cls.shape[0])

        for b in range(bsz):
            mol_id = str(mol_ids[b])
            valid = ~key_padding_mask[b]
            L = int(valid.sum().item())
            if L <= 0:
                continue
            confs = [str(x) for x in conf_pad[b, :L].tolist()]

            for t in range(NUM_TASKS):
                if float(y_cls[b, t].detach().item()) <= 0.5:
                    y_pos = False
                else:
                    y_pos = True
                w = attn_tensor[b, t, :L]
                w = w / (w.sum() + 1e-8)

                target_pairs = self.rl_target_conf_pairs_by_task[t]
                idx = [i for i, cid in enumerate(confs) if (mol_id, str(cid)) in target_pairs]
                if not idx:
                    continue
                idx_t = torch.tensor(idx, dtype=torch.long, device=attn_tensor.device)
                score = w.index_select(0, idx_t).sum()
                if y_pos:
                    total = total + score
                    denom += 1
                else:
                    neg_w = float(max(0.0, self.rl_negative_penalty))
                    if neg_w > 0.0:
                        total = total - (neg_w * score)
                        denom += neg_w

        if denom <= 0:
            return torch.zeros((), dtype=attn_tensor.dtype, device=attn_tensor.device)
        return total / float(denom)

    def _resolve_alignment_attention(self, attn: Any) -> Optional[torch.Tensor]:
        """
        Convert attention payload into a conformer-level [B,T,N] tensor for RL alignment.

        If modality-aware attention is available, use 3D modality gates to combine geometry
        and QM branch attentions. 2D gates are ignored here because RL alignment is defined
        over conformer-level targets only.
        """
        if attn is None:
            return None
        if torch.is_tensor(attn):
            return attn
        if not isinstance(attn, dict):
            return None

        attn_geom = attn.get("attn_geom")
        attn_qm = attn.get("attn_qm")
        if attn_geom is None and attn_qm is None:
            return None
        if attn_geom is None:
            return attn_qm
        if attn_qm is None:
            return attn_geom

        modality_gates = attn.get("modality_gates")
        modality_order = tuple(str(x) for x in (attn.get("modality_order") or ()))
        if modality_gates is None or len(modality_order) != int(modality_gates.shape[-1]):
            return 0.5 * (attn_geom + attn_qm)

        try:
            geom_idx = modality_order.index("3d_geom")
        except ValueError:
            geom_idx = -1
        try:
            qm_idx = modality_order.index("3d_qm")
        except ValueError:
            qm_idx = -1

        if geom_idx < 0 and qm_idx < 0:
            return 0.5 * (attn_geom + attn_qm)
        if geom_idx < 0:
            return attn_qm
        if qm_idx < 0:
            return attn_geom

        g_geom = modality_gates[..., int(geom_idx)]
        g_qm = modality_gates[..., int(qm_idx)]
        g_sum = (g_geom + g_qm).clamp_min(1e-8)
        w_geom = (g_geom / g_sum).unsqueeze(-1)
        w_qm = (g_qm / g_sum).unsqueeze(-1)
        return (w_geom * attn_geom) + (w_qm * attn_qm)

    def forward(
        self,
        x2d: torch.Tensor,              # [B,F2]
        x3d_pad: torch.Tensor,          # [B,N,F3]
        key_padding_mask: torch.Tensor, # [B,N] True=PAD
        return_attn: bool = False,
        return_attn_modalities: bool = False,
        return_bitmask: bool = False,
    ):
        self._validate_forward_inputs(x2d=x2d, x3d_pad=x3d_pad, key_padding_mask=key_padding_mask)
        need_attn = bool(return_attn or return_attn_modalities)
        forward_out = self._compute_outputs(
            x2d=x2d,
            x3d_pad=x3d_pad,
            key_padding_mask=key_padding_mask,
            need_attn=need_attn,
        )
        logits = forward_out["logits"]
        abs_out = forward_out["abs_out"]
        fluo_out = forward_out["fluo_out"]
        bitmask_logits = forward_out["bitmask_logits"]
        fusion_info = forward_out["fusion_info"]
        attn_geom = forward_out["attn_geom"]
        attn_qm = forward_out["attn_qm"]

        attn_fused = None
        if return_attn and (not return_attn_modalities):
            maps = [m for m in (attn_geom, attn_qm) if m is not None]
            if len(maps) > 0:
                attn_fused = torch.stack(maps, dim=0).mean(dim=0)
        attn_payload: Any = attn_fused
        if return_attn_modalities:
            attn_payload = {
                "attn_geom": attn_geom,
                "attn_qm": attn_qm,
                "modality_gates": fusion_info.get("modality_gates"),
                "modality_scores": fusion_info.get("modality_scores"),
                "modality_order": fusion_info.get("modality_order"),
                "modality_attn": fusion_info.get("modality_attn"),
                "mixer_info": fusion_info.get("mixer_info"),
            }

        if need_attn and return_bitmask:
            return logits, abs_out, fluo_out, bitmask_logits, attn_payload
        if need_attn:
            return logits, abs_out, fluo_out, attn_payload
        if return_bitmask:
            return logits, abs_out, fluo_out, bitmask_logits
        return logits, abs_out, fluo_out

    def _bitmask_group_targets(self, y_cls: torch.Tensor) -> Optional[torch.Tensor]:
        if self.bitmask_head is None:
            return None
        yb = (y_cls > 0.5).long()
        bits = (1 << torch.arange(NUM_TASKS, device=yb.device, dtype=torch.long)).reshape(1, -1)
        mask_ids = (yb * bits).sum(dim=1).clamp(min=0, max=int((1 << NUM_TASKS) - 1))
        return self.bitmask_mask_to_group[mask_ids]

    @staticmethod
    def _validate_forward_inputs(
        *,
        x2d: torch.Tensor,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> None:
        if x2d.dim() != 2:
            raise ValueError(f"x2d must be rank-2 [B,F2], got shape={tuple(x2d.shape)}")
        if x3d_pad.dim() != 3:
            raise ValueError(f"x3d_pad must be rank-3 [B,N,F3], got shape={tuple(x3d_pad.shape)}")
        if key_padding_mask.dim() != 2:
            raise ValueError(f"key_padding_mask must be rank-2 [B,N], got shape={tuple(key_padding_mask.shape)}")
        if x2d.shape[0] != x3d_pad.shape[0]:
            raise ValueError(
                f"Batch mismatch: x2d batch={int(x2d.shape[0])} vs x3d_pad batch={int(x3d_pad.shape[0])}"
            )
        if x3d_pad.shape[:2] != key_padding_mask.shape:
            raise ValueError(
                f"Mask mismatch: x3d_pad[:2]={tuple(x3d_pad.shape[:2])} vs key_padding_mask={tuple(key_padding_mask.shape)}"
            )

    def _split_instance_modalities(self, x3d_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_dim = int(x3d_pad.shape[-1])
        gdim = int(max(0, self.inst_geom_dim))
        qdim = int(max(0, self.inst_qm_dim))
        if gdim + qdim > feat_dim:
            # Prefer configured geometry slice and consume remaining as QM.
            gdim = min(gdim, feat_dim)
            qdim = max(0, feat_dim - gdim)
        if gdim <= 0:
            geom = x3d_pad[..., :0]
        else:
            geom = x3d_pad[..., :gdim]
        if qdim <= 0:
            qm = x3d_pad[..., :0]
        else:
            qm = x3d_pad[..., gdim : gdim + qdim]
        return geom, qm

    def _pool_one_branch(
        self,
        *,
        x3d_mod: torch.Tensor,
        key_padding_mask: torch.Tensor,
        embedder: nn.Module,
        post_norm: nn.Module,
        aggregator: nn.Module,
        return_attn: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, n_instances, feature_dim = x3d_mod.shape
        tok = embedder(
            x3d_mod.reshape(batch_size * n_instances, feature_dim)
        ).reshape(batch_size, n_instances, -1)
        tok = post_norm(tok)
        return aggregator(tok, key_padding_mask=key_padding_mask, return_attn=return_attn)

    def _pool_task_tokens(
        self,
        *,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
        return_attn: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x3d_geom, x3d_qm = self._split_instance_modalities(x3d_pad)
        batch_size = int(x3d_pad.shape[0])
        pooled_geom = torch.zeros(
            (batch_size, NUM_TASKS, int(self.inst_hidden)),
            dtype=x3d_pad.dtype,
            device=x3d_pad.device,
        )
        pooled_qm = torch.zeros_like(pooled_geom)
        attn_geom: Optional[torch.Tensor] = None
        attn_qm: Optional[torch.Tensor] = None
        if self.inst_geom_enc is not None and self.attn_pool_geom is not None and int(x3d_geom.shape[-1]) > 0:
            pooled_geom, attn_geom = self._pool_one_branch(
                x3d_mod=x3d_geom,
                key_padding_mask=key_padding_mask,
                embedder=self.inst_geom_enc,
                post_norm=self.inst_geom_post_embed_norm,
                aggregator=self.attn_pool_geom,
                return_attn=return_attn,
            )
        if self.inst_qm_enc is not None and self.attn_pool_qm is not None and int(x3d_qm.shape[-1]) > 0:
            pooled_qm, attn_qm = self._pool_one_branch(
                x3d_mod=x3d_qm,
                key_padding_mask=key_padding_mask,
                embedder=self.inst_qm_enc,
                post_norm=self.inst_qm_post_embed_norm,
                aggregator=self.attn_pool_qm,
                return_attn=return_attn,
            )
        return pooled_geom, pooled_qm, attn_geom, attn_qm

    def _build_task_representations(
        self,
        *,
        x2d: torch.Tensor,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
        pooled_geom: torch.Tensor,
        pooled_qm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size = x2d.shape[0]
        modality_parts: dict[str, torch.Tensor] = {}
        if self.mol_enc is not None and self.mol_post_embed_norm is not None and self.proj2d is not None:
            mol_emb = self.mol_post_embed_norm(self.mol_enc(x2d))
            e2d = self.proj2d(mol_emb)  # [B,proj]
            e2d_rep = e2d.unsqueeze(1).expand(-1, NUM_TASKS, -1)  # [B,4,proj]
            if (
                self.fusion_use_task_2d_adapter
                and self.task_2d_tokens is not None
                and self.task_2d_adapter is not None
                and self.task_2d_post_norm is not None
            ):
                task_ctx = self.task_2d_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                delta = self.task_2d_adapter(torch.cat([e2d_rep, task_ctx], dim=-1))
                e2d_rep = self.task_2d_post_norm(e2d_rep + delta)
            modality_parts["2d"] = e2d_rep
        if self.proj3d_geom is not None and self.agg_geom_post_norm is not None:
            pooled_geom = self.agg_geom_post_norm(pooled_geom)
            e3d_geom = self.proj3d_geom(
                pooled_geom.reshape(batch_size * NUM_TASKS, -1)
            ).reshape(batch_size, NUM_TASKS, -1)
            modality_parts["3d_geom"] = e3d_geom
        if self.proj3d_qm is not None and self.agg_qm_post_norm is not None:
            pooled_qm = self.agg_qm_post_norm(pooled_qm)
            e3d_qm = self.proj3d_qm(
                pooled_qm.reshape(batch_size * NUM_TASKS, -1)
            ).reshape(batch_size, NUM_TASKS, -1)
            modality_parts["3d_qm"] = e3d_qm
        if len(modality_parts) == 0:
            raise RuntimeError("No active modality projections found for mixer input.")
        enabled_modalities = self._enabled_modalities_for_current_stage()
        sample_has_real_3d = self._non_dummy_valid_instance_mask(
            x3d_pad=x3d_pad,
            key_padding_mask=key_padding_mask,
        ).any(dim=1)
        modality_presence: Dict[str, torch.Tensor] = {}
        for name in self.active_modalities:
            name_str = str(name)
            present = torch.ones((batch_size,), dtype=torch.bool, device=x2d.device)
            if name_str not in enabled_modalities:
                present = torch.zeros((batch_size,), dtype=torch.bool, device=x2d.device)
            elif name_str in {"3d_geom", "3d_qm"}:
                present = sample_has_real_3d
            modality_presence[name_str] = present
        for name in tuple(modality_parts.keys()):
            presence = modality_presence.get(str(name))
            if presence is None:
                continue
            modality_parts[str(name)] = modality_parts[str(name)] * presence.to(
                dtype=modality_parts[str(name)].dtype
            ).view(batch_size, 1, 1)
        tokens = torch.stack(
            [modality_parts[str(name)] for name in self.active_modalities],
            dim=2,
        )  # [B,T,M,proj]
        n_modalities = int(tokens.shape[2])
        type_embed = self.modality_type_embeddings[:n_modalities].view(1, 1, n_modalities, self.proj_dim)
        tokens_with_type = tokens + type_embed
        active_mask = torch.stack(
            [
                modality_presence[str(name)].to(dtype=tokens.dtype, device=tokens.device)
                for name in self.active_modalities
            ],
            dim=1,
        ).view(batch_size, 1, n_modalities)
        active_mask_bt = active_mask.expand(batch_size, NUM_TASKS, n_modalities)

        if n_modalities == 1:
            modality_scores = torch.ones(
                (batch_size, NUM_TASKS, 1),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            modality_gates = active_mask_bt
        else:
            flat_tokens = tokens_with_type.reshape(batch_size * NUM_TASKS * n_modalities, self.proj_dim)
            modality_scores = self.modality_gate_net(flat_tokens).reshape(batch_size, NUM_TASKS, n_modalities)
            if bool(torch.any(active_mask_bt < 0.5)):
                # Use a dtype-safe floor so mixed-precision runs do not overflow on fp16.
                modality_scores = modality_scores.masked_fill(
                    active_mask_bt < 0.5,
                    float(torch.finfo(modality_scores.dtype).min),
                )
            if self.fusion_use_modality_gates:
                modality_gates = torch.softmax(modality_scores, dim=-1)
                modality_gates = modality_gates * active_mask_bt
                modality_gates = modality_gates / modality_gates.sum(dim=-1, keepdim=True).clamp_min(1.0)
            else:
                modality_gates = active_mask_bt / active_mask_bt.sum(dim=-1, keepdim=True).clamp_min(1.0)

        gated_tokens = tokens * modality_gates.unsqueeze(-1)
        modality_attn = None
        fused_tokens = gated_tokens
        if (
            self.fusion_use_modality_interaction
            and self.modality_interaction_attn is not None
            and self.modality_interaction_ln1 is not None
            and self.modality_interaction_ffn is not None
            and self.modality_interaction_ln2 is not None
            and n_modalities > 1
        ):
            seq = gated_tokens.reshape(batch_size * NUM_TASKS, n_modalities, self.proj_dim)
            seq_active_mask = active_mask_bt.reshape(batch_size * NUM_TASKS, n_modalities).bool()
            seq_type = type_embed.expand(batch_size, NUM_TASKS, -1, -1).reshape(
                batch_size * NUM_TASKS, n_modalities, self.proj_dim
            )
            seq_in = seq + (seq_type * seq_active_mask.to(dtype=seq.dtype).unsqueeze(-1))
            row_has_active = seq_active_mask.any(dim=1)
            fused_seq = seq
            modality_attn_full = torch.zeros(
                (batch_size * NUM_TASKS, n_modalities, n_modalities),
                dtype=seq.dtype,
                device=seq.device,
            )
            if bool(torch.any(row_has_active)):
                active_rows = torch.nonzero(row_has_active, as_tuple=False).squeeze(1)
                seq_in_sel = seq_in.index_select(0, active_rows)
                seq_sel = seq.index_select(0, active_rows)
                seq_mask_sel = seq_active_mask.index_select(0, active_rows)
                attn_out, attn_w = self.modality_interaction_attn(
                    query=seq_in_sel,
                    key=seq_in_sel,
                    value=seq_in_sel,
                    need_weights=True,
                    average_attn_weights=False,
                    key_padding_mask=(~seq_mask_sel),
                )
                seq_sel = self.modality_interaction_ln1(seq_sel + self.modality_interaction_dropout(attn_out))
                seq_sel = self.modality_interaction_ln2(seq_sel + self.modality_interaction_ffn(seq_sel))
                seq_sel = seq_sel * seq_mask_sel.to(dtype=seq_sel.dtype).unsqueeze(-1)
                fused_seq = fused_seq.clone()
                fused_seq.index_copy_(0, active_rows, seq_sel)
                modality_attn_full.index_copy_(0, active_rows, attn_w.mean(dim=1))
            fused_tokens = fused_seq.reshape(batch_size, NUM_TASKS, n_modalities, self.proj_dim)
            modality_attn = modality_attn_full.reshape(batch_size, NUM_TASKS, n_modalities, n_modalities)

        mix_in = fused_tokens.reshape(batch_size * NUM_TASKS, n_modalities * self.proj_dim)
        mixer_info: Dict[str, Any] = {}
        mix_out = self.mixer(mix_in)
        z_tasks = mix_out.reshape(batch_size, NUM_TASKS, -1)  # [B,4,mixer_hidden]
        z_tasks = self.mixer_post_norm(z_tasks)
        fusion_info = {
            "modality_order": tuple(str(x) for x in self.active_modalities),
            "modality_scores": modality_scores,
            "modality_gates": modality_gates,
            "modality_attn": modality_attn,
            "modality_parts": modality_parts,
            "mixer_info": mixer_info,
            "sample_has_real_3d": sample_has_real_3d,
            "modality_presence": {str(k): v for k, v in modality_presence.items()},
        }
        return z_tasks, fusion_info

    @staticmethod
    def _summary_latents_from_modality_parts(modality_parts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            str(name): part.mean(dim=1)
            for name, part in modality_parts.items()
            if torch.is_tensor(part) and part.ndim == 3 and int(part.shape[1]) > 0
        }

    def _project_contrastive_latents(self, summaries: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, tensor in summaries.items():
            if str(name) not in self.contrastive_heads:
                continue
            out[str(name)] = F.normalize(self.contrastive_heads[str(name)](tensor), p=2, dim=-1)
        return out

    @staticmethod
    def _positive_mask_from_ids(
        *,
        batch_size: int,
        device: torch.device,
        mol_ids: Optional[Sequence[Any]],
    ) -> torch.Tensor:
        if mol_ids is None or len(mol_ids) != int(batch_size):
            return torch.eye(int(batch_size), dtype=torch.bool, device=device)
        ids = [str(x) for x in mol_ids]
        mask = [[ids[i] == ids[j] for j in range(len(ids))] for i in range(len(ids))]
        return torch.tensor(mask, dtype=torch.bool, device=device)

    @staticmethod
    def _mask_positive_pairs_by_sample_eligibility(
        positive_mask: torch.Tensor,
        sample_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if sample_mask is None:
            return positive_mask
        sample_mask = sample_mask.to(device=positive_mask.device, dtype=torch.bool).reshape(-1)
        if int(sample_mask.numel()) != int(positive_mask.shape[0]):
            raise ValueError(
                f"Expected sample eligibility to have length {int(positive_mask.shape[0])}, "
                f"got {int(sample_mask.numel())}"
            )
        pair_mask = sample_mask.unsqueeze(1) & sample_mask.unsqueeze(0)
        return positive_mask & pair_mask

    @staticmethod
    def _non_dummy_valid_instance_mask(
        *,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
        zero_tol: float = 1e-12,
    ) -> torch.Tensor:
        if x3d_pad.ndim != 3:
            raise ValueError(f"Expected x3d_pad shape [B,N,F], got {tuple(x3d_pad.shape)}")
        if key_padding_mask.ndim != 2:
            raise ValueError(f"Expected key_padding_mask shape [B,N], got {tuple(key_padding_mask.shape)}")
        if tuple(x3d_pad.shape[:2]) != tuple(key_padding_mask.shape):
            raise ValueError(
                f"Expected x3d_pad/key_padding_mask to agree on [B,N], "
                f"got {tuple(x3d_pad.shape[:2])} vs {tuple(key_padding_mask.shape)}"
            )
        valid_mask = ~key_padding_mask.bool()
        nonzero_mask = x3d_pad.abs().sum(dim=-1) > float(zero_tol)
        return valid_mask & nonzero_mask

    @staticmethod
    def _multi_positive_info_nce(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        anchor_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.ndim != 2 or positive_mask.shape != logits.shape:
            raise ValueError(
                f"Expected logits/positive_mask shape [B,B], got logits={tuple(logits.shape)} mask={tuple(positive_mask.shape)}"
            )
        row_has_pos = positive_mask.any(dim=1)
        if not bool(torch.any(row_has_pos)):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        log_denom = torch.logsumexp(logits, dim=1)
        pos_logits = logits.masked_fill(~positive_mask, float("-inf"))
        log_num = torch.logsumexp(pos_logits, dim=1)
        per_row = -(log_num[row_has_pos] - log_denom[row_has_pos])
        if anchor_weight is None:
            return per_row.mean()
        row_weight = anchor_weight[row_has_pos].to(dtype=per_row.dtype).clamp_min(0.0)
        denom = row_weight.sum()
        if not bool(torch.isfinite(denom)) or float(denom.detach().item()) <= 0.0:
            return per_row.mean()
        return (per_row * row_weight).sum() / denom

    def _contrastive_pair_loss(
        self,
        *,
        za: Optional[torch.Tensor],
        zb: Optional[torch.Tensor],
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        if za is None or zb is None:
            return positive_mask.new_zeros((), dtype=torch.float32)
        if int(za.shape[0]) <= 1 or int(zb.shape[0]) <= 1:
            return za.new_zeros(())
        logits = (za @ zb.transpose(0, 1)) / float(self.contrastive_temperature)
        loss_ab = self._multi_positive_info_nce(logits, positive_mask)
        loss_ba = self._multi_positive_info_nce(logits.transpose(0, 1), positive_mask.transpose(0, 1))
        return 0.5 * (loss_ab + loss_ba)

    def _task_supervised_contrastive_loss(
        self,
        *,
        z_task: torch.Tensor,
        labels: torch.Tensor,
        anchor_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if int(z_task.shape[0]) <= 1:
            return z_task.new_zeros(())
        labels = (labels > 0.5).long().reshape(-1)
        if int(labels.numel()) != int(z_task.shape[0]):
            raise ValueError(
                f"Expected task labels to match embeddings, got labels={tuple(labels.shape)} z_task={tuple(z_task.shape)}"
            )
        unique_labels = torch.unique(labels)
        if int(unique_labels.numel()) < 2:
            return z_task.new_zeros(())
        logits = (z_task @ z_task.transpose(0, 1)) / float(self.contrastive_temperature)
        eye = torch.eye(int(z_task.shape[0]), dtype=torch.bool, device=z_task.device)
        logits = logits.masked_fill(eye, float("-inf"))

        class_losses: List[torch.Tensor] = []
        for cls in unique_labels.tolist():
            class_mask = labels == int(cls)
            if int(class_mask.sum().item()) < 2:
                continue
            positive_mask = class_mask.unsqueeze(1) & class_mask.unsqueeze(0) & (~eye)
            if not bool(torch.any(positive_mask)):
                continue
            class_weight = None
            if anchor_weight is not None:
                class_weight = anchor_weight.to(dtype=z_task.dtype).reshape(-1)
            class_losses.append(
                self._multi_positive_info_nce(
                    logits,
                    positive_mask,
                    anchor_weight=class_weight,
                )
            )
        if len(class_losses) == 0:
            return z_task.new_zeros(())
        return torch.stack(class_losses, dim=0).mean()

    def _cross_modal_contrastive_loss(
        self,
        *,
        fusion_info: Dict[str, Any],
        mol_ids: Optional[Sequence[Any]],
        x3d_pad: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        modality_parts = fusion_info.get("modality_parts")
        if not isinstance(modality_parts, dict) or len(modality_parts) <= 1:
            return self.lam.new_zeros(())
        enabled_modalities = self._enabled_modalities_for_current_stage()
        modality_parts = {
            str(name): tensor
            for name, tensor in modality_parts.items()
            if str(name) in enabled_modalities
        }
        if len(modality_parts) <= 1:
            return self.lam.new_zeros(())
        summaries = self._summary_latents_from_modality_parts(modality_parts)
        proj = self._project_contrastive_latents(summaries)
        if len(proj) <= 1:
            return self.lam.new_zeros(())
        ref = next(iter(proj.values()))
        positive_mask = self._positive_mask_from_ids(
            batch_size=int(ref.shape[0]),
            device=ref.device,
            mol_ids=mol_ids,
        )
        has_real_3d: Optional[torch.Tensor] = None
        if x3d_pad is not None and key_padding_mask is not None:
            real_valid = self._non_dummy_valid_instance_mask(
                x3d_pad=x3d_pad,
                key_padding_mask=key_padding_mask,
            )
            has_real_3d = real_valid.any(dim=1)
        pair_losses: List[torch.Tensor] = []
        for name_a, name_b in (("2d", "3d_geom"), ("2d", "3d_qm")):
            if name_a in proj and name_b in proj:
                pair_positive_mask = self._mask_positive_pairs_by_sample_eligibility(positive_mask, has_real_3d)
                pair_losses.append(
                    self._contrastive_pair_loss(
                        za=proj[name_a],
                        zb=proj[name_b],
                        positive_mask=pair_positive_mask,
                    )
                )
        if self.cross_modal_include_geom_qm and ("3d_geom" in proj) and ("3d_qm" in proj):
            pair_positive_mask = self._mask_positive_pairs_by_sample_eligibility(positive_mask, has_real_3d)
            pair_losses.append(
                self._contrastive_pair_loss(
                    za=proj["3d_geom"],
                    zb=proj["3d_qm"],
                    positive_mask=pair_positive_mask,
                )
            )
        if len(pair_losses) == 0:
            return self.lam.new_zeros(())
        return torch.stack(pair_losses, dim=0).mean()

    @staticmethod
    def _sample_subset_padding_mask(
        *,
        key_padding_mask: torch.Tensor,
        keep_rate: float,
    ) -> torch.Tensor:
        keep = float(np.clip(keep_rate, 0.25, 1.0))
        valid = ~key_padding_mask
        subset_valid = torch.zeros_like(valid)
        for b in range(int(valid.shape[0])):
            idx = torch.nonzero(valid[b], as_tuple=False).squeeze(1)
            n = int(idx.numel())
            if n <= 0:
                continue
            keep_n = int(round(keep * float(n)))
            keep_n = max(1, min(n, keep_n))
            if keep_n >= n:
                subset_valid[b, idx] = True
                continue
            perm = torch.randperm(n, device=idx.device)[:keep_n]
            subset_valid[b, idx.index_select(0, perm)] = True
        return ~subset_valid

    def _project_3d_modality_parts(
        self,
        *,
        pooled_geom: torch.Tensor,
        pooled_qm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = int(pooled_geom.shape[0])
        modality_parts: Dict[str, torch.Tensor] = {}
        if self.proj3d_geom is not None and self.agg_geom_post_norm is not None:
            pooled_geom = self.agg_geom_post_norm(pooled_geom)
            modality_parts["3d_geom"] = self.proj3d_geom(
                pooled_geom.reshape(batch_size * NUM_TASKS, -1)
            ).reshape(batch_size, NUM_TASKS, -1)
        if self.proj3d_qm is not None and self.agg_qm_post_norm is not None:
            pooled_qm = self.agg_qm_post_norm(pooled_qm)
            modality_parts["3d_qm"] = self.proj3d_qm(
                pooled_qm.reshape(batch_size * NUM_TASKS, -1)
            ).reshape(batch_size, NUM_TASKS, -1)
        return modality_parts

    def _three_d_view_consistency_loss(
        self,
        *,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
        mol_ids: Optional[Sequence[Any]],
    ) -> torch.Tensor:
        enabled_modalities = self._enabled_modalities_for_current_stage()
        if ("3d_geom" not in enabled_modalities) and ("3d_qm" not in enabled_modalities):
            return self.lam.new_zeros(())
        if (self.proj3d_geom is None) and (self.proj3d_qm is None):
            return self.lam.new_zeros(())
        real_valid = self._non_dummy_valid_instance_mask(
            x3d_pad=x3d_pad,
            key_padding_mask=key_padding_mask,
        )
        n_real_conformers = real_valid.sum(dim=1)
        eligible_samples = n_real_conformers >= 2
        if not bool(torch.any(eligible_samples)):
            return self.lam.new_zeros(())
        effective_kpm = key_padding_mask | (~real_valid)
        mask_a = self._sample_subset_padding_mask(
            key_padding_mask=effective_kpm,
            keep_rate=float(self.consistency_view_keep_rate),
        )
        mask_b = self._sample_subset_padding_mask(
            key_padding_mask=effective_kpm,
            keep_rate=float(self.consistency_view_keep_rate),
        )
        pooled_geom_a, pooled_qm_a, _, _ = self._pool_task_tokens(
            x3d_pad=x3d_pad,
            key_padding_mask=mask_a,
            return_attn=False,
        )
        pooled_geom_b, pooled_qm_b, _, _ = self._pool_task_tokens(
            x3d_pad=x3d_pad,
            key_padding_mask=mask_b,
            return_attn=False,
        )
        proj_a = self._project_contrastive_latents(
            self._summary_latents_from_modality_parts(
                self._project_3d_modality_parts(pooled_geom=pooled_geom_a, pooled_qm=pooled_qm_a)
            )
        )
        proj_b = self._project_contrastive_latents(
            self._summary_latents_from_modality_parts(
                self._project_3d_modality_parts(pooled_geom=pooled_geom_b, pooled_qm=pooled_qm_b)
            )
        )
        if len(proj_a) == 0 or len(proj_b) == 0:
            return self.lam.new_zeros(())
        ref = next(iter(proj_a.values()))
        positive_mask = self._positive_mask_from_ids(
            batch_size=int(ref.shape[0]),
            device=ref.device,
            mol_ids=mol_ids,
        )
        positive_mask = self._mask_positive_pairs_by_sample_eligibility(positive_mask, eligible_samples)
        losses: List[torch.Tensor] = []
        for name in ("3d_geom", "3d_qm"):
            if name in proj_a and name in proj_b:
                losses.append(
                    self._contrastive_pair_loss(
                        za=proj_a[name],
                        zb=proj_b[name],
                        positive_mask=positive_mask,
                    )
                )
        if len(losses) == 0:
            return self.lam.new_zeros(())
        return torch.stack(losses, dim=0).mean()

    def _supervised_task_contrastive_loss(
        self,
        *,
        z_tasks: torch.Tensor,
        y_cls: torch.Tensor,
        w_cls: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if z_tasks.ndim != 3:
            raise ValueError(f"Expected z_tasks shape [B,T,H], got {tuple(z_tasks.shape)}")
        batch_size, n_tasks, _ = z_tasks.shape
        if int(batch_size) <= 1 or int(n_tasks) <= 0:
            return z_tasks.new_zeros(())
        proj = self.task_contrastive_head(
            z_tasks.reshape(batch_size * n_tasks, -1)
        ).reshape(batch_size, n_tasks, -1)
        proj = F.normalize(proj, p=2, dim=-1)

        task_losses: List[torch.Tensor] = []
        for task_idx in range(int(n_tasks)):
            anchor_weight = None
            if w_cls is not None:
                anchor_weight = w_cls[:, task_idx]
            task_loss = self._task_supervised_contrastive_loss(
                z_task=proj[:, task_idx, :],
                labels=y_cls[:, task_idx],
                anchor_weight=anchor_weight,
            )
            labels = (y_cls[:, task_idx] > 0.5).long()
            has_both_classes = int(torch.unique(labels).numel()) >= 2
            if has_both_classes and bool(torch.isfinite(task_loss)):
                task_losses.append(task_loss)
        if len(task_losses) == 0:
            return z_tasks.new_zeros(())
        return torch.stack(task_losses, dim=0).mean()

    def _compute_outputs(
        self,
        *,
        x2d: torch.Tensor,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
        need_attn: bool,
    ) -> Dict[str, Any]:
        pooled_geom, pooled_qm, attn_geom, attn_qm = self._pool_task_tokens(
            x3d_pad=x3d_pad,
            key_padding_mask=key_padding_mask,
            return_attn=need_attn,
        )
        z_tasks, fusion_info = self._build_task_representations(
            x2d=x2d,
            x3d_pad=x3d_pad,
            key_padding_mask=key_padding_mask,
            pooled_geom=pooled_geom,
            pooled_qm=pooled_qm,
        )
        logits = apply_task_heads(z_tasks, self.cls_heads)
        z_aux = z_tasks.mean(dim=1)
        abs_out = apply_shared_heads(z_aux, self.abs_heads)
        fluo_out = apply_shared_heads(z_aux, self.fluo_heads)
        bitmask_logits = self.bitmask_head(z_aux) if self.bitmask_head is not None else None
        return {
            "logits": logits,
            "abs_out": abs_out,
            "fluo_out": fluo_out,
            "bitmask_logits": bitmask_logits,
            "z_tasks": z_tasks,
            "fusion_info": fusion_info,
            "attn_geom": attn_geom,
            "attn_qm": attn_qm,
        }

    def training_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo = batch[:11]
        self._accumulate_real_3d_counts(split="train", x3d_pad=x3d, key_padding_mask=kpm)
        mol_ids = batch[11] if len(batch) >= 13 else None
        conf_pad = batch[12] if len(batch) >= 13 else None
        use_attn_guidance = bool(self.rl_enabled and mol_ids is not None and conf_pad is not None)

        forward_out = self._compute_outputs(
            x2d=x2d,
            x3d_pad=x3d,
            key_padding_mask=kpm,
            need_attn=bool(use_attn_guidance),
        )
        logits = forward_out["logits"]
        abs_out = forward_out["abs_out"]
        fluo_out = forward_out["fluo_out"]
        bitmask_logits = forward_out["bitmask_logits"]
        attn = None
        if use_attn_guidance:
            attn = {
                "attn_geom": forward_out.get("attn_geom"),
                "attn_qm": forward_out.get("attn_qm"),
                "modality_gates": forward_out["fusion_info"].get("modality_gates"),
                "modality_scores": forward_out["fusion_info"].get("modality_scores"),
                "modality_order": forward_out["fusion_info"].get("modality_order"),
                "modality_attn": forward_out["fusion_info"].get("modality_attn"),
            }
        bitmask_targets = self._bitmask_group_targets(y_cls)

        with autocast(enabled=False):
            losses = compute_training_losses(
                cls_loss_fn=self.cls_loss,
                logits=logits,
                y_cls=y_cls,
                w_cls=w_cls,
                lam=self.lam,
                abs_out=abs_out,
                y_abs=y_abs,
                m_abs=m_abs,
                w_abs=w_abs,
                fluo_out=fluo_out,
                y_fluo=y_fluo,
                m_fluo=m_fluo,
                w_fluo=w_fluo,
                bitmask_logits=bitmask_logits,
                bitmask_targets=bitmask_targets,
                bitmask_class_weight=(
                    self.bitmask_group_class_weight
                    if self.bitmask_head is not None
                    else None
                ),
                reg_loss_type=self.reg_loss_type,
                lambda_aux_abs=self.lambda_aux_abs,
                lambda_aux_fluo=self.lambda_aux_fluo,
                lambda_aux_bitmask=self.lambda_aux_bitmask,
                task_loss_log_vars=(self.task_loss_log_vars if self.learnable_task_uncertainty else None),
                task_uncertainty_reg=float(self.task_uncertainty_reg),
            )
            contrastive_cross_modal = torch.zeros((), dtype=losses.total.dtype, device=losses.total.device)
            contrastive_3d_consistency = torch.zeros((), dtype=losses.total.dtype, device=losses.total.device)
            contrastive_supervised = torch.zeros((), dtype=losses.total.dtype, device=losses.total.device)
            if float(self.lambda_contrastive_cross_modal) > 0.0:
                contrastive_cross_modal = self._cross_modal_contrastive_loss(
                    fusion_info=forward_out["fusion_info"],
                    mol_ids=mol_ids,
                    x3d_pad=x3d,
                    key_padding_mask=kpm,
                ).to(dtype=losses.total.dtype)
            if float(self.lambda_contrastive_3d_consistency) > 0.0:
                contrastive_3d_consistency = self._three_d_view_consistency_loss(
                    x3d_pad=x3d,
                    key_padding_mask=kpm,
                    mol_ids=mol_ids,
                ).to(dtype=losses.total.dtype)
            if float(self.lambda_contrastive_supervised) > 0.0:
                contrastive_supervised = self._supervised_task_contrastive_loss(
                    z_tasks=forward_out["z_tasks"],
                    y_cls=y_cls,
                    w_cls=w_cls,
                ).to(dtype=losses.total.dtype)
            weighted_contrastive_cross_modal = float(self.lambda_contrastive_cross_modal) * contrastive_cross_modal
            weighted_contrastive_3d_consistency = (
                float(self.lambda_contrastive_3d_consistency) * contrastive_3d_consistency
            )
            weighted_contrastive_supervised = (
                float(self.lambda_contrastive_supervised) * contrastive_supervised
            )
            concept_alignment = torch.zeros((), dtype=losses.total.dtype, device=losses.total.device)
            concept_bonus = torch.zeros((), dtype=losses.total.dtype, device=losses.total.device)
            if use_attn_guidance and attn is not None and float(self.rl_guidance_scale) > 0.0:
                concept_alignment = self._concept_alignment_score(
                    y_cls=y_cls,
                    attn=attn,
                    key_padding_mask=kpm,
                    mol_ids=[str(x) for x in mol_ids],
                    conf_pad=conf_pad,
                )
                concept_bonus = float(self.rl_guidance_scale) * concept_alignment
            total_loss = (
                losses.total
                + weighted_contrastive_cross_modal
                + weighted_contrastive_3d_consistency
                + weighted_contrastive_supervised
                - concept_bonus
            )

        bs = int(y_cls.shape[0])
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_cls_loss", losses.cls, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_bitmask_loss", losses.bitmask, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_per_task_loss_mean", losses.per_task.mean(), on_step=False, on_epoch=True, batch_size=bs)
        self.log(
            "train_weighted_per_task_loss_mean",
            losses.weighted_per_task.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_base_weighted_per_task_loss_mean",
            losses.base_weighted_per_task.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_task_uncertainty_reg_mean",
            losses.task_uncertainty_reg.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        for task_idx in range(NUM_TASKS):
            self.log(
                f"train_task_loss_t{int(task_idx)}",
                losses.per_task[int(task_idx)],
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )
            self.log(
                f"train_task_weighted_loss_t{int(task_idx)}",
                losses.weighted_per_task[int(task_idx)],
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )
            self.log(
                f"train_task_base_weighted_loss_t{int(task_idx)}",
                losses.base_weighted_per_task[int(task_idx)],
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )
        if self.learnable_task_uncertainty and self.task_loss_log_vars is not None:
            task_precision = torch.exp(-self.task_loss_log_vars.detach())
            for task_idx in range(NUM_TASKS):
                self.log(
                    f"train_task_log_var_t{int(task_idx)}",
                    self.task_loss_log_vars[int(task_idx)].detach(),
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                )
                self.log(
                    f"train_task_precision_t{int(task_idx)}",
                    task_precision[int(task_idx)],
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                )
        self.log("train_concept_alignment", concept_alignment, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_concept_bonus", concept_bonus, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_rl_guidance_scale", float(self.rl_guidance_scale), on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_contrastive_cross_modal", contrastive_cross_modal, on_step=False, on_epoch=True, batch_size=bs)
        self.log(
            "train_contrastive_cross_modal_weighted",
            weighted_contrastive_cross_modal,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_contrastive_3d_consistency",
            contrastive_3d_consistency,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_contrastive_3d_consistency_weighted",
            weighted_contrastive_3d_consistency,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_contrastive_supervised",
            contrastive_supervised,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_contrastive_supervised_weighted",
            weighted_contrastive_supervised,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        if bool(self.automatic_optimization):
            return total_loss

        opt = self.optimizers()
        accum_steps = int(self._gradient_accumulation_steps())
        if (int(batch_idx) % accum_steps) == 0:
            opt.zero_grad()

        shared_params = self._shared_training_parameters()
        task_grad_lists: List[Sequence[Optional[torch.Tensor]]] = []
        if len(shared_params) > 0 and str(self.multitask_gradient_mode) == "pcgrad_shared":
            for task_idx in range(NUM_TASKS):
                task_component = losses.weighted_per_task[int(task_idx)] / float(NUM_TASKS * accum_steps)
                task_grads = torch.autograd.grad(
                    task_component,
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                task_grad_lists.append(task_grads)
            self._log_task_gradient_diagnostics(task_grads=task_grad_lists, batch_size=int(bs))

        scaled_total_loss = total_loss / float(accum_steps)
        self.manual_backward(scaled_total_loss)

        if len(shared_params) > 0 and len(task_grad_lists) > 0 and str(self.multitask_gradient_mode) == "pcgrad_shared":
            full_shared_grads = [
                (None if param.grad is None else param.grad.detach().clone())
                for param in shared_params
            ]
            classification_shared_grads = self._sum_grad_lists(task_grad_lists)
            extra_shared_grads = self._subtract_grad_lists(full_shared_grads, classification_shared_grads)
            merged_task_grads = self._pcgrad_merge(task_grad_lists)
            merged_shared_grads = self._sum_grad_lists([merged_task_grads, extra_shared_grads])
            for param, merged_grad in zip(shared_params, merged_shared_grads):
                if merged_grad is None:
                    param.grad = None
                else:
                    param.grad = merged_grad.to(device=param.device, dtype=param.dtype)

        if self._should_step_optimizer(int(batch_idx)):
            opt.step()
            opt.zero_grad()
        return total_loss.detach()

    def on_validation_epoch_start(self):
        self._val_p, self._val_y, self._val_w = [], [], []
        self._val_real_3d_with = 0
        self._val_real_3d_without = 0

    def validation_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, *_ = batch
        self._accumulate_real_3d_counts(split="val", x3d_pad=x3d, key_padding_mask=kpm)
        forward_out = self._compute_outputs(
            x2d=x2d,
            x3d_pad=x3d,
            key_padding_mask=kpm,
            need_attn=False,
        )
        logits = forward_out["logits"]

        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = y_cls.detach().cpu().numpy().astype(int)
        w = w_cls.detach().cpu().numpy().astype(np.float32)

        self._val_p.append(p)
        self._val_y.append(y)
        self._val_w.append(w)

    def on_validation_epoch_end(self):
        self._log_real_3d_epoch_summary(split="val")
        if not self._val_p:
            return
        p_all = np.concatenate(self._val_p, axis=0)
        y_all = np.concatenate(self._val_y, axis=0).astype(int)
        w_all = np.concatenate(self._val_w, axis=0).astype(np.float32)

        aps = ap_per_task(y_all, p_all, w_cls=w_all, weighted_tasks=(0, 1))
        macro_ap = float(np.mean(aps))
        min_ap = float(np.min(aps))
        if str(self.objective_mode) in {"macro_plus_min"}:
            objective_ap = float((1.0 - float(self.objective_min_w)) * macro_ap + float(self.objective_min_w) * min_ap)
        else:
            objective_ap = float(macro_ap)

        stage_mode = str(self._stage_mode_for_epoch(int(getattr(self, "current_epoch", 0))))
        has_2d = bool("2d" in self.active_modalities)
        has_3d = bool(("3d_geom" in self.active_modalities) or ("3d_qm" in self.active_modalities))
        is_stagewise_multimodal = bool(self._stagewise_active and has_2d and has_3d)
        stage_is_selection_eligible = (not is_stagewise_multimodal) or (stage_mode == "joint")
        monitored_objective_ap = float(objective_ap if stage_is_selection_eligible else -1.0)

        for task_idx in range(NUM_TASKS):
            self.log(f"val_ap_{task_idx}", float(aps[task_idx]), prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_macro_ap", float(macro_ap), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_min_ap", float(min_ap), prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_objective_ap", float(monitored_objective_ap), prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_objective_ap_stage", float(objective_ap), prog_bar=False, on_step=False, on_epoch=True)
        log_event(
            "INFO",
            "mil.val.epoch_metrics",
            epoch=int(getattr(self, "current_epoch", 0)),
            stage_mode=str(stage_mode),
            stage_selection_eligible=bool(stage_is_selection_eligible),
            val_macro_ap=float(macro_ap),
            val_min_ap=float(min_ap),
            val_objective_ap=float(objective_ap),
            val_objective_ap_monitored=float(monitored_objective_ap),
            val_ap_t0=float(aps[0]),
            val_ap_t1=float(aps[1]),
            val_ap_t2=float(aps[2]),
            val_ap_t3=float(aps[3]),
        )

    def configure_optimizers(self):
        param_groups: Dict[str, List[torch.nn.Parameter]] = {"2d": [], "3d": [], "fusion": [], "heads": []}
        for name, param in self.named_parameters():
            group = self._parameter_group_name(str(name))
            param_groups[group].append(param)

        optimizer_groups: List[Dict[str, Any]] = []
        active_group_names: List[str] = []
        for group_name in ("2d", "3d", "fusion", "heads"):
            params = param_groups[group_name]
            if len(params) == 0:
                continue
            active_group_names.append(str(group_name))
            optimizer_groups.append(
                {
                    "params": params,
                    "lr": float(self.lr) * float(self.lr_group_scales[group_name]),
                    "weight_decay": float(self.weight_decay) * float(self.weight_decay_group_scales[group_name]),
                }
            )
        self._optimizer_group_names = tuple(active_group_names)
        group_summaries = [s for s in self.get_optimizer_group_summaries() if str(s["group"]) in set(self._optimizer_group_names)]
        total_params = int(sum(int(s["n_params"]) for s in group_summaries))
        total_trainable_params = int(sum(int(s["n_trainable_params"]) for s in group_summaries))
        log_event(
            "INFO",
            "mil.optimizer.plan",
            optimizer="AdamW",
            n_groups=int(len(self._optimizer_group_names)),
            group_order=",".join(self._optimizer_group_names) if len(self._optimizer_group_names) > 0 else "none",
            total_params=int(total_params),
            total_trainable_params=int(total_trainable_params),
        )
        for summary in group_summaries:
            log_event("INFO", "mil.optimizer.group", **summary)
        return torch.optim.AdamW(optimizer_groups, lr=self.lr, weight_decay=self.weight_decay)
