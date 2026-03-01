from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from ...losses.multi_task_focal import MultiTaskFocal
from ...utils.metrics import ap_per_task
from .aggregators import build_aggregator
from .configs import MILModelConfig
from .constants import NUM_ABS_HEADS, NUM_FLUO_HEADS, NUM_TASKS
from .embedder_mlp_v3_base import build_mlp_v3_embedder
from .embedders import build_2d_embedder, build_3d_embedder
from .head_mlp_v3 import MLPPredictorV3Like
from .head_utils import apply_shared_heads, apply_task_heads, make_projection
from .predictors import build_predictor_heads
from .training import compute_training_losses


class MILTaskAttnMixerWithAux(pl.LightningModule):
    """
    - 2D embedder -> e2d (no aggregator)
    - 3D geometry embedder -> tokens -> geometry aggregator
    - 3D quantum embedder -> tokens -> quantum aggregator
    - project 2D/3D-geom/3D-qm to same dim, concat, mixer -> z_task
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
            pos_weight=pos_weight,
            gamma=gamma,
            lam=lam,
            lambda_aux_abs=float(loss.lambda_aux_abs),
            lambda_aux_fluo=float(loss.lambda_aux_fluo),
            lambda_aux_bitmask=float(loss.lambda_aux_bitmask),
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
            predictor_name=str(h.predictor_name),
            head_num_layers=int(h.num_layers),
            head_dropout=float(h.dropout),
            head_stochastic_depth=float(h.stochastic_depth),
            head_fc2_gain_non_last=float(h.fc2_gain_non_last),
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
        pos_weight: torch.Tensor,
        gamma: torch.Tensor,
        lam: np.ndarray,
        lambda_aux_abs: float,
        lambda_aux_fluo: float,
        lambda_aux_bitmask: float,
        reg_loss_type: str,
        bitmask_group_top_ids: Optional[List[int]] = None,
        bitmask_group_class_weight: Optional[List[float]] = None,
        activation: str = "GELU",
        mol_embedder_name: str = "mlp_v3_2d",
        inst_embedder_name: str = "mlp_v3_3d",
        aggregator_name: str = "task_attention_pool",
        aggregator_kwargs: Optional[Dict[str, Any]] = None,
        predictor_name: str = "mlp_v3",
        head_num_layers: int = 2,
        head_dropout: float = 0.1,
        head_stochastic_depth: float = 0.1,
        head_fc2_gain_non_last: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight", "gamma", "lam"])

        self.mol_dim = int(mol_dim)
        self.inst_dim = int(inst_dim)
        self.inst_geom_dim = int(inst_geom_dim)
        self.inst_qm_dim = int(inst_qm_dim)
        self.inst_hidden = int(inst_hidden)
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

        # Optional concept-guidance RL controls (configured externally for final runs).
        self.rl_enabled: bool = False
        self.rl_guidance_scale: float = 0.0
        self.rl_guidance_scale_max: float = 0.2
        self.rl_negative_penalty: float = 0.15
        self.rl_task_target_concepts: list[tuple[str, ...]] = [tuple() for _ in range(NUM_TASKS)]
        self.rl_target_conf_pairs_by_task: list[set[tuple[str, str]]] = [set() for _ in range(NUM_TASKS)]
        self.rl_target_mols_by_task: list[set[str]] = [set() for _ in range(NUM_TASKS)]

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
        attn: torch.Tensor,
        key_padding_mask: torch.Tensor,
        mol_ids: Sequence[str],
        conf_pad: np.ndarray,
    ) -> torch.Tensor:
        """Average attention mass on task-target concepts for positive labels."""
        if (not self.rl_enabled) or attn is None:
            return torch.zeros((), dtype=y_cls.dtype, device=y_cls.device)

        total = torch.zeros((), dtype=attn.dtype, device=attn.device)
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
                w = attn[b, t, :L]
                w = w / (w.sum() + 1e-8)

                target_pairs = self.rl_target_conf_pairs_by_task[t]
                idx = [i for i, cid in enumerate(confs) if (mol_id, str(cid)) in target_pairs]
                if not idx:
                    continue
                idx_t = torch.tensor(idx, dtype=torch.long, device=attn.device)
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
            return torch.zeros((), dtype=attn.dtype, device=attn.device)
        return total / float(denom)

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
        pooled_geom, pooled_qm, attn_geom, attn_qm = self._pool_task_tokens(
            x3d_pad=x3d_pad,
            key_padding_mask=key_padding_mask,
            return_attn=need_attn,
        )
        z_tasks = self._build_task_representations(
            x2d=x2d,
            pooled_geom=pooled_geom,
            pooled_qm=pooled_qm,
        )

        logits = apply_task_heads(z_tasks, self.cls_heads)  # [B,4]

        z_aux = z_tasks.mean(dim=1)  # [B,mixer_hidden]
        abs_out = apply_shared_heads(z_aux, self.abs_heads)    # [B,2]
        fluo_out = apply_shared_heads(z_aux, self.fluo_heads)  # [B,4]
        bitmask_logits = self.bitmask_head(z_aux) if self.bitmask_head is not None else None

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
        pooled_geom: torch.Tensor,
        pooled_qm: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x2d.shape[0]
        mix_parts: list[torch.Tensor] = []
        if self.mol_enc is not None and self.mol_post_embed_norm is not None and self.proj2d is not None:
            mol_emb = self.mol_post_embed_norm(self.mol_enc(x2d))
            e2d = self.proj2d(mol_emb)  # [B,proj]
            e2d_rep = e2d.unsqueeze(1).expand(-1, NUM_TASKS, -1)  # [B,4,proj]
            mix_parts.append(e2d_rep)
        if self.proj3d_geom is not None and self.agg_geom_post_norm is not None:
            pooled_geom = self.agg_geom_post_norm(pooled_geom)
            e3d_geom = self.proj3d_geom(
                pooled_geom.reshape(batch_size * NUM_TASKS, -1)
            ).reshape(batch_size, NUM_TASKS, -1)
            mix_parts.append(e3d_geom)
        if self.proj3d_qm is not None and self.agg_qm_post_norm is not None:
            pooled_qm = self.agg_qm_post_norm(pooled_qm)
            e3d_qm = self.proj3d_qm(
                pooled_qm.reshape(batch_size * NUM_TASKS, -1)
            ).reshape(batch_size, NUM_TASKS, -1)
            mix_parts.append(e3d_qm)
        if len(mix_parts) == 0:
            raise RuntimeError("No active modality projections found for mixer input.")

        mix_in = torch.cat(mix_parts, dim=2).reshape(batch_size * NUM_TASKS, -1)
        z_tasks = self.mixer(mix_in).reshape(batch_size, NUM_TASKS, -1)  # [B,4,mixer_hidden]
        return self.mixer_post_norm(z_tasks)

    def training_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo = batch[:11]
        mol_ids = batch[11] if len(batch) >= 13 else None
        conf_pad = batch[12] if len(batch) >= 13 else None
        use_attn_guidance = bool(self.rl_enabled and mol_ids is not None and conf_pad is not None)

        if use_attn_guidance:
            logits, abs_out, fluo_out, bitmask_logits, attn = self(
                x2d,
                x3d,
                kpm,
                return_attn=True,
                return_bitmask=True,
            )
        else:
            logits, abs_out, fluo_out, bitmask_logits = self(
                x2d,
                x3d,
                kpm,
                return_attn=False,
                return_bitmask=True,
            )
            attn = None
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
            total_loss = losses.total - concept_bonus

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
        self.log("train_concept_alignment", concept_alignment, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_concept_bonus", concept_bonus, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_rl_guidance_scale", float(self.rl_guidance_scale), on_step=False, on_epoch=True, batch_size=bs)
        return total_loss

    def on_validation_epoch_start(self):
        self._val_p, self._val_y, self._val_w = [], [], []

    def validation_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, *_ = batch
        logits, _, _ = self(x2d, x3d, kpm, return_attn=False)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = y_cls.detach().cpu().numpy().astype(int)
        w = w_cls.detach().cpu().numpy().astype(np.float32)

        self._val_p.append(p)
        self._val_y.append(y)
        self._val_w.append(w)

    def on_validation_epoch_end(self):
        if not self._val_p:
            return
        p_all = np.concatenate(self._val_p, axis=0)
        y_all = np.concatenate(self._val_y, axis=0).astype(int)
        w_all = np.concatenate(self._val_w, axis=0).astype(np.float32)

        aps = ap_per_task(y_all, p_all, w_cls=w_all, weighted_tasks=(0, 1))
        macro_ap = float(np.mean(aps))
        min_ap = float(np.min(aps))

        for task_idx in range(NUM_TASKS):
            self.log(f"val_ap_{task_idx}", float(aps[task_idx]), prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_macro_ap", float(macro_ap), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_min_ap", float(min_ap), prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
