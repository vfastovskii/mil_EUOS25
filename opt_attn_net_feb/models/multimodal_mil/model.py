from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
    - 3D embedder -> tokens
    - 3D aggregator -> pooled per task + attn maps
    - project 2D and pooled-3D to same dim, concat, mixer -> z_task
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

        self.mol_enc = build_2d_embedder(
            name=str(mol_embedder_name),
            input_dim=int(mol_dim),
            hidden_dim=int(mol_hidden),
            layers=int(mol_layers),
            dropout=float(mol_dropout),
            activation=str(activation),
        )
        self.inst_enc = build_3d_embedder(
            name=str(inst_embedder_name),
            input_dim=int(inst_dim),
            hidden_dim=int(inst_hidden),
            layers=int(inst_layers),
            dropout=float(inst_dropout),
            activation=str(activation),
        )
        self.mol_post_embed_norm = nn.LayerNorm(int(mol_hidden))
        self.inst_post_embed_norm = nn.LayerNorm(int(inst_hidden))

        agg_kwargs = dict(aggregator_kwargs or {})
        overlap = {"dim", "n_heads", "dropout", "n_tasks"}.intersection(agg_kwargs.keys())
        if overlap:
            raise ValueError(f"aggregator_kwargs cannot override reserved keys: {sorted(overlap)}")
        self.attn_pool = build_aggregator(
            name=str(aggregator_name),
            dim=int(inst_hidden),
            n_heads=int(attn_heads),
            dropout=float(attn_dropout),
            n_tasks=NUM_TASKS,
            **agg_kwargs,
        )
        self.agg_post_norm = nn.LayerNorm(int(inst_hidden))

        self.proj2d = make_projection(int(mol_hidden), int(proj_dim))
        self.proj3d = make_projection(int(inst_hidden), int(proj_dim))

        self.mixer = build_mlp_v3_embedder(
            input_dim=int(2 * proj_dim),
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

    def forward(
        self,
        x2d: torch.Tensor,              # [B,F2]
        x3d_pad: torch.Tensor,          # [B,N,F3]
        key_padding_mask: torch.Tensor, # [B,N] True=PAD
        return_attn: bool = False,
        return_bitmask: bool = False,
    ):
        self._validate_forward_inputs(x2d=x2d, x3d_pad=x3d_pad, key_padding_mask=key_padding_mask)
        pooled_tasks, attn = self._pool_task_tokens(x3d_pad=x3d_pad, key_padding_mask=key_padding_mask, return_attn=return_attn)
        z_tasks = self._build_task_representations(x2d=x2d, pooled_tasks=pooled_tasks)

        logits = apply_task_heads(z_tasks, self.cls_heads)  # [B,4]

        z_aux = z_tasks.mean(dim=1)  # [B,mixer_hidden]
        abs_out = apply_shared_heads(z_aux, self.abs_heads)    # [B,2]
        fluo_out = apply_shared_heads(z_aux, self.fluo_heads)  # [B,4]
        bitmask_logits = self.bitmask_head(z_aux) if self.bitmask_head is not None else None

        if return_attn and return_bitmask:
            return logits, abs_out, fluo_out, bitmask_logits, attn
        if return_attn:
            return logits, abs_out, fluo_out, attn
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

    def _pool_task_tokens(
        self,
        *,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
        return_attn: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, n_instances, feature_dim = x3d_pad.shape
        tok = self.inst_enc(x3d_pad.reshape(batch_size * n_instances, feature_dim)).reshape(batch_size, n_instances, -1)
        tok = self.inst_post_embed_norm(tok)
        return self.attn_pool(tok, key_padding_mask=key_padding_mask, return_attn=return_attn)

    def _build_task_representations(self, *, x2d: torch.Tensor, pooled_tasks: torch.Tensor) -> torch.Tensor:
        batch_size = x2d.shape[0]
        mol_emb = self.mol_post_embed_norm(self.mol_enc(x2d))
        e2d = self.proj2d(mol_emb)  # [B,proj]
        e2d_rep = e2d.unsqueeze(1).expand(-1, NUM_TASKS, -1)  # [B,4,proj]

        pooled_tasks = self.agg_post_norm(pooled_tasks)
        e3d = self.proj3d(pooled_tasks.reshape(batch_size * NUM_TASKS, -1)).reshape(batch_size, NUM_TASKS, -1)  # [B,4,proj]

        mix_in = torch.cat([e2d_rep, e3d], dim=2).reshape(batch_size * NUM_TASKS, -1)  # [B*4,2*proj]
        z_tasks = self.mixer(mix_in).reshape(batch_size, NUM_TASKS, -1)  # [B,4,mixer_hidden]
        return self.mixer_post_norm(z_tasks)

    def training_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo = batch
        logits, abs_out, fluo_out, bitmask_logits = self(
            x2d,
            x3d,
            kpm,
            return_attn=False,
            return_bitmask=True,
        )
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

        bs = int(y_cls.shape[0])
        self.log("train_loss", losses.total, on_step=False, on_epoch=True, batch_size=bs)
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
        return losses.total

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
