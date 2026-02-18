from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ...losses.multi_task_focal import MultiTaskFocal
from ...utils.metrics import ap_per_task
from ...utils.mlp import make_mlp
from ..task_attention import TaskAttentionPool
from .constants import NUM_ABS_HEADS, NUM_FLUO_HEADS, NUM_TASKS
from .heads import apply_shared_heads, apply_task_heads, make_linear_heads, make_projection
from .training import compute_training_losses


class MILTaskAttnMixerWithAux(pl.LightningModule):
    """
    - 2D MLP encoder -> e2d
    - 3D MLP encoder -> tokens
    - task-specific attention pooling -> pooled per task + attn maps
    - project 2D and pooled-3D to same dim, concat, mixer -> z_task
    - cls logits from task-specific z_task
    - aux heads from mean(z_task)
    """

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
        reg_loss_type: str,
        activation: str = "GELU",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight", "gamma", "lam"])

        self.mol_enc = make_mlp(
            int(mol_dim),
            int(mol_hidden),
            int(mol_layers),
            float(mol_dropout),
            activation=str(activation),
        )
        self.inst_enc = make_mlp(
            int(inst_dim),
            int(inst_hidden),
            int(inst_layers),
            float(inst_dropout),
            activation=str(activation),
        )

        self.attn_pool = TaskAttentionPool(
            dim=int(inst_hidden),
            n_heads=int(attn_heads),
            dropout=float(attn_dropout),
            n_tasks=NUM_TASKS,
        )

        self.proj2d = make_projection(int(mol_hidden), int(proj_dim))
        self.proj3d = make_projection(int(inst_hidden), int(proj_dim))

        self.mixer = make_mlp(
            int(2 * proj_dim),
            int(mixer_hidden),
            int(mixer_layers),
            float(mixer_dropout),
            activation=str(activation),
        )

        self.cls_heads = make_linear_heads(int(mixer_hidden), NUM_TASKS)
        self.abs_heads = make_linear_heads(int(mixer_hidden), NUM_ABS_HEADS)
        self.fluo_heads = make_linear_heads(int(mixer_hidden), NUM_FLUO_HEADS)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pooled_tasks, attn = self._pool_task_tokens(x3d_pad=x3d_pad, key_padding_mask=key_padding_mask, return_attn=return_attn)
        z_tasks = self._build_task_representations(x2d=x2d, pooled_tasks=pooled_tasks)

        logits = apply_task_heads(z_tasks, self.cls_heads)  # [B,4]

        z_aux = z_tasks.mean(dim=1)  # [B,mixer_hidden]
        abs_out = apply_shared_heads(z_aux, self.abs_heads)    # [B,2]
        fluo_out = apply_shared_heads(z_aux, self.fluo_heads)  # [B,4]

        if return_attn:
            return logits, abs_out, fluo_out, attn
        return logits, abs_out, fluo_out

    def _pool_task_tokens(
        self,
        *,
        x3d_pad: torch.Tensor,
        key_padding_mask: torch.Tensor,
        return_attn: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, n_instances, feature_dim = x3d_pad.shape
        tok = self.inst_enc(x3d_pad.reshape(batch_size * n_instances, feature_dim)).reshape(batch_size, n_instances, -1)
        tok = F.layer_norm(tok, (tok.shape[-1],))
        return self.attn_pool(tok, key_padding_mask=key_padding_mask, return_attn=return_attn)

    def _build_task_representations(self, *, x2d: torch.Tensor, pooled_tasks: torch.Tensor) -> torch.Tensor:
        batch_size = x2d.shape[0]
        e2d = self.proj2d(self.mol_enc(x2d))  # [B,proj]
        e2d_rep = e2d.unsqueeze(1).expand(-1, NUM_TASKS, -1)  # [B,4,proj]

        e3d = self.proj3d(pooled_tasks.reshape(batch_size * NUM_TASKS, -1)).reshape(batch_size, NUM_TASKS, -1)  # [B,4,proj]

        mix_in = torch.cat([e2d_rep, e3d], dim=2).reshape(batch_size * NUM_TASKS, -1)  # [B*4,2*proj]
        return self.mixer(mix_in).reshape(batch_size, NUM_TASKS, -1)  # [B,4,mixer_hidden]

    def training_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo = batch
        logits, abs_out, fluo_out = self(x2d, x3d, kpm, return_attn=False)

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
                reg_loss_type=self.reg_loss_type,
                lambda_aux_abs=self.lambda_aux_abs,
                lambda_aux_fluo=self.lambda_aux_fluo,
            )

        bs = int(y_cls.shape[0])
        self.log("train_loss", losses.total, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_cls_loss", losses.cls, on_step=False, on_epoch=True, batch_size=bs)
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
