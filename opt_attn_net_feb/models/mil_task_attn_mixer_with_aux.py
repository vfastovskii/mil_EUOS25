from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.cuda.amp import autocast

from ..utils.mlp import make_mlp
from ..models.task_attention_pool import TaskAttentionPool
from ..losses.multi_task_focal import MultiTaskFocal
from ..losses.regression import reg_loss_weighted
from ..utils.metrics import ap_per_task


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

        self.mol_enc = make_mlp(int(mol_dim), int(mol_hidden), int(mol_layers), float(mol_dropout), activation=str(activation))
        self.inst_enc = make_mlp(int(inst_dim), int(inst_hidden), int(inst_layers), float(inst_dropout), activation=str(activation))

        self.attn_pool = TaskAttentionPool(dim=int(inst_hidden), n_heads=int(attn_heads), dropout=float(attn_dropout), n_tasks=4)

        self.proj2d = nn.Sequential(nn.Linear(int(mol_hidden), int(proj_dim)), nn.LayerNorm(int(proj_dim)))
        self.proj3d = nn.Sequential(nn.Linear(int(inst_hidden), int(proj_dim)), nn.LayerNorm(int(proj_dim)))

        self.mixer = make_mlp(int(2 * proj_dim), int(mixer_hidden), int(mixer_layers), float(mixer_dropout), activation=str(activation))

        self.cls_heads = nn.ModuleList([nn.Linear(int(mixer_hidden), 1) for _ in range(4)])
        self.abs_heads = nn.ModuleList([nn.Linear(int(mixer_hidden), 1) for _ in range(2)])
        self.fluo_heads = nn.ModuleList([nn.Linear(int(mixer_hidden), 1) for _ in range(4)])

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
        B, N, F3 = x3d_pad.shape
        tok = self.inst_enc(x3d_pad.reshape(B * N, F3)).reshape(B, N, -1)  # [B,N,inst_hidden]
        tok = F.layer_norm(tok, (tok.shape[-1],))

        pooled_tasks, attn = self.attn_pool(tok, key_padding_mask=key_padding_mask, return_attn=return_attn)
        # pooled_tasks: [B,4,inst_hidden]; attn: [B,4,N] if return_attn else None

        e2d = self.proj2d(self.mol_enc(x2d))  # [B,proj]
        e2d_rep = e2d.unsqueeze(1).expand(-1, 4, -1)  # [B,4,proj]

        e3d = self.proj3d(pooled_tasks.reshape(B * 4, -1)).reshape(B, 4, -1)  # [B,4,proj]

        mix_in = torch.cat([e2d_rep, e3d], dim=2).reshape(B * 4, -1)  # [B*4,2*proj]
        z_tasks = self.mixer(mix_in).reshape(B, 4, -1)  # [B,4,mixer_hidden]

        logits = torch.cat([self.cls_heads[t](z_tasks[:, t, :]) for t in range(4)], dim=1)  # [B,4]

        z_aux = z_tasks.mean(dim=1)  # [B,mixer_hidden]  # allow aux losses to train shared representation
        abs_out = torch.cat([hd(z_aux) for hd in self.abs_heads], dim=1)   # [B,2]
        fluo_out = torch.cat([hd(z_aux) for hd in self.fluo_heads], dim=1) # [B,4]

        if return_attn:
            return logits, abs_out, fluo_out, attn
        return logits, abs_out, fluo_out

    def training_step(self, batch, batch_idx):
        x2d, x3d, kpm, y_cls, w_cls, y_abs, m_abs, w_abs, y_fluo, m_fluo, w_fluo = batch
        logits, abs_out, fluo_out = self(x2d, x3d, kpm, return_attn=False)

        with autocast(enabled=False):
            per_task = self.cls_loss(logits.float(), y_cls.float(), w_cls.float())
            weighted_per_task = per_task * self.lam.float()
            loss_cls = weighted_per_task.mean()

            loss_abs = reg_loss_weighted(abs_out.float(), y_abs.float(), m_abs, w_abs.float(), self.reg_loss_type)
            loss_fluo = reg_loss_weighted(fluo_out.float(), y_fluo.float(), m_fluo, w_fluo.float(), self.reg_loss_type)

            loss = loss_cls + self.lambda_aux_abs * loss_abs + self.lambda_aux_fluo * loss_fluo

        bs = int(y_cls.shape[0])
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=bs)
        # Optional detailed training logs (aggregated over epoch)
        self.log("train_cls_loss", loss_cls, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_per_task_loss_mean", per_task.mean(), on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_weighted_per_task_loss_mean", weighted_per_task.mean(), on_step=False, on_epoch=True, batch_size=bs)
        return loss

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
        P = np.concatenate(self._val_p, axis=0)
        Y = np.concatenate(self._val_y, axis=0).astype(int)
        W = np.concatenate(self._val_w, axis=0).astype(np.float32)

        aps = ap_per_task(Y, P, w_cls=W, weighted_tasks=(0, 1))
        mac = float(np.mean(aps))
        mn = float(np.min(aps))

        for t in range(4):
            self.log(f"val_ap_{t}", float(aps[t]), prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_macro_ap", float(mac), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_min_ap", float(mn), prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
