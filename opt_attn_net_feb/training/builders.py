from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.multimodal_mil import MILTaskAttnMixerWithAux


@dataclass(frozen=True)
class LoaderConfig:
    num_workers: int
    pin_memory: bool


def _resolve_head_stochastic_depth(params: Dict[str, Any]) -> float:
    if "head_stochastic_depth" in params:
        return float(params["head_stochastic_depth"])
    return 0.0 if int(params.get("head_num_layers", 2)) == 1 else 0.1


class MILModelBuilder:
    """Centralized model construction from flat HPO parameters."""

    @staticmethod
    def build(
        *,
        params: Dict[str, Any],
        mol_dim: int,
        inst_dim: int,
        pos_weight: torch.Tensor,
        gamma: torch.Tensor,
        lam: np.ndarray,
    ) -> MILTaskAttnMixerWithAux:
        return MILTaskAttnMixerWithAux(
            mol_dim=int(mol_dim),
            inst_dim=int(inst_dim),
            mol_hidden=int(params["mol_hidden"]),
            mol_layers=int(params["mol_layers"]),
            mol_dropout=float(params["mol_dropout"]),
            inst_hidden=int(params["inst_hidden"]),
            inst_layers=int(params["inst_layers"]),
            inst_dropout=float(params["inst_dropout"]),
            proj_dim=int(params["proj_dim"]),
            attn_heads=int(params["attn_heads"]),
            attn_dropout=float(params["attn_dropout"]),
            mixer_hidden=int(params["mixer_hidden"]),
            mixer_layers=int(params["mixer_layers"]),
            mixer_dropout=float(params["mixer_dropout"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            pos_weight=pos_weight,
            gamma=gamma,
            lam=lam,
            lambda_aux_abs=float(params["lambda_aux_abs"]),
            lambda_aux_fluo=float(params["lambda_aux_fluo"]),
            reg_loss_type=str(params["reg_loss_type"]),
            activation=str(params.get("activation", "GELU")),
            mol_embedder_name=str(params.get("mol_embedder_name", "mlp_v3_2d")),
            inst_embedder_name=str(params.get("inst_embedder_name", "mlp_v3_3d")),
            aggregator_name=str(params.get("aggregator_name", "task_attention_pool")),
            predictor_name=str(params.get("predictor_name", "mlp_v3")),
            head_num_layers=int(params.get("head_num_layers", 2)),
            head_dropout=float(params.get("head_dropout", 0.1)),
            head_stochastic_depth=_resolve_head_stochastic_depth(params),
            head_fc2_gain_non_last=float(params.get("head_fc2_gain_non_last", 1e-2)),
        )


class DataLoaderBuilder:
    """Creates dataloaders with consistent worker/prefetch settings."""

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

    def train_loader(self, dataset, *, batch_size: int, sampler, collate_fn):
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            sampler=sampler,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
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
