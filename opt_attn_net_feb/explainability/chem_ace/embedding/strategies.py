from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from ..types import PatchEmbeddingRecord, PatchRecord
from .base import EmbeddingContext, PatchEmbedder, forward_model, to_1d_embedding
from .cache import EmbeddingCache
from .hooks import LayerActivationHook

logger = logging.getLogger(__name__)


def _move_to_device(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_move_to_device(v, device) for v in data)
    return data


def _pool_nodes(activation: torch.Tensor, atom_indices: Sequence[int]) -> np.ndarray:
    act = activation.detach().cpu().float()
    if act.ndim == 3:
        act2 = act[0]
    elif act.ndim == 2:
        act2 = act
    else:
        return to_1d_embedding(act)

    valid = [int(i) for i in atom_indices if 0 <= int(i) < int(act2.shape[0])]
    if not valid:
        pooled = act2.mean(dim=0)
    else:
        pooled = act2[valid].mean(dim=0)
    return pooled.numpy().astype(np.float32)


class MaskedInputPatchEmbedder(PatchEmbedder):
    """Embed patch by forwarding masked input and reading chosen layer activation."""

    strategy_name = "masked_input"

    def embed_patch(
        self,
        *,
        patch: PatchRecord,
        context: EmbeddingContext,
        cache: Optional[EmbeddingCache] = None,
    ) -> PatchEmbeddingRecord:
        if cache is not None:
            rec = cache.load(
                patch=patch,
                layer_name=context.layer_name,
                strategy=self.strategy_name,
                metadata={"strategy": self.strategy_name},
            )
            if rec is not None:
                return rec

        device = torch.device(context.device)
        model = context.model.to(device)
        model.eval()

        model_input = context.input_builder.build_masked_input(patch)
        model_input = _move_to_device(model_input, device=device)

        with torch.no_grad(), LayerActivationHook(model, context.layer_name) as hook:
            _ = forward_model(model, model_input)

        if hook.last_activation is None:
            raise RuntimeError(
                f"No activation captured for layer '{context.layer_name}' during masked embedding"
            )

        vec = to_1d_embedding(hook.last_activation)
        if cache is None:
            return PatchEmbeddingRecord(
                patch_id=patch.patch_id,
                layer_name=context.layer_name,
                strategy=self.strategy_name,
                vector=vec,
                embedding_uri=None,
                metadata={"strategy": self.strategy_name},
            )
        return cache.save(
            patch=patch,
            layer_name=context.layer_name,
            strategy=self.strategy_name,
            vector=vec,
            metadata={"strategy": self.strategy_name},
        )


class NodePoolingPatchEmbedder(PatchEmbedder):
    """Embed patch by pooling node-level activations on patch atom indices."""

    strategy_name = "node_pooling"

    def embed_patch(
        self,
        *,
        patch: PatchRecord,
        context: EmbeddingContext,
        cache: Optional[EmbeddingCache] = None,
    ) -> PatchEmbeddingRecord:
        if cache is not None:
            rec = cache.load(
                patch=patch,
                layer_name=context.layer_name,
                strategy=self.strategy_name,
                metadata={"strategy": self.strategy_name},
            )
            if rec is not None:
                return rec

        device = torch.device(context.device)
        model = context.model.to(device)
        model.eval()

        model_input, atom_indices = context.input_builder.build_full_input(patch)
        model_input = _move_to_device(model_input, device=device)

        with torch.no_grad(), LayerActivationHook(model, context.layer_name) as hook:
            _ = forward_model(model, model_input)

        if hook.last_activation is None:
            raise RuntimeError(
                f"No activation captured for layer '{context.layer_name}' during node-pooling embedding"
            )

        vec = _pool_nodes(hook.last_activation, atom_indices=atom_indices)
        if cache is None:
            return PatchEmbeddingRecord(
                patch_id=patch.patch_id,
                layer_name=context.layer_name,
                strategy=self.strategy_name,
                vector=vec,
                embedding_uri=None,
                metadata={"strategy": self.strategy_name, "n_patch_atoms": len(atom_indices)},
            )
        return cache.save(
            patch=patch,
            layer_name=context.layer_name,
            strategy=self.strategy_name,
            vector=vec,
            metadata={"strategy": self.strategy_name, "n_patch_atoms": len(atom_indices)},
        )



def embed_patches(
    *,
    patches: Iterable[PatchRecord],
    embedder: PatchEmbedder,
    context: EmbeddingContext,
    cache: Optional[EmbeddingCache] = None,
) -> list[PatchEmbeddingRecord]:
    """Batch helper to embed a patch iterable with one strategy."""
    out: list[PatchEmbeddingRecord] = []
    for patch in patches:
        out.append(embedder.embed_patch(patch=patch, context=context, cache=cache))
    return out


__all__ = [
    "MaskedInputPatchEmbedder",
    "NodePoolingPatchEmbedder",
    "embed_patches",
]
