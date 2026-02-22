from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any, Optional

import numpy as np
import torch

from ..types import PatchEmbeddingRecord, PatchInputBuilder, PatchRecord
from .cache import EmbeddingCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingContext:
    """Execution context for patch embedding extraction."""

    model: torch.nn.Module
    layer_name: str
    input_builder: PatchInputBuilder
    device: str = "cpu"


class PatchEmbedder(ABC):
    """Abstract patch embedding strategy interface."""

    strategy_name: str

    @abstractmethod
    def embed_patch(
        self,
        *,
        patch: PatchRecord,
        context: EmbeddingContext,
        cache: Optional[EmbeddingCache] = None,
    ) -> PatchEmbeddingRecord:
        """Embed one patch and optionally cache to disk."""



def forward_model(model: torch.nn.Module, model_input: Any) -> Any:
    """Run model forward for dict / tuple / scalar input variants."""
    if isinstance(model_input, dict):
        return model(**model_input)
    if isinstance(model_input, (tuple, list)):
        return model(*model_input)
    return model(model_input)



def to_1d_embedding(tensor: torch.Tensor) -> np.ndarray:
    """Convert activation tensor to one vector via robust pooling."""
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim >= 2:
        arr2 = arr.reshape(-1, arr.shape[-1])
        return arr2.mean(axis=0).astype(np.float32)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


__all__ = [
    "EmbeddingContext",
    "PatchEmbedder",
    "forward_model",
    "to_1d_embedding",
]
