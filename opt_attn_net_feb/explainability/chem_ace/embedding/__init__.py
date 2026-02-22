"""Embedding extraction and caching for Chem-ACE patches."""

from .base import EmbeddingContext, PatchEmbedder
from .cache import EmbeddingCache
from .hooks import LayerActivationHook, resolve_module
from .strategies import MaskedInputPatchEmbedder, NodePoolingPatchEmbedder, embed_patches

__all__ = [
    "EmbeddingCache",
    "EmbeddingContext",
    "LayerActivationHook",
    "MaskedInputPatchEmbedder",
    "NodePoolingPatchEmbedder",
    "PatchEmbedder",
    "embed_patches",
    "resolve_module",
]
