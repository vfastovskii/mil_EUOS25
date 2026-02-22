from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..types import PatchEmbeddingRecord, PatchRecord


class EmbeddingCache:
    """Deterministic disk cache for patch embeddings."""

    def __init__(self, cache_dir: str, *, fmt: str = "npy", overwrite: bool = False):
        self.cache_dir = Path(cache_dir)
        self.format = str(fmt).lower()
        self.overwrite = bool(overwrite)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_uri(self, *, patch_id: str, layer_name: str, strategy: str) -> Path:
        key = sha1(f"{patch_id}|{layer_name}|{strategy}".encode("utf-8")).hexdigest()
        sub = self.cache_dir / str(layer_name).replace(".", "_")
        sub.mkdir(parents=True, exist_ok=True)
        ext = "npy" if self.format not in {"safetensors"} else "npy"
        return sub / f"{key}.{ext}"

    def load(
        self,
        *,
        patch: PatchRecord,
        layer_name: str,
        strategy: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[PatchEmbeddingRecord]:
        uri = self._build_uri(patch_id=patch.patch_id, layer_name=layer_name, strategy=strategy)
        if not uri.exists():
            return None
        vector = np.load(uri)
        return PatchEmbeddingRecord(
            patch_id=patch.patch_id,
            layer_name=str(layer_name),
            strategy=str(strategy),
            vector=np.asarray(vector, dtype=np.float32),
            embedding_uri=str(uri),
            metadata=metadata or {},
        )

    def save(
        self,
        *,
        patch: PatchRecord,
        layer_name: str,
        strategy: str,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PatchEmbeddingRecord:
        uri = self._build_uri(patch_id=patch.patch_id, layer_name=layer_name, strategy=strategy)
        if self.overwrite or not uri.exists():
            np.save(uri, np.asarray(vector, dtype=np.float32))
        md = metadata or {}
        meta_uri = uri.with_suffix(".json")
        if self.overwrite or not meta_uri.exists():
            meta_uri.write_text(json.dumps(md, indent=2, sort_keys=True))
        return PatchEmbeddingRecord(
            patch_id=patch.patch_id,
            layer_name=str(layer_name),
            strategy=str(strategy),
            vector=np.asarray(vector, dtype=np.float32),
            embedding_uri=str(uri),
            metadata=md,
        )


__all__ = ["EmbeddingCache"]
