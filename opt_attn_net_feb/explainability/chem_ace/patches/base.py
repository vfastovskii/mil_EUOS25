from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha1
import json
import logging
from typing import Any, Iterable, Optional, Sequence

from ..types import PatchRecord

logger = logging.getLogger(__name__)


def stable_json(value: Any) -> str:
    """Serialize value deterministically for hashing and storage."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def patch_hash(
    *,
    patch_type: str,
    atom_indices: Sequence[int],
    smarts: Optional[str],
    fragment_repr: Optional[str],
    feature_metadata: dict[str, Any],
) -> str:
    """Compute deterministic hash for a molecular patch signature."""
    payload = {
        "patch_type": str(patch_type),
        "atom_indices": [int(x) for x in sorted(set(atom_indices))],
        "smarts": smarts,
        "fragment_repr": fragment_repr,
        "feature_metadata": feature_metadata,
    }
    return sha1(stable_json(payload).encode("utf-8")).hexdigest()


def patch_id(mol_id: str, conf_id: Optional[str], p_hash: str) -> str:
    """Compute deterministic patch identifier for molecule/conformer/hash tuple."""
    conf = "null" if conf_id is None else str(conf_id)
    return sha1(f"{mol_id}|{conf}|{p_hash}".encode("utf-8")).hexdigest()


class PatchGenerator(ABC):
    """Abstract base class for molecular patch generators."""

    patch_type: str
    requires_conformer: bool = False

    @abstractmethod
    def generate(self, *, mol_id: str, mol: Any, conf_id: Optional[str] = None) -> list[PatchRecord]:
        """Generate patches for a molecule."""


class CompositePatchGenerator(PatchGenerator):
    """Run multiple generators and merge unique patch records."""

    patch_type = "composite"

    def __init__(self, generators: Iterable[PatchGenerator]):
        self.generators = list(generators)

    def generate(self, *, mol_id: str, mol: Any, conf_id: Optional[str] = None) -> list[PatchRecord]:
        merged: dict[str, PatchRecord] = {}
        for generator in self.generators:
            try:
                generated = generator.generate(mol_id=mol_id, mol=mol, conf_id=conf_id)
            except Exception:
                logger.exception(
                    "Patch generator failed",
                    extra={"generator": type(generator).__name__, "mol_id": mol_id},
                )
                continue
            for patch in generated:
                merged[patch.patch_id] = patch
        return list(merged.values())


def make_patch_record(
    *,
    mol_id: str,
    conf_id: Optional[str],
    patch_type: str,
    atom_indices: Sequence[int],
    smarts: Optional[str] = None,
    fragment_repr: Optional[str] = None,
    feature_metadata: Optional[dict[str, Any]] = None,
) -> PatchRecord:
    """Construct a validated PatchRecord with deterministic identifiers."""
    md = feature_metadata or {}
    atoms = tuple(sorted(set(int(x) for x in atom_indices)))
    if len(atoms) == 0:
        raise ValueError("atom_indices cannot be empty")
    p_hash = patch_hash(
        patch_type=patch_type,
        atom_indices=atoms,
        smarts=smarts,
        fragment_repr=fragment_repr,
        feature_metadata=md,
    )
    return PatchRecord(
        patch_id=patch_id(mol_id=mol_id, conf_id=conf_id, p_hash=p_hash),
        mol_id=str(mol_id),
        conf_id=(None if conf_id is None else str(conf_id)),
        patch_type=str(patch_type),
        atom_indices=atoms,
        patch_hash=p_hash,
        smarts=smarts,
        fragment_repr=fragment_repr,
        feature_metadata=md,
    )


__all__ = [
    "CompositePatchGenerator",
    "PatchGenerator",
    "make_patch_record",
    "patch_hash",
    "patch_id",
    "stable_json",
]
