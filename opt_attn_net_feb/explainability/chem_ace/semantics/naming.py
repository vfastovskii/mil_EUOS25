from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class NamingRule:
    """One naming rule over semantic tags."""

    label: str
    required_tags: tuple[str, ...]
    forbidden_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class NamingRegistry:
    """Rule registry for auto-generated concept names."""

    rules: tuple[NamingRule, ...]

    @staticmethod
    def default() -> "NamingRegistry":
        return NamingRegistry(
            rules=(
                NamingRule(label="carboxylate-like anion", required_tags=("anionic", "HBA")),
                NamingRule(label="cationic amine", required_tags=("cationic", "HBD")),
                NamingRule(label="planar aromatic pi-system", required_tags=("planar", "aromatic pi-system")),
                NamingRule(label="HBD/HBA pair", required_tags=("HBD/HBA pair",)),
            )
        )

    @staticmethod
    def from_json(path: str | Path) -> "NamingRegistry":
        payload = json.loads(Path(path).read_text())
        rules = []
        for row in payload.get("rules", []):
            rules.append(
                NamingRule(
                    label=str(row["label"]),
                    required_tags=tuple(str(x) for x in row.get("required_tags", [])),
                    forbidden_tags=tuple(str(x) for x in row.get("forbidden_tags", [])),
                )
            )
        return NamingRegistry(rules=tuple(rules))


def choose_label(*, tags: Sequence[str], registry: NamingRegistry, descriptor_rank: Sequence[str]) -> str:
    """Choose human-readable concept label using rules then fallback descriptors."""
    tag_set = set(str(t) for t in tags)
    for rule in registry.rules:
        if all(t in tag_set for t in rule.required_tags) and all(t not in tag_set for t in rule.forbidden_tags):
            return str(rule.label)

    ranked = [d for d in descriptor_rank if d]
    if not ranked:
        return "unlabeled molecular motif"
    prefix = ", ".join(ranked[:3])
    return f"{prefix} motif"


__all__ = ["NamingRegistry", "NamingRule", "choose_label"]
