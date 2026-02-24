from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping


def _slug(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def default_functional_rules_path() -> Path:
    return Path(__file__).resolve().parent / "default_functional_group_rules.json"


def load_fragment_functions() -> Mapping[str, Any]:
    try:
        from rdkit.Chem import Fragments
    except Exception as exc:
        raise RuntimeError(
            "RDKit is required for fragment-rule generation. Install rdkit to use this functionality."
        ) from exc

    out: Dict[str, Any] = {}
    for name, fn in inspect.getmembers(Fragments):
        if not str(name).startswith("fr_"):
            continue
        if callable(fn):
            out[str(name)] = fn
    return out


def scan_smiles_with_fragments(
    *,
    smiles_iter: Iterable[str],
    fragment_functions: Mapping[str, Any],
) -> tuple[int, int, Dict[str, int], Dict[str, float]]:
    try:
        from rdkit import Chem
    except Exception as exc:
        raise RuntimeError(
            "RDKit is required for fragment-rule generation. Install rdkit to use this functionality."
        ) from exc

    hits = {name: 0 for name in fragment_functions.keys()}
    counts = {name: 0.0 for name in fragment_functions.keys()}
    n_seen = 0
    n_valid = 0

    for smi in smiles_iter:
        n_seen += 1
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        n_valid += 1
        for name, fn in fragment_functions.items():
            try:
                value = float(fn(mol))
            except Exception:
                value = 0.0
            if value > 0.0:
                hits[name] += 1
                counts[name] += float(value)
    return n_seen, n_valid, hits, counts


def build_fragment_rules(
    *,
    n_valid: int,
    hits: Mapping[str, int],
    counts: Mapping[str, float],
    min_count: int,
    min_prevalence: float,
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if n_valid <= 0:
        return rules

    for name in sorted(hits.keys()):
        hit = int(hits[name])
        prevalence = float(hit) / float(n_valid)
        if hit < int(min_count):
            continue
        if prevalence < float(min_prevalence):
            continue

        tag = f"fg_rdkit_{_slug(name[3:])}"
        confidence = min(0.95, 0.65 + 0.45 * min(1.0, prevalence / 0.2))
        min_patch_rate = max(0.01, min(0.20, 0.50 * prevalence))

        rules.append(
            {
                "tag": str(tag),
                "rdkit_fragment": str(name),
                "min_patch_rate": float(round(min_patch_rate, 6)),
                "confidence": float(round(confidence, 6)),
                "provenance": "rdkit_fragment_tagger",
                "source_prevalence": float(round(prevalence, 8)),
                "source_hit_molecules": int(hit),
                "source_total_count": float(round(float(counts.get(name, 0.0)), 4)),
            }
        )
    return rules


def merge_rules(
    *,
    base_rules: list[dict[str, Any]],
    generated_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = list(base_rules)
    idx_by_tag = {
        str(row.get("tag", "")): i
        for i, row in enumerate(out)
        if str(row.get("tag", "")).strip()
    }
    for row in generated_rules:
        tag = str(row.get("tag", "")).strip()
        if not tag:
            continue
        if tag in idx_by_tag:
            out[idx_by_tag[tag]] = row
        else:
            idx_by_tag[tag] = len(out)
            out.append(row)
    return out


@dataclass(frozen=True)
class FragmentRuleGenerationSummary:
    n_smiles_seen: int
    n_smiles_valid: int
    n_fragment_functions: int
    n_generated_rules: int
    min_count: int
    min_prevalence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_fragment_rules_from_smiles(
    *,
    smiles_iter: Iterable[str],
    min_count: int = 25,
    min_prevalence: float = 0.001,
) -> tuple[list[dict[str, Any]], FragmentRuleGenerationSummary, list[dict[str, Any]]]:
    fragment_functions = load_fragment_functions()
    n_seen, n_valid, hits, counts = scan_smiles_with_fragments(
        smiles_iter=smiles_iter,
        fragment_functions=fragment_functions,
    )
    generated_rules = build_fragment_rules(
        n_valid=n_valid,
        hits=hits,
        counts=counts,
        min_count=int(min_count),
        min_prevalence=float(min_prevalence),
    )
    summary = FragmentRuleGenerationSummary(
        n_smiles_seen=int(n_seen),
        n_smiles_valid=int(n_valid),
        n_fragment_functions=int(len(fragment_functions)),
        n_generated_rules=int(len(generated_rules)),
        min_count=int(min_count),
        min_prevalence=float(min_prevalence),
    )
    fragment_stats = sorted(
        (
            {
                "fragment": str(name),
                "hit_molecules": int(hits.get(name, 0)),
                "prevalence": float(hits.get(name, 0)) / float(max(1, n_valid)),
                "total_count": float(counts.get(name, 0.0)),
            }
            for name in fragment_functions.keys()
        ),
        key=lambda x: x["hit_molecules"],
        reverse=True,
    )
    return generated_rules, summary, fragment_stats
