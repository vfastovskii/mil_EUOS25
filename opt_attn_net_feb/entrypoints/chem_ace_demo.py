from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..explainability.chem_ace import ChemACEConfig
from ..explainability.chem_ace.config import DatabaseConfig, EmbeddingConfig
from ..explainability.chem_ace.analytics import ConceptQueryService
from ..explainability.chem_ace.cav import collect_gradients_for_layer
from ..explainability.chem_ace.concepts.pipeline import ChemACEPipeline, MoleculeSource
from ..explainability.chem_ace.db.repository import MILConceptEpochMetric
from ..explainability.chem_ace.optional_deps import OptionalDependencyError, require_rdkit

LOGGER = logging.getLogger("chem_ace_demo")


@dataclass(frozen=True)
class ToyMolInputBuilder:
    """Patch input builder for toy node-feature model."""

    features_by_mol: Mapping[str, np.ndarray]

    def build_masked_input(self, patch):
        x = np.asarray(self.features_by_mol[patch.mol_id], dtype=np.float32).copy()
        keep = set(int(i) for i in patch.atom_indices)
        for idx in range(x.shape[0]):
            if idx not in keep:
                x[idx, :] = 0.0
        return {"x": torch.from_numpy(x).unsqueeze(0)}

    def build_full_input(self, patch):
        x = np.asarray(self.features_by_mol[patch.mol_id], dtype=np.float32)
        return {"x": torch.from_numpy(x).unsqueeze(0)}, list(int(i) for i in patch.atom_indices)


class ToyTaskAdapter:
    """Task adapter for TCAV gradient collection from toy model."""

    def __init__(self, model: nn.Module, task_to_idx: Mapping[str, int]):
        self.model = model
        self.task_to_idx = dict(task_to_idx)

    def forward(self, model_input: Any) -> Any:
        if isinstance(model_input, dict):
            return self.model(**model_input)
        raise TypeError("ToyTaskAdapter expects dict model_input")

    def get_task_scalar(self, model_output: Any, task_id: str) -> Any:
        idx = int(self.task_to_idx[task_id])
        return model_output[:, idx].sum()


class ToyNodeModel(nn.Module):
    """Simple node-feature model exposing `encoder` layer for activation hooks."""

    def __init__(self, in_dim: int, hidden_dim: int, n_tasks: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.head = nn.Linear(int(hidden_dim), int(n_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def _toy_smiles() -> list[tuple[str, str]]:
    return [
        ("mol_1", "CC(=O)O"),
        ("mol_2", "c1ccccc1N"),
        ("mol_3", "CCN(CC)CC"),
        ("mol_4", "O=C([O-])c1ccccc1"),
        ("mol_5", "c1ncccc1"),
        ("mol_6", "CCOC(=O)N"),
        ("mol_7", "CCS(=O)(=O)N"),
        ("mol_8", "CC(C)O"),
    ]


def _load_molecules(smiles_csv: str | None, max_mols: int, with_3d: bool) -> tuple[list[MoleculeSource], dict[str, Any]]:
    require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if smiles_csv:
        df = pd.read_csv(smiles_csv)
        if not {"mol_id", "smiles"}.issubset(set(df.columns)):
            raise ValueError("CSV must contain columns: mol_id, smiles")
        pairs = [(str(r.mol_id), str(r.smiles)) for r in df.itertuples(index=False)]
    else:
        pairs = _toy_smiles()

    mol_sources: list[MoleculeSource] = []
    mol_map: dict[str, Any] = {}
    for mol_id, smi in pairs[: max(1, int(max_mols))]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        conf_ids: tuple[str, ...] = ()
        if with_3d:
            _ = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
            _ = AllChem.UFFOptimizeMolecule(mol)
            conf_ids = ("0",)
        mol_sources.append(MoleculeSource(mol_id=str(mol_id), mol=mol, conf_ids=conf_ids))
        mol_map[str(mol_id)] = mol
    return mol_sources, mol_map


def _featurize_molecules(mol_map: Mapping[str, Any]) -> dict[str, np.ndarray]:
    require_rdkit()
    from rdkit.Chem import rdchem

    atom_vocab = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    idx = {z: i for i, z in enumerate(atom_vocab)}

    out: dict[str, np.ndarray] = {}
    for mol_id, mol in mol_map.items():
        feats: list[list[float]] = []
        for atom in mol.GetAtoms():
            onehot = [0.0] * len(atom_vocab)
            z = int(atom.GetAtomicNum())
            if z in idx:
                onehot[idx[z]] = 1.0
            feats.append(
                onehot
                + [
                    float(atom.GetDegree()) / 4.0,
                    float(atom.GetFormalCharge()) / 3.0,
                    1.0 if atom.GetIsAromatic() else 0.0,
                    float(atom.GetMass()) / 200.0,
                ]
            )
        out[mol_id] = np.asarray(feats, dtype=np.float32)
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Chem-ACE demo pipeline")
    ap.add_argument("--smiles_csv", default=None, help="Optional CSV with columns: mol_id, smiles")
    ap.add_argument("--output_dir", default="chem_ace_demo_out")
    ap.add_argument("--db_uri", default=None, help="Optional SQLAlchemy URI, default uses output_dir/chem_ace.sqlite3")
    ap.add_argument("--max_mols", type=int, default=8)
    ap.add_argument("--with_3d", action="store_true")
    ap.add_argument("--layer_name", default="encoder")
    ap.add_argument("--embedding_strategy", default="masked_input", choices=["masked_input", "node_pooling"])
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        mol_sources, mol_map = _load_molecules(args.smiles_csv, args.max_mols, args.with_3d)
    except OptionalDependencyError as exc:
        raise SystemExit(f"RDKit is required for chem_ace_demo: {exc}")

    if not mol_sources:
        raise SystemExit("No valid molecules available for demo")

    features_by_mol = _featurize_molecules(mol_map)
    in_dim = int(next(iter(features_by_mol.values())).shape[1])

    task_ids = ["task0", "task1", "task2", "task3"]
    task_to_idx = {t: i for i, t in enumerate(task_ids)}

    db_uri = args.db_uri or f"sqlite:///{(outdir / 'chem_ace.sqlite3').as_posix()}"
    config = ChemACEConfig(
        run_name="chem_ace_demo",
        seed=int(args.seed),
        output_dir=str(outdir),
        embedding=EmbeddingConfig(
            layer_name=str(args.layer_name),
            strategy=str(args.embedding_strategy),
        ),
        database=DatabaseConfig(uri=db_uri),
    )

    pipeline = ChemACEPipeline(config=config)
    run_id = pipeline.start_run(task_ids=task_ids)

    model = ToyNodeModel(in_dim=in_dim, hidden_dim=32, n_tasks=len(task_ids))
    input_builder = ToyMolInputBuilder(features_by_mol=features_by_mol)
    task_adapter = ToyTaskAdapter(model=model, task_to_idx=task_to_idx)

    patches = pipeline.generate_patches(molecules=mol_sources)
    if not patches:
        raise SystemExit("Patch generation produced zero patches")

    embeddings = pipeline.embed_patches(
        patches=patches,
        model=model,
        input_builder=input_builder,
        device="cpu",
    )

    concept_set, concept_set_id = pipeline.discover_and_store_concepts(
        run_id=run_id,
        embeddings=embeddings,
    )

    tagging = pipeline.tag_and_store_concepts(
        concept_set=concept_set,
        patches=patches,
        molecules_by_id=mol_map,
    )

    # Build gradient matrices for each task at demo epoch=0.
    model_inputs = [{"x": torch.from_numpy(features_by_mol[item.mol_id]).unsqueeze(0)} for item in mol_sources]
    gradients_by_task_epoch: dict[tuple[str, int], np.ndarray] = {}
    for task_id in task_ids:
        grads = collect_gradients_for_layer(
            model=model,
            layer_name=str(args.layer_name),
            task_id=task_id,
            adapter=task_adapter,
            model_inputs=model_inputs,
            device="cpu",
        )
        gradients_by_task_epoch[(task_id, 0)] = grads

    tcav_summary = pipeline.run_tcav_and_store(
        run_id=run_id,
        concept_set=concept_set,
        embeddings=embeddings,
        gradients_by_task_epoch=gradients_by_task_epoch,
    )

    # Populate optional MIL concept epoch metrics for query demos.
    n_patches = max(1, len(patches))
    support_by_concept = {cand.concept_local_id: cand.support for cand in concept_set.candidates}
    for task_id in task_ids:
        for cand in concept_set.candidates:
            prevalence = float(support_by_concept[cand.concept_local_id] / n_patches)
            pipeline.repository.upsert_mil_concept_epoch(
                MILConceptEpochMetric(
                    run_id=run_id,
                    epoch=0,
                    concept_id=cand.concept_local_id,
                    task_id=task_id,
                    attention_support=float(prevalence),
                    witness_rate=float(min(1.0, prevalence * 1.5)),
                    attention_entropy=float(max(0.0, 1.0 - prevalence)),
                    prevalence=float(prevalence),
                    metadata={"source": "demo_synthetic"},
                )
            )

    query = ConceptQueryService(session_factory=pipeline.repository.SessionFactory)
    query_outputs = {
        "planar_conjugated_rising_tcav": query.planar_conjugated_rising_tcav(task_id="task0", last_n_epochs=3),
        "top_attention_support": query.top_attention_support(task_id="task0", epoch=0, limit=10),
        "high_tcav_low_prevalence": query.high_tcav_low_prevalence(
            task_id="task0",
            epoch=0,
            tcav_thr=0.5,
            prevalence_thr=0.2,
        ),
        "concept_collapse_indicator": query.concept_collapse_indicator(task_id="task0", epoch=0, top_k=5),
    }

    summary = {
        "run_id": run_id,
        "concept_set_id": concept_set_id,
        "n_molecules": len(mol_sources),
        "n_patches": len(patches),
        "n_embeddings": len(embeddings),
        "n_concepts": len(concept_set.candidates),
        "n_tags": int(sum(len(x.tags) for x in tagging)),
        "n_tcav_summaries": len(tcav_summary),
        "query_outputs": query_outputs,
    }
    summary_path = outdir / "chem_ace_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    LOGGER.info("Chem-ACE demo finished")
    LOGGER.info("Summary path: %s", summary_path)


if __name__ == "__main__":
    main()
