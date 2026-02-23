from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..cav.tcav import run_tcav_from_arrays
from ..config import ChemACEConfig
from ..db.repository import ChemACERepository
from ..embedding import EmbeddingCache, EmbeddingContext, MaskedInputPatchEmbedder, NodePoolingPatchEmbedder, embed_patches
from ..semantics.taggers import SemanticTagger, SemanticTaggingResult
from ..types import PatchEmbeddingRecord, PatchInputBuilder, PatchRecord
from .clustering import DiscoveredConceptSet, discover_concepts
from ..patches import (
    BRICSPatchGenerator,
    CompositePatchGenerator,
    LocalSubgraphPatchGenerator,
    MurckoPatchGenerator,
    Pharm3DPatchGenerator,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoleculeSource:
    """Input molecule descriptor for Chem-ACE batch processing."""

    mol_id: str
    mol: Any
    conf_ids: tuple[str, ...] = ()


class ChemACEPipeline:
    """End-to-end Chem-ACE orchestration service."""

    def __init__(self, *, config: ChemACEConfig, repository: Optional[ChemACERepository] = None):
        self.config = config
        outdir = Path(config.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        self.repository = repository or ChemACERepository(
            db_uri=str(config.database.uri),
            artifact_dir=str(outdir / "artifacts"),
        )

        cache_cfg = config.embedding.cache
        self.embedding_cache = EmbeddingCache(
            cache_dir=str(outdir / cache_cfg.cache_dir),
            fmt=cache_cfg.format,
            overwrite=cache_cfg.overwrite,
        )

        generators = [
            LocalSubgraphPatchGenerator(config.patch_generation.local_subgraph),
            BRICSPatchGenerator(config.patch_generation.brics),
            MurckoPatchGenerator(config.patch_generation.murcko),
            Pharm3DPatchGenerator(config.patch_generation.pharm3d),
        ]
        self.patch_generator = CompositePatchGenerator(generators)
        self.semantic_tagger = SemanticTagger(config.semantics)

    def start_run(self, *, task_ids: Sequence[str]) -> str:
        """Create run in DB and register tasks."""
        run_id = self.repository.create_run(run_name=self.config.run_name, config=self.config.to_dict())
        self.repository.ensure_tasks(task_ids)
        return run_id

    def _cpu_workers(self) -> int:
        try:
            return max(0, int(self.config.cpu_workers))
        except Exception:
            return 0

    def generate_patches(self, *, molecules: Sequence[MoleculeSource]) -> list[PatchRecord]:
        """Generate patches for a set of molecules and persist them."""
        workers = self._cpu_workers()

        def _gen_for_molecule(item: MoleculeSource) -> list[PatchRecord]:
            out: list[PatchRecord] = []
            if item.conf_ids:
                for conf_id in item.conf_ids:
                    out.extend(
                        self.patch_generator.generate(
                            mol_id=item.mol_id,
                            mol=item.mol,
                            conf_id=conf_id,
                        )
                    )
            else:
                out.extend(
                    self.patch_generator.generate(
                        mol_id=item.mol_id,
                        mol=item.mol,
                        conf_id=None,
                    )
                )
            return out

        all_patches: list[PatchRecord] = []
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for patch_list in ex.map(_gen_for_molecule, molecules):
                    all_patches.extend(patch_list)
        else:
            for item in molecules:
                all_patches.extend(_gen_for_molecule(item))
        self.repository.upsert_patches(all_patches)
        logger.info("Generated patches", extra={"n_patches": len(all_patches), "cpu_workers": workers})
        return all_patches

    def embed_patches(
        self,
        *,
        patches: Sequence[PatchRecord],
        model: Any,
        input_builder: PatchInputBuilder,
        device: str = "cpu",
    ) -> list[PatchEmbeddingRecord]:
        """Embed patches at configured layer and persist metadata."""
        context = EmbeddingContext(
            model=model,
            layer_name=str(self.config.embedding.layer_name),
            input_builder=input_builder,
            device=str(device),
        )
        strategy = str(self.config.embedding.strategy).lower()
        if strategy == "node_pooling":
            embedder = NodePoolingPatchEmbedder()
        else:
            embedder = MaskedInputPatchEmbedder()

        recs = embed_patches(
            patches=patches,
            embedder=embedder,
            context=context,
            cache=self.embedding_cache,
        )
        for rec in recs:
            self.repository.upsert_patch_embedding(rec)
        logger.info("Embedded patches", extra={"n_embeddings": len(recs), "strategy": strategy})
        return recs

    def discover_and_store_concepts(
        self,
        *,
        run_id: str,
        embeddings: Sequence[PatchEmbeddingRecord],
    ) -> tuple[DiscoveredConceptSet, str]:
        """Run concept discovery, persist snapshot, concepts, and memberships."""
        concept_set = discover_concepts(
            embeddings=embeddings,
            config=self.config.discovery,
            seed=int(self.config.seed),
        )
        concept_set_id = self.repository.create_concept_set_snapshot(
            run_id=run_id,
            layer_name=concept_set.layer_name,
            config={
                "discovery": self.config.discovery.__dict__,
                "embedding": self.config.embedding.__dict__,
            },
            metadata=concept_set.metadata,
        )
        self.repository.upsert_concepts(concept_set_id=concept_set_id, candidates=concept_set.candidates)
        self.repository.upsert_memberships(concept_set.memberships)
        self.repository.set_run_concept_set(run_id=run_id, concept_set_id=concept_set_id)
        logger.info(
            "Discovered concepts",
            extra={
                "n_concepts": len(concept_set.candidates),
                "n_memberships": len(concept_set.memberships),
                "concept_set_id": concept_set_id,
            },
        )
        return concept_set, concept_set_id

    def tag_and_store_concepts(
        self,
        *,
        concept_set: DiscoveredConceptSet,
        patches: Sequence[PatchRecord],
        molecules_by_id: Mapping[str, Any],
    ) -> list[SemanticTaggingResult]:
        """Compute semantic tags for each concept and persist them."""
        workers = self._cpu_workers()
        patch_by_id = {p.patch_id: p for p in patches}
        members_by_concept: dict[str, list[PatchRecord]] = {}
        for m in concept_set.memberships:
            patch = patch_by_id.get(m.patch_id)
            if patch is None:
                continue
            members_by_concept.setdefault(m.concept_local_id, []).append(patch)

        candidates_with_patches = [
            cand
            for cand in concept_set.candidates
            if members_by_concept.get(cand.concept_local_id, [])
        ]

        def _tag_one(cand) -> SemanticTaggingResult:
            cpatches = members_by_concept[cand.concept_local_id]
            return self.semantic_tagger.tag_concept(
                concept_id=cand.concept_local_id,
                concept_patches=cpatches,
                molecules_by_id=molecules_by_id,
            )

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_tag_one, candidates_with_patches))
        else:
            results = [_tag_one(cand) for cand in candidates_with_patches]

        out: list[SemanticTaggingResult] = []
        for result in results:
            self.repository.set_concept_label(concept_id=result.concept_id, label_auto=result.label_auto)
            self.repository.upsert_tags(result.tags)
            out.append(result)
        logger.info("Tagged concepts", extra={"n_tagged": len(out), "cpu_workers": workers})
        return out

    def run_tcav_and_store(
        self,
        *,
        run_id: str,
        concept_set: DiscoveredConceptSet,
        embeddings: Sequence[PatchEmbeddingRecord],
        gradients_by_task_epoch: Mapping[tuple[str, int], np.ndarray],
    ) -> dict[tuple[str, int, str], dict[str, float]]:
        """Run repeated CAV/TCAV for each concept/task/epoch using provided gradients."""
        emb_by_patch = {e.patch_id: np.asarray(e.vector, dtype=np.float32) for e in embeddings}
        members_by_concept: dict[str, list[str]] = {}
        for m in concept_set.memberships:
            members_by_concept.setdefault(m.concept_local_id, []).append(m.patch_id)

        all_patch_vectors = np.stack([np.asarray(e.vector, dtype=np.float32) for e in embeddings], axis=0)

        summary_out: dict[tuple[str, int, str], dict[str, float]] = {}
        for cand in concept_set.candidates:
            concept_patch_ids = members_by_concept.get(cand.concept_local_id, [])
            concept_vectors = [emb_by_patch[pid] for pid in concept_patch_ids if pid in emb_by_patch]
            if len(concept_vectors) < 2:
                continue
            x_pos = np.stack(concept_vectors, axis=0)

            mask = np.ones((all_patch_vectors.shape[0],), dtype=bool)
            # remove current concept members from random pool when possible
            member_set = set(concept_patch_ids)
            all_patch_ids = [e.patch_id for e in embeddings]
            for i, pid in enumerate(all_patch_ids):
                if pid in member_set:
                    mask[i] = False
            x_pool = all_patch_vectors[mask] if mask.any() else all_patch_vectors

            for (task_id, epoch), grads in gradients_by_task_epoch.items():
                cav_records, tcav_records, summary = run_tcav_from_arrays(
                    run_id=run_id,
                    epoch=int(epoch),
                    concept_id=cand.concept_local_id,
                    task_id=str(task_id),
                    layer_name=concept_set.layer_name,
                    concept_embeddings=x_pos,
                    random_pool_embeddings=x_pool,
                    target_gradients=np.asarray(grads, dtype=np.float32),
                    config=self.config.cav,
                    seed=int(self.config.seed),
                )
                for rec in cav_records:
                    self.repository.upsert_cav(rec)
                for rec in tcav_records:
                    self.repository.upsert_tcav_epoch(rec)

                summary_out[(str(task_id), int(epoch), cand.concept_local_id)] = {
                    "mean_sign_rate": float(summary.mean_sign_rate),
                    "std_sign_rate": float(summary.std_sign_rate),
                    "mean_directional_derivative": float(summary.mean_directional_derivative),
                    "std_directional_derivative": float(summary.std_directional_derivative),
                }

        logger.info("Computed TCAV", extra={"n_summaries": len(summary_out)})
        return summary_out


__all__ = ["ChemACEPipeline", "MoleculeSource"]
