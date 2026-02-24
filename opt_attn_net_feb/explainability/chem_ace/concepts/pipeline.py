from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import time

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
    LocalSubgraphPatchGenerator,
    MurckoPatchGenerator,
    Pharm3DPatchGenerator,
)
from ....utils.progress import log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoleculeSource:
    """
    Represents a source of molecular data.

    This class is used to store information about a molecule, including its
    identifier, its structure, and optionally the IDs of its conformers.

    Attributes:
        mol_id: A unique identifier for the molecule.
        mol: The molecular structure. The type of this attribute allows flexibility
            for various representations.
        conf_ids: A tuple of identifiers for conformers associated with the molecule.
    """

    mol_id: str
    mol: Any
    conf_ids: tuple[str, ...] = ()


class ChemACEPipeline:
    """
    Represents a processing pipeline for ChemACE, a tool for chemical structure-based
    explainable AI. This pipeline orchestrates molecule patch generation, caching,
    semantic tagging, and repository management.

    The class initializes necessary components such as patch generators, embedding cache,
    and a semantic tagger while handling input configuration and repository management.
    It supports 2D and 3D molecular patch generation and ensures patches are constrained
    by user-defined or automatically calculated limits.

    Attributes:
        config: ChemACE configuration object containing all essential settings for the pipeline.
        repository: Optional repository for managing database interactions and artifacts.
                    If not provided, a default repository is initialized.
        embedding_cache: Manages caching for embeddings used in the analytics pipeline.
        patch_generators_2d: List of 2D patch generators deployed in the pipeline.
        patch_generators_3d: List of 3D patch generators deployed for processing molecules
                             with conformer data.
        semantic_tagger: Processes and tags molecules semantically based on the provided
                         configuration.
    """

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
        self.patch_generators_2d = [
            g for g in generators if not bool(getattr(g, "requires_conformer", False))
        ]
        self.patch_generators_3d = [
            g for g in generators if bool(getattr(g, "requires_conformer", False))
        ]
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

    def _patch_progress_every(self) -> int:
        try:
            return max(0, int(getattr(self.config, "progress_log_every_molecules", 250)))
        except Exception:
            return 250

    def _resolve_patch_cap(self, *, total_molecules: int) -> int:
        try:
            configured_cap = int(getattr(self.config, "max_patches_per_molecule", 0))
        except Exception:
            configured_cap = 0
        if configured_cap > 0:
            return int(configured_cap)

        try:
            target_total = int(getattr(self.config, "target_total_patches", 0))
        except Exception:
            target_total = 0
        if target_total <= 0 or int(total_molecules) <= 0:
            return 0

        raw_cap = int(math.ceil(float(target_total) / float(total_molecules)))
        # Keep dynamic cap in a sane range for stability.
        return int(max(16, min(256, raw_cap)))

    @staticmethod
    def _deterministic_cap_patches(
        *,
        mol_id: str,
        patches: Sequence[PatchRecord],
        cap: int,
    ) -> list[PatchRecord]:
        if cap <= 0 or len(patches) <= cap:
            return list(patches)
        ranked: list[tuple[str, PatchRecord]] = []
        for patch in patches:
            h = sha1(f"{mol_id}|{patch.patch_id}|chemace_patch_cap_v1".encode("utf-8")).hexdigest()
            ranked.append((h, patch))
        ranked.sort(key=lambda x: x[0])
        return [p for _, p in ranked[:cap]]

    def generate_patches(
        self,
        *,
        molecules: Sequence[MoleculeSource],
        progress_extras: Optional[Mapping[str, Any]] = None,
    ) -> list[PatchRecord]:
        """Generate patches for a set of molecules and persist them."""
        workers = self._cpu_workers()
        total_molecules = int(len(molecules))
        progress_every = self._patch_progress_every()
        patch_cap = self._resolve_patch_cap(total_molecules=total_molecules)
        t0 = time.perf_counter()
        extras = {str(k): v for k, v in dict(progress_extras or {}).items()}

        log_event(
            "INFO",
            "explainability.chem_ace.patch_budget",
            n_molecules=int(total_molecules),
            cap_per_molecule=int(patch_cap),
            target_total_patches=int(getattr(self.config, "target_total_patches", 0)),
            **extras,
        )

        three_d_patch_types = {
            str(getattr(g, "patch_type", "")) for g in self.patch_generators_3d
        }
        n_molecules_with_conf_total = int(sum(1 for item in molecules if bool(item.conf_ids)))

        def _gen_for_molecule(item: MoleculeSource) -> tuple[list[PatchRecord], int, int, bool, bool]:
            out: dict[str, PatchRecord] = {}
            for generator in self.patch_generators_2d:
                try:
                    generated = generator.generate(
                        mol_id=item.mol_id,
                        mol=item.mol,
                        conf_id=None,
                    )
                except Exception:
                    logger.exception(
                        "2D patch generator failed",
                        extra={"generator": type(generator).__name__, "mol_id": item.mol_id},
                    )
                    continue
                for patch in generated:
                    out[patch.patch_id] = patch

            if item.conf_ids and self.patch_generators_3d:
                for conf_id in item.conf_ids:
                    for generator in self.patch_generators_3d:
                        try:
                            generated = generator.generate(
                                mol_id=item.mol_id,
                                mol=item.mol,
                                conf_id=str(conf_id),
                            )
                        except Exception:
                            logger.exception(
                                "3D patch generator failed",
                                extra={
                                    "generator": type(generator).__name__,
                                    "mol_id": item.mol_id,
                                    "conf_id": conf_id,
                                },
                            )
                            continue
                        for patch in generated:
                            out[patch.patch_id] = patch

            patch_list = list(out.values())
            kept = self._deterministic_cap_patches(
                mol_id=str(item.mol_id),
                patches=patch_list,
                cap=int(patch_cap),
            )
            kept_3d = int(sum(1 for p in kept if str(p.patch_type) in three_d_patch_types))
            kept_2d = int(len(kept) - kept_3d)
            has_conf = bool(item.conf_ids)
            has_3d_kept = bool(kept_3d > 0)
            return kept, kept_2d, kept_3d, has_conf, has_3d_kept

        def _emit_progress(
            done_molecules: int,
            n_patches: int,
            n_2d_patches: int,
            n_3d_patches: int,
            done_mols_with_conf: int,
            done_mols_with_3d: int,
        ) -> None:
            if total_molecules <= 0:
                return
            elapsed = max(1e-9, float(time.perf_counter() - t0))
            rate = float(done_molecules) / elapsed
            eta_s = float(total_molecules - done_molecules) / rate if rate > 0.0 else None
            log_event(
                "PROGRESS",
                "explainability.chem_ace.generate_patches",
                done=f"{int(done_molecules)}/{int(total_molecules)}",
                pct=f"{(100.0 * done_molecules / float(total_molecules)):.1f}",
                patches=int(n_patches),
                patches_2d=int(n_2d_patches),
                patches_3d=int(n_3d_patches),
                mols_with_conf_done=f"{int(done_mols_with_conf)}/{int(n_molecules_with_conf_total)}",
                mols_with_3d_patches_done=int(done_mols_with_3d),
                mol_per_s=f"{rate:.2f}",
                eta_s=(f"{eta_s:.1f}" if eta_s is not None else "na"),
                cpu_workers=int(workers),
                **extras,
            )

        all_patches: list[PatchRecord] = []
        done = 0
        kept_2d_total = 0
        kept_3d_total = 0
        done_with_conf = 0
        done_with_3d = 0
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_gen_for_molecule, item) for item in molecules]
                for fut in as_completed(futures):
                    patch_list, kept_2d, kept_3d, has_conf, has_3d = fut.result()
                    all_patches.extend(patch_list)
                    kept_2d_total += int(kept_2d)
                    kept_3d_total += int(kept_3d)
                    done_with_conf += int(1 if has_conf else 0)
                    done_with_3d += int(1 if has_3d else 0)
                    done += 1
                    if done == total_molecules or (
                        progress_every > 0 and (done % progress_every == 0)
                    ):
                        _emit_progress(
                            done,
                            len(all_patches),
                            kept_2d_total,
                            kept_3d_total,
                            done_with_conf,
                            done_with_3d,
                        )
        else:
            for item in molecules:
                patch_list, kept_2d, kept_3d, has_conf, has_3d = _gen_for_molecule(item)
                all_patches.extend(patch_list)
                kept_2d_total += int(kept_2d)
                kept_3d_total += int(kept_3d)
                done_with_conf += int(1 if has_conf else 0)
                done_with_3d += int(1 if has_3d else 0)
                done += 1
                if done == total_molecules or (
                    progress_every > 0 and (done % progress_every == 0)
                ):
                    _emit_progress(
                        done,
                        len(all_patches),
                        kept_2d_total,
                        kept_3d_total,
                        done_with_conf,
                        done_with_3d,
                    )
        log_event(
            "INFO",
            "explainability.chem_ace.generate_patches.summary",
            n_molecules=int(total_molecules),
            n_molecules_with_conf=int(n_molecules_with_conf_total),
            n_molecules_with_3d_patches=int(done_with_3d),
            patches_total=int(len(all_patches)),
            patches_2d=int(kept_2d_total),
            patches_3d=int(kept_3d_total),
            **extras,
        )
        log_event(
            "START",
            "explainability.chem_ace.persist_patches",
            n_patches=int(len(all_patches)),
        )
        self.repository.upsert_patches(all_patches)
        log_event(
            "DONE",
            "explainability.chem_ace.persist_patches",
            n_patches=int(len(all_patches)),
        )
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
        log_event(
            "INFO",
            "explainability.chem_ace.discover_concepts.config",
            n_embeddings=int(len(embeddings)),
            algorithms=",".join(str(x) for x in self.config.discovery.algorithms),
            kmeans_k=int(self.config.discovery.kmeans_k),
            hierarchical_max_samples=int(self.config.discovery.hierarchical_max_samples),
            hierarchical_max_pairwise_gb=float(self.config.discovery.hierarchical_max_pairwise_gb),
            hdbscan_max_samples=int(self.config.discovery.hdbscan_max_samples),
        )
        concept_set = discover_concepts(
            embeddings=embeddings,
            config=self.config.discovery,
            seed=int(self.config.seed),
        )
        concept_set_id = self.repository.create_concept_set_snapshot(
            run_id=run_id,
            layer_name=concept_set.layer_name,
            config={
                "discovery": asdict(self.config.discovery),
                "embedding": asdict(self.config.embedding),
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
        inst_by_pair: Optional[Mapping[tuple[str, str], np.ndarray]] = None,
        inst_mean_by_id: Optional[Mapping[str, np.ndarray]] = None,
        inst_geom_dim: int = 0,
        inst_qm_dim: int = 0,
        geom_feature_names: Optional[Sequence[str]] = None,
        qm_feature_names: Optional[Sequence[str]] = None,
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
                inst_by_pair=inst_by_pair,
                inst_mean_by_id=inst_mean_by_id,
                inst_geom_dim=int(inst_geom_dim),
                inst_qm_dim=int(inst_qm_dim),
                geom_feature_names=geom_feature_names,
                qm_feature_names=qm_feature_names,
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
