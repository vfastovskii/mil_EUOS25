from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from ..explainability.chem_ace.analytics import ConceptQueryService
from ..explainability.chem_ace.cav import run_tcav_from_arrays
from ..explainability.chem_ace.config import CAVConfig, ConceptDiscoveryConfig
from ..explainability.chem_ace.concepts import discover_concepts
from ..explainability.chem_ace.db.repository import ChemACERepository
from ..explainability.chem_ace.patches.base import make_patch_record
from ..explainability.chem_ace.types import PatchEmbeddingRecord


class ChemACETest(unittest.TestCase):
    def test_patch_id_deterministic(self) -> None:
        p1 = make_patch_record(
            mol_id="m1",
            conf_id="0",
            patch_type="x",
            atom_indices=[2, 1, 2],
            smarts="[*]-[*]",
            fragment_repr="CC",
            feature_metadata={"a": 1},
        )
        p2 = make_patch_record(
            mol_id="m1",
            conf_id="0",
            patch_type="x",
            atom_indices=[1, 2],
            smarts="[*]-[*]",
            fragment_repr="CC",
            feature_metadata={"a": 1},
        )
        self.assertEqual(p1.patch_hash, p2.patch_hash)
        self.assertEqual(p1.patch_id, p2.patch_id)

    def test_concept_discovery(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(loc=-1.0, scale=0.05, size=(20, 8)).astype(np.float32)
        b = rng.normal(loc=1.0, scale=0.05, size=(20, 8)).astype(np.float32)
        x = np.concatenate([a, b], axis=0)
        embeddings = [
            PatchEmbeddingRecord(
                patch_id=f"p{i}",
                layer_name="encoder",
                strategy="masked_input",
                vector=x[i],
                embedding_uri=None,
                metadata={},
            )
            for i in range(x.shape[0])
        ]
        cfg = ConceptDiscoveryConfig(
            algorithms=("hierarchical",),
            hierarchical_distance_threshold=1.2,
            min_support=5,
            min_coherence=0.0,
            dedup_centroid_similarity_threshold=0.999,
        )
        result = discover_concepts(embeddings=embeddings, config=cfg, seed=0)
        self.assertGreaterEqual(len(result.candidates), 2)
        self.assertGreaterEqual(len(result.memberships), 30)

    def test_tcav_arrays(self) -> None:
        rng = np.random.default_rng(0)
        concept = rng.normal(size=(16, 8)).astype(np.float32)
        random_pool = rng.normal(size=(64, 8)).astype(np.float32)
        grads = rng.normal(size=(24, 8)).astype(np.float32)

        cavs, tcavs, summary = run_tcav_from_arrays(
            run_id="r1",
            epoch=0,
            concept_id="c1",
            task_id="t0",
            layer_name="encoder",
            concept_embeddings=concept,
            random_pool_embeddings=random_pool,
            target_gradients=grads,
            config=CAVConfig(n_random_repeats=3, random_counterexamples_per_repeat=16),
            seed=7,
        )
        self.assertEqual(len(cavs), 3)
        self.assertEqual(len(tcavs), 3)
        self.assertGreaterEqual(summary.mean_sign_rate, 0.0)
        self.assertLessEqual(summary.mean_sign_rate, 1.0)

    def test_repository_and_queries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_uri = f"sqlite:///{Path(td) / 'chem_ace.sqlite3'}"
            repo = ChemACERepository(db_uri=db_uri, artifact_dir=str(Path(td) / "artifacts"))

            run_id = repo.create_run(run_name="test", config={"a": 1})
            repo.ensure_tasks(["t0"])

            patch = make_patch_record(
                mol_id="mol_1",
                conf_id=None,
                patch_type="toy",
                atom_indices=[0, 1],
                smarts="[*]-[*]",
                fragment_repr="CC",
                feature_metadata={},
            )
            repo.upsert_patch(patch)

            emb_uri = Path(td) / "emb.npy"
            np.save(emb_uri, np.ones((8,), dtype=np.float32))
            repo.upsert_patch_embedding(
                PatchEmbeddingRecord(
                    patch_id=patch.patch_id,
                    layer_name="encoder",
                    strategy="masked_input",
                    vector=np.ones((8,), dtype=np.float32),
                    embedding_uri=str(emb_uri),
                    metadata={},
                )
            )

            concept_set_id = repo.create_concept_set_snapshot(
                run_id=run_id,
                layer_name="encoder",
                config={"x": 1},
                metadata={"m": 1},
            )

            from explainability.chem_ace.types import ConceptCandidate, ConceptMembership, TCAVRecord

            cand = ConceptCandidate(
                concept_local_id="c1",
                layer_name="encoder",
                algorithm="kmeans",
                support=1,
                coherence=0.5,
                centroid=np.ones((8,), dtype=np.float32),
                medoid_patch_id=patch.patch_id,
                metadata={},
            )
            repo.upsert_concept(concept_set_id=concept_set_id, cand=cand)
            repo.upsert_memberships([
                ConceptMembership(
                    concept_local_id="c1",
                    patch_id=patch.patch_id,
                    membership_score=1.0,
                    distance_to_centroid=0.0,
                )
            ])

            repo.upsert_tcav_epoch(
                TCAVRecord(
                    run_id=run_id,
                    epoch=0,
                    concept_id="c1",
                    task_id="t0",
                    layer_name="encoder",
                    seed=0,
                    tcav_sign_rate=0.8,
                    tcav_mean_directional_derivative=0.2,
                    n_samples=10,
                    p_value=0.1,
                    metadata={},
                )
            )

            query = ConceptQueryService(session_factory=repo.SessionFactory)
            collapse = query.concept_collapse_indicator(task_id="t0", epoch=0, top_k=1)
            self.assertIn("collapse_index", collapse)
            self.assertEqual(collapse["n_concepts"], 1)


if __name__ == "__main__":
    unittest.main()
