from __future__ import annotations

import inspect
import unittest
from pathlib import Path
import sys

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from opt_attn_net_feb.explainability.chem_ace.types import PatchEmbeddingRecord, PatchRecord
pytest.importorskip("torch")
from opt_attn_net_feb.training.explainability_runtime import _build_hybrid_patch_embeddings, prepare_chem_ace_bundle


class _DummyEmbeddingCache:
    def save(self, *, patch, layer_name, strategy, vector, metadata):
        return PatchEmbeddingRecord(
            patch_id=str(patch.patch_id),
            layer_name=str(layer_name),
            strategy=str(strategy),
            vector=np.asarray(vector, dtype=np.float32),
            embedding_uri=None,
            metadata=dict(metadata),
        )


class _DummyRepo:
    def upsert_patch_embeddings(self, _recs):
        return None


class _DummyPipeline:
    embedding_cache = _DummyEmbeddingCache()
    repository = _DummyRepo()


class ChemACEHybridEmbeddingTest(unittest.TestCase):
    def test_modality_dims(self) -> None:
        patches = [
            PatchRecord(
                patch_id="p2d",
                mol_id="m1",
                conf_id=None,
                patch_type="local_subgraph",
                atom_indices=(0, 1),
                patch_hash="h2d",
            ),
            PatchRecord(
                patch_id="p3dg",
                mol_id="m1",
                conf_id="c0",
                patch_type="pharm3d",
                atom_indices=(0, 1, 2),
                patch_hash="h3dg",
            ),
            PatchRecord(
                patch_id="p3dq",
                mol_id="m1",
                conf_id="c1",
                patch_type="pharm3d",
                atom_indices=(1, 2, 3),
                patch_hash="h3dq",
            ),
        ]
        x2d = {"m1": np.ones((10,), dtype=np.float32)}
        # conf c0 has zero QM slice -> 3d_geom
        # conf c1 has non-zero QM slice -> 3d_qm
        xinst = {
            ("m1", "c0"): np.asarray([1, 1, 1, 1, 0, 0, 0], dtype=np.float32),
            ("m1", "c1"): np.asarray([1, 1, 1, 1, 0.1, -0.2, 0.3], dtype=np.float32),
        }
        out, _ = _build_hybrid_patch_embeddings(
            pipeline=_DummyPipeline(),
            patches=patches,
            molecules_by_id={"m1": None},
            x2d_by_id=x2d,
            xinst_by_pair=xinst,
            xinst_mean_by_id={},
            inst_geom_dim=4,
            inst_qm_dim=3,
            embed_dim_2d=32,
            embed_dim_3d_geom=24,
            embed_dim_3d_qm=16,
            context_dim=8,
            context_alpha=0.2,
            qm_gating=True,
            fit_descriptor_scaler=False,
            descriptor_scaler=None,
            persist_embeddings=False,
            n_workers=0,
            seed=7,
        )
        self.assertEqual(len(out["2d"]), 1)
        self.assertEqual(len(out["3d_geom"]), 1)
        self.assertEqual(len(out["3d_qm"]), 1)
        self.assertEqual(int(out["2d"][0].vector.shape[0]), 32)
        self.assertEqual(int(out["3d_geom"][0].vector.shape[0]), 24)
        self.assertEqual(int(out["3d_qm"][0].vector.shape[0]), 16)

    def test_different_patches_not_identical(self) -> None:
        patches = [
            PatchRecord(
                patch_id="p_a",
                mol_id="m1",
                conf_id=None,
                patch_type="local_subgraph",
                atom_indices=(0, 1),
                patch_hash="ha",
            ),
            PatchRecord(
                patch_id="p_b",
                mol_id="m1",
                conf_id=None,
                patch_type="brics",
                atom_indices=(2, 3),
                patch_hash="hb",
            ),
        ]
        out, _ = _build_hybrid_patch_embeddings(
            pipeline=_DummyPipeline(),
            patches=patches,
            molecules_by_id={"m1": None},
            x2d_by_id={"m1": np.ones((12,), dtype=np.float32)},
            xinst_by_pair={},
            xinst_mean_by_id={},
            inst_geom_dim=0,
            inst_qm_dim=0,
            embed_dim_2d=32,
            embed_dim_3d_geom=24,
            embed_dim_3d_qm=16,
            context_dim=8,
            context_alpha=0.2,
            qm_gating=True,
            fit_descriptor_scaler=False,
            descriptor_scaler=None,
            persist_embeddings=False,
            n_workers=0,
            seed=5,
        )
        v0 = out["2d"][0].vector
        v1 = out["2d"][1].vector
        self.assertFalse(np.allclose(v0, v1))

    def test_context_alpha_clipped(self) -> None:
        patches = [
            PatchRecord(
                patch_id="p",
                mol_id="m1",
                conf_id=None,
                patch_type="local_subgraph",
                atom_indices=(0, 1),
                patch_hash="h",
            )
        ]
        out_hi, _ = _build_hybrid_patch_embeddings(
            pipeline=_DummyPipeline(),
            patches=patches,
            molecules_by_id={"m1": None},
            x2d_by_id={"m1": np.ones((6,), dtype=np.float32)},
            xinst_by_pair={},
            xinst_mean_by_id={},
            inst_geom_dim=0,
            inst_qm_dim=0,
            embed_dim_2d=16,
            embed_dim_3d_geom=16,
            embed_dim_3d_qm=16,
            context_dim=4,
            context_alpha=2.0,
            qm_gating=True,
            fit_descriptor_scaler=False,
            descriptor_scaler=None,
            persist_embeddings=False,
            n_workers=0,
            seed=1,
        )
        out_lo, _ = _build_hybrid_patch_embeddings(
            pipeline=_DummyPipeline(),
            patches=patches,
            molecules_by_id={"m1": None},
            x2d_by_id={"m1": np.ones((6,), dtype=np.float32)},
            xinst_by_pair={},
            xinst_mean_by_id={},
            inst_geom_dim=0,
            inst_qm_dim=0,
            embed_dim_2d=16,
            embed_dim_3d_geom=16,
            embed_dim_3d_qm=16,
            context_dim=4,
            context_alpha=-1.0,
            qm_gating=True,
            fit_descriptor_scaler=False,
            descriptor_scaler=None,
            persist_embeddings=False,
            n_workers=0,
            seed=1,
        )
        self.assertEqual(float(out_hi["2d"][0].metadata["context_alpha"]), 1.0)
        self.assertEqual(float(out_lo["2d"][0].metadata["context_alpha"]), 0.0)

    def test_no_legacy_feature_projection_in_final_runtime(self) -> None:
        src = inspect.getsource(prepare_chem_ace_bundle)
        self.assertIn("_build_hybrid_patch_embeddings", src)
        self.assertNotIn("_build_feature_patch_embeddings", src)


if __name__ == "__main__":
    unittest.main()
