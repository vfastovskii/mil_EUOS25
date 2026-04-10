from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

torch = pytest.importorskip("torch")

from opt_attn_net_feb.models.attention_pooling.pool import TaskAttentionPool
from opt_attn_net_feb.models.multimodal_mil.model import MILTaskAttnMixerWithAux


class MultimodalFusionModelTest(unittest.TestCase):
    def test_attention_dropout_keeps_at_least_one_valid_conformer(self) -> None:
        pool = TaskAttentionPool(
            dim=8,
            n_heads=2,
            dropout=0.95,
            n_tasks=4,
            pool_from="normed_inputs",
        )
        pool.train()

        tokens = torch.randn(3, 4, 8)
        key_padding_mask = torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
            ],
            dtype=torch.bool,
        )

        for _ in range(8):
            pooled, attn = pool(tokens, key_padding_mask=key_padding_mask, return_attn=True)
            self.assertTrue(torch.isfinite(pooled).all())
            self.assertIsNotNone(attn)
            attn_t = attn if attn is not None else torch.zeros(3, 4, 4)
            self.assertTrue(torch.allclose(attn_t.sum(dim=-1), torch.ones(3, 4), atol=1e-6))
            self.assertTrue(torch.allclose(attn_t[0, :, 0], torch.ones(4), atol=1e-6))
            self.assertTrue(bool(torch.all(attn_t.masked_select(key_padding_mask.unsqueeze(1)) == 0.0).item()))

    def test_alignment_attention_uses_3d_modality_gates(self) -> None:
        model = MILTaskAttnMixerWithAux(
            mol_dim=6,
            inst_dim=7,
            inst_geom_dim=4,
            inst_qm_dim=3,
            mol_hidden=16,
            mol_layers=2,
            mol_dropout=0.1,
            inst_hidden=8,
            inst_layers=2,
            inst_dropout=0.1,
            proj_dim=8,
            attn_heads=2,
            attn_dropout=0.1,
            mixer_hidden=16,
            mixer_layers=2,
            mixer_dropout=0.1,
            lr=1e-3,
            weight_decay=1e-5,
            pos_weight=torch.ones((4,), dtype=torch.float32),
            gamma=torch.zeros((4,), dtype=torch.float32),
            lam=[1.0, 1.0, 1.0, 1.0],
            lambda_aux_abs=0.0,
            lambda_aux_fluo=0.0,
            lambda_aux_bitmask=0.0,
            reg_loss_type="mse",
        )

        attn_geom = torch.tensor([[[0.8, 0.2], [0.1, 0.9], [0.3, 0.7], [0.4, 0.6]]], dtype=torch.float32)
        attn_qm = torch.tensor([[[0.2, 0.8], [0.6, 0.4], [0.9, 0.1], [0.5, 0.5]]], dtype=torch.float32)
        modality_gates = torch.tensor(
            [[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.25, 0.55], [0.5, 0.0, 1.0]]],
            dtype=torch.float32,
        )

        fused = model._resolve_alignment_attention(
            {
                "attn_geom": attn_geom,
                "attn_qm": attn_qm,
                "modality_gates": modality_gates,
                "modality_order": ("2d", "3d_geom", "3d_qm"),
            }
        )

        self.assertIsNotNone(fused)
        fused_t = fused if fused is not None else torch.zeros_like(attn_geom)

        # Task 0: 3d weights are renormalized from geom=0.2, qm=0.1 -> 2/3 and 1/3
        expected_t0 = (2.0 / 3.0) * attn_geom[0, 0] + (1.0 / 3.0) * attn_qm[0, 0]
        self.assertTrue(torch.allclose(fused_t[0, 0], expected_t0, atol=1e-6))

        # Task 1: geom dominates 0.8 vs 0.1
        expected_t1 = (8.0 / 9.0) * attn_geom[0, 1] + (1.0 / 9.0) * attn_qm[0, 1]
        self.assertTrue(torch.allclose(fused_t[0, 1], expected_t1, atol=1e-6))

        # Task 3: geom gate is zero, so output should equal QM attention.
        self.assertTrue(torch.allclose(fused_t[0, 3], attn_qm[0, 3], atol=1e-6))

    def test_forward_exposes_modality_gates_and_attention(self) -> None:
        model = MILTaskAttnMixerWithAux(
            mol_dim=6,
            inst_dim=7,
            inst_geom_dim=4,
            inst_qm_dim=3,
            mol_hidden=16,
            mol_layers=2,
            mol_dropout=0.1,
            inst_hidden=8,
            inst_layers=2,
            inst_dropout=0.1,
            proj_dim=8,
            attn_heads=2,
            attn_dropout=0.1,
            mixer_hidden=16,
            mixer_layers=2,
            mixer_dropout=0.1,
            lr=1e-3,
            weight_decay=1e-5,
            pos_weight=torch.ones((4,), dtype=torch.float32),
            gamma=torch.zeros((4,), dtype=torch.float32),
            lam=[1.0, 1.0, 1.0, 1.0],
            lambda_aux_abs=0.0,
            lambda_aux_fluo=0.0,
            lambda_aux_bitmask=0.0,
            reg_loss_type="mse",
        )
        x2d = torch.randn(3, 6)
        x3d = torch.randn(3, 5, 7)
        kpm = torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, True, True, True],
                [False, False, False, False, True],
            ],
            dtype=torch.bool,
        )

        logits, _, _, attn = model(
            x2d,
            x3d,
            kpm,
            return_attn=True,
            return_attn_modalities=True,
        )
        self.assertEqual(tuple(logits.shape), (3, 4))
        self.assertIsInstance(attn, dict)
        self.assertIn("modality_gates", attn)
        self.assertEqual(tuple(attn["modality_gates"].shape), (3, 4, 3))
        sums = attn["modality_gates"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        self.assertIn("modality_channel_gate_mean", attn)
        self.assertEqual(tuple(attn["modality_channel_gate_mean"].shape), (3, 4, 3))
        self.assertIn("modality_attn", attn)
        self.assertEqual(tuple(attn["modality_attn"].shape), (3, 4, 3, 3))
        self.assertIn("pairwise_weights", attn)
        self.assertEqual(tuple(attn["pairwise_weights"].shape), (3, 4, 3))
        self.assertEqual(tuple(attn["pairwise_order"]), ("2d|3d_geom", "2d|3d_qm", "3d_geom|3d_qm"))
        self.assertEqual(tuple(attn["attn_geom"].shape), (3, 4, 5))
        self.assertEqual(tuple(attn["attn_qm"].shape), (3, 4, 5))

    def test_single_modality_family_has_unit_gate(self) -> None:
        model = MILTaskAttnMixerWithAux(
            mol_dim=6,
            inst_dim=1,
            inst_geom_dim=0,
            inst_qm_dim=0,
            mol_hidden=16,
            mol_layers=2,
            mol_dropout=0.1,
            inst_hidden=8,
            inst_layers=2,
            inst_dropout=0.1,
            proj_dim=8,
            attn_heads=2,
            attn_dropout=0.1,
            mixer_hidden=16,
            mixer_layers=2,
            mixer_dropout=0.1,
            lr=1e-3,
            weight_decay=1e-5,
            pos_weight=torch.ones((4,), dtype=torch.float32),
            gamma=torch.zeros((4,), dtype=torch.float32),
            lam=[1.0, 1.0, 1.0, 1.0],
            lambda_aux_abs=0.0,
            lambda_aux_fluo=0.0,
            lambda_aux_bitmask=0.0,
            reg_loss_type="mse",
        )
        x2d = torch.randn(2, 6)
        x3d = torch.zeros(2, 1, 1)
        kpm = torch.tensor([[False], [False]], dtype=torch.bool)

        logits, _, _, attn = model(
            x2d,
            x3d,
            kpm,
            return_attn=True,
            return_attn_modalities=True,
        )
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(attn["modality_gates"].shape), (2, 4, 1))
        self.assertTrue(torch.allclose(attn["modality_gates"], torch.ones_like(attn["modality_gates"])))
        self.assertEqual(tuple(attn["modality_channel_gate_mean"].shape), (2, 4, 1))
        self.assertEqual(tuple(attn["pairwise_weights"].shape), (2, 4, 0))


if __name__ == "__main__":
    unittest.main()
