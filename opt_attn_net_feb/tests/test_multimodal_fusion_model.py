from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

torch = pytest.importorskip("torch")

from opt_attn_net_feb.models.multimodal_mil.model import MILTaskAttnMixerWithAux


class MultimodalFusionModelTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
