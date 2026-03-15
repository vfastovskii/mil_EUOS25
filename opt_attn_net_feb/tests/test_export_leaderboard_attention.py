from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional test dependency
    torch = None  # type: ignore
    nn = object  # type: ignore
    _HAS_TORCH = False

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from opt_attn_net_feb.data.exports import export_attention_dataset_summary
from opt_attn_net_feb.utils.constants import TASK_COLS

if _HAS_TORCH:
    from opt_attn_net_feb.data.exports import export_leaderboard_attention
else:  # pragma: no cover - unavailable torch runtime
    export_leaderboard_attention = None  # type: ignore


if _HAS_TORCH:
    class _DummyExportModel(nn.Module):
        def forward(self, x2d, x3d, kpm, return_attn: bool = True, return_attn_modalities: bool = False):
            b, _n, _f = x3d.shape
            t = len(TASK_COLS)
            logits = x2d[:, :t]
            attn_geom = torch.ones((b, t, x3d.shape[1]), dtype=x3d.dtype, device=x3d.device)
            attn_qm = torch.ones((b, t, x3d.shape[1]), dtype=x3d.dtype, device=x3d.device)
            gates = torch.full((b, t, 3), 1.0 / 3.0, dtype=x3d.dtype, device=x3d.device)
            if return_attn_modalities:
                attn = {
                    "attn_geom": attn_geom,
                    "attn_qm": attn_qm,
                    "modality_gates": gates,
                    "modality_order": ("2d", "3d_geom", "3d_qm"),
                }
            else:
                attn = 0.5 * (attn_geom + attn_qm)
            return logits, None, None, attn
else:  # pragma: no cover - unavailable torch runtime
    class _DummyExportModel:  # type: ignore[no-redef]
        pass


class ExportLeaderboardAttentionTest(unittest.TestCase):
    def test_attention_dataset_summary_exports_gate_stats_and_top_conformers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "leaderboard_mt_2d3d_attention.csv"
            rows = [
                {
                    "ID": "m1",
                    "conf_id": "c1",
                    f"pred_{task}": 0.8 if i == 0 else 0.1,
                    f"pred_label_{task}": 1 if i == 0 else 0,
                    f"true_label_{task}": 1 if i == 0 else 0,
                    f"fusion_gate_2d_{task}": 0.7,
                    f"fusion_gate_3d_geom_{task}": 0.2,
                    f"fusion_gate_3d_qm_{task}": 0.1,
                    f"attn_geom_{task}": 0.8 if i == 0 else 0.5,
                    f"attn_qm_{task}": 0.3 if i == 0 else 0.2,
                }
                for i, task in enumerate(TASK_COLS)
            ]
            row_m1_c1 = {"ID": "m1", "conf_id": "c1"}
            row_m1_c2 = {"ID": "m1", "conf_id": "c2"}
            row_m2_d1 = {"ID": "m2", "conf_id": "d1"}
            row_m2_d2 = {"ID": "m2", "conf_id": "d2"}
            for i, task in enumerate(TASK_COLS):
                row_m1_c1.update(rows[i])
                row_m1_c2.update(
                    {
                        f"pred_{task}": row_m1_c1[f"pred_{task}"],
                        f"pred_label_{task}": row_m1_c1[f"pred_label_{task}"],
                        f"true_label_{task}": row_m1_c1[f"true_label_{task}"],
                        f"fusion_gate_2d_{task}": 0.7,
                        f"fusion_gate_3d_geom_{task}": 0.2,
                        f"fusion_gate_3d_qm_{task}": 0.1,
                        f"attn_geom_{task}": 0.2 if i == 0 else 0.1,
                        f"attn_qm_{task}": 0.7 if i == 0 else 0.1,
                    }
                )
                row_m2_d1.update(
                    {
                        f"pred_{task}": 0.2 if i == 0 else 0.05,
                        f"pred_label_{task}": 0,
                        f"true_label_{task}": 0,
                        f"fusion_gate_2d_{task}": 0.2,
                        f"fusion_gate_3d_geom_{task}": 0.3,
                        f"fusion_gate_3d_qm_{task}": 0.5,
                        f"attn_geom_{task}": 0.6 if i == 0 else 0.4,
                        f"attn_qm_{task}": 0.1 if i == 0 else 0.6,
                    }
                )
                row_m2_d2.update(
                    {
                        f"pred_{task}": row_m2_d1[f"pred_{task}"],
                        f"pred_label_{task}": row_m2_d1[f"pred_label_{task}"],
                        f"true_label_{task}": row_m2_d1[f"true_label_{task}"],
                        f"fusion_gate_2d_{task}": 0.2,
                        f"fusion_gate_3d_geom_{task}": 0.3,
                        f"fusion_gate_3d_qm_{task}": 0.5,
                        f"attn_geom_{task}": 0.4 if i == 0 else 0.6,
                        f"attn_qm_{task}": 0.9 if i == 0 else 0.4,
                    }
                )
            pd.DataFrame([row_m1_c1, row_m1_c2, row_m2_d1, row_m2_d2]).to_csv(src, index=False)

            out = export_attention_dataset_summary(pred_table_path=src, top_k_per_task=2)

            gate_df = pd.read_csv(out["fusion_gate_summary"])
            top_df = pd.read_csv(out["top_conformers"])

            gate_row = gate_df[
                (gate_df["task"] == str(TASK_COLS[0]))
                & (gate_df["subset"] == "all")
                & (gate_df["modality"] == "2d")
            ].iloc[0]
            self.assertEqual(int(gate_row["n_molecules"]), 2)
            self.assertAlmostEqual(float(gate_row["mean"]), 0.45, places=6)

            pred_pos_row = gate_df[
                (gate_df["task"] == str(TASK_COLS[0]))
                & (gate_df["subset"] == "predicted_positive")
                & (gate_df["modality"] == "2d")
            ].iloc[0]
            self.assertEqual(int(pred_pos_row["n_molecules"]), 1)
            self.assertAlmostEqual(float(pred_pos_row["mean"]), 0.7, places=6)

            top_geom = top_df[
                (top_df["task"] == str(TASK_COLS[0]))
                & (top_df["modality"] == "geom")
                & (top_df["rank_global"] == 1)
            ].iloc[0]
            self.assertEqual(str(top_geom["ID"]), "m1")
            self.assertEqual(str(top_geom["conf_id"]), "c1")
            self.assertAlmostEqual(float(top_geom["attention"]), 0.8, places=6)

            top_qm = top_df[
                (top_df["task"] == str(TASK_COLS[0]))
                & (top_df["modality"] == "qm")
                & (top_df["rank_global"] == 1)
            ].iloc[0]
            self.assertEqual(str(top_qm["ID"]), "m2")
            self.assertEqual(str(top_qm["conf_id"]), "d2")
            self.assertAlmostEqual(float(top_qm["attention"]), 0.9, places=6)

    @unittest.skipUnless(_HAS_TORCH, "torch is required for export_leaderboard_attention test")
    def test_binary_pred_columns_are_exported(self) -> None:
        model = _DummyExportModel()
        device = torch.device("cpu")

        mol_ids = ["m1", "m2"]
        conf_pad = np.array(
            [
                ["c1", "c2", ""],
                ["d1", "", ""],
            ],
            dtype=object,
        )
        x2d = torch.tensor(
            [
                [0.0, 1.0, -1.0, 2.0],
                [-0.01, 0.0, 0.01, -0.5],
            ],
            dtype=torch.float32,
        )
        x3d = torch.zeros((2, 3, 2), dtype=torch.float32)
        kpm = torch.tensor(
            [
                [False, False, True],
                [False, True, True],
            ],
            dtype=torch.bool,
        )

        dl = [(mol_ids, conf_pad, x2d, x3d, kpm)]

        with tempfile.TemporaryDirectory() as td:
            out_csv = Path(td) / "leaderboard_attn.csv"
            export_leaderboard_attention(
                model=model,
                dl_lb_export=dl,
                device=device,
                out_path=out_csv,
            )

            df = pd.read_csv(out_csv)
            self.assertEqual(len(df), 3)

            for task in TASK_COLS:
                self.assertIn(f"pred_{task}", df.columns)
                self.assertIn(f"pred_label_{task}", df.columns)
                self.assertIn(f"attn_geom_{task}", df.columns)
                self.assertIn(f"attn_qm_{task}", df.columns)
                self.assertIn(f"fusion_gate_2d_{task}", df.columns)
                self.assertIn(f"fusion_gate_3d_geom_{task}", df.columns)
                self.assertIn(f"fusion_gate_3d_qm_{task}", df.columns)

            row_m1 = df[(df["ID"] == "m1") & (df["conf_id"] == "c1")].iloc[0]
            row_m2 = df[(df["ID"] == "m2") & (df["conf_id"] == "d1")].iloc[0]

            expected_m1 = [1, 1, 0, 1]  # sigmoid([0,1,-1,2]) >= 0.5
            expected_m2 = [0, 1, 1, 0]  # sigmoid([-0.01,0,0.01,-0.5]) >= 0.5

            for i, task in enumerate(TASK_COLS):
                self.assertEqual(int(row_m1[f"pred_label_{task}"]), expected_m1[i])
                self.assertEqual(int(row_m2[f"pred_label_{task}"]), expected_m2[i])

    @unittest.skipUnless(_HAS_TORCH, "torch is required for export_leaderboard_attention test")
    def test_true_binary_label_columns_are_exported_when_provided(self) -> None:
        model = _DummyExportModel()
        device = torch.device("cpu")

        mol_ids = ["m1", "m2"]
        conf_pad = np.array(
            [
                ["c1", "c2", ""],
                ["d1", "", ""],
            ],
            dtype=object,
        )
        x2d = torch.tensor(
            [
                [0.0, 1.0, -1.0, 2.0],
                [-0.01, 0.0, 0.01, -0.5],
            ],
            dtype=torch.float32,
        )
        x3d = torch.zeros((2, 3, 2), dtype=torch.float32)
        kpm = torch.tensor(
            [
                [False, False, True],
                [False, True, True],
            ],
            dtype=torch.bool,
        )
        dl = [(mol_ids, conf_pad, x2d, x3d, kpm)]

        true_labels = {
            "m1": [1, 0, 1, 0],
            "m2": [0, 1, 0, 1],
        }

        with tempfile.TemporaryDirectory() as td:
            out_csv = Path(td) / "leaderboard_attn.csv"
            export_leaderboard_attention(
                model=model,
                dl_lb_export=dl,
                device=device,
                out_path=out_csv,
                true_labels_by_id=true_labels,
            )
            df = pd.read_csv(out_csv)
            for task in TASK_COLS:
                self.assertIn(f"true_label_{task}", df.columns)
            row_m1 = df[(df["ID"] == "m1") & (df["conf_id"] == "c1")].iloc[0]
            row_m2 = df[(df["ID"] == "m2") & (df["conf_id"] == "d1")].iloc[0]
            expected_m1 = [1, 0, 1, 0]
            expected_m2 = [0, 1, 0, 1]
            for i, task in enumerate(TASK_COLS):
                self.assertEqual(int(row_m1[f"true_label_{task}"]), expected_m1[i])
                self.assertEqual(int(row_m2[f"true_label_{task}"]), expected_m2[i])

    @unittest.skipUnless(_HAS_TORCH, "torch is required for export_leaderboard_attention test")
    def test_pmapper_signature_columns_and_mass_are_exported(self) -> None:
        model = _DummyExportModel()
        device = torch.device("cpu")

        mol_ids = ["m1"]
        conf_pad = np.array([["c1", "c2", ""]], dtype=object)
        x2d = torch.tensor([[0.0, 1.0, -1.0, 2.0]], dtype=torch.float32)
        x3d = torch.zeros((1, 3, 2), dtype=torch.float32)
        kpm = torch.tensor([[False, False, True]], dtype=torch.bool)
        dl = [(mol_ids, conf_pad, x2d, x3d, kpm)]

        sig_map = {"c1": "sigA", "c2": "sigA"}
        sig_map_alt = {"c1": "sigX", "c2": "sigY"}

        with tempfile.TemporaryDirectory() as td:
            out_csv = Path(td) / "leaderboard_attn.csv"
            export_leaderboard_attention(
                model=model,
                dl_lb_export=dl,
                device=device,
                out_path=out_csv,
                conf_signature_map=sig_map,
                conf_signature_alt_map=sig_map_alt,
            )
            df = pd.read_csv(out_csv)

            self.assertIn("pmapper_sig_md5", df.columns)
            self.assertIn("pmapper_sig_md5_alt", df.columns)
            for task in TASK_COLS:
                self.assertIn(f"pmapper_sig_md5_mass_{task}", df.columns)
                self.assertIn(f"pmapper_sig_md5_rank_{task}", df.columns)
                self.assertIn(f"pmapper_sig_md5_top_{task}", df.columns)

            # Both conformers share primary signature sigA, so per-task mass should be 1.0.
            row0 = df.iloc[0]
            for task in TASK_COLS:
                self.assertAlmostEqual(float(row0[f"pmapper_sig_md5_mass_{task}"]), 1.0, places=6)
                self.assertEqual(int(row0[f"pmapper_sig_md5_top_{task}"]), 1)


if __name__ == "__main__":
    unittest.main()
