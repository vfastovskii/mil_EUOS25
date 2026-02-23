from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    from ..data.exports import export_prediction_text_explanations
    from ..utils.constants import TASK_COLS
except Exception:  # pragma: no cover
    from data.exports import export_prediction_text_explanations
    from utils.constants import TASK_COLS


class PredictionTextExplanationsTest(unittest.TestCase):
    def test_export_prediction_text_explanations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_path = root / "leaderboard_attn.csv"
            out_path = root / "leaderboard_attn_explained.csv"
            ricci_path = root / "ricci_edges_long.csv"

            rows = []
            for mol_id, conf_id, p0, a0 in [
                ("m1", "c1", 0.83, 0.72),
                ("m2", "d1", 0.21, 0.18),
            ]:
                row = {"ID": mol_id, "conf_id": conf_id}
                for ti, task in enumerate(TASK_COLS):
                    p = p0 if ti == 0 else (0.40 + 0.05 * ti)
                    row[f"pred_{task}"] = float(p)
                    row[f"pred_label_{task}"] = int(float(p) >= 0.5)
                    row[f"attn_{task}"] = float(a0 if ti == 0 else 0.10 + 0.02 * ti)
                rows.append(row)
            pd.DataFrame(rows).to_csv(pred_path, index=False)

            pd.DataFrame(
                [
                    {
                        "epoch": 2,
                        "task_id": str(TASK_COLS[0]),
                        "concept_src": "c_cat",
                        "concept_dst": "c_arom",
                        "weight_raw": 0.8,
                        "curvature": -0.9,
                        "weight_flow": 0.7,
                    },
                    {
                        "epoch": 2,
                        "task_id": str(TASK_COLS[0]),
                        "concept_src": "c_cat",
                        "concept_dst": "c_other",
                        "weight_raw": 0.2,
                        "curvature": -0.2,
                        "weight_flow": 0.1,
                    },
                ]
            ).to_csv(ricci_path, index=False)

            concept_ids = ["c_arom", "c_cat", "c_other"]
            concept_metadata = {
                "c_arom": {
                    "label_auto": "planar aromatic pi-system",
                    "tags": ["aromatic pi-system", "planar"],
                    "support": 120,
                },
                "c_cat": {
                    "label_auto": "cationic amine",
                    "tags": ["cationic center", "HBD"],
                    "support": 64,
                },
                "c_other": {
                    "label_auto": "aliphatic neutral motif",
                    "tags": ["aliphatic"],
                    "support": 18,
                },
            }
            concept_mol_map = {
                "c_arom": {"m1"},
                "c_other": {"m2"},
            }
            concept_conf_map = {
                "c_cat": {("m1", "c1")},
            }

            written = export_prediction_text_explanations(
                pred_table_path=pred_path,
                out_path=out_path,
                concept_ids=concept_ids,
                concept_metadata=concept_metadata,
                concept_mol_map=concept_mol_map,
                concept_conf_map=concept_conf_map,
                task_cols=TASK_COLS,
                ricci_edges_csv=str(ricci_path),
                top_k=2,
                bridge_threshold=0.15,
            )

            self.assertEqual(Path(written), out_path)
            self.assertTrue(out_path.exists())

            df = pd.read_csv(out_path)
            self.assertIn("prediction_explanation", df.columns)
            for task in TASK_COLS:
                self.assertIn(f"top_concepts_{task}", df.columns)
                self.assertIn(f"top_concept_labels_{task}", df.columns)
                self.assertIn(f"prediction_explanation_{task}", df.columns)

            row_m1 = df[(df["ID"] == "m1") & (df["conf_id"] == "c1")].iloc[0]
            expl0 = str(row_m1[f"prediction_explanation_{TASK_COLS[0]}"])
            self.assertIn("planar aromatic pi-system", expl0)
            self.assertIn("cationic amine", expl0)
            self.assertIn("Ricci note", expl0)

            row_m2 = df[(df["ID"] == "m2") & (df["conf_id"] == "d1")].iloc[0]
            expl_m2 = str(row_m2[f"prediction_explanation_{TASK_COLS[0]}"])
            self.assertIn("aliphatic neutral motif", expl_m2)


if __name__ == "__main__":
    unittest.main()

