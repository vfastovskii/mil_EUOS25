from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    from ..entrypoints.lambda_vol_demo import main as lambda_vol_demo_main
except Exception:  # pragma: no cover
    from opt_attn_net_feb.entrypoints.lambda_vol_demo import main as lambda_vol_demo_main


class LambdaVolDemoSmokeTest(unittest.TestCase):
    def test_demo_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "demo"
            lambda_vol_demo_main(
                [
                    "--output_dir",
                    str(out_dir),
                    "--epochs",
                    "5",
                    "--num_tasks",
                    "3",
                    "--num_concepts",
                    "8",
                    "--no_plotly",
                    "--no_parquet",
                ]
            )

            summary_path = out_dir / "lambda_vol_demo_summary.json"
            self.assertTrue(summary_path.exists())

            summary = json.loads(summary_path.read_text())
            self.assertIn("run_id", summary)
            self.assertIn("artifacts", summary)

            tensor_npz = Path(summary["artifacts"]["tensor_npz"])
            long_csv = Path(summary["artifacts"]["long_csv"])
            alerts_json = Path(summary["artifacts"]["alerts_json"])
            self.assertTrue(tensor_npz.exists())
            self.assertTrue(long_csv.exists())
            self.assertTrue(alerts_json.exists())


if __name__ == "__main__":
    unittest.main()
