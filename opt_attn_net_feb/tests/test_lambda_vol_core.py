from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from explainability.lambda_vol.config import DetectorConfig, ExportConfig, LambdaVolConfig, StoreConfig, TrackerConfig
from explainability.lambda_vol.detectors import ConceptPressureDetector, concentration_metrics
from explainability.lambda_vol.monitor import LambdaVolMonitor
from explainability.lambda_vol.tracker import ConceptPressureTracker
from explainability.lambda_vol.types import RegimeLabel


class LambdaVolCoreTest(unittest.TestCase):
    def _frames(self, tcav: np.ndarray, attn: np.ndarray, prev: np.ndarray, task_ids, concept_ids):
        tcav_rows = []
        concept_rows = []
        for ti, task_id in enumerate(task_ids):
            for ci, concept_id in enumerate(concept_ids):
                tcav_rows.append({"task_id": task_id, "concept_id": concept_id, "tcav": float(tcav[ti, ci])})
                concept_rows.append(
                    {
                        "task_id": task_id,
                        "concept_id": concept_id,
                        "attention_support": float(attn[ti, ci]),
                        "prevalence": float(prev[ti, ci]),
                    }
                )

        task_attention = pd.DataFrame(
            [{"task_id": t, "attention_entropy": 0.5, "witness_rate": 0.2} for t in task_ids]
        )
        task_metrics = pd.DataFrame(
            [{"task_id": t, "train_metric": 0.2, "val_metric": 0.18, "loss": 1.0, "calibration_error": 0.1} for t in task_ids]
        )
        return pd.DataFrame(tcav_rows), pd.DataFrame(concept_rows), task_attention, task_metrics

    def test_tracker_rho_and_drift(self) -> None:
        task_ids = ["t0", "t1"]
        concept_ids = ["c0", "c1", "c2"]

        tracker = ConceptPressureTracker(
            task_ids=task_ids,
            concept_ids=concept_ids,
            config=TrackerConfig(alpha=0.5, tcav_ema_beta=0.0, drift_clip=10.0),
        )

        tcav0 = np.array([[0.2, 0.1, 0.0], [0.3, 0.2, 0.1]], dtype=np.float32)
        attn0 = np.array([[0.6, 0.4, 0.2], [0.7, 0.5, 0.3]], dtype=np.float32)
        prev0 = np.full_like(tcav0, 0.2)
        t0, c0, ta0, tm0 = self._frames(tcav0, attn0, prev0, task_ids, concept_ids)
        _, _, state0 = tracker.update_from_frames(
            epoch=0,
            regime_label=RegimeLabel.WARMUP,
            tcav_df=t0,
            concept_attention_df=c0,
            task_attention_df=ta0,
            task_metrics_df=tm0,
            context_covariates={"x": 0.1},
        )

        expected_rho0 = 0.5 * tcav0 + 0.5 * attn0
        self.assertTrue(np.allclose(state0.rho, expected_rho0, atol=1e-6))
        self.assertTrue(np.allclose(state0.drift, 0.0, atol=1e-6))

        tcav1 = tcav0 + 0.1
        attn1 = attn0 + 0.05
        t1, c1, ta1, tm1 = self._frames(tcav1, attn1, prev0, task_ids, concept_ids)
        _, _, state1 = tracker.update_from_frames(
            epoch=1,
            regime_label=RegimeLabel.FITTING,
            tcav_df=t1,
            concept_attention_df=c1,
            task_attention_df=ta1,
            task_metrics_df=tm1,
            context_covariates={"x": 0.2},
        )

        expected_rho1 = 0.5 * tcav1 + 0.5 * attn1
        self.assertTrue(np.allclose(state1.rho, expected_rho1, atol=1e-6))
        self.assertTrue(np.allclose(state1.drift, expected_rho1 - expected_rho0, atol=1e-6))

    def test_concentration_and_alerts(self) -> None:
        cm = concentration_metrics(np.array([0.9, 0.05, 0.05], dtype=np.float32), top_k=1)
        self.assertGreater(cm.topk_mass, 0.89)

        detector = ConceptPressureDetector(
            DetectorConfig(
                runaway_threshold=0.8,
                concentration_top_k=1,
                concentration_topk_mass_alert=0.8,
                blocked_concept_positive_drift=0.02,
            )
        )

        rho = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)
        drift = np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
        diss = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)

        out = detector.detect(
            run_id="r1",
            epoch=2,
            task_ids=["t0"],
            concept_ids=["c0", "c1", "c2"],
            rho=rho,
            drift=drift,
            dissipation=diss,
            blocked_concepts=["c0"],
            prev_concentration=None,
        )

        codes = {a.code for a in out.alerts}
        self.assertIn("concept_collapse_topk_mass", codes)
        self.assertIn("runaway_concept_pressure", codes)
        self.assertIn("blocked_concept_positive_drift", codes)

    def test_monitor_end_to_end_export(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            cfg = LambdaVolConfig(
                run_name="unit_test",
                blocked_concepts=("c0",),
                exporter=ExportConfig(
                    output_dir=str(out_dir),
                    export_parquet=False,
                    export_plotly_html=False,
                    export_vtk=False,
                ),
                store=StoreConfig(db_uri=f"sqlite:///{Path(td) / 'lv.sqlite3'}"),
            )

            monitor = LambdaVolMonitor(
                config=cfg,
                task_ids=["t0", "t1"],
                concept_ids=["c0", "c1", "c2"],
                concept_metadata={"c0": {"family": "charge"}},
            )

            for epoch in range(3):
                tcav = np.array(
                    [
                        [0.10 + 0.03 * epoch, 0.04, 0.03],
                        [0.08, 0.07 + 0.02 * epoch, 0.02],
                    ],
                    dtype=np.float32,
                )
                attn = np.array(
                    [
                        [0.35 + 0.04 * epoch, 0.12, 0.08],
                        [0.20, 0.24 + 0.03 * epoch, 0.07],
                    ],
                    dtype=np.float32,
                )
                prev = np.array(
                    [
                        [0.30, 0.10, 0.06],
                        [0.18, 0.22, 0.05],
                    ],
                    dtype=np.float32,
                )

                tcav_df, concept_df, task_attn_df, task_metrics_df = self._frames(
                    tcav,
                    attn,
                    prev,
                    ["t0", "t1"],
                    ["c0", "c1", "c2"],
                )
                monitor.step_from_frames(
                    epoch=epoch,
                    tcav_df=tcav_df,
                    concept_attention_df=concept_df,
                    task_attention_df=task_attn_df,
                    task_metrics_df=task_metrics_df,
                    context_covariates={"val_macro": 0.2 + 0.01 * epoch},
                )

            artifacts = monitor.finalize()
            self.assertTrue(Path(artifacts.tensor_npz).exists())
            self.assertTrue(Path(artifacts.long_csv).exists())
            self.assertTrue(Path(artifacts.metadata_json).exists())
            self.assertTrue(Path(artifacts.alerts_json).exists())
            self.assertTrue(Path(artifacts.recommendations_json).exists())


if __name__ == "__main__":
    unittest.main()
