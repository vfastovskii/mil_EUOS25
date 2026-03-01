from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    from opt_attn_net_feb.explainability.lambda_vol.config import DetectorConfig, RicciConfig
    from opt_attn_net_feb.explainability.lambda_vol.detectors import ConceptPressureDetector
    from opt_attn_net_feb.explainability.lambda_vol.ricci import ConceptRicciFlowAnalyzer
    from opt_attn_net_feb.explainability.lambda_vol.types import RicciTaskSummary
except Exception:  # pragma: no cover
    from opt_attn_net_feb.explainability.lambda_vol.config import DetectorConfig, RicciConfig
    from opt_attn_net_feb.explainability.lambda_vol.detectors import ConceptPressureDetector
    from opt_attn_net_feb.explainability.lambda_vol.ricci import ConceptRicciFlowAnalyzer
    from opt_attn_net_feb.explainability.lambda_vol.types import RicciTaskSummary


class LambdaVolRicciTest(unittest.TestCase):
    def test_ricci_analyzer_produces_edges_and_flow(self) -> None:
        task_ids = ["t0", "t1"]
        concept_ids = ["c0", "c1", "c2", "c3"]

        analyzer = ConceptRicciFlowAnalyzer(
            task_ids=task_ids,
            concept_ids=concept_ids,
            concept_modalities=["2d", "2d", "3d_geom", "3d_qm"],
            config=RicciConfig(
                enabled=True,
                edge_keep_quantile=0.4,
                min_edge_weight=0.01,
                top_k_per_node=2,
                node_top_k_per_task=4,
                flow_enabled=True,
                flow_steps=3,
                flow_step_size=0.2,
            ),
        )

        tcav = np.asarray(
            [
                [0.40, 0.32, 0.06, 0.10],
                [0.30, 0.20, 0.15, 0.04],
            ],
            dtype=np.float32,
        )
        attn = np.asarray(
            [
                [0.55, 0.40, 0.05, 0.07],
                [0.48, 0.22, 0.18, 0.03],
            ],
            dtype=np.float32,
        )
        n_samples = 24
        concept_activity = np.zeros((len(task_ids), n_samples, len(concept_ids)), dtype=np.float32)
        # task 0: c0/c1 co-activate strongly; c2/c3 co-activate with attention-weighted mass
        concept_activity[0, :12, 0] = 1.0
        concept_activity[0, :10, 1] = 1.0
        concept_activity[0, 8:18, 2] = 0.8
        concept_activity[0, 9:19, 3] = 0.7
        # task 1: weaker but still non-trivial co-activation structure
        concept_activity[1, 3:14, 0] = 1.0
        concept_activity[1, 4:12, 1] = 1.0
        concept_activity[1, 10:20, 2] = 0.6
        concept_activity[1, 11:22, 3] = 0.5

        out = analyzer.analyze_epoch(
            epoch=3,
            tcav_smoothed=tcav,
            attention_support=attn,
            concept_activity_samples=concept_activity,
        )

        self.assertEqual(len(out.task_summaries), len(task_ids))
        self.assertGreater(len(out.edge_rows), 0)
        self.assertEqual(out.mean_flowed_similarity.shape, (len(concept_ids), len(concept_ids)))
        self.assertEqual(
            out.flowed_similarity_by_task.shape,
            (len(task_ids), len(concept_ids), len(concept_ids)),
        )

        flow = out.flowed_similarity_by_task
        self.assertTrue(np.allclose(flow, np.swapaxes(flow, -1, -2), atol=1e-6))
        self.assertTrue(np.allclose(np.diagonal(flow, axis1=1, axis2=2), 0.0, atol=1e-6))

        self.assertTrue(any(abs(float(r.weight_flow) - float(r.weight_raw)) > 1e-8 for r in out.edge_rows))

    def test_detector_emits_ricci_alerts(self) -> None:
        detector = ConceptPressureDetector(
            DetectorConfig(
                runaway_threshold=10.0,
                ricci_negative_edge_fraction_alert=0.25,
                ricci_strong_negative_fraction_alert=0.10,
                ricci_min_curvature_alert=-0.30,
            )
        )
        rho = np.asarray([[0.3, 0.2, 0.1]], dtype=np.float32)
        drift = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        diss = np.asarray([[0.1, 0.1, 0.1]], dtype=np.float32)
        ricci_summary = RicciTaskSummary(
            epoch=2,
            task_id="t0",
            n_nodes=3,
            n_edges=2,
            mean_curvature=-0.12,
            std_curvature=0.05,
            min_curvature=-0.42,
            max_curvature=-0.03,
            negative_edge_fraction=0.5,
            strong_negative_edge_fraction=0.5,
            top_negative_src="c0",
            top_negative_dst="c1",
            top_negative_curvature=-0.42,
        )

        out = detector.detect(
            run_id="r1",
            epoch=2,
            task_ids=["t0"],
            concept_ids=["c0", "c1", "c2"],
            rho=rho,
            drift=drift,
            dissipation=diss,
            blocked_concepts=[],
            ricci_summaries=[ricci_summary],
        )
        codes = {a.code for a in out.alerts}
        self.assertIn("ricci_negative_curvature_surge", codes)
        self.assertIn("ricci_bridge_concentration", codes)
        self.assertIn("ricci_extreme_negative_bridge", codes)


if __name__ == "__main__":
    unittest.main()
