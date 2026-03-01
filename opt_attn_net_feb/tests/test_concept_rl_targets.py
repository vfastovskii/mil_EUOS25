from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

if _HAS_TORCH:
    from opt_attn_net_feb.training.explainability_runtime import (  # type: ignore
        ChemACEBundle,
        FinalExplainabilityConfig,
        build_positive_concept_targets_with_report,
    )


@unittest.skipUnless(_HAS_TORCH, "torch is required for explainability runtime import")
class ConceptRLTargetSelectionTest(unittest.TestCase):
    def test_task_and_overlap_targets_are_selected(self) -> None:
        ids = ["m1", "m2", "m3", "m4", "m5", "m6"]
        # 4-task binary labels (T340, T450, F340450, Fgt480)
        y = np.asarray(
            [
                [1, 0, 0, 0],  # m1
                [1, 1, 0, 0],  # m2 (overlap)
                [0, 1, 0, 0],  # m3
                [0, 0, 1, 1],  # m4 (overlap)
                [0, 0, 0, 0],  # m5
                [1, 0, 1, 0],  # m6 (overlap)
            ],
            dtype=np.float32,
        )

        bundle = ChemACEBundle(
            output_dir=".",
            db_uri="sqlite:///tmp.db",
            run_id="r",
            concept_set_id="cs",
            concept_ids=("c_t0", "c_t1", "c_overlap", "c_noise"),
            concept_metadata={},
            concept_support={},
            concept_mol_map={
                "c_t0": {"m1", "m2", "m6"},
                "c_t1": {"m2", "m3"},
                "c_overlap": {"m2", "m4", "m6"},
                "c_noise": {"m5"},
            },
            concept_conf_map={
                "c_t0": {("m1", "c1"), ("m2", "c1"), ("m6", "c1")},
                "c_t1": {("m2", "c1"), ("m3", "c1")},
                "c_overlap": {("m2", "c1"), ("m4", "c1"), ("m6", "c1")},
                "c_noise": {("m5", "c1")},
            },
        )

        cfg = FinalExplainabilityConfig(
            run_concept_rl=True,
            concept_rl_top_k_per_task=1,
            concept_rl_min_pos_coverage=0.10,
            concept_rl_min_pos_hits=1,
            concept_rl_min_lift=1.0,
            concept_rl_overlap_weight=0.50,
            concept_rl_global_weight=0.20,
            concept_rl_lift_weight=0.10,
            concept_rl_general_top_k=1,
            concept_rl_min_multi_active_count=1,
            concept_rl_require_conf_support=True,
            concept_rl_min_conf_pos_coverage=0.0,
        )

        targets, report = build_positive_concept_targets_with_report(
            config=cfg,
            ids_train=ids,
            y_cls_train=y,
            chem_bundle=bundle,
        )

        # Task 0 should contain its own task concept.
        self.assertIn("c_t0", set(targets.get(0, ())))
        # Task 1 should contain its own task concept.
        self.assertIn("c_t1", set(targets.get(1, ())))
        # A shared overlap concept should be injected as general for at least one task.
        self.assertTrue(any("c_overlap" in set(v) for v in targets.values()))

        self.assertTrue(bool(report.get("enabled", False)))
        self.assertIn("general_active_concepts", report)
        self.assertIn("tasks", report)

    def test_top_k_zero_means_all_passing_concepts(self) -> None:
        ids = ["m1", "m2", "m3", "m4", "m5", "m6"]
        y = np.asarray(
            [
                [1, 0, 0, 0],  # m1
                [1, 1, 0, 0],  # m2
                [0, 1, 0, 0],  # m3
                [0, 0, 1, 1],  # m4
                [0, 0, 0, 0],  # m5
                [1, 0, 1, 0],  # m6
            ],
            dtype=np.float32,
        )
        bundle = ChemACEBundle(
            output_dir=".",
            db_uri="sqlite:///tmp.db",
            run_id="r",
            concept_set_id="cs",
            concept_ids=("c_t0", "c_t1", "c_overlap", "c_noise"),
            concept_metadata={},
            concept_support={},
            concept_mol_map={
                "c_t0": {"m1", "m2", "m6"},
                "c_t1": {"m2", "m3"},
                "c_overlap": {"m2", "m4", "m6"},
                "c_noise": {"m5"},
            },
            concept_conf_map={
                "c_t0": {("m1", "c1"), ("m2", "c1"), ("m6", "c1")},
                "c_t1": {("m2", "c1"), ("m3", "c1")},
                "c_overlap": {("m2", "c1"), ("m4", "c1"), ("m6", "c1")},
                "c_noise": {("m5", "c1")},
            },
        )
        cfg = FinalExplainabilityConfig(
            run_concept_rl=True,
            concept_rl_top_k_per_task=0,  # all-passing mode
            concept_rl_general_top_k=0,
            concept_rl_min_pos_coverage=0.05,
            concept_rl_min_pos_hits=1,
            concept_rl_min_lift=1.0,
            concept_rl_require_conf_support=False,
        )
        targets, report = build_positive_concept_targets_with_report(
            config=cfg,
            ids_train=ids,
            y_cls_train=y,
            chem_bundle=bundle,
        )
        t0 = set(targets.get(0, ()))
        self.assertIn("c_t0", t0)
        self.assertIn("c_overlap", t0)
        self.assertGreaterEqual(len(t0), 2)
        self.assertEqual(report.get("config", {}).get("top_k_per_task_mode"), "all_passing")


if __name__ == "__main__":
    unittest.main()
