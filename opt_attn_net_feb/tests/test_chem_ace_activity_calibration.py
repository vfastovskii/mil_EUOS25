from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from ..utils.constants import TASK_COLS
from ..explainability.chem_ace.types import TagAssignment
from ..explainability.chem_ace.semantics.calibration import (
    ActivityAwareSemanticCalibrator,
    ActivityCalibrationConfig,
)
from ..explainability.chem_ace.semantics.taggers import SemanticTaggingResult


class ChemACEActivityCalibrationTest(unittest.TestCase):
    def test_activity_enriched_concept_gets_higher_confidence(self) -> None:
        ids = [f"m{i}" for i in range(100)]
        y = np.zeros((100, len(TASK_COLS)), dtype=np.int64)
        y[:20, 0] = 1  # task-0 actives
        y[40:55, 2] = 1  # another task, unrelated to c_neg

        concept_map = {
            "c_pos": set(ids[:20]),
            "c_neg": set(ids[20:40]),
        }
        tagging = [
            SemanticTaggingResult(
                concept_id="c_pos",
                label_auto="positive-like",
                tags=[
                    TagAssignment(
                        concept_id="c_pos",
                        tag="tag_pos",
                        confidence=0.60,
                        provenance="rule",
                        evidence_json={},
                    )
                ],
                evidence_json={},
            ),
            SemanticTaggingResult(
                concept_id="c_neg",
                label_auto="negative-like",
                tags=[
                    TagAssignment(
                        concept_id="c_neg",
                        tag="tag_neg",
                        confidence=0.60,
                        provenance="rule",
                        evidence_json={},
                    )
                ],
                evidence_json={},
            ),
        ]

        calibrator = ActivityAwareSemanticCalibrator(
            config=ActivityCalibrationConfig(
                enabled=True,
                min_concept_support=5,
                min_tag_support=5,
                prior_strength=4.0,
                mix_base=0.8,
                keep_threshold=0.5,
                ratio_cap=6.0,
            ),
            task_cols=TASK_COLS,
        )
        calibrated, rows, summary = calibrator.calibrate(
            tagging_results=tagging,
            concept_mol_map_train=concept_map,
            ids_train=ids,
            y_train=y,
        )

        self.assertEqual(len(calibrated), 2)
        self.assertEqual(int(summary["n_concepts"]), 2)
        rows_by_concept = {str(r.concept_id): r for r in rows}
        self.assertIn("c_pos", rows_by_concept)
        self.assertIn("c_neg", rows_by_concept)
        self.assertGreater(
            float(rows_by_concept["c_pos"].calibrated_confidence),
            float(rows_by_concept["c_neg"].calibrated_confidence),
        )

    def test_fallback_top1_keeps_one_tag_when_all_filtered(self) -> None:
        ids = [f"m{i}" for i in range(30)]
        y = np.zeros((30, len(TASK_COLS)), dtype=np.int64)
        concept_map = {"c0": set(ids[:5])}
        tagging = [
            SemanticTaggingResult(
                concept_id="c0",
                label_auto="c0",
                tags=[
                    TagAssignment(
                        concept_id="c0",
                        tag="weak_tag",
                        confidence=0.10,
                        provenance="rule",
                        evidence_json={},
                    )
                ],
                evidence_json={},
            )
        ]

        calibrator = ActivityAwareSemanticCalibrator(
            config=ActivityCalibrationConfig(
                enabled=True,
                keep_threshold=0.99,
                fallback_top1_if_empty=True,
                min_concept_support=5,
                min_tag_support=5,
                prior_strength=8.0,
            ),
            task_cols=TASK_COLS,
        )
        calibrated, rows, _summary = calibrator.calibrate(
            tagging_results=tagging,
            concept_mol_map_train=concept_map,
            ids_train=ids,
            y_train=y,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(calibrated), 1)
        self.assertEqual(len(calibrated[0].tags), 1)
        self.assertEqual(str(calibrated[0].tags[0].tag), "weak_tag")


if __name__ == "__main__":
    unittest.main()

