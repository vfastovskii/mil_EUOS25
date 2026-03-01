from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from opt_attn_net_feb.explainability.chem_ace.config import SemanticTaggingConfig
from opt_attn_net_feb.explainability.chem_ace.semantics.taggers import SemanticTagger


class ChemACESemanticsTest(unittest.TestCase):
    def _make_tagger(self) -> SemanticTagger:
        return SemanticTagger(
            SemanticTaggingConfig(
                use_openbabel_descriptors=False,
            )
        )

    def test_geom_family_summary_from_feature_names(self) -> None:
        tagger = self._make_tagger()
        geom_vectors = [
            np.asarray([1.20, -0.80, 0.90, 1.10], dtype=np.float32),
            np.asarray([1.00, -0.60, 0.70, 1.30], dtype=np.float32),
            np.asarray([0.90, -0.50, 0.60, 1.20], dtype=np.float32),
        ]
        geom_names = [
            "torsion_phi",
            "shape_asphericity",
            "planarity_score",
            "ring_strain_energy",
        ]
        summary = tagger._summarize_geom_vectors(
            geom_vectors=geom_vectors,
            geom_feature_names=geom_names,
        )
        self.assertEqual(int(summary["n_vectors"]), 3)
        self.assertEqual(int(summary["n_features"]), 4)
        family_stats = summary["family_stats"]
        self.assertIn("dihedral", family_stats)
        self.assertIn("shape", family_stats)
        self.assertIn("planarity", family_stats)
        self.assertIn("ring_strain", family_stats)

    def test_geometry_descriptor_tags_emit(self) -> None:
        tagger = self._make_tagger()
        descriptors = {
            "planarity_rmsd_mean": 0.18,
            "rotatable_bond_count_mean": 0.25,
            "geom_summary": {
                "n_vectors": 24,
                "family_stats": {
                    "dihedral": {"abs_z_mean": 1.25, "z_mean": 1.10},
                    "planarity": {"abs_z_mean": 1.40, "z_mean": 1.05},
                    "shape": {"abs_z_mean": 0.90, "z_mean": 0.70},
                    "ring_strain": {"abs_z_mean": 0.75, "z_mean": 0.60},
                    "surface_volume": {"abs_z_mean": 0.72, "z_mean": 0.51},
                },
            },
        }
        tags = tagger._geometry_tags(concept_id="c0", descriptors=descriptors)
        names = {t.tag for t in tags}
        self.assertIn("planar", names)
        self.assertIn("rigid", names)
        self.assertIn("torsionally active geometry", names)
        self.assertIn("planarity-enriched geometry", names)
        self.assertIn("shape-anisotropic geometry", names)
        self.assertIn("ring-strained geometry", names)
        self.assertIn("surface/volume-driven geometry", names)

    def test_cross_modal_tags_emit(self) -> None:
        tagger = self._make_tagger()
        descriptors = {
            "aromatic_atom_fraction_mean": 0.55,
            "pharmacophore_counts": {"Donor": 2, "Acceptor": 2},
            "smarts_rx_role_rate": {"electrophile": 0.08, "nucleophile": 0.05},
            "geom_summary": {
                "n_vectors": 30,
                "family_stats": {"planarity": {"abs_z_mean": 0.9, "z_mean": 0.7}},
            },
            "qm_summary": {
                "n_vectors": 30,
                "family_stats": {
                    "electrophilicity": {"abs_z_mean": 1.2, "z_mean": 1.1},
                    "nucleophilicity": {"abs_z_mean": 1.1, "z_mean": 1.0},
                    "gap": {"abs_z_mean": 0.8, "z_mean": 0.7},
                    "dipole": {"abs_z_mean": 0.9, "z_mean": 0.8},
                },
            },
        }
        tags = tagger._cross_modal_tags(concept_id="c0", descriptors=descriptors)
        names = {t.tag for t in tags}
        self.assertIn("electrophilic reaction-center motif", names)
        self.assertIn("nucleophilic reaction-center motif", names)
        self.assertIn("planar conjugated electronic motif", names)
        self.assertIn("polar donor-acceptor electronic motif", names)


if __name__ == "__main__":
    unittest.main()
