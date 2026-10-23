"""The restated shoot-out predictors: candidate layers on top of the headline stack, and the DE440 perturber catalogue they read.

Ground truth that does not come from the code under test: the DE440 ASCII header values (Ceres GM 62.6289 km^3/s^2, Vesta 17.2882, the ring's 36 equal points), and the
construction rules (layers off = the stored predictor, bit for bit).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import sigma_ground.field.interface.rolling_shootout as rs

FIX = Path(__file__).parent / "fixtures"
KERNEL = Path("D:/datasets/jpl/de440s.bsp")


class TestCatalogue(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads((FIX / "de440_perturbers_v2physics.json").read_text())

    def test_set_sizes_match_the_de440_paper(self):                                  # BUG: 343 asteroids, 30 KBOs, a ring of 36 equal points (Park et al. 2021 sec. 2.2)
        self.assertEqual(len(self.doc["asteroids"]), 343)
        self.assertEqual(len(self.doc["kbo"]), 30)
        self.assertEqual(self.doc["ring"]["n_points"], 36)
        self.assertTrue(self.doc["ring"]["all_equal"])
        self.assertEqual(self.doc["ring"]["a_au"], 44.0)

    def test_header_masses(self):                                                    # BUG: values read off the DE440 header, converted au^3/day^2 -> km^3/s^2
        gm = {a["number"]: a["gm_km3_s2"] for a in self.doc["asteroids"]}
        self.assertAlmostEqual(gm[1], 62.6289, delta=2e-4)                          # Ceres
        self.assertAlmostEqual(gm[4], 17.2882, delta=2e-4)                          # Vesta
        self.assertAlmostEqual(gm[2], 13.6659, delta=2e-4)                          # Pallas

    def test_kbo_placement_follows_the_one_percent_unique_match_rule(self):         # BUG: names are never guessed; an ambiguous or absent mass match stays unplaced
        placed = {k["id"]: k for k in self.doc["kbo"] if k["elements"]}
        self.assertEqual(sorted(placed), [8001, 8002, 8003, 8007, 8009, 8012, 8018])    # Eris, Haumea, Quaoar, Orcus, 2002 UX25, Varda, 2003 AZ84
        for k in placed.values():
            self.assertLess(abs(k["match_pct"]), 1.0)
            self.assertAlmostEqual(k["gm_km3_s2"] / k["paper_gm_km3_s2"] - 1.0, k["match_pct"] / 100.0, delta=1e-4)
        self.assertEqual(sorted(self.doc["metadata"]["kbo_unplaced_named"]), ["Gonggong", "Makemake", "Salacia"])
        self.assertIn("Eris", placed[8001]["name"])

    def test_missing_elements_are_reported_not_hidden(self):
        missing = self.doc["metadata"]["asteroids_without_elements"]
        self.assertEqual(sorted(missing), sorted(a["number"] for a in self.doc["asteroids"] if not a["elements"]))   # the report is exactly the gap, however large (ef3211d closed it: 0 now)


class TestPredictors(unittest.TestCase):
    def test_stored_predictors_are_unchanged(self):                                  # BY CONSTRUCTION: the new fields default to "off"
        for p in rs.PREDICTORS:
            if not p.name.startswith("restated_"):
                self.assertEqual(p.lunar_c22, 0.0)
                self.assertEqual(p.perturber_sets, ())

    def test_restated_predictors_differ_from_the_headline_only_in_the_layers(self):  # BUG
        import dataclasses
        base = [p for p in rs.canonical_predictors(0.025) if p.name == "de440_stack"][0]
        for p in rs.restated_predictors(0.025):
            same = dataclasses.replace(p, name=base.name, description=base.description, lunar_c22=0.0, perturber_sets=(), zonal_overrides=())
            self.assertEqual(same, base)
        by = {p.name: p for p in rs.restated_predictors(0.025)}
        self.assertEqual(by["restated_ast_ring_c22"].perturber_sets, ("asteroids", "ring"))
        self.assertEqual(by["restated_c22"].perturber_sets, ())
        self.assertEqual(by["restated_zon"].perturber_sets, ("asteroids", "ring", "kbo"))              # the new label is the KBO candidate plus measured zonals, nothing else
        self.assertEqual(by["restated_ast_ring_kbo_c22"].zonal_overrides, ())

    @unittest.skipUnless(KERNEL.exists(), "DE440s kernel not available")
    def test_layers_are_built_for_the_right_bodies(self):                            # BUG
        fx = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())
        pred = [p for p in rs.restated_predictors(0.025) if p.name == "restated_ast_ring_c22"][0]
        bodies, names, sun_idx, parent = rs._build_bodies_at_snapshot("j2010", fx, pred)
        out_bodies, toggles, kw = rs._candidate_layers(pred, bodies, names, parent, fx["snapshots"]["j2010"]["epoch"]["jd_tdb"], 40.0)
        self.assertTrue(toggles.sectoral_c22 and toggles.prescribed_sources)
        for i, b in enumerate(out_bodies):
            self.assertEqual(b.c22 != 0.0, names[i] == "Moon")                      # only the Moon carries a figure
        src = kw["prescribed_sources"]
        doc = json.loads((FIX / "de440_perturbers_v2physics.json").read_text())
        self.assertEqual(src.gm.shape[0], sum(1 for a in doc["asteroids"] if a["elements"]) + 36)
        self.assertGreaterEqual(src.t_end, 40.0 * 86400.0)                          # the table covers the horizon
        self.assertEqual(sorted(kw["rotation_models"]), [names.index("Moon")])
        self.assertAlmostEqual(float(src.gm[:-36].sum()) / 1e9, sum(a["gm_km3_s2"] for a in doc["asteroids"] if a["elements"]), delta=1e-6)


class TestZonalOverrides(unittest.TestCase):
    """C12: measured zonals replace the table's values only for the new label; every stored predictor keeps the table's."""

    def _bodies(self, name):
        pred = next(p for p in rs.PREDICTORS if p.name == name)
        fix = json.loads((FIX / "de440_state_vectors_v2physics_r4.json").read_text())
        bodies, names, _, _ = rs._build_bodies_at_snapshot("j2010", fix, pred)
        return {n: b for n, b in zip(names, bodies)}

    def test_new_label_carries_the_measured_values(self):                           # BUG: Jupiter J2 14696.514e-6 (Juno), Saturn J2 16290.573e-6 and J4 -935.314e-6 on 60330 km
        b = self._bodies("restated_zon")
        self.assertAlmostEqual(b["Jupiter"].j2, 14696.514e-6, delta=1e-12)
        self.assertAlmostEqual(b["Saturn"].j2, 16290.573e-6, delta=1e-12)
        self.assertAlmostEqual(b["Saturn"].j4, -935.314e-6, delta=1e-12)
        self.assertAlmostEqual(b["Saturn"].radius_m, 60330e3, delta=1.0)
        self.assertAlmostEqual(b["Uranus"].j2, 3.5107e-3, delta=1e-12)               # untouched

    def test_stored_predictors_keep_the_table(self):                                 # BY CONSTRUCTION: the old results stay reproducible
        b = self._bodies("restated_ast_ring_kbo_c22")
        self.assertAlmostEqual(b["Jupiter"].j2, 1.4736e-2, delta=1e-12)
        self.assertAlmostEqual(b["Saturn"].j2, 1.6298e-2, delta=1e-12)
        self.assertAlmostEqual(b["Saturn"].radius_m, 60268e3, delta=1.0)

    def test_the_measured_j2_r_squared_matches_the_equivalent_value_on_60268_km(self):   # BUG: J2 R^2 is what the force reads; 16290.573e-6 on 60330 km == 16324.108e-6 on 60268 km
        self.assertAlmostEqual(16290.573e-6 * 60330.0 ** 2, 16324.108e-6 * 60268.0 ** 2, delta=2e-5 * 16324.108e-6 * 60268.0 ** 2)


if __name__ == "__main__":
    unittest.main()
