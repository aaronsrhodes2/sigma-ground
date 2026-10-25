"""DE440's own constants as one local data file read by both the predictor and Mentat's body lookup.

Ground truth that is not the code under test: the DE440 ASCII header itself (parsed here independently of the generator for the checks that matter) and the published DE440 Earth and Moon GM values
(398600.435507 and 4902.800118 km^3/s^2, Park et al. 2021).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
HEADER = Path("D:/datasets/jpl/de440_ascii_header/header.440")
FILE = ROOT / "sigma_ground/field/interface/fixtures/de440_header_constants.json"


class TestConstantsFile(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import de440_constants
        self.dc = de440_constants

    def test_earth_and_moon_gm_are_the_published_de440_values(self):                           # BUG: GM split from GMB with EMRAT, converted with the header's AU
        self.assertAlmostEqual(self.dc.gm_km3_s2("Earth"), 398600.435507, delta=2e-6)
        self.assertAlmostEqual(self.dc.gm_km3_s2("Moon"), 4902.800118, delta=2e-6)
        self.assertAlmostEqual(self.dc.gm_km3_s2("Earth") + self.dc.gm_km3_s2("Moon"), self.dc.gm_km3_s2("Earth-Moon barycentre"), delta=1e-9)

    def test_the_file_carries_every_header_constant_unchanged(self):                            # BY CONSTRUCTION
        doc = self.dc.load()
        self.assertEqual(len(doc["raw"]), 645)
        self.assertEqual(doc["raw"]["J2E"], 0.00108262539)
        self.assertEqual(doc["raw"]["RE"], 6378.1366)
        self.assertEqual(doc["raw"]["AM"], 1738.0)

    @unittest.skipUnless(HEADER.exists(), "DE440 header not available")
    def test_values_match_a_fresh_parse_of_the_header(self):                                    # BUG
        import make_perturber_catalog as mp
        c = mp.parse_header(HEADER)
        e, m, s = self.dc.body("Earth"), self.dc.body("Moon"), self.dc.body("Sun")
        self.assertEqual((e["j2"], e["j3"], e["j4"], e["reference_radius_km"]), (c["J2E"], c["J3E"], c["J4E"], c["RE"]))
        self.assertEqual((m["j2"], m["j3"], m["j4"], m["reference_radius_km"]), (c["J2M"], c["J3M"], c["J4M"], c["AM"]))
        self.assertEqual((s["j2"], s["reference_radius_km"]), (c["J2SUN"], c["ASUN"]))

    def test_jupiter_and_saturn_come_from_the_cited_papers_and_say_what_was_read(self):          # BY CONSTRUCTION
        doc = self.dc.load()["other_sources"]
        self.assertAlmostEqual(doc["Jupiter"]["j2"], 14696.514e-6, delta=1e-12)
        self.assertIn("NOT printed", doc["Saturn"]["read_2026-10-23"])
        self.assertIn("arXiv:2304.09215", doc["Saturn"]["citation"])


class TestOneSource(unittest.TestCase):
    def test_mentat_returns_the_same_field_constants_as_the_predictor(self):                    # BUG: two body tables used to disagree; they now read one file
        from sigma_ground.field.interface import de440_constants as dc
        from sigma_ground.mcp.tools import astronomy as ast
        for name in ("sun", "earth", "moon", "jupiter", "saturn"):
            v = ast.solar_system_body(name).to_dict()["value"]
            d = dc.body(name)
            self.assertEqual(v["field_reference_radius_km"], d["reference_radius_km"], name)
            self.assertEqual(v["j2"], d["j2"], name)
            self.assertEqual(v["field_constants_source"], d["source"], name)
        self.assertEqual(ast.solar_system_body("earth").to_dict()["value"]["gm_km3_s2"], dc.gm_km3_s2("Earth"))

    def test_mentat_keeps_its_old_fields_and_adds_none_for_bodies_without_a_source(self):       # BY CONSTRUCTION
        from sigma_ground.mcp.tools import astronomy as ast
        v = ast.solar_system_body("mars").to_dict()["value"]
        self.assertNotIn("j2", v)
        self.assertEqual(v["radius_m"], 3.3895e6)
        e = ast.solar_system_body("earth").to_dict()["value"]
        self.assertEqual(e["radius_m"], 6.3781e6)                                                # the Fact Sheet radius is still there, next to field_reference_radius_km

    def test_predictor_label_applies_the_header_and_stored_predictors_keep_the_table(self):     # BUG: old results must stay reproducible
        import numpy as np
        from sigma_ground.field.interface import rolling_shootout as rs
        fix = json.loads((ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics_r4.json").read_text())

        def bodies(name):
            pred = next(p for p in rs.PREDICTORS if p.name == name)
            b, n, _, _ = rs._build_bodies_at_snapshot("j2010", fix, pred)
            return dict(zip(n, b))

        h, z, base = bodies("restated_hdr"), bodies("restated_zon"), bodies("de440_stack")
        self.assertAlmostEqual(h["Earth"].radius_m, 6378136.6, delta=1e-3)
        self.assertAlmostEqual(h["Earth"].j2, 0.00108262539, delta=1e-15)
        self.assertAlmostEqual(h["Moon"].j4, -9.7043900244e-06, delta=1e-15)
        self.assertAlmostEqual(h["Sun"].j2, 2.1961391516529825e-07, delta=1e-18)
        self.assertAlmostEqual(z["Earth"].radius_m, 6371000.0, delta=1e-3)                        # restated_zon and the stored stack are unchanged
        self.assertAlmostEqual(base["Earth"].radius_m, 6371000.0, delta=1e-3)
        self.assertAlmostEqual(base["Sun"].j2, 2.1106e-7, delta=1e-18)
        self.assertTrue(np.isclose(h["Jupiter"].j2, z["Jupiter"].j2))                             # the measured Jupiter and Saturn values are the same in both labels

    def test_measured_zonals_of_restated_zon_agree_with_the_constants_file(self):               # BY CONSTRUCTION: one number, two places, kept equal
        from sigma_ground.field.interface import de440_constants as dc
        from sigma_ground.field.interface import rolling_shootout as rs
        tab = {(n, a): v for n, a, v in rs._MEASURED_ZONALS}
        self.assertEqual(tab[("Jupiter", "j2")], dc.body("Jupiter")["j2"])
        self.assertEqual(tab[("Saturn", "j2")], dc.body("Saturn")["j2"])
        self.assertEqual(tab[("Saturn", "j4")], dc.body("Saturn")["j4"])
        self.assertEqual(tab[("Saturn", "radius_km")], dc.body("Saturn")["reference_radius_km"])

    def test_every_difference_in_the_audit_has_a_stated_reason(self):                           # BY CONSTRUCTION
        import subprocess
        out = ROOT / "sigma_ground/field/interface/fixtures/v2physics_restated/CONSTANTS_AUDIT.json"
        rows = json.loads(out.read_text())
        self.assertTrue(rows)
        self.assertTrue(all(r["class"] in ("convention", "rounding", "vintage", "unresolved") for r in rows), [r for r in rows if r["class"] == "unclassified"])


if __name__ == "__main__":
    unittest.main()
