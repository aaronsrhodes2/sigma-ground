"""small_body: orbit, physical data and non-gravitational terms of asteroids, Kuiper-belt objects and comets.

Ground truth that is not the code under test: the values printed in the Horizons header of 10P/Tempel 2 (read off the response text), the period and perihelion distance Horizons itself gives for each of the 20
comets (compared with what we CALCULATE from a, e and the Sun's GM), and textbook periods (Ceres 4.60 yr, Halley 75.9 yr, Eris about 559 yr).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestComets(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import small_body_data
        self.sb = small_body_data

    def test_twenty_comets_with_their_solutions(self):                                          # BY CONSTRUCTION
        self.assertEqual(len(self.sb.comets()), 20)

    def test_10p_matches_the_horizons_header_text(self):                                        # BUG if a number were mis-parsed (EC= .5375 has no leading zero)
        e, _ = self.sb.resolve("10P")
        d = self.sb.describe(e)
        self.assertEqual(d["name"], "10P/Tempel 2")
        self.assertAlmostEqual(d["orbit"]["eccentricity"], 0.53753449495463, delta=1e-14)
        self.assertAlmostEqual(d["orbit"]["perihelion_distance_au_given"], 1.416518875712098, delta=1e-12)
        self.assertAlmostEqual(d["nongravitational"]["A1_au_per_day2"], 2.55888774991e-10, delta=1e-20)
        self.assertEqual(d["nongravitational"]["DT_days"], 0.0)
        self.assertEqual(d["sources"]["orbit"].count("7070"), 1)                                # the observation count is carried into the source

    def test_calculated_period_and_perihelion_equal_what_horizons_gives_for_every_comet(self):   # BUG if the Sun's GM or the year length were wrong
        for des, p in self.sb.comets().items():
            d = self.sb.describe(self.sb.resolve(des)[0])["orbit"]
            self.assertAlmostEqual(d["calculated"]["period_years"] / d["period_years_given"], 1.0, delta=1e-4, msg=des)
            self.assertAlmostEqual(d["calculated"]["perihelion_au"] / d["perihelion_distance_au_given"], 1.0, delta=1e-9, msg=des)

    def test_names_and_designations_reach_the_same_comet(self):                                 # BY CONSTRUCTION
        ref = self.sb.resolve("1P")[0]
        for q in ("Halley", "Comet Halley", "1P/Halley", "halley"):
            self.assertIs(self.sb.resolve(q)[0], ref, q)
        self.assertIs(self.sb.resolve("67P")[0], self.sb.resolve("Churyumov-Gerasimenko")[0])

    def test_a_comet_without_a_published_mass_says_so(self):                                    # BUG: GM = n.a. must not become zero
        d = self.sb.describe(self.sb.resolve("1P")[0])
        self.assertFalse(d["mass_known"])
        self.assertTrue(any("No published mass" in n for n in d["notes"]))
        self.assertNotIn("mass_kg", d)


class TestAsteroidsAndKuiperBelt(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import small_body_data
        self.sb = small_body_data

    def test_period_is_calculated_from_the_elements_and_the_suns_gm(self):                      # BUG if Kepler's third law were applied with the wrong GM
        ceres = self.sb.describe(self.sb.resolve("Ceres")[0])
        self.assertAlmostEqual(ceres["orbit"]["calculated"]["period_years"], 4.60, delta=0.02)
        eris = self.sb.describe(self.sb.resolve("Eris")[0])
        self.assertAlmostEqual(eris["orbit"]["calculated"]["period_years"], 559.0, delta=3.0)
        self.assertAlmostEqual(eris["orbit"]["calculated"]["perihelion_au"], 38.2, delta=0.3)

    def test_derived_orbit_is_a_calculation_not_a_stored_value(self):                           # BY CONSTRUCTION: no 'period' is stored in the catalogue or the fixtures
        import json
        pc = json.loads((ROOT / "sigma_ground/field/interface/fixtures/de440_perturbers_v2physics.json").read_text())
        self.assertNotIn("period_years", json.dumps(pc["asteroids"][0]))
        self.assertAlmostEqual(self.sb.derived_orbit(1.0, 0.0)["period_days"], 365.25, delta=0.02)   # a = 1 au is one Julian year (to the Earth-Moon mass 3e-6)

    def test_physical_data_of_a_heavy_asteroid_comes_with_its_source(self):                     # BY CONSTRUCTION
        v = self.sb.describe(self.sb.resolve("4 Vesta")[0])
        self.assertAlmostEqual(v["physical"]["radius_km"], 261.385, delta=0.01)
        self.assertEqual(v["sources"]["radius_km"], "JPL Horizons asteroid physical parameters (header)")
        self.assertAlmostEqual(v["gm_km3_s2"], 17.2882, delta=3e-4)

    def test_planets_moons_and_unknown_names_are_not_small_bodies(self):                        # BY CONSTRUCTION
        for q in ("Earth", "Pan", "Io"):
            e, _ = self.sb.resolve(q)
            self.assertTrue(e is None or e["kind"] in ("asteroid", "comet", "kuiper-belt object"), q)
        self.assertIsNone(self.sb.resolve("Earth")[0])


class TestTool(unittest.TestCase):
    def test_tool_returns_orbit_and_refuses_with_a_pointer(self):
        from sigma_ground.mcp.tools import astronomy
        r = astronomy.small_body("halley").to_dict()
        self.assertEqual(r["provenance_tag"], "VERIFIED")
        self.assertAlmostEqual(r["value"]["orbit"]["calculated"]["period_years"], 75.9, delta=0.1)
        self.assertIn("A1_au_per_day2", r["value"]["nongravitational"])
        r = astronomy.small_body("earth").to_dict()
        self.assertIsNone(r["value"])
        self.assertIn("body_constants", r["notes"])


if __name__ == "__main__":
    unittest.main()
