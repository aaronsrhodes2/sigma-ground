"""body_constants and the GM / mass lookups: exact GM, derived mass with G's uncertainty, honest "mass unknown", names that collide, and one source for the old Fact Sheet table.

Ground truth that is not the code under test: the DE440 header values (Earth GM 398600.435507, Moon 4902.800118 km^3/s^2, Park et al. 2021), CODATA's G (6.6743e-11, relative uncertainty 2.2e-5) and the asteroid GMs
printed in the DE440 header constants (Ceres 62.6289, Vesta 17.2882 km^3/s^2).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

G = 6.6743e-11
G_REL = 1.5e-15 / 6.6743e-11


class TestCatalogue(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import body_data
        self.bd = body_data

    def test_earth_gm_is_the_published_de440_value_and_mass_is_gm_over_g(self):                  # BUG
        e, _ = self.bd.resolve("Earth")
        self.assertAlmostEqual(e["gm_km3_s2"], 398600.435507, delta=2e-6)
        m = self.bd.describe(e)["mass_kg"]
        self.assertAlmostEqual(m["value"], 398600.435507e9 / G, delta=1e15)
        self.assertAlmostEqual(m["uncertainty"] / m["value"], G_REL, delta=1e-9)              # the mass inherits G's relative uncertainty, and nothing more

    def test_nicknames_and_numbers_resolve(self):                                                # BY CONSTRUCTION
        for q in ("Terra", "earth", "The Earth"):
            self.assertEqual(self.bd.resolve(q)[0]["name"], "Earth", q)
        for q in ("1", "Ceres", "1 Ceres", "(1) Ceres"):
            self.assertEqual(self.bd.resolve(q)[0]["number"], 1, q)
        self.assertAlmostEqual(self.bd.resolve("Ceres")[0]["gm_km3_s2"], 62.6289, delta=3e-4)
        self.assertAlmostEqual(self.bd.resolve("Vesta")[0]["gm_km3_s2"], 17.2882, delta=3e-4)

    def test_a_planet_or_moon_keeps_its_bare_name_and_the_asteroid_is_reached_by_number(self):    # BUG: 'Io' once returned asteroid 85 Io
        io, _ = self.bd.resolve("Io")
        self.assertEqual(io["kind"], "moon")
        self.assertAlmostEqual(io["gm_km3_s2"], 5959.9155, delta=1e-3)
        self.assertTrue(any("85 Io" in n for n in io["notes"]))
        a85, _ = self.bd.resolve("85 Io")
        self.assertEqual((a85["kind"], a85["number"]), ("asteroid", 85))
        self.assertEqual(self.bd.resolve("Europa")[0]["kind"], "moon")
        self.assertEqual(self.bd.resolve("52 Europa")[0]["kind"], "asteroid")

    def test_a_moon_without_a_published_mass_is_unknown_not_zero(self):                          # BUG: massless test particles must not be reported as massless bodies
        for name in ("Pan", "Janus", "Puck", "Nereid"):
            e, _ = self.bd.resolve(name)
            self.assertNotIn("gm_km3_s2", e, name)
            d = self.bd.describe(e)
            self.assertNotIn("mass_kg", d, name)
            self.assertFalse(d["mass_known"], name)
            self.assertTrue(any("not a claim" in n for n in d["notes"]), name)

    def test_giants_carry_planet_and_system_gm_separately(self):                                  # BUG: the Sun sees the planet plus its moons
        j, _ = self.bd.resolve("Jupiter")
        self.assertGreater(j["system_gm_km3_s2"], j["gm_km3_s2"])
        self.assertAlmostEqual(j["system_gm_km3_s2"], 126712764.1, delta=0.1)
        self.assertAlmostEqual(j["j2"], 14696.514e-6, delta=1e-12)                               # Juno

    def test_kuiper_belt_objects_by_name_and_designation(self):                                  # BY CONSTRUCTION
        for q in ("Eris", "136199 Eris", "2003 UB313"):
            e, _ = self.bd.resolve(q)
            self.assertEqual(e["kind"], "kuiper-belt object", q)
            self.assertAlmostEqual(e["gm_km3_s2"], 1114.69, delta=0.01)

    def test_unknown_names_give_close_matches_not_a_guess(self):                                 # BY CONSTRUCTION: suggest, never fill in
        e, close = self.bd.resolve("Cerez")
        self.assertIsNone(e)
        self.assertIn("ceres", close)

    def test_the_catalogue_covers_56_bodies_343_asteroids_30_kbos(self):                          # BY CONSTRUCTION
        cat = self.bd.catalog()
        names = {e["name"] for e in cat.values() if e["kind"] in ("star", "planet", "moon")}
        self.assertEqual(len(names), 56)
        self.assertEqual(len({e["number"] for e in cat.values() if e["kind"] == "asteroid"}), 343)
        self.assertEqual(len({id(e) for e in cat.values() if e["kind"] == "kuiper-belt object"}), 30)


class TestTools(unittest.TestCase):
    def test_body_constants_tool_returns_values_sources_and_the_g_statement(self):               # BUG
        from sigma_ground.mcp.tools import astronomy
        r = astronomy.body_constants("earth").to_dict()
        self.assertEqual(r["provenance_tag"], "VERIFIED")
        v = r["value"]
        self.assertAlmostEqual(v["gm_km3_s2"], 398600.435507, delta=2e-6)
        self.assertEqual(v["sources"]["gm_km3_s2"], "JPL DE440 header (header.440; Park et al. 2021)")
        self.assertIn("2.2e-5", r["notes"])

    def test_unknown_body_says_so(self):                                                         # BY CONSTRUCTION
        from sigma_ground.mcp.tools import astronomy
        r = astronomy.body_constants("planet x").to_dict()
        self.assertIsNone(r["value"])
        self.assertIn("not in the catalogue", r["notes"])

    def test_lookup_constant_answers_gm_and_mass_names(self):                                    # BUG: 'GM_earth' and 'earth_mass' used to be 'not found'
        from sigma_ground.mcp.tools import constants
        gm = constants.lookup_constant("GM_earth").to_dict()
        self.assertAlmostEqual(gm["value"], 398600.435507e9, delta=1e3)
        self.assertEqual(gm["units"], "m^3/s^2")
        ms = constants.lookup_constant("earth_mass").to_dict()
        self.assertAlmostEqual(ms["value"], 5.9721684e24, delta=5e17)
        self.assertAlmostEqual(ms["uncertainty"] / ms["value"], G_REL, delta=1e-9)
        self.assertAlmostEqual(constants.lookup_constant("mass of the Moon").to_dict()["value"], 7.34579e22, delta=1e17)

    def test_lookup_constant_does_not_shadow_the_curated_constants(self):                        # BY CONSTRUCTION: the body route is tried last
        from sigma_ground.mcp.tools import constants
        self.assertAlmostEqual(constants.lookup_constant("G").to_dict()["value"], 6.6743e-11, delta=1e-16)
        self.assertAlmostEqual(constants.lookup_constant("m_e").to_dict()["value"], 9.109e-31, delta=1e-33)
        self.assertEqual(constants.lookup_constant("M_sun").to_dict()["inputs"]["resolved_via"], "alias")

    def test_a_body_with_no_published_mass_is_not_reported_as_a_missing_constant(self):         # BUG
        from sigma_ground.mcp.tools import constants
        r = constants.lookup_constant("GM_pan").to_dict()
        self.assertIsNone(r["value"])
        self.assertIn("No published mass", r["notes"])

    def test_the_old_fact_sheet_masses_agree_with_gm_over_g_to_their_four_digits(self):          # BUG: the Moon was 5.2e-4 low (7.342e22 against 7.34579e22)
        from sigma_ground.field.interface import body_data
        from sigma_ground.mcp.tools import astronomy
        for key in ("mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"):
            entry, _ = body_data.resolve(key)
            want = body_data.mass_from_gm(entry["gm_km3_s2"])["value"]
            got = astronomy._SOLAR_SYSTEM_BODIES[key]["mass_kg"]
            self.assertLess(abs(got - want) / want, 5e-4, key)                                   # a four-digit table cannot be closer than this


if __name__ == "__main__":
    unittest.main()
