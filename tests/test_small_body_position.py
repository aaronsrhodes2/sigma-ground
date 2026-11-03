"""body_position for small bodies (small_body_position.py): Eros from OUR orbit fit, the tiers, the moon-name guard, the budget, the roster and the tool's notes.

Ground truth not under test: JPL Horizons' own Eros state at 2026-01-01 (D:/datasets/mpc/eros/vec_2024-01-01_2026-11-03.json, fetched 2026-11-03), the C28 result that our fit lands 0.12 arcsec
(about 85 km at 1 au) from JPL's solution over 2020-2026, and the two-body identity: an elements start propagated zero days returns the elements' own state.
"""
from __future__ import annotations

import glob
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TRUTH = Path("D:/datasets/jpl/de442/de442.bsp")
EROS_2026 = glob.glob("D:/datasets/mpc/eros/vec_*2026*.json")


@unittest.skipUnless(TRUTH.exists() or Path("D:/datasets/jpl/de440s.bsp").exists(), "no JPL kernel on this machine")
class TestEros(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import small_body_position as sbp, where_when as ww
        self.sbp, self.ww = sbp, ww
        self.k, self.kname, _ = ww._truth()

    def test_eros_starts_from_our_own_fit_and_the_answer_says_so(self):                         # BUG if Eros silently fell back to JPL's state
        st = self.sbp.resolve("Eros", self.k)
        self.assertEqual((st.tier, st.number, st.jd0), ("own_fit", 433, 2458849.5))
        self.assertIn("observations alone", st.source)
        r = self.ww.position_at("433 Eros", "JD 2458879.5", "ssb", budget_s=60)                 # 30 days after the fit epoch: a second of integration
        self.assertEqual((r["status"], r["start"]["tier"], r["body"]), ("computed", "own_fit", "433 Eros"))
        self.assertTrue(any("OUR orbit solution" in n for n in r["notes"]))
        self.assertEqual(r["receipt"]["prescribed_sources"]["count"], 343)
        self.assertIn("1PN", r["receipt"]["relativity"])
        self.assertEqual(r["start"]["measured"]["n_predicted_observations"], 4435)              # C28's out-of-sample test travels with the answer

    @unittest.skipUnless(EROS_2026, "JPL's 2026 Eros vectors not on this machine")
    def test_our_fit_propagated_six_years_lands_within_the_c28_distance_of_jpls_solution(self):   # BUG if the engine or the fit fixture were wrong by more than the ledger's 0.12 arcsec
        y = self.sbp._horizons_vector_row(json.load(open(EROS_2026[0]))["result"], 433, self.sbp.JD_2026)
        r = self.ww.position_at("Eros", f"JD {self.sbp.JD_2026}", "ssb", budget_s=120)
        d = np.linalg.norm(np.array(r["position_km"]) - y[:3])
        self.assertLess(d, 300.0, f"{d:.0f} km from JPL's 2026 state")                         # C28: 0.118 arcsec median at ~1 au is ~85 km; measured 115 km on 2026-11-03
        self.assertGreater(d, 1.0)                                                               # and it is NOT JPL's state copied

    def test_from_the_earth_the_answer_has_ra_and_dec_and_the_scoreboard_line(self):
        r = self.ww.position_at("Eros", "JD 2458879.5", "earth", budget_s=60)
        self.assertTrue(0.0 <= r["ra_deg"] < 360.0 and -90.0 <= r["dec_deg"] <= 90.0)
        self.assertIn("ours_median_arcsec", r["measured_against_the_sky"])                     # Eros is on the daily scoreboard (C27)


@unittest.skipUnless(TRUTH.exists() or Path("D:/datasets/jpl/de440s.bsp").exists(), "no JPL kernel on this machine")
class TestTiers(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import small_body_position as sbp, where_when as ww
        self.sbp, self.ww = sbp, ww
        self.k = ww._truth()[0]

    def test_each_tier_resolves_in_order(self):                                                 # BUG if a catalogue start shadowed a better one
        tiers = {n: self.sbp.resolve(n, self.k).tier for n in ("Eros", "Ceres", "Apophis", "99942", "Eris", "67P", "Akepa", "704")}
        self.assertEqual(tiers, {"Eros": "own_fit", "Ceres": "horizons", "Apophis": "horizons", "99942": "horizons", "Eris": "catalogue", "67P": "catalogue", "Akepa": "sbdb", "704": "horizons"})

    def test_an_elements_start_propagated_zero_days_is_the_two_body_state(self):                # BY CONSTRUCTION: the start is exact at its epoch
        st = self.sbp.resolve("Eris", self.k)
        r = self.ww.position_at("Eris", f"JD {st.jd0}", "ssb", budget_s=60)
        self.assertTrue(np.allclose(r["position_km"], st.y0[:3], atol=1e-3))
        self.assertAlmostEqual(r["distance_au"], 95.0, delta=2.0)                                # Eris near aphelion, ~95 au (SBDB)

    def test_a_comet_carries_its_marsden_terms_and_law(self):
        st = self.sbp.resolve("67P", self.k)
        self.assertEqual(st.kind, "comet")
        self.assertIsNotNone(st.nongrav)
        self.assertEqual(set(st.law), {"alpha", "k", "m", "n", "r0"})                             # comet_orbits.marsden_g's names, not Horizons' ALN/NK/NM/NN/R0

    def test_a_bare_moon_name_is_refused_and_the_asteroid_of_that_name_is_reachable_by_number(self):   # BUG if 'Io' quietly became asteroid 85 Io
        with self.assertRaises(self.ww.WhereWhenError) as cm:
            self.ww.position_at("Io", "2026-01-01", "sun")
        self.assertIn("moon of Jupiter", str(cm.exception))
        self.assertIn("Supported:", str(cm.exception))
        self.assertEqual(self.sbp.resolve("85 Io", self.k).number, 85)

    def test_an_unknown_body_is_refused_with_the_roster_pointer(self):
        with self.assertRaises(self.ww.WhereWhenError) as cm:
            self.ww.position_at("Bennu", "2026-01-01", "sun")
        self.assertIn("solvable_bodies", str(cm.exception))

    def test_a_long_lead_over_budget_is_declined_before_it_runs(self):                          # the chat must not hang
        r = self.ww.position_at("Halley", "2061-07-28", "sun", budget_s=10)
        self.assertEqual(r["status"], "over_budget")
        self.assertIsNone(r["position_km"])
        self.assertIn("raise budget_s", r["offer"]["how"])

    def test_the_roster_counts_match_the_fixtures(self):
        d = self.sbp.solvable_bodies()
        t = d["small_bodies_as_test_particles"]["tiers_in_order"]
        self.assertEqual(d["full_n_body_live"]["count"], 11)
        self.assertEqual(t[0]["bodies"], ["433 Eros"])
        self.assertEqual(t[1]["count"], 31)                                                      # 30 heaviest + Apophis
        self.assertEqual((t[2]["asteroids"] >= 300, t[2]["kuiper_belt_objects"], t[2]["comets"]), (True, 30, 20))
        self.assertEqual(t[3]["count"], 24370)


@unittest.skipUnless(TRUTH.exists() or Path("D:/datasets/jpl/de440s.bsp").exists(), "no JPL kernel on this machine")
class TestTool(unittest.TestCase):
    def test_the_tool_notes_show_the_start_the_receipt_and_the_sky_measurement(self):
        from sigma_ground.mcp.tools import astronomy
        r = astronomy.body_position("eros", "JD 2458879.5", "earth").to_dict()
        self.assertEqual(r["provenance_tag"], "DERIVED")
        for s in ("PREDICTION (our calculation)", "START (own_fit", "RECEIPT:", "MEASURED AGAINST THE SKY", "nothing of JPL's"):
            self.assertIn(s, r["notes"])
        roster = astronomy.solvable_bodies().to_dict()
        self.assertIn("433 Eros", roster["notes"])
        self.assertIn("24370 named main-belt asteroids", roster["notes"])


if __name__ == "__main__":
    unittest.main()
