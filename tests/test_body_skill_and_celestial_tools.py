"""body_skill (the measured-skill table) and the celestial tools' answer shape.

The table is a cache of runs against JPL DE440; these tests pin it to the ledger (C14, C16, C17, C19), check that it rebuilds from its own sources, and that the class rule cannot be loosened without a test noticing."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "sigma_ground/field/interface/fixtures"

# misc/CANDIDATE_LEDGER.md C14: "Not supported: Pan, Atlas, Prometheus, Pandora, Janus, Epimetheus, Juliet, Portia, Belinda, Puck, Despina, Galatea, Larissa and Proteus lose their orbital phase within a year"
C14_LOSE_PHASE = {"Pan", "Atlas", "Prometheus", "Pandora", "Janus", "Epimetheus", "Juliet", "Portia", "Belinda", "Puck", "Despina", "Galatea", "Larissa", "Proteus"}


def _table() -> dict:
    return json.loads((FIX / "body_skill_table.json").read_text(encoding="utf-8"))["bodies"]


class TestTable(unittest.TestCase):
    def test_the_class_rule_reproduces_the_ledgers_list_of_moons_that_lose_phase(self):
        got = {n for n, b in _table().items() if b["class"] == "loses_orbital_phase"}
        self.assertEqual(got, C14_LOSE_PHASE)

    def test_the_two_groups_are_separated_by_two_orders_of_magnitude_so_the_threshold_is_not_a_fit(self):
        lost = [b["converged"]["error_at_1yr_over_orbital_radius"] for b in _table().values() if b["class"] == "loses_orbital_phase"]
        kept = [b["converged"]["error_at_1yr_over_orbital_radius"] for b in _table().values() if "converged" in b and b["class"] != "loses_orbital_phase"]
        self.assertGreater(min(lost) / max(kept), 50)
        self.assertGreater(min(lost), 0.1)
        self.assertLess(max(kept), 0.1)

    def test_numbers_equal_the_ledger(self):
        t = _table()
        self.assertAlmostEqual(t["Jupiter"]["forecast_step_0.025d"]["5"]["median_km"], 5.79, delta=0.01)       # C16 "Jupiter 5.54 -> 5.79 km (+5 yr)"
        self.assertAlmostEqual(t["Callisto"]["forecast_step_0.025d"]["5"]["median_km"], 31.9, delta=0.1)       # C16 "Callisto 210 -> 31.9 km"
        self.assertAlmostEqual(t["Mercury"]["hindcast"]["5"]["median_km"], 0.0252, delta=0.0005)                # C17 "Mercury +5 0.025"
        self.assertAlmostEqual(t["Moon"]["hindcast"]["5"]["median_km"], 0.375, delta=0.005)                      # C17 "Moon ... +5 0.38"
        self.assertAlmostEqual(t["Io"]["converged"]["error_km"]["5"], 159, delta=1)                              # C14 "Io 159"
        self.assertAlmostEqual(t["Enceladus"]["converged"]["error_km"]["5"], 358, delta=1)                       # C14 "Enceladus 358"
        self.assertEqual(len(t), 55)

    def test_the_three_outer_moons_whose_two_measurements_disagree_are_flagged_not_averaged(self):
        t = _table()
        for n in ("Nereid", "Himalia", "Phoebe"):
            self.assertEqual(t[n]["class"], "measurements_disagree", n)
            self.assertIsNone(t[n]["best_estimate_5yr_km"])
            self.assertGreater(t[n]["converged"]["converged_over_full_system_at_5yr"], 10)
        for n in ("Titan", "Callisto", "Hyperion", "Iapetus", "Titania", "Oberon", "Charon"):
            self.assertEqual(t[n]["class"], "predictable", n)                                                      # the others agree within a factor of 2

    def test_no_body_with_no_position_claim_carries_a_best_estimate(self):
        for n, b in _table().items():
            if b["class"] in ("loses_orbital_phase", "measurements_disagree", "not_scored_at_a_converged_step"):
                self.assertIsNone(b.get("best_estimate_5yr_km"), n)
            else:
                self.assertGreater(b["best_estimate_5yr_km"], 0, n)

    def test_it_rebuilds_from_its_sources(self):
        r = subprocess.run([sys.executable, "-X", "utf8", str(REPO / "scripts/make_body_skill_table.py"), "--check"], capture_output=True, text=True, cwd=REPO)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_the_sources_still_match_the_hashes_it_was_built_from(self):
        from sigma_ground.field.interface import skill
        self.assertEqual(skill.stale_sources(), [])


class TestTools(unittest.TestCase):
    def test_body_skill_words_each_class_without_a_claim_it_cannot_back(self):
        from sigma_ground.mcp.tools import celestial_tools as T
        pan = T.body_skill("pan").to_dict()
        self.assertEqual(pan["value"]["class"], "loses_orbital_phase")
        self.assertIn("give no position", pan["value"]["summary"])
        nereid = T.body_skill("nereid").to_dict()
        self.assertIn("Neither number is offered as the skill", nereid["value"]["summary"])
        mercury = T.body_skill("mercury").to_dict()
        self.assertIn("integration error cancelling", mercury["value"]["summary"])                                # the C19 caveat travels with the number
        io = T.body_skill("io").to_dict()
        self.assertIn("converged step", io["value"]["summary"])
        self.assertIsNone(io["value"]["table_sources_changed_since_build"])

    def test_names_resolve_through_the_aliases_the_other_tools_accept(self):
        from sigma_ground.mcp.tools import celestial_tools as T
        self.assertEqual(T.body_skill("Luna").to_dict()["value"]["body"], "Moon")

    def test_unknown_and_unscored_bodies_are_declined_with_the_reason(self):
        from sigma_ground.mcp.tools import celestial_tools as T
        for fn, args in ((T.body_skill, ("ceres",)), (T.orbital_elements, ("ceres",)), (T.hill_radius, ("pluto-charon-xyz",)), (T.roche_limit, ("sun", "sun"))):
            r = fn(*args).to_dict()
            self.assertIsNone(r["value"], args)
            self.assertTrue(r["notes"])

    def test_questions_the_data_cannot_answer_are_declined_not_crashed(self):                  # BUG if any of these raised or returned a number (roche_limit('sun','sun') once returned one)
        from sigma_ground.mcp.tools import celestial_tools as T
        cases = [(T.hill_radius, ("sun", "earth")), (T.sphere_of_influence, ("sun", "earth")), (T.lagrange_points, ("sun", "earth")), (T.synodic_period, ("mars", "mars")),
                 (T.hohmann_transfer, ("earth", "earth")), (T.step_check, ("mercury", 100.0)), (T.roche_limit, ("earth", "moon")), (T.roche_limit, ("sun", "sun")),
                 (T.orbital_elements, ("earth", "earth")), (T.orbital_elements, ("mars", "sun", "not a date")), (T.orbital_elements, ("mars", "sun", "1700-01-01")),
                 (T.orbital_elements, ("mars", "sun", "2300-01-01")), (T.orbital_elements, ("io", "jupiter", "1900-01-01"))]
        for fn, args in cases:
            r = fn(*args).to_dict()
            self.assertIsNone(r["value"], (fn.__name__, args))
            self.assertTrue(r["notes"], (fn.__name__, args))

    def test_every_tool_answer_is_json_and_names_its_source(self):
        from sigma_ground.mcp.tools import celestial_tools as T
        for fn, args in ((T.orbital_elements, ("io", "jupiter")), (T.roche_limit, ("moon", "earth")), (T.step_check, ("jupiter", 1080.0)), (T.body_skill, ("io",)), (T.lagrange_points, ("io", "jupiter"))):
            r = fn(*args).to_dict()
            json.dumps(r)
            self.assertTrue(r["source"], fn.__name__)
            self.assertIsNotNone(r["value"], fn.__name__)

    def test_step_check_reproduces_the_c19_counts_and_names_the_step_that_resolves_all(self):
        from sigma_ground.mcp.tools import celestial_tools as T
        r = T.step_check("jupiter", 0.0125 * 86400).to_dict()["value"]
        io = [x for x in r["rows"] if x["body"] == "Io"][0]
        self.assertAlmostEqual(io["steps_per_orbit"], 142, delta=2)                                              # C19: "142 steps per orbit"
        self.assertIn("Io", r["unresolved"])
        fine = T.step_check("jupiter", r["step_that_resolves_all_s"]).to_dict()["value"]
        self.assertTrue(fine["all_adequate"])
        self.assertEqual(T.step_check("jupiter", -5.0).to_dict()["value"], None)


if __name__ == "__main__":
    unittest.main()
