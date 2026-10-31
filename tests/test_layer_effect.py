"""layer_effect: how much each physics layer matters for a body, checked against things the table does not contain.

* The Mercury relativity effect against the closed-form apsidal advance 6 pi G M / (c^2 a (1 - e^2)) per orbit with scipy constants (an order-of-magnitude bound, not a fit).
* Three cells recomputed from the raw ablation runs, when those are on disk.
* The ordering relativity must have (stronger nearer the Sun) and the ledger's C23b numbers (misc/CANDIDATE_LEDGER.md).
* Declines and aliases."""
from __future__ import annotations

import json
import math
import statistics
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = Path("D:/datasets/sigma_runs")


def _tool(name: str) -> dict:
    from sigma_ground.mcp.tools import celestial_tools as T
    return T.layer_effect(name).to_dict()


class TestPhysicsAnchors(unittest.TestCase):
    def test_mercury_without_relativity_is_off_by_about_the_apsidal_advance_times_the_orbit_radius(self):
        from scipy import constants as sc
        gm_sun, a, e, c = 1.32712440018e20, 5.790905e10, 0.2056, sc.c
        per_orbit = 6 * math.pi * gm_sun / (c ** 2 * a * (1 - e ** 2))              # radians of apsidal advance per orbit
        orbits_5yr = 5 * 365.25 / 87.969
        estimate_km = per_orbit * orbits_5yr * a / 1e3                              # the arc the orbit's pericentre moves in 5 yr, about 600 km
        got = _tool("mercury")["value"]["layers"]["eih"]["without_layer_km"]["5"]
        self.assertGreater(got, 0.8 * estimate_km)                                  # the C23b ledger recorded 1,046 km against a prediction of at most 1,000: failed, but the same order
        self.assertLess(got, 2.5 * estimate_km)

    def test_relativity_gets_weaker_with_distance_from_the_sun(self):
        r = [_tool(b)["value"]["layers"]["eih"]["ratio"]["5"] for b in ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")]
        self.assertEqual(r, sorted(r, reverse=True))

    def test_the_ledger_numbers(self):                                              # misc/CANDIDATE_LEDGER.md C23b
        self.assertAlmostEqual(_tool("moon")["value"]["layers"]["zonals"]["ratio"]["5"], 459, delta=5)
        self.assertAlmostEqual(_tool("jupiter")["value"]["layers"]["asteroids"]["ratio"]["5"], 52, delta=1)
        self.assertAlmostEqual(_tool("neptune")["value"]["layers"]["ring"]["ratio"]["5"], 26, delta=1)
        self.assertAlmostEqual(_tool("pluto")["value"]["layers"]["kbo"]["ratio"]["5"], 2.9, delta=0.1)
        self.assertAlmostEqual(_tool("moon")["value"]["layers"]["c22"]["ratio"]["5"], 5.0, delta=0.1)
        self.assertLess(_tool("mars")["value"]["layers"]["zonals"]["ratio"]["5"], 1.0)         # the unexplained lead: Mars agrees better without the zonals


@unittest.skipUnless((RUNS / "abl_eih").is_dir(), "the ablation runs are not on this machine")
class TestAgainstTheRawRuns(unittest.TestCase):
    def _median(self, folder: str, prefix: str, body: str, lead: str) -> float:
        return statistics.median(json.loads((RUNS / folder / f"{prefix}_{y}.json").read_text(encoding="utf-8"))["per_start"][str(y)][body][lead] for y in (2000, 2005, 2010, 2015))

    def test_three_cells_equal_the_raw_medians(self):
        for layer, body, lead in (("eih", "Venus", "3"), ("asteroids", "Saturn", "5"), ("ring", "Uranus", "1")):
            ratio = self._median(f"abl_{layer}", f"restated_zon_no_{layer}", body, lead) / self._median("zon_fine2", "restated_zon", body, lead)
            self.assertAlmostEqual(_tool(body.lower())["value"]["layers"][layer]["ratio"][lead], ratio, places=9)


class TestShape(unittest.TestCase):
    def test_every_body_has_every_layer_and_the_ranking_is_descending(self):
        for b in ("mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"):
            v = _tool(b)["value"]
            self.assertEqual(set(v["layers"]), {"eih", "zonals", "tides", "srp", "asteroids", "ring", "kbo", "c22"}, b)
            rk = [v["layers"][k]["ratio"]["5"] for k in v["ranked_by_5yr_ratio"]]
            self.assertEqual(rk, sorted(rk, reverse=True), b)
            self.assertTrue(all(x > 0 for k in v["layers"] for x in v["layers"][k]["ratio"].values()), b)
            self.assertEqual(set(v["error_with_all_layers_km"]), {"1", "2", "3", "4", "5"})

    def test_the_sentence_names_the_biggest_layer_and_flags_a_layer_whose_removal_helped(self):
        s = _tool("mercury")["value"]["summary"]
        self.assertIn("general relativity", s)
        self.assertNotIn("BETTER", s)
        m = _tool("mars")["value"]["summary"]
        self.assertIn("BETTER", m)
        self.assertIn("recorded, not resolved", m)

    def test_aliases_resolve_and_the_unscored_are_declined_with_the_reason(self):
        self.assertEqual(_tool("Luna")["value"]["body"], "Moon")
        r = _tool("ceres")
        self.assertIsNone(r["value"])
        self.assertIn("was not ablated", r["notes"])

    def test_the_table_still_matches_the_model_it_was_measured_with(self):
        self.assertFalse(_tool("earth")["value"]["model_changed_since_build"])             # True would mean the model changed and the table needs re-measuring


if __name__ == "__main__":
    unittest.main()
