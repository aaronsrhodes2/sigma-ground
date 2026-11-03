"""body_position(calculate=True): the N-body chain runs and shows its work even where the kernel knows the answer (the Captain, 2026-11-03: "when I say calculate I want
the full N-body calculation chain and the work shown, not a lookup").

Ground truth not under test: the kernel's own value at the asked date, which the calculated answer must land near (a short lead keeps the test to seconds and the miss to metres).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HAVE_KERNEL = Path("D:/datasets/jpl/de442/de442.bsp").exists() or Path("D:/datasets/jpl/de440s.bsp").exists()


@unittest.skipUnless(HAVE_KERNEL, "no JPL kernel on this machine")
class TestCalculateOnRequest(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import where_when
        self.ww = where_when
        self.ww.forget_live()

    def test_a_known_date_is_a_lookup_by_default_and_a_calculation_on_request(self):           # BUG if calculate=True still returned the kernel's value
        plain = self.ww.position_at("Mars", "2015-06-15", "sun", budget_s=60)
        self.assertEqual(plain["kind"], "known")
        self.assertNotIn("receipt", plain)
        calc = self.ww.position_at("Mars", "2015-06-15", "sun", budget_s=120, calculate=True, calc_lead_years=0.02)     # about a week of integration
        self.assertEqual((calc["kind"], calc["status"]), ("calculated", "computed"))
        rc = calc["receipt"]
        self.assertIn("calculate=True", rc["start"]["what_it_is"])
        self.assertAlmostEqual(rc["lead_years"], 0.02, delta=1e-9)
        self.assertGreater(rc["force_evaluations_this_call"], 0)
        self.assertTrue(any(l["applied"] for l in rc["layers"]))
        self.assertIn("reality here", calc["jpl_same_instant"]["note"])
        self.assertIsNone(calc["expected_distance_from_de440"])                                 # the hindcast table is for leads from the edge, not this
        # the work lands where reality is: a week of this model's Mars is well under a kilometre from the kernel, and not exactly on it (it was computed, not copied)
        d = calc["cross_check"]["difference_km"]
        self.assertLess(d, 1.0, f"{d} km")
        self.assertGreater(d, 0.0)
        self.assertLess(np.linalg.norm(np.array(calc["position_km"]) - np.array(plain["position_km"])), 1.0)

    def test_the_flag_changes_nothing_past_the_edge(self):                                      # BY CONSTRUCTION: there it is already a calculation
        edge = self.ww._truth()[2]
        a = self.ww.position_at("Mars", f"JD {edge + 1.0}", "sun", budget_s=60)
        b = self.ww.position_at("Mars", f"JD {edge + 1.0}", "sun", budget_s=60, calculate=True)
        self.assertEqual((a["kind"], b["kind"]), ("prediction", "prediction"))
        self.assertEqual(a["position_km"], b["position_km"])

    def test_a_start_before_the_kernel_is_refused(self):
        lo = self.ww._truth()[0][0, 3].start_jd
        with self.assertRaises(self.ww.WhereWhenError):
            self.ww.position_at("Mars", f"JD {lo + 10.0}", "sun", calculate=True)

    def test_the_tool_says_calculated_and_carries_the_receipt(self):
        from sigma_ground.mcp.tools import astronomy
        t = astronomy.body_position("mars", "2015-06-15", budget_s=150, calculate=True).to_dict()          # the default half-year lead: about 35 s on the loaded PC, measured 2026-11-03
        self.assertEqual(t["inputs"]["calculate"], True)
        self.assertTrue(t["notes"].startswith("CALCULATED."))
        self.assertIn("RECEIPT:", t["notes"])


if __name__ == "__main__":
    unittest.main()
