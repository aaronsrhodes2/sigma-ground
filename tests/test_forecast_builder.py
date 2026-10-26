"""scripts/make_forecast_table.py must refuse to seal a table built with a step too coarse for Mercury and the Moon, or one whose elapsed part is far from DE440.

The case that motivated it (2026-10-26): a first build at 0.25 d was 102 km (Mercury) and 259 km (Moon) from DE440 after 0.82 yr and 3.0 km for the Earth, against 0.03, 0.1 and 0.04 km at the shoot-out's 0.025 d.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


class TestRefusal(unittest.TestCase):
    def setUp(self):
        import make_forecast_table as m
        self.m = m

    def test_a_coarse_step_is_refused(self):                                                   # BUG: the 0.25 d build
        self.assertIn("too coarse", self.m.refusal(0.25, {}))
        self.assertIsNone(self.m.refusal(0.025, {}))
        self.assertIsNone(self.m.refusal(0.05, {}))

    def test_a_table_far_from_de440_is_refused_and_names_the_bodies(self):                      # BUG: the same build's check
        why = self.m.refusal(0.025, {"Mercury": 101.7, "Earth": 2.97, "Moon": 259.5, "Mars": 0.011})
        self.assertIn("Mercury", why)
        self.assertIn("Moon", why)
        self.assertNotIn("Mars", why)

    def test_a_good_check_passes(self):                                                        # BY CONSTRUCTION: values of the size the shoot-out measures at +1 yr
        self.assertIsNone(self.m.refusal(0.025, {"Mercury": 0.005, "Venus": 0.02, "Earth": 0.04, "Moon": 0.10, "Mars": 0.011}))


if __name__ == "__main__":
    unittest.main()
