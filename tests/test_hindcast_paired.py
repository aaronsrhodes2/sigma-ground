"""hindcast_paired_report.compare: pairs the same body, lead and start in both predictors, uses only starts that have the lead, and counts the starts where the new one is closer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import hindcast_paired_report as hp  # noqa: E402


class TestCompare(unittest.TestCase):
    def test_median_ratio_and_starts_better(self):                                              # BUG if starts were mixed up or a missing lead were treated as zero
        control = {"2000": {"Moon": {"1": 1.0, "5": 4.0}}, "2010": {"Moon": {"1": 3.0}}, "2015": {"Moon": {"1": 2.0}}}
        new = {"2000": {"Moon": {"1": 0.5, "5": 4.0}}, "2010": {"Moon": {"1": 3.0}}, "2015": {"Moon": {"1": 3.0}}}
        rows = {(r["body"], r["lead"]): r for r in hp.compare(control, new)}
        r1 = rows[("Moon", 1)]
        self.assertEqual(r1["starts"], 3)
        self.assertEqual(r1["starts_better"], 1)                                                # only 2000 improved
        self.assertAlmostEqual(r1["control_km"], 2.0)                                           # median of 1, 3, 2
        self.assertAlmostEqual(r1["new_km"], 3.0)                                               # median of 0.5, 3, 3
        self.assertAlmostEqual(r1["ratio"], 1.5)
        self.assertEqual(rows[("Moon", 5)]["starts"], 1)                                        # only the 2000 start reaches +5 yr
        self.assertNotIn(("Moon", 10), rows)


if __name__ == "__main__":
    unittest.main()
