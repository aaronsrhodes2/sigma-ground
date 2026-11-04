"""The obs80 reader with designations (stage 2, phase 1): lines copied from the MPC archive and from the scoreboard's per-body files."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NUMBERED_OLD = "00001         A1801 01 01.82630 03 38 23.07 +16 17 25.5                 MC004535"            # photographic/legacy: note 2 'A', dropped
CCD = "00433         C2026 03 21.17320 11 32 18.23 -00 46 23.1          20.6 RX~9bzPX19"
UNNUMBERED = "     K25MM4T  C2025 06 15.30188 15 10 44.655-19 40 48.63         20.4 GV~8KbyF51"
EXTENDED = "~19mAK26D18M  C2026 03 21.20251 11 32 16.39 -00 46 11.2          20.8 RX~9bzPX19"


class TestReader(unittest.TestCase):
    def test_designations_and_fields(self):
        from sigma_ground.field.interface.obs80 import packed_key, parse_line
        r = parse_line(CCD)
        self.assertEqual((r[0], r[1], r[2], r[3], r[4]), ("00433", "", "C", 2026, 3))
        self.assertAlmostEqual(r[5], 21.17320)
        self.assertAlmostEqual(math.degrees(r[6]), (11 + 32 / 60 + 18.23 / 3600) * 15, places=9)
        self.assertAlmostEqual(math.degrees(r[7]), -(0 + 46 / 60 + 23.1 / 3600), places=9)
        self.assertEqual((r[8], r[9], r[10]), (20.6, "R", "X19"))
        self.assertEqual(packed_key(r[0], r[1]), "00433")
        u = parse_line(UNNUMBERED)
        self.assertEqual((u[0], u[1], packed_key(u[0], u[1]), u[10]), ("", "K25MM4T", "K25MM4T", "F51"))
        self.assertLess(u[7], 0.0)                                                                 # '-19 40 48.63' with no space before the sign
        e = parse_line(EXTENDED)
        self.assertEqual((e[0], e[1]), ("~19mA", "K26D18M"))                                      # a numbered body above 619999 carries both

    def test_non_optical_and_short_lines_are_dropped(self):
        from sigma_ground.field.interface.obs80 import parse_line
        self.assertIsNone(parse_line(NUMBERED_OLD))
        self.assertIsNone(parse_line(CCD[:60]))
        self.assertIsNone(parse_line(CCD[:14] + "S" + CCD[15:]))

    def test_columns(self):
        from sigma_ground.field.interface.obs80 import COLUMNS, parse_line, rows_to_columns
        cols = rows_to_columns([parse_line(CCD), parse_line(UNNUMBERED)])
        self.assertEqual(list(cols), COLUMNS)
        self.assertEqual(cols["packed"], ["00433", "K25MM4T"])


if __name__ == "__main__":
    unittest.main()
