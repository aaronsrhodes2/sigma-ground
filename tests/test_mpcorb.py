"""The MPCORB reader (stage 2, phase 1): packed designations, epochs and the fixed columns, on lines copied from DAILY.DAT of 2026-11-04 and the MPC's documented examples."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LINES = [
    "00005    6.95  0.15 K2669 181.47820  359.37561  141.44724    5.35968  0.1875740  0.23827641   2.5768077  0 E2026-V37  3693  91 1845-2026 0.85 M-v 3Ek MPCORBFIT  4000      (5) Astraea            20261029",
    "15231   12.10  0.15 K2669 331.04045  216.03521  158.23554    6.81530  0.2926780  0.12598094   3.9409370  0 E2026-V37  3520  31 1987-2026 0.74 M-v 3Ek MPCORBFIT  0008  (15231) Ehdita             20261101",
    "A0029   15.16  0.15 K2669 337.64068   85.37472  315.72787    4.11621  0.1010758  0.19285389   2.9669971  0 E2026-V37  1183  22 1990-2026 0.81 M-v 3Ek MPCORBFIT  0000 (100029) Varnhagen          20261103",
    "~0J43   17.43  0.15 K2669 322.44505   32.16256   12.15601   14.90479  0.2144189  0.22587091   2.6703151  0 E2026-V37   151   8 2013-2026 0.98 M-v 3Ek MPCORBFIT  0000 (693287) 2015 FX219         20261103",
    "K25MM4T 18.25  0.15 K2669 295.08532   80.83803    6.58120    3.15012  0.0567374  0.21540279   2.7561435  0 E2026-V37    70   7 2012-2026 0.80 M-v 3Ek MPCORBFIT  0000          2025 MT224         20261103",
]


class TestPacked(unittest.TestCase):
    def test_numbers(self):
        from sigma_ground.field.interface.mpcorb import unpack_number
        self.assertEqual([unpack_number(p) for p in ("00005", "15231", "A0029", "a0017", "~0J43", "~AZaz")], [5, 15231, 100029, 360017, 693287, 620000 + ((10 * 62 + 35) * 62 + 36) * 62 + 61])

    def test_provisional_and_epoch(self):
        from sigma_ground.field.interface.mpcorb import unpack_epoch_jd, unpack_provisional
        self.assertEqual(unpack_provisional("K25MM4T"), "2025 MT224")
        self.assertEqual(unpack_provisional("J95X00A"), "1995 XA")
        self.assertEqual(unpack_provisional("K07Tf8A"), "2007 TA418")
        self.assertEqual(unpack_provisional("PLS2040"), "2040 P-L")
        self.assertAlmostEqual(unpack_epoch_jd("K2669"), 2461200.5, places=9)                     # the SBDB epoch of the same solutions (2026-06-09 00:00 TT)
        self.assertAlmostEqual(unpack_epoch_jd("J9611"), 2450083.5, places=9)                     # 1996-01-01


class TestRecords(unittest.TestCase):
    def test_the_sampled_lines(self):
        from sigma_ground.field.interface.mpcorb import parse_lines
        o = parse_lines(LINES)
        self.assertEqual([x.number for x in o], [5, 15231, 100029, 693287, None])
        self.assertEqual(o[-1].designation, "2025 MT224")
        self.assertEqual((o[0].h_mag, o[0].a_au, o[0].e, o[0].i_deg), (6.95, 2.5768077, 0.187574, 5.35968))
        self.assertEqual((o[0].n_obs, o[0].n_opp, o[0].arc, o[0].rms_arcsec, o[0].u_param, o[0].last_obs_yyyymmdd), (3693, 91, "1845-2026", 0.85, "0", "20261029"))
        self.assertEqual(o[0].readable, "(5) Astraea")
        self.assertEqual({x.orbit_class for x in o}, {"mba"})

    def test_header_and_junk_are_skipped(self):
        from sigma_ground.field.interface.mpcorb import parse_lines
        o = parse_lines(["MINOR PLANET CENTER ORBIT DATABASE (MPCORB)", "", "Des'n     H     G   Epoch     M        Peri.", "-" * 160] + LINES[:1] + ["", "garbage"])
        self.assertEqual(len(o), 1)


if __name__ == "__main__":
    unittest.main()
