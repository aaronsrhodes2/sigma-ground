"""comet_truth.parse_horizons_comet against a minimal response in the real Horizons layout (header lines copied from the 1P/Halley response of 2026-10-15)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sigma_ground.field.interface.comet_truth import parse_horizons_comet

SAMPLE = """Rec #:90000030        Soln.date: 2025-Nov-21_15:57:34   # obs: 8518 (1835-1994)
Comet non-gravitational force model (AMRAT=m^2/kg;A1-A3=au/d^2;DT=days;R0=au):
   AMRAT=  0.                                      DT=  0.
   A1= 4.887055233121E-10  A2= 1.554720290005E-10  A3= 0.
 Non-standard or simulated/proxy model:
   ALN=  .1112620426   NK=  4.6142   NM=  2.15     NN=  5.093    R0=  2.808
Target body name: 1P/Halley                       {source: JPL#75}
$$SOE
2455197.500000000, A.D. 2010-Jan-01 00:00:00.0000, -3.0E+09,  3.7E+09,  9.7E+07, -5.3E+04,  1.9E+05,  3.3E+04,
2455562.750000000, A.D. 2011-Jan-01 06:00:00.0000, -3.1E+09,  3.8E+09,  1.0E+08, -4.3E+04,  1.8E+05,  3.3E+04,
$$EOE
"""


class TestParser(unittest.TestCase):
    def _parse(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            p.write_text(json.dumps({"result": text}))
            return parse_horizons_comet(p)

    def test_fields(self):                                                           # BUG
        r = self._parse(SAMPLE)
        self.assertEqual(r["target"], "1P/Halley")
        self.assertEqual(r["solution_source"], "JPL#75")
        self.assertEqual(r["A"], (4.887055233121e-10, 1.554720290005e-10, 0.0))
        self.assertEqual(r["law"]["r0"], 2.808)
        self.assertEqual(r["state_km_kmday"].shape, (2, 6))
        self.assertEqual(float(r["jd"][1]), 2455562.75)
        self.assertEqual(float(r["state_km_kmday"][0, 3]), -5.3e4)

    def test_missing_force_parameter_is_an_error_not_a_default(self):                # BUG: a silent A = 0 would corrupt the test
        with self.assertRaises(ValueError):
            self._parse(SAMPLE.replace("A2= 1.554720290005E-10", ""))

    def test_no_ephemeris_block_is_an_error(self):
        with self.assertRaises(ValueError):
            self._parse("Multiple major-bodies match string 'DES=9P'")


if __name__ == "__main__":
    unittest.main()
