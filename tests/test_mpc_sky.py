"""The whole-sky pipeline's pure parts (stage 2): time scales match the scoreboard's, API files are read by packed designation, the seal's hash is stable."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

CCD = "00433         C2026 03 21.17320 11 32 18.23 -00 46 23.1          20.6 RX~9bzPX19"


class TestTimes(unittest.TestCase):
    def test_tdb_matches_the_scoreboards_obs_class(self):
        import erfa
        import eros_sky_residuals as esr, mpc_sky_score as sc
        jd, _ = sc.jd_utc_to_tdb(np.array([2026]), np.array([3]), np.array([21.1732]))
        u1, u2 = erfa.dtf2d("UTC", 2026, 3, 21, 0, 0, 0.0); t1, t2 = erfa.taitt(*erfa.utctai(u1, u2 + 0.1732))
        self.assertAlmostEqual(jd[0], t1 + t2, delta=0.002 / 86400.0)
        import mpc_sky_seal as ss
        from datetime import date
        self.assertAlmostEqual((ss.midnight_tdb(date(2026, 1, 1)) - 2461041.5) * 86400.0, 69.184, delta=0.01)


class TestFiles(unittest.TestCase):
    def test_api_files_are_keyed_by_packed_designation(self):
        import mpc_sky_score as sc
        with tempfile.TemporaryDirectory() as d:
            old = sc.SKY; sc.SKY = Path(d)
            try:
                day = Path(d) / "obs_api" / "2026-11-04"; day.mkdir(parents=True)
                (day / "00433.txt").write_text(CCD + "\n" + CCD[:14] + "S" + CCD[15:] + "\n", encoding="utf-8")
                (day / "K25MM4T.txt").write_text("", encoding="utf-8")
                obs = sc.load_api_obs("2026-11-04", None)
            finally:
                sc.SKY = old
        self.assertEqual(list(obs), ["00433"]); self.assertEqual(len(obs["00433"]), 1); self.assertEqual(obs["00433"][0][10], "X19")

    def test_seal_hash_covers_state_and_direction_only(self):
        import pyarrow as pa
        import mpc_sky_seal as ss
        base = {"packed": "00433", "date_utc": "2026-11-05T00:00:00Z", "state_x": 1.0, "state_y": 2.0, "state_z": 3.0, "state_vx": 0.1, "state_vy": 0.2, "state_vz": 0.3, "ra_deg": 10.0, "dec_deg": -5.0, "light_time_s": 400.0}
        a = pa.Table.from_pylist([base]); b = pa.Table.from_pylist([{**base, "light_time_s": 401.0}]); c = pa.Table.from_pylist([{**base, "ra_deg": 10.0000001}])
        self.assertEqual(ss.rows_hash(a), ss.rows_hash(b)); self.assertNotEqual(ss.rows_hash(a), ss.rows_hash(c))


if __name__ == "__main__":
    unittest.main()
