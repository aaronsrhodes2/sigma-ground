"""The sealed forward predictions (C32): the sealed file is intact, complete and never edited; the seal script's conventions are the scoreboard's.

Ground truth not under test: the sealed numbers themselves are the prediction (judged later by `--judge` and the scoreboard); here only their integrity and shape.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
SEALS = sorted((ROOT / "sigma_ground/field/interface/fixtures/mpc_live").glob("forecast_sealed_*.json"))


class TestSealedFile(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SEALS, "no sealed forecast file")
        self.doc = json.loads(SEALS[-1].read_text(encoding="utf-8"))

    def test_rows_hash_is_the_recorded_one(self):                                             # an edited sealed row changes the hash: the seal is only a seal if this holds
        import mpc_forecast_seal as fs
        self.assertEqual(fs.rows_hash(self.doc["rows"]), self.doc["metadata"]["rows_sha256"])

    def test_every_tracked_body_at_every_date(self):
        import mpc_forecast_seal as fs, mpc_live_score as m
        want = {(n, d) for n, _ in m.tracked() for d in fs.DATES_ALL} | {(n, d) for n, ds in fs.DATES_EXTRA.items() for d in ds}
        got = {(r["body"], r["date_utc"][:10]) for r in self.doc["rows"]}
        self.assertEqual(got, want)
        self.assertEqual(self.doc["metadata"]["n_rows"], len(self.doc["rows"]))

    def test_rows_carry_state_direction_and_their_start(self):
        for r in self.doc["rows"]:
            self.assertEqual(len(r["state_bary_km_kms"]), 6, r["body"])
            self.assertTrue(0.0 <= r["ra_deg"] < 360.0 and -90.0 <= r["dec_deg"] <= 90.0, r["body"])
            self.assertGreater(r["lead_days"], 300.0)
            self.assertIn("JPL Horizons barycentric state", r["start_source"])
        apophis = [r for r in self.doc["rows"] if r["body"] == 99942]
        self.assertTrue(all(r["yarkovsky"] for r in apophis))                                   # C31's term is in the sealed Apophis prediction
        self.assertTrue(all(r["yarkovsky"] is None for r in self.doc["rows"] if r["body"] in (433, 1, 4)))
        close = min(apophis, key=lambda r: r["geocentric_distance_km"])
        self.assertEqual(close["date_utc"][:10], "2029-04-14")                                 # 00:00 UTC on the 14th is two hours after closest approach (2029-04-13 ~21:46 UTC)
        self.assertLess(close["geocentric_distance_km"], 100_000.0)


class TestConventions(unittest.TestCase):
    def test_tdb_of_utc_midnight_matches_the_scoreboards_scale(self):
        import mpc_forecast_seal as fs
        jd = fs.tdb_of_utc_midnight("2026-01-01")
        self.assertAlmostEqual((jd - 2461041.5) * 86400.0, 69.184, delta=0.01)                  # TT - UTC = 69.184 s in 2026; TDB - TT under 2 ms

    def test_rows_hash_is_order_insensitive_in_keys_and_sensitive_in_values(self):
        import mpc_forecast_seal as fs
        a = [{"x": 1, "y": 2.5}]; b = [{"y": 2.5, "x": 1}]; c = [{"x": 1, "y": 2.5000001}]
        self.assertEqual(fs.rows_hash(a), fs.rows_hash(b)); self.assertNotEqual(fs.rows_hash(a), fs.rows_hash(c))


if __name__ == "__main__":
    unittest.main()
