"""body_position: known values from DE440, labelled predictions, honest refusals.

Ground truth that is not the code under test: the DE440 fixture's own states (same kernel, read independently), the leap-second arithmetic (TT-UTC = 64.184 s at 2000-01-01: 32 leap seconds + 32.184 s),
and the physical range of the Moon's distance (356,400 to 406,700 km).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = ROOT / "sigma_ground/field/interface/fixtures"
KERNEL = Path("D:/datasets/jpl/de440s.bsp")


class TestTime(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import where_when
        self.ww = where_when

    def test_iso_is_utc_and_converted_to_tdb_with_the_leap_seconds(self):                     # BUG if the 64.184 s were dropped
        jd, iso, how = self.ww.parse_when("2000-01-01T12:00:00")
        self.assertAlmostEqual(jd, 2451545.0 + 64.184 / 86400.0, delta=2e-8)                   # 2 ms
        self.assertEqual(iso, "2000-01-01T12:00:00Z")

    def test_a_julian_date_is_read_as_tdb_unchanged(self):                                      # BY CONSTRUCTION
        self.assertEqual(self.ww.parse_when("JD 2462000.5")[0], 2462000.5)
        self.assertEqual(self.ww.parse_when("2462000.5")[0], 2462000.5)

    def test_now_is_the_clock(self):                                                            # BY CONSTRUCTION
        jd, iso, _ = self.ww.parse_when("now", datetime.datetime(2026, 10, 26, 12, 0, 0, tzinfo=datetime.timezone.utc))
        # 2026-10-26 12:00 UTC is 298.5 days after 2026-01-01 00:00 (JD 2461041.5, the fixture's J2026 epoch): JD 2461340.0; TT-UTC = 69.184 s
        self.assertAlmostEqual(jd, 2461340.0 + 69.184 / 86400.0, delta=2e-8)

    def test_an_unreadable_time_says_what_to_write(self):                                       # BY CONSTRUCTION: suggest, never guess
        with self.assertRaises(self.ww.WhereWhenError) as cm:
            self.ww.parse_when("next tuesday")
        self.assertIn("ISO date", str(cm.exception))


class TestNames(unittest.TestCase):
    def test_nicknames_resolve_and_unsupported_bodies_are_refused_with_the_list(self):
        from sigma_ground.field.interface import where_when as ww
        self.assertEqual(ww.canonical_body("Luna"), "Moon")
        self.assertEqual(ww.canonical_body("the red planet"), "Mars")
        with self.assertRaises(ww.WhereWhenError) as cm:
            ww.canonical_body("Io")
        self.assertIn("Supported:", str(cm.exception))

    def test_the_earth_from_the_earth_is_refused(self):
        from sigma_ground.field.interface import where_when as ww
        with self.assertRaises(ww.WhereWhenError):
            ww.position_at("Earth", "2020-01-01", "earth")


class TestErrorTable(unittest.TestCase):
    def test_loglog_interpolation_and_the_flags_outside_the_measured_leads(self):               # BUG if a lead beyond the table were presented as measured
        from sigma_ground.field.interface import where_when as ww
        doc = {"metadata": {"starts": [2000]}, "table": {"Earth": {"1": {"median_km": 1.0, "max_km": 2.0, "n_starts": 8}, "10": {"median_km": 10.0, "max_km": 20.0, "n_starts": 4},
                                                                    "20": {"median_km": 40.0, "max_km": 80.0, "n_starts": 2}}}}
        with mock.patch.object(ww, "_error_table", return_value=doc):
            e = ww.expected_error("Earth", 10 ** 0.5)
            self.assertAlmostEqual(e["typical_km"], 10 ** 0.5, delta=1e-9)                       # power law between 1 and 10: 1 km * lead
            self.assertIsNone(e["note"])
            e = ww.expected_error("Earth", 0.5)
            self.assertAlmostEqual(e["typical_km"], 0.5, delta=1e-9)
            self.assertIn("scaled linearly", e["note"])
            e = ww.expected_error("Earth", 30.0)
            self.assertEqual(e["typical_km"], 40.0)
            self.assertIn("floor", e["note"])
            self.assertIsNone(ww.expected_error("Mars", 5.0))


@unittest.skipUnless(KERNEL.exists(), "DE440s kernel not available")
class TestKnown(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import where_when
        self.ww = where_when

    def test_earth_at_a_fixture_epoch_equals_the_fixture(self):                                  # BUG: same kernel, independent read of the state
        fx = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"]["j2010"]
        jd = fx["epoch"]["jd_tdb"]
        by = {b["name"]: b["state_vector"] for b in fx["bodies"]}
        want = np.array([by["Earth"][c] - by["Sun"][c] for c in ("x_km", "y_km", "z_km")])
        r = self.ww.position_at("Earth", f"JD {jd}", "sun", now=datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc))
        self.assertEqual(r["kind"], "known")
        self.assertLess(np.linalg.norm(np.array(r["position_km"]) - want), 1e-3)
        self.assertFalse(r["beyond_fit_data"])
        self.assertAlmostEqual(r["distance_au"], 0.983, delta=0.01)                              # January perihelion

    def test_a_date_after_the_fit_data_is_known_but_flagged(self):                               # BUG if 2023 were presented as an observation
        r = self.ww.position_at("Mars", "2023-03-01", now=datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc))
        self.assertEqual(r["kind"], "known")
        self.assertTrue(r["beyond_fit_data"])
        self.assertIn("not a new observation", r["basis"])

    def test_moon_from_the_earth_has_a_physical_distance_ra_and_dec(self):
        r = self.ww.position_at("Moon", "2019-07-02T19:24:00", "earth", now=datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc))   # the total eclipse of 2019-07-02: new Moon
        self.assertTrue(356000 < r["distance_km"] < 407000)
        self.assertTrue(0.0 <= r["ra_deg"] < 360.0 and -30.0 < r["dec_deg"] < 30.0)

    def test_centers_agree_with_each_other(self):                                                # BY CONSTRUCTION: sun-centred = ssb-centred minus the Sun's ssb position
        now = datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc)
        a = np.array(self.ww.position_at("Jupiter", "2012-05-05", "sun", now=now)["position_km"])
        b = np.array(self.ww.position_at("Jupiter", "2012-05-05", "ssb", now=now)["position_km"])
        s = np.array(self.ww.position_at("Sun", "2012-05-05", "ssb", now=now)["position_km"])
        self.assertLess(np.linalg.norm(a - (b - s)), 1e-6)

    def test_outside_the_kernel_span_is_refused(self):                                           # BY CONSTRUCTION
        with self.assertRaises(self.ww.WhereWhenError) as cm:
            self.ww.position_at("Venus", "1700-01-01", now=datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc))
        self.assertIn("1849", str(cm.exception))

    def test_a_prediction_without_our_table_falls_back_to_jpl_and_says_so(self):                 # BUG if JPL's value were passed off as ours
        with mock.patch.object(self.ww, "_forecast", return_value=(None, "no forecast has been built")):
            r = self.ww.position_at("Venus", "2033-01-01", now=datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc))
        self.assertEqual(r["kind"], "prediction")
        self.assertIn("JPL DE440 extrapolation", r["method"])
        self.assertIn("JPL's own prediction, not ours", " ".join(r["notes"]))


class TestSealedTable(unittest.TestCase):
    def test_a_table_that_no_longer_matches_its_seal_is_refused(self):                           # BUG if an edited forecast were served
        from sigma_ground.field.interface import where_when as ww
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "forecast").mkdir()
            f = t / "forecast" / "f.npz"
            np.savez_compressed(f, jd=np.arange(3.0), pos_km=np.zeros((3, 1, 3)), vel_km_s=np.zeros((3, 1, 3)), names=np.array(["Sun"]))
            seal = hashlib.sha256(f.read_bytes()).hexdigest()
            man = t / "forecast_manifest.json"
            man.write_text(json.dumps({"tables": [{"file": "f.npz", "sha256": seal, "created_utc": "x", "git_commit": "y"}]}))
            with mock.patch.object(ww, "_FIX", t), mock.patch.dict(os.environ, {"SIGMA_DATASETS": str(t)}):
                tb, info = ww._forecast.__wrapped__()
                self.assertIsNotNone(tb)                                                         # intact: served
                f.write_bytes(f.read_bytes() + b"x")                                             # altered after the seal
                tb, info = ww._forecast.__wrapped__()
                self.assertIsNone(tb)
                self.assertIn("does not match its sealed sha256", info)


class TestTool(unittest.TestCase):
    def test_tool_labels_a_pre_2020_value_verified_and_refuses_a_moon(self):
        from sigma_ground.mcp.tools import astronomy
        if KERNEL.exists():
            r = astronomy.body_position("mars", "2015-06-15").to_dict()
            self.assertEqual(r["provenance_tag"], "VERIFIED")
            self.assertTrue(r["notes"].startswith("KNOWN."))
        r = astronomy.body_position("io", "2020-01-01").to_dict()
        self.assertIsNone(r["value"])
        self.assertIn("Supported:", r["notes"])


if __name__ == "__main__":
    unittest.main()
