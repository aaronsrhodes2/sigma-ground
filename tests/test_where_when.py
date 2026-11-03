"""body_position: known values from the truth kernel (DE442), live predictions computed when asked, honest refusals and budgets.

Ground truth that is not the code under test: the kernel's own states read independently with jplephem, the leap-second arithmetic (TT-UTC = 64.184 s at 2000-01-01: 32 leap seconds + 32.184 s),
and the physical range of the Moon's distance (356,400 to 406,700 km). The live tests integrate a day or two past the edge (seconds), never a year.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = ROOT / "sigma_ground/field/interface/fixtures"
KERNEL = Path("D:/datasets/jpl/de440s.bsp")
TRUTH = Path("D:/datasets/jpl/de442/de442.bsp")
NOW = datetime.datetime(2026, 11, 2, tzinfo=datetime.timezone.utc)


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

    def test_mercury_and_the_moon_carry_the_step_caveat_and_the_others_do_not(self):          # BUG if a row's low value at 0.025 d were quoted without saying what it is made of (C19)
        from sigma_ground.field.interface import where_when as ww
        row = {"1": {"median_km": 1.0, "max_km": 2.0, "n_starts": 4}, "5": {"median_km": 2.0, "max_km": 4.0, "n_starts": 4}}
        doc = {"metadata": {"starts": [2000], "dt_days": 0.025}, "table": {n: row for n in ("Mercury", "Moon", "Venus")}}
        with mock.patch.object(ww, "_error_table", return_value=doc):
            m = ww.expected_error("Mercury", 3.0)
            self.assertEqual(m["integration_step_days"], 0.025)
            self.assertIn("integration error cancelling", m["step_caveat"])
            self.assertIn("integration error", ww.expected_error("Moon", 3.0)["step_caveat"])
            self.assertIsNone(ww.expected_error("Venus", 3.0)["step_caveat"])


@unittest.skipUnless(KERNEL.exists(), "DE440s kernel not available")
class TestKnown(unittest.TestCase):
    def setUp(self):
        from sigma_ground.field.interface import where_when
        self.ww = where_when

    def test_earth_at_a_fixture_epoch_equals_the_truth_kernel_read_independently(self):          # BUG: same kernel, independent read of the state
        from jplephem.spk import SPK
        fx = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"]["j2010"]
        jd = fx["epoch"]["jd_tdb"]
        k = SPK.open(str(TRUTH if TRUTH.exists() else KERNEL))
        want = (k[0, 3].compute(jd) + k[3, 399].compute(jd)) - k[0, 10].compute(jd)
        r = self.ww.position_at("Earth", f"JD {jd}", "sun", now=NOW)
        self.assertEqual(r["kind"], "known")
        self.assertLess(np.linalg.norm(np.array(r["position_km"]) - want), 1e-6)
        self.assertFalse(r["beyond_fit_data"])
        self.assertIn(r["reality_edge"]["kernel"], ("DE442", "DE440 (fallback: DE442 not installed)"))
        self.assertAlmostEqual(r["distance_au"], 0.983, delta=0.01)                              # January perihelion

    def test_a_date_inside_the_truth_kernels_data_is_known_and_a_date_after_its_edge_is_not(self):   # BUG if 2025 were presented as known
        r = self.ww.position_at("Mars", "2023-03-01", now=NOW)
        self.assertEqual(r["kind"], "known")
        self.assertTrue(r["method"].startswith("JPL"))
        if TRUTH.exists():
            self.assertEqual(r["reality_edge"]["date"], "2024-01-01")
            self.assertEqual(self.ww.position_at("Mars", "2025-06-01", now=NOW, budget_s=0.0)["kind"], "prediction")   # budget 0: the build only, no year of integration

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
            self.ww.position_at("Venus", "1400-01-01", now=NOW)
        self.assertIn("outside", str(cm.exception))


@unittest.skipUnless(KERNEL.exists(), "DE440s kernel not available")
class TestLive(unittest.TestCase):
    """A date after the edge is OUR calculation made now (the Captain, 2026-11-02): no table is read, the memo is declared, the budget is honoured."""
    def setUp(self):
        from sigma_ground.field.interface import where_when
        self.ww = where_when
        self.ww.forget_live()
        self.edge = self.ww._truth()[2]

    def test_a_day_past_the_edge_is_computed_now_and_lands_on_jpls_extrapolation(self):          # BUG if the live path drifted from JPL by more than a day of this model's error
        with mock.patch.object(self.ww, "_forecast", side_effect=AssertionError("the sealed table must not be read for a position")):
            r = self.ww.position_at("Mars", f"JD {self.edge + 1.0}", "sun", now=NOW, budget_s=60)
        self.assertEqual((r["kind"], r["status"]), ("prediction", "computed"))
        self.assertIn("computed now", r["method"])
        self.assertLess(r["cross_check"]["difference_km"], 1e-3)                                  # measured 1.2e-5 km after one day (2026-11-02)
        self.assertEqual(r["live"]["predictor"], self.ww.SERVED["predictor"])
        self.assertEqual(r["live"]["integrator"], self.ww.SERVED["integrator"])
        self.assertFalse(r["live"]["memo"]["used"])
        self.assertEqual(r["live"]["model_fingerprint"], self.ww._current_fingerprint(self.ww.SERVED["predictor"]))
        self.assertIsNotNone(r["jpl_same_instant"]["position_km"])
        self.assertAlmostEqual(r["lead_years_from_edge"], 1.0 / 365.25, delta=1e-9)

    def test_the_answer_shows_its_work_in_a_receipt_with_the_layers_and_their_sources(self):     # the Captain, 2026-11-02: the time it took is the exhibit; a bare number is not done
        seen = []
        r = self.ww.position_at("Mars", f"JD {self.edge + 3.0}", "sun", now=NOW, budget_s=60, progress=seen.append)
        rc = r["receipt"]
        for key in ("machine", "wall_seconds_this_call", "wall_seconds_whole_integration", "integrator", "steps_this_call", "force_evaluations_this_call", "force_evaluations_whole_integration",
                    "bodies_integrated", "prescribed_sources", "start", "lead_years", "predictor", "model_fingerprint", "memo", "layers"):
            self.assertIn(key, rc)
        self.assertTrue(rc["machine"]["host"])
        self.assertEqual(rc["integrator"]["name"], self.ww.SERVED["integrator"])
        self.assertEqual(rc["integrator"]["step_days"], self.ww.SERVED["dt_days"])
        self.assertEqual(rc["bodies_integrated"], 11)
        self.assertEqual(rc["prescribed_sources"], 386)                                        # 343 asteroids + the 36-point ring + 7 Kuiper-belt objects
        self.assertEqual(rc["force_evaluations_this_call"], 30 * 28)                            # 30 steps of 0.1 d, Yoshida-6 with the relativistic layer: 14 implicit kicks of 2 evaluations
        self.assertEqual(rc["start"]["kernel"], r["reality_edge"]["kernel"])
        self.assertIn("JPL's measured state", rc["start"]["what_it_is"])
        self.assertFalse(rc["memo"]["used"])
        on = {l["layer"] for l in rc["layers"] if l["applied"]}
        self.assertEqual(on, {"newton", "eih", "zonals", "tides", "c22", "srp", "asteroids", "ring", "kbo"})
        self.assertFalse(any(l["applied"] for l in rc["layers"] if l["layer"] in ("de_tide", "gm_rate")))
        for l in rc["layers"]:
            self.assertTrue(l["source"])
        sizes = {l["layer"]: l["measured_size"] for l in rc["layers"] if "measured_size" in l}
        self.assertIn("eih", sizes)                                                             # the ledger measured the layers for Mars (C23b)
        self.assertGreater(sizes["eih"]["error_ratio_without_it"], 1.0)
        self.assertTrue(seen and seen[-1]["force_evaluations"] > 0 and seen[-1]["years_asked"] > 0)   # progress was reported while it ran
        from sigma_ground.mcp.tools import astronomy
        t = astronomy.body_position("mars", f"JD {self.edge + 3.5}", budget_s=60).to_dict()
        self.assertIn("RECEIPT:", t["notes"])
        self.assertIn("memo used", t["notes"])

    def test_a_second_question_inside_the_memo_is_instant_and_says_so(self):                    # BY CONSTRUCTION: the memo is keyed by fingerprint and start, and declared
        self.ww.position_at("Earth", f"JD {self.edge + 2.0}", "sun", now=NOW, budget_s=60)
        t0 = time.perf_counter()
        r = self.ww.position_at("Moon", f"JD {self.edge + 1.5}", "earth", now=NOW, budget_s=60)
        self.assertLess(time.perf_counter() - t0, 1.0)
        self.assertTrue(r["live"]["memo"]["used"])
        self.assertIn("memo", r["basis"])
        self.assertTrue(356000 < r["distance_km"] < 407000)
        self.assertLess(r["cross_check"]["difference_km"], 0.01)

    def test_over_budget_says_how_far_it_got_and_gives_no_position_of_ours(self):               # BUG if the chat hung or JPL's value were passed off as ours
        r = self.ww.position_at("Mars", f"JD {self.edge + 30 * 365.25}", "sun", now=NOW, budget_s=0.5)
        self.assertEqual(r["status"], "over_budget")
        self.assertIsNone(r["position_km"])
        self.assertGreater(r["offer"]["reached_lead_years"], 0.0)
        self.assertLess(r["offer"]["reached_lead_years"], 1.0)
        self.assertGreater(r["offer"]["seconds_to_finish_estimate"], 0.0)
        self.assertIn("ask again", r["offer"]["how"])
        self.assertIn("not ours", " ".join(r["notes"]))
        from sigma_ground.mcp.tools import astronomy
        t = astronomy.body_position("mars", f"JD {self.edge + 30 * 365.25}", budget_s=0.5).to_dict()
        self.assertTrue(t["notes"].startswith("NOT COMPUTED YET"))

    def test_the_memo_continues_the_same_integration(self):                                      # BY CONSTRUCTION: a second call adds steps, it does not restart
        r1 = self.ww.live_states(self.edge + 1.0, budget_s=60)
        r2 = self.ww.live_states(self.edge + 2.0, budget_s=60)
        self.assertTrue(r1["reached"] and r2["reached"])
        self.assertFalse(r2["memo"]["fresh_this_call"])
        self.assertEqual(r2["memo"]["rows"], r1["memo"]["rows"] + r2["steps_this_call"])


class TestFreshness(unittest.TestCase):
    """A stored table is a cache of a calculation: it must notice when the calculation has changed (the Captain, 2026-10-26)."""

    def test_matching_fingerprint_is_fresh_changed_is_stale_missing_is_unchecked(self):         # BUG if an out-of-date table were served as current
        from sigma_ground.field.interface import where_when as ww
        with mock.patch.object(ww, "_current_fingerprint", return_value="abc"):
            self.assertEqual(ww.freshness("abc", "restated_zon")["stale"], False)
            r = ww.freshness("old", "restated_zon")
            self.assertTrue(r["stale"])
            self.assertIn("Rebuild", r["note"])
            r = ww.freshness(None, "restated_zon")
            self.assertIsNone(r["stale"])
            self.assertFalse(r["checked"])

    def test_the_fingerprint_is_stable_and_depends_on_the_predictor(self):                      # BY CONSTRUCTION
        from sigma_ground.field.interface import reduced_model as rm
        a = rm.model_fingerprint("restated_zon")
        self.assertEqual(a, rm.model_fingerprint("restated_zon"))
        self.assertNotEqual(a, rm.model_fingerprint("de440_stack"))

    def test_the_fingerprint_changes_when_a_body_parameter_changes(self):                       # BUG if a constants edit left a table looking current
        import dataclasses
        from sigma_ground.field.interface import reduced_model as rm
        from sigma_ground.field.interface import rolling_shootout as rs
        before = rm.model_fingerprint("restated_zon")
        e = rs._BODY_PARAMS["Earth"]
        with mock.patch.dict(rs._BODY_PARAMS, {"Earth": dataclasses.replace(e, j2=e.j2 * 1.001)}):
            self.assertNotEqual(before, rm.model_fingerprint("restated_zon"))
        self.assertEqual(before, rm.model_fingerprint("restated_zon"))


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


@unittest.skipUnless(KERNEL.exists() and (FIX / "forecast_manifest.json").exists(), "DE440s kernel or sealed forecast table not available")
class TestRealForecast(unittest.TestCase):
    """The sealed 30-year table built on 2026-10-27 is a RECORD (its start state is DE440's, 2026-01-01): still readable for audits, no longer served."""
    def setUp(self):
        from sigma_ground.field.interface import where_when
        self.ww = where_when
        self.ww.forget_live()
        self.table, self.info = self.ww._forecast()
        if self.table is None:
            self.skipTest("the forecast table file is not installed on this machine")

    def test_the_record_stays_readable_and_carries_a_fingerprint(self):                          # BY CONSTRUCTION: the record is kept, its freshness is checkable
        self.assertEqual(self.info["predictor"], self.ww.SERVED["predictor"])
        self.assertTrue(self.ww.freshness(self.info.get("model_fingerprint"), self.info["predictor"])["checked"])

    def test_the_live_path_from_the_tables_own_start_reproduces_its_first_days(self):             # BUG if the live integration and the sealed record were different calculations
        from jplephem.spk import SPK
        k = SPK.open(str(KERNEL))
        jd0 = float(self.info["start_jd_tdb"])
        r = self.ww.live_states(jd0 + 2.0, budget_s=60, start_jd=jd0, start_kernel=k, start_kernel_name="DE440")
        self.assertTrue(r["reached"])
        for body in ("Mars", "Moon", "Mercury"):
            ours = r["state"][body][0] - r["state"]["Sun"][0]
            tab = self.ww._table_state(self.table, body, jd0 + 2.0)[0] - self.ww._table_state(self.table, "Sun", jd0 + 2.0)[0]
            self.assertLess(float(np.linalg.norm(ours - tab)), 1e-4, body)                     # 10 cm after two days: Yoshida-6 0.1 d against the record's Forest-Ruth 0.025 d


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
