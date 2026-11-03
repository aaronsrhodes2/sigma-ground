"""scripts/mpc_live_score.py (misc/CANDIDATE_LEDGER.md C27): the scoring arithmetic reproduces C24's stored Eros residuals, and the incremental bookkeeping selects exactly the new observations.

Ground truth for the arithmetic is misc/EROS_SKY_RESIDUALS.json (C24, 2026-10-30): arm J, JPL's Eros vectors through the pipeline, 4,195 observations, RMS 0.3461 arcsec. The test re-runs that
arm from the raw files on D: (skipped when they are absent) and must reproduce every stored number to 1e-9. The bookkeeping tests use synthetic rows and need no data.
"""
from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import mpc_live_score as m  # noqa: E402
import eros_sky_residuals as esr  # noqa: E402

STORED = ROOT / "misc" / "EROS_SKY_RESIDUALS.json"
OBS = m.MPC_OLD / "obs80_433.txt"
CODES = m.MPC_OLD / "ObsCodes.html"
KERNEL = Path("D:/datasets/jpl/de440s.bsp")
PROBE = Path("D:/datasets/mpc_live/2026-11-02/horizons/probe_433_W68.json")


def row(y, mo, d, code="W68", ra=0.1, dec=0.2):
    return (y, mo, d, ra, dec, code)


class TestC24Reproduction(unittest.TestCase):

    @unittest.skipUnless(OBS.exists() and CODES.exists() and KERNEL.exists() and STORED.exists(), "C24's raw data on D: is needed")
    def test_arm_j_reproduces_the_stored_c24_residuals(self):
        from jplephem.spk import SPK
        stored = json.loads(STORED.read_text(encoding="utf-8"))["J_all"]
        kernel = SPK.open(str(KERNEL))
        rows = esr.parse_obs80(OBS)
        codes = esr.parse_obscodes(CODES)
        obs = esr.Obs([r for r in rows if r[0] >= 2021], codes, kernel)
        self.assertEqual(obs.n, stored["n"])
        tj, pj, vj = esr.load_horizons()
        fn = lambda T: esr.hermite(tj, pj, vj, T)
        T, _ = esr.light_time_targets(fn, obs)
        res = esr.radec_residuals(obs, fn(T))
        got = esr.summarize(res)
        for k in ("rms_total_arcsec", "median_total_arcsec", "mean_dra_cosdec", "mean_ddec", "sem_dra", "sem_ddec"):
            self.assertAlmostEqual(got[k], stored[k], delta=1e-9, msg=k)
        rb = m.robust(res)
        self.assertAlmostEqual(rb["median_arcsec"], stored["median_total_arcsec"], delta=1e-12)
        self.assertAlmostEqual(rb["rms_arcsec"], stored["rms_total_arcsec"], delta=1e-12)
        self.assertGreater(rb["robust_sigma_arcsec"], 0.0)
        self.assertLess(rb["robust_sigma_arcsec"], rb["rms_arcsec"])

    def test_utc_jd_matches_the_pipeline_time_scale(self):
        """The UTC JD sent to Horizons and the pipeline's TT JD differ by exactly TT - UTC (69.184 s since 2017)."""
        rows = [row(2026, 9, 19.025603), row(2026, 1, 1.5)]
        jd = m.utc_jd(rows)
        self.assertAlmostEqual(jd[0], 2461302.525603, delta=1e-9)
        self.assertAlmostEqual(jd[1], 2461042.0, delta=1e-9)
        import erfa
        u1, u2 = erfa.dtf2d("UTC", 2026, 9, 19, 0, 0, 0.0)
        t1, t2 = erfa.taitt(*erfa.utctai(u1, u2 + 0.025603))
        self.assertAlmostEqual((t1 + t2 - jd[0]) * 86400.0, 69.184, delta=1e-4)


class TestBookkeeping(unittest.TestCase):

    def test_first_run_starts_at_the_backfill_date_and_stops_before_the_run_date(self):
        rows = [row(2025, 12, 31.9), row(2026, 1, 1.0), row(2026, 1, 1.5), row(2026, 3, 2.25), row(2026, 11, 1.99), row(2026, 11, 2.0), row(2026, 11, 2.1)]
        sel, times = m.select_new(rows, None, "2026-11-02")
        self.assertEqual([r[2] for r in sel], [1.5, 2.25, 1.99])              # 2026-01-01 00:00 itself is not after the backfill start; 2026-11-02 00:00 is not before the run date
        self.assertEqual(times[0], datetime(2026, 1, 1, 12, tzinfo=timezone.utc))

    def test_watermark_is_exclusive_and_the_second_run_sees_only_newer_rows(self):
        rows = [row(2026, 1, 5.5), row(2026, 1, 9.25), row(2026, 2, 1.0), row(2026, 2, 1.75)]
        sel1, t1 = m.select_new(rows, None, "2026-02-01")
        self.assertEqual(len(sel1), 2)
        watermark = t1[-1].isoformat()                                       # what the scorer stores after a successful row
        sel2, t2 = m.select_new(rows, watermark, "2026-03-01")
        self.assertEqual([r[2] for r in sel2], [1.0, 1.75])
        sel3, _ = m.select_new(rows, t2[-1].isoformat(), "2026-03-01")
        self.assertEqual(sel3, [])

    def test_unsorted_input_comes_out_oldest_first(self):
        rows = [row(2026, 5, 3.0), row(2026, 2, 1.0), row(2026, 4, 1.0)]
        sel, times = m.select_new(rows, None, "2026-06-01")
        self.assertEqual([r[1] for r in sel], [2, 4, 5])
        self.assertEqual(times, sorted(times))

    def test_backfill_windows_are_30_days_of_observation_date_and_a_later_run_is_one_window(self):
        times = [datetime(2026, 1, 2, tzinfo=timezone.utc), datetime(2026, 1, 20, tzinfo=timezone.utc), datetime(2026, 2, 1, tzinfo=timezone.utc),
                 datetime(2026, 2, 15, tzinfo=timezone.utc), datetime(2026, 6, 1, tzinfo=timezone.utc)]
        self.assertEqual(m.windows(times, first_run=True), [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(m.windows(times, first_run=False), [(0, 5)])
        self.assertEqual(m.windows([], first_run=True), [])

    def test_robust_statistics(self):
        res = np.array([[3.0, 4.0], [0.0, 1.0], [0.0, 0.0], [0.0, -2.0], [1.0, 0.0]])
        r = m.robust(res)
        self.assertEqual(r["median_arcsec"], 1.0)                             # totals 5, 1, 0, 2, 1
        self.assertAlmostEqual(r["robust_sigma_arcsec"], 1.4826 * 1.0)         # |tot - 1| = 4, 0, 1, 1, 0 -> MAD 1
        self.assertEqual(r["median_dra_cosdec"], 0.0)
        self.assertEqual(r["median_ddec"], 0.0)


class TestHorizonsRequests(unittest.TestCase):

    def test_observer_url_carries_the_pre_registered_parameters(self):
        u = m.horizons_observer_url(433, "W68", [2461302.525603, 2461302.526504])
        for needle in ("EPHEM_TYPE=%27OBSERVER%27", "CENTER=%27W68%40399%27", "QUANTITIES=%271%27", "TIME_TYPE=%27UT%27", "TLIST_TYPE=%27JD%27", "ANG_FORMAT=%27DEG%27", "EXTRA_PREC=%27YES%27",
                       "CSV_FORMAT=%27YES%27", "COMMAND=%27433%3B%27", "2461302.5256030"):
            self.assertIn(needle, u)
        self.assertLess(len(m.horizons_observer_url(99942, "G96", [2461302.0 + 0.001 * i for i in range(m.CHUNK)])), 8000)

    def test_vectors_url_is_the_asteroid_group_request_with_one_epoch(self):
        u = m.horizons_vectors_url(99942, m.JD_2026)
        self.assertIn("COMMAND=%2799942%3B%27", u)
        self.assertIn("TLIST=%272461041.50%27", u)
        self.assertIn("EPHEM_TYPE=%27VECTORS%27", u)

    @unittest.skipUnless(PROBE.exists(), "the saved probe response on D: is needed")
    def test_probe_response_parses_to_astrometric_positions(self):
        res = json.loads(PROBE.read_text(encoding="utf-8"))["result"]
        jd, ra, dec = m.parse_horizons_observer(res, 433)
        self.assertEqual(len(jd), 2)
        self.assertAlmostEqual(jd[0], 2461302.525603, delta=1e-6)
        self.assertTrue(227.0 < ra[0] < 228.0 and -25.0 < dec[0] < -24.0)
        with self.assertRaises(ValueError):
            m.parse_horizons_observer(res, 1)                                  # the wrong body is refused, never used

    def test_vector_rows_honour_the_file_units(self):
        """C24's Eros vectors are km/s, the asteroid group km/day: both must come out in km/day (a solar-system speed), a missing units line is refused."""
        eros = m.MPC_OLD / "eros" / "vec_2019-12-31_2022-01-02.json"
        ceres = Path("D:/datasets/jpl/horizons_asteroids/horizons_ast_1.json")
        if eros.exists():
            y = m.parse_horizons_vector_row(json.loads(eros.read_text(encoding="utf-8"))["result"], 433, m.JD_EROS)
            self.assertTrue(1e6 < np.linalg.norm(y[3:]) < 3e6, np.linalg.norm(y[3:]))
            tj, pj, vj = esr.load_horizons()
            i0 = int(np.argmin(np.abs(tj - m.JD_EROS)))
            np.testing.assert_allclose(y, np.concatenate([pj[i0], vj[i0]]), rtol=0, atol=1e-9)       # exactly C24's y_jpl
        if ceres.exists():
            y = m.parse_horizons_vector_row(json.loads(ceres.read_text(encoding="utf-8"))["result"], 1, m.JD_2026)
            self.assertTrue(1e6 < np.linalg.norm(y[3:]) < 2e6, np.linalg.norm(y[3:]))
        fake = "Target body name: 1 Ceres (A801 AA)  {source: x}\n$$SOE\n2461041.5, A.D., 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,\n$$EOE\n"
        with self.assertRaises(ValueError):
            m.parse_horizons_vector_row(fake, 1, 2461041.5)

    def test_refused_request_is_split_down_to_the_instants_that_work(self):
        """A stand-in for the day-one C40 case: Horizons refuses any request with more than 3 instants; the scorer must still obtain every instant through halving."""
        import types
        from datetime import datetime, timezone
        n = 10
        times = [datetime(2026, 1, 1, i, tzinfo=timezone.utc) for i in range(n)]
        jd_utc = np.array([2461041.5 + i / 24 for i in range(n)])
        calls = []

        def fake_fetch(url, out, log, check):
            k = url.count("%20") + 1                                            # instants in the TLIST
            calls.append(k)
            if k > 3:
                return None
            tl = url.split("TLIST=%27")[1].split("%27")[0].split("%20")
            res = "Target body name: 433 Eros (A898 PA)  {source: x}\nDate_________JDUT, , , R.A.___(ICRF), DEC____(ICRF),\n$$SOE\n" + "".join(f"{float(t):.9f}, , , 10.0, 20.0,\n" for t in tl) + "$$EOE\n"
            return {"result": res}

        ra_j = np.full(n, np.nan); dec_j = np.full(n, np.nan); files = []
        saved = m.fetch_url
        m.fetch_url = fake_fetch
        try:
            lost = m.fetch_part(433, "C40", list(range(n)), times, jd_utc, ra_j, dec_j, files, Path("D:/datasets/mpc_live/_test"), lambda s: None)
        finally:
            m.fetch_url = saved
        self.assertEqual(lost, [])
        self.assertTrue(np.all(~np.isnan(ra_j)) and np.all(ra_j == 10.0))
        self.assertEqual(calls, [10, 5, 2, 3, 5, 2, 3])                          # 10 refused -> 5 + 5; each 5 refused -> 2 + 3, which are answered
        self.assertEqual(len(files), 4)

    def test_apparent_table_is_refused(self):
        fake = "Target body name: 433 Eros (A898 PA)  {source: x}\nDate_________JDUT, , , R.A._(a-app), DEC_(a-app),\n$$SOE\n2461302.5, , , 1.0, 2.0,\n$$EOE\n"
        with self.assertRaises(ValueError):
            m.parse_horizons_observer(fake, 433)


if __name__ == "__main__":
    unittest.main()
