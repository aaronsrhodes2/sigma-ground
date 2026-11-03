"""scripts/orbit_fit.py (misc/CANDIDATE_LEDGER.md C28): a synthetic orbit is recovered to the noise, and the finite-difference partials agree with half-step partials.

The synthetic case uses the same engine as the fit (DE440s planets, 1PN, no asteroid sources for speed), a real station (W68) and 60 observation instants over 120 days around the
epoch; the 'truth' is JPL's Eros state at 2020-01-01 (any bound orbit would do), the observations carry 0.3 arcsec Gaussian noise, and the fit starts 300 km and 0.05 km/day away.
Needs D:/datasets/jpl/de440s.bsp and the MPC ObsCodes file; skipped without them.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import orbit_fit as of  # noqa: E402
import mpc_live_score as mls  # noqa: E402
import eros_sky_residuals as esr  # noqa: E402

KERNEL = Path("D:/datasets/jpl/de440s.bsp")
CODES = mls.MPC_OLD / "ObsCodes.html"
EROS_VEC = mls.MPC_OLD / "eros" / "vec_2019-12-31_2022-01-02.json"
HAVE = KERNEL.exists() and CODES.exists() and EROS_VEC.exists()


def synthetic_rows(n=100, span_days=365.0):
    """n instants spread over span_days after the epoch, alternating between two real stations (ATLAS Chile W68, Catalina G96)."""
    import erfa
    rows = []
    for i in range(n):
        y, m, dd, fd = erfa.jd2cal(2458849.5 + 1.5 + i * span_days / n, 0.0)
        rows.append((int(y), int(m), float(dd) + float(fd), 0.0, 0.0, "W68" if i % 2 else "G96"))
    return rows


@unittest.skipUnless(HAVE, "DE440s kernel, ObsCodes and the Eros vectors on D: are needed")
class TestOrbitFitSynthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = mls.Engine(lambda m: None, asteroids=False)
        cls.codes = esr.parse_obscodes(CODES)
        cls.y_true, _ = of.jpl_state_at(433, mls.JD_EROS)
        cls.obs = esr.Obs(synthetic_rows(), cls.codes, cls.engine.kernel)
        cls.fn = staticmethod(lambda y, T: cls.engine.positions(433, mls.JD_EROS, np.asarray(y), np.asarray(T, float)))
        rng = np.random.default_rng(7)
        T = of.light_time_fixed(lambda TT: cls.fn(cls.y_true, TT), cls.obs)
        of.synthesize(cls.obs, cls.fn(cls.y_true, T), 0.3, rng)

    def test_true_state_is_recovered_to_the_noise(self):
        y0 = self.y_true + np.array([300.0, -200.0, 100.0, 0.05, -0.03, 0.02])
        fit = of.fit_state(self.fn, self.obs, y0, 0.3, log=lambda m: None)
        y = np.array(fit["state_km_kmday"])
        self.assertTrue(fit["converged"])
        self.assertGreaterEqual(fit["n_kept"], self.obs.n - 2)                 # 0.3-arcsec noise reaches the 3-sigma gate (0.9 arcsec total) in about 1 draw of 100
        self.assertAlmostEqual(fit["rms_arcsec"], 0.3 * math.sqrt(2), delta=0.3 * math.sqrt(2) * 0.2)     # total residual of two 0.3-arcsec components
        pulls = (y - self.y_true) / np.array(fit["sigma_formal"])
        self.assertTrue(np.all(np.abs(pulls) < 3.0), pulls)
        self.assertLess(np.linalg.norm((y - self.y_true)[:3]), np.linalg.norm(y0[:3] - self.y_true[:3]))     # nearer than the 374 km it started from

    def test_partials_agree_with_half_step(self):
        T = of.light_time_fixed(lambda TT: self.fn(self.y_true, TT), self.obs)
        _, J1 = of.partials(self.fn, self.obs, self.y_true, T, of.STEPS)
        _, J2 = of.partials(self.fn, self.obs, self.y_true, T, of.STEPS / 2)
        for k in range(6):
            scale = np.abs(J1[:, k]).max()
            self.assertLess(np.abs(J1[:, k] - J2[:, k]).max() / scale, 0.01, f"column {k}")

    def test_osculating_elements_round_trip_a_known_orbit(self):
        gm = 1.32712440041279419e11                                          # km^3/s^2: the DE440 header's GMS (fixtures/de440_header_constants.json, au^3/day^2) in km and s; only the scale matters here
        a_km = 1.458 * esr.AU_KM
        r = np.array([a_km * (1 - 0.2228), 0.0, 0.0])                      # perihelion of an e = 0.2228 orbit in the x-y plane
        v = np.array([0.0, math.sqrt(gm * (1 + 0.2228) / (a_km * (1 - 0.2228))), 0.0])
        el = of.osculating_elements(r, v, gm)
        self.assertAlmostEqual(el["a_au"], 1.458, places=9)
        self.assertAlmostEqual(el["e"], 0.2228, places=9)
        self.assertAlmostEqual(el["i_deg"], 0.0, places=9)
        self.assertAlmostEqual(el["ma_deg"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
