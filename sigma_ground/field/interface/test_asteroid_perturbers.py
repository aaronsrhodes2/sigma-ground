"""field.interface.asteroid_perturbers: Kepler conversion invariants, a Sun-only propagation against the exact two-body advance, and a round trip.

Ground truth is independent of the code: energy and angular momentum of the two-body orbit, the exact mean-anomaly advance M(t) = M0 + n (t - t0) solved with an
independent Kepler solver written here, and time reversal. No kernel is needed (the planets are switched off).
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.interface import asteroid_perturbers as ap

GM_SUN = 132712440041.93938                  # km^3/s^2 (the IAU/DE440 value; any positive number serves these tests)
ZERO = lambda jd: (np.zeros(3), np.zeros(3))


def _independent_kepler(p: ap.Perturber, jd: float) -> np.ndarray:
    """Heliocentric ecliptic position by solving Kepler's equation with plain bisection (independent of the module's Newton iteration)."""
    a = p.a_au * ap.AU_KM
    n = math.sqrt(GM_SUN / a ** 3) * ap.DAY_S                     # rad/day
    M = (math.radians(p.ma_deg) + n * (jd - p.epoch_jd)) % (2 * math.pi)
    lo, hi = 0.0, 2 * math.pi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - p.e * math.sin(mid) < M:
            lo = mid
        else:
            hi = mid
    E = 0.5 * (lo + hi)
    xp, yp = a * (math.cos(E) - p.e), a * math.sqrt(1 - p.e ** 2) * math.sin(E)
    Om, w, i = math.radians(p.om_deg), math.radians(p.w_deg), math.radians(p.i_deg)
    x = (math.cos(w) * math.cos(Om) - math.sin(w) * math.sin(Om) * math.cos(i)) * xp + (-math.sin(w) * math.cos(Om) - math.cos(w) * math.sin(Om) * math.cos(i)) * yp
    y = (math.cos(w) * math.sin(Om) + math.sin(w) * math.cos(Om) * math.cos(i)) * xp + (-math.sin(w) * math.sin(Om) + math.cos(w) * math.cos(Om) * math.cos(i)) * yp
    z = (math.sin(w) * math.sin(i)) * xp + (math.cos(w) * math.sin(i)) * yp
    return ap._ECL_TO_EQ @ np.array([x, y, z])


class TestFixture(unittest.TestCase):
    def test_seven_perturbers_with_measured_masses(self):
        ps = ap.load_perturbers()
        self.assertEqual([p.name for p in ps], list(ap.DEFAULT_PERTURBERS))
        ceres = ps[0]
        self.assertAlmostEqual(ceres.gm_km3_s2, 62.6284, delta=1e-4)
        self.assertTrue(all(p.gm_km3_s2 > 0 for p in ps))

    def test_an_asteroid_without_a_measured_mass_is_rejected(self):
        with self.assertRaises(ValueError):
            ap.load_perturbers(("Bennu",))


class TestKeplerConversion(unittest.TestCase):
    def test_energy_and_angular_momentum_match_the_elements(self):          # BUG: invariants of the two-body orbit
        for p in ap.load_perturbers():
            r, v = ap.kepler_state_ecliptic(p, GM_SUN)
            a = p.a_au * ap.AU_KM
            energy = 0.5 * float(v @ v) - GM_SUN / float(np.linalg.norm(r))
            self.assertAlmostEqual(energy / (-GM_SUN / (2 * a)), 1.0, delta=1e-10, msg=p.name)
            h = float(np.linalg.norm(np.cross(r, v)))
            self.assertAlmostEqual(h / math.sqrt(GM_SUN * a * (1 - p.e ** 2)), 1.0, delta=1e-10, msg=p.name)
            rn = float(np.linalg.norm(r))
            self.assertTrue(a * (1 - p.e) * (1 - 1e-9) <= rn <= a * (1 + p.e) * (1 + 1e-9), msg=p.name)


class TestPropagation(unittest.TestCase):
    def setUp(self):
        self.ceres = ap.load_perturbers(("Ceres",))[0]
        self.sun_only = lambda jd: [(GM_SUN, np.zeros(3))]

    def test_sun_only_propagation_matches_the_exact_two_body_advance(self):   # BUG: 10 yr back and forward, independent Kepler solver
        targets = [self.ceres.epoch_jd - 3652.5, self.ceres.epoch_jd - 1000.0, self.ceres.epoch_jd + 1500.0]
        out = ap.propagate_to(self.ceres, targets, self.sun_only, ZERO, GM_SUN)
        for jd in targets:
            truth = _independent_kepler(self.ceres, jd)
            self.assertLess(float(np.linalg.norm(np.array(out[jd][:3]) - truth)), 1.0, msg=f"jd offset {jd - self.ceres.epoch_jd:.0f}")   # km

    def test_round_trip_in_time(self):                                         # BUG: back 15 yr and forward again recovers the start
        back = ap.propagate_to(self.ceres, [self.ceres.epoch_jd - 5478.0], self.sun_only, ZERO, GM_SUN)[self.ceres.epoch_jd - 5478.0]
        r0, v0 = ap.kepler_state_ecliptic(self.ceres, GM_SUN)
        y0 = np.concatenate([ap._ECL_TO_EQ @ r0, (ap._ECL_TO_EQ @ v0) * ap.DAY_S])
        # the state at the epoch, integrated backward then forward by the same routine, must return to y0
        from sigma_ground.kernel.ode import dopri5

        def f(t, y):
            r = np.array(y[:3])
            a = -GM_SUN * ap.DAY_S ** 2 * r / np.linalg.norm(r) ** 3
            return [y[3], y[4], y[5], a[0], a[1], a[2]]
        t0 = self.ceres.epoch_jd
        b = dopri5(f, (t0, t0 - 5478.0), list(y0), rtol=1e-11, atol=1e-6).y[-1]
        fwd = dopri5(f, (t0 - 5478.0, t0), list(b), rtol=1e-11, atol=1e-6).y[-1]
        # rtol 1e-11 on a 4e8 km orbit allows ~4e-3 km per unit of relative error; 15 yr of steps accumulates to ~1 km (2e-9 of the orbit)
        self.assertLess(float(np.linalg.norm(np.array(fwd[:3]) - y0[:3])), 2.0)     # km
        self.assertLess(float(np.linalg.norm(np.array(back[:3]) - np.array(b[:3]))), 1e-3)

    def test_the_epoch_itself_is_returned_unchanged(self):                     # BY CONSTRUCTION
        out = ap.propagate_to(self.ceres, [self.ceres.epoch_jd], self.sun_only, ZERO, GM_SUN)
        r, _ = ap.kepler_state_ecliptic(self.ceres, GM_SUN)
        self.assertLess(float(np.linalg.norm(np.array(out[self.ceres.epoch_jd][:3]) - ap._ECL_TO_EQ @ r)), 1e-6)


if __name__ == "__main__":
    unittest.main()
