"""comet_orbits: the Marsden law, the Sun's 1PN term and the propagator, against closed forms.

Ground truth that does not come from the code under test: (1) the exact two-body (Kepler) solution, (2) the textbook 1PN perihelion advance 6 pi GM / (c^2 a (1 - e^2)) per orbit,
(3) the Marsden function evaluated by hand at r = r0 (g = alpha 2^-k) and the geometry of the R, T, N triad at a state where it is obvious.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.interface import comet_orbits as co

GM_SUN = 1.32712440041279419e11
AU = co.AU_KM
DAY = co.DAY_S


class _Seg:
    def __init__(self, p):
        self.p = np.array(p, dtype=float)

    def compute(self, jd):
        return self.p

    def compute_and_differentiate(self, jd):
        return self.p, np.zeros(3)


class _SunOnly:
    def __getitem__(self, key):
        return _Seg([0.0, 0.0, 0.0] if key == (0, 10) else [1e13, 0.0, 0.0])


GM_SUN_ONLY = [GM_SUN] + [0.0] * 9


def _orbit(a_au, e):
    a = a_au * AU
    rp = a * (1 - e)
    vp = math.sqrt(GM_SUN * (1 + e) / rp) * DAY
    return [rp, 0.0, 0.0, 0.0, vp, 0.0], 2 * math.pi * math.sqrt(a ** 3 / GM_SUN) / DAY


class TestMarsden(unittest.TestCase):
    def test_g_at_r0_is_alpha_times_two_to_minus_k(self):                           # BUG: hand evaluation of the law at x = 1
        self.assertAlmostEqual(co.marsden_g(2.808), 0.1112620426 * 2.0 ** (-4.6142), delta=1e-12)

    def test_g_at_one_au(self):                                                      # BUG: hand evaluation, x = 1/2.808
        x = 1.0 / 2.808
        self.assertAlmostEqual(co.marsden_g(1.0), 0.1112620426 * x ** -2.15 * (1 + x ** 5.093) ** -4.6142, delta=1e-12)

    def test_triad_geometry(self):                                                   # BUG: r along +x, motion along +y, so R = +x, T = +y, N = +z (prograde in the xy plane)
        r = np.array([AU, 0.0, 0.0])
        v = np.array([0.0, 30.0 * DAY, 0.0])
        g = co.marsden_g(1.0) * AU
        for A, axis in (((1e-8, 0, 0), 0), ((0, 1e-8, 0), 1), ((0, 0, 1e-8), 2)):
            a = co.nongrav_acceleration(r, v, A, None)
            self.assertAlmostEqual(a[axis] / (g * 1e-8), 1.0, delta=1e-12)
            self.assertLess(abs(np.delete(a, axis)).max(), 1e-12 * abs(a[axis]))

    def test_retrograde_motion_flips_the_normal(self):                               # BUG: N = R x v / |R x v|
        r = np.array([AU, 0.0, 0.0])
        a = co.nongrav_acceleration(r, np.array([0.0, -30.0 * DAY, 0.0]), (0, 0, 1e-8), None)
        self.assertLess(a[2], 0.0)


class TestPropagation(unittest.TestCase):
    def test_kepler_orbit_returns_after_whole_periods(self):                          # BUG: three periods of an e = 0.85 comet orbit
        y0, period = _orbit(3.0, 0.85)
        out = co.propagate(y0, 2455197.5, [2455197.5 + 3 * period], _SunOnly(), GM_SUN_ONLY, gr=False)
        y = out[2455197.5 + 3 * period]
        self.assertLess(np.linalg.norm(y[:3] - np.array(y0[:3])), 0.3)               # km (0.12 measured), against a perihelion distance of 4.5e7 km
        self.assertLess(np.linalg.norm(y[3:] - np.array(y0[3:])), 0.02)             # km/day: the position error times the perihelion angular rate (0.116 /day) is 0.014

    def test_1pn_perihelion_advance(self):                                            # BUG: the textbook advance per orbit
        a_au, e = 0.4, 0.5
        y0, period = _orbit(a_au, e)

        def periapsis_angle(y):
            r, v = np.array(y[:3]), np.array(y[3:]) / DAY
            ev = ((v @ v) / GM_SUN - 1.0 / np.linalg.norm(r)) * r - (r @ v) / GM_SUN * v
            return math.atan2(ev[1], ev[0])
        n_orb = 6
        t1 = 2455197.5 + n_orb * period
        with_gr = periapsis_angle(co.propagate(y0, 2455197.5, [t1], _SunOnly(), GM_SUN_ONLY, gr=True)[t1])
        without = periapsis_angle(co.propagate(y0, 2455197.5, [t1], _SunOnly(), GM_SUN_ONLY, gr=False)[t1])
        a = a_au * AU
        want = n_orb * 6 * math.pi * GM_SUN / (co.C_KM_S ** 2 * a * (1 - e ** 2))
        self.assertAlmostEqual((with_gr - without) / want, 1.0, delta=0.02)

    def test_backward_propagation_undoes_forward(self):                                # BY CONSTRUCTION: time reversal of the conservative force
        y0, period = _orbit(2.5, 0.6)
        t0 = 2455197.5
        fwd = co.propagate(y0, t0, [t0 + 400.0], _SunOnly(), GM_SUN_ONLY, gr=True)[t0 + 400.0]
        back = co.propagate(list(fwd), t0 + 400.0, [t0], _SunOnly(), GM_SUN_ONLY, gr=True)[t0]
        self.assertLess(np.linalg.norm(back[:3] - np.array(y0[:3])), 1.0)             # km

    def test_nongravitational_term_changes_the_orbit_in_the_expected_sense(self):      # BUG: a prograde transverse push (A2 > 0) raises the orbital energy
        y0, period = _orbit(3.0, 0.3)
        t1 = 2455197.5 + 2 * period

        def energy(y):
            r, v = np.array(y[:3]), np.array(y[3:]) / DAY
            return 0.5 * v @ v - GM_SUN / np.linalg.norm(r)
        base = co.propagate(y0, 2455197.5, [t1], _SunOnly(), GM_SUN_ONLY, gr=False)[t1]
        push = co.propagate(y0, 2455197.5, [t1], _SunOnly(), GM_SUN_ONLY, gr=False, nongrav=(0.0, 1e-8, 0.0))[t1]
        self.assertGreater(energy(push), energy(base))


if __name__ == "__main__":
    unittest.main()
