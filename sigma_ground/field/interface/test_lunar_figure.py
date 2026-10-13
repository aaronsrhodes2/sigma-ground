"""field.interface.lunar_figure: the quadrupole force against independent closed forms, Newton's third law, and the zero-C22 limit.

Ground truth that does not come from the code under test: (1) the engine's own axisymmetric J2 acceleration (Vallado 2013, s9.4 form, already tested against the zonal block),
(2) the spherical-harmonic potential of a sectoral C22 term, V = -(GM R^2 / r^3) 3 C22 cos(2 lambda) cos^2(phi), differentiated by hand on the long axis and on the intermediate axis,
(3) momentum conservation, (4) C22 = 0 reproducing the base system exactly.
"""
from __future__ import annotations

import unittest

import numpy as np

from sigma_ground.field.constants import G
from sigma_ground.field.interface import lunar_figure as lf
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

M_MOON = 7.342e22
R = 1.738e6
GM = G * M_MOON


class TestQuadrupoleForce(unittest.TestCase):
    def test_axisymmetric_limit_equals_the_engine_j2_acceleration(self):         # BUG
        rng = np.random.default_rng(3)
        n = np.array([0.3, -0.2, 0.93]); n /= np.linalg.norm(n)
        j2 = 2.034e-4
        T = lf.axisymmetric_T(M_MOON, R, j2, n)
        for _ in range(5):
            r = rng.normal(size=3); r *= 4e8 / np.linalg.norm(r)
            r2, rr = float(r @ r), float(np.linalg.norm(r))
            c2 = float(r @ n) ** 2 / r2
            ref = 1.5 * GM * j2 * R ** 2 / (r2 * r2 * rr) * ((5 * c2 - 1) * r - 2 * float(r @ n) * n)
            self.assertLess(np.linalg.norm(lf.quadrupole_acceleration(r, T) - ref) / np.linalg.norm(ref), 1e-12)

    def test_c22_on_the_long_axis_matches_the_spherical_harmonic_derivative(self):   # BUG: a = -9 G M C22 R^2 / d^4 toward the Moon
        c22 = 2.2382e-5
        x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        T = lf.c22_T(M_MOON, R, c22, x, y)
        d = 3.844e8
        a = lf.quadrupole_acceleration(np.array([d, 0, 0]), T)
        self.assertAlmostEqual(a[0] / (-9 * GM * c22 * R ** 2 / d ** 4), 1.0, delta=1e-12)
        self.assertAlmostEqual(a[1], 0.0, delta=1e-20)

    def test_c22_on_the_intermediate_axis_has_the_opposite_sign(self):               # BUG: lambda = 90 deg: cos 2 lambda = -1, so the pull is weaker: a = +9 G M C22 R^2 / d^4 (outward)
        c22 = 2.2382e-5
        x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        T = lf.c22_T(M_MOON, R, c22, x, y)
        d = 3.844e8
        a = lf.quadrupole_acceleration(np.array([0, d, 0]), T)
        self.assertAlmostEqual(a[1] / (9 * GM * c22 * R ** 2 / d ** 4), 1.0, delta=1e-12)

    def test_the_c22_force_vanishes_on_the_spin_axis_to_first_order_in_the_quadrupole_direction(self):   # BUG: on the pole the sectoral term is zero by symmetry
        T = lf.c22_T(M_MOON, R, 2.2382e-5, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        a = lf.quadrupole_acceleration(np.array([0, 0, 3.844e8]), T)
        self.assertLess(np.linalg.norm(a), 1e-25)

    def test_the_tensor_is_traceless(self):                                      # BY CONSTRUCTION
        T = lf.c22_T(M_MOON, R, 2.2382e-5, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        self.assertAlmostEqual(float(np.trace(T)), 0.0, delta=1e-6)


def _earth_moon(moon_pole=(0.0, 0.0, 1.0)):
    earth = CelestialBody(5.972e24, np.zeros(3), np.zeros(3), 6.371e6, 0.0)
    moon = CelestialBody(M_MOON, np.array([3.844e8, 0.0, 0.0]), np.array([0.0, 1.022e3, 0.0]), R, 0.0, pole_axis_unit=np.array(moon_pole))
    sun_like = CelestialBody(1.989e30, np.array([1.5e11, 2e10, 3e9]), np.zeros(3), 6.96e8, 0.0)
    return [earth, moon, sun_like]


class TestSystem(unittest.TestCase):
    def test_zero_c22_is_the_base_system(self):                                  # BY CONSTRUCTION
        bodies = _earth_moon()
        base = NBodySystem(bodies, toggles=PhysicsToggles()).compute_accelerations()
        locked = lf.LockedMoonSystem(bodies, toggles=PhysicsToggles(), moon_index=1, earth_index=0, c22=0.0).compute_accelerations()
        self.assertTrue(np.array_equal(base, locked))

    def test_total_momentum_is_conserved_by_the_added_force(self):               # BUG: Newton's third law, tested on the extra force itself (no cancellation against the base)
        bodies = _earth_moon((0.1, 0.2, 0.97))
        extra = lf.c22_extra_accelerations(bodies, moon_index=1, earth_index=0)
        m = np.array([b.mass_kg for b in bodies])
        F = m[:, None] * extra
        self.assertLess(np.linalg.norm(F.sum(0)) / np.abs(F).sum(), 1e-13)

    def test_the_earth_feels_the_pull_of_the_bulge(self):                        # BUG: the Earth lies on the Moon's long axis, so the extra pull on it points toward the Moon (+x)
        bodies = _earth_moon()
        extra = lf.c22_extra_accelerations(bodies, moon_index=1, earth_index=0)
        expected = 9 * GM * lf.C22_MOON * lf.R_REF_MOON_M ** 2 / 3.844e8 ** 4
        self.assertAlmostEqual(extra[0][0] / expected, 1.0, delta=1e-12)
        self.assertGreater(extra[0][0], 0.0)

    def test_the_subclass_adds_exactly_that_force_to_the_base_accelerations(self):   # BUG: tolerance set by round-off of the large base term (~1e-9 relative to the extra)
        bodies = _earth_moon((0.1, 0.2, 0.97))
        base = NBodySystem(bodies, toggles=PhysicsToggles()).compute_accelerations()
        locked = lf.LockedMoonSystem(bodies, toggles=PhysicsToggles(), moon_index=1, earth_index=0).compute_accelerations()
        extra = lf.c22_extra_accelerations(bodies, moon_index=1, earth_index=0)
        self.assertLess(np.linalg.norm(locked - base - extra), 1e-6 * np.linalg.norm(extra[0]) + 1e-18)


if __name__ == "__main__":
    unittest.main()
