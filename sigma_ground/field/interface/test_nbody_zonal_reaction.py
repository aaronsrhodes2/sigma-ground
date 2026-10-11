"""Regression sentinels: the J2/J3/J4 zonal layers obey Newton's third law (fixed 2026-10-11).

Before the fix the zonal layers accelerated every body i by the oblate body j's multipole field and applied NO reaction on j, so
the force between planet and moon was not equal and opposite: net force with only the zonal layers on was 2.2e-5 of the total
force scale on the real Jupiter system (1e-17 for Newton), Jupiter received about 10^4 times too little acceleration from its
moons' pull, and Jupiter and Saturn came out WORSE with the zonal layers on in the N-body shoot-out against JPL DE440
(Jupiter 630 km at +5 yr, against 31 km without them) while their moons improved 10-100 fold. See KNOWN_GAPS.md and
sigma_ground/field/interface/fixtures/v2physics/README.md.

The tests use independent ground truth: momentum conservation (a closed-system identity, not the code's own formula), and the
two-body relative acceleration of a mutual force, a_rel = a_on_test_particle x (1 + m_i/m_j).
"""
from __future__ import annotations

import unittest

import numpy as np

from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

ZONALS = PhysicsToggles(j2_zonal=True, j3_zonal=True, j4_zonal=True)
NEWTON = PhysicsToggles()


def _planet(pole=(0.3, -0.2, 0.93)):
    p = np.array(pole, dtype=float)
    return CelestialBody(1.898e27, np.zeros(3), np.zeros(3), 7.1492e7, 0.0,
                         j2=1.4736e-2, j3=-4.2e-8, j4=-5.87e-4, pole_axis_unit=p / np.linalg.norm(p))


def _moon(mass, pos):
    return CelestialBody(mass, np.array(pos, dtype=float), np.array([0.0, 1.0e4, 0.0]), 1.8e6, 0.0)


def _accel(bodies, toggles):
    return NBodySystem(bodies, toggles=toggles).compute_accelerations()


def _net_force_fraction(bodies, toggles):
    a = _accel(bodies, toggles)
    m = np.array([b.mass_kg for b in bodies])
    F = m[:, None] * a
    return float(np.linalg.norm(F.sum(0)) / np.abs(F).sum())


class TestZonalNewtonThirdLaw(unittest.TestCase):

    def test_two_body_zonal_force_is_equal_and_opposite(self):
        bodies = [_planet(), _moon(8.93e22, (2.0e8, 1.3e8, -0.7e8))]
        # zonal part only = (all layers) - (Newton), so Newton's own cancellation drops out
        a_z = _accel(bodies, ZONALS)
        a_n = _accel(bodies, NEWTON)
        m = np.array([b.mass_kg for b in bodies])
        F = m[:, None] * (a_z - a_n)
        self.assertLess(np.linalg.norm(F.sum(0)) / np.abs(F).sum(), 1e-12)

    def test_five_body_system_conserves_momentum_with_zonals(self):
        bodies = [_planet(), _moon(8.93e22, (4.2e8, 0, 0.2e8)), _moon(4.8e22, (-6.7e8, 1e8, -0.4e8)),
                  _moon(1.48e23, (1.07e9, 2e8, 0.3e8)), _moon(1.08e23, (0.3e9, -1.88e9, 0.2e8))]
        a_z = _accel(bodies, ZONALS)
        a_n = _accel(bodies, NEWTON)
        m = np.array([b.mass_kg for b in bodies])
        F = m[:, None] * (a_z - a_n)
        self.assertLess(np.linalg.norm(F.sum(0)) / np.abs(F).sum(), 1e-12)

    def test_relative_acceleration_is_the_mutual_one(self):
        """For two bodies the relative zonal acceleration is a_on_moon x (1 + m_moon/m_planet): the planet recoils."""
        for mass in (1.0e20, 8.93e22, 1.9e26):          # moon, big moon, a comparable-mass companion
            bodies = [_planet(), _moon(mass, (2.0e8, 1.3e8, -0.7e8))]
            d = (_accel(bodies, ZONALS) - _accel(bodies, NEWTON))
            a_moon, a_planet = d[1], d[0]
            ratio = np.linalg.norm(a_planet) / np.linalg.norm(a_moon)
            self.assertAlmostEqual(ratio, mass / bodies[0].mass_kg, delta=1e-9 * mass / bodies[0].mass_kg + 1e-18)
            # opposite sign: the planet is pushed the other way
            self.assertLess(float(np.dot(a_planet, a_moon)), 0.0)

    def test_test_particle_limit_is_unchanged(self):
        """With a negligible moon the moon's acceleration is exactly what the layer gave before (the reaction is tiny)."""
        bodies = [_planet(), _moon(1.0, (2.0e8, 1.3e8, -0.7e8))]
        d = _accel(bodies, PhysicsToggles(j2_zonal=True)) - _accel(bodies, NEWTON)
        # independent J2-only closed form: a = 1.5 GM J2 R^2/r^5 [ (5 c^2 - 1) r - 2 (r.n) n ]  (Vallado 2013 s9.4)
        b = bodies[0]
        r = bodies[1].position_m
        rn = float(r @ b.pole_axis_unit); rr = float(np.linalg.norm(r)); c2 = rn * rn / (rr * rr)
        j2 = 1.5 * b.gm_m3_s2 * b.j2 * b.radius_m ** 2 / rr ** 5 * ((5 * c2 - 1) * r - 2 * rn * b.pole_axis_unit)
        self.assertLess(np.linalg.norm(d[1] - j2) / np.linalg.norm(j2), 1e-9)

    def test_newtonian_path_is_untouched(self):
        bodies = [_planet(), _moon(8.93e22, (2.0e8, 1.3e8, -0.7e8))]
        self.assertLess(_net_force_fraction(bodies, NEWTON), 1e-15)


if __name__ == "__main__":
    unittest.main()
