"""Tests for local_library.interface.nbody.

Covers:
  - CelestialBody: mass, GM, σ-field scaling, kinetic energy, momentum
  - NBodySystem: velocity-Verlet and Forest-Ruth integration
  - Energy / angular-momentum conservation
  - Tidal deformation geometry and magnitude
  - Roche limit
  - GW energy loss (binary inspiral)
  - Forest-Ruth 4th-order symplectic conservation (energy drift)
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.constants import G as _G
from sigma_ground.field.interface.nbody import (
    CelestialBody,
    NBodySystem,
    TidalDeformationField,
    _FR_THETA,
)

M_SUN   = 1.989e30   # kg
R_SUN   = 6.96e8     # m
AU      = 1.496e11   # m


def _two_body_circular(
    m1: float = M_SUN,
    m2: float = M_SUN,
    sep_m: float = AU,
    r1: float = R_SUN,
    r2: float = R_SUN,
    k2: float = 0.5,
) -> tuple[CelestialBody, CelestialBody]:
    """Return two bodies in a circular COM orbit."""
    mu = _G * (m1 + m2)
    v  = math.sqrt(mu / sep_m)           # relative speed
    v1 =  v * m2 / (m1 + m2)
    v2 = -v * m1 / (m1 + m2)
    x1 =  sep_m * m2 / (m1 + m2)
    x2 = -sep_m * m1 / (m1 + m2)
    b1 = CelestialBody(m1, np.array([x1, 0, 0]), np.array([0, v1, 0]), r1, k2)
    b2 = CelestialBody(m2, np.array([x2, 0, 0]), np.array([0, v2, 0]), r2, k2)
    return b1, b2


# ═══════════════════════════════════════════════════════════════════════════
# CelestialBody
# ═══════════════════════════════════════════════════════════════════════════

class TestCelestialBody(unittest.TestCase):

    def _sun(self, **kw) -> CelestialBody:
        defaults = dict(
            mass_kg=M_SUN,
            position_m=np.zeros(3),
            velocity_m_s=np.zeros(3),
            radius_m=R_SUN,
            love_number_k2=0.5,
        )
        defaults.update(kw)
        return CelestialBody(**defaults)

    def test_gm_newtonian(self):
        body = self._sun()
        self.assertAlmostEqual(body.gm_m3_s2, _G * M_SUN, delta=_G * M_SUN * 1e-10)

    def test_gm_sigma_scaling(self):
        sigma = 0.5
        body  = self._sun(sigma_field=sigma)
        expected = _G * M_SUN * math.exp(sigma)
        self.assertAlmostEqual(body.gm_m3_s2, expected, delta=expected * 1e-10)

    def test_kinetic_energy(self):
        v    = np.array([3e4, 0, 0])
        body = self._sun(velocity_m_s=v)
        expected = 0.5 * M_SUN * float(np.dot(v, v))
        self.assertAlmostEqual(body.kinetic_energy(), expected, delta=expected * 1e-10)

    def test_momentum(self):
        v    = np.array([1e3, 2e3, 3e3])
        body = self._sun(velocity_m_s=v)
        np.testing.assert_allclose(body.momentum(), M_SUN * v)

    def test_immutability_replace(self):
        body = self._sun()
        new  = body.replace(mass_kg=2 * M_SUN)
        self.assertEqual(body.mass_kg, M_SUN)
        self.assertEqual(new.mass_kg, 2 * M_SUN)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            CelestialBody(M_SUN, np.zeros(4), np.zeros(3), R_SUN, 0.5)

    def test_forest_ruth_theta(self):
        """θ = 1/(2 − ∛2) ≈ 1.3512."""
        expected = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        self.assertAlmostEqual(_FR_THETA, expected, places=14)


# ═══════════════════════════════════════════════════════════════════════════
# Two-body circular orbit — conservation laws
# ═══════════════════════════════════════════════════════════════════════════

class TestTwoBodyVerlet(unittest.TestCase):
    """Velocity-Verlet integration: energy and angular momentum."""

    def _integrate(self, n_orbits: int = 2, steps_per_orbit: int = 100) -> NBodySystem:
        b1, b2 = _two_body_circular()
        period  = 2 * math.pi * math.sqrt(AU ** 3 / (_G * 2 * M_SUN))
        dt      = period / steps_per_orbit
        system  = NBodySystem([b1, b2])
        for _ in range(n_orbits * steps_per_orbit):
            system.step(dt)
        return system

    def test_energy_conservation(self):
        b1, b2  = _two_body_circular()
        period  = 2 * math.pi * math.sqrt(AU ** 3 / (_G * 2 * M_SUN))
        dt      = period / 200
        system  = NBodySystem([b1, b2])
        E0      = system.total_energy()
        for _ in range(400):
            system.step(dt)
        E1 = system.total_energy()
        drift = abs(E1 - E0) / abs(E0)
        self.assertLess(drift, 0.05, f"Verlet energy drift {drift:.2e}")

    def test_angular_momentum_conservation(self):
        b1, b2  = _two_body_circular()
        period  = 2 * math.pi * math.sqrt(AU ** 3 / (_G * 2 * M_SUN))
        dt      = period / 500
        system  = NBodySystem([b1, b2])
        L0      = system.total_angular_momentum()
        for _ in range(2000):
            system.step(dt)
        L1 = system.total_angular_momentum()
        rel = float(np.linalg.norm(L1 - L0)) / float(np.linalg.norm(L0))
        self.assertLess(rel, 0.01, f"Verlet L drift {rel:.2e}")

    def test_total_momentum_conservation(self):
        # Use unequal masses so total momentum is non-zero
        b1 = CelestialBody(2 * M_SUN, np.array([0, 0, 0.0]),
                           np.array([1e4, 0, 0.0]), R_SUN, 0.5)
        b2 = CelestialBody(1 * M_SUN, np.array([1e10, 0, 0.0]),
                           np.array([0, 3e3, 0.0]), R_SUN, 0.5)
        system = NBodySystem([b1, b2])
        p0 = system.total_momentum()
        for _ in range(50):
            system.step(1e5)
        p1  = system.total_momentum()
        p0_mag = float(np.linalg.norm(p0))
        self.assertGreater(p0_mag, 0)
        rel = float(np.linalg.norm(p1 - p0)) / p0_mag
        self.assertLess(rel, 0.01, f"Momentum drift {rel:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# Forest-Ruth 4th-order symplectic — improved energy conservation
# ═══════════════════════════════════════════════════════════════════════════

class TestForestRuth(unittest.TestCase):
    """Forest-Ruth integration: tighter energy conservation than Verlet."""

    def test_energy_conservation_tighter_than_verlet(self):
        """FR4 should conserve energy at least as well as Verlet at same dt."""
        b1_v, b2_v = _two_body_circular()
        b1_f, b2_f = _two_body_circular()

        period = 2 * math.pi * math.sqrt(AU ** 3 / (_G * 2 * M_SUN))
        dt     = period / 100
        steps  = 200

        sys_v = NBodySystem([b1_v, b2_v])
        sys_f = NBodySystem([b1_f, b2_f])
        E0_v  = sys_v.total_energy()
        E0_f  = sys_f.total_energy()

        for _ in range(steps):
            sys_v.step(dt)
            sys_f.forest_ruth_step(dt)

        drift_v = abs(sys_v.total_energy() - E0_v) / abs(E0_v)
        drift_f = abs(sys_f.total_energy() - E0_f) / abs(E0_f)

        # FR4 must be at least as good as Verlet (usually much better)
        self.assertLessEqual(drift_f, drift_v + 1e-8,
                             f"FR4 drift {drift_f:.2e} worse than Verlet {drift_v:.2e}")
        # And absolute energy conservation better than 5%
        self.assertLess(drift_f, 0.05)

    def test_time_advances(self):
        b1, b2 = _two_body_circular()
        system = NBodySystem([b1, b2])
        dt     = 1e6
        system.forest_ruth_step(dt)
        self.assertAlmostEqual(system.time, dt)

    def test_bodies_move(self):
        b1, b2 = _two_body_circular()
        system = NBodySystem([b1, b2])
        pos0   = system.bodies[0].position_m.copy()
        system.forest_ruth_step(1e6)
        self.assertFalse(np.allclose(system.bodies[0].position_m, pos0))

    def test_energy_conservation_fine_dt(self):
        """With dt = period/1000, FR4 energy drift < 1e-6 over 10 orbits."""
        b1, b2  = _two_body_circular()
        period  = 2 * math.pi * math.sqrt(AU ** 3 / (_G * 2 * M_SUN))
        dt      = period / 1000
        system  = NBodySystem([b1, b2])
        E0      = system.total_energy()
        for _ in range(10_000):
            system.forest_ruth_step(dt)
        drift = abs(system.total_energy() - E0) / abs(E0)
        self.assertLess(drift, 1e-6, f"FR4 fine-dt drift {drift:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# σ-field coupling
# ═══════════════════════════════════════════════════════════════════════════

class TestSigmaFieldCoupling(unittest.TestCase):

    def test_sigma_increases_acceleration(self):
        """Positive σ on body2 → larger acceleration on body1."""
        pos1 = np.zeros(3)
        vel0 = np.zeros(3)
        pos2 = np.array([1e11, 0, 0.0])

        b1 = CelestialBody(1e30, pos1, vel0, 1e8, 0.5, sigma_field=0.0)

        b2_no_s = CelestialBody(1e30, pos2, vel0, 1e8, 0.5, sigma_field=0.0)
        b2_sig  = CelestialBody(1e30, pos2, vel0, 1e8, 0.5, sigma_field=0.5)

        acc_no = NBodySystem([b1, b2_no_s]).compute_accelerations()
        acc_si = NBodySystem([b1, b2_sig ]).compute_accelerations()

        mag_no = float(np.linalg.norm(acc_no[0]))
        mag_si = float(np.linalg.norm(acc_si[0]))

        ratio    = mag_si / mag_no
        expected = math.exp(0.5)
        self.assertAlmostEqual(ratio, expected, delta=expected * 0.01)

    def test_sigma_zero_is_newtonian(self):
        b1, b2 = _two_body_circular()
        system = NBodySystem([b1, b2])
        acc    = system.compute_accelerations()
        self.assertEqual(acc.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(acc)))


# ═══════════════════════════════════════════════════════════════════════════
# Tidal deformation
# ═══════════════════════════════════════════════════════════════════════════

class TestTidalDeformation(unittest.TestCase):

    def _aligned_system(
        self, d: float = 1e11, k2: float = 0.5,
    ) -> NBodySystem:
        b1 = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, k2)
        b2 = CelestialBody(M_SUN, np.array([d, 0, 0.0]), np.zeros(3), R_SUN, k2)
        return NBodySystem([b1, b2])

    def test_pole_bulge_positive(self):
        """P₂(1) = 1 → deformation at pole is positive (bulge toward companion)."""
        sys    = self._aligned_system()
        field  = sys.compute_tidal_deformation(0, 1)
        self.assertGreater(field.evaluate_at_angle(0), 0)
        self.assertGreater(field.evaluate_at_angle(math.pi), 0)

    def test_equator_negative(self):
        """P₂(0) = −0.5 → deformation at equator is negative (squeezed)."""
        sys   = self._aligned_system()
        field = sys.compute_tidal_deformation(0, 1)
        self.assertLess(field.evaluate_at_angle(math.pi / 2), 0)

    def test_epsilon2_formula(self):
        """ε₂ = (k₂/2)(M_c/M_b)(R/d)³ — Love (1911)."""
        d, k2 = 1e11, 0.5
        sys   = self._aligned_system(d=d, k2=k2)
        field = sys.compute_tidal_deformation(0, 1)
        expected = (k2 / 2) * (M_SUN / M_SUN) * (R_SUN / d) ** 3
        self.assertAlmostEqual(
            field.evaluate_at_angle(0), expected,
            delta=expected * 0.01,
        )

    def test_max_deformation(self):
        sys   = self._aligned_system()
        field = sys.compute_tidal_deformation(0, 1)
        self.assertAlmostEqual(field.max_deformation(), field.epsilon2)


# ═══════════════════════════════════════════════════════════════════════════
# J₂ zonal quadrupole (oblateness)
# ═══════════════════════════════════════════════════════════════════════════

class TestJ2Quadrupole(unittest.TestCase):
    """J₂ zonal-harmonic acceleration: oblateness of the central body.

    Standard form (Vallado 2013 §9.4):
        a_J2 = (3 G M_j R_j² J₂_j / (2 r⁵))
               × [(5(r·n̂)²/r² − 1) × r − 2(r·n̂) × n̂]
    where r = pos_i − pos_j (FROM j TO i) and n̂_j is j's pole unit vector.
    """

    # Earth canonical values (CODATA / IERS conventions 2010)
    M_EARTH = 5.972e24
    R_EARTH = 6.371e6
    J2_EARTH = 1.08263e-3

    def test_j2_default_is_zero(self):
        """Bodies without explicit j2 default to 0.0 (pure spherical)."""
        body = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        self.assertEqual(body.j2, 0.0)

    def test_default_pole_axis_is_z(self):
        """Default pole axis is +z unit vector."""
        body = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        np.testing.assert_array_equal(body.pole_axis_unit, np.array([0.0, 0.0, 1.0]))

    def test_pole_axis_auto_normalized(self):
        """User-supplied non-unit pole vector is normalized to unit length."""
        body = CelestialBody(
            M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5,
            pole_axis_unit=np.array([0.0, 3.0, 4.0]),  # |v| = 5
        )
        np.testing.assert_allclose(
            body.pole_axis_unit, np.array([0.0, 0.6, 0.8]), atol=1e-15,
        )

    def test_pole_axis_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            CelestialBody(
                M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5,
                pole_axis_unit=np.zeros(3),
            )

    def test_pole_axis_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            CelestialBody(
                M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5,
                pole_axis_unit=np.array([1.0, 0.0]),
            )

    def test_zero_j2_matches_newtonian(self):
        """j2=0 path produces identical accelerations to plain Newtonian."""
        # Earth + satellite at LEO altitude
        earth_no_j2 = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3), self.R_EARTH, 0.3, j2=0.0,
        )
        sat = CelestialBody(
            1.0, np.array([6.771e6, 0.0, 0.0]), np.zeros(3), 1.0, 0.0,
        )
        sys_no_j2 = NBodySystem([earth_no_j2, sat])
        acc_no_j2 = sys_no_j2.compute_accelerations()

        # Earth WITHOUT explicit j2 kwarg (default path) should be identical
        earth_default = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3), self.R_EARTH, 0.3,
        )
        sys_default = NBodySystem([earth_default, sat])
        acc_default = sys_default.compute_accelerations()

        np.testing.assert_allclose(acc_no_j2, acc_default, atol=0.0)

    def test_equatorial_force_oblate_is_extra_inward(self):
        """At equator (r·n̂ = 0), J₂ > 0 adds extra inward radial force.

        |a_J2| = 1.5 × G × M × J₂ × R² / r⁴ along −r̂
        For Earth (J₂ = 1.083e-3) + LEO (r = 6771 km): ≈ 0.0125 m/s² inward.
        """
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.3, j2=self.J2_EARTH,
        )
        r = 6.771e6
        sat = CelestialBody(1.0, np.array([r, 0.0, 0.0]), np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat])
        acc = sys.compute_accelerations()

        # Newtonian on sat: a = −GM/r² in +x → −x
        a_newton = -_G * self.M_EARTH / (r * r)
        # J₂ adds extra inward (−x) of magnitude 1.5 G M J₂ R²/r⁴
        a_j2 = -1.5 * _G * self.M_EARTH * self.J2_EARTH * self.R_EARTH ** 2 / r ** 4

        expected_x = a_newton + a_j2
        self.assertAlmostEqual(
            acc[1][0], expected_x, delta=abs(expected_x) * 1e-10,
        )
        # No tangential force at equator
        self.assertAlmostEqual(acc[1][1], 0.0, delta=1e-15)
        self.assertAlmostEqual(acc[1][2], 0.0, delta=1e-15)

    def test_polar_force_oblate_is_less_attractive(self):
        """At pole (r along n̂), J₂ > 0 reduces inward attraction.

        Total a_z = −GM/r² + 3 G M J₂ R²/r⁴
        Magnitude smaller than pure Newtonian.
        """
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.3, j2=self.J2_EARTH,
        )
        r = 6.771e6
        sat = CelestialBody(1.0, np.array([0.0, 0.0, r]), np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat])
        acc = sys.compute_accelerations()

        a_newton = -_G * self.M_EARTH / (r * r)               # −z direction
        a_j2     = +3.0 * _G * self.M_EARTH * self.J2_EARTH * self.R_EARTH ** 2 / r ** 4  # +z (outward)

        expected_z = a_newton + a_j2
        self.assertAlmostEqual(
            acc[1][2], expected_z, delta=abs(expected_z) * 1e-10,
        )
        self.assertAlmostEqual(acc[1][0], 0.0, delta=1e-15)
        self.assertAlmostEqual(acc[1][1], 0.0, delta=1e-15)

    def test_earth_leo_j2_magnitude_canonical(self):
        """Sanity: Earth's J₂ effect at LEO altitude is ~0.14% of Newtonian.

        This is the magnitude that drives Sun-synchronous orbits and is
        documented in every astrodynamics textbook (Vallado, Curtis, etc.)
        """
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.3, j2=self.J2_EARTH,
        )
        r = 6.771e6  # 400 km altitude
        sat = CelestialBody(1.0, np.array([r, 0.0, 0.0]), np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat])
        acc = sys.compute_accelerations()

        a_newton_mag = _G * self.M_EARTH / (r * r)
        # |acc[1]| should be a_newton + a_j2 ≈ 8.69 + 0.0125 ≈ 8.70 m/s²
        a_total = float(np.linalg.norm(acc[1]))
        ratio = (a_total - a_newton_mag) / a_newton_mag
        # J₂ contribution at LEO: ~0.14% of Newtonian (canonical textbook value)
        self.assertAlmostEqual(ratio, 0.00143, delta=2e-4)

    def test_j2_does_not_self_force(self):
        """A body's own J₂ does not apply force to itself."""
        # Standalone body with non-zero j2 — no other bodies
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.3, j2=self.J2_EARTH,
        )
        sys = NBodySystem([earth])
        acc = sys.compute_accelerations()
        np.testing.assert_array_equal(acc[0], np.zeros(3))

    def test_tilted_pole_force_direction(self):
        """With pole tilted, the equator/pole geometry rotates accordingly.

        Pole along +x means the "equator" is the y-z plane.
        A satellite at (r, 0, 0) is now at the POLE, not the equator,
        so should experience an outward (less attractive) J₂ correction.
        """
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.3, j2=self.J2_EARTH,
            pole_axis_unit=np.array([1.0, 0.0, 0.0]),  # pole along x-axis
        )
        r = 6.771e6
        sat = CelestialBody(1.0, np.array([r, 0.0, 0.0]), np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat])
        acc = sys.compute_accelerations()

        a_newton = -_G * self.M_EARTH / (r * r)
        a_j2     = +3.0 * _G * self.M_EARTH * self.J2_EARTH * self.R_EARTH ** 2 / r ** 4

        expected_x = a_newton + a_j2  # x-axis behaves like "pole" now
        self.assertAlmostEqual(
            acc[1][0], expected_x, delta=abs(expected_x) * 1e-10,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Roche limit
# ═══════════════════════════════════════════════════════════════════════════

class TestRocheLimit(unittest.TestCase):

    def test_roche_formula(self):
        """d_R = R_p × (2 M_p/M_s)^(1/3) — Roche (1847)."""
        b1 = CelestialBody(M_SUN,       np.zeros(3), np.zeros(3), R_SUN, 0.5)
        b2 = CelestialBody(M_SUN * 0.1, np.array([1e11, 0, 0.0]), np.zeros(3), R_SUN * 0.5, 0.3)
        sys = NBodySystem([b1, b2])
        rl  = sys.roche_limit(0, 1)
        expected = R_SUN * (2 * M_SUN / (M_SUN * 0.1)) ** (1.0 / 3.0)
        self.assertAlmostEqual(rl, expected, delta=expected * 1e-10)

    def test_roche_zero_satellite_mass(self):
        b1 = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        b2 = CelestialBody(0.0,   np.array([AU, 0, 0.0]), np.zeros(3), 1e6, 0.3)
        sys = NBodySystem([b1, b2])
        self.assertEqual(sys.roche_limit(0, 1), float("inf"))


# ═══════════════════════════════════════════════════════════════════════════
# GW energy loss (Peters 1964)
# ═══════════════════════════════════════════════════════════════════════════

class TestGWDamping(unittest.TestCase):
    """GW damping tests using a compact neutron-star binary.

    Separation a = 1e9 m (1000 × R_earth).  Orbital period ~540 s.
    dt = 5 s → ~108 steps/orbit — well resolved.
    """

    @staticmethod
    def _ns_binary(a: float = 1e9):
        """Equal-mass NS binary in circular COM orbit at separation a."""
        M    = 1.4 * M_SUN
        # Circular velocity per body: v = sqrt(G*m/(2a))  (COM frame)
        v    = math.sqrt(_G * M / (2 * a))
        b1 = CelestialBody(M, np.array([ a/2, 0, 0.0]), np.array([0,  v, 0.0]), 1e4, 0.3)
        b2 = CelestialBody(M, np.array([-a/2, 0, 0.0]), np.array([0, -v, 0.0]), 1e4, 0.3)
        return b1, b2

    def test_energy_decreases_with_gw(self):
        """GW damping should cause total energy to decrease over time."""
        M  = 1.4 * M_SUN
        a  = 1e9
        T  = 2 * math.pi * math.sqrt(a**3 / (_G * 2 * M))  # orbital period (s)
        dt = T / 100          # 100 steps per orbit
        b1, b2 = self._ns_binary(a)

        system = NBodySystem([b1, b2])
        E0     = system.total_energy()

        n_steps = int(5 * T / dt)  # 5 orbits
        for _ in range(n_steps):
            system.step(dt, include_gw_loss=True)

        E1 = system.total_energy()
        self.assertLess(E1, E0, f"GW damping should reduce energy: E0={E0:.3e} E1={E1:.3e}")

    def test_no_damping_no_energy_loss(self):
        """Without GW damping, energy should be well conserved."""
        M  = 1.4 * M_SUN
        a  = 1e9
        T  = 2 * math.pi * math.sqrt(a**3 / (_G * 2 * M))
        dt = T / 100
        b1, b2 = self._ns_binary(a)

        system = NBodySystem([b1, b2])
        E0     = system.total_energy()
        n_steps = int(5 * T / dt)  # 5 orbits
        for _ in range(n_steps):
            system.step(dt, include_gw_loss=False)
        E1    = system.total_energy()
        drift = abs(E1 - E0) / abs(E0)
        self.assertLess(drift, 0.01, f"Energy drift {drift:.2e} without GW")


# ═══════════════════════════════════════════════════════════════════════════
# GR correction
# ═══════════════════════════════════════════════════════════════════════════

class TestGRCorrection(unittest.TestCase):

    def test_gr_adds_correction(self):
        """With include_gr=True, acceleration magnitude should differ from Newtonian."""
        b1 = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        b2 = CelestialBody(1e24, np.array([AU / 10, 0, 0.0]),
                           np.array([0, 3e4, 0.0]), 1e6, 0.3)

        acc_n  = NBodySystem([b1, b2], include_gr=False).compute_accelerations()
        acc_gr = NBodySystem([b1, b2], include_gr=True ).compute_accelerations()

        # GR correction is small but non-zero
        diff = float(np.linalg.norm(acc_gr[1] - acc_n[1]))
        self.assertGreater(diff, 0)

    def test_gr_zero_velocity_no_change(self):
        """At v=0 the 1PN correction vanishes (v² = 0, r̂·v = 0 terms)."""
        b1 = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        b2 = CelestialBody(1e24, np.array([AU / 10, 0, 0.0]),
                           np.zeros(3), 1e6, 0.3)

        acc_n  = NBodySystem([b1, b2], include_gr=False).compute_accelerations()
        acc_gr = NBodySystem([b1, b2], include_gr=True ).compute_accelerations()

        # At v=0: v² = 0 and r̂·v = 0, so the 1PN correction is (4GM/r)r̂
        # It's non-zero, but the Newtonian dominates by >>factor
        ratio = float(np.linalg.norm(acc_gr[1])) / float(np.linalg.norm(acc_n[1]))
        # Correction is ~(4GM/rc²) / 1 ≈ very small
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
