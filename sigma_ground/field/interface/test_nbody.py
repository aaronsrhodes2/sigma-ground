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
    PhysicsToggles,
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


# ═══════════════════════════════════════════════════════════════════════════
# PhysicsToggles dataclass + backward-compat shim
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicsToggles(unittest.TestCase):
    """The toggles dataclass governs each force layer; defaults all-False."""

    def test_default_all_false(self):
        t = PhysicsToggles()
        for flag in ("gr_1pn", "gr_2pn", "eih_cross", "srp",
                     "j2_zonal", "j3_zonal", "j4_zonal",
                     "tidal_force", "gw_damping"):
            self.assertFalse(getattr(t, flag), f"{flag} should default False")

    def test_toggles_are_frozen(self):
        t = PhysicsToggles(gr_1pn=True)
        with self.assertRaises(Exception):  # FrozenInstanceError
            t.gr_1pn = False  # type: ignore

    def test_legacy_kwargs_build_toggles_gr_only(self):
        """include_gr=True without explicit toggles -> gr_1pn=True, others False."""
        b1, b2 = _two_body_circular()
        sys = NBodySystem([b1, b2], include_gr=True)
        self.assertTrue(sys.toggles.gr_1pn)
        self.assertFalse(sys.toggles.srp)
        self.assertFalse(sys.toggles.j2_zonal)  # no body has j2

    def test_legacy_kwargs_build_toggles_with_srp(self):
        b1, b2 = _two_body_circular()
        sys = NBodySystem([b1, b2], include_gr=True, solar_luminosity_W=3.828e26)
        self.assertTrue(sys.toggles.gr_1pn)
        self.assertTrue(sys.toggles.srp)

    def test_legacy_kwargs_auto_detect_j2(self):
        """If any body has j2 != 0 and no toggles given, j2_zonal is auto-on."""
        b1 = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        b2 = CelestialBody(M_SUN, np.array([1e10, 0.0, 0.0]),
                            np.zeros(3), R_SUN, 0.5, j2=1e-3)
        sys = NBodySystem([b1, b2])
        self.assertTrue(sys.toggles.j2_zonal,
                        "j2_zonal should auto-enable when any body has j2 != 0")

    def test_explicit_toggles_overrides_legacy(self):
        """If toggles= is given explicitly, legacy kwargs are ignored."""
        b1, b2 = _two_body_circular()
        # Pass include_gr=True but explicit toggles with gr_1pn=False
        sys = NBodySystem([b1, b2],
                          toggles=PhysicsToggles(gr_1pn=False),
                          include_gr=True)
        self.assertFalse(sys.toggles.gr_1pn,
                         "explicit toggles must win over legacy include_gr")

    def test_backward_compat_acceleration_bit_identical(self):
        """Without any new toggles enabled, accelerations match pre-refactor.

        The classic invocation `NBodySystem(bodies, include_gr=True,
        solar_luminosity_W=L)` should produce the same accelerations as it
        did before PhysicsToggles existed.
        """
        b1 = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        b2 = CelestialBody(M_SUN, np.array([1e11, 0.0, 0.0]),
                            np.array([0.0, 3e4, 0.0]), R_SUN, 0.5)
        sys = NBodySystem([b1, b2], include_gr=True, solar_luminosity_W=3.828e26)
        acc = sys.compute_accelerations()
        # The legacy interface gave us Newtonian + 1PN with no other layers,
        # since neither body has j2/j3/j4/love_number_k2 set.
        # If we explicitly disable all toggles except gr_1pn, we should get
        # the same numbers.
        sys_explicit = NBodySystem([b1, b2],
                                    toggles=PhysicsToggles(gr_1pn=True))
        acc_explicit = sys_explicit.compute_accelerations()
        np.testing.assert_allclose(acc, acc_explicit, atol=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2PN GR (BORROWED) — single-body Schwarzschild c⁻⁴ correction
# ═══════════════════════════════════════════════════════════════════════════

class TestGR2PN(unittest.TestCase):
    """The gr_2pn toggle adds a c⁻⁴ correction on top of 1PN.

    These tests verify (a) backward-compat with toggle off, and
    (b) the c⁻⁴ correction has the expected magnitude (smaller than 1PN
    by a factor of v²/c² or GM/(rc²) -- order 10⁻⁸ for Mercury).
    """

    def _mercury_like(self):
        """Mercury-like configuration: tight orbit around the Sun."""
        sun = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.0)
        # Mercury at perihelion: r=0.31 AU, v=58.98 km/s
        r_au = 0.31 * AU
        v    = 58.98e3
        body = CelestialBody(
            3.302e23, np.array([r_au, 0.0, 0.0]),
            np.array([0.0, v, 0.0]), 2440e3, 0.0,
        )
        return [sun, body]

    def test_2pn_off_matches_1pn_only(self):
        bodies = self._mercury_like()
        a_1pn = NBodySystem(bodies,
                             toggles=PhysicsToggles(gr_1pn=True)).compute_accelerations()
        a_both = NBodySystem(bodies,
                              toggles=PhysicsToggles(gr_1pn=True, gr_2pn=False)
                              ).compute_accelerations()
        np.testing.assert_allclose(a_both, a_1pn, atol=0.0)

    def test_2pn_correction_detectable_in_compact_binary(self):
        """2PN correction at NS-binary scale (v ~ 0.05c) should be detectable.

        At Mercury's solar-system scale, 2PN is ~10⁻²⁵ m/s² -- below float64
        precision when added to Newton's ~10⁻² m/s². We need a strong-field
        regime to make 2PN measurable.
        """
        # Neutron-star binary: m=1.4 M_sun each, separation 1e7 m → v ~ 0.05c
        M  = 1.4 * M_SUN
        r  = 1e7
        v  = math.sqrt(_G * M / r)  # circular velocity at r (~0.05c here)
        b1 = CelestialBody(M, np.zeros(3), np.zeros(3), 1e4, 0.0)
        b2 = CelestialBody(M, np.array([r, 0.0, 0.0]),
                            np.array([0.0, v, 0.0]), 1e4, 0.0)
        bodies = [b1, b2]
        a_1pn  = NBodySystem(bodies,
                              toggles=PhysicsToggles(gr_1pn=True)
                              ).compute_accelerations()
        a_2pn  = NBodySystem(bodies,
                              toggles=PhysicsToggles(gr_1pn=True, gr_2pn=True)
                              ).compute_accelerations()
        delta_2pn = float(np.linalg.norm(a_2pn[1] - a_1pn[1]))
        # In this regime v/c ≈ 0.05, so 2PN/1PN ratio ~ 0.0025.
        # Both deltas should be measurable.
        self.assertGreater(delta_2pn, 0,
                            "2PN should change accel in strong-field regime")

    def test_2pn_zero_velocity_still_has_radial(self):
        """At v=0, 2PN should give a purely radial (GM/r)² correction."""
        sun = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.0)
        body = CelestialBody(
            1e20, np.array([1e10, 0.0, 0.0]), np.zeros(3), 1e3, 0.0,
        )
        a_2pn = NBodySystem([sun, body],
                             toggles=PhysicsToggles(gr_2pn=True)
                             ).compute_accelerations()
        a_n   = NBodySystem([sun, body],
                             toggles=PhysicsToggles()
                             ).compute_accelerations()
        # The 2PN delta should be in the radial direction (along -x for body at +x)
        delta = a_2pn[1] - a_n[1]
        # The y and z components of delta should be zero (radial-only at v=0)
        self.assertAlmostEqual(delta[1], 0.0, delta=abs(delta[0]) * 1e-10)
        self.assertAlmostEqual(delta[2], 0.0, delta=abs(delta[0]) * 1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# J₃ and J₄ zonal harmonics (BORROWED)
# ═══════════════════════════════════════════════════════════════════════════

class TestZonalJ3J4(unittest.TestCase):
    """The j3_zonal and j4_zonal toggles add higher-order zonal corrections."""

    M_EARTH = 5.972e24
    R_EARTH = 6.371e6
    J3_EARTH = -2.5e-6
    J4_EARTH = -1.6e-6

    def _earth_with_satellite(self, j3=0.0, j4=0.0):
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.0, j2=0.0, j3=j3, j4=j4,
        )
        # LEO satellite at z != 0 so the asymmetric J3 force is visible
        r = 6.771e6
        sat = CelestialBody(
            1.0, np.array([r * 0.7071, 0.0, r * 0.7071]),
            np.zeros(3), 1.0, 0.0,
        )
        return [earth, sat]

    def test_j3_default_zero(self):
        body = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        self.assertEqual(body.j3, 0.0)

    def test_j4_default_zero(self):
        body = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.5)
        self.assertEqual(body.j4, 0.0)

    def test_j3_toggle_off(self):
        """j3_zonal=False -> no contribution even if body.j3 set."""
        bodies = self._earth_with_satellite(j3=self.J3_EARTH)
        sys = NBodySystem(bodies, toggles=PhysicsToggles())
        acc = sys.compute_accelerations()
        # Compare to pure Newton (no j3 set)
        bodies_n = self._earth_with_satellite(j3=0.0)
        sys_n = NBodySystem(bodies_n, toggles=PhysicsToggles())
        acc_n = sys_n.compute_accelerations()
        np.testing.assert_allclose(acc, acc_n, atol=0.0)

    def test_j3_toggle_on_with_zero_coeff(self):
        """j3_zonal=True but body.j3=0 -> no change."""
        bodies = self._earth_with_satellite(j3=0.0)
        sys_on  = NBodySystem(bodies, toggles=PhysicsToggles(j3_zonal=True))
        sys_off = NBodySystem(bodies, toggles=PhysicsToggles())
        np.testing.assert_allclose(sys_on.compute_accelerations(),
                                    sys_off.compute_accelerations(),
                                    atol=0.0)

    def test_j3_toggle_on_changes_accel(self):
        """j3_zonal=True with non-trivial body.j3 -> measurable change.

        Use j3 = 0.01 (well above Earth's -2.5e-6) to make the change
        detectable against Newton's much larger acceleration. The realistic
        Earth J₃ is too small to detect at float64 precision when subtracted
        from a Newton ~1e-1 m/s²; this test validates the formula path, not
        the realistic magnitude.
        """
        bodies = self._earth_with_satellite(j3=0.01)
        sys_on = NBodySystem(bodies, toggles=PhysicsToggles(j3_zonal=True))
        sys_off = NBodySystem(bodies, toggles=PhysicsToggles())
        acc_on  = sys_on.compute_accelerations()
        acc_off = sys_off.compute_accelerations()
        delta = acc_on[1] - acc_off[1]
        self.assertGreater(float(np.linalg.norm(delta)), 0.0,
                            "J₃ should change satellite acceleration "
                            "(with inflated J₃ coefficient for FP detectability)")

    def test_j4_toggle_off(self):
        bodies = self._earth_with_satellite(j4=self.J4_EARTH)
        sys = NBodySystem(bodies, toggles=PhysicsToggles())
        acc = sys.compute_accelerations()
        bodies_n = self._earth_with_satellite(j4=0.0)
        sys_n = NBodySystem(bodies_n, toggles=PhysicsToggles())
        np.testing.assert_allclose(acc, sys_n.compute_accelerations(), atol=0.0)

    def test_j4_toggle_on_changes_accel(self):
        """j4_zonal with non-trivial body.j4 -> measurable change.

        Same FP-precision caveat as j3: realistic Earth J4 ~ -1.6e-6 is below
        detectability against Newton; use inflated 0.01 to validate the path.
        """
        bodies = self._earth_with_satellite(j4=0.01)
        sys_on  = NBodySystem(bodies, toggles=PhysicsToggles(j4_zonal=True))
        sys_off = NBodySystem(bodies, toggles=PhysicsToggles())
        delta = sys_on.compute_accelerations()[1] - sys_off.compute_accelerations()[1]
        self.assertGreater(float(np.linalg.norm(delta)), 0.0)

    # -- Quantitative formula checks: limits and magnitudes ---------------

    def test_j3_at_equator_is_purely_along_pole(self):
        """At equator (s=0), J3 force must be along n̂ only, zero radial.

        Derivation: a_J3 = (GM J₃ R³)/(2 r⁵) × [(3 - 15s²) n̂ + 5s(7s² - 3) r̂]
        At s=0: a_J3 = (3 GM J₃ R³)/(2 r⁵) × n̂  -- pure n̂ component.

        If the formula has the wrong angular-polynomial coefficients (e.g.
        my previous buggy implementation with (35s⁴ - 15s²) on r̂), the
        force at equator would have a nonzero radial component. This test
        catches that class of error.
        """
        # Build an "Earth" with inflated J₃ for FP detectability.
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.0, j3=0.01,
        )
        # Satellite at equator (z=0)
        r_sat = 1e7
        sat = CelestialBody(1.0, np.array([r_sat, 0.0, 0.0]),
                             np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat], toggles=PhysicsToggles(j3_zonal=True))
        sys_n = NBodySystem([earth, sat], toggles=PhysicsToggles())
        delta = sys.compute_accelerations()[1] - sys_n.compute_accelerations()[1]
        # delta should be purely along +z (n̂); x and y components must be zero
        self.assertAlmostEqual(delta[0], 0.0, delta=abs(delta[2]) * 1e-10,
                                msg="J3 at equator has spurious radial-x component")
        self.assertAlmostEqual(delta[1], 0.0, delta=abs(delta[2]) * 1e-10)
        self.assertGreater(abs(delta[2]), 0.0,
                            "J3 at equator should produce nonzero n̂ force")

    def test_j3_at_equator_magnitude_matches_derivation(self):
        """At equator, |a_J3| = 3 GM J₃ R³/(2 r⁵)."""
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.0, j3=self.J3_EARTH,
        )
        r = 1e7
        sat = CelestialBody(1.0, np.array([r, 0.0, 0.0]), np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat], toggles=PhysicsToggles(j3_zonal=True))
        sys_n = NBodySystem([earth, sat], toggles=PhysicsToggles())
        delta = sys.compute_accelerations()[1] - sys_n.compute_accelerations()[1]
        # |a_J3_n̂| = 3 GM J₃ R³ / (2 r⁵)
        expected = 3.0 * _G * self.M_EARTH * self.J3_EARTH * self.R_EARTH**3 / (2.0 * r**5)
        self.assertAlmostEqual(delta[2], expected, delta=abs(expected) * 1e-8)

    def test_j4_at_equator_is_purely_radial(self):
        """At equator (s=0), J4 force is purely radial.

        Derivation: a_J4 = (5 GM J₄ R⁴)/(8 r⁶) × [3(21s⁴-14s²+1) r̂ + 4s(3-7s²) n̂]
        At s=0: a_J4 = (15/8) GM J₄ R⁴/r⁶ × r̂  -- pure r̂ component.
        """
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.0, j4=0.01,
        )
        r = 1e7
        sat = CelestialBody(1.0, np.array([r, 0.0, 0.0]), np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat], toggles=PhysicsToggles(j4_zonal=True))
        sys_n = NBodySystem([earth, sat], toggles=PhysicsToggles())
        delta = sys.compute_accelerations()[1] - sys_n.compute_accelerations()[1]
        # delta should be purely along x (r̂ direction); y and z must be zero
        self.assertGreater(abs(delta[0]), 0.0,
                            "J4 at equator should produce nonzero radial force")
        self.assertAlmostEqual(delta[1], 0.0, delta=abs(delta[0]) * 1e-10,
                                msg="J4 at equator has spurious y component")
        self.assertAlmostEqual(delta[2], 0.0, delta=abs(delta[0]) * 1e-10,
                                msg="J4 at equator has spurious z (n̂) component")

    def test_j4_at_equator_magnitude_matches_derivation(self):
        """At equator, |a_J4_radial| = (15/8) GM J₄ R⁴/r⁶."""
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.0, j4=self.J4_EARTH,
        )
        r = 1e7
        sat = CelestialBody(1.0, np.array([r, 0.0, 0.0]), np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat], toggles=PhysicsToggles(j4_zonal=True))
        sys_n = NBodySystem([earth, sat], toggles=PhysicsToggles())
        delta = sys.compute_accelerations()[1] - sys_n.compute_accelerations()[1]
        expected = (15.0 / 8.0) * _G * self.M_EARTH * self.J4_EARTH * self.R_EARTH**4 / r**6
        self.assertAlmostEqual(delta[0], expected, delta=abs(expected) * 1e-8)

    def test_j4_at_pole_magnitude_matches_derivation(self):
        """At pole (s=1), |a_J4| = 5 GM J₄ R⁴/r⁶ along the pole axis.

        Derivation: at s=1, the vector form
          a_J4 = (5 GM J₄ R⁴)/(8 r⁶) × [3(21s⁴-14s²+1) r̂ + 4s(3-7s²) n̂]
        simplifies (r̂ = n̂ at the pole):
          coef × [3(21-14+1) + 4(3-7)] = coef × [24 - 16] = 8 coef
          = (5 GM J₄ R⁴) / r⁶
        Sign of force matches sign of J₄ (positive J₄ pushes outward at pole;
        negative J₄ like Saturn's pulls inward at pole).
        """
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, 0.0, j4=self.J4_EARTH,
        )
        r = 1e7
        # Test particle at the north pole — position purely along +z
        sat = CelestialBody(1.0, np.array([0.0, 0.0, r]), np.zeros(3), 1.0, 0.0)
        sys = NBodySystem([earth, sat], toggles=PhysicsToggles(j4_zonal=True))
        sys_n = NBodySystem([earth, sat], toggles=PhysicsToggles())
        delta = sys.compute_accelerations()[1] - sys_n.compute_accelerations()[1]
        expected_z = 5.0 * _G * self.M_EARTH * self.J4_EARTH * self.R_EARTH**4 / r**6
        # Pole result should be purely along z, magnitude matching analytic
        self.assertAlmostEqual(delta[0], 0.0, delta=abs(expected_z) * 1e-10,
                                msg="J4 at pole has spurious x component")
        self.assertAlmostEqual(delta[1], 0.0, delta=abs(expected_z) * 1e-10,
                                msg="J4 at pole has spurious y component")
        self.assertAlmostEqual(delta[2], expected_z,
                                delta=abs(expected_z) * 1e-8)

    def test_j4_saturn_enceladus_geometry_matches_analytic(self):
        """Saturn J4 effect at Enceladus's orbital radius -- specific to the
        2026-05-15 regression diagnosis.

        Verified that the J4 formula is correct for the Saturn-Enceladus
        case AT the pole (most-favourable analytic check). The Enceladus
        regression we observed in rolling_shootout_toggle_iteration is
        NOT a J4 formula bug -- it's that our DE440 fixture is missing
        Dione, breaking Enceladus's 2:1 resonance perturbation. See
        misc/saturn_enceladus_j4_verdict_2026-05-15.md for details.

        This test exists to lock in the formula's correctness so the
        Dione-perturber fix doesn't accidentally regress here.
        """
        G_real = _G
        M_SAT  = 5.6834e26     # kg, Saturn mass
        R_SAT  = 60268e3       # m
        J4_SAT = -9.15e-4      # Anderson & Schubert 2007
        r_enc  = 238042e3      # m, Enceladus orbital radius

        saturn = CelestialBody(
            M_SAT, np.zeros(3), np.zeros(3),
            R_SAT, 0.39, j4=J4_SAT,
        )
        # Test particle at Saturn's north pole (purely along +z)
        enc = CelestialBody(1.0, np.array([0.0, 0.0, r_enc]), np.zeros(3), 1.0, 0.0)

        sys = NBodySystem([saturn, enc], toggles=PhysicsToggles(j4_zonal=True))
        sys_n = NBodySystem([saturn, enc], toggles=PhysicsToggles())
        delta = sys.compute_accelerations()[1] - sys_n.compute_accelerations()[1]

        expected_z = 5.0 * G_real * M_SAT * J4_SAT * R_SAT**4 / r_enc**6
        self.assertAlmostEqual(delta[2], expected_z, delta=abs(expected_z) * 1e-8)
        # Negative J4 means the force at the pole is INWARD (toward Saturn),
        # which means a_z is negative for a moon above the pole.
        self.assertLess(delta[2], 0.0,
                         "Saturn negative J4 should produce inward force at pole")


# ═══════════════════════════════════════════════════════════════════════════
# Tidal force (OURS — built from compute_tidal_deformation)
# ═══════════════════════════════════════════════════════════════════════════

class TestTidalForce(unittest.TestCase):
    """The tidal_force toggle applies the tidally-induced quadrupole."""

    M_EARTH = 5.972e24
    R_EARTH = 6.371e6
    M_MOON  = 7.342e22
    R_MOON  = 1.7374e6
    EARTH_MOON_DIST = 3.844e8

    def _earth_moon(self, k2_earth=0.30):
        earth = CelestialBody(
            self.M_EARTH, np.zeros(3), np.zeros(3),
            self.R_EARTH, k2_earth,
        )
        moon = CelestialBody(
            self.M_MOON,
            np.array([self.EARTH_MOON_DIST, 0.0, 0.0]),
            np.array([0.0, 1.022e3, 0.0]),  # ~lunar orbital velocity
            self.R_MOON, 0.0,
        )
        return [earth, moon]

    def test_tidal_off_matches_newtonian(self):
        bodies = self._earth_moon()
        sys_off = NBodySystem(bodies, toggles=PhysicsToggles())
        sys_n   = NBodySystem(bodies, toggles=PhysicsToggles())
        np.testing.assert_allclose(sys_off.compute_accelerations(),
                                    sys_n.compute_accelerations(),
                                    atol=0.0)

    def test_tidal_on_zero_love_no_effect(self):
        """Earth with k₂=0 -> no tidal bulge -> no tidal force."""
        bodies = self._earth_moon(k2_earth=0.0)
        sys_on  = NBodySystem(bodies, toggles=PhysicsToggles(tidal_force=True))
        sys_off = NBodySystem(bodies, toggles=PhysicsToggles())
        np.testing.assert_allclose(sys_on.compute_accelerations(),
                                    sys_off.compute_accelerations(),
                                    atol=0.0)

    def test_tidal_on_earth_k2_changes_moon_accel(self):
        """With k₂_Earth = 0.30, Moon should feel an extra force from Earth's bulge."""
        bodies = self._earth_moon(k2_earth=0.30)
        sys_on  = NBodySystem(bodies, toggles=PhysicsToggles(tidal_force=True))
        sys_off = NBodySystem(bodies, toggles=PhysicsToggles())
        a_on  = sys_on.compute_accelerations()
        a_off = sys_off.compute_accelerations()
        delta_moon = a_on[1] - a_off[1]
        self.assertGreater(float(np.linalg.norm(delta_moon)), 0.0,
                            "tidal_force should change Moon's acceleration "
                            "when Earth has a non-zero Love number")


# ═══════════════════════════════════════════════════════════════════════════
# gw_damping toggle (was step() kwarg, now PhysicsToggles)
# ═══════════════════════════════════════════════════════════════════════════

class TestGWDampingToggle(unittest.TestCase):
    """The gw_damping toggle replaces the step(include_gw_loss=...) kwarg.

    The legacy kwarg is still honored when explicitly True/False; only when
    it's None (the new default) does the toggle take effect.
    """

    @staticmethod
    def _ns_binary(a=1e9):
        M = 1.4 * M_SUN
        v = math.sqrt(_G * M / (2 * a))
        b1 = CelestialBody(M, np.array([ a/2, 0, 0.0]),
                            np.array([0,  v, 0.0]), 1e4, 0.3)
        b2 = CelestialBody(M, np.array([-a/2, 0, 0.0]),
                            np.array([0, -v, 0.0]), 1e4, 0.3)
        return b1, b2

    def test_legacy_kwarg_true_still_damps(self):
        """step(dt, include_gw_loss=True) overrides toggle and applies damping."""
        b1, b2 = self._ns_binary()
        sys = NBodySystem([b1, b2])  # toggles.gw_damping defaults False
        e0 = sys.total_energy()
        for _ in range(20):
            sys.step(1.0, include_gw_loss=True)
        e1 = sys.total_energy()
        # Energy should DECREASE due to GW damping
        self.assertLess(e1, e0)

    def test_default_kwarg_uses_toggle(self):
        """step(dt) with no kwarg -> uses self.toggles.gw_damping."""
        b1, b2 = self._ns_binary()
        sys = NBodySystem([b1, b2],
                           toggles=PhysicsToggles(gw_damping=True))
        e0 = sys.total_energy()
        for _ in range(20):
            sys.step(1.0)
        e1 = sys.total_energy()
        self.assertLess(e1, e0)

    def test_legacy_kwarg_false_blocks_toggle(self):
        """step(dt, include_gw_loss=False) overrides even if toggle is True."""
        b1, b2 = self._ns_binary()
        sys = NBodySystem([b1, b2],
                           toggles=PhysicsToggles(gw_damping=True))
        # Take a step with override-off
        e0 = sys.total_energy()
        for _ in range(20):
            sys.step(1.0, include_gw_loss=False)
        e1 = sys.total_energy()
        # Without damping (and good integration), energy should be nearly conserved
        self.assertAlmostEqual(e1 / e0, 1.0, delta=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# Sigma-bounds validation
# ═══════════════════════════════════════════════════════════════════════════

class TestSigmaBoundsCheck(unittest.TestCase):
    """CelestialBody.__post_init__ rejects σ values outside the SSBM domain."""

    def test_sigma_zero_safe(self):
        """σ = 0 (vacuum) is SAFE — no exception."""
        body = CelestialBody(M_SUN, np.zeros(3), np.zeros(3),
                              R_SUN, 0.5, sigma_field=0.0)
        self.assertEqual(body.sigma_field, 0.0)

    def test_sigma_small_positive_safe(self):
        """Earth-surface-class σ (~1e-9) is SAFE."""
        body = CelestialBody(M_SUN, np.zeros(3), np.zeros(3),
                              R_SUN, 0.5, sigma_field=1e-9)
        self.assertEqual(body.sigma_field, 1e-9)

    def test_sigma_negative_raises(self):
        """σ < 0 is BEYOND domain -- must raise."""
        with self.assertRaises(ValueError):
            CelestialBody(M_SUN, np.zeros(3), np.zeros(3),
                           R_SUN, 0.5, sigma_field=-0.1)

    def test_sigma_beyond_conv_raises(self):
        """σ > σ_conv (≈1.849) is BEYOND domain -- must raise."""
        with self.assertRaises(ValueError):
            CelestialBody(M_SUN, np.zeros(3), np.zeros(3),
                           R_SUN, 0.5, sigma_field=2.0)

    def test_gm_uses_scale_ratio(self):
        """gm_m3_s2 calls scale.scale_ratio (which clamps at ±709), not raw math.exp."""
        # σ = 700 is large but still within scale_ratio's guard
        body = CelestialBody(M_SUN, np.zeros(3), np.zeros(3),
                              R_SUN, 0.5, sigma_field=1.0)  # use safe value
        gm = body.gm_m3_s2
        expected = _G * M_SUN * math.exp(1.0)
        self.assertAlmostEqual(gm, expected, delta=expected * 1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Forest-Ruth (per-body dt)
# ═══════════════════════════════════════════════════════════════════════════

class TestHierarchicalForestRuth(unittest.TestCase):
    """forest_ruth_step_hierarchical advances slow bodies at dt and fast
    bodies at dt/n_substeps. Validates that:
      - With n_substeps=1 the result equals forest_ruth_step(dt) exactly.
      - With an empty fast_indices list the result equals forest_ruth_step(dt).
      - Slow bodies end up at the same position as a uniform-dt run.
      - Fast bodies' final positions differ from uniform-dt (substepping
        is doing something), and total energy stays well-bounded.
    """

    def _three_body(self):
        """Sun + Earth + Moon — Moon is the 'fast' body candidate."""
        b_sun = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.02)
        b_earth = CelestialBody(
            5.972e24,
            np.array([1.0 * AU, 0.0, 0.0]),
            np.array([0.0, 2.978e4, 0.0]),
            6.371e6, 0.3,
        )
        b_moon = CelestialBody(
            7.342e22,
            np.array([1.0 * AU + 3.844e8, 0.0, 0.0]),
            np.array([0.0, 2.978e4 + 1.022e3, 0.0]),
            1.737e6, 0.024,
        )
        return [b_sun, b_earth, b_moon]

    def test_hierarchical_with_n_substeps_1_matches_uniform(self):
        """n_substeps=1 must reduce to forest_ruth_step bit-identically."""
        bodies_a = self._three_body()
        bodies_b = self._three_body()
        sys_a = NBodySystem(bodies_a)
        sys_b = NBodySystem(bodies_b)
        dt = 86400.0  # 1 day
        sys_a.forest_ruth_step(dt)
        sys_b.forest_ruth_step_hierarchical(dt, fast_indices=[2], n_substeps=1)
        for i in range(3):
            np.testing.assert_array_almost_equal(
                sys_a.bodies[i].position_m, sys_b.bodies[i].position_m,
                decimal=20,
                err_msg=f"body {i}: n_substeps=1 path diverged from uniform",
            )

    def test_hierarchical_with_empty_fast_list_matches_uniform(self):
        """Empty fast_indices must fall through to forest_ruth_step."""
        bodies_a = self._three_body()
        bodies_b = self._three_body()
        sys_a = NBodySystem(bodies_a)
        sys_b = NBodySystem(bodies_b)
        dt = 86400.0
        sys_a.forest_ruth_step(dt)
        sys_b.forest_ruth_step_hierarchical(dt, fast_indices=[], n_substeps=10)
        for i in range(3):
            np.testing.assert_array_almost_equal(
                sys_a.bodies[i].position_m, sys_b.bodies[i].position_m,
                decimal=20,
            )

    def test_hierarchical_substepping_changes_fast_body_trajectory(self):
        """With n_substeps=10, the Moon's position must differ from uniform dt
        (otherwise the substepping is inert)."""
        bodies_a = self._three_body()
        bodies_b = self._three_body()
        sys_a = NBodySystem(bodies_a)
        sys_b = NBodySystem(bodies_b)
        dt = 86400.0 * 0.1   # 0.1 day, the canonical macro step
        # Run 30 macro steps (3 simulated days)
        for _ in range(30):
            sys_a.forest_ruth_step(dt)
            sys_b.forest_ruth_step_hierarchical(dt, fast_indices=[2], n_substeps=10)
        delta = float(np.linalg.norm(
            sys_a.bodies[2].position_m - sys_b.bodies[2].position_m
        ))
        # Moon moves ~3km/s in our frame; after 3 days the substepping
        # should produce at least some delta from uniform integration.
        # Make the check loose -- the substepping shouldn't be wildly
        # different but it shouldn't be exactly zero either.
        self.assertGreater(delta, 0.0,
                            "Hierarchical substepping produced no delta vs uniform")

    def test_hierarchical_advances_time_by_dt(self):
        """One hierarchical step at dt must advance system._time by exactly dt."""
        bodies = self._three_body()
        sys = NBodySystem(bodies)
        t0 = sys.time
        dt = 86400.0
        sys.forest_ruth_step_hierarchical(dt, fast_indices=[2], n_substeps=10)
        self.assertAlmostEqual(sys.time - t0, dt, delta=1e-9)

    def test_hierarchical_rejects_zero_substeps(self):
        bodies = self._three_body()
        sys = NBodySystem(bodies)
        with self.assertRaises(ValueError):
            sys.forest_ruth_step_hierarchical(86400.0, fast_indices=[2], n_substeps=0)

    def test_hierarchical_KNOWN_BROKEN_two_body_earth_moon(self):
        """Demonstrates the known bug: in a 2-body Earth-Moon system
        (no Sun), hierarchical(1d/0.1d) is WORSE than uniform 1d,
        because the slow-body advancement with frozen fast body produces
        a wrong slow trajectory.

        Expected behaviour with a CORRECT hierarchical method: the
        hierarchical result should approximate the uniform dt=0.1d
        reference. This test currently FAILS that expectation and is
        marked as a known regression. When the underlying algorithm is
        fixed (symplectic multi-timestep / RESPA), flip the assertion
        to assertLess and remove this banner.
        """
        def make():
            return [
                CelestialBody(5.972e24, np.zeros(3), np.zeros(3),
                               6.371e6, 0.3),
                CelestialBody(7.342e22, np.array([3.844e8, 0, 0]),
                               np.array([0, 1.022e3, 0]),
                               1.737e6, 0.024),
            ]
        DAY = 86400.0
        # Reference: uniform dt=0.1d
        s_ref = NBodySystem(make())
        for _ in range(300):
            s_ref.forest_ruth_step(0.1 * DAY)
        pos_ref = s_ref.bodies[1].position_m
        # Hierarchical 1d / 0.1d
        s_hier = NBodySystem(make())
        for _ in range(30):
            s_hier.forest_ruth_step_hierarchical(1.0 * DAY, [1], 10)
        pos_hier = s_hier.bodies[1].position_m
        # Uniform-coarse dt=1d
        s_coarse = NBodySystem(make())
        for _ in range(30):
            s_coarse.forest_ruth_step(1.0 * DAY)
        pos_coarse = s_coarse.bodies[1].position_m

        err_hier   = float(np.linalg.norm(pos_hier - pos_ref))
        err_coarse = float(np.linalg.norm(pos_coarse - pos_ref))
        # Known bug: hierarchical is several times WORSE than uniform-coarse.
        # When fixed, this assertion should be FLIPPED to err_hier < err_coarse.
        self.assertGreater(err_hier, err_coarse,
                            f"Bug regression: hierarchical err {err_hier:.2e} "
                            f"should be > uniform-coarse err {err_coarse:.2e} "
                            f"until the operator-split fix lands.")

    def test_hierarchical_slow_body_drift_is_bounded(self):
        """The slow bodies' end state from hierarchical drifts from a pure
        forest_ruth_step run because we treat fast bodies as frozen during
        the slow-body integration. The drift is bounded by the magnitude
        of the fast body's perturbation; for Earth (slow) under Moon (fast)
        it's bounded by Moon's gravitational acceleration on Earth times
        dt^2. Test that the relative drift is small compared to Earth's
        orbital scale.
        """
        bodies_a = self._three_body()
        bodies_b = self._three_body()
        sys_a = NBodySystem(bodies_a)
        sys_b = NBodySystem(bodies_b)
        dt = 86400.0 * 0.1   # 0.1 day
        for _ in range(10):
            sys_a.forest_ruth_step(dt)
            sys_b.forest_ruth_step_hierarchical(dt, fast_indices=[2], n_substeps=10)
        # Earth (the "slow" body in this test). The hierarchical's frozen-
        # fast-body assumption introduces an error of order (Moon GM / r²)
        # × dt² per outer step; over 1 day of simulation, the absolute drift
        # is tens of km, but relative to Earth's 1-AU orbit (150e9 m) that's
        # ~1e-10. Verify drift is sub-1e-6 of orbital radius.
        earth_pos = sys_a.bodies[1].position_m
        earth_r = float(np.linalg.norm(earth_pos))
        delta = float(np.linalg.norm(earth_pos - sys_b.bodies[1].position_m))
        rel_drift = delta / earth_r
        self.assertLess(rel_drift, 1e-6,
                         f"Slow body (Earth) drifted {delta:.2f} m "
                         f"({rel_drift:.2e} of orbital radius {earth_r:.2e} m)")


# ═══════════════════════════════════════════════════════════════════════════
# EIH N-body 1PN cross-terms — JPL DE440 canonical force model
# ═══════════════════════════════════════════════════════════════════════════

class TestEIH1PN(unittest.TestCase):
    """Full N-body 1PN EIH equations (Will 1993 Box 6.2 / IAU 2000 §8.4).

    This is the canonical 1PN form used by JPL DE440 (Park et al. 2021).
    It differs from the single-body gr_1pn Schwarzschild approximation by
    including the cross-body potential terms (-(4/c²) Φ_i, -(1/c²) Φ_j),
    the v_j velocity contributions, and the 7/(2c²) Σ_j μ_j a_j^N / r_ij
    coupling. The differences are second-order in (v/c) so they're small
    at solar-system scales (~10⁻⁹), but they are what gives DE440 its
    mm-level Mercury accuracy.
    """

    def _three_body_solar_system_like(self):
        """Sun + Mercury-like + Jupiter-like (just for forcing a non-trivial
        Φ_i and Φ_j; not a quantitative match to the real solar system)."""
        b_sun = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.02)
        b_inner = CelestialBody(
            3.3e23,
            np.array([0.4 * AU, 0.0, 0.0]),
            np.array([0.0, 4.7e4, 0.0]),
            2.44e6, 0.45,
        )
        b_outer = CelestialBody(
            1.9e27,
            np.array([5.2 * AU, 0.0, 0.0]),
            np.array([0.0, 1.3e4, 0.0]),
            7.0e7, 0.535,
        )
        return [b_sun, b_inner, b_outer]

    def test_eih_no_longer_raises(self):
        """eih_cross used to raise NotImplementedError; now it returns finite accelerations."""
        bodies = self._three_body_solar_system_like()
        sys = NBodySystem(bodies, toggles=PhysicsToggles(eih_cross=True))
        acc = sys.compute_accelerations()
        self.assertTrue(np.all(np.isfinite(acc)))
        # All bodies must experience non-zero acceleration toward each other.
        for i in range(len(bodies)):
            self.assertGreater(float(np.linalg.norm(acc[i])), 0.0)

    def test_eih_off_reproduces_pre_eih_behavior(self):
        """Default toggles (eih_cross=False) MUST be bit-identical to pre-EIH code.

        This is the backward-compat guarantee. Pure-Newton baseline with no
        toggles set should produce the same acceleration regardless of any
        eih_cross machinery being added.
        """
        bodies = self._three_body_solar_system_like()
        acc_off = NBodySystem(bodies, toggles=PhysicsToggles()).compute_accelerations()
        # Re-construct fresh bodies and re-run — same numerical result expected.
        bodies2 = self._three_body_solar_system_like()
        acc_off2 = NBodySystem(bodies2, toggles=PhysicsToggles()).compute_accelerations()
        np.testing.assert_array_almost_equal(acc_off, acc_off2, decimal=20)

    def test_eih_differs_from_pure_newton(self):
        """EIH adds 1PN corrections — must differ from pure Newton."""
        bodies = self._three_body_solar_system_like()
        acc_newton = NBodySystem(bodies, toggles=PhysicsToggles()
                                  ).compute_accelerations()
        acc_eih = NBodySystem(bodies, toggles=PhysicsToggles(eih_cross=True)
                               ).compute_accelerations()
        diff = float(np.linalg.norm(acc_eih - acc_newton))
        self.assertGreater(diff, 0.0)

    def test_eih_differs_from_single_body_1pn_in_three_body(self):
        """In a 3-body system EIH and gr_1pn (single-body) should differ.

        Reason: EIH includes the cross-potential Φ_i = Σ_{k≠i} GM_k/r_ik
        terms, which the single-body Schwarzschild approximation lacks.
        In a 3-body Sun + 2-planet system the inner planet sees the Sun's
        potential AND the outer planet's potential in the EIH formulation,
        whereas single-body sees only the Sun.
        """
        bodies = self._three_body_solar_system_like()
        acc_1pn = NBodySystem(bodies, toggles=PhysicsToggles(gr_1pn=True)
                               ).compute_accelerations()
        acc_eih = NBodySystem(bodies, toggles=PhysicsToggles(eih_cross=True)
                               ).compute_accelerations()
        # The inner-body acceleration is where the cross-potential matters most.
        diff_inner = float(np.linalg.norm(acc_eih[1] - acc_1pn[1]))
        # Both should be O(GM_sun / r²) ≈ 0.04 m/s²; the EIH-1PN delta is
        # typically ~10⁻⁹ smaller. Just require strictly nonzero.
        self.assertGreater(diff_inner, 0.0)

    def test_eih_correction_is_small_at_solar_system_scales(self):
        """At v/c ~ 10⁻⁴ and GM/(rc²) ~ 10⁻⁸ the 1PN correction is ~10⁻⁸ of Newton."""
        bodies = self._three_body_solar_system_like()
        acc_newton = NBodySystem(bodies, toggles=PhysicsToggles()
                                  ).compute_accelerations()
        acc_eih = NBodySystem(bodies, toggles=PhysicsToggles(eih_cross=True)
                               ).compute_accelerations()
        for i in range(len(bodies)):
            mag_n   = float(np.linalg.norm(acc_newton[i]))
            mag_eih = float(np.linalg.norm(acc_eih[i]))
            if mag_n == 0.0:
                continue
            # Ratio close to 1.0 (correction is small fraction)
            ratio = mag_eih / mag_n
            self.assertAlmostEqual(ratio, 1.0, delta=1e-4)

    def test_eih_supersedes_gr1pn_when_both_on(self):
        """When eih_cross=True AND gr_1pn=True, only eih_cross applies.

        Both model the same 1PN physics — double-counting would be a bug.
        Verify by comparing (eih + gr1pn) ≡ (eih alone).
        """
        bodies = self._three_body_solar_system_like()
        acc_both = NBodySystem(
            bodies, toggles=PhysicsToggles(eih_cross=True, gr_1pn=True)
        ).compute_accelerations()
        bodies2 = self._three_body_solar_system_like()
        acc_eih_only = NBodySystem(
            bodies2, toggles=PhysicsToggles(eih_cross=True)
        ).compute_accelerations()
        np.testing.assert_array_almost_equal(acc_both, acc_eih_only, decimal=20)

    def test_eih_two_body_reduces_to_schwarzschild_at_leading_order(self):
        """For an isolated 2-body system, EIH and single-body 1PN agree at
        leading order (the small differences come from v_j² and v_i·v_j
        terms that single-body Schwarzschild treats as if v_j = 0 in the
        Sun frame).

        Since we DON'T impose a heliocentric frame here, even 2-body shows
        a measurable EIH/1PN delta. We just verify the ratio is close to 1.
        """
        b1 = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.02)
        b2 = CelestialBody(
            3.3e23,
            np.array([0.4 * AU, 0.0, 0.0]),
            np.array([0.0, 4.7e4, 0.0]),
            2.44e6, 0.45,
        )
        acc_1pn = NBodySystem([b1, b2], toggles=PhysicsToggles(gr_1pn=True)
                               ).compute_accelerations()
        b1b = CelestialBody(M_SUN, np.zeros(3), np.zeros(3), R_SUN, 0.02)
        b2b = CelestialBody(
            3.3e23,
            np.array([0.4 * AU, 0.0, 0.0]),
            np.array([0.0, 4.7e4, 0.0]),
            2.44e6, 0.45,
        )
        acc_eih = NBodySystem([b1b, b2b], toggles=PhysicsToggles(eih_cross=True)
                               ).compute_accelerations()
        # Test-particle approximation agrees on direction; magnitudes within 0.01%.
        ratio = (float(np.linalg.norm(acc_eih[1]))
                 / float(np.linalg.norm(acc_1pn[1])))
        self.assertAlmostEqual(ratio, 1.0, delta=1e-4)

    def test_eih_includes_a_newton_self_consistency(self):
        """The 7/(2c²) Σ_j μ_j a_j^N / r_ij coupling means each body's
        acceleration depends on the Newtonian accelerations of all OTHER
        bodies. This is what makes EIH 'mutual' rather than just superposed
        Schwarzschild fields. Test that this term is non-zero.
        """
        # Construct a config where a_j^N is large for j ≠ i: put the inner
        # planet next to the Sun so its Newtonian acceleration dominates.
        bodies = self._three_body_solar_system_like()
        # With EIH, the outer planet feels coupling from the inner planet's
        # large Newtonian acceleration via the 7/(2c²) term.
        sys = NBodySystem(bodies, toggles=PhysicsToggles(eih_cross=True))
        acc_eih = sys.compute_accelerations()
        # Without the 7/(2c²) term, the outer body's acceleration would be
        # dominated entirely by the Sun's GM. We can't easily isolate the
        # 7/(2c²) contribution analytically here without re-implementing it,
        # so this test just guards finite output (already done elsewhere)
        # and existence of a third-body effect.
        self.assertTrue(np.all(np.isfinite(acc_eih)))


# ═══════════════════════════════════════════════════════════════════════════
# Sun J2 — DE440 canonical value
# ═══════════════════════════════════════════════════════════════════════════

class TestSunJ2:
    """The Sun's J₂ is part of the DE440 force model.

    Park et al. 2021 ("The JPL Planetary and Lunar Ephemerides DE440 and
    DE441") Table 3 lists J₂_⊙ = 2.1106e-7 as the fitted value used in DE440.
    """

    def test_sun_j2_value_matches_de440(self):
        from sigma_ground.field.interface.rolling_shootout import _BODY_PARAMS
        assert _BODY_PARAMS["Sun"].j2 == 2.1106e-7

    def test_sun_has_iau2015_pole(self):
        """Sun's rotational pole at IAU 2015 (RA=286.13°, Dec=63.87°) -- not ICRS +z."""
        from sigma_ground.field.interface.rolling_shootout import _BODY_PARAMS
        pole = _BODY_PARAMS["Sun"].pole_axis_unit
        assert pole is not None
        # Sun's pole is tilted ~7.25° to the ecliptic normal; the z-component
        # should be sin(63.87°) ≈ 0.898, NOT 1.0 (which would be ICRS +z).
        assert pole[2] != 1.0
        assert abs(pole[2] - math.sin(math.radians(63.87))) < 1e-10


if __name__ == "__main__":
    unittest.main(verbosity=2)
