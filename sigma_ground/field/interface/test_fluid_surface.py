"""Tests for free-surface fluid dynamics (buoyancy, wind stress, ripples).

Validated against textbook/closed-form values (Golden Rule 8). Water seeds come
from liquid_water.py (NIST-anchored), so these also guard the seed fix.
"""
import math
import unittest

from .fluid_surface import (
    STANDARD_GRAVITY,
    submerged_fraction, floats,
    wind_shear_stress, friction_velocity,
    capillary_gravity_phase_speed, capillary_gravity_angular_frequency,
    capillary_gravity_group_speed, capillary_gravity_min_phase_speed,
)
from .liquid_water import water_surface_tension, water_density

_T = 293.15                       # 20 °C
_SIGMA = water_surface_tension(_T)
_RHO = water_density(_T)


class TestBuoyancy(unittest.TestCase):
    def test_copper_sinks(self):
        self.assertFalse(floats(8960.0, _RHO))
        self.assertEqual(submerged_fraction(8960.0, _RHO), 1.0)

    def test_ice_floats_about_92_percent_submerged(self):
        self.assertTrue(floats(917.0, _RHO))
        self.assertAlmostEqual(submerged_fraction(917.0, _RHO), 0.919, delta=0.01)

    def test_wood_floats_partly(self):
        f = submerged_fraction(600.0, _RHO)
        self.assertTrue(floats(600.0, _RHO))
        self.assertAlmostEqual(f, 0.601, delta=0.01)


class TestWindStress(unittest.TestCase):
    def test_stress_at_5_m_s(self):
        tau = wind_shear_stress(5.0)
        self.assertAlmostEqual(tau, 0.0398, delta=0.002)  # C_d ρ_air U²

    def test_scales_as_velocity_squared(self):
        self.assertAlmostEqual(wind_shear_stress(10.0) / wind_shear_stress(5.0),
                               4.0, delta=1e-6)

    def test_friction_velocity(self):
        tau = wind_shear_stress(5.0)
        u = friction_velocity(tau, _RHO)
        self.assertAlmostEqual(u, math.sqrt(tau / _RHO), places=9)
        self.assertGreater(u, 0.0)


class TestCapillaryGravityWaves(unittest.TestCase):
    def test_min_phase_speed_water(self):
        """Capillary-gravity minimum: ~0.23 m/s at ~1.7 cm (textbook)."""
        c_min, lam_min = capillary_gravity_min_phase_speed(_SIGMA, _RHO)
        self.assertAlmostEqual(c_min, 0.231, delta=0.004)
        self.assertAlmostEqual(lam_min, 0.0171, delta=0.0005)

    def test_phase_speed_at_min_equals_c_min(self):
        c_min, lam_min = capillary_gravity_min_phase_speed(_SIGMA, _RHO)
        c = capillary_gravity_phase_speed(lam_min, _SIGMA, _RHO)
        self.assertAlmostEqual(c, c_min, places=6)

    def test_gravity_wave_limit(self):
        """Long λ → pure gravity wave c = √(gλ/2π); at 1 m ≈ 1.25 m/s."""
        c = capillary_gravity_phase_speed(1.0, _SIGMA, _RHO)
        self.assertAlmostEqual(c, math.sqrt(STANDARD_GRAVITY * 1.0 / (2 * math.pi)),
                               delta=0.01)

    def test_phase_speed_has_a_minimum(self):
        """c(λ) rises on both sides of λ_min (capillary up, gravity up)."""
        c_min, lam_min = capillary_gravity_min_phase_speed(_SIGMA, _RHO)
        self.assertGreater(capillary_gravity_phase_speed(lam_min / 5, _SIGMA, _RHO), c_min)
        self.assertGreater(capillary_gravity_phase_speed(lam_min * 5, _SIGMA, _RHO), c_min)

    def test_dispersion_consistency(self):
        """ω = c · k at any wavelength."""
        lam = 0.05
        k = 2 * math.pi / lam
        omega = capillary_gravity_angular_frequency(lam, _SIGMA, _RHO)
        c = capillary_gravity_phase_speed(lam, _SIGMA, _RHO)
        self.assertAlmostEqual(omega, c * k, places=6)

    def test_group_speed_positive(self):
        for lam in (0.005, 0.0171, 0.1, 1.0):
            self.assertGreater(capillary_gravity_group_speed(lam, _SIGMA, _RHO), 0.0)


if __name__ == "__main__":
    unittest.main()
