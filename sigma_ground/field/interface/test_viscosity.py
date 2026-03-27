"""
Tests for the viscosity module.

Test structure:
  1. Stokes drag — force on sphere
  2. Terminal velocity — force balance
  3. Drag coefficient — Re-dependent regimes
  4. Poiseuille flow — pipe flow rate
  5. Viscous dissipation — energy loss
  6. Nabarro-Herring creep — diffusion-controlled
  7. Boundary layer — Blasius thickness
  8. Nagatha export
"""

import math
import unittest

from .viscosity import (
    stokes_drag,
    terminal_velocity_stokes,
    particle_reynolds_number,
    drag_coefficient_sphere,
    general_drag_force,
    poiseuille_flow_rate,
    poiseuille_max_velocity,
    viscous_dissipation_simple_shear,
    viscous_heating_temperature_rise,
    nabarro_herring_strain_rate,
    boundary_layer_thickness,
    wall_shear_stress,
    viscous_flow_properties,
)


class TestStokesDrag(unittest.TestCase):
    """Stokes drag force F = 6πηrv."""

    def test_positive(self):
        """Drag force is always positive."""
        F = stokes_drag(1e-3, 1e-3, 1.0)
        self.assertGreater(F, 0)

    def test_formula(self):
        """F = 6πηrv."""
        eta = 1e-3
        r = 1e-4
        v = 0.01
        F = stokes_drag(eta, r, v)
        expected = 6 * math.pi * eta * r * v
        self.assertAlmostEqual(F, expected, places=15)

    def test_proportional_to_velocity(self):
        """Doubling velocity doubles drag."""
        F1 = stokes_drag(1e-3, 1e-3, 1.0)
        F2 = stokes_drag(1e-3, 1e-3, 2.0)
        self.assertAlmostEqual(F2 / F1, 2.0, places=10)

    def test_proportional_to_radius(self):
        """Doubling radius doubles drag."""
        F1 = stokes_drag(1e-3, 1e-4, 1.0)
        F2 = stokes_drag(1e-3, 2e-4, 1.0)
        self.assertAlmostEqual(F2 / F1, 2.0, places=10)

    def test_water_sphere_known(self):
        """1 mm sphere at 1 cm/s in water: F ~ 10⁻⁷ N."""
        F = stokes_drag(1e-3, 0.5e-3, 0.01)
        self.assertGreater(F, 1e-9)
        self.assertLess(F, 1e-5)


class TestTerminalVelocity(unittest.TestCase):
    """Terminal velocity — Stokes settling."""

    def test_sinking(self):
        """Dense particle in lighter fluid sinks (v > 0)."""
        v = terminal_velocity_stokes(1e-4, 8000, 1000, 1e-3)
        self.assertGreater(v, 0)

    def test_rising(self):
        """Light particle in denser fluid rises (v < 0)."""
        v = terminal_velocity_stokes(1e-4, 500, 1000, 1e-3)
        self.assertLess(v, 0)

    def test_neutral_buoyancy(self):
        """Equal density: v = 0."""
        v = terminal_velocity_stokes(1e-4, 1000, 1000, 1e-3)
        self.assertAlmostEqual(v, 0.0, places=15)

    def test_scales_with_r_squared(self):
        """v_t ∝ r²."""
        v1 = terminal_velocity_stokes(1e-4, 3000, 1000, 1e-3)
        v2 = terminal_velocity_stokes(2e-4, 3000, 1000, 1e-3)
        self.assertAlmostEqual(v2 / v1, 4.0, places=10)

    def test_invalid_viscosity(self):
        """Zero or negative viscosity raises ValueError."""
        with self.assertRaises(ValueError):
            terminal_velocity_stokes(1e-4, 3000, 1000, 0)


class TestDragCoefficient(unittest.TestCase):
    """Drag coefficient regimes."""

    def test_stokes_regime(self):
        """Re < 1: C_D = 24/Re."""
        Re = 0.1
        C_D = drag_coefficient_sphere(Re)
        self.assertAlmostEqual(C_D, 24.0 / Re, places=10)

    def test_transition_regime(self):
        """1 < Re < 1000: Schiller-Naumann."""
        Re = 100
        C_D = drag_coefficient_sphere(Re)
        # Should be between 24/Re and 0.44
        self.assertGreater(C_D, 0.24)
        self.assertLess(C_D, 2.0)

    def test_newton_regime(self):
        """Re > 1000: C_D ≈ 0.44."""
        C_D = drag_coefficient_sphere(5000)
        self.assertAlmostEqual(C_D, 0.44, places=2)

    def test_decreases_then_levels(self):
        """C_D decreases from Stokes through transition."""
        CD_low = drag_coefficient_sphere(0.01)
        CD_mid = drag_coefficient_sphere(10)
        CD_high = drag_coefficient_sphere(10000)
        self.assertGreater(CD_low, CD_mid)
        self.assertGreater(CD_mid, CD_high)

    def test_invalid_Re(self):
        """Re ≤ 0 raises ValueError."""
        with self.assertRaises(ValueError):
            drag_coefficient_sphere(0)
        with self.assertRaises(ValueError):
            drag_coefficient_sphere(-1)


class TestPoiseuilleFlow(unittest.TestCase):
    """Hagen-Poiseuille pipe flow."""

    def test_positive_flow(self):
        """Positive ΔP gives positive flow rate."""
        Q = poiseuille_flow_rate(0.01, 1000, 1e-3, 1.0)
        self.assertGreater(Q, 0)

    def test_r4_dependence(self):
        """Q ∝ R⁴."""
        Q1 = poiseuille_flow_rate(0.01, 1000, 1e-3, 1.0)
        Q2 = poiseuille_flow_rate(0.02, 1000, 1e-3, 1.0)
        self.assertAlmostEqual(Q2 / Q1, 16.0, places=5)

    def test_inverse_viscosity(self):
        """Q ∝ 1/η."""
        Q1 = poiseuille_flow_rate(0.01, 1000, 1e-3, 1.0)
        Q2 = poiseuille_flow_rate(0.01, 1000, 2e-3, 1.0)
        self.assertAlmostEqual(Q1 / Q2, 2.0, places=10)

    def test_max_velocity_formula(self):
        """v_max = 2 × v_avg = 2Q/(πR²)."""
        R = 0.01
        dP = 1000
        eta = 1e-3
        L = 1.0
        Q = poiseuille_flow_rate(R, dP, eta, L)
        v_avg = Q / (math.pi * R ** 2)
        v_max = poiseuille_max_velocity(R, dP, eta, L)
        self.assertAlmostEqual(v_max / v_avg, 2.0, places=10)

    def test_invalid_inputs(self):
        """Invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(0.01, 1000, 0, 1.0)
        with self.assertRaises(ValueError):
            poiseuille_flow_rate(0.01, 1000, 1e-3, 0)


class TestViscousDissipation(unittest.TestCase):
    """Viscous dissipation rate."""

    def test_positive(self):
        """Dissipation is always positive (irreversible)."""
        Phi = viscous_dissipation_simple_shear(1e-3, 100)
        self.assertGreater(Phi, 0)

    def test_quadratic_in_shear_rate(self):
        """Φ ∝ (du/dy)²."""
        Phi1 = viscous_dissipation_simple_shear(1e-3, 100)
        Phi2 = viscous_dissipation_simple_shear(1e-3, 200)
        self.assertAlmostEqual(Phi2 / Phi1, 4.0, places=10)

    def test_heating(self):
        """Viscous heating gives positive ΔT."""
        dT = viscous_heating_temperature_rise(
            viscosity=1.0, shear_rate=100,
            time=10.0, density=1000, specific_heat=4000)
        self.assertGreater(dT, 0)


class TestNabarroHerringCreep(unittest.TestCase):
    """Nabarro-Herring diffusion creep."""

    def test_positive(self):
        """Strain rate is positive."""
        eps_dot = nabarro_herring_strain_rate(
            diffusivity=1e-15, stress=10e6,
            atomic_volume=1.2e-29, grain_size=50e-6, T=1000)
        self.assertGreater(eps_dot, 0)

    def test_proportional_to_stress(self):
        """ε̇ ∝ σ (Newtonian viscous creep)."""
        e1 = nabarro_herring_strain_rate(1e-15, 10e6, 1.2e-29, 50e-6, 1000)
        e2 = nabarro_herring_strain_rate(1e-15, 20e6, 1.2e-29, 50e-6, 1000)
        self.assertAlmostEqual(e2 / e1, 2.0, places=10)

    def test_inverse_grain_size_squared(self):
        """ε̇ ∝ 1/d²."""
        e1 = nabarro_herring_strain_rate(1e-15, 10e6, 1.2e-29, 50e-6, 1000)
        e2 = nabarro_herring_strain_rate(1e-15, 10e6, 1.2e-29, 100e-6, 1000)
        self.assertAlmostEqual(e1 / e2, 4.0, places=10)

    def test_invalid_temperature(self):
        """T ≤ 0 raises ValueError."""
        with self.assertRaises(ValueError):
            nabarro_herring_strain_rate(1e-15, 10e6, 1.2e-29, 50e-6, 0)


class TestBoundaryLayer(unittest.TestCase):
    """Blasius boundary layer."""

    def test_positive(self):
        """Boundary layer thickness is positive."""
        delta = boundary_layer_thickness(0.1, 1.2, 10, 1.8e-5)
        self.assertGreater(delta, 0)

    def test_grows_with_x(self):
        """δ grows downstream (∝ √x)."""
        d1 = boundary_layer_thickness(0.1, 1.2, 10, 1.8e-5)
        d2 = boundary_layer_thickness(0.4, 1.2, 10, 1.8e-5)
        self.assertGreater(d2, d1)

    def test_wall_shear_decreases(self):
        """Wall shear stress decreases downstream."""
        tau1 = wall_shear_stress(1.2, 10, 1.8e-5, 0.1)
        tau2 = wall_shear_stress(1.2, 10, 1.8e-5, 0.5)
        self.assertGreater(tau1, tau2)

    def test_invalid_inputs(self):
        """Nonpositive inputs raise ValueError."""
        with self.assertRaises(ValueError):
            boundary_layer_thickness(0, 1.2, 10, 1.8e-5)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_basic_properties(self):
        """Basic export includes required fields."""
        props = viscous_flow_properties(
            viscosity=1e-3, rho_fluid=1000, velocity=1.0,
            particle_radius=1e-3, rho_particle=3000)
        self.assertIn('stokes_drag_N', props)
        self.assertIn('terminal_velocity_m_s', props)
        self.assertIn('origin_tag', props)

    def test_pipe_flow(self):
        """Pipe flow export includes Poiseuille quantities."""
        props = viscous_flow_properties(
            viscosity=1e-3, rho_fluid=1000, velocity=1.0,
            pipe_radius=0.01, pipe_length=1.0)
        self.assertIn('poiseuille_flow_rate_m3_s', props)


if __name__ == '__main__':
    unittest.main()
