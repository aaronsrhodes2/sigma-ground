"""
Tests for the diffusion module.

Test structure:
  1. Solid-state diffusion — Arrhenius, known values
  2. Fick's laws — flux, concentration profiles
  3. Diffusion length — penetration estimates
  4. Thermal diffusivity — κ/(ρc_p)
  5. Einstein-Stokes — liquid diffusion
  6. σ-dependence — diffusion shift through nuclear mass
  7. Nagatha export — complete format
"""

import math
import unittest

from .diffusion import (
    activation_energy_ev,
    solid_diffusivity,
    ficks_first_law,
    ficks_second_law_erf,
    diffusion_length,
    time_to_penetrate,
    thermal_diffusivity,
    einstein_stokes_diffusivity,
    darken_interdiffusion,
    sigma_diffusion_shift,
    material_diffusion_properties,
    DIFFUSION_DATA,
)
from .surface import MATERIALS


class TestActivationEnergy(unittest.TestCase):
    """Activation energy — measured values and σ-scaling."""

    def test_positive(self):
        """E_a is positive for all materials."""
        for mat in DIFFUSION_DATA:
            E_a = activation_energy_ev(mat)
            self.assertGreater(E_a, 0, f"{mat}: E_a must be positive")

    def test_known_iron(self):
        """Iron self-diffusion E_a ≈ 2.87 eV (Mehrer 2007)."""
        E_a = activation_energy_ev('iron')
        self.assertAlmostEqual(E_a, 2.87, places=2)

    def test_known_copper(self):
        """Copper E_a ≈ 2.19 eV (Rothman & Peterson 1969)."""
        E_a = activation_energy_ev('copper')
        self.assertAlmostEqual(E_a, 2.19, places=2)

    def test_covalent_high_Ea(self):
        """Covalent materials (Si, W) have high E_a (> 4 eV)."""
        E_a_si = activation_energy_ev('silicon')
        E_a_w = activation_energy_ev('tungsten')
        self.assertGreater(E_a_si, 4.0)
        self.assertGreater(E_a_w, 4.0)

    def test_sigma_increases_Ea(self):
        """E_a increases with σ (stiffer lattice)."""
        for mat in DIFFUSION_DATA:
            E_0 = activation_energy_ev(mat, 0.0)
            E_s = activation_energy_ev(mat, 0.1)
            self.assertGreater(E_s, E_0,
                f"{mat}: E_a should increase with σ")


class TestSolidDiffusivity(unittest.TestCase):
    """Arrhenius diffusion coefficient."""

    def test_positive(self):
        """D is always positive."""
        for mat in DIFFUSION_DATA:
            D = solid_diffusivity(mat, T=1000.0)
            self.assertGreater(D, 0)

    def test_increases_with_temperature(self):
        """D increases with T (Arrhenius)."""
        for mat in DIFFUSION_DATA:
            D_low = solid_diffusivity(mat, T=800.0)
            D_high = solid_diffusivity(mat, T=1200.0)
            self.assertGreater(D_high, D_low,
                f"{mat}: D should increase with temperature")

    def test_iron_1000K_order(self):
        """Iron at 1000K: D ~ 10⁻¹⁷ to 10⁻¹⁵ m²/s."""
        D = solid_diffusivity('iron', T=1000.0)
        self.assertGreater(D, 1e-20)
        self.assertLess(D, 1e-12)

    def test_aluminum_higher_than_iron(self):
        """Aluminum diffuses faster than iron at same T (lower E_a)."""
        D_al = solid_diffusivity('aluminum', T=800.0)
        D_fe = solid_diffusivity('iron', T=800.0)
        self.assertGreater(D_al, D_fe)

    def test_invalid_temperature(self):
        """T ≤ 0 raises ValueError."""
        with self.assertRaises(ValueError):
            solid_diffusivity('iron', T=0.0)
        with self.assertRaises(ValueError):
            solid_diffusivity('iron', T=-100.0)


class TestFicksLaws(unittest.TestCase):
    """Fick's first and second laws."""

    def test_first_law_sign(self):
        """Flux goes from high to low concentration (J < 0 for dC/dx > 0)."""
        J = ficks_first_law(1e-15, 1e20)  # positive gradient
        self.assertLess(J, 0)

    def test_first_law_magnitude(self):
        """J = -D × dC/dx."""
        D = 1e-14
        dC = 1e18
        J = ficks_first_law(D, dC)
        self.assertAlmostEqual(J, -D * dC, places=5)

    def test_second_law_surface(self):
        """At x=0: C = C_surface (boundary condition)."""
        C = ficks_second_law_erf(1.0, 0.0, 0.0, 1e-14, 3600.0)
        self.assertAlmostEqual(C, 1.0, places=10)

    def test_second_law_deep(self):
        """Deep inside: C → C_initial."""
        C = ficks_second_law_erf(1.0, 0.0, 1.0, 1e-14, 1.0)
        self.assertAlmostEqual(C, 0.0, places=6)

    def test_second_law_monotonic(self):
        """Concentration decreases with depth."""
        D = 1e-14
        t = 3600.0
        C_prev = 1.0
        for x in [0, 1e-6, 1e-5, 1e-4, 1e-3]:
            C = ficks_second_law_erf(1.0, 0.0, x, D, t)
            self.assertLessEqual(C, C_prev + 1e-10)
            C_prev = C

    def test_second_law_invalid_time(self):
        """t ≤ 0 raises ValueError."""
        with self.assertRaises(ValueError):
            ficks_second_law_erf(1.0, 0.0, 1e-6, 1e-14, 0.0)


class TestDiffusionLength(unittest.TestCase):
    """Characteristic diffusion length."""

    def test_formula(self):
        """L = √(Dt)."""
        D = 1e-14
        t = 3600.0
        L = diffusion_length(D, t)
        self.assertAlmostEqual(L, math.sqrt(D * t), places=15)

    def test_increases_with_time(self):
        """Longer time → deeper penetration."""
        D = 1e-14
        L1 = diffusion_length(D, 100.0)
        L2 = diffusion_length(D, 10000.0)
        self.assertGreater(L2, L1)

    def test_time_to_penetrate_inverse(self):
        """t = depth²/D (inverse of L = √(Dt))."""
        D = 1e-14
        depth = 1e-6
        t = time_to_penetrate(D, depth)
        self.assertAlmostEqual(t, depth ** 2 / D, places=5)

    def test_round_trip(self):
        """L → t → L round trip."""
        D = 1e-14
        t = 3600.0
        L = diffusion_length(D, t)
        t_back = time_to_penetrate(D, L)
        self.assertAlmostEqual(t_back, t, places=5)


class TestThermalDiffusivity(unittest.TestCase):
    """Thermal diffusivity α = κ/(ρc_p)."""

    def test_positive(self):
        """α > 0 for all materials."""
        for mat in DIFFUSION_DATA:
            if mat in MATERIALS:
                alpha = thermal_diffusivity(mat)
                self.assertGreater(alpha, 0,
                    f"{mat}: thermal diffusivity must be positive")

    def test_order_of_magnitude(self):
        """Metals: α ~ 10⁻⁶ to 10⁻⁴ m²/s."""
        alpha = thermal_diffusivity('iron')
        self.assertGreater(alpha, 1e-7)
        self.assertLess(alpha, 1e-3)

    def test_aluminum_high(self):
        """Aluminum has high thermal diffusivity (good conductor)."""
        alpha_al = thermal_diffusivity('aluminum')
        alpha_fe = thermal_diffusivity('iron')
        self.assertGreater(alpha_al, alpha_fe)


class TestEinsteinStokes(unittest.TestCase):
    """Einstein-Stokes diffusion in liquids."""

    def test_positive(self):
        """D > 0."""
        D = einstein_stokes_diffusivity(300.0, 1e-3, 1e-9)
        self.assertGreater(D, 0)

    def test_increases_with_temperature(self):
        """Higher T → faster diffusion."""
        D1 = einstein_stokes_diffusivity(300.0, 1e-3, 1e-9)
        D2 = einstein_stokes_diffusivity(350.0, 1e-3, 1e-9)
        self.assertGreater(D2, D1)

    def test_decreases_with_viscosity(self):
        """Higher viscosity → slower diffusion."""
        D1 = einstein_stokes_diffusivity(300.0, 1e-3, 1e-9)
        D2 = einstein_stokes_diffusivity(300.0, 2e-3, 1e-9)
        self.assertLess(D2, D1)

    def test_nanometer_particle_order(self):
        """1 nm particle in water at 300K: D ~ 10⁻¹⁰ m²/s."""
        D = einstein_stokes_diffusivity(300.0, 1e-3, 1e-9)
        self.assertGreater(D, 1e-12)
        self.assertLess(D, 1e-8)

    def test_invalid_inputs(self):
        """Negative viscosity or radius raises ValueError."""
        with self.assertRaises(ValueError):
            einstein_stokes_diffusivity(300.0, -1e-3, 1e-9)
        with self.assertRaises(ValueError):
            einstein_stokes_diffusivity(300.0, 1e-3, -1e-9)


class TestDarken(unittest.TestCase):
    """Darken interdiffusion."""

    def test_pure_A(self):
        """x_A=1, x_B=0 → D̃ = 0×D_A + 1×D_B = ... wait, D̃ = x_B D_A + x_A D_B."""
        D = darken_interdiffusion(1e-14, 2e-14, 1.0, 0.0)
        self.assertAlmostEqual(D, 2e-14, places=20)

    def test_equal_mix(self):
        """x_A=x_B=0.5: D̃ = average."""
        D = darken_interdiffusion(1e-14, 3e-14, 0.5, 0.5)
        self.assertAlmostEqual(D, 2e-14, places=20)


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts diffusion."""

    def test_shift_unity_at_zero(self):
        """D(σ=0)/D(0) = 1."""
        for mat in DIFFUSION_DATA:
            ratio = sigma_diffusion_shift(mat, T=1000.0, sigma=0.0)
            self.assertAlmostEqual(ratio, 1.0, places=10)

    def test_shift_decreases_with_sigma(self):
        """D(σ) < D(0) for σ > 0 (higher barrier, slower diffusion)."""
        for mat in DIFFUSION_DATA:
            ratio = sigma_diffusion_shift(mat, T=1000.0, sigma=0.1)
            self.assertLess(ratio, 1.0,
                f"{mat}: diffusivity should decrease with σ")

    def test_earth_sigma_negligible(self):
        """At Earth σ ~ 7×10⁻¹⁰: shift < 10⁻⁷."""
        ratio = sigma_diffusion_shift('iron', T=1000.0, sigma=7e-10)
        self.assertAlmostEqual(ratio, 1.0, places=7)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_all_materials_export(self):
        """All diffusion materials produce valid export dicts."""
        for mat in DIFFUSION_DATA:
            props = material_diffusion_properties(mat)
            self.assertIn('self_diffusivity_m2_s', props)
            self.assertIn('activation_energy_ev', props)
            self.assertIn('origin_tag', props)

    def test_sigma_propagates(self):
        """σ value appears in export."""
        props = material_diffusion_properties('iron', sigma=0.1)
        self.assertEqual(props['sigma'], 0.1)

    def test_honest_origin_tags(self):
        """Origin tag includes derivation info."""
        props = material_diffusion_properties('iron')
        self.assertIn('FIRST_PRINCIPLES', props['origin_tag'])
        self.assertIn('MEASURED', props['origin_tag'])


if __name__ == '__main__':
    unittest.main()
