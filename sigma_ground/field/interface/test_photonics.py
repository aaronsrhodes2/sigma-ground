"""
Tests for the photonics module.

Test structure:
  1. Waveguide fundamentals — NA, critical angle, V-number
  2. Fiber modes — single-mode cutoff, mode counting
  3. Photonic bandgap — Bragg wavelength, bandwidth, reflectance
  4. χ² nonlinear optics — SHG, phase matching
  5. χ³ Kerr effect — self-focusing, B-integral
  6. Absorption edge — direct and indirect gaps
  7. σ-dependence — Bragg shift through lattice
  8. Nagatha export
"""

import math
import unittest

from .photonics import (
    numerical_aperture,
    critical_angle,
    v_number_slab,
    v_number_fiber,
    is_single_mode_fiber,
    number_of_modes_fiber,
    slab_modes_count,
    bragg_wavelength,
    bragg_bandwidth_fraction,
    bragg_reflectance,
    shg_phase_mismatch,
    shg_efficiency_factor,
    kerr_refractive_index,
    self_focusing_critical_power,
    nonlinear_phase_shift,
    absorption_coefficient_direct,
    absorption_coefficient_indirect,
    sigma_bragg_shift,
    waveguide_properties,
    NONLINEAR_CRYSTALS,
    KERR_MATERIALS,
)


class TestNumericalAperture(unittest.TestCase):
    """NA = √(n_core² − n_clad²)."""

    def test_standard_fiber(self):
        """SMF-28: NA ≈ 0.12 (n_core=1.4681, n_clad=1.4629)."""
        NA = numerical_aperture(1.4681, 1.4629)
        self.assertAlmostEqual(NA, 0.12, delta=0.02)

    def test_multimode_fiber(self):
        """Multimode fiber: NA ≈ 0.22."""
        NA = numerical_aperture(1.48, 1.46)
        self.assertGreater(NA, 0.15)
        self.assertLess(NA, 0.35)

    def test_higher_contrast_higher_NA(self):
        """Larger index difference → higher NA."""
        NA1 = numerical_aperture(1.50, 1.48)
        NA2 = numerical_aperture(1.50, 1.40)
        self.assertGreater(NA2, NA1)

    def test_invalid_indices(self):
        """n_core ≤ n_clad raises ValueError."""
        with self.assertRaises(ValueError):
            numerical_aperture(1.45, 1.50)


class TestCriticalAngle(unittest.TestCase):
    """Total internal reflection."""

    def test_glass_air(self):
        """Glass (n=1.5) / air (n=1.0): θ_c ≈ 41.8°."""
        theta = critical_angle(1.5, 1.0)
        self.assertAlmostEqual(math.degrees(theta), 41.8, delta=0.5)

    def test_increases_with_clad_index(self):
        """Higher cladding index → larger critical angle."""
        t1 = critical_angle(1.5, 1.0)
        t2 = critical_angle(1.5, 1.3)
        self.assertGreater(t2, t1)


class TestVNumber(unittest.TestCase):
    """V-number for waveguides."""

    def test_slab_positive(self):
        """V > 0 for valid waveguide."""
        V = v_number_slab(1e-6, 1.55e-6, 1.50, 1.45)
        self.assertGreater(V, 0)

    def test_fiber_smf28(self):
        """SMF-28 at 1550 nm: V ≈ 2.0-2.2 (single mode)."""
        # Core radius 4.1 μm, n_core=1.4681, n_clad=1.4629
        V = v_number_fiber(4.1e-6, 1.55e-6, 1.4681, 1.4629)
        self.assertGreater(V, 1.5)
        self.assertLess(V, 2.5)

    def test_single_mode_smf28(self):
        """SMF-28 is single-mode at 1550 nm."""
        self.assertTrue(
            is_single_mode_fiber(4.1e-6, 1.55e-6, 1.4681, 1.4629))

    def test_multimode_at_visible(self):
        """50 μm core at 633 nm: many modes."""
        N = number_of_modes_fiber(25e-6, 633e-9, 1.48, 1.46)
        self.assertGreater(N, 10)


class TestSlabModes(unittest.TestCase):
    """Slab waveguide mode counting."""

    def test_at_least_one(self):
        """Always at least 1 guided mode."""
        m = slab_modes_count(0.1e-6, 1.55e-6, 1.50, 1.45)
        self.assertGreaterEqual(m, 1)

    def test_more_modes_thicker(self):
        """Thicker slab → more modes."""
        m1 = slab_modes_count(1e-6, 1.55e-6, 1.50, 1.45)
        m2 = slab_modes_count(5e-6, 1.55e-6, 1.50, 1.45)
        self.assertGreaterEqual(m2, m1)


class TestBraggStack(unittest.TestCase):
    """Photonic bandgap — Bragg reflector."""

    def test_quarter_wave_wavelength(self):
        """Quarter-wave stack at 550 nm green light."""
        # For λ=550 nm: n₁d₁ + n₂d₂ = λ/2 = 275 nm
        # Quarter-wave: n₁d₁ = n₂d₂ = λ/4 = 137.5 nm
        d1 = 137.5e-9 / 1.46  # SiO₂
        d2 = 137.5e-9 / 2.30  # TiO₂
        lam = bragg_wavelength(1.46, d1, 2.30, d2)
        self.assertAlmostEqual(lam * 1e9, 550, delta=1)

    def test_bandwidth_increases_with_contrast(self):
        """Higher index contrast → wider bandgap."""
        bw1 = bragg_bandwidth_fraction(1.46, 1.50)  # low contrast
        bw2 = bragg_bandwidth_fraction(1.46, 2.30)  # high contrast
        self.assertGreater(bw2, bw1)

    def test_bandwidth_zero_same_index(self):
        """Same indices → zero bandwidth."""
        bw = bragg_bandwidth_fraction(1.5, 1.5)
        self.assertAlmostEqual(bw, 0.0, places=10)

    def test_reflectance_increases_with_pairs(self):
        """More layer pairs → higher reflectance."""
        R1 = bragg_reflectance(1.46, 2.30, 3)
        R2 = bragg_reflectance(1.46, 2.30, 10)
        self.assertGreater(R2, R1)

    def test_reflectance_approaches_unity(self):
        """Many pairs → R → 1."""
        R = bragg_reflectance(1.46, 2.30, 20)
        self.assertGreater(R, 0.999)


class TestSHG(unittest.TestCase):
    """Second harmonic generation — χ²."""

    def test_phase_mismatch_sign(self):
        """For normal dispersion (n_2ω > n_ω): Δk < 0."""
        dk = shg_phase_mismatch(1.50, 1.52, 1064e-9)
        self.assertLess(dk, 0)

    def test_phase_matched(self):
        """Perfect phase matching: Δk = 0 when n_ω = n_2ω."""
        dk = shg_phase_mismatch(1.50, 1.50, 1064e-9)
        self.assertAlmostEqual(dk, 0.0, places=10)

    def test_efficiency_increases_with_length(self):
        """Longer crystal → higher efficiency (before walk-off)."""
        eff1 = shg_efficiency_factor(3.0, 0.01, 1.5, 1.5, 1064e-9)
        eff2 = shg_efficiency_factor(3.0, 0.02, 1.5, 1.5, 1064e-9)
        self.assertGreater(eff2, eff1)

    def test_efficiency_increases_with_d(self):
        """Higher d_eff → higher efficiency."""
        eff1 = shg_efficiency_factor(1.0, 0.01, 1.5, 1.5, 1064e-9)
        eff2 = shg_efficiency_factor(4.0, 0.01, 1.5, 1.5, 1064e-9)
        self.assertGreater(eff2, eff1)

    def test_all_crystals_have_data(self):
        """All nonlinear crystals have required fields."""
        for key, data in NONLINEAR_CRYSTALS.items():
            self.assertIn('d_eff_pm_V', data)
            self.assertIn('n_omega', data)
            self.assertIn('n_2omega', data)
            self.assertGreater(data['d_eff_pm_V'], 0)


class TestKerr(unittest.TestCase):
    """χ³ Kerr effect."""

    def test_zero_intensity_linear(self):
        """At I=0: n = n₀."""
        n = kerr_refractive_index(1.45, 2.7e-20, 0)
        self.assertAlmostEqual(n, 1.45, places=10)

    def test_positive_n2(self):
        """Positive n₂: index increases with intensity."""
        n0 = kerr_refractive_index(1.45, 2.7e-20, 0)
        n_high = kerr_refractive_index(1.45, 2.7e-20, 1e16)
        self.assertGreater(n_high, n0)

    def test_critical_power_silica(self):
        """Silica at 1064 nm: P_cr ~ 4 MW."""
        P = self_focusing_critical_power(1064e-9, 1.45, 2.7e-20)
        # P_cr ≈ 3-5 MW for silica
        self.assertGreater(P, 1e6)
        self.assertLess(P, 10e6)

    def test_b_integral(self):
        """B-integral is proportional to I×L."""
        phi1 = nonlinear_phase_shift(2.7e-20, 1e14, 0.1, 1064e-9)
        phi2 = nonlinear_phase_shift(2.7e-20, 1e14, 0.2, 1064e-9)
        self.assertAlmostEqual(phi2 / phi1, 2.0, places=10)

    def test_all_kerr_materials(self):
        """All Kerr materials have required fields."""
        for key, data in KERR_MATERIALS.items():
            self.assertIn('n0', data)
            self.assertIn('n2_m2_W', data)
            self.assertGreater(data['n2_m2_W'], 0)


class TestAbsorptionEdge(unittest.TestCase):
    """Optical absorption near bandgap."""

    def test_below_gap_zero(self):
        """Below bandgap: α = 0 (transparent)."""
        alpha = absorption_coefficient_direct(1.0, 1.5)
        self.assertEqual(alpha, 0.0)

    def test_above_gap_positive(self):
        """Above bandgap: α > 0 (absorbing)."""
        alpha = absorption_coefficient_direct(2.0, 1.5)
        self.assertGreater(alpha, 0)

    def test_direct_sqrt_dependence(self):
        """Direct gap: α ∝ √(E − E_g)."""
        a1 = absorption_coefficient_direct(2.0, 1.0)  # ΔE = 1.0
        a2 = absorption_coefficient_direct(5.0, 1.0)  # ΔE = 4.0
        self.assertAlmostEqual(a2 / a1, 2.0, places=5)

    def test_indirect_quadratic(self):
        """Indirect gap: α ∝ (E − E_g)²."""
        a1 = absorption_coefficient_indirect(2.0, 1.0)  # ΔE = 1.0
        a2 = absorption_coefficient_indirect(3.0, 1.0)  # ΔE = 2.0
        self.assertAlmostEqual(a2 / a1, 4.0, places=5)


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts Bragg wavelength through lattice."""

    def test_zero_sigma_unchanged(self):
        """σ=0: no shift."""
        d1 = 100e-9
        d2 = 80e-9
        lam_0, lam_s = sigma_bragg_shift(1.46, d1, 2.30, d2, 0.0)
        self.assertAlmostEqual(lam_0, lam_s, places=15)

    def test_positive_sigma_redshifts(self):
        """σ > 0 expands lattice → redshift."""
        d1 = 100e-9
        d2 = 80e-9
        lam_0, lam_s = sigma_bragg_shift(1.46, d1, 2.30, d2, 0.1)
        self.assertGreater(lam_s, lam_0)

    def test_earth_sigma_negligible(self):
        """At Earth σ: shift < 10⁻⁸ nm."""
        d1 = 100e-9
        d2 = 80e-9
        lam_0, lam_s = sigma_bragg_shift(1.46, d1, 2.30, d2, 7e-10)
        self.assertAlmostEqual(lam_0, lam_s, places=15)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_waveguide_export(self):
        """Waveguide export includes all required fields."""
        props = waveguide_properties(4.1e-6, 1.55e-6, 1.4681, 1.4629)
        self.assertIn('numerical_aperture', props)
        self.assertIn('v_number', props)
        self.assertIn('n_modes', props)
        self.assertIn('single_mode', props)
        self.assertIn('critical_angle_deg', props)
        self.assertIn('origin_tag', props)


if __name__ == '__main__':
    unittest.main()
