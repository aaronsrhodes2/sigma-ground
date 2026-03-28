"""
Tests for atomic_spectra.py — energy levels, spectral lines, selection rules.

Strategy:
  - Test Rydberg energy ≈ 13.6 eV (MEASURED)
  - Test Rydberg constant ≈ 1.0974e7 m⁻¹ (MEASURED)
  - Test hydrogen ionization energy ≈ 13.598 eV (MEASURED, with reduced mass)
  - Test Balmer series: Hα ≈ 656 nm, Hβ ≈ 486 nm (MEASURED)
  - Test He⁺ (Z=2) has 4× the energy of hydrogen
  - Test fine structure splitting of 2p (MEASURED)
  - Test Landé g-factor special cases
  - Test Zeeman splitting count
  - Test selection rules
  - Test QHO equally spaced levels
  - Test σ-dependence through reduced mass
  - Test Rule 9: full_report covers all fields

Reference values:
  Rydberg energy:       13.6057 eV (infinite mass)
  Hydrogen IE:          13.5984 eV (with reduced mass)
  Hα (Balmer):          656.28 nm
  Hβ:                   486.13 nm
  Hγ:                   434.05 nm
  Lyman α:              121.57 nm
  2p fine structure:    ~4.5×10⁻⁵ eV
  Rydberg constant R∞:  1.0974e7 m⁻¹
"""

import math
import unittest

from sigma_ground.field.interface.atomic_spectra import (
    RYDBERG_ENERGY_EV,
    RYDBERG_CONSTANT,
    hydrogen_energy_eV,
    hydrogen_like_energy_eV,
    hydrogen_reduced_mass,
    ionization_energy_hydrogen_eV,
    transition_energy_eV,
    transition_wavelength_nm,
    transition_frequency_Hz,
    transition_wavenumber,
    lyman_series,
    balmer_series,
    paschen_series,
    series_limit_nm,
    multi_electron_energy_eV,
    fine_structure_shift_eV,
    fine_structure_splitting_eV,
    lande_g_factor,
    zeeman_shift_eV,
    zeeman_splitting_count,
    zeeman_pattern,
    is_allowed_transition,
    allowed_transitions,
    qho_energy_eV,
    qho_zero_point_energy_eV,
    qho_transition_energy_eV,
    qho_classical_amplitude,
    qho_level_spacing_eV,
    rydberg_constant_at_sigma,
    sigma_spectral_shift,
    is_visible,
    visible_lines,
    wavelength_to_rgb,
    emission_spectrum,
    atomic_spectra_report,
    full_report,
)


class TestRydbergConstant(unittest.TestCase):
    """Rydberg energy and constant validation."""

    def test_rydberg_energy(self):
        """E_R ≈ 13.6 eV (MEASURED: 13.6057 eV infinite mass)."""
        self.assertAlmostEqual(RYDBERG_ENERGY_EV, 13.6057, delta=0.01)

    def test_rydberg_constant(self):
        """R∞ ≈ 1.0974e7 m⁻¹ (MEASURED)."""
        self.assertAlmostEqual(RYDBERG_CONSTANT / 1e7, 1.0974, delta=0.001)

    def test_rydberg_derived(self):
        """R∞ = E_R/(hc) — self-consistent."""
        from sigma_ground.field.constants import H_PLANCK, C
        R_check = RYDBERG_ENERGY_EV * 1.602176634e-19 / (H_PLANCK * C)
        self.assertAlmostEqual(RYDBERG_CONSTANT, R_check, delta=1e3)


class TestHydrogenLevels(unittest.TestCase):
    """Hydrogen atom energy levels."""

    def test_ground_state(self):
        """E₁ ≈ −13.6 eV (with reduced mass: −13.598 eV)."""
        E1 = hydrogen_energy_eV(1)
        self.assertAlmostEqual(E1, -13.598, delta=0.02)

    def test_n2(self):
        """E₂ = E₁/4."""
        E1 = hydrogen_energy_eV(1)
        E2 = hydrogen_energy_eV(2)
        self.assertAlmostEqual(E2, E1 / 4, delta=0.001)

    def test_negative(self):
        """All bound states have negative energy."""
        for n in range(1, 10):
            self.assertLess(hydrogen_energy_eV(n), 0)

    def test_approaches_zero(self):
        """E_n → 0 as n → ∞."""
        E100 = hydrogen_energy_eV(100)
        self.assertAlmostEqual(E100, 0.0, delta=0.01)

    def test_ionization_energy(self):
        """IE of hydrogen ≈ 13.598 eV (MEASURED: 13.5984 eV)."""
        IE = ionization_energy_hydrogen_eV()
        self.assertAlmostEqual(IE, 13.598, delta=0.02)

    def test_reduced_mass_correction(self):
        """Reduced mass < bare electron mass (finite proton)."""
        mu = hydrogen_reduced_mass()
        from sigma_ground.field.constants import M_ELECTRON_KG
        self.assertLess(mu, M_ELECTRON_KG)
        # μ/m_e ≈ 0.99946
        self.assertAlmostEqual(mu / M_ELECTRON_KG, 0.99946, delta=0.0001)


class TestHydrogenLike(unittest.TestCase):
    """Hydrogen-like ions (He⁺, Li²⁺, etc.)."""

    def test_He_plus(self):
        """He⁺ ground state = 4 × H ground state."""
        E_H = hydrogen_like_energy_eV(1, 1)
        E_He = hydrogen_like_energy_eV(2, 1)
        self.assertAlmostEqual(E_He / E_H, 4.0, delta=0.01)

    def test_Li2_plus(self):
        """Li²⁺ ground state = 9 × H ground state."""
        E_H = hydrogen_like_energy_eV(1, 1)
        E_Li = hydrogen_like_energy_eV(3, 1)
        self.assertAlmostEqual(E_Li / E_H, 9.0, delta=0.01)


class TestSpectralLines(unittest.TestCase):
    """Spectral line wavelengths."""

    def test_lyman_alpha(self):
        """Lyman α (2→1) ≈ 121.6 nm (MEASURED)."""
        lam = transition_wavelength_nm(1, 2, 1)
        self.assertAlmostEqual(lam, 121.6, delta=0.5)

    def test_balmer_alpha(self):
        """Hα (3→2) ≈ 656.3 nm (MEASURED: 656.28 nm)."""
        lam = transition_wavelength_nm(1, 3, 2)
        self.assertAlmostEqual(lam, 656.3, delta=1.0)

    def test_balmer_beta(self):
        """Hβ (4→2) ≈ 486.1 nm (MEASURED)."""
        lam = transition_wavelength_nm(1, 4, 2)
        self.assertAlmostEqual(lam, 486.1, delta=1.0)

    def test_balmer_gamma(self):
        """Hγ (5→2) ≈ 434.0 nm (MEASURED)."""
        lam = transition_wavelength_nm(1, 5, 2)
        self.assertAlmostEqual(lam, 434.0, delta=1.0)

    def test_energy_positive(self):
        """Emission energy is positive."""
        E = transition_energy_eV(1, 3, 2)
        self.assertGreater(E, 0)

    def test_frequency_consistent(self):
        """ν = E/h."""
        E = transition_energy_eV(1, 3, 2)
        freq = transition_frequency_Hz(1, 3, 2)
        from sigma_ground.field.constants import H_PLANCK, EV_TO_J
        self.assertAlmostEqual(freq, E * EV_TO_J / H_PLANCK,
                               delta=freq * 1e-6)

    def test_wavenumber_inverse_wavelength(self):
        """ν̃ = 1/λ (in cm⁻¹)."""
        lam_nm = transition_wavelength_nm(1, 3, 2)
        wn = transition_wavenumber(1, 3, 2)
        lam_cm = lam_nm * 1e-7
        self.assertAlmostEqual(wn, 1.0 / lam_cm, delta=1.0)


class TestSpectralSeries(unittest.TestCase):
    """Named spectral series."""

    def test_lyman_all_UV(self):
        """All Lyman lines are UV (< 122 nm)."""
        lines = lyman_series()
        for n, lam, E in lines:
            self.assertLess(lam, 122)

    def test_balmer_visible(self):
        """First few Balmer lines are in visible range."""
        lines = balmer_series()
        # Hα should be visible (656 nm)
        self.assertTrue(380 < lines[0][1] < 750)

    def test_paschen_IR(self):
        """Paschen lines are infrared (> 800 nm)."""
        lines = paschen_series()
        for n, lam, E in lines:
            self.assertGreater(lam, 800)

    def test_series_limit(self):
        """Series limit = ionization from that level."""
        # Balmer series limit ≈ 364.6 nm
        lim = series_limit_nm(1, 2)
        self.assertAlmostEqual(lim, 364.6, delta=1.0)

    def test_series_converges(self):
        """Lines in a series converge to the limit."""
        lines = balmer_series(n_max=20)
        limit = series_limit_nm(1, 2)
        # Last line should be close to limit
        self.assertAlmostEqual(lines[-1][1], limit, delta=5.0)


class TestMultiElectron(unittest.TestCase):
    """Multi-electron approximate energy levels."""

    def test_negative(self):
        """Orbital energies are negative (bound)."""
        E = multi_electron_energy_eV(26, 3, 2)  # Fe 3d
        self.assertLess(E, 0)

    def test_deeper_for_higher_Z(self):
        """Higher Z → more tightly bound."""
        E_C = multi_electron_energy_eV(6, 2, 1)   # Carbon 2p
        E_Ne = multi_electron_energy_eV(10, 2, 1)  # Neon 2p
        self.assertLess(E_Ne, E_C)  # Ne more tightly bound


class TestFineStructure(unittest.TestCase):
    """Fine structure (spin-orbit splitting)."""

    def test_2p_splitting(self):
        """Hydrogen 2p fine structure ≈ 4.5×10⁻⁵ eV (MEASURED)."""
        dE = fine_structure_splitting_eV(1, 2, 1)
        self.assertAlmostEqual(dE, 4.5e-5, delta=2e-5)

    def test_s_orbital_no_splitting(self):
        """l=0 → no fine structure splitting."""
        dE = fine_structure_splitting_eV(1, 1, 0)
        self.assertEqual(dE, 0.0)

    def test_Z_scaling(self):
        """Fine structure ∝ Z⁴ (for hydrogen-like)."""
        dE_H = fine_structure_splitting_eV(1, 2, 1)
        dE_He = fine_structure_splitting_eV(2, 2, 1)
        # Should scale as Z⁴ ≈ 16
        ratio = dE_He / dE_H
        self.assertAlmostEqual(ratio, 16.0, delta=3.0)

    def test_two_j_values(self):
        """l=1 gives j=1/2 and j=3/2."""
        shift_half = fine_structure_shift_eV(1, 2, 1, 0.5)
        shift_three_half = fine_structure_shift_eV(1, 2, 1, 1.5)
        self.assertNotAlmostEqual(shift_half, shift_three_half)


class TestZeeman(unittest.TestCase):
    """Zeeman effect in magnetic field."""

    def test_g_factor_pure_spin(self):
        """l=0, s=1/2 → g_J = 2."""
        g = lande_g_factor(0, 0.5, 0.5)
        self.assertAlmostEqual(g, 2.0, delta=0.01)

    def test_g_factor_pure_orbital(self):
        """s=0, l=1, j=1 → g_J = 1."""
        g = lande_g_factor(1, 0, 1)
        self.assertAlmostEqual(g, 1.0, delta=0.01)

    def test_splitting_count(self):
        """j=1 → 3 substates, j=3/2 → 4 substates."""
        self.assertEqual(zeeman_splitting_count(1), 3)
        self.assertEqual(zeeman_splitting_count(1.5), 4)

    def test_zeeman_shift_proportional_to_B(self):
        """Shift doubles when field doubles."""
        dE_1 = zeeman_shift_eV(1, 2.0, 1.0)
        dE_2 = zeeman_shift_eV(1, 2.0, 2.0)
        self.assertAlmostEqual(dE_2 / dE_1, 2.0, delta=0.01)

    def test_zeeman_pattern_symmetric(self):
        """Zeeman pattern is symmetric around m_j = 0."""
        pattern = zeeman_pattern(1, 0.5, 1.5, 1.0)
        energies = [dE for mj, dE in pattern]
        # Sum of shifts should be ~0 (symmetric)
        self.assertAlmostEqual(sum(energies), 0.0, delta=1e-10)

    def test_zero_field_no_splitting(self):
        """B = 0 → no Zeeman shift."""
        dE = zeeman_shift_eV(1, 2.0, 0.0)
        self.assertEqual(dE, 0.0)


class TestSelectionRules(unittest.TestCase):
    """Electric dipole selection rules."""

    def test_allowed_s_to_p(self):
        """s→p (Δl=+1) is allowed."""
        self.assertTrue(is_allowed_transition(0, 1))

    def test_allowed_p_to_s(self):
        """p→s (Δl=−1) is allowed."""
        self.assertTrue(is_allowed_transition(1, 0))

    def test_forbidden_s_to_s(self):
        """s→s (Δl=0) is forbidden."""
        self.assertFalse(is_allowed_transition(0, 0))

    def test_forbidden_s_to_d(self):
        """s→d (Δl=+2) is forbidden."""
        self.assertFalse(is_allowed_transition(0, 2))

    def test_forbidden_d_to_d(self):
        """d→d (Δl=0) is forbidden by electric dipole rules."""
        self.assertFalse(is_allowed_transition(2, 2))

    def test_delta_m_allowed(self):
        """Δm_l = 0, ±1 allowed."""
        self.assertTrue(is_allowed_transition(1, 0, 0, 0))
        self.assertTrue(is_allowed_transition(1, 0, 1, 0))
        self.assertTrue(is_allowed_transition(1, 0, -1, 0))

    def test_delta_m_forbidden(self):
        """|Δm_l| > 1 forbidden."""
        self.assertFalse(is_allowed_transition(2, 1, 2, 0))

    def test_allowed_transitions_list(self):
        """All listed transitions satisfy selection rules."""
        trans = allowed_transitions(n_max=4)
        self.assertGreater(len(trans), 0)
        for n_i, l_i, n_f, l_f, lam in trans:
            self.assertTrue(is_allowed_transition(l_i, l_f))


class TestQHO(unittest.TestCase):
    """Quantum harmonic oscillator."""

    def test_zero_point_energy(self):
        """E₀ = ℏω/2 > 0."""
        omega = 1e14
        E0 = qho_zero_point_energy_eV(omega)
        self.assertGreater(E0, 0)

    def test_equally_spaced(self):
        """Eₙ₊₁ − Eₙ = ℏω (same for all n)."""
        omega = 1e14
        spacing = qho_level_spacing_eV(omega)
        for n in range(5):
            dE = qho_energy_eV(omega, n + 1) - qho_energy_eV(omega, n)
            self.assertAlmostEqual(dE, spacing, delta=spacing * 1e-10)

    def test_energy_formula(self):
        """Eₙ = ℏω(n + ½)."""
        omega = 5e13
        for n in range(5):
            E = qho_energy_eV(omega, n)
            from sigma_ground.field.constants import HBAR, EV_TO_J
            expected = HBAR * omega * (n + 0.5) / EV_TO_J
            self.assertAlmostEqual(E, expected, delta=E * 1e-10)

    def test_classical_amplitude_increases(self):
        """Higher n → larger classical turning point."""
        omega = 1e14
        from sigma_ground.field.constants import M_ELECTRON_KG
        x0 = qho_classical_amplitude(omega, M_ELECTRON_KG, 0)
        x5 = qho_classical_amplitude(omega, M_ELECTRON_KG, 5)
        self.assertGreater(x5, x0)

    def test_transition_energy(self):
        """Δn = 1 transition energy = level spacing."""
        omega = 1e14
        dE = qho_transition_energy_eV(omega, 3, 2)
        spacing = qho_level_spacing_eV(omega)
        self.assertAlmostEqual(dE, spacing, delta=spacing * 1e-10)


class TestSigmaEffects(unittest.TestCase):
    """σ-field effects on spectra."""

    def test_sigma_zero_matches_standard(self):
        """At σ=0, results match standard hydrogen."""
        E0 = hydrogen_energy_eV(1, 0.0)
        E_std = hydrogen_energy_eV(1)
        self.assertAlmostEqual(E0, E_std, delta=1e-10)

    def test_sigma_shifts_rydberg(self):
        """At σ>0, Rydberg constant changes (slightly)."""
        R0 = rydberg_constant_at_sigma(0.0)
        R1 = rydberg_constant_at_sigma(1.0)
        self.assertNotAlmostEqual(R0, R1)
        # Higher σ → heavier proton → μ closer to m_e → R closer to R∞
        self.assertGreater(R1, R0)

    def test_sigma_spectral_shift_small(self):
        """Spectral shift at σ=0.01 is tiny."""
        shift = sigma_spectral_shift(1, 3, 2, 0.01)
        self.assertAlmostEqual(shift, 0.0, delta=0.001)

    def test_sigma_spectral_shift_sign(self):
        """Higher σ → heavier proton → lines blue-shift (slightly)."""
        shift = sigma_spectral_shift(1, 3, 2, 1.0)
        # Higher σ → R increases → wavelength decreases → negative shift
        self.assertLess(shift, 0)


class TestVisibleAndColor(unittest.TestCase):
    """Visible range and color mapping."""

    def test_visible_range(self):
        """380-750 nm is visible."""
        self.assertTrue(is_visible(500))
        self.assertFalse(is_visible(300))
        self.assertFalse(is_visible(800))

    def test_hydrogen_visible_lines(self):
        """Hydrogen has 4 visible lines (Balmer series)."""
        vis = visible_lines(1, n_max=10)
        # Hα(656), Hβ(486), Hγ(434), Hδ(410)
        self.assertGreaterEqual(len(vis), 4)

    def test_rgb_red(self):
        """~650 nm → red."""
        r, g, b = wavelength_to_rgb(650)
        self.assertGreater(r, 0.5)
        self.assertLess(g, 0.3)

    def test_rgb_blue(self):
        """~450 nm → blue."""
        r, g, b = wavelength_to_rgb(450)
        self.assertGreater(b, 0.5)

    def test_rgb_out_of_range(self):
        """UV and IR → black."""
        self.assertEqual(wavelength_to_rgb(300), (0, 0, 0))
        self.assertEqual(wavelength_to_rgb(800), (0, 0, 0))


class TestEmissionSpectrum(unittest.TestCase):
    """Full emission spectrum."""

    def test_has_lines(self):
        """Hydrogen has many spectral lines."""
        spec = emission_spectrum(1, n_max=6)
        self.assertGreater(len(spec), 10)

    def test_line_structure(self):
        """Each line has required keys."""
        spec = emission_spectrum(1, n_max=4)
        required = ['n_upper', 'n_lower', 'wavelength_nm', 'energy_eV',
                     'frequency_Hz', 'wavenumber_cm', 'series', 'visible', 'rgb']
        for line in spec:
            for key in required:
                self.assertIn(key, line)

    def test_balmer_series_labeled(self):
        """Lines to n=2 labeled 'Balmer'."""
        spec = emission_spectrum(1, n_max=5)
        balmer = [l for l in spec if l['n_lower'] == 2]
        for line in balmer:
            self.assertEqual(line['series'], 'Balmer')


class TestReports(unittest.TestCase):
    """Rule 9: full_report covers everything."""

    def test_report_complete(self):
        """atomic_spectra_report has all required fields."""
        r = atomic_spectra_report()
        required = ['Z', 'rydberg_energy_eV', 'rydberg_constant_m_inv',
                     'ionization_energy_eV', 'energy_levels_eV',
                     'balmer_series', 'visible_line_count',
                     'fine_structure_2p_eV']
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_full_report(self):
        """full_report returns a dict with additional fields."""
        r = full_report()
        self.assertIsInstance(r, dict)
        self.assertIn('qho_zero_point_eV', r)
        self.assertIn('zeeman_2p_pattern_1T', r)

    def test_full_report_He(self):
        """full_report works for He⁺ (Z=2)."""
        r = full_report(Z=2)
        self.assertEqual(r['Z'], 2)
        # He⁺ ionization ≈ 4 × hydrogen
        self.assertGreater(r['ionization_energy_eV'], 50)


if __name__ == '__main__':
    unittest.main()
