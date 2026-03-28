"""
Tests for solution.py — solubility, activity coefficients, colligative properties.

Strategy:
  - Test Ksp → molar solubility against MEASURED values
  - Test common ion effect reduces solubility
  - Test precipitation prediction (Q vs Ksp)
  - Test Debye-Hückel activity coefficients in valid range
  - Test colligative properties against MEASURED values
  - Test Debye length order of magnitude
  - Test derived K_b and K_f against MEASURED
  - Test Rule 9: every salt gets a report

Reference values (MEASURED, CRC Handbook):
  AgCl solubility:     1.33e-5 mol/L (from Ksp=1.77e-10)
  BaSO4 solubility:    1.04e-5 mol/L (from Ksp=1.08e-10)
  NaCl solubility:     6.15 mol/L (MEASURED directly)
  K_b(water):          0.512 K·kg/mol
  K_f(water):          1.86 K·kg/mol
  ΔT_f(1 m NaCl):     3.72 K (i=2)
  π(0.1 M NaCl):      ~4.89 atm at 25°C
  Debye length:        0.304/√I nm at 25°C in water
"""

import math
import unittest

from sigma_ground.field.interface.solution import (
    molar_solubility,
    solubility_with_common_ion,
    will_precipitate,
    ionic_strength_from_salt,
    ionic_strength,
    debye_huckel_A,
    debye_huckel_B,
    activity_coefficient_dh,
    activity_coefficient_salt,
    molality_from_molarity,
    boiling_point_elevation,
    freezing_point_depression,
    osmotic_pressure,
    boiling_point_elevation_salt,
    freezing_point_depression_salt,
    osmotic_pressure_salt,
    dilution,
    mixing_concentration,
    debye_length,
    sigma_Ksp_shift,
    solution_report,
    full_report,
    SOLUBILITY_DATA,
    ION_DATA,
    K_B_WATER,
    K_F_WATER,
)


class TestDerivedConstants(unittest.TestCase):
    """K_b and K_f derived from solvent properties."""

    def test_Kb_water(self):
        """K_b(water) ≈ 0.512 K·kg/mol (MEASURED)."""
        self.assertAlmostEqual(K_B_WATER, 0.512, delta=0.02)

    def test_Kf_water(self):
        """K_f(water) ≈ 1.86 K·kg/mol (MEASURED)."""
        self.assertAlmostEqual(K_F_WATER, 1.86, delta=0.05)

    def test_Kf_greater_than_Kb(self):
        """K_f > K_b (freezing is more sensitive than boiling)."""
        self.assertGreater(K_F_WATER, K_B_WATER)


class TestMolarSolubility(unittest.TestCase):
    """Molar solubility from Ksp."""

    def test_AgCl(self):
        """AgCl: s = sqrt(Ksp) = 1.33e-5 mol/L."""
        s = molar_solubility('silver_chloride')
        expected = math.sqrt(1.77e-10)
        self.assertAlmostEqual(s, expected, delta=expected * 0.01)

    def test_BaSO4(self):
        """BaSO4: s = sqrt(Ksp) = 1.04e-5 mol/L."""
        s = molar_solubility('barium_sulfate')
        expected = math.sqrt(1.08e-10)
        self.assertAlmostEqual(s, expected, delta=expected * 0.01)

    def test_CaF2(self):
        """CaF2: Ksp = 4s³ → s = (Ksp/4)^(1/3) = 2.05e-4 mol/L."""
        s = molar_solubility('calcium_fluoride')
        expected = (3.45e-11 / 4) ** (1.0 / 3.0)
        self.assertAlmostEqual(s, expected, delta=expected * 0.01)

    def test_FeOH3(self):
        """Fe(OH)3: Ksp = 27s⁴ → extremely insoluble."""
        s = molar_solubility('iron_hydroxide_iii')
        self.assertLess(s, 1e-9)  # essentially insoluble

    def test_NaCl_very_soluble(self):
        """NaCl is very soluble (Ksp = 37)."""
        s = molar_solubility('sodium_chloride')
        self.assertGreater(s, 5.0)

    def test_all_positive(self):
        """Every salt has positive solubility."""
        for key in SOLUBILITY_DATA:
            with self.subTest(salt=key):
                self.assertGreater(molar_solubility(key), 0)

    def test_ordering(self):
        """More insoluble salts have lower solubility."""
        # AgCl (Ksp=1.77e-10) > AgBr (5.35e-13) > AgI (8.52e-17)
        s_AgCl = molar_solubility('silver_chloride')
        s_AgBr = molar_solubility('silver_bromide')
        s_AgI = molar_solubility('silver_iodide')
        self.assertGreater(s_AgCl, s_AgBr)
        self.assertGreater(s_AgBr, s_AgI)

    def test_HgS_least_soluble(self):
        """HgS is the least soluble salt in the database."""
        s_HgS = molar_solubility('mercury_sulfide')
        for key in SOLUBILITY_DATA:
            if key != 'mercury_sulfide':
                with self.subTest(salt=key):
                    self.assertLessEqual(s_HgS, molar_solubility(key))


class TestCommonIonEffect(unittest.TestCase):
    """Solubility decreases with common ion."""

    def test_AgCl_with_NaCl(self):
        """Adding Cl- reduces AgCl solubility."""
        s_pure = molar_solubility('silver_chloride')
        s_common = solubility_with_common_ion(
            'silver_chloride', 0.1, ion_type='anion')
        self.assertLess(s_common, s_pure)

    def test_no_common_ion(self):
        """Zero common ion → same as pure solubility."""
        s_pure = molar_solubility('silver_chloride')
        s_zero = solubility_with_common_ion(
            'silver_chloride', 0.0, ion_type='anion')
        self.assertAlmostEqual(s_zero, s_pure, delta=s_pure * 0.01)

    def test_large_common_ion(self):
        """Large common ion → very small solubility."""
        s = solubility_with_common_ion(
            'silver_chloride', 1.0, ion_type='anion')
        self.assertLess(s, 1e-9)  # Ksp/[Cl-] ≈ 1.77e-10


class TestPrecipitation(unittest.TestCase):
    """Precipitation prediction."""

    def test_will_precipitate_above_Ksp(self):
        """Q > Ksp → precipitation."""
        ppt, Q, Ksp = will_precipitate('silver_chloride', 0.01, 0.01)
        self.assertTrue(ppt)
        self.assertGreater(Q, Ksp)

    def test_no_precipitate_below_Ksp(self):
        """Q < Ksp → no precipitation."""
        ppt, Q, Ksp = will_precipitate('silver_chloride', 1e-7, 1e-7)
        self.assertFalse(ppt)
        self.assertLess(Q, Ksp)

    def test_very_insoluble_easy_to_precipitate(self):
        """Fe(OH)3 precipitates at tiny concentrations."""
        ppt, Q, Ksp = will_precipitate('iron_hydroxide_iii', 1e-8, 1e-8)
        # Q = (1e-8)^1 × (1e-8)^3 = 1e-32, Ksp = 2.79e-39
        self.assertTrue(ppt)


class TestIonicStrength(unittest.TestCase):
    """Ionic strength calculations."""

    def test_NaCl_1_1(self):
        """0.1 M NaCl: I = 0.1 (1:1 electrolyte)."""
        I = ionic_strength_from_salt('sodium_chloride', 0.1)
        self.assertAlmostEqual(I, 0.1, places=5)

    def test_CaSO4_2_2(self):
        """0.01 M CaSO4: I = 0.04 (2:2 electrolyte)."""
        I = ionic_strength_from_salt('calcium_sulfate', 0.01)
        self.assertAlmostEqual(I, 0.04, places=5)

    def test_PbCl2_1_2(self):
        """0.01 M PbCl2: I = 0.5*(0.01×4 + 0.02×1) = 0.03."""
        I = ionic_strength_from_salt('lead_chloride', 0.01)
        self.assertAlmostEqual(I, 0.03, places=5)

    def test_from_ions(self):
        """From individual ions: 0.1 M Na+ + 0.1 M Cl- → I = 0.1."""
        I = ionic_strength({'Na+': 0.1, 'Cl-': 0.1})
        self.assertAlmostEqual(I, 0.1, places=5)


class TestDebyeHuckel(unittest.TestCase):
    """Debye-Hückel activity coefficients."""

    def test_A_at_25C(self):
        """A ≈ 0.509 at 25°C in water."""
        A = debye_huckel_A(298.15)
        self.assertAlmostEqual(A, 0.509, delta=0.02)

    def test_gamma_unity_at_zero_I(self):
        """At I=0, γ± = 1.0 (ideal solution)."""
        gamma = activity_coefficient_dh(1, 1, 0.0)
        self.assertAlmostEqual(gamma, 1.0, places=10)

    def test_gamma_less_than_one(self):
        """Activity coefficient < 1 at finite I (ion-ion interactions)."""
        gamma = activity_coefficient_dh(1, 1, 0.01)
        self.assertLess(gamma, 1.0)
        self.assertGreater(gamma, 0.0)

    def test_gamma_decreases_with_I(self):
        """Higher ionic strength → lower activity coefficient."""
        g_low = activity_coefficient_dh(1, 1, 0.001)
        g_high = activity_coefficient_dh(1, 1, 0.1)
        self.assertGreater(g_low, g_high)

    def test_divalent_more_depressed(self):
        """2:2 electrolyte has lower γ± than 1:1 at same I."""
        g_11 = activity_coefficient_dh(1, 1, 0.01)
        g_22 = activity_coefficient_dh(2, 2, 0.01)
        self.assertGreater(g_11, g_22)

    def test_NaCl_0_01M(self):
        """NaCl at 0.01 M: γ± ≈ 0.90 (MEASURED: 0.903)."""
        gamma = activity_coefficient_salt('sodium_chloride', 0.01)
        self.assertGreater(gamma, 0.85)
        self.assertLess(gamma, 0.95)

    def test_bounded(self):
        """γ± should be in (0, 1] for I < 0.1."""
        for I in [0.001, 0.01, 0.05, 0.1]:
            gamma = activity_coefficient_dh(1, 1, I)
            with self.subTest(I=I):
                self.assertGreater(gamma, 0)
                self.assertLessEqual(gamma, 1.0)


class TestColligativeProperties(unittest.TestCase):
    """Boiling point elevation, freezing point depression, osmotic pressure."""

    def test_bpe_NaCl_1m(self):
        """1 m NaCl: ΔT_b = 2 × 0.512 × 1.0 = 1.024 K."""
        dT = boiling_point_elevation(1.0, i_factor=2)
        self.assertAlmostEqual(dT, 2 * K_B_WATER, delta=0.01)

    def test_fpd_NaCl_1m(self):
        """1 m NaCl: ΔT_f = 2 × 1.86 × 1.0 = 3.72 K."""
        dT = freezing_point_depression(1.0, i_factor=2)
        self.assertAlmostEqual(dT, 2 * K_F_WATER, delta=0.05)

    def test_bpe_positive(self):
        """Boiling point elevation is always positive."""
        dT = boiling_point_elevation(0.5, i_factor=1)
        self.assertGreater(dT, 0)

    def test_fpd_positive(self):
        """Freezing point depression is always positive."""
        dT = freezing_point_depression(0.5, i_factor=1)
        self.assertGreater(dT, 0)

    def test_proportional_to_molality(self):
        """ΔT ∝ m (colligative)."""
        dT_1 = freezing_point_depression(1.0)
        dT_2 = freezing_point_depression(2.0)
        self.assertAlmostEqual(dT_2 / dT_1, 2.0, places=5)

    def test_proportional_to_i(self):
        """ΔT ∝ i (more particles → more effect)."""
        dT_i1 = boiling_point_elevation(1.0, i_factor=1)
        dT_i3 = boiling_point_elevation(1.0, i_factor=3)
        self.assertAlmostEqual(dT_i3 / dT_i1, 3.0, places=5)

    def test_osmotic_pressure_0_1M_NaCl(self):
        """0.1 M NaCl: π ≈ 4.89 atm at 25°C."""
        pi = osmotic_pressure(0.1, i_factor=2, T=298.15)
        pi_atm = pi / 101325.0
        self.assertGreater(pi_atm, 3.0)
        self.assertLess(pi_atm, 7.0)

    def test_osmotic_increases_with_C(self):
        """More concentrated → higher osmotic pressure."""
        pi_low = osmotic_pressure(0.01, i_factor=1)
        pi_high = osmotic_pressure(1.0, i_factor=1)
        self.assertGreater(pi_high, pi_low)

    def test_salt_convenience_functions(self):
        """Salt-specific functions should give same results."""
        dT_b = boiling_point_elevation_salt('sodium_chloride', 0.1)
        dT_f = freezing_point_depression_salt('sodium_chloride', 0.1)
        pi = osmotic_pressure_salt('sodium_chloride', 0.1)
        self.assertGreater(dT_b, 0)
        self.assertGreater(dT_f, 0)
        self.assertGreater(pi, 0)


class TestDebyeLength(unittest.TestCase):
    """Debye screening length."""

    def test_0_1M_NaCl(self):
        """0.1 M 1:1 salt: λ_D ≈ 0.96 nm."""
        lam = debye_length(0.1)
        lam_nm = lam * 1e9
        self.assertGreater(lam_nm, 0.5)
        self.assertLess(lam_nm, 2.0)

    def test_0_001M(self):
        """0.001 M: λ_D ≈ 9.6 nm."""
        lam = debye_length(0.001)
        lam_nm = lam * 1e9
        self.assertGreater(lam_nm, 5.0)
        self.assertLess(lam_nm, 20.0)

    def test_decreases_with_I(self):
        """Higher ionic strength → shorter Debye length."""
        lam_low = debye_length(0.001)
        lam_high = debye_length(0.1)
        self.assertGreater(lam_low, lam_high)

    def test_inverse_sqrt_scaling(self):
        """λ_D ∝ 1/√I."""
        lam_1 = debye_length(0.01)
        lam_2 = debye_length(0.04)
        ratio = lam_1 / lam_2
        self.assertAlmostEqual(ratio, 2.0, delta=0.1)

    def test_infinite_at_zero_I(self):
        """At I=0, Debye length is infinite."""
        self.assertEqual(debye_length(0.0), float('inf'))


class TestDilutionMixing(unittest.TestCase):
    """Dilution and mixing calculations."""

    def test_dilution_halves(self):
        """Doubling volume halves concentration."""
        C = dilution(1.0, 50.0, 100.0)
        self.assertAlmostEqual(C, 0.5, places=10)

    def test_mixing_equal(self):
        """Mixing equal volumes of same concentration → same concentration."""
        C = mixing_concentration(0.5, 50.0, 0.5, 50.0)
        self.assertAlmostEqual(C, 0.5, places=10)

    def test_mixing_different(self):
        """Mixing 0.1 M (50 mL) + 0.3 M (50 mL) → 0.2 M."""
        C = mixing_concentration(0.1, 50.0, 0.3, 50.0)
        self.assertAlmostEqual(C, 0.2, places=10)


class TestSigmaDependence(unittest.TestCase):
    """σ-field coupling."""

    def test_Ksp_invariant(self):
        """At sigma_here, Ksp is unchanged."""
        from sigma_ground.field.constants import SIGMA_HERE
        Ksp = sigma_Ksp_shift('silver_chloride', SIGMA_HERE)
        self.assertAlmostEqual(Ksp, SOLUBILITY_DATA['silver_chloride']['Ksp'],
                                places=15)


class TestDiagnostics(unittest.TestCase):
    """Report generation."""

    def test_report_complete(self):
        r = solution_report('silver_chloride')
        required = [
            'salt', 'formula', 'Ksp', 'molar_solubility_mol_L',
            'i_factor', 'activity_coefficient', 'ionic_strength_mol_L',
            'boiling_point_elevation_K', 'freezing_point_depression_K',
            'osmotic_pressure_Pa',
        ]
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_full_report_all_salts(self):
        """Rule 9: covers every salt."""
        reports = full_report()
        self.assertEqual(set(reports.keys()), set(SOLUBILITY_DATA.keys()))

    def test_all_solubilities_positive(self):
        """Every salt has positive solubility in report."""
        reports = full_report()
        for key, r in reports.items():
            with self.subTest(salt=key):
                self.assertGreater(r['molar_solubility_mol_L'], 0)


if __name__ == '__main__':
    unittest.main()
