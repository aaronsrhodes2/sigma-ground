"""
Tests for acid_base.py — pH, pKa, buffers, titrations.

Strategy:
  - Test pH of known solutions against MEASURED values
  - Test Henderson-Hasselbalch at known buffer points
  - Test buffer capacity maximum at pH = pKa
  - Test titration curve shape (4 regions)
  - Test Kw temperature dependence
  - Test polyprotic speciation sums to 1
  - Test strong acid/base limits
  - Test Rule 9: every species gets a report

Reference values (MEASURED, CRC/IUPAC):
  0.1 M HCl:           pH = 1.0
  0.1 M NaOH:          pH = 13.0
  0.1 M acetic acid:   pH ≈ 2.88
  0.1 M NH3:           pH ≈ 11.12
  0.1 M NaCl:          pH = 7.0
  Kw(25°C):            1.012e-14
  Kw(37°C):            2.4e-14
  Neutral pH(25°C):    7.0
  Neutral pH(37°C):    6.8
"""

import math
import unittest

from sigma_ground.field.interface.acid_base import (
    water_ion_product,
    neutral_pH,
    pKa,
    Ka,
    pKb,
    Kb,
    pH_strong_acid,
    pH_strong_base,
    pH_weak_acid,
    pH_weak_base,
    pH_solution,
    henderson_hasselbalch,
    buffer_capacity,
    buffer_capacity_max,
    buffer_range,
    titration_strong_acid_strong_base,
    titration_weak_acid_strong_base,
    titration_curve,
    percent_dissociation,
    polyprotic_alpha,
    sigma_pKa_shift,
    acid_base_report,
    full_report,
    ACID_BASE_DATA,
)


class TestWaterAutoionization(unittest.TestCase):
    """Kw(T) from van't Hoff equation."""

    def test_Kw_25C(self):
        """Kw at 25°C = 1.012e-14 (MEASURED, IUPAC)."""
        Kw = water_ion_product(298.15)
        self.assertAlmostEqual(Kw, 1.012e-14, delta=0.01e-14)

    def test_Kw_increases_with_T(self):
        """Kw increases with temperature (endothermic autoionization)."""
        Kw_25 = water_ion_product(298.15)
        Kw_37 = water_ion_product(310.15)
        Kw_60 = water_ion_product(333.15)
        self.assertLess(Kw_25, Kw_37)
        self.assertLess(Kw_37, Kw_60)

    def test_Kw_37C(self):
        """Kw at 37°C ≈ 2.4e-14 (MEASURED, body temperature)."""
        Kw = water_ion_product(310.15)
        self.assertGreater(Kw, 1.5e-14)
        self.assertLess(Kw, 4.0e-14)

    def test_Kw_zero_T(self):
        """T=0 → Kw = 0."""
        self.assertEqual(water_ion_product(0), 0.0)

    def test_Kw_positive(self):
        """Kw > 0 for all positive temperatures."""
        for T in [100, 200, 273, 298, 310, 373]:
            with self.subTest(T=T):
                self.assertGreater(water_ion_product(T), 0)


class TestNeutralPH(unittest.TestCase):
    """Neutral pH of pure water."""

    def test_neutral_25C(self):
        """pH of pure water at 25°C = 7.0."""
        self.assertAlmostEqual(neutral_pH(298.15), 7.0, delta=0.05)

    def test_neutral_37C(self):
        """pH of pure water at 37°C ≈ 6.8."""
        pH = neutral_pH(310.15)
        self.assertGreater(pH, 6.5)
        self.assertLess(pH, 7.0)

    def test_neutral_decreases_with_T(self):
        """Neutral pH decreases with temperature."""
        pH_25 = neutral_pH(298.15)
        pH_60 = neutral_pH(333.15)
        self.assertGreater(pH_25, pH_60)


class TestPKa(unittest.TestCase):
    """pKa retrieval and consistency."""

    def test_acetic_acid(self):
        """Acetic acid pKa = 4.756 (MEASURED)."""
        self.assertAlmostEqual(pKa('acetic_acid'), 4.756, places=2)

    def test_formic_acid_stronger_than_acetic(self):
        """Formic acid (pKa 3.75) is stronger than acetic (4.756)."""
        self.assertLess(pKa('formic_acid'), pKa('acetic_acid'))

    def test_HF(self):
        """HF pKa = 3.17 (MEASURED)."""
        self.assertAlmostEqual(pKa('hydrofluoric_acid'), 3.17, places=1)

    def test_strong_acids_negative_pKa(self):
        """Strong acids have pKa < 0."""
        for key in ACID_BASE_DATA:
            if ACID_BASE_DATA[key]['type'] == 'strong_acid':
                with self.subTest(acid=key):
                    self.assertLess(pKa(key), 0)

    def test_weak_acids_positive_pKa(self):
        """Weak acids have pKa > 0."""
        for key in ACID_BASE_DATA:
            if ACID_BASE_DATA[key]['type'] == 'weak_acid':
                with self.subTest(acid=key):
                    self.assertGreater(pKa(key), 0)

    def test_pKa_pKb_sum_equals_pKw(self):
        """pKa + pKb = pKw for conjugate acid-base pair."""
        pKw = -math.log10(water_ion_product(298.15))
        for key in ACID_BASE_DATA:
            data = ACID_BASE_DATA[key]
            if data['type'] in ('weak_acid', 'weak_base'):
                with self.subTest(species=key):
                    total = pKa(key) + pKb(key)
                    self.assertAlmostEqual(total, pKw, delta=0.1)

    def test_Ka_Kb_product_equals_Kw(self):
        """Ka × Kb = Kw for conjugate pairs."""
        Kw = water_ion_product(298.15)
        for key in ACID_BASE_DATA:
            data = ACID_BASE_DATA[key]
            if data['type'] in ('weak_acid', 'weak_base'):
                with self.subTest(species=key):
                    product = Ka(key) * Kb(key)
                    # Compare log scale
                    self.assertAlmostEqual(
                        math.log10(product), math.log10(Kw), delta=0.2)

    def test_base_pKa_from_pKb(self):
        """Ammonia pKa(conjugate) = 14 - 4.75 ≈ 9.25."""
        pKa_val = pKa('ammonia')
        self.assertGreater(pKa_val, 9.0)
        self.assertLess(pKa_val, 9.5)


class TestStrongAcidPH(unittest.TestCase):
    """pH of strong acid solutions."""

    def test_0_1M_HCl(self):
        """0.1 M HCl → pH = 1.0."""
        pH = pH_strong_acid(0.1)
        self.assertAlmostEqual(pH, 1.0, delta=0.05)

    def test_1M_HCl(self):
        """1.0 M HCl → pH = 0.0."""
        pH = pH_strong_acid(1.0)
        self.assertAlmostEqual(pH, 0.0, delta=0.05)

    def test_very_dilute(self):
        """Very dilute acid → pH approaches 7."""
        pH = pH_strong_acid(1e-10)
        self.assertGreater(pH, 6.5)
        self.assertLess(pH, 7.5)

    def test_zero_concentration(self):
        """Zero concentration → neutral pH."""
        pH = pH_strong_acid(0.0)
        self.assertAlmostEqual(pH, 7.0, delta=0.1)

    def test_pH_decreases_with_concentration(self):
        """More acid → lower pH."""
        pH_low = pH_strong_acid(0.01)
        pH_high = pH_strong_acid(1.0)
        self.assertGreater(pH_low, pH_high)


class TestStrongBasePH(unittest.TestCase):
    """pH of strong base solutions."""

    def test_0_1M_NaOH(self):
        """0.1 M NaOH → pH = 13.0."""
        pH = pH_strong_base(0.1)
        self.assertAlmostEqual(pH, 13.0, delta=0.1)

    def test_1M_NaOH(self):
        """1.0 M NaOH → pH = 14.0."""
        pH = pH_strong_base(1.0)
        self.assertAlmostEqual(pH, 14.0, delta=0.1)

    def test_zero_concentration(self):
        """Zero concentration → neutral pH."""
        pH = pH_strong_base(0.0)
        self.assertAlmostEqual(pH, 7.0, delta=0.1)


class TestWeakAcidPH(unittest.TestCase):
    """pH of weak acid solutions."""

    def test_0_1M_acetic(self):
        """0.1 M acetic acid → pH ≈ 2.88 (MEASURED)."""
        pH = pH_weak_acid('acetic_acid', 0.1)
        self.assertGreater(pH, 2.5)
        self.assertLess(pH, 3.3)

    def test_more_concentrated_lower_pH(self):
        """Higher concentration → lower pH."""
        pH_dilute = pH_weak_acid('acetic_acid', 0.001)
        pH_conc = pH_weak_acid('acetic_acid', 1.0)
        self.assertGreater(pH_dilute, pH_conc)

    def test_stronger_acid_lower_pH(self):
        """Formic acid (pKa 3.75) gives lower pH than acetic (4.756) at same C."""
        pH_formic = pH_weak_acid('formic_acid', 0.1)
        pH_acetic = pH_weak_acid('acetic_acid', 0.1)
        self.assertLess(pH_formic, pH_acetic)

    def test_weak_acid_pH_above_strong(self):
        """Weak acid pH > strong acid pH at same concentration."""
        pH_weak = pH_weak_acid('acetic_acid', 0.1)
        pH_strong = pH_strong_acid(0.1)
        self.assertGreater(pH_weak, pH_strong)


class TestWeakBasePH(unittest.TestCase):
    """pH of weak base solutions."""

    def test_0_1M_ammonia(self):
        """0.1 M NH3 → pH ≈ 11.1 (MEASURED)."""
        pH = pH_weak_base('ammonia', 0.1)
        self.assertGreater(pH, 10.5)
        self.assertLess(pH, 11.7)

    def test_methylamine_stronger_than_ammonia(self):
        """Methylamine (pKb 3.36) gives higher pH than ammonia (pKb 4.75)."""
        pH_methyl = pH_weak_base('methylamine', 0.1)
        pH_ammonia = pH_weak_base('ammonia', 0.1)
        self.assertGreater(pH_methyl, pH_ammonia)

    def test_weak_base_pH_below_strong(self):
        """Weak base pH < strong base pH at same concentration."""
        pH_weak = pH_weak_base('ammonia', 0.1)
        pH_strong = pH_strong_base(0.1)
        self.assertLess(pH_weak, pH_strong)


class TestUnifiedPH(unittest.TestCase):
    """pH_solution dispatches correctly."""

    def test_strong_acid(self):
        pH = pH_solution('hydrochloric_acid', 0.1)
        self.assertAlmostEqual(pH, 1.0, delta=0.1)

    def test_weak_acid(self):
        pH = pH_solution('acetic_acid', 0.1)
        self.assertGreater(pH, 2.5)
        self.assertLess(pH, 3.3)

    def test_strong_base(self):
        pH = pH_solution('sodium_hydroxide', 0.1)
        self.assertAlmostEqual(pH, 13.0, delta=0.1)

    def test_weak_base(self):
        pH = pH_solution('ammonia', 0.1)
        self.assertGreater(pH, 10.5)
        self.assertLess(pH, 11.7)


class TestHendersonHasselbalch(unittest.TestCase):
    """Henderson-Hasselbalch equation."""

    def test_equal_ratio(self):
        """[A-]/[HA] = 1 → pH = pKa."""
        pH = henderson_hasselbalch('acetic_acid', 1.0)
        self.assertAlmostEqual(pH, pKa('acetic_acid'), places=3)

    def test_ratio_10(self):
        """[A-]/[HA] = 10 → pH = pKa + 1."""
        pH = henderson_hasselbalch('acetic_acid', 10.0)
        self.assertAlmostEqual(pH, pKa('acetic_acid') + 1.0, places=3)

    def test_ratio_0_1(self):
        """[A-]/[HA] = 0.1 → pH = pKa - 1."""
        pH = henderson_hasselbalch('acetic_acid', 0.1)
        self.assertAlmostEqual(pH, pKa('acetic_acid') - 1.0, places=3)


class TestBufferCapacity(unittest.TestCase):
    """Buffer capacity calculations."""

    def test_maximum_at_pKa(self):
        """Buffer capacity is maximum at pH = pKa."""
        pka = pKa('acetic_acid')
        beta_at_pKa = buffer_capacity('acetic_acid', 0.1, pka)
        beta_off = buffer_capacity('acetic_acid', 0.1, pka + 2.0)
        self.assertGreater(beta_at_pKa, beta_off)

    def test_proportional_to_concentration(self):
        """Buffer capacity scales with concentration."""
        pka = pKa('acetic_acid')
        beta_01 = buffer_capacity('acetic_acid', 0.1, pka)
        beta_1 = buffer_capacity('acetic_acid', 1.0, pka)
        # Should be ~10× larger
        self.assertGreater(beta_1 / beta_01, 5.0)
        self.assertLess(beta_1 / beta_01, 15.0)

    def test_positive_everywhere(self):
        """Buffer capacity is always positive."""
        for pH in [1, 3, 5, 7, 9, 11, 13]:
            beta = buffer_capacity('acetic_acid', 0.1, pH)
            with self.subTest(pH=pH):
                self.assertGreater(beta, 0)

    def test_buffer_range(self):
        """Buffer range is pKa ± 1."""
        low, high = buffer_range('acetic_acid')
        pka = pKa('acetic_acid')
        self.assertAlmostEqual(low, pka - 1.0, places=5)
        self.assertAlmostEqual(high, pka + 1.0, places=5)


class TestTitration(unittest.TestCase):
    """Titration curve behavior."""

    def test_strong_strong_equivalence(self):
        """Strong acid + strong base at equivalence → pH 7."""
        pH = titration_strong_acid_strong_base(0.1, 50.0, 0.1, 50.0)
        self.assertAlmostEqual(pH, 7.0, delta=0.1)

    def test_strong_strong_before_eq(self):
        """Before equivalence: excess acid → pH < 7."""
        pH = titration_strong_acid_strong_base(0.1, 50.0, 0.1, 25.0)
        self.assertLess(pH, 7.0)

    def test_strong_strong_after_eq(self):
        """After equivalence: excess base → pH > 7."""
        pH = titration_strong_acid_strong_base(0.1, 50.0, 0.1, 75.0)
        self.assertGreater(pH, 7.0)

    def test_weak_strong_equivalence_basic(self):
        """Weak acid + strong base at equivalence → pH > 7 (conjugate base)."""
        # V_eq = C_acid × V_acid / C_base = 0.1 × 50 / 0.1 = 50 mL
        pH = titration_weak_acid_strong_base(
            'acetic_acid', 0.1, 50.0, 0.1, 50.0)
        self.assertGreater(pH, 7.0)
        self.assertLess(pH, 11.0)

    def test_weak_strong_half_equivalence(self):
        """At half-equivalence: pH = pKa (Henderson-Hasselbalch)."""
        pH = titration_weak_acid_strong_base(
            'acetic_acid', 0.1, 50.0, 0.1, 25.0)
        self.assertAlmostEqual(pH, pKa('acetic_acid'), delta=0.2)

    def test_titration_curve_monotonic(self):
        """Titration curve should be monotonically increasing."""
        curve = titration_curve('acetic_acid', 0.1, 50.0, 0.1, n_points=50)
        for i in range(len(curve) - 1):
            self.assertLessEqual(curve[i][1], curve[i + 1][1] + 0.01)

    def test_titration_curve_length(self):
        """Curve has n_points + 1 data points."""
        curve = titration_curve('acetic_acid', 0.1, 50.0, 0.1, n_points=20)
        self.assertEqual(len(curve), 21)


class TestPercentDissociation(unittest.TestCase):
    """Percent dissociation of acids."""

    def test_strong_acid_100pct(self):
        """Strong acids are 100% dissociated."""
        self.assertAlmostEqual(
            percent_dissociation('hydrochloric_acid', 0.1), 100.0, delta=0.1)

    def test_weak_acid_partial(self):
        """Weak acids are partially dissociated."""
        alpha = percent_dissociation('acetic_acid', 0.1)
        self.assertGreater(alpha, 0)
        self.assertLess(alpha, 100)

    def test_dissociation_increases_with_dilution(self):
        """Dilution increases percent dissociation (Ostwald)."""
        alpha_conc = percent_dissociation('acetic_acid', 1.0)
        alpha_dilute = percent_dissociation('acetic_acid', 0.001)
        self.assertGreater(alpha_dilute, alpha_conc)

    def test_acetic_at_0_1M(self):
        """0.1 M acetic acid: ~1.3% dissociated (MEASURED)."""
        alpha = percent_dissociation('acetic_acid', 0.1)
        self.assertGreater(alpha, 0.5)
        self.assertLess(alpha, 5.0)


class TestPolyproticSpeciation(unittest.TestCase):
    """Polyprotic acid speciation (alpha fractions)."""

    def test_sum_to_one(self):
        """All alpha fractions must sum to 1.0."""
        pKa_list = [2.15, 7.20, 12.35]  # phosphoric acid
        for pH in [0, 2, 4, 7, 10, 12, 14]:
            alphas = polyprotic_alpha(pKa_list, pH)
            with self.subTest(pH=pH):
                self.assertAlmostEqual(sum(alphas), 1.0, places=10)

    def test_protonated_at_low_pH(self):
        """At pH << pKa1, fully protonated species dominates."""
        pKa_list = [2.15, 7.20, 12.35]
        alphas = polyprotic_alpha(pKa_list, 0.0)
        self.assertGreater(alphas[0], 0.9)

    def test_deprotonated_at_high_pH(self):
        """At pH >> pKa3, fully deprotonated species dominates."""
        pKa_list = [2.15, 7.20, 12.35]
        alphas = polyprotic_alpha(pKa_list, 16.0)
        self.assertGreater(alphas[-1], 0.9)

    def test_crossover_at_pKa(self):
        """At pH = pKa1, alpha_0 ≈ alpha_1 (for well-separated pKa values)."""
        pKa_list = [2.15, 7.20, 12.35]
        alphas = polyprotic_alpha(pKa_list, 2.15)
        # alpha_0 and alpha_1 should be similar
        self.assertAlmostEqual(alphas[0], alphas[1], delta=0.15)

    def test_carbonate_system(self):
        """CO2/HCO3-/CO3^2- at ocean pH 8.1: HCO3- dominates."""
        pKa_list = [6.35, 10.33]
        alphas = polyprotic_alpha(pKa_list, 8.1)
        # HCO3- is alpha_1
        self.assertGreater(alphas[1], 0.9)

    def test_monoprotic_consistent(self):
        """Monoprotic acid: alpha_0 + alpha_1 = 1."""
        pKa_list = [4.756]  # acetic acid
        alphas = polyprotic_alpha(pKa_list, 4.756)
        self.assertAlmostEqual(alphas[0], 0.5, delta=0.05)
        self.assertAlmostEqual(alphas[1], 0.5, delta=0.05)


class TestSigmaDependence(unittest.TestCase):
    """σ-field coupling (should be negligible)."""

    def test_shift_at_sigma_here(self):
        """At sigma_here, pKa shift is zero."""
        from sigma_ground.field.constants import SIGMA_HERE
        pKa_shifted = sigma_pKa_shift('acetic_acid', SIGMA_HERE)
        self.assertAlmostEqual(pKa_shifted, pKa('acetic_acid'), places=5)


class TestDiagnostics(unittest.TestCase):
    """Report generation."""

    def test_report_complete(self):
        r = acid_base_report('acetic_acid')
        required = [
            'species', 'formula', 'type', 'concentration_mol_L',
            'temperature_K', 'pKa', 'Ka', 'pH', 'percent_dissociation',
            'Kw', 'neutral_pH',
        ]
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_weak_acid_has_buffer_info(self):
        r = acid_base_report('acetic_acid')
        self.assertIn('buffer_range', r)
        self.assertIn('buffer_capacity_at_pKa', r)

    def test_full_report_all_species(self):
        """Rule 9: covers every species."""
        reports = full_report()
        self.assertEqual(set(reports.keys()), set(ACID_BASE_DATA.keys()))

    def test_all_pH_physical(self):
        """Every species should give a physical pH (0-14 range)."""
        reports = full_report(concentration=0.1)
        for key, r in reports.items():
            with self.subTest(species=key):
                self.assertGreater(r['pH'], -2)
                self.assertLess(r['pH'], 16)


if __name__ == '__main__':
    unittest.main()
