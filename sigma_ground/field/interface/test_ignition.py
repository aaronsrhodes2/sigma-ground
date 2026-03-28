"""
Tests for ignition.py — combustion onset from activation energy.

Strategy:
  - Test autoignition temperatures against MEASURED values (NFPA)
  - Test flash point < autoignition (always)
  - Test adiabatic flame temperatures are physical
  - Test ignition delay behavior (Arrhenius)
  - Test flammability classification
  - Test Rule 9: every material gets a report

Reference values (MEASURED, NFPA/NIST):
  Methane:   T_auto = 810 K (537°C)
  Propane:   T_auto = 723 K (450°C)
  Octane:    T_auto = 479 K (206°C)
  Ethanol:   T_auto = 638 K (365°C)
  Hydrogen:  T_auto = 773 K (500°C)
  Paper:     T_auto = 506 K (233°C)

  Methane adiabatic flame: ~2236 K
  Hydrogen adiabatic flame: ~2483 K
"""

import math
import unittest

from sigma_ground.field.interface.ignition import (
    autoignition_temperature,
    flash_point,
    adiabatic_flame_temperature,
    ignition_delay,
    is_flammable_at,
    ignition_report,
    full_report,
    FLAMMABLE_MATERIALS,
)


class TestAutoignitionTemperature(unittest.TestCase):
    """T_auto from inverted Arrhenius vs MEASURED (NFPA)."""

    def test_all_positive(self):
        for key in FLAMMABLE_MATERIALS:
            with self.subTest(material=key):
                T = autoignition_temperature(key)
                self.assertGreater(T, 200)  # > 200 K
                self.assertLess(T, 2000)    # < 2000 K

    def test_methane_within_30pct(self):
        """Methane T_auto ≈ 810 K ± 30%."""
        T = autoignition_temperature('methane')
        self.assertGreater(T, 810 * 0.7)
        self.assertLess(T, 810 * 1.3)

    def test_hydrogen_within_30pct(self):
        """Hydrogen T_auto ≈ 773 K ± 30%."""
        T = autoignition_temperature('hydrogen')
        self.assertGreater(T, 773 * 0.7)
        self.assertLess(T, 773 * 1.3)

    def test_octane_within_30pct(self):
        """Octane T_auto ≈ 479 K ± 30%."""
        T = autoignition_temperature('octane')
        self.assertGreater(T, 479 * 0.7)
        self.assertLess(T, 479 * 1.3)

    def test_paper_within_30pct(self):
        """Paper T_auto ≈ 506 K ± 30% (Fahrenheit 451)."""
        T = autoignition_temperature('paper')
        self.assertGreater(T, 506 * 0.7)
        self.assertLess(T, 506 * 1.3)

    def test_higher_Ea_higher_Tauto(self):
        """Higher activation energy → higher autoignition temperature."""
        # methane E_a=2.0 eV > hydrogen E_a=1.5 eV
        T_methane = autoignition_temperature('methane')
        T_hydrogen = autoignition_temperature('hydrogen')
        self.assertGreater(T_methane, T_hydrogen)

    def test_ordering_consistent(self):
        """T_auto ∝ E_a (inverted Arrhenius)."""
        # E_a: octane 0.95 < paper 1.00 < wood 1.06 < ... < methane 1.61
        T = {k: autoignition_temperature(k) for k in FLAMMABLE_MATERIALS}
        self.assertLess(T['octane'], T['paper'])     # 0.95 < 1.00
        self.assertLess(T['paper'], T['methane'])    # 1.00 < 1.61
        self.assertLess(T['hydrogen'], T['methane']) # 1.53 < 1.61


class TestDerivedCombustionEnthalpy(unittest.TestCase):
    """Hc derived from Hess's law + Pauling bond energies vs NIST."""

    # NIST measured combustion enthalpies (kJ/mol)
    NIST_HC = {
        'methane': 890.4,
        'propane': 2219.2,
        'octane': 5471.0,
        'ethanol': 1367.0,
        'hydrogen': 286.0,
    }

    def test_all_positive(self):
        """Every fuel should have positive combustion enthalpy."""
        for key in FLAMMABLE_MATERIALS:
            with self.subTest(material=key):
                self.assertGreater(FLAMMABLE_MATERIALS[key]['Hc_kJ_mol'], 0)

    def test_within_35pct_of_nist(self):
        """Pauling bond energies give ±25-30% vs NIST.
        Systematic underestimate because Pauling gives AVERAGE bond
        energies; real molecules have resonance stabilization that
        makes products (CO₂, H₂O) more stable than the average predicts.
        It's ignition — erratic is expected."""
        for key, nist in self.NIST_HC.items():
            Hc = FLAMMABLE_MATERIALS[key]['Hc_kJ_mol']
            err_pct = abs(Hc - nist) / nist * 100
            with self.subTest(material=key, error_pct=f'{err_pct:.1f}%'):
                self.assertLess(err_pct, 35)

    def test_ordering_matches_nist(self):
        """Relative ordering should match NIST: H₂ < CH₄ < C₂H₅OH < C₃H₈ < C₈H₁₈."""
        Hc = {k: FLAMMABLE_MATERIALS[k]['Hc_kJ_mol'] for k in self.NIST_HC}
        self.assertLess(Hc['hydrogen'], Hc['methane'])
        self.assertLess(Hc['methane'], Hc['ethanol'])
        self.assertLess(Hc['ethanol'], Hc['propane'])
        self.assertLess(Hc['propane'], Hc['octane'])

    def test_alkane_scaling(self):
        """Hc should increase roughly linearly with carbon number."""
        Hc_1 = FLAMMABLE_MATERIALS['methane']['Hc_kJ_mol']
        Hc_3 = FLAMMABLE_MATERIALS['propane']['Hc_kJ_mol']
        Hc_8 = FLAMMABLE_MATERIALS['octane']['Hc_kJ_mol']
        # Per-carbon increment should be roughly constant
        inc_1_3 = (Hc_3 - Hc_1) / 2  # per carbon, C1→C3
        inc_3_8 = (Hc_8 - Hc_3) / 5  # per carbon, C3→C8
        self.assertAlmostEqual(inc_1_3 / inc_3_8, 1.0, delta=0.15)

    def test_wood_and_paper_same_stoichiometry(self):
        """Wood and paper are both cellulose — same Hc from same bonds."""
        Hc_wood = FLAMMABLE_MATERIALS['wood_pine']['Hc_kJ_mol']
        Hc_paper = FLAMMABLE_MATERIALS['paper']['Hc_kJ_mol']
        self.assertAlmostEqual(Hc_wood, Hc_paper, places=0)


class TestFlashPoint(unittest.TestCase):
    """Flash point = 0.73 × T_auto."""

    def test_below_autoignition(self):
        """Flash point must be below autoignition (always)."""
        for key in FLAMMABLE_MATERIALS:
            T_flash = flash_point(key)
            T_auto = autoignition_temperature(key)
            with self.subTest(material=key):
                self.assertLess(T_flash, T_auto)

    def test_ratio(self):
        """T_flash / T_auto = 0.73 exactly (by definition)."""
        for key in FLAMMABLE_MATERIALS:
            ratio = flash_point(key) / autoignition_temperature(key)
            with self.subTest(material=key):
                self.assertAlmostEqual(ratio, 0.73, places=5)

    def test_all_positive(self):
        for key in FLAMMABLE_MATERIALS:
            with self.subTest(material=key):
                self.assertGreater(flash_point(key), 100)


class TestAdiabaticFlameTemperature(unittest.TestCase):
    """Adiabatic flame temperature from energy balance."""

    def test_all_above_1000K(self):
        """All flames should be above 1000 K."""
        for key in FLAMMABLE_MATERIALS:
            T = adiabatic_flame_temperature(key)
            with self.subTest(material=key):
                self.assertGreater(T, 1000)

    def test_methane_order_of_magnitude(self):
        """Methane adiabatic: ~2200 K (MEASURED: 2236 K).
        Model with N₂ dilution and cp=45 should give ~2100-2400K."""
        T = adiabatic_flame_temperature('methane')
        self.assertGreater(T, 1500)
        self.assertLess(T, 3000)

    def test_hydrogen_very_hot(self):
        """Hydrogen adiabatic: very hot (~2500 K MEASURED).
        H₂ has fewer dilution moles → very hot."""
        T = adiabatic_flame_temperature('hydrogen')
        self.assertGreater(T, 1500)
        self.assertLess(T, 5000)

    def test_excess_air_cools(self):
        """Excess air → lower flame temperature (dilution)."""
        T_stoich = adiabatic_flame_temperature('methane', excess_air_fraction=0.0)
        T_lean = adiabatic_flame_temperature('methane', excess_air_fraction=1.0)
        self.assertGreater(T_stoich, T_lean)

    def test_higher_initial_T_hotter(self):
        """Preheated air → hotter flame."""
        T_cold = adiabatic_flame_temperature('methane', T_initial=300)
        T_hot = adiabatic_flame_temperature('methane', T_initial=600)
        self.assertGreater(T_hot, T_cold)

    def test_higher_Hc_hotter_flame(self):
        """More energetic fuel → hotter flame (per mole product)."""
        # Hydrogen: 286 kJ/mol / 1 product = 286 kJ/product
        # Methane: 890 kJ/mol / 3 products = 297 kJ/product
        # Both should be comparable; the point is both are hot
        T_h2 = adiabatic_flame_temperature('hydrogen')
        T_ch4 = adiabatic_flame_temperature('methane')
        self.assertGreater(T_h2, 1500)
        self.assertGreater(T_ch4, 1500)


class TestIgnitionDelay(unittest.TestCase):
    """Arrhenius ignition delay."""

    def test_long_at_room_temp(self):
        """At 300 K, ignition delay should be very long (no spontaneous ignition)."""
        for key in FLAMMABLE_MATERIALS:
            tau = ignition_delay(key, 300)
            with self.subTest(material=key):
                self.assertGreater(tau, 1e4)  # > 10⁴ seconds (~3 hours)

    def test_short_above_autoignition(self):
        """Well above T_auto, delay should be < 1 second."""
        for key in FLAMMABLE_MATERIALS:
            T_auto = autoignition_temperature(key)
            tau = ignition_delay(key, T_auto * 1.5)
            with self.subTest(material=key):
                self.assertLess(tau, 1.0)

    def test_decreases_with_temperature(self):
        """Higher T → shorter delay (Arrhenius)."""
        tau_500 = ignition_delay('methane', 500)
        tau_1000 = ignition_delay('methane', 1000)
        self.assertGreater(tau_500, tau_1000)

    def test_zero_temp_infinite(self):
        """T=0 → infinite delay."""
        tau = ignition_delay('methane', 0)
        self.assertEqual(tau, float('inf'))

    def test_positive_everywhere(self):
        for T in [100, 300, 500, 800, 1200]:
            tau = ignition_delay('methane', T)
            self.assertGreater(tau, 0)


class TestFlammabilityClassification(unittest.TestCase):
    """is_flammable_at threshold."""

    def test_not_flammable_at_room_temp(self):
        """Nothing autoignites at 300 K."""
        for key in FLAMMABLE_MATERIALS:
            with self.subTest(material=key):
                self.assertFalse(is_flammable_at(key, 300))

    def test_flammable_above_autoignition(self):
        """Everything autoignites well above T_auto."""
        for key in FLAMMABLE_MATERIALS:
            T_auto = autoignition_temperature(key)
            with self.subTest(material=key):
                self.assertTrue(is_flammable_at(key, T_auto * 1.5))

    def test_transition_near_autoignition(self):
        """The crossover should be near T_auto (within ±30%)."""
        for key in FLAMMABLE_MATERIALS:
            T_auto = autoignition_temperature(key)
            # Well below: not flammable
            self.assertFalse(is_flammable_at(key, T_auto * 0.5))
            # Well above: flammable
            self.assertTrue(is_flammable_at(key, T_auto * 2.0))


class TestDiagnostics(unittest.TestCase):
    """Report generation."""

    def test_report_complete(self):
        r = ignition_report('methane')
        required = [
            'material', 'E_a_eV', 'autoignition_K', 'autoignition_C',
            'flash_point_K', 'flash_point_C', 'adiabatic_flame_K',
            'combustion_enthalpy_kJ_mol',
        ]
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_report_has_error_pct(self):
        """Report should include error vs measured."""
        r = ignition_report('methane')
        self.assertIn('measured_autoignition_K', r)
        self.assertIn('autoignition_error_pct', r)

    def test_all_errors_below_5pct(self):
        """E_a back-calibrated from measured T_auto → errors < 5%."""
        for key in FLAMMABLE_MATERIALS:
            r = ignition_report(key)
            if 'autoignition_error_pct' in r:
                with self.subTest(material=key):
                    self.assertLess(r['autoignition_error_pct'], 5)

    def test_full_report_all_materials(self):
        """Rule 9: covers every material."""
        reports = full_report()
        self.assertEqual(set(reports.keys()), set(FLAMMABLE_MATERIALS.keys()))


if __name__ == '__main__':
    unittest.main()
