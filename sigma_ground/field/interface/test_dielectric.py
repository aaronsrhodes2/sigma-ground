"""
Tests for dielectric.py — permittivity from electronic structure.

Strategy:
  - Test static ε_r values against MEASURED data
  - Test frequency dependence (Debye relaxation)
  - Test Clausius-Mossotti gives correct ε_r
  - Test breakdown field scales with bandgap
  - Test energy density formula
  - Test Rule 9: every material gets a report

Reference values (MEASURED, CRC Handbook):
  Silicon: ε_r = 11.7
  SiO₂: ε_r = 3.9
  Water (25°C): ε_r = 78.4 (static), 1.77 (optical)
  BaTiO₃: ε_r ≈ 1700 (ferroelectric)
  Diamond: ε_r = 5.7
"""

import math
import unittest

from sigma_ground.field.interface.dielectric import (
    dielectric_constant,
    dielectric_loss_tangent,
    clausius_mossotti,
    breakdown_field,
    breakdown_voltage,
    energy_density,
    max_energy_density,
    dielectric_report,
    full_report,
    DIELECTRIC_DATA,
)
from sigma_ground.field.constants import EPS_0


class TestStaticPermittivity(unittest.TestCase):
    """Static dielectric constants match stored MEASURED values."""

    def test_silicon(self):
        self.assertAlmostEqual(dielectric_constant('silicon'), 11.7, places=1)

    def test_sio2(self):
        self.assertAlmostEqual(dielectric_constant('silicon_dioxide'), 3.9, places=1)

    def test_water(self):
        self.assertAlmostEqual(dielectric_constant('water_25C'), 78.4, places=1)

    def test_batio3_very_high(self):
        """BaTiO₃ is ferroelectric: ε_r ≈ 1700."""
        eps = dielectric_constant('barium_titanate')
        self.assertGreater(eps, 1000)
        self.assertLess(eps, 3000)

    def test_diamond(self):
        self.assertAlmostEqual(dielectric_constant('diamond'), 5.7, places=1)

    def test_metals_return_eps_inf(self):
        """Metals return optical ε_∞ (DC is meaningless)."""
        for key in DIELECTRIC_DATA:
            if DIELECTRIC_DATA[key]['type'] == 'metal':
                eps = dielectric_constant(key)
                with self.subTest(metal=key):
                    self.assertEqual(eps, DIELECTRIC_DATA[key]['eps_inf'])

    def test_all_positive(self):
        """Every ε_r should be positive."""
        for key in DIELECTRIC_DATA:
            with self.subTest(material=key):
                self.assertGreater(dielectric_constant(key), 0)


class TestFrequencyDependence(unittest.TestCase):
    """Debye relaxation model."""

    def test_water_decreases_at_microwave(self):
        """Water ε_r drops from 78 at DC to ~5 at 100 GHz."""
        eps_dc = dielectric_constant('water_25C', 0)
        eps_100ghz = dielectric_constant('water_25C', 100e9)
        self.assertGreater(eps_dc, eps_100ghz)
        self.assertLess(eps_100ghz, 20)

    def test_optical_approaches_eps_inf(self):
        """At optical frequencies (10¹⁴ Hz), ε → ε_∞."""
        for key in ['water_25C', 'barium_titanate', 'alumina']:
            eps_opt = dielectric_constant(key, 1e14)
            eps_inf = DIELECTRIC_DATA[key]['eps_inf']
            with self.subTest(material=key):
                self.assertAlmostEqual(eps_opt, eps_inf, delta=eps_inf * 0.1)

    def test_monotone_decrease(self):
        """ε_r should decrease monotonically with frequency."""
        freqs = [1e6, 1e9, 1e10, 1e11, 1e12, 1e14]
        eps_prev = dielectric_constant('water_25C', 0)
        for f in freqs:
            eps = dielectric_constant('water_25C', f)
            self.assertLessEqual(eps, eps_prev + 0.01)
            eps_prev = eps

    def test_nonpolar_insulator_no_dispersion(self):
        """Polyethylene (nonpolar): ε_static ≈ ε_∞, no dispersion."""
        eps_dc = dielectric_constant('polyethylene', 0)
        eps_ghz = dielectric_constant('polyethylene', 1e9)
        self.assertAlmostEqual(eps_dc, eps_ghz, delta=0.1)


class TestDielectricLoss(unittest.TestCase):
    """Loss tangent tan δ."""

    def test_zero_at_dc(self):
        """tan δ = 0 at f=0."""
        self.assertEqual(dielectric_loss_tangent('water_25C', 0), 0.0)

    def test_metals_zero(self):
        """Metals return 0 (conduction loss not modeled here)."""
        self.assertEqual(dielectric_loss_tangent('copper', 1e9), 0.0)

    def test_water_peak_near_relaxation(self):
        """Water loss peaks near 18 GHz (Debye relaxation)."""
        tan_1ghz = dielectric_loss_tangent('water_25C', 1e9)
        tan_18ghz = dielectric_loss_tangent('water_25C', 18e9)
        tan_1thz = dielectric_loss_tangent('water_25C', 1e12)
        # Peak should be near 18 GHz
        self.assertGreater(tan_18ghz, tan_1ghz)
        self.assertGreater(tan_18ghz, tan_1thz)

    def test_positive(self):
        """Loss tangent should be non-negative."""
        for key in DIELECTRIC_DATA:
            if DIELECTRIC_DATA[key]['type'] != 'metal':
                tan_d = dielectric_loss_tangent(key, 1e9)
                with self.subTest(material=key):
                    self.assertGreaterEqual(tan_d, 0)


class TestClausiusMossotti(unittest.TestCase):
    """ε_r from molecular polarizability."""

    def test_low_density_gives_near_one(self):
        """Very dilute gas → ε_r ≈ 1."""
        # Code expects SI polarizability α in F·m² (not volume polarizability)
        # Typical: ~1e-40 F·m². Low density: 1e20 /m³
        eps = clausius_mossotti(1e-40, 1e20)  # very low N×α
        self.assertAlmostEqual(eps, 1.0, delta=0.01)

    def test_increases_with_density(self):
        """Higher density → higher ε_r."""
        eps_lo = clausius_mossotti(1e-40, 1e27)
        eps_hi = clausius_mossotti(1e-40, 1e28)
        self.assertGreater(eps_hi, eps_lo)

    def test_increases_with_polarizability(self):
        eps_lo = clausius_mossotti(1e-41, 1e28)
        eps_hi = clausius_mossotti(1e-40, 1e28)
        self.assertGreater(eps_hi, eps_lo)

    def test_caps_at_ferroelectric(self):
        """Very high χ → caps at 1000 (ferroelectric instability)."""
        eps = clausius_mossotti(1e-20, 1e30)  # absurdly high
        self.assertEqual(eps, 1000.0)


class TestBreakdownField(unittest.TestCase):
    """Dielectric breakdown from avalanche model."""

    def test_metals_zero(self):
        """Metals don't break down (they conduct)."""
        for key in DIELECTRIC_DATA:
            if DIELECTRIC_DATA[key]['type'] == 'metal':
                with self.subTest(metal=key):
                    self.assertEqual(breakdown_field(key), 0.0)

    def test_wider_gap_higher_breakdown(self):
        """Wider bandgap → higher breakdown field."""
        E_si = breakdown_field('silicon')       # 1.12 eV
        E_sio2 = breakdown_field('silicon_dioxide')  # 9.0 eV
        self.assertGreater(E_sio2, E_si)

    def test_sio2_order_of_magnitude(self):
        """SiO₂ intrinsic breakdown: model gives upper bound.
        E_b = 0.1 × E_g/(e×d) ≈ 1800 MV/m for SiO₂.
        Real measured: ~15-25 MV/m (defects, thickness effects).
        Model gives the avalanche THRESHOLD, not practical breakdown."""
        E_b = breakdown_field('silicon_dioxide') / 1e6  # MV/m
        self.assertGreater(E_b, 100)   # intrinsic limit is high
        self.assertLess(E_b, 5000)

    def test_silicon_breakdown(self):
        """Silicon intrinsic breakdown: model gives upper bound.
        Real measured: ~30 MV/m. Model: ~200 MV/m (avalanche threshold)."""
        E_b = breakdown_field('silicon') / 1e6
        self.assertGreater(E_b, 50)
        self.assertLess(E_b, 500)

    def test_all_insulators_positive(self):
        for key in DIELECTRIC_DATA:
            if DIELECTRIC_DATA[key]['type'] != 'metal':
                with self.subTest(material=key):
                    self.assertGreater(breakdown_field(key), 0)


class TestBreakdownVoltage(unittest.TestCase):
    """V_b = E_b × d."""

    def test_linear_in_thickness(self):
        """Voltage scales linearly with thickness."""
        V1 = breakdown_voltage('silicon_dioxide', 1e-6)
        V2 = breakdown_voltage('silicon_dioxide', 2e-6)
        self.assertAlmostEqual(V2 / V1, 2.0, delta=0.01)

    def test_sio2_thin_film(self):
        """100 nm SiO₂: V_b = E_b × d. Model gives intrinsic limit.
        Intrinsic: ~180 V for 100nm. Real gate oxide: ~1-3 V (defects)."""
        V = breakdown_voltage('silicon_dioxide', 100e-9)
        self.assertGreater(V, 10)
        self.assertLess(V, 500)


class TestEnergyDensity(unittest.TestCase):
    """Electrostatic energy storage."""

    def test_formula(self):
        """u = ½ ε₀ ε_r E²."""
        E_field = 1e6  # 1 MV/m
        eps_r = dielectric_constant('silicon_dioxide', 0)
        u = energy_density('silicon_dioxide', E_field)
        expected = 0.5 * EPS_0 * eps_r * E_field**2
        self.assertAlmostEqual(u, expected, places=5)

    def test_higher_eps_higher_energy(self):
        """Higher ε_r stores more energy at same field."""
        u_sio2 = energy_density('silicon_dioxide', 1e6)
        u_batio3 = energy_density('barium_titanate', 1e6)
        self.assertGreater(u_batio3, u_sio2)

    def test_max_energy_density_positive(self):
        for key in DIELECTRIC_DATA:
            if DIELECTRIC_DATA[key]['type'] != 'metal':
                with self.subTest(material=key):
                    self.assertGreater(max_energy_density(key), 0)


class TestDiagnostics(unittest.TestCase):
    """Report generation."""

    def test_report_complete(self):
        r = dielectric_report('silicon')
        required = ['material', 'type', 'eps_r_static', 'eps_r_optical']
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_insulator_has_breakdown(self):
        r = dielectric_report('silicon_dioxide')
        self.assertIn('breakdown_field_V_m', r)
        self.assertIn('max_energy_density_J_m3', r)

    def test_frequency_report(self):
        r = dielectric_report('water_25C', frequency_hz=1e9)
        self.assertIn('eps_r_at_freq', r)
        self.assertIn('loss_tangent', r)

    def test_full_report_all_materials(self):
        """Rule 9: covers every material."""
        reports = full_report()
        self.assertEqual(set(reports.keys()), set(DIELECTRIC_DATA.keys()))


if __name__ == '__main__':
    unittest.main()
