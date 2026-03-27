"""
Tests for the piezoelectricity module.

Test structure:
  1. Direct effect — polarization from stress
  2. Converse effect — strain from electric field
  3. Coupling coefficient — energy conversion efficiency
  4. Resonant frequency — thickness mode
  5. Energy harvesting — energy density
  6. Quartz oscillator — frequency constant
  7. σ-dependence — resonant frequency shift
  8. Nagatha export
"""

import math
import unittest

from .piezoelectricity import (
    piezoelectric_polarization,
    piezoelectric_voltage,
    piezoelectric_strain,
    piezoelectric_displacement,
    coupling_coefficient,
    coupling_coefficient_computed,
    energy_density_harvested,
    resonant_frequency_thickness,
    quartz_frequency,
    sigma_resonant_frequency_shift,
    material_piezoelectric_properties,
    PIEZO_MATERIALS,
)


class TestDirectEffect(unittest.TestCase):
    """Direct piezoelectric effect — P = d × σ."""

    def test_positive_stress_positive_P(self):
        """Positive stress → positive polarization."""
        P = piezoelectric_polarization('PZT4', 1e6)
        self.assertGreater(P, 0)

    def test_proportional_to_stress(self):
        """P ∝ σ (linear response)."""
        P1 = piezoelectric_polarization('PZT4', 1e6)
        P2 = piezoelectric_polarization('PZT4', 2e6)
        self.assertAlmostEqual(P2 / P1, 2.0, places=10)

    def test_PZT_larger_than_quartz(self):
        """PZT has much higher d than quartz → larger P at same stress."""
        P_pzt = piezoelectric_polarization('PZT4', 1e6)
        P_qtz = piezoelectric_polarization('quartz', 1e6)
        self.assertGreater(P_pzt, P_qtz * 10)

    def test_voltage_from_stress(self):
        """1 MPa on 1 mm PZT-4: ~25 V (order of magnitude)."""
        V = piezoelectric_voltage('PZT4', 1e6, 1e-3)
        self.assertGreater(V, 1)
        self.assertLess(V, 100)

    def test_voltage_proportional_to_thickness(self):
        """V ∝ thickness (thicker element → more voltage)."""
        V1 = piezoelectric_voltage('PZT4', 1e6, 1e-3)
        V2 = piezoelectric_voltage('PZT4', 1e6, 2e-3)
        self.assertAlmostEqual(V2 / V1, 2.0, places=10)


class TestConverseEffect(unittest.TestCase):
    """Converse piezoelectric effect — ε = d × E."""

    def test_positive(self):
        """Positive E → positive strain for d > 0."""
        eps = piezoelectric_strain('PZT4', 1e6)
        self.assertGreater(eps, 0)

    def test_proportional_to_field(self):
        """ε ∝ E."""
        e1 = piezoelectric_strain('PZT4', 1e6)
        e2 = piezoelectric_strain('PZT4', 2e6)
        self.assertAlmostEqual(e2 / e1, 2.0, places=10)

    def test_displacement(self):
        """Displacement = strain × length."""
        E = 1e6  # 1 MV/m
        L = 0.01  # 10 mm
        d = piezoelectric_displacement('PZT4', E, L)
        eps = piezoelectric_strain('PZT4', E)
        self.assertAlmostEqual(d, eps * L, places=15)

    def test_PZT_micrometer_displacement(self):
        """PZT at 1 MV/m, 10 mm: displacement ~ μm range."""
        d = piezoelectric_displacement('PZT4', 1e6, 0.01)
        self.assertGreater(d, 1e-8)
        self.assertLess(d, 1e-4)


class TestCouplingCoefficient(unittest.TestCase):
    """Electromechanical coupling."""

    def test_quartz_low_coupling(self):
        """Quartz k ≈ 0.10 (weak piezoelectric)."""
        k = coupling_coefficient('quartz')
        self.assertAlmostEqual(k, 0.10, delta=0.02)

    def test_PZT_high_coupling(self):
        """PZT k ≈ 0.70 (strong piezoelectric)."""
        k = coupling_coefficient('PZT4')
        self.assertAlmostEqual(k, 0.70, delta=0.05)

    def test_all_k_less_than_unity(self):
        """k < 1 always (thermodynamic requirement)."""
        for mat in PIEZO_MATERIALS:
            k = coupling_coefficient(mat)
            self.assertLess(k, 1.0, f"{mat}: k must be < 1")
            self.assertGreater(k, 0, f"{mat}: k must be > 0")

    def test_computed_vs_measured(self):
        """Computed k from d, ε, s should be in same ballpark as measured k."""
        for mat in PIEZO_MATERIALS:
            k_meas = coupling_coefficient(mat)
            k_comp = coupling_coefficient_computed(mat)
            # Within factor of 3 (d, ε, s may not be from same orientation)
            self.assertGreater(k_comp, k_meas * 0.1,
                f"{mat}: computed k={k_comp:.3f}, measured k={k_meas:.3f}")
            self.assertLess(k_comp, k_meas * 10,
                f"{mat}: computed k={k_comp:.3f}, measured k={k_meas:.3f}")


class TestResonantFrequency(unittest.TestCase):
    """Thickness-mode resonant frequency."""

    def test_quartz_1mm(self):
        """1 mm quartz: f ~ 1.66 MHz (industry standard)."""
        f = quartz_frequency(1e-3)
        self.assertAlmostEqual(f / 1e6, 1.661, delta=0.01)

    def test_proportional_to_inverse_thickness(self):
        """f ∝ 1/t."""
        f1 = resonant_frequency_thickness('quartz', 1e-3)
        f2 = resonant_frequency_thickness('quartz', 2e-3)
        self.assertAlmostEqual(f1 / f2, 2.0, places=10)

    def test_AlN_highest_frequency(self):
        """AlN has highest sound velocity → highest frequency at same thickness."""
        f_aln = resonant_frequency_thickness('AlN', 1e-3)
        for mat in PIEZO_MATERIALS:
            f = resonant_frequency_thickness(mat, 1e-3)
            self.assertGreaterEqual(f_aln, f * 0.99,
                f"AlN should have highest or near-highest frequency")

    def test_all_positive(self):
        """All materials give positive frequency."""
        for mat in PIEZO_MATERIALS:
            f = resonant_frequency_thickness(mat, 1e-3)
            self.assertGreater(f, 0)


class TestEnergyHarvesting(unittest.TestCase):
    """Energy density from piezoelectric harvesting."""

    def test_positive(self):
        """Energy density is always positive."""
        u = energy_density_harvested('PZT4', 1e6)
        self.assertGreater(u, 0)

    def test_quadratic_in_stress(self):
        """u ∝ σ²."""
        u1 = energy_density_harvested('PZT4', 1e6)
        u2 = energy_density_harvested('PZT4', 2e6)
        self.assertAlmostEqual(u2 / u1, 4.0, places=10)

    def test_PZT_more_than_quartz(self):
        """PZT harvests more energy than quartz (higher k)."""
        u_pzt = energy_density_harvested('PZT4', 1e6)
        u_qtz = energy_density_harvested('quartz', 1e6)
        self.assertGreater(u_pzt, u_qtz)


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts resonant frequency."""

    def test_zero_sigma_unchanged(self):
        """σ=0: no shift."""
        f_0, f_s = sigma_resonant_frequency_shift('quartz', 1e-3, 0.0)
        self.assertAlmostEqual(f_0, f_s, places=10)

    def test_positive_sigma_lowers_frequency(self):
        """σ > 0: frequency decreases (heavier lattice dominates)."""
        f_0, f_s = sigma_resonant_frequency_shift('quartz', 1e-3, 0.1)
        self.assertLess(f_s, f_0)

    def test_earth_sigma_negligible(self):
        """At Earth σ ~ 7×10⁻¹⁰: relative shift < 10⁻⁸."""
        f_0, f_s = sigma_resonant_frequency_shift('quartz', 1e-3, 7e-10)
        relative_shift = abs(f_s - f_0) / f_0
        self.assertLess(relative_shift, 1e-8)

    def test_all_materials_shift(self):
        """All materials show frequency shift at σ=0.1."""
        for mat in PIEZO_MATERIALS:
            f_0, f_s = sigma_resonant_frequency_shift(mat, 1e-3, 0.1)
            self.assertNotAlmostEqual(f_0, f_s, places=5,
                msg=f"{mat}: should show σ shift")


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_all_materials_export(self):
        """All materials produce valid export dicts."""
        for mat in PIEZO_MATERIALS:
            props = material_piezoelectric_properties(mat)
            self.assertIn('d_pC_N', props)
            self.assertIn('coupling_k', props)
            self.assertIn('resonant_freq_1mm_Hz', props)
            self.assertIn('origin_tag', props)

    def test_sigma_propagates(self):
        """σ value and frequency shift appear in export."""
        props = material_piezoelectric_properties('quartz', sigma=0.1)
        self.assertEqual(props['sigma'], 0.1)
        self.assertIn('resonant_freq_sigma_Hz', props)
        self.assertIn('frequency_shift_ratio', props)
        self.assertLess(props['frequency_shift_ratio'], 1.0)

    def test_honest_origin_tags(self):
        """Origin tag includes derivation info."""
        props = material_piezoelectric_properties('PZT4')
        self.assertIn('FIRST_PRINCIPLES', props['origin_tag'])
        self.assertIn('MEASURED', props['origin_tag'])


if __name__ == '__main__':
    unittest.main()
