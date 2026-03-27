"""
Tests for the electrochemistry module.

Test structure:
  1. Nernst equation — standard and non-standard conditions
  2. Cell potential — galvanic cells
  3. Gibbs energy — spontaneity
  4. Activity series — metal ordering
  5. Faraday's laws — electrolysis
  6. Tafel kinetics — overpotential
  7. Ionic conductivity — Kohlrausch
  8. σ-dependence — thermal shift
  9. Nagatha export
"""

import math
import unittest

from .electrochemistry import (
    nernst_potential,
    cell_potential,
    gibbs_energy_cell,
    is_spontaneous,
    activity_series,
    can_displace,
    faraday_mass_deposited,
    faraday_charge_required,
    faraday_time_required,
    tafel_slope,
    tafel_overpotential,
    molar_conductivity_dilute,
    solution_conductivity,
    sigma_nernst_shift,
    material_electrochemical_properties,
    STANDARD_POTENTIALS,
    FARADAY,
)


class TestNernstEquation(unittest.TestCase):
    """Nernst equation — electrode potentials."""

    def test_standard_conditions(self):
        """Q=1 gives E = E°."""
        E0 = -0.447  # iron
        E = nernst_potential(E0, 2, 1.0)
        self.assertAlmostEqual(E, E0, places=10)

    def test_higher_Q_lowers_potential(self):
        """Higher Q → lower E (Le Chatelier)."""
        E0 = 0.342  # copper
        E1 = nernst_potential(E0, 2, 1.0)
        E2 = nernst_potential(E0, 2, 10.0)
        self.assertGreater(E1, E2)

    def test_lower_Q_raises_potential(self):
        """Lower Q → higher E."""
        E0 = 0.342
        E1 = nernst_potential(E0, 2, 1.0)
        E2 = nernst_potential(E0, 2, 0.1)
        self.assertLess(E1, E2)

    def test_nernst_at_25C(self):
        """At 25°C, RT/F ≈ 25.7 mV (thermal voltage)."""
        # E = E° - (RT/nF) ln(Q) = E° - (0.0257/n) ln(Q)
        E0 = 0.0
        E = nernst_potential(E0, 1, math.e)  # Q = e → ln(Q) = 1
        # Should be -0.0257 V at 298.15 K
        self.assertAlmostEqual(E, -0.0257, delta=0.001)

    def test_invalid_Q(self):
        """Q ≤ 0 raises ValueError."""
        with self.assertRaises(ValueError):
            nernst_potential(0.0, 1, 0.0)
        with self.assertRaises(ValueError):
            nernst_potential(0.0, 1, -1.0)


class TestCellPotential(unittest.TestCase):
    """Cell potential for galvanic cells."""

    def test_daniell_cell(self):
        """Daniell cell: Zn|Cu → E ≈ 1.10 V."""
        E = cell_potential('copper', 'zinc')
        self.assertAlmostEqual(E, 1.104, delta=0.01)

    def test_gold_iron(self):
        """Au³⁺/Fe cell: highly spontaneous."""
        E = cell_potential('gold', 'iron')
        self.assertGreater(E, 1.5)

    def test_same_electrode(self):
        """Same cathode and anode: E = 0."""
        E = cell_potential('copper', 'copper')
        self.assertAlmostEqual(E, 0.0, places=10)

    def test_reversed_gives_negative(self):
        """Reversed cell: E < 0 (non-spontaneous)."""
        E = cell_potential('zinc', 'copper')
        self.assertLess(E, 0)


class TestGibbsEnergy(unittest.TestCase):
    """Gibbs free energy of cell reactions."""

    def test_spontaneous_negative_gibbs(self):
        """Spontaneous cell (E>0) → ΔG < 0."""
        dG = gibbs_energy_cell('copper', 'zinc')
        self.assertLess(dG, 0)

    def test_non_spontaneous_positive_gibbs(self):
        """Non-spontaneous cell (E<0) → ΔG > 0."""
        dG = gibbs_energy_cell('zinc', 'copper')
        self.assertGreater(dG, 0)

    def test_is_spontaneous(self):
        """is_spontaneous matches sign of E."""
        self.assertTrue(is_spontaneous('copper', 'zinc'))
        self.assertFalse(is_spontaneous('zinc', 'copper'))

    def test_gibbs_magnitude(self):
        """Daniell cell ΔG ≈ −213 kJ/mol (n=2, E≈1.1V)."""
        dG = gibbs_energy_cell('copper', 'zinc')
        self.assertAlmostEqual(dG / 1000, -213, delta=5)


class TestActivitySeries(unittest.TestCase):
    """Activity series ordering."""

    def test_ordered(self):
        """Series is sorted by E° (most negative first)."""
        series = activity_series()
        potentials = [e for _, e in series]
        self.assertEqual(potentials, sorted(potentials))

    def test_lithium_most_reactive(self):
        """Lithium is most reactive (most negative E°)."""
        series = activity_series()
        self.assertEqual(series[0][0], 'lithium')

    def test_gold_least_reactive(self):
        """Gold is least reactive (most positive E°)."""
        series = activity_series()
        self.assertEqual(series[-1][0], 'gold')

    def test_hydrogen_in_middle(self):
        """Hydrogen (E°=0) is the reference point."""
        series = activity_series()
        h_pos = next(i for i, (k, _) in enumerate(series) if k == 'hydrogen')
        # Some metals above, some below
        self.assertGreater(h_pos, 0)
        self.assertLess(h_pos, len(series) - 1)

    def test_displacement_iron_copper(self):
        """Iron can displace copper from solution."""
        self.assertTrue(can_displace('iron', 'copper'))
        self.assertFalse(can_displace('copper', 'iron'))

    def test_displacement_zinc_iron(self):
        """Zinc can displace iron from solution."""
        self.assertTrue(can_displace('zinc', 'iron'))


class TestFaradayLaws(unittest.TestCase):
    """Faraday's laws of electrolysis."""

    def test_copper_deposition(self):
        """1 A for 1 hour deposits ~1.19 g of Cu (M=63.5g/mol, n=2)."""
        m = faraday_mass_deposited(63.5e-3, 1.0, 3600, 2)
        self.assertAlmostEqual(m * 1000, 1.185, delta=0.01)

    def test_charge_mass_consistency(self):
        """Charge to deposit mass M/n = F."""
        q = faraday_charge_required(63.5e-3, 63.5e-3 / 2, 2)
        self.assertAlmostEqual(q, FARADAY, delta=1)

    def test_time_current_consistency(self):
        """t × I = q."""
        m = 1e-3  # 1 g
        I = 2.0   # 2 A
        M = 63.5e-3
        n = 2
        t = faraday_time_required(M, m, I, n)
        q = faraday_charge_required(M, m, n)
        self.assertAlmostEqual(t * I, q, places=5)

    def test_proportional_to_current(self):
        """Doubling current halves deposition time."""
        t1 = faraday_time_required(63.5e-3, 1e-3, 1.0, 2)
        t2 = faraday_time_required(63.5e-3, 1e-3, 2.0, 2)
        self.assertAlmostEqual(t1 / t2, 2.0, places=10)


class TestTafelKinetics(unittest.TestCase):
    """Tafel equation — overpotential vs current density."""

    def test_tafel_slope_at_25C(self):
        """Tafel slope at 25°C, α=0.5, n=1: b ≈ 118 mV/decade."""
        b = tafel_slope(298.15, 0.5, 1)
        self.assertAlmostEqual(b * 1000, 118.3, delta=1)

    def test_slope_increases_with_T(self):
        """Higher T → steeper Tafel slope."""
        b1 = tafel_slope(298.15)
        b2 = tafel_slope(373.15)
        self.assertGreater(b2, b1)

    def test_overpotential_positive(self):
        """j > j₀ gives positive overpotential (anodic)."""
        eta = tafel_overpotential(10.0, 1.0)
        self.assertGreater(eta, 0)

    def test_overpotential_at_j0(self):
        """j = j₀ gives η = 0."""
        eta = tafel_overpotential(1.0, 1.0)
        self.assertAlmostEqual(eta, 0.0, places=10)

    def test_invalid_current(self):
        """Non-positive current raises ValueError."""
        with self.assertRaises(ValueError):
            tafel_overpotential(0, 1.0)


class TestIonicConductivity(unittest.TestCase):
    """Kohlrausch law — electrolyte conductivity."""

    def test_additive(self):
        """Λ° = λ₊ + λ₋."""
        Lam = molar_conductivity_dilute(7.35e-3, 7.63e-3)  # NaCl
        self.assertAlmostEqual(Lam, 14.98e-3, delta=1e-5)

    def test_conductivity_proportional(self):
        """κ = c × Λ."""
        kappa = solution_conductivity(1000, 0.01)  # 1 mol/m³ × 0.01
        self.assertAlmostEqual(kappa, 10.0, places=5)


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts Nernst potential."""

    def test_zero_sigma_unchanged(self):
        """σ=0 gives standard Nernst potential."""
        E_std = nernst_potential(-0.447, 2, 1.0)
        E_sigma = sigma_nernst_shift(-0.447, 2, 1.0, 298.15, 0.0)
        self.assertAlmostEqual(E_std, E_sigma, places=10)

    def test_nonzero_sigma_shifts(self):
        """σ > 0 shifts the Nernst potential (when Q ≠ 1)."""
        # Q=1 has ln(Q)=0, so thermal term vanishes — use Q≠1
        E_0 = sigma_nernst_shift(-0.447, 2, 0.01, 298.15, 0.0)
        E_s = sigma_nernst_shift(-0.447, 2, 0.01, 298.15, 0.1)
        self.assertNotAlmostEqual(E_0, E_s, places=5)

    def test_earth_sigma_negligible(self):
        """At Earth σ ~ 7×10⁻¹⁰: shift < 10⁻⁸ V."""
        E_0 = sigma_nernst_shift(0.342, 2, 1.0, 298.15, 0.0)
        E_s = sigma_nernst_shift(0.342, 2, 1.0, 298.15, 7e-10)
        self.assertAlmostEqual(E_0, E_s, places=8)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_all_elements_export(self):
        """All standard potential elements produce valid export."""
        for elem in STANDARD_POTENTIALS:
            props = material_electrochemical_properties(elem)
            self.assertIn('standard_potential_V', props)
            self.assertIn('n_electrons', props)
            self.assertIn('origin_tag', props)

    def test_sigma_propagates(self):
        """σ value appears in export."""
        props = material_electrochemical_properties('copper', sigma=0.1)
        self.assertEqual(props['sigma'], 0.1)

    def test_unknown_element(self):
        """Unknown element returns error dict."""
        props = material_electrochemical_properties('unobtainium')
        self.assertIn('error', props)


if __name__ == '__main__':
    unittest.main()
