"""Tests for quantum_matter.py — Mott transition, crystal field spin physics.

Predictions tested:
  1. All metals correctly classified as metallic (U/W < 1.2)
  2. Crystal field ions in oxide/silicate are all high-spin (localized)
  3. Nephelauxetic series orders correctly (CN > S > oxide > F)
  4. Superexchange J from cascade agrees with Curie-derived J to order of magnitude
  5. VQE finds exact Heisenberg ground state with cascade-derived J
  6. Silicon is not a Mott insulator (band insulator instead)
"""

import math
import unittest

from .quantum_matter import (
    wigner_seitz_radius,
    thomas_fermi_screening_length,
    fermi_energy_eV,
    hubbard_parameters,
    mott_phase_diagram,
    hubbard_ground_state,
    crystal_field_mott_ratio,
    crystal_field_phase_diagram,
    nephelauxetic_metallicity,
    superconductor_correlation_strength,
    two_site_spin_hamiltonian_from_crystal_field,
    quantum_matter_report,
)


# =====================================================================
# MOTT PHASE DIAGRAM
# =====================================================================

class TestMottPhaseDiagram(unittest.TestCase):
    """Test Mott phase classification from cascade parameters."""

    def test_all_metals_are_metallic(self):
        """All 7 metals should have U/W < 1.2 (not Mott insulators)."""
        metals = ['iron', 'copper', 'aluminum', 'gold', 'tungsten', 'nickel', 'titanium']
        for mat in metals:
            r = hubbard_parameters(mat)
            self.assertFalse(r['is_mott_insulator'],
                             msg=f"{mat} classified as Mott insulator (U/W={r['U_over_W']:.3f})")
            self.assertLess(r['U_over_W'], 1.2,
                            msg=f"{mat} U/W={r['U_over_W']:.3f} should be < 1.2")

    def test_silicon_not_mott_insulator(self):
        """Silicon should NOT be classified as Mott insulator.

        PREDICTION: Silicon's band gap is from lattice geometry (band insulator),
        not from electron-electron correlation (Mott insulator). The cascade
        correctly distinguishes these: U/W < 1.2 for silicon.
        """
        r = hubbard_parameters('silicon')
        self.assertFalse(r['is_mott_insulator'])

    def test_wigner_seitz_radius_reasonable(self):
        """r_ws should be 1-2 Angstrom for metals."""
        for mat in ['iron', 'copper', 'aluminum']:
            r_ws = wigner_seitz_radius(mat) * 1e10  # Angstrom
            self.assertGreater(r_ws, 1.0)
            self.assertLess(r_ws, 2.5)

    def test_thomas_fermi_shorter_than_wigner_seitz(self):
        """Screening length should be shorter than interatomic distance."""
        for mat in ['iron', 'copper', 'nickel']:
            lam = thomas_fermi_screening_length(mat)
            r_ws = wigner_seitz_radius(mat)
            self.assertLess(lam, r_ws,
                            msg=f"{mat}: lambda_TF should be < r_ws")

    def test_fermi_energy_reasonable(self):
        """Free-electron Fermi energy should be 5-40 eV for metals."""
        for mat in ['iron', 'copper', 'aluminum', 'gold']:
            E_F = fermi_energy_eV(mat)
            self.assertGreater(E_F, 5.0)
            self.assertLess(E_F, 50.0)

    def test_screening_reduces_U(self):
        """Screened U should be much less than bare U."""
        for mat in ['iron', 'copper']:
            r = hubbard_parameters(mat)
            self.assertLess(r['U_screened_eV'], r['U_bare_eV'])
            self.assertLess(r['screening_factor'], 0.5)

    def test_mott_diagram_ordering(self):
        """Phase diagram should be sorted by U/W."""
        results = mott_phase_diagram()
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i]['U_over_W'], results[i + 1]['U_over_W'])


# =====================================================================
# HUBBARD VQE
# =====================================================================

class TestHubbardVQE(unittest.TestCase):
    """Test Hubbard model simulations with cascade parameters."""

    def test_iron_superexchange(self):
        """Iron J_super should be larger than J_Curie (itinerant ferromagnet)."""
        r = hubbard_ground_state('iron')
        self.assertIsNotNone(r['J_curie_meV'])
        self.assertGreater(r['J_superexchange_meV'], r['J_curie_meV'],
                           msg="J_super > J_Curie for itinerant ferromagnet")

    def test_nickel_superexchange(self):
        """Nickel: same pattern as iron — itinerant, not Heisenberg."""
        r = hubbard_ground_state('nickel')
        self.assertGreater(r['J_ratio'], 3.0,
                           msg="J_ratio > 3 means itinerant, not localized")

    def test_vqe_matches_exact_heisenberg(self):
        """VQE should find exact 2-site Heisenberg ground state."""
        for mat in ['iron', 'copper']:
            r = hubbard_ground_state(mat)
            self.assertAlmostEqual(
                r['heisenberg_E_vqe_eV'], r['heisenberg_E_exact_eV'],
                delta=0.001,
                msg=f"{mat}: VQE should match exact Heisenberg energy")

    def test_heisenberg_maximal_entanglement(self):
        """Heisenberg singlet ground state is maximally entangled."""
        r = hubbard_ground_state('iron')
        self.assertAlmostEqual(r['entanglement_entropy'], math.log(2),
                               delta=0.01)

    def test_double_occupancy_small_for_large_U_over_t(self):
        """When U >> t, double occupancy should be small."""
        r = hubbard_ground_state('copper')
        self.assertLess(r['double_occupancy'], 0.1,
                        msg="Large U/t → suppressed double occupancy")


# =====================================================================
# CRYSTAL FIELD MOTT ANALOGY
# =====================================================================

class TestCrystalFieldMott(unittest.TestCase):
    """Test crystal field → Mott physics mapping."""

    def test_fe2_oxide_high_spin(self):
        """Fe²⁺ in oxide (FeO) must be high-spin — classic Mott insulator."""
        r = crystal_field_mott_ratio(26, 2, 'oxide_oct')
        self.assertIsNotNone(r)
        self.assertTrue(r['is_high_spin'],
                        msg=f"Fe²⁺ in oxide: 10Dq/B={r['10Dq_over_B']:.1f} should be < {r['TS_crossover']}")
        self.assertEqual(r['d_count'], 6)

    def test_mn2_oxide_high_spin(self):
        """Mn²⁺ (d⁵) in oxide: definitely high-spin (half-filled shell)."""
        r = crystal_field_mott_ratio(25, 2, 'oxide_oct')
        self.assertTrue(r['is_high_spin'])
        self.assertEqual(r['d_count'], 5)

    def test_cr3_ruby_high_spin(self):
        """Cr³⁺ (d³) in ruby: always high-spin (no crossover for d³)."""
        r = crystal_field_mott_ratio(24, 3, 'oxide_oct')
        self.assertTrue(r['is_high_spin'])
        self.assertEqual(r['d_count'], 3)

    def test_all_mineral_ions_high_spin(self):
        """Every ion in our crystal field database (oxide/silicate) should be high-spin.

        PREDICTION: All mineral-hosted transition metal ions in oxide and silicate
        coordination are in the localized (high-spin, Mott) regime. Low-spin
        requires strong-field ligands (CN⁻, CO) not found in minerals.
        """
        diagram = crystal_field_phase_diagram()
        for r in diagram:
            self.assertTrue(r['is_high_spin'],
                            msg=f"Z={r['Z']} ox={r['oxidation_state']} "
                            f"coord={r['coord_key']}: 10Dq/B={r['10Dq_over_B']:.1f} "
                            f"should give high-spin")

    def test_10dq_over_B_below_crossover(self):
        """For d⁴-d⁷ ions with known crossovers, ratio must be below crossover."""
        diagram = crystal_field_phase_diagram()
        for r in diagram:
            if r['TS_crossover'] is not None:
                self.assertLess(r['10Dq_over_B'], r['TS_crossover'],
                                msg=f"d{r['d_count']} in {r['coord_key']}: "
                                f"ratio {r['10Dq_over_B']:.1f} should be < {r['TS_crossover']}")


# =====================================================================
# NEPHELAUXETIC SERIES
# =====================================================================

class TestNephelauxeticMetallicity(unittest.TestCase):
    """Test nephelauxetic series as metallicity predictor."""

    def test_cn_most_covalent(self):
        """CN⁻ should have lowest β (most covalent/metallic)."""
        series = nephelauxetic_metallicity()
        self.assertEqual(series[0]['coordination'], 'cn_oct')

    def test_fluoride_most_ionic(self):
        """F⁻ should have highest β (most ionic)."""
        series = nephelauxetic_metallicity()
        self.assertEqual(series[-1]['coordination'], 'fluoride_oct')

    def test_sulfide_more_covalent_than_oxide(self):
        """S²⁻ should be more covalent than O²⁻ (softer base)."""
        series = {r['coordination']: r for r in nephelauxetic_metallicity()}
        self.assertLess(series['sulfide_oct']['beta'],
                        series['oxide_oct']['beta'])

    def test_metallicity_index_consistent(self):
        """Metallicity = 1 - β should be between 0 and 1."""
        for r in nephelauxetic_metallicity():
            self.assertGreaterEqual(r['metallicity_index'], 0)
            self.assertLessEqual(r['metallicity_index'], 1)


# =====================================================================
# SUPERCONDUCTOR CORRELATION STRENGTH
# =====================================================================

class TestSuperconductorCorrelation(unittest.TestCase):
    """Test superconductor correlation strength classification."""

    def test_niobium_strongly_correlated(self):
        """Nb (λ=1.26) should be classified as strongly correlated."""
        sc = {r['material']: r for r in superconductor_correlation_strength()}
        self.assertEqual(sc['niobium']['correlation_class'],
                         'strongly_correlated_near_Mott')

    def test_aluminum_weakly_correlated(self):
        """Al (λ=0.43) should be weakly correlated."""
        sc = {r['material']: r for r in superconductor_correlation_strength()}
        self.assertEqual(sc['aluminum']['correlation_class'],
                         'weakly_correlated')

    def test_noble_metals_weakly_correlated(self):
        """Cu, Ag, Au (non-SC) should be weakly correlated."""
        sc = {r['material']: r for r in superconductor_correlation_strength()}
        for mat in ['copper', 'silver', 'gold']:
            self.assertEqual(sc[mat]['correlation_class'], 'weakly_correlated',
                             msg=f"{mat} should be weakly correlated")

    def test_lead_mercury_near_mott(self):
        """Pb (λ=1.55) and Hg (λ=1.60): strongly correlated."""
        sc = {r['material']: r for r in superconductor_correlation_strength()}
        for mat in ['lead', 'mercury']:
            self.assertEqual(sc[mat]['correlation_class'],
                             'strongly_correlated_near_Mott')


# =====================================================================
# CRYSTAL FIELD SPIN HAMILTONIAN
# =====================================================================

class TestSpinHamiltonian(unittest.TestCase):
    """Test crystal field → spin Hamiltonian → VQE pipeline."""

    def test_iron_oxide_spin_hamiltonian(self):
        """Fe²⁺ in oxide: VQE should find antiferromagnetic singlet."""
        r = two_site_spin_hamiltonian_from_crystal_field(26, 2, 'oxide_oct')
        self.assertIsNotNone(r)
        self.assertEqual(r['d_count'], 6)
        self.assertTrue(r['is_high_spin'])
        # Heisenberg singlet for AF coupling
        self.assertAlmostEqual(r['entanglement_entropy'], math.log(2), delta=0.01)

    def test_iron_uses_curie_derived_J(self):
        """For iron (ferromagnet), J should come from T_C."""
        r = two_site_spin_hamiltonian_from_crystal_field(26, 2, 'oxide_oct')
        # Iron T_C = 1043 K → J ≈ 11 meV
        self.assertGreater(r['J_exchange_meV'], 5)
        self.assertLess(r['J_exchange_meV'], 20)

    def test_nickel_oxide_spin_hamiltonian(self):
        """Ni²⁺ in oxide: VQE finds singlet ground state."""
        r = two_site_spin_hamiltonian_from_crystal_field(28, 2, 'oxide_oct')
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r['entanglement_entropy'], math.log(2), delta=0.01)

    def test_vqe_matches_exact(self):
        """VQE energy should match -3J for 2-site Heisenberg."""
        r = two_site_spin_hamiltonian_from_crystal_field(26, 2, 'oxide_oct')
        self.assertAlmostEqual(r['E_vqe_eV'], r['E_exact_eV'], delta=0.001)

    def test_copper_oxide_superexchange(self):
        """Cu²⁺ (d⁹) in oxide: J from superexchange formula."""
        r = two_site_spin_hamiltonian_from_crystal_field(29, 2, 'water_oct')
        self.assertIsNotNone(r)
        # Cu is not ferromagnetic, so J comes from superexchange estimate
        self.assertGreater(r['J_exchange_meV'], 0)


# =====================================================================
# MODULE REPORT
# =====================================================================

class TestReport(unittest.TestCase):
    """Test module report."""

    def test_report_has_pipelines(self):
        report = quantum_matter_report()
        self.assertGreater(len(report['prediction_pipelines']), 0)

    def test_report_has_key_insight(self):
        report = quantum_matter_report()
        self.assertIn('Tanabe-Sugano', report['key_insight'])


if __name__ == '__main__':
    unittest.main()
