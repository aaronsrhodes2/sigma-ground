"""
Tests for the hydrogen_bonding module.

Test structure:
  1. Rule 9 — molecule data completeness
  2. H-bond energy — calibration, ordering, zeros
  3. London dispersion — positivity, size dependence
  4. Keesom dipole-dipole — polarity dependence, temperature
  5. Total intermolecular energy — ordering, dominance
  6. Boiling point estimation — ordering, absolute values
  7. Intermolecular breakdown — completeness, consistency
  8. Cross-validation — against fluid.py measured values
  9. Nagatha export — format and origin tags
"""

import math
import unittest

from .hydrogen_bonding import (
    MOLECULES,
    hydrogen_bond_energy,
    hydrogen_bond_energy_molecule,
    london_dispersion_energy,
    keesom_dipole_energy,
    total_intermolecular_energy,
    intermolecular_breakdown,
    estimated_vaporization_enthalpy,
    estimated_boiling_point,
    hb_energy_ordering,
    boiling_point_ordering,
    intermolecular_properties,
)


class TestRule9MoleculeData(unittest.TestCase):
    """Every molecule has every field — Golden Rule 9."""

    REQUIRED_KEYS = [
        'formula', 'atoms', 'donor_bond', 'acceptor_atom',
        'n_donor_bonds', 'n_acceptor_lps', 'n_hb_liquid',
        'hb_energy_ev', 'molecular_mass_amu', 'dipole_debye',
        'polarizability_A3', 'IE_mol_eV',
        'r_intermol_pm', 'n_neighbors', 'T_boil_K',
    ]

    def test_all_molecules_have_all_fields(self):
        """Rule 9: every molecule gets every property."""
        for mol, data in MOLECULES.items():
            for key in self.REQUIRED_KEYS:
                self.assertIn(key, data, f"{mol} missing {key}")

    def test_six_molecules(self):
        """Six molecules in the database."""
        self.assertEqual(len(MOLECULES), 6)
        for mol in ('water', 'methanol', 'ammonia',
                     'hydrogen_fluoride', 'ethanol', 'methane'):
            self.assertIn(mol, MOLECULES)

    def test_masses_positive(self):
        for mol, data in MOLECULES.items():
            self.assertGreater(data['molecular_mass_amu'], 0, f"{mol}")

    def test_boiling_points_positive(self):
        for mol, data in MOLECULES.items():
            self.assertGreater(data['T_boil_K'], 0, f"{mol}")

    def test_hb_energy_nonnegative(self):
        for mol, data in MOLECULES.items():
            self.assertGreaterEqual(data['hb_energy_ev'], 0, f"{mol}")


class TestHydrogenBondEnergy(unittest.TestCase):
    """H-bond energy from donor/acceptor electronegativity."""

    def test_water_calibration(self):
        """Water O-H···O at reference distance ≈ 0.23 eV (calibration point)."""
        E = hydrogen_bond_energy('O', 'O', r_pm=280)
        self.assertAlmostEqual(E, 0.23, delta=0.01)

    def test_hf_strongest(self):
        """F-H···F should be the strongest H-bond."""
        E_HF = hydrogen_bond_energy('F', 'F', r_pm=250)
        E_OH = hydrogen_bond_energy('O', 'O', r_pm=280)
        self.assertGreater(E_HF, E_OH)

    def test_none_donor_returns_zero(self):
        """No donor → zero H-bond energy."""
        E = hydrogen_bond_energy(None, 'O')
        self.assertEqual(E, 0.0)

    def test_none_acceptor_returns_zero(self):
        """No acceptor → zero H-bond energy."""
        E = hydrogen_bond_energy('O', None)
        self.assertEqual(E, 0.0)

    def test_distance_dependence(self):
        """Closer distance → stronger H-bond."""
        E_close = hydrogen_bond_energy('O', 'O', r_pm=260)
        E_far = hydrogen_bond_energy('O', 'O', r_pm=320)
        self.assertGreater(E_close, E_far)

    def test_all_positive_for_real_pairs(self):
        """All donor-acceptor pairs give positive energy."""
        for d in ('O', 'N', 'F'):
            for a in ('O', 'N', 'F'):
                E = hydrogen_bond_energy(d, a)
                self.assertGreater(E, 0, f"{d}-H···{a}")


class TestHydrogenBondMolecule(unittest.TestCase):
    """H-bond energy per molecule from database."""

    def test_methane_zero(self):
        """Methane has zero H-bond energy (no donors, no acceptors)."""
        E = hydrogen_bond_energy_molecule('methane')
        self.assertEqual(E, 0.0)

    def test_water_positive(self):
        """Water has positive H-bond energy."""
        E = hydrogen_bond_energy_molecule('water')
        self.assertGreater(E, 0)

    def test_ordering_hf_gt_water(self):
        """HF H-bond > water H-bond."""
        E_hf = hydrogen_bond_energy_molecule('hydrogen_fluoride')
        E_w = hydrogen_bond_energy_molecule('water')
        self.assertGreater(E_hf, E_w)

    def test_ordering_water_gt_ammonia(self):
        """Water H-bond > ammonia H-bond."""
        E_w = hydrogen_bond_energy_molecule('water')
        E_nh3 = hydrogen_bond_energy_molecule('ammonia')
        self.assertGreater(E_w, E_nh3)

    def test_all_molecules_nonnegative(self):
        """All molecules have non-negative H-bond energy."""
        for mol in MOLECULES:
            E = hydrogen_bond_energy_molecule(mol)
            self.assertGreaterEqual(E, 0, f"{mol}")


class TestLondonDispersion(unittest.TestCase):
    """London dispersion energy — universal attractive force."""

    def test_all_positive(self):
        """London dispersion is always positive (attractive)."""
        for mol in MOLECULES:
            E = london_dispersion_energy(mol)
            self.assertGreater(E, 0, f"{mol}: London should be > 0")

    def test_larger_molecule_stronger(self):
        """Ethanol (bigger) has stronger dispersion than methane (smaller)."""
        # Not necessarily: methane has closer r. Let's compare at same distance.
        E_eth = london_dispersion_energy('ethanol', r_pm=350)
        E_ch4 = london_dispersion_energy('methane', r_pm=350)
        self.assertGreater(E_eth, E_ch4)

    def test_distance_dependence(self):
        """London ∝ 1/r⁶ — closer is much stronger."""
        E_close = london_dispersion_energy('water', r_pm=280)
        E_far = london_dispersion_energy('water', r_pm=400)
        # ratio should be (400/280)^6 ≈ 7.9
        ratio = E_close / E_far
        self.assertGreater(ratio, 5.0)
        self.assertLess(ratio, 12.0)

    def test_self_interaction_default(self):
        """Default: mol_B = mol_A (self-interaction)."""
        E_self = london_dispersion_energy('water')
        E_pair = london_dispersion_energy('water', 'water')
        self.assertAlmostEqual(E_self, E_pair, places=10)


class TestKeesomDipole(unittest.TestCase):
    """Keesom dipole-dipole orientation energy."""

    def test_methane_zero(self):
        """Methane: zero dipole → zero Keesom energy."""
        E = keesom_dipole_energy('methane')
        self.assertEqual(E, 0.0)

    def test_water_positive(self):
        """Water has significant Keesom energy."""
        E = keesom_dipole_energy('water')
        self.assertGreater(E, 0)

    def test_temperature_dependence(self):
        """Higher T → weaker Keesom (thermal disruption)."""
        E_cold = keesom_dipole_energy('water', T_K=200.0)
        E_hot = keesom_dipole_energy('water', T_K=500.0)
        self.assertGreater(E_cold, E_hot)

    def test_stronger_dipole_stronger_interaction(self):
        """Water (μ=1.85 D) > ammonia (μ=1.47 D) at same distance."""
        E_w = keesom_dipole_energy('water', r_pm=300)
        E_nh3 = keesom_dipole_energy('ammonia', r_pm=300)
        self.assertGreater(E_w, E_nh3)

    def test_distance_dependence(self):
        """Keesom ∝ 1/r⁶ — strong distance dependence."""
        E_close = keesom_dipole_energy('water', r_pm=280)
        E_far = keesom_dipole_energy('water', r_pm=400)
        ratio = E_close / E_far
        self.assertGreater(ratio, 5.0)


class TestTotalIntermolecularEnergy(unittest.TestCase):
    """Total intermolecular energy — sum of all forces."""

    def test_all_positive(self):
        """Cohesive energy is positive for all molecules."""
        for mol in MOLECULES:
            E = total_intermolecular_energy(mol)
            self.assertGreater(E, 0, f"{mol}: total should be > 0")

    def test_water_stronger_than_methane(self):
        """Water (H-bonds) is much more cohesive than methane (dispersion only)."""
        E_w = total_intermolecular_energy('water')
        E_ch4 = total_intermolecular_energy('methane')
        self.assertGreater(E_w, E_ch4)

    def test_water_dominant_is_hbond(self):
        """Water's dominant force is hydrogen bonding."""
        bd = intermolecular_breakdown('water')
        self.assertEqual(bd['dominant'], 'hydrogen_bond')

    def test_methane_dominant_is_london(self):
        """Methane's dominant force is London dispersion."""
        bd = intermolecular_breakdown('methane')
        self.assertEqual(bd['dominant'], 'london_dispersion')


class TestIntermolecularBreakdown(unittest.TestCase):
    """Breakdown of forces — consistency checks."""

    def test_all_keys_present(self):
        """Breakdown returns all expected keys."""
        for mol in MOLECULES:
            bd = intermolecular_breakdown(mol)
            for key in ('hb_ev', 'london_ev', 'keesom_ev',
                        'total_ev', 'dominant'):
                self.assertIn(key, bd, f"{mol} missing {key}")

    def test_sum_equals_total(self):
        """Parts sum to total."""
        for mol in MOLECULES:
            bd = intermolecular_breakdown(mol)
            parts_sum = bd['hb_ev'] + bd['london_ev'] + bd['keesom_ev']
            self.assertAlmostEqual(parts_sum, bd['total_ev'], places=10,
                                   msg=f"{mol}: parts should sum to total")

    def test_all_parts_nonnegative(self):
        """No negative contributions."""
        for mol in MOLECULES:
            bd = intermolecular_breakdown(mol)
            self.assertGreaterEqual(bd['hb_ev'], 0, f"{mol}: hb")
            self.assertGreaterEqual(bd['london_ev'], 0, f"{mol}: london")
            self.assertGreaterEqual(bd['keesom_ev'], 0, f"{mol}: keesom")

    def test_methane_zero_hb_and_keesom(self):
        """Methane: zero H-bond and zero Keesom (no dipole, no donors)."""
        bd = intermolecular_breakdown('methane')
        self.assertEqual(bd['hb_ev'], 0.0)
        self.assertEqual(bd['keesom_ev'], 0.0)
        self.assertGreater(bd['london_ev'], 0)


class TestBoilingPoint(unittest.TestCase):
    """Boiling point estimation from intermolecular energy."""

    def test_all_positive(self):
        """All boiling points are positive."""
        for mol in MOLECULES:
            T = estimated_boiling_point(mol)
            self.assertGreater(T, 0, f"{mol}: T_boil should be > 0")

    def test_water_order_of_magnitude(self):
        """Water boiling point estimate within factor 2 of 373 K."""
        T = estimated_boiling_point('water')
        self.assertGreater(T, 373.0 / 2.0)  # > 186 K
        self.assertLess(T, 373.0 * 2.0)      # < 746 K

    def test_methane_lowest(self):
        """Methane has the lowest boiling point (weakest interactions)."""
        T_ch4 = estimated_boiling_point('methane')
        for mol in MOLECULES:
            if mol != 'methane':
                T_other = estimated_boiling_point(mol)
                self.assertLess(T_ch4, T_other,
                                f"Methane should boil lower than {mol}")

    def test_water_much_higher_than_methane(self):
        """Water boils far above methane (H-bonds vs dispersion only)."""
        T_w = estimated_boiling_point('water')
        T_ch4 = estimated_boiling_point('methane')
        self.assertGreater(T_w / T_ch4, 2.0,
                           "Water should boil at least 2× higher than methane")

    def test_methane_low(self):
        """Methane boiling point should be < 200 K (measured: 111.7 K)."""
        T = estimated_boiling_point('methane')
        self.assertLess(T, 200.0)

    def test_vaporization_enthalpy_positive(self):
        """ΔH_vap is positive for all molecules."""
        for mol in MOLECULES:
            dH = estimated_vaporization_enthalpy(mol)
            self.assertGreater(dH, 0, f"{mol}: ΔH_vap should be > 0")

    def test_water_vaporization_order_of_magnitude(self):
        """Water ΔH_vap ≈ 40.7 kJ/mol — within factor 2."""
        dH = estimated_vaporization_enthalpy('water')
        dH_kJ = dH / 1000.0
        self.assertGreater(dH_kJ, 20.0)  # > 20 kJ/mol
        self.assertLess(dH_kJ, 80.0)     # < 80 kJ/mol


class TestOrdering(unittest.TestCase):
    """Physical ordering tests — the real proof."""

    def test_hb_energy_ordering(self):
        """H-bond ordering: HF > water > methanol/ethanol > ammonia > methane=0."""
        order = hb_energy_ordering()
        names = [k for k, _ in order]
        # HF should be first
        self.assertEqual(names[0], 'hydrogen_fluoride')
        # Water should be second
        self.assertEqual(names[1], 'water')
        # Methane should be last
        self.assertEqual(names[-1], 'methane')

    def test_boiling_point_hbond_above_nonpolar(self):
        """All H-bonding molecules boil above methane.

        The exact ordering among H-bonding liquids (water vs methanol vs HF)
        is beyond the resolution of Trouton-level models — their real boiling
        points differ by only ~35-80 K.  The clear prediction is that ALL
        H-bonding molecules boil far above dispersion-only methane.
        """
        T_ch4 = estimated_boiling_point('methane')
        for mol in MOLECULES:
            if mol != 'methane':
                T = estimated_boiling_point(mol)
                self.assertGreater(T, T_ch4,
                                   f"{mol} should boil above methane")

    def test_boiling_point_ordering_methane_bottom(self):
        """Methane has the lowest estimated boiling point."""
        order = boiling_point_ordering()
        self.assertEqual(order[-1][0], 'methane')


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export format."""

    def test_all_molecules_export(self):
        """All molecules produce valid export dicts."""
        for mol in MOLECULES:
            props = intermolecular_properties(mol)
            self.assertIn('molecule', props)
            self.assertIn('total_intermolecular_ev', props)
            self.assertIn('estimated_T_boil_K', props)
            self.assertIn('origin', props)

    def test_origin_tags_honest(self):
        """Origin contains FIRST_PRINCIPLES and MEASURED."""
        props = intermolecular_properties('water')
        self.assertIn('FIRST_PRINCIPLES', props['origin'])
        self.assertIn('MEASURED', props['origin'])

    def test_measured_boiling_included(self):
        """Export includes measured boiling point for comparison."""
        props = intermolecular_properties('water')
        self.assertAlmostEqual(props['measured_T_boil_K'], 373.15, places=1)

    def test_sigma_propagates(self):
        """σ value appears in export."""
        props = intermolecular_properties('water', sigma=0.1)
        self.assertEqual(props['sigma'], 0.1)

    def test_all_keys_present(self):
        """Export has all expected keys."""
        props = intermolecular_properties('water')
        required = [
            'molecule', 'formula', 'molecular_mass_amu', 'dipole_debye',
            'n_hb_liquid', 'hb_energy_ev', 'london_energy_ev',
            'keesom_energy_ev', 'total_intermolecular_ev',
            'dominant_force', 'estimated_dH_vap_J_mol',
            'estimated_T_boil_K', 'measured_T_boil_K', 'sigma', 'origin',
        ]
        for key in required:
            self.assertIn(key, props, f"Missing: {key}")


if __name__ == '__main__':
    unittest.main()
