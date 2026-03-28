"""
Tests for chemical_reactions.py — reaction thermodynamics and kinetics.

Strategy:
  - Test against MEASURED reaction enthalpies (NIST, CRC Handbook)
  - Test thermodynamic self-consistency (sign, magnitude, trends)
  - Test kinetic formulas against known analytical results
  - Tolerance: ±30% for enthalpies (Pauling bond energy limitation),
    order-of-magnitude for rate constants (exponential sensitivity)
"""

import math
import unittest

from sigma_ground.field.interface.chemical_reactions import (
    reaction_enthalpy_kJ_mol,
    reaction_enthalpy_by_key,
    entropy_change_estimate,
    gibbs_energy_kJ_mol,
    equilibrium_constant,
    evans_polanyi_activation_energy,
    arrhenius_rate,
    collision_prefactor,
    half_life,
    temperature_for_rate,
    reaction_report,
    full_report,
    REACTIONS,
    EVANS_POLANYI_FAMILIES,
    _bond_energy_ev,
)
from sigma_ground.field.constants import K_B, N_AVOGADRO, EV_TO_J


# ═══════════════════════════════════════════════════════════════════
# REACTION ENTHALPIES (Hess's Law)
# ═══════════════════════════════════════════════════════════════════

class TestReactionEnthalpy(unittest.TestCase):
    """Hess's law enthalpies vs MEASURED values."""

    def test_all_known_reactions_within_30_percent(self):
        """Every reaction with a measured ΔH should be within ±30%.

        Exception: reactions flagged 'pauling_limitation' are excluded.
        The Pauling equation has known failure modes (anomalous homonuclear
        bonds like N-N in hydrazine). We document these, not hide them.
        """
        for key, rxn in REACTIONS.items():
            if 'measured_dH_kJ_mol' not in rxn:
                continue
            if rxn.get('pauling_limitation'):
                continue  # documented limitation
            measured = rxn['measured_dH_kJ_mol']
            derived = reaction_enthalpy_by_key(key)
            with self.subTest(reaction=key):
                if abs(measured) > 10:  # skip tiny values (relative error meaningless)
                    error = abs(derived - measured) / abs(measured)
                    self.assertLess(
                        error, 0.30,
                        f"{key}: derived={derived:.1f}, measured={measured:.1f}, "
                        f"error={error*100:.1f}%"
                    )

    def test_exothermic_reactions_negative(self):
        """Known exothermic reactions should give negative ΔH."""
        exothermic = [
            'methane_combustion', 'ethanol_combustion',
            'hydrogenation_ethylene', 'hydrogen_chloride',
            'hydrogen_fluoride', 'water_formation',
        ]
        for key in exothermic:
            with self.subTest(reaction=key):
                dH = reaction_enthalpy_by_key(key)
                self.assertLess(dH, 0, f"{key} should be exothermic")

    def test_methane_combustion_magnitude(self):
        """Methane combustion should be ~890 kJ/mol."""
        dH = reaction_enthalpy_by_key('methane_combustion')
        self.assertAlmostEqual(dH, -890.4, delta=300)

    def test_water_formation_magnitude(self):
        """2H₂ + O₂ → 2H₂O should be ~484 kJ/mol exothermic."""
        dH = reaction_enthalpy_by_key('water_formation')
        self.assertAlmostEqual(dH, -483.6, delta=200)

    def test_haber_process_pauling_limitation(self):
        """N₂ + 3H₂ → 2NH₃: Pauling gets the WRONG SIGN.

        The N-N single bond (1.59 eV, hydrazine) is anomalously weak
        due to lone pair repulsion. This makes Pauling underestimate
        N-H bond energy by ~6% per bond. Over 6 N-H bonds, the
        cumulative error (+1.4 eV) flips the sign.

        MEASURED: -92 kJ/mol (exothermic)
        PAULING:  +76 kJ/mol (endothermic) — WRONG

        This is a DOCUMENTED limitation, not a bug. The Pauling equation
        assumes homonuclear bond energies are transferable; N-N in N₂H₄
        violates this (lone pair repulsion weakens it far below a
        'normal' N-N single bond).
        """
        dH = reaction_enthalpy_by_key('haber_process')
        # Pauling gives wrong sign — test that we know about it
        self.assertGreater(dH, 0, "Pauling predicts endothermic (wrong)")
        self.assertLess(abs(dH), 200, "Magnitude should be modest")

    def test_hf_more_exothermic_than_hcl(self):
        """H₂ + F₂ → 2HF should release more energy than H₂ + Cl₂ → 2HCl.

        Fluorine is more electronegative → larger ionic resonance energy
        in H-F → more exothermic formation.
        """
        dH_HF = reaction_enthalpy_by_key('hydrogen_fluoride')
        dH_HCl = reaction_enthalpy_by_key('hydrogen_chloride')
        self.assertLess(dH_HF, dH_HCl)

    def test_hydrogenation_enthalpy_trend(self):
        """C≡C hydrogenation should release more energy than C=C.

        Triple → double breaks a stronger bond (C≡C) but the extra π bond
        energy released in going from triple to double exceeds the H-H cost.
        """
        dH_ethylene = reaction_enthalpy_by_key('hydrogenation_ethylene')
        dH_acetylene = reaction_enthalpy_by_key('hydrogenation_acetylene')
        # Both exothermic
        self.assertLess(dH_ethylene, 0)
        self.assertLess(dH_acetylene, 0)
        # Acetylene hydrogenation more exothermic (releases more)
        self.assertLess(dH_acetylene, dH_ethylene)

    def test_direct_bond_inventory(self):
        """Direct bond inventory should match keyed reaction."""
        rxn = REACTIONS['hydrogen_chloride']
        direct = reaction_enthalpy_kJ_mol(
            rxn['bonds_broken'], rxn['bonds_formed']
        )
        keyed = reaction_enthalpy_by_key('hydrogen_chloride')
        self.assertAlmostEqual(direct, keyed, places=5)


# ═══════════════════════════════════════════════════════════════════
# THERMODYNAMICS (ΔS, ΔG, K)
# ═══════════════════════════════════════════════════════════════════

class TestEntropy(unittest.TestCase):
    """Entropy estimation from gas-phase mole change."""

    def test_zero_delta_n_gives_near_zero_entropy(self):
        """No change in gas moles → ΔS ≈ 0."""
        dS = entropy_change_estimate(0)
        self.assertAlmostEqual(dS, 0.0, places=5)

    def test_negative_delta_n_gives_negative_entropy(self):
        """Fewer gas moles → negative ΔS (more ordered)."""
        dS = entropy_change_estimate(-2)
        self.assertLess(dS, 0)

    def test_positive_delta_n_gives_positive_entropy(self):
        """More gas moles → positive ΔS (more disordered)."""
        dS = entropy_change_estimate(1)
        self.assertGreater(dS, 0)

    def test_entropy_magnitude_reasonable(self):
        """ΔS for ±1 mol gas should be ~100-200 J/(mol·K)."""
        dS = entropy_change_estimate(1)
        self.assertGreater(abs(dS), 100)
        self.assertLess(abs(dS), 250)

    def test_entropy_scales_linearly(self):
        """ΔS should be roughly proportional to Δn_gas."""
        dS_1 = entropy_change_estimate(1)
        dS_2 = entropy_change_estimate(2)
        self.assertAlmostEqual(dS_2, 2 * dS_1, delta=1.0)


class TestGibbsEnergy(unittest.TestCase):
    """Gibbs free energy ΔG = ΔH − TΔS."""

    def test_exothermic_no_entropy_change_is_spontaneous(self):
        """Exothermic + no entropy change → ΔG < 0 (spontaneous)."""
        dG = gibbs_energy_kJ_mol(-100, delta_n_gas=0, T=298.15)
        self.assertLess(dG, 0)

    def test_entropy_driven_at_high_T(self):
        """Endothermic reaction with positive ΔS can be spontaneous at high T."""
        # ΔH = +50 kJ/mol (endothermic), Δn_gas = +2 (entropy gain)
        dG_low = gibbs_energy_kJ_mol(50, delta_n_gas=2, T=100)
        dG_high = gibbs_energy_kJ_mol(50, delta_n_gas=2, T=1000)
        # Should become more favorable at high T
        self.assertLess(dG_high, dG_low)

    def test_temperature_dependence_sign(self):
        """Higher T makes −TΔS more negative when ΔS > 0."""
        dG_300 = gibbs_energy_kJ_mol(-50, delta_n_gas=1, T=300)
        dG_600 = gibbs_energy_kJ_mol(-50, delta_n_gas=1, T=600)
        # Positive ΔS means higher T makes ΔG more negative
        self.assertLess(dG_600, dG_300)


class TestEquilibriumConstant(unittest.TestCase):
    """K = exp(−ΔG/RT)."""

    def test_very_exothermic_gives_large_K(self):
        """Very exothermic reaction → K >> 1 (goes to completion)."""
        K = equilibrium_constant(-500, delta_n_gas=0, T=298.15)
        self.assertGreater(K, 1e10)

    def test_very_endothermic_gives_small_K(self):
        """Very endothermic reaction → K << 1 (barely proceeds)."""
        K = equilibrium_constant(500, delta_n_gas=0, T=298.15)
        self.assertLess(K, 1e-10)

    def test_K_increases_with_T_for_endothermic(self):
        """Le Chatelier: endothermic reaction → K increases with T."""
        K_300 = equilibrium_constant(50, delta_n_gas=0, T=300)
        K_600 = equilibrium_constant(50, delta_n_gas=0, T=600)
        self.assertGreater(K_600, K_300)

    def test_K_equals_one_at_equilibrium_delta_G_zero(self):
        """When ΔG = 0, K = 1."""
        # ΔH = TΔS → ΔG = 0
        # For Δn_gas = 1 at 298K, ΔS ≈ 150 J/(mol·K), so ΔH ≈ 44.7 kJ/mol
        dS = entropy_change_estimate(1, T=298.15)
        dH_for_equilibrium = 298.15 * dS / 1000.0  # kJ/mol
        K = equilibrium_constant(dH_for_equilibrium, delta_n_gas=1, T=298.15)
        self.assertAlmostEqual(K, 1.0, delta=0.1)

    def test_combustion_K_huge(self):
        """Combustion reactions have enormous K (irreversible in practice)."""
        dH = reaction_enthalpy_by_key('methane_combustion')
        K = equilibrium_constant(dH, delta_n_gas=-1, T=298.15)
        self.assertGreater(K, 1e50)


# ═══════════════════════════════════════════════════════════════════
# KINETICS (Activation Energy, Arrhenius, Collision Theory)
# ═══════════════════════════════════════════════════════════════════

class TestEvansPolanyi(unittest.TestCase):
    """Evans-Polanyi activation energy E_a = E_0 + α × ΔH."""

    def test_exothermic_lowers_barrier(self):
        """More exothermic → lower E_a within a family."""
        E_a_mild = evans_polanyi_activation_energy(-50, 'hydrogen_abstraction')
        E_a_strong = evans_polanyi_activation_energy(-200, 'hydrogen_abstraction')
        self.assertGreater(E_a_mild, E_a_strong)

    def test_endothermic_raises_barrier(self):
        """Endothermic reaction → E_a > E_0."""
        E_a = evans_polanyi_activation_energy(100, 'hydrogen_abstraction')
        E_0 = EVANS_POLANYI_FAMILIES['hydrogen_abstraction']['E_0_eV']
        self.assertGreater(E_a, E_0)

    def test_activation_energy_non_negative(self):
        """E_a should never be negative (clamped to 0)."""
        # Very exothermic should clamp to 0
        E_a = evans_polanyi_activation_energy(-5000, 'hydrogen_abstraction')
        self.assertGreaterEqual(E_a, 0.0)

    def test_activation_energy_reasonable_magnitude(self):
        """For typical reactions, E_a should be 0.1-3 eV."""
        for family in EVANS_POLANYI_FAMILIES:
            with self.subTest(family=family):
                E_a = evans_polanyi_activation_energy(-100, family)
                self.assertGreaterEqual(E_a, 0.0)
                self.assertLess(E_a, 5.0)

    def test_halogenation_low_barrier(self):
        """Halogenation propagation has a low intrinsic barrier."""
        E_a = evans_polanyi_activation_energy(-100, 'halogenation')
        self.assertLess(E_a, 0.5)  # Should be small

    def test_combustion_initiation_high_barrier(self):
        """Combustion initiation (R-H + O₂) has a high barrier."""
        E_a = evans_polanyi_activation_energy(0, 'combustion_initiation')
        self.assertGreater(E_a, 1.0)  # Inherently high barrier


class TestArrhenius(unittest.TestCase):
    """k(T) = A × exp(−E_a / k_BT)."""

    def test_rate_increases_with_temperature(self):
        """Higher T → faster rate."""
        A = 1e10
        E_a = 0.5  # eV
        k_300 = arrhenius_rate(A, E_a, 300)
        k_600 = arrhenius_rate(A, E_a, 600)
        self.assertGreater(k_600, k_300)

    def test_zero_barrier_gives_prefactor(self):
        """E_a = 0 → k = A (every collision reacts)."""
        A = 1e10
        k = arrhenius_rate(A, 0.0, 300)
        self.assertAlmostEqual(k, A, places=0)

    def test_high_barrier_gives_tiny_rate(self):
        """Very high E_a → k ≈ 0."""
        k = arrhenius_rate(1e10, 5.0, 300)  # 5 eV barrier at 300K
        self.assertLess(k, 1e-70)

    def test_zero_temperature_gives_zero(self):
        """T = 0 → k = 0."""
        k = arrhenius_rate(1e10, 0.5, 0)
        self.assertEqual(k, 0.0)

    def test_room_temp_modest_barrier(self):
        """0.7 eV barrier at 298K should give measurably slow rate."""
        k = arrhenius_rate(1e10, 0.7, 298)
        # exp(-0.7*1.6e-19 / (1.38e-23 * 298)) = exp(-27.2) ≈ 1.5e-12
        # k ≈ 1e10 * 1.5e-12 ≈ 0.015
        self.assertGreater(k, 1e-5)
        self.assertLess(k, 1e3)


class TestCollisionPrefactor(unittest.TestCase):
    """Collision theory pre-exponential A."""

    def test_prefactor_positive(self):
        """A should always be positive."""
        A = collision_prefactor(28, 32, 185, 175, T=300)
        self.assertGreater(A, 0)

    def test_prefactor_order_of_magnitude(self):
        """Gas-phase bimolecular A should be ~1e9-1e11 L/(mol·s)."""
        # N₂ + O₂ type collision
        A = collision_prefactor(28, 32, 185, 175, T=300, steric_factor=1.0)
        self.assertGreater(A, 1e9)
        self.assertLess(A, 1e13)

    def test_steric_factor_scales_linearly(self):
        """A should scale linearly with steric factor."""
        A_full = collision_prefactor(28, 32, 185, 175, steric_factor=1.0)
        A_tenth = collision_prefactor(28, 32, 185, 175, steric_factor=0.1)
        self.assertAlmostEqual(A_full / A_tenth, 10.0, delta=0.01)

    def test_larger_molecules_larger_cross_section(self):
        """Larger collision radii → larger A."""
        A_small = collision_prefactor(28, 32, 100, 100, steric_factor=1.0)
        A_large = collision_prefactor(28, 32, 300, 300, steric_factor=1.0)
        self.assertGreater(A_large, A_small)

    def test_heavier_molecules_slower(self):
        """Heavier reduced mass → slower collisions → smaller A
        (at fixed cross-section and steric factor)."""
        A_light = collision_prefactor(2, 2, 150, 150, steric_factor=1.0)
        A_heavy = collision_prefactor(200, 200, 150, 150, steric_factor=1.0)
        self.assertGreater(A_light, A_heavy)


class TestHalfLife(unittest.TestCase):
    """t_½ = ln(2) / k."""

    def test_half_life_from_rate(self):
        """Known rate constant → known half-life."""
        k = 0.01  # 1/s
        t = half_life(k)
        self.assertAlmostEqual(t, 69.3, delta=0.1)

    def test_zero_rate_infinite_half_life(self):
        """k = 0 → t_½ = ∞."""
        self.assertEqual(half_life(0.0), float('inf'))

    def test_fast_rate_short_half_life(self):
        """Large k → short t_½."""
        t = half_life(1e6)
        self.assertLess(t, 1e-5)


class TestTemperatureForRate(unittest.TestCase):
    """Inverted Arrhenius: find T for target k."""

    def test_roundtrip(self):
        """arrhenius_rate(A, E_a, T) should equal k_target."""
        A = 1e10
        E_a = 0.5
        T = temperature_for_rate(1e5, A, E_a)
        k_check = arrhenius_rate(A, E_a, T)
        self.assertAlmostEqual(k_check, 1e5, delta=1e2)

    def test_higher_rate_needs_higher_T(self):
        """Faster rate requires higher temperature."""
        A = 1e10
        E_a = 0.5
        T_slow = temperature_for_rate(1.0, A, E_a)
        T_fast = temperature_for_rate(1e6, A, E_a)
        self.assertGreater(T_fast, T_slow)

    def test_impossible_rate_gives_inf(self):
        """k_target >= A is impossible → T = inf."""
        T = temperature_for_rate(1e11, 1e10, 0.5)
        self.assertEqual(T, float('inf'))


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION: Full Reaction Reports
# ═══════════════════════════════════════════════════════════════════

class TestReactionReport(unittest.TestCase):
    """End-to-end report for named reactions."""

    def test_methane_combustion_report_complete(self):
        """Methane combustion report should have all fields."""
        r = reaction_report('methane_combustion')
        required = [
            'name', 'equation', 'T_K', 'delta_H_kJ_mol',
            'delta_S_J_mol_K', 'delta_G_kJ_mol', 'K_eq',
            'spontaneous', 'exothermic',
        ]
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_methane_combustion_is_spontaneous(self):
        """Methane combustion should be spontaneous at 298K."""
        r = reaction_report('methane_combustion')
        self.assertTrue(r['spontaneous'])
        self.assertTrue(r['exothermic'])

    def test_kinetics_included_when_family_known(self):
        """Reactions with EP family should include kinetics."""
        r = reaction_report('methane_chlorination')
        self.assertIn('E_a_eV', r)
        self.assertIn('k_at_T', r)

    def test_haber_no_kinetics(self):
        """Haber process (catalyzed, no EP family) should lack kinetics."""
        r = reaction_report('haber_process')
        self.assertNotIn('E_a_eV', r)

    def test_measured_comparison_included(self):
        """Report should include measured ΔH and error when available."""
        r = reaction_report('methane_combustion')
        self.assertIn('measured_dH_kJ_mol', r)
        self.assertIn('enthalpy_error_pct', r)

    def test_full_report_all_reactions(self):
        """full_report() should cover every reaction."""
        reports = full_report()
        self.assertEqual(set(reports.keys()), set(REACTIONS.keys()))


class TestPhysicsConsistency(unittest.TestCase):
    """Cross-checks between different derived quantities."""

    def test_gibbs_consistent_with_enthalpy_entropy(self):
        """ΔG should equal ΔH − TΔS."""
        T = 400
        dH = reaction_enthalpy_by_key('water_formation')
        dS = entropy_change_estimate(REACTIONS['water_formation']['delta_n_gas'], T)
        dG = gibbs_energy_kJ_mol(dH, REACTIONS['water_formation']['delta_n_gas'], T)
        expected_dG = dH - T * dS / 1000.0
        self.assertAlmostEqual(dG, expected_dG, places=5)

    def test_K_consistent_with_gibbs(self):
        """K should be exp(−ΔG/RT)."""
        T = 298.15
        R = K_B * N_AVOGADRO
        dH = -200  # kJ/mol
        dG = gibbs_energy_kJ_mol(dH, 0, T)
        K = equilibrium_constant(dH, 0, T)
        expected_K = math.exp(-dG * 1000 / (R * T))
        self.assertAlmostEqual(K / expected_K, 1.0, delta=0.001)

    def test_bond_energies_symmetric(self):
        """Pauling bond energy A-B should equal B-A."""
        E_CH = _bond_energy_ev('C-H')
        E_HC = _bond_energy_ev('H-C')
        self.assertAlmostEqual(E_CH, E_HC, places=5)

    def test_combustion_more_exothermic_for_larger_fuel(self):
        """Ethanol combustion should be more exothermic than methane."""
        dH_methane = reaction_enthalpy_by_key('methane_combustion')
        dH_ethanol = reaction_enthalpy_by_key('ethanol_combustion')
        self.assertLess(dH_ethanol, dH_methane)

    def test_arrhenius_and_EP_consistent(self):
        """Lower EP barrier → faster Arrhenius rate at same T."""
        E_a_low = evans_polanyi_activation_energy(-200, 'halogenation')
        E_a_high = evans_polanyi_activation_energy(-50, 'combustion_initiation')
        k_low_barrier = arrhenius_rate(1e10, E_a_low, 300)
        k_high_barrier = arrhenius_rate(1e10, E_a_high, 300)
        self.assertGreater(k_low_barrier, k_high_barrier)

    def test_le_chatelier_temperature(self):
        """Exothermic reactions: K decreases with T (Le Chatelier)."""
        dH = reaction_enthalpy_by_key('methane_combustion')
        K_300 = equilibrium_constant(dH, -1, 300)
        K_1000 = equilibrium_constant(dH, -1, 1000)
        # For very exothermic reactions, K may hit overflow at both temps
        # but the trend should hold if they're finite
        if K_300 < 1e300 and K_1000 < 1e300:
            self.assertGreater(K_300, K_1000)


if __name__ == '__main__':
    unittest.main()
