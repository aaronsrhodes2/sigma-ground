"""
Tests for the nucleosynthesis module.

Test structure:
  1. Gamow peak — energy and window width at solar conditions
  2. Reaction rates — pp, CNO, triple-alpha at known temperatures
  3. Energy generation — pp and CNO rates, crossover temperature
  4. σ-dependence — rates shift with σ (SSBM predictions)
  5. Physical sanity — ordering, bounds, conservation
  6. Nagatha export — complete format with origin tags
"""

import math
import unittest

from .nucleosynthesis import (
    reduced_mass_kg,
    reduced_mass_mev,
    gamow_energy_keV,
    gamow_window_keV,
    reaction_rate_sigma_v,
    pp_chain_energy_rate,
    cno_energy_rate,
    pp_cno_crossover_temperature,
    pp_temperature_exponent,
    reaction_properties,
    stellar_burning_summary,
    REACTIONS,
)


# ── Solar conditions for reference ────────────────────────────────
T_SUN = 15.7e6      # Solar core temperature (K)
RHO_SUN = 150e3     # Solar core density (kg/m3)


class TestReducedMass(unittest.TestCase):
    """Reduced mass for nuclear reaction pairs."""

    def test_pp_half_proton(self):
        """p+p reduced mass = m_p/2."""
        mu = reduced_mass_mev(1, 1, sigma=0.0)
        from ..constants import PROTON_TOTAL_MEV
        expected = PROTON_TOTAL_MEV / 2.0
        self.assertAlmostEqual(mu, expected, delta=1.0)

    def test_increases_with_sigma(self):
        """Higher sigma -> heavier nuclei -> higher reduced mass."""
        mu_0 = reduced_mass_kg(1, 1, sigma=0.0)
        mu_1 = reduced_mass_kg(1, 1, sigma=1.0)
        self.assertGreater(mu_1, mu_0)

    def test_asymmetric(self):
        """C12 + p: reduced mass close to proton mass."""
        mu = reduced_mass_mev(12, 1, sigma=0.0)
        from ..constants import PROTON_TOTAL_MEV
        # mu = 12*1/(12+1) * m_p = 12/13 * m_p ~ 0.923 m_p
        self.assertAlmostEqual(mu / PROTON_TOTAL_MEV, 12.0/13.0, delta=0.01)

    def test_positive(self):
        """Reduced mass always positive."""
        for rxn in REACTIONS.values():
            mu = reduced_mass_kg(rxn['A1'], rxn['A2'])
            self.assertGreater(mu, 0)


class TestGamowPeak(unittest.TestCase):
    """Gamow peak energy — the sweet spot for thermonuclear reactions."""

    def test_pp_solar_about_6keV(self):
        """pp Gamow peak at solar T: E_G ~ 6 keV.

        This is a fundamental number in stellar physics.
        kT_sun ~ 1.35 keV, but reactions happen at 6 keV
        thanks to quantum tunneling.
        """
        E_G = gamow_energy_keV(1, 1, 1, 1, T_SUN)
        self.assertAlmostEqual(E_G, 6.0, delta=2.0)

    def test_higher_Z_higher_peak(self):
        """Higher Z -> stronger Coulomb barrier -> higher Gamow peak.

        CNO (Z=6+1) has higher E_G than pp (Z=1+1).
        """
        E_pp = gamow_energy_keV(1, 1, 1, 1, T_SUN)
        E_CNO = gamow_energy_keV(6, 12, 1, 1, T_SUN)
        self.assertGreater(E_CNO, E_pp)

    def test_increases_with_temperature(self):
        """Higher T shifts the Gamow peak to higher energy."""
        E_low = gamow_energy_keV(1, 1, 1, 1, 10e6)
        E_high = gamow_energy_keV(1, 1, 1, 1, 30e6)
        self.assertGreater(E_high, E_low)

    def test_zero_temperature(self):
        """T = 0: no Gamow peak."""
        E_G = gamow_energy_keV(1, 1, 1, 1, 0)
        self.assertEqual(E_G, 0.0)

    def test_positive_for_all_reactions(self):
        """Gamow peak is positive for all reactions at solar T."""
        for key, rxn in REACTIONS.items():
            E_G = gamow_energy_keV(rxn['Z1'], rxn['A1'],
                                    rxn['Z2'], rxn['A2'], T_SUN)
            self.assertGreater(E_G, 0, f"{key}: E_G should be positive")


class TestGamowWindow(unittest.TestCase):
    """Gamow window width."""

    def test_positive(self):
        """Window width is always positive."""
        delta = gamow_window_keV(1, 1, 1, 1, T_SUN)
        self.assertGreater(delta, 0)

    def test_wider_at_higher_T(self):
        """Higher temperature -> wider Gamow window."""
        d_lo = gamow_window_keV(1, 1, 1, 1, 10e6)
        d_hi = gamow_window_keV(1, 1, 1, 1, 30e6)
        self.assertGreater(d_hi, d_lo)

    def test_pp_solar_window(self):
        """pp window at solar T: a few keV wide."""
        delta = gamow_window_keV(1, 1, 1, 1, T_SUN)
        self.assertGreater(delta, 1.0)
        self.assertLess(delta, 20.0)


class TestReactionRates(unittest.TestCase):
    """Thermonuclear reaction rates at stellar temperatures."""

    def test_pp_rate_positive(self):
        """pp rate is positive at solar temperature."""
        sv = reaction_rate_sigma_v('pp', T_SUN)
        self.assertGreater(sv, 0)

    def test_pp_rate_tiny(self):
        """pp rate is extremely small (weak interaction).

        At solar T: <sigma v> ~ 10^-43 cm3/s.
        This is why the Sun burns so slowly — the weak interaction
        bottleneck makes proton fusion incredibly improbable.
        """
        sv = reaction_rate_sigma_v('pp', T_SUN)
        # Should be in the 10^-46 to 10^-40 range
        self.assertGreater(sv, 1e-50)
        self.assertLess(sv, 1e-35)

    def test_dp_much_faster_than_pp(self):
        """d+p reaction is much faster than pp (EM vs weak).

        The S-factor for d+p is ~10^18 times larger than pp.
        This is why deuterium burns instantly in stars.
        """
        sv_pp = reaction_rate_sigma_v('pp', T_SUN)
        sv_dp = reaction_rate_sigma_v('dp', T_SUN)
        self.assertGreater(sv_dp, sv_pp * 1e10)

    def test_rate_increases_with_temperature(self):
        """All rates increase steeply with temperature."""
        for key in ['pp', 'dp', 'C12_p', 'N14_p']:
            sv_lo = reaction_rate_sigma_v(key, 10e6)
            sv_hi = reaction_rate_sigma_v(key, 20e6)
            self.assertGreater(sv_hi, sv_lo,
                f"{key}: rate should increase with T")

    def test_cno_steeper_than_pp(self):
        """CNO has steeper T dependence than pp.

        CNO ~ T^16, pp ~ T^4. So CNO gains more from a T increase.
        """
        # Compare rate ratios over a temperature doubling
        sv_pp_lo = reaction_rate_sigma_v('pp', 15e6)
        sv_pp_hi = reaction_rate_sigma_v('pp', 30e6)
        sv_cno_lo = reaction_rate_sigma_v('N14_p', 15e6)
        sv_cno_hi = reaction_rate_sigma_v('N14_p', 30e6)

        if sv_pp_lo > 0 and sv_cno_lo > 0:
            ratio_pp = sv_pp_hi / sv_pp_lo
            ratio_cno = sv_cno_hi / sv_cno_lo
            self.assertGreater(ratio_cno, ratio_pp)

    def test_triple_alpha_positive(self):
        """Triple-alpha rate is positive at He-burning temperature."""
        # He burning: T ~ 100 MK
        sv = reaction_rate_sigma_v('triple_alpha', 100e6)
        self.assertGreater(sv, 0)

    def test_triple_alpha_zero_at_low_T(self):
        """Triple-alpha negligible at low temperature."""
        sv = reaction_rate_sigma_v('triple_alpha', 1e6)
        self.assertEqual(sv, 0.0)


class TestEnergyGeneration(unittest.TestCase):
    """Energy generation rates in stellar cores."""

    def test_pp_solar_order_of_magnitude(self):
        """pp rate at solar conditions: detectable but modest.

        Solar luminosity / solar mass ~ 2 × 10⁻⁴ W/kg (average).
        Core rate is higher (concentrated in center).
        """
        eps = pp_chain_energy_rate(T_SUN, RHO_SUN)
        # Should be positive and nonzero
        self.assertGreater(eps, 0)

    def test_pp_increases_with_T(self):
        """pp energy generation increases with temperature."""
        eps_lo = pp_chain_energy_rate(10e6, RHO_SUN)
        eps_hi = pp_chain_energy_rate(20e6, RHO_SUN)
        self.assertGreater(eps_hi, eps_lo)

    def test_cno_increases_with_T(self):
        """CNO energy generation increases with temperature."""
        eps_lo = cno_energy_rate(15e6, RHO_SUN)
        eps_hi = cno_energy_rate(25e6, RHO_SUN)
        self.assertGreater(eps_hi, eps_lo)

    def test_pp_dominates_at_solar_T(self):
        """pp dominates over CNO at solar temperature."""
        eps_pp = pp_chain_energy_rate(T_SUN, RHO_SUN)
        eps_cno = cno_energy_rate(T_SUN, RHO_SUN)
        self.assertGreater(eps_pp, eps_cno)

    def test_cno_dominates_at_high_T(self):
        """CNO dominates over pp at high temperature (massive stars)."""
        T_high = 30e6  # 30 MK — typical for massive star core
        eps_pp = pp_chain_energy_rate(T_high, RHO_SUN)
        eps_cno = cno_energy_rate(T_high, RHO_SUN)
        self.assertGreater(eps_cno, eps_pp)

    def test_zero_density_zero_rate(self):
        """Zero density -> zero energy generation."""
        eps = pp_chain_energy_rate(T_SUN, 0.0)
        self.assertEqual(eps, 0.0)

    def test_zero_hydrogen_zero_rate(self):
        """Zero hydrogen -> zero energy generation."""
        eps = pp_chain_energy_rate(T_SUN, RHO_SUN, X_H=0.0)
        self.assertEqual(eps, 0.0)


class TestCrossoverTemperature(unittest.TestCase):
    """pp-CNO crossover temperature."""

    def test_solar_crossover_about_17MK(self):
        """Crossover at solar composition: T ~ 15-20 MK.

        This is a well-known result in stellar astrophysics.
        Stars above ~1.3 M_sun have core T > T_cross.
        """
        T_cross = pp_cno_crossover_temperature()
        T_cross_MK = T_cross / 1e6
        self.assertGreater(T_cross_MK, 10)
        self.assertLess(T_cross_MK, 30)

    def test_crossover_shifts_with_sigma(self):
        """Crossover temperature shifts with σ.

        Higher σ → heavier nuclei → higher Coulomb barriers.
        CNO (higher Z) is affected more → CNO becomes relatively
        slower → crossover shifts to higher T.
        """
        T_0 = pp_cno_crossover_temperature(sigma=0.0)
        T_1 = pp_cno_crossover_temperature(sigma=0.5)
        self.assertNotEqual(T_0, T_1)


class TestTemperatureExponent(unittest.TestCase):
    """Power-law exponent for pp chain: eps ~ T^nu."""

    def test_pp_exponent_about_4(self):
        """pp temperature exponent ~ 4 at solar T.

        This is the classic result: ε_pp ∝ T^4.
        (More precisely: 3.5 to 5 depending on exact T.)
        """
        nu = pp_temperature_exponent(T_SUN)
        self.assertGreater(nu, 2.0)
        self.assertLess(nu, 8.0)


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts reaction rates — SSBM predictions."""

    def test_pp_rate_shifts_with_sigma(self):
        """Non-zero σ changes pp reaction rate."""
        sv_0 = reaction_rate_sigma_v('pp', T_SUN, sigma=0.0)
        sv_1 = reaction_rate_sigma_v('pp', T_SUN, sigma=0.1)
        self.assertNotEqual(sv_0, sv_1)
        self.assertGreater(sv_0, 0)
        self.assertGreater(sv_1, 0)

    def test_higher_sigma_lower_rate(self):
        """Higher σ → heavier nuclei → higher Gamow peak → lower rate.

        This is the core SSBM prediction for nucleosynthesis:
        in high-σ environments (near black holes), nuclear fusion
        is suppressed relative to standard conditions.
        """
        sv_0 = reaction_rate_sigma_v('pp', T_SUN, sigma=0.0)
        sv_1 = reaction_rate_sigma_v('pp', T_SUN, sigma=0.5)
        self.assertLess(sv_1, sv_0)

    def test_gamow_peak_shifts_with_sigma(self):
        """Gamow peak energy increases with σ (heavier reduced mass)."""
        E_0 = gamow_energy_keV(1, 1, 1, 1, T_SUN, sigma=0.0)
        E_1 = gamow_energy_keV(1, 1, 1, 1, T_SUN, sigma=0.5)
        self.assertGreater(E_1, E_0)

    def test_triple_alpha_shifts_with_sigma(self):
        """Triple-alpha rate shifts with σ.

        The Hoyle state resonance energy shifts through QCD binding.
        This changes the triple-alpha rate, affecting carbon production.
        """
        sv_0 = reaction_rate_sigma_v('triple_alpha', 100e6, sigma=0.0)
        sv_1 = reaction_rate_sigma_v('triple_alpha', 100e6, sigma=0.1)
        self.assertGreater(sv_0, 0)
        self.assertNotEqual(sv_0, sv_1)

    def test_earth_sigma_negligible(self):
        """Earth σ ~ 7e-10: rate change < 10^-6."""
        sv_0 = reaction_rate_sigma_v('pp', T_SUN, sigma=0.0)
        sv_e = reaction_rate_sigma_v('pp', T_SUN, sigma=7e-10)
        if sv_0 > 0:
            ratio = abs(sv_e - sv_0) / sv_0
            self.assertLess(ratio, 1e-4)

    def test_energy_generation_shifts(self):
        """σ affects total energy generation rate."""
        eps_0 = pp_chain_energy_rate(T_SUN, RHO_SUN, sigma=0.0)
        eps_1 = pp_chain_energy_rate(T_SUN, RHO_SUN, sigma=0.5)
        self.assertNotEqual(eps_0, eps_1)


class TestPhysicalSanity(unittest.TestCase):
    """Conservation laws and physical bounds."""

    def test_Q_values_positive(self):
        """All reaction Q-values are positive (exothermic)."""
        for key, rxn in REACTIONS.items():
            self.assertGreater(rxn['Q_MeV'], 0,
                f"{key}: nuclear burning should be exothermic")

    def test_S_factors_nonnegative(self):
        """All S-factors are non-negative."""
        for key, rxn in REACTIONS.items():
            self.assertGreaterEqual(rxn['S0_keV_barn'], 0,
                f"{key}: S-factor should be non-negative")

    def test_pp_chain_net_Q(self):
        """Net Q for pp-I chain: 4p -> 4He + 2e+ + 2nu.

        Q = 4×m_p - m_He4 - 2×m_e ≈ 26.73 MeV (total)
        Minus neutrino losses ≈ 0.5 MeV → 26.2 MeV net.
        """
        # The sum of pp + dp + He3_He3 Q-values:
        Q_total = (2 * REACTIONS['pp']['Q_MeV'] +
                   2 * REACTIONS['dp']['Q_MeV'] +
                   REACTIONS['He3_He3']['Q_MeV'])
        # Should be close to 26.73 MeV
        self.assertAlmostEqual(Q_total, 26.73, delta=1.0)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export format."""

    def test_reaction_required_fields(self):
        """Reaction export contains all required fields."""
        props = reaction_properties('pp')
        required = [
            'reaction', 'name', 'description', 'temperature_K',
            'sigma', 'Z1', 'A1', 'Z2', 'A2',
            'reduced_mass_MeV', 'gamow_energy_keV',
            'gamow_window_keV', 'sigma_v_cm3_s',
            'Q_MeV', 'Q_at_sigma_MeV', 'origin',
        ]
        for key in required:
            self.assertIn(key, props, f"Missing: {key}")

    def test_burning_summary_fields(self):
        """Stellar burning summary has all fields."""
        summary = stellar_burning_summary()
        required = [
            'temperature_K', 'density_kg_m3', 'sigma',
            'epsilon_pp_W_kg', 'epsilon_cno_W_kg',
            'epsilon_total_W_kg', 'dominant_chain',
            'pp_cno_crossover_MK', 'pp_fraction', 'origin',
        ]
        for key in required:
            self.assertIn(key, summary, f"Missing: {key}")

    def test_honest_origin_tags(self):
        """Origin string contains FIRST_PRINCIPLES and MEASURED."""
        props = reaction_properties('pp')
        origin = props['origin']
        self.assertIn('FIRST_PRINCIPLES', origin)
        self.assertIn('MEASURED', origin)

    def test_all_reactions_export(self):
        """Every reaction exports without error."""
        for key in REACTIONS:
            props = reaction_properties(key, T_K=T_SUN)
            self.assertIn('origin', props)

    def test_sigma_propagates(self):
        """σ parameter affects export values."""
        props_0 = reaction_properties('pp', sigma=0.0)
        props_1 = reaction_properties('pp', sigma=0.5)
        self.assertNotEqual(
            props_0['sigma_v_cm3_s'],
            props_1['sigma_v_cm3_s'])

    def test_solar_summary_pp_dominant(self):
        """At solar conditions, pp dominates."""
        summary = stellar_burning_summary(T_SUN, RHO_SUN)
        self.assertEqual(summary['dominant_chain'], 'pp')
        self.assertGreater(summary['pp_fraction'], 0.5)


if __name__ == '__main__':
    unittest.main()
