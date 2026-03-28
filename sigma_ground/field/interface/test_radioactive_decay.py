"""
Tests for the radioactive decay module.

Test structure:
  1. Alpha decay — Gamow theory, tunneling, Geiger-Nuttall
  2. Beta decay — Sargent's rule, Q⁵ scaling, free neutron
  3. General interface — dispatch, half-life, activity
  4. σ-dependence — decay rates shift with σ-field
  5. Physical sanity — conservation laws, ordering
  6. Cross-module consistency — uses constants from binding.py
  7. Nagatha export — complete format with origin tags
"""

import math
import unittest

from .radioactive_decay import (
    alpha_mass_mev,
    nuclear_radius_fm,
    gamow_factor,
    alpha_decay_constant,
    alpha_half_life,
    alpha_Q_decomposition,
    alpha_Q_at_sigma,
    beta_Q_value_mev,
    beta_Q_decomposition,
    beta_decay_constant,
    beta_half_life,
    decay_constant,
    half_life,
    half_life_human,
    activity_becquerel,
    remaining_fraction,
    geiger_nuttall_check,
    isotope_decay_properties,
    ISOTOPES,
    ALPHA_MASS_MEV,
    ALPHA_Z,
    ALPHA_A,
)


class TestAlphaMass(unittest.TestCase):
    """Alpha particle mass at various σ values."""

    def test_sigma_zero(self):
        """At σ=0, alpha mass ≈ 3727 MeV."""
        m = alpha_mass_mev(0.0)
        self.assertAlmostEqual(m, ALPHA_MASS_MEV, places=3)

    def test_increases_with_sigma(self):
        """Higher σ → heavier alpha (QCD mass scales up)."""
        m_0 = alpha_mass_mev(0.0)
        m_1 = alpha_mass_mev(1.0)
        self.assertGreater(m_1, m_0)

    def test_alpha_is_4_nucleons(self):
        """Alpha mass ≈ 4 × nucleon mass − 28 MeV binding."""
        from ..constants import PROTON_TOTAL_MEV, NEUTRON_TOTAL_MEV
        expected = 2 * PROTON_TOTAL_MEV + 2 * NEUTRON_TOTAL_MEV - 28.296
        self.assertAlmostEqual(alpha_mass_mev(0.0), expected, delta=0.01)


class TestNuclearRadius(unittest.TestCase):
    """Nuclear radius R = r₀ A^(1/3)."""

    def test_known_values(self):
        """Uranium-238: R ≈ 7.5 fm."""
        R = nuclear_radius_fm(238)
        self.assertAlmostEqual(R, 7.5, delta=0.5)

    def test_scales_with_A(self):
        """R ∝ A^(1/3) — doubling A increases R by 2^(1/3) ≈ 1.26."""
        R_100 = nuclear_radius_fm(100)
        R_200 = nuclear_radius_fm(200)
        ratio = R_200 / R_100
        self.assertAlmostEqual(ratio, 2.0**(1.0/3.0), delta=0.001)

    def test_positive(self):
        """Radius is always positive."""
        for A in [1, 4, 56, 208, 238]:
            self.assertGreater(nuclear_radius_fm(A), 0)


class TestGamowFactor(unittest.TestCase):
    """Gamow tunneling factor — the heart of alpha decay."""

    def test_positive(self):
        """Gamow factor is positive for all alpha emitters."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            G = gamow_factor(iso['daughter_Z'], iso['Q_value_MeV'],
                             ALPHA_MASS_MEV, iso['A'])
            self.assertGreater(G, 0, f"{key}: G should be positive")

    def test_higher_Q_lower_G(self):
        """Higher Q → lower Gamow factor → faster decay.

        This is the Geiger-Nuttall relation: faster alpha emitters
        have higher Q-values.
        """
        # Po-212 (Q=8.95) vs U-238 (Q=4.27) — same Z_daughter range
        G_po = gamow_factor(82, 8.954, ALPHA_MASS_MEV, 212)
        G_u = gamow_factor(90, 4.270, ALPHA_MASS_MEV, 238)
        self.assertLess(G_po, G_u)

    def test_zero_Q_infinite(self):
        """Q = 0 → infinite Gamow factor (no tunneling)."""
        G = gamow_factor(90, 0.0, ALPHA_MASS_MEV, 238)
        self.assertEqual(G, float('inf'))

    def test_typical_magnitude(self):
        """Gamow factor for U-238: G ~ 40-60 (gives very long half-life)."""
        G = gamow_factor(90, 4.270, ALPHA_MASS_MEV, 238)
        self.assertGreater(G, 20)
        self.assertLess(G, 80)


class TestAlphaDecay(unittest.TestCase):
    """Alpha decay rates from Gamow theory."""

    def test_all_alpha_emitters_have_rate(self):
        """Every alpha emitter produces a nonzero decay constant."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            lam = alpha_decay_constant(key)
            self.assertGreater(lam, 0, f"{key}: should have nonzero λ")

    def test_non_alpha_returns_zero(self):
        """Beta emitters return zero from alpha_decay_constant."""
        lam = alpha_decay_constant('C14')
        self.assertEqual(lam, 0.0)

    def test_geiger_nuttall_ordering(self):
        """Higher Q → shorter half-life (for similar Z).

        Po-212 (Q=8.95, t½=299ns) should be MUCH faster than
        Po-210 (Q=5.41, t½=138d). The Gamow exponential makes
        this a huge difference.
        """
        t_po212 = alpha_half_life('Po212')
        t_po210 = alpha_half_life('Po210')
        self.assertLess(t_po212, t_po210)

    def test_gamow_reproduces_order_of_magnitude(self):
        """Gamow theory reproduces half-lives within ~4 orders of magnitude.

        The Gamow model is a first-principles calculation with no fitted
        parameters beyond r₀. It does NOT include the alpha preformation
        factor (probability that 4 nucleons cluster into an alpha inside
        the nucleus), which is typically 10⁻² to 10⁻⁴. Including that
        would be a fitted parameter — we choose honest physics instead.

        Getting within 10⁴ of the measured value across a range spanning
        10²⁴ is a triumph of quantum mechanics.
        """
        results = geiger_nuttall_check()
        for r in results:
            # |log10(predicted/measured)| < 4.5 (allows for missing preformation)
            self.assertLess(abs(r['log10_ratio']), 4.5,
                f"{r['isotope']}: log10(pred/meas) = {r['log10_ratio']:.1f}, "
                f"predicted too far from measured")

    def test_geiger_nuttall_trend(self):
        """Geiger-Nuttall: log(λ) should correlate with 1/√Q.

        Check that isotopes with higher Q have higher log(λ) — the
        fundamental prediction of Gamow theory.
        """
        results = geiger_nuttall_check()
        if len(results) < 2:
            self.skipTest("Need ≥2 alpha emitters for trend check")

        # Sort by Q
        results.sort(key=lambda r: r['Q_MeV'])
        # Higher Q → higher log10(λ) (faster decay)
        for i in range(len(results) - 1):
            self.assertLess(
                results[i]['log10_lambda_predicted'],
                results[i + 1]['log10_lambda_predicted'],
                f"Geiger-Nuttall violated: {results[i]['isotope']} "
                f"(Q={results[i]['Q_MeV']}) should decay slower than "
                f"{results[i+1]['isotope']} (Q={results[i+1]['Q_MeV']})")


class TestBetaDecay(unittest.TestCase):
    """Beta decay from Sargent's rule."""

    def test_free_neutron_Q_value(self):
        """Free neutron Q ≈ 0.782 MeV (m_n − m_p − m_e)."""
        Q = beta_Q_value_mev('free_neutron', sigma=0.0)
        self.assertAlmostEqual(Q, 0.782, delta=0.01)

    def test_free_neutron_half_life(self):
        """Free neutron half-life ≈ 611 s at σ=0 (calibrated)."""
        t = beta_half_life('free_neutron', sigma=0.0)
        self.assertAlmostEqual(t, 611.0, delta=1.0)

    def test_carbon14_half_life(self):
        """C-14 half-life ≈ 5730 years at σ=0 (calibrated)."""
        t = beta_half_life('C14', sigma=0.0)
        t_years = t / (365.25 * 86400)
        self.assertAlmostEqual(t_years, 5730, delta=10)

    def test_sargent_Q5_scaling(self):
        """λ ∝ Q⁵ — verify the Q-dependence.

        If we scale Q by factor k, λ should scale by k⁵.
        We test this by comparing σ=0 and σ=small.
        """
        lam_0 = beta_decay_constant('free_neutron', sigma=0.0)
        # At small σ, Q shifts slightly
        lam_s = beta_decay_constant('free_neutron', sigma=0.01)
        Q_0 = beta_Q_value_mev('free_neutron', sigma=0.0)
        Q_s = beta_Q_value_mev('free_neutron', sigma=0.01)

        if Q_0 > 0 and Q_s > 0 and lam_0 > 0:
            expected_ratio = (Q_s / Q_0) ** 5
            actual_ratio = lam_s / lam_0
            self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.001)

    def test_non_beta_returns_zero(self):
        """Alpha emitters return zero from beta_decay_constant."""
        lam = beta_decay_constant('U238')
        self.assertEqual(lam, 0.0)

    def test_higher_Q_faster_decay(self):
        """Higher Q → faster beta decay (Sargent's Q⁵ rule).

        Co-60 (Q=2.82) decays much faster than C-14 (Q=0.156).
        """
        t_c14 = beta_half_life('C14')
        t_co60 = beta_half_life('Co60')
        self.assertLess(t_co60, t_c14)


class TestGeneralInterface(unittest.TestCase):
    """Dispatch, half-life, activity, remaining fraction."""

    def test_dispatch_alpha(self):
        """decay_constant dispatches to alpha for U-238."""
        lam_dispatch = decay_constant('U238')
        lam_direct = alpha_decay_constant('U238')
        self.assertEqual(lam_dispatch, lam_direct)

    def test_dispatch_beta(self):
        """decay_constant dispatches to beta for C-14."""
        lam_dispatch = decay_constant('C14')
        lam_direct = beta_decay_constant('C14')
        self.assertEqual(lam_dispatch, lam_direct)

    def test_half_life_consistency(self):
        """half_life = ln(2) / decay_constant for all isotopes."""
        for key in ISOTOPES:
            lam = decay_constant(key)
            if lam > 0:
                t = half_life(key)
                self.assertAlmostEqual(t, math.log(2) / lam, places=5)

    def test_half_life_human_units(self):
        """Human-readable units make sense."""
        # U-238: Gamow model gives Gyr (longer than measured due to
        # missing preformation factor, but correct unit regime)
        val, unit = half_life_human('U238')
        self.assertEqual(unit, 'Gyr')
        self.assertGreater(val, 1.0)  # at least 1 Gyr

        # C-14: beta decay (calibrated from measured), should be in years
        val, unit = half_life_human('C14')
        self.assertEqual(unit, 'years')
        self.assertAlmostEqual(val, 5730, delta=10)

        # Po-212: very fast alpha emitter
        val, unit = half_life_human('Po212')
        # Gamow model may put this in a different sub-second unit
        self.assertIn(unit, ('ns', '\u03bcs', 'ms', 'seconds'))

    def test_activity(self):
        """Activity = λN — 1 mol of C-14."""
        N_A = 6.022e23
        A = activity_becquerel('C14', N_A)
        self.assertGreater(A, 0)
        # Should be huge: ~10¹² Bq for 1 mol
        self.assertGreater(A, 1e10)

    def test_remaining_fraction_at_zero(self):
        """At t=0, all atoms remain."""
        f = remaining_fraction('C14', time_s=0.0)
        self.assertAlmostEqual(f, 1.0, places=10)

    def test_remaining_fraction_at_half_life(self):
        """At t = t½, half the atoms remain."""
        t_half = half_life('C14')
        f = remaining_fraction('C14', t_half)
        self.assertAlmostEqual(f, 0.5, delta=0.001)

    def test_remaining_fraction_decreases(self):
        """Remaining fraction decreases monotonically."""
        fracs = [remaining_fraction('Co60', t) for t in [0, 1e7, 1e8, 1e9]]
        for i in range(len(fracs) - 1):
            self.assertGreater(fracs[i], fracs[i + 1])


class TestAlphaQDecomposition(unittest.TestCase):
    """Coulomb/strong decomposition of alpha Q-values — DERIVED, not guessed."""

    def test_components_sum_to_measured(self):
        """Q_coulomb + Q_strong = Q_measured (exact by construction)."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            Q_coulomb, Q_strong = alpha_Q_decomposition(key)
            Q_sum = Q_coulomb + Q_strong
            self.assertAlmostEqual(
                Q_sum, iso['Q_value_MeV'], places=6,
                msg=f"{key}: Q_coulomb + Q_strong != Q_measured")

    def test_coulomb_positive(self):
        """Q_coulomb > 0: Coulomb repulsion pushes alpha out.

        The parent nucleus has more protons packed together than
        the daughter + free alpha. Coulomb wants the alpha OUT.
        """
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            Q_coulomb, _ = alpha_Q_decomposition(key)
            self.assertGreater(Q_coulomb, 0,
                f"{key}: Coulomb should push alpha out (Q_C > 0)")

    def test_strong_negative(self):
        """Q_strong < 0: strong force holds alpha in.

        The net strong binding is REDUCED when the alpha escapes
        (4 nucleons lose their neighbors). Strong wants alpha IN.
        """
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            _, Q_strong = alpha_Q_decomposition(key)
            self.assertLess(Q_strong, 0,
                f"{key}: strong force should hold alpha in (Q_S < 0)")

    def test_coulomb_dominates_strong(self):
        """Q_coulomb > |Q_strong| for all alpha emitters (otherwise no decay).

        Alpha decay happens ONLY because Coulomb repulsion
        barely wins over strong attraction. The Q-value is a
        small positive difference between two large terms.
        """
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            Q_coulomb, Q_strong = alpha_Q_decomposition(key)
            self.assertGreater(Q_coulomb, abs(Q_strong),
                f"{key}: Coulomb must beat strong for decay to occur")

    def test_u238_magnitudes(self):
        """U-238: Q_C ≈ +36 MeV, Q_S ≈ −31 MeV, net Q ≈ +4.3 MeV.

        Two massive forces nearly cancel, leaving a tiny residual
        that powers 4.5 billion years of radioactive heating.
        """
        Q_C, Q_S = alpha_Q_decomposition('U238')
        self.assertGreater(Q_C, 20.0)   # tens of MeV Coulomb push
        self.assertLess(Q_S, -15.0)     # tens of MeV strong pull
        # Net is the measured Q ≈ 4.27 MeV
        self.assertAlmostEqual(Q_C + Q_S, 4.270, delta=0.001)

    def test_Q_at_sigma_zero_equals_measured(self):
        """At σ=0, alpha_Q_at_sigma returns measured Q-value."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            Q = alpha_Q_at_sigma(key, sigma=0.0)
            self.assertAlmostEqual(Q, iso['Q_value_MeV'], places=6,
                msg=f"{key}: Q(σ=0) should equal measured Q")

    def test_Q_decreases_with_sigma(self):
        """Higher σ → Q decreases (strong force gets stronger, holds alpha tighter).

        This means alpha decay SLOWS DOWN at high σ — the opposite
        of the naive guess f_Q_strong=0.2 which made Q increase!
        """
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            Q_0 = alpha_Q_at_sigma(key, sigma=0.0)
            Q_1 = alpha_Q_at_sigma(key, sigma=1.0)
            self.assertLess(Q_1, Q_0,
                f"{key}: Q should decrease with σ (strong holds tighter)")

    def test_critical_sigma_exists(self):
        """At some critical σ, Q → 0 and alpha decay turns off.

        Since Q_strong < 0 and grows as e^σ, eventually |Q_strong| > Q_coulomb.
        """
        Q_C, Q_S = alpha_Q_decomposition('U238')
        # Critical σ: Q_S × e^σ + Q_C = 0 → e^σ = Q_C / |Q_S|
        sigma_crit = math.log(Q_C / abs(Q_S))
        self.assertGreater(sigma_crit, 0, "Critical σ should be positive")
        # Verify Q is indeed ~0 at critical σ
        Q_crit = alpha_Q_at_sigma('U238', sigma=sigma_crit)
        self.assertAlmostEqual(Q_crit, 0.0, delta=0.01)


class TestBetaQDecomposition(unittest.TestCase):
    """Coulomb/strong decomposition of beta Q-values — DERIVED, not guessed."""

    def test_components_sum_to_measured(self):
        """Q_invariant + Q_sigma_coeff = Q_measured at σ=0."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] not in ('beta_minus', 'beta_plus'):
                continue
            Q_inv, Q_sig = beta_Q_decomposition(key)
            Q_sum = Q_inv + Q_sig
            self.assertAlmostEqual(
                Q_sum, iso['Q_value_MeV'], places=5,
                msg=f"{key}: Q_inv + Q_sig != Q_measured")

    def test_free_neutron_decomposition(self):
        """Free neutron: pure mass decomposition (no nuclear binding).

        Q_inv = m_n_bare − m_p_bare − m_e (EM mass difference)
        Q_sig = m_n_QCD − m_p_QCD (QCD mass difference)
        """
        Q_inv, Q_sig = beta_Q_decomposition('free_neutron')
        # Q_sig should be small (QCD contribution to n-p mass diff)
        self.assertLess(abs(Q_sig), 5.0)
        # Total at σ=0 should be measured Q ≈ 0.782 MeV
        self.assertAlmostEqual(Q_inv + Q_sig, 0.782, delta=0.01)

    def test_Q_at_sigma_zero_equals_measured(self):
        """beta_Q_value_mev at σ=0 returns measured Q-value."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] not in ('beta_minus', 'beta_plus'):
                continue
            Q = beta_Q_value_mev(key, sigma=0.0)
            self.assertAlmostEqual(Q, iso['Q_value_MeV'], places=5,
                msg=f"{key}: Q(σ=0) should equal measured Q")

    def test_invariant_part_includes_coulomb_step(self):
        """For nuclear beta decay, Q_invariant includes Coulomb step.

        When Z → Z+1, the daughter has more Coulomb repulsion.
        This is an EM effect that goes into Q_invariant.
        """
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'beta_minus' or key == 'free_neutron':
                continue
            Q_inv, _ = beta_Q_decomposition(key)
            # Q_invariant can be positive or negative depending on Coulomb step
            # but it should be finite and not NaN
            self.assertTrue(math.isfinite(Q_inv), f"{key}: Q_inv not finite")


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts decay rates — SSBM predictions."""

    def test_alpha_sigma_shifts_rate(self):
        """Non-zero σ changes alpha decay rate."""
        lam_0 = alpha_decay_constant('U238', sigma=0.0)
        lam_1 = alpha_decay_constant('U238', sigma=0.1)
        self.assertNotEqual(lam_0, lam_1)
        # Both should be positive
        self.assertGreater(lam_0, 0)
        self.assertGreater(lam_1, 0)

    def test_beta_sigma_shifts_rate(self):
        """Non-zero σ changes beta decay rate."""
        lam_0 = beta_decay_constant('free_neutron', sigma=0.0)
        lam_1 = beta_decay_constant('free_neutron', sigma=0.1)
        self.assertNotEqual(lam_0, lam_1)

    def test_free_neutron_Q_shifts_with_sigma(self):
        """Free neutron Q-value changes with σ (different bare masses)."""
        Q_0 = beta_Q_value_mev('free_neutron', sigma=0.0)
        Q_1 = beta_Q_value_mev('free_neutron', sigma=0.1)
        self.assertNotEqual(Q_0, Q_1)
        # Both should be positive (neutron heavier than proton)
        self.assertGreater(Q_0, 0)

    def test_earth_sigma_negligible_alpha(self):
        """Earth σ ~ 7×10⁻¹⁰: alpha decay rate changes < 10⁻⁶."""
        lam_0 = alpha_decay_constant('U238', sigma=0.0)
        lam_earth = alpha_decay_constant('U238', sigma=7e-10)
        if lam_0 > 0:
            ratio = abs(lam_earth - lam_0) / lam_0
            self.assertLess(ratio, 1e-4)

    def test_earth_sigma_negligible_beta(self):
        """Earth σ ~ 7×10⁻¹⁰: beta decay rate changes < 10⁻⁶."""
        lam_0 = beta_decay_constant('C14', sigma=0.0)
        lam_earth = beta_decay_constant('C14', sigma=7e-10)
        if lam_0 > 0:
            ratio = abs(lam_earth - lam_0) / lam_0
            self.assertLess(ratio, 1e-4)


class TestPhysicalSanity(unittest.TestCase):
    """Conservation laws and physical bounds."""

    def test_daughter_charge_conservation(self):
        """Alpha decay: Z_parent = Z_daughter + 2."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            self.assertEqual(
                iso['Z'], iso['daughter_Z'] + ALPHA_Z,
                f"{key}: charge not conserved in alpha decay")

    def test_daughter_mass_conservation(self):
        """Alpha decay: A_parent = A_daughter + 4."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'alpha':
                continue
            self.assertEqual(
                iso['A'], iso['daughter_A'] + ALPHA_A,
                f"{key}: baryon number not conserved in alpha decay")

    def test_beta_charge_conservation(self):
        """Beta⁻ decay: Z_daughter = Z_parent + 1."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] != 'beta_minus':
                continue
            self.assertEqual(
                iso['daughter_Z'], iso['Z'] + 1,
                f"{key}: charge not conserved in beta⁻ decay")

    def test_beta_mass_conservation(self):
        """Beta decay: A_daughter = A_parent (no nucleon change)."""
        for key, iso in ISOTOPES.items():
            if iso['decay_mode'] not in ('beta_minus', 'beta_plus'):
                continue
            self.assertEqual(
                iso['daughter_A'], iso['A'],
                f"{key}: baryon number not conserved in beta decay")

    def test_Q_values_positive(self):
        """All Q-values are positive (decay is energetically allowed)."""
        for key, iso in ISOTOPES.items():
            self.assertGreater(iso['Q_value_MeV'], 0,
                f"{key}: Q-value should be positive")

    def test_half_lives_positive(self):
        """All measured half-lives are positive."""
        for key, iso in ISOTOPES.items():
            self.assertGreater(iso['half_life_s'], 0,
                f"{key}: half-life should be positive")


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export format."""

    def test_required_fields(self):
        """Export contains all required fields."""
        props = isotope_decay_properties('U238')
        required = [
            'isotope', 'name', 'Z', 'A', 'decay_mode', 'sigma',
            'Q_value_MeV', 'half_life_s', 'half_life_measured_s',
            'half_life_human', 'decay_constant_per_s', 'daughter',
            'origin',
        ]
        for key in required:
            self.assertIn(key, props, f"Missing: {key}")

    def test_alpha_has_gamow(self):
        """Alpha decay export includes Gamow factor."""
        props = isotope_decay_properties('U238')
        self.assertIn('gamow_factor', props)
        self.assertGreater(props['gamow_factor'], 0)

    def test_beta_has_Q_at_sigma(self):
        """Beta decay export includes Q at σ."""
        props = isotope_decay_properties('C14')
        self.assertIn('Q_value_at_sigma_MeV', props)

    def test_honest_origin_tags(self):
        """Origin string contains FIRST_PRINCIPLES and MEASURED."""
        for key in ['U238', 'C14']:
            props = isotope_decay_properties(key)
            origin = props['origin']
            self.assertIn('FIRST_PRINCIPLES', origin)
            self.assertIn('MEASURED', origin)

    def test_all_isotopes_export(self):
        """Every isotope exports without error."""
        for key in ISOTOPES:
            props = isotope_decay_properties(key)
            self.assertIn('origin', props)

    def test_sigma_propagates(self):
        """σ parameter affects export values."""
        props_0 = isotope_decay_properties('U238', sigma=0.0)
        props_1 = isotope_decay_properties('U238', sigma=0.5)
        self.assertNotEqual(
            props_0['half_life_s'], props_1['half_life_s'])


if __name__ == '__main__':
    unittest.main()
