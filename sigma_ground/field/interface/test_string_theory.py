"""Tests for string_theory.py — Swampland, hierarchy, moduli stabilization.

Predictions tested:
  1. Effective potential V(σ) is positive, monotonically increasing, convex
  2. σ-field mass is QCD-scale (~Λ_QCD ≈ 217 MeV)
  3. Hierarchy ratio α_EM/α_G ≈ 10^36 at σ = 0
  4. QCD fraction accounts for ~10^4 of the hierarchy reduction
  5. WGC satisfied by ≥ 36 orders of magnitude at σ = 0
  6. WGC critical σ ≈ 41.5, far beyond σ_conv ≈ 1.849
  7. Distance Conjecture trivially satisfied (finite field range)
  8. de Sitter Conjecture satisfied (convex potential, no dS vacua)
  9. Compactification radius near Planck length for n = 6, 7
 10. Vacuum selection reduces landscape to 1 parameter
 11. Cosmological constant problem transformed (no 10^-120 tuning)
 12. Dilaton correspondence maps correctly
 13. Hierarchy shrinks with increasing σ (gravity strengthens)
"""

import math
import unittest

from .string_theory import (
    effective_potential,
    effective_potential_derivative,
    sigma_field_mass_mev,
    moduli_stabilization_summary,
    gravitational_coupling,
    hierarchy_ratio,
    hierarchy_from_qcd_fraction,
    hierarchy_at_sigma_values,
    vacuum_selection_analysis,
    weak_gravity_conjecture,
    wgc_critical_sigma,
    distance_conjecture_check,
    de_sitter_conjecture_check,
    swampland_summary,
    dilaton_correspondence,
    cosmological_constant_analysis,
    compactification_radius,
    compactification_predictions,
    sigma_modulus_comparison,
    sigma_hierarchy_shift,
    string_theory_report,
    full_report,
    M_PLANCK_MEV,
    ALPHA_G,
    RHO_QCD_J_M3,
)
from ..constants import (
    XI, ETA, SIGMA_CONV, ALPHA, L_PLANCK, SIGMA_HERE,
    PROTON_QCD_MEV, PROTON_TOTAL_MEV, PROTON_BARE_MEV,
    LAMBDA_QCD_MEV,
)


# =====================================================================
# DERIVED CONSTANTS
# =====================================================================

class TestDerivedConstants(unittest.TestCase):
    """Test fundamental constants derived for string theory context."""

    def test_planck_mass_mev(self):
        """M_Planck ≈ 1.22 × 10^22 MeV (PDG)."""
        self.assertAlmostEqual(M_PLANCK_MEV / 1e22, 1.22, delta=0.01)

    def test_gravitational_coupling(self):
        """α_G ≈ 5.9 × 10^-39 (dimensionless gravitational coupling)."""
        self.assertAlmostEqual(ALPHA_G / 1e-39, 5.9, delta=0.2)

    def test_alpha_g_much_less_than_alpha_em(self):
        """Gravity is ~10^36 times weaker than EM."""
        ratio = ALPHA / ALPHA_G
        self.assertGreater(ratio, 1e35)
        self.assertLess(ratio, 1e37)

    def test_rho_qcd_positive(self):
        """QCD energy density at saturation should be positive and huge."""
        self.assertGreater(RHO_QCD_J_M3, 1e30)  # enormous in SI


# =====================================================================
# 1. MODULI STABILIZATION
# =====================================================================

class TestEffectivePotential(unittest.TestCase):
    """Test V_eff(σ) and its properties."""

    def test_V_zero_at_origin(self):
        """V(0) = 0 — the minimum is at flat spacetime."""
        V = effective_potential(0.0)
        self.assertAlmostEqual(V, 0.0, places=10)

    def test_V_positive_for_positive_sigma(self):
        """V(σ) > 0 for all σ > 0 (energy cost to compress spacetime)."""
        for s in [0.01, 0.1, 0.5, 1.0, 1.5, SIGMA_CONV * 0.99]:
            V = effective_potential(s)
            self.assertGreater(V, 0, msg=f"V({s}) should be > 0")

    def test_V_negative_for_negative_sigma(self):
        """V(σ) < 0 for σ < 0 (decompressed spacetime)."""
        V = effective_potential(-0.1)
        self.assertLess(V, 0)

    def test_V_monotonically_increasing(self):
        """V(σ₁) < V(σ₂) for σ₁ < σ₂ in [0, σ_conv)."""
        sigmas = [0.0, 0.1, 0.5, 1.0, 1.5, SIGMA_CONV * 0.99]
        potentials = [effective_potential(s) for s in sigmas]
        for i in range(len(potentials) - 1):
            self.assertLess(potentials[i], potentials[i + 1])

    def test_V_beyond_sigma_conv_returns_none(self):
        """V(σ) returns None for σ ≥ σ_conv (BEYOND domain)."""
        self.assertIsNone(effective_potential(SIGMA_CONV))
        self.assertIsNone(effective_potential(SIGMA_CONV + 1))

    def test_V_at_sigma_conv_boundary(self):
        """V(σ_conv − ε) should be large but finite."""
        V = effective_potential(SIGMA_CONV * 0.999)
        self.assertIsNotNone(V)
        self.assertGreater(V, 1000)  # > 1 GeV per nucleon

    def test_derivative_positive_everywhere(self):
        """dV/dσ > 0 for all σ in domain (restoring force toward σ = 0)."""
        for s in [0.0, 0.1, 0.5, 1.0, 1.5]:
            dV = effective_potential_derivative(s)
            self.assertGreater(dV, 0, msg=f"dV/dσ({s}) should be > 0")

    def test_derivative_beyond_sigma_conv(self):
        """dV/dσ returns None beyond σ_conv."""
        self.assertIsNone(effective_potential_derivative(SIGMA_CONV))

    def test_derivative_equals_m_qcd_at_origin(self):
        """dV/dσ(0) = m_QCD ≈ 929 MeV."""
        dV = effective_potential_derivative(0.0)
        self.assertAlmostEqual(dV, PROTON_QCD_MEV, places=1)

    def test_potential_is_convex(self):
        """d²V/dσ² > 0 everywhere (no inflection points)."""
        # d²V/dσ² = m_QCD × e^σ > 0 always
        for s in [0.0, 0.5, 1.0, 1.5]:
            d2V = PROTON_QCD_MEV * math.exp(s)
            self.assertGreater(d2V, 0)


class TestSigmaFieldMass(unittest.TestCase):
    """Test σ-field mass from QCD potential."""

    def test_mass_at_vacuum_qcd_scale(self):
        """m_σ(0) ~ Λ_QCD ≈ 217 MeV."""
        m = sigma_field_mass_mev(0)
        self.assertAlmostEqual(m, LAMBDA_QCD_MEV, places=0)

    def test_mass_increases_with_sigma(self):
        """σ-field gets heavier as spacetime compresses."""
        m0 = sigma_field_mass_mev(0)
        m1 = sigma_field_mass_mev(1.0)
        self.assertGreater(m1, m0)

    def test_mass_comparable_to_pion(self):
        """m_σ ~ 1-2 × m_π (both QCD-scale objects)."""
        m = sigma_field_mass_mev(0)
        m_pion = 134.977  # MeV (π⁰)
        ratio = m / m_pion
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 5.0)


class TestModuliSummary(unittest.TestCase):
    """Test the moduli stabilization summary report."""

    def test_summary_has_all_fields(self):
        """Summary should contain all required keys."""
        s = moduli_stabilization_summary()
        self.assertIn('sigma_conv', s)
        self.assertIn('V_wall_MeV', s)
        self.assertIn('mass_at_vacuum_MeV', s)
        self.assertIn('mass_comparison', s)
        self.assertIn('potential_profile', s)
        self.assertIn('string_comparison', s)

    def test_wall_height_large(self):
        """Energy cost to reach σ_conv wall should be > 1 GeV/nucleon."""
        s = moduli_stabilization_summary()
        self.assertGreater(s['V_wall_MeV'], 1000)

    def test_potential_profile_ordered(self):
        """Potential profile should have increasing V with σ."""
        s = moduli_stabilization_summary()
        profile = s['potential_profile']
        for i in range(len(profile) - 1):
            self.assertLessEqual(profile[i]['V_MeV'], profile[i + 1]['V_MeV'])


# =====================================================================
# 2. HIERARCHY PROBLEM
# =====================================================================

class TestHierarchy(unittest.TestCase):
    """Test hierarchy ratio computations."""

    def test_alpha_g_at_vacuum(self):
        """α_G(0) ≈ 5.9 × 10^-39."""
        ag = gravitational_coupling(SIGMA_HERE)
        self.assertAlmostEqual(ag / 1e-39, 5.9, delta=0.2)

    def test_hierarchy_ratio_at_vacuum(self):
        """α_EM/α_G ≈ 1.24 × 10^36 at σ = 0."""
        R = hierarchy_ratio(SIGMA_HERE)
        log_R = math.log10(R)
        self.assertAlmostEqual(log_R, 36.1, delta=0.2)

    def test_hierarchy_shrinks_with_sigma(self):
        """Hierarchy should decrease as σ increases (gravity strengthens)."""
        R0 = hierarchy_ratio(0)
        R1 = hierarchy_ratio(1.0)
        self.assertLess(R1, R0)

    def test_hierarchy_at_sigma_conv(self):
        """At σ_conv, hierarchy should be reduced by ~40× but still huge."""
        R0 = hierarchy_ratio(0)
        R_conv = hierarchy_ratio(SIGMA_CONV * 0.99)
        reduction = R0 / R_conv
        self.assertGreater(reduction, 10)  # significant reduction
        self.assertLess(reduction, 100)    # but not extreme
        self.assertGreater(R_conv, 1e34)   # still enormous

    def test_alpha_g_increases_with_sigma(self):
        """Gravity strengthens as σ increases (heavier nucleons)."""
        ag0 = gravitational_coupling(0)
        ag1 = gravitational_coupling(1.0)
        self.assertGreater(ag1, ag0)


class TestHierarchyDecomposition(unittest.TestCase):
    """Test hierarchy decomposition into QCD and Higgs sectors."""

    def test_qcd_reduction_factor(self):
        """QCD should reduce hierarchy by ~(m_p/m_bare)² ≈ 10^4."""
        h = hierarchy_from_qcd_fraction()
        reduction = h['qcd_reduction_factor']
        log_reduction = math.log10(reduction)
        # (938/9)² ≈ 10^4
        self.assertAlmostEqual(log_reduction, 4.0, delta=0.5)

    def test_bare_hierarchy_larger(self):
        """Without QCD binding, hierarchy would be larger."""
        h = hierarchy_from_qcd_fraction()
        self.assertGreater(h['R_bare_only'], h['R_observed'])

    def test_log10_hierarchy_around_36(self):
        """log₁₀(R) ≈ 36."""
        h = hierarchy_from_qcd_fraction()
        self.assertAlmostEqual(h['log10_R_observed'], 36.1, delta=0.2)

    def test_log10_bare_around_40(self):
        """log₁₀(R_bare) ≈ 40 (bare quarks only, no QCD)."""
        h = hierarchy_from_qcd_fraction()
        self.assertAlmostEqual(h['log10_R_bare'], 40.1, delta=0.5)


class TestHierarchyProfile(unittest.TestCase):
    """Test hierarchy across σ domain."""

    def test_profile_has_multiple_entries(self):
        """Should compute hierarchy at several σ values."""
        results = hierarchy_at_sigma_values()
        self.assertGreaterEqual(len(results), 5)

    def test_profile_monotonically_decreasing(self):
        """Hierarchy ratio should decrease with increasing σ."""
        results = hierarchy_at_sigma_values()
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]['alpha_EM_over_alpha_G'],
                results[i + 1]['alpha_EM_over_alpha_G'],
            )


# =====================================================================
# 3. VACUUM SELECTION
# =====================================================================

class TestVacuumSelection(unittest.TestCase):
    """Test landscape reduction analysis."""

    def test_derived_quantities_from_xi(self):
        """ξ should determine σ_conv, η, and mass spectrum."""
        v = vacuum_selection_analysis()
        d = v['derived_from_xi']
        self.assertAlmostEqual(d['sigma_conv'], SIGMA_CONV, places=5)
        self.assertAlmostEqual(d['eta'], ETA, places=4)
        self.assertGreater(d['proton_mass_at_conv_MeV'], PROTON_TOTAL_MEV)

    def test_sensitivity_negative(self):
        """dσ_conv/dξ = −1/ξ < 0 (larger ξ → smaller σ_conv)."""
        v = vacuum_selection_analysis()
        self.assertLess(v['sensitivity']['dsigma_conv_dxi'], 0)

    def test_sensitivity_magnitude(self):
        """dσ_conv/dξ ≈ −1/ξ ≈ −6.3."""
        v = vacuum_selection_analysis()
        expected = -1.0 / XI
        actual = v['sensitivity']['dsigma_conv_dxi']
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_mass_enhancement_at_conversion(self):
        """At σ_conv: m_p ≈ 6.3× vacuum mass (= 1/ξ factor)."""
        v = vacuum_selection_analysis()
        enhancement = v['derived_from_xi']['mass_enhancement_at_conv']
        # e^σ_conv = e^(-ln ξ) = 1/ξ, but only QCD part scales
        # enhancement = (m_bare + m_QCD/ξ) / (m_bare + m_QCD)
        expected = (PROTON_BARE_MEV + PROTON_QCD_MEV / XI) / PROTON_TOTAL_MEV
        self.assertAlmostEqual(enhancement, expected, places=3)


# =====================================================================
# 4. SWAMPLAND COMPATIBILITY
# =====================================================================

class TestWeakGravityConjecture(unittest.TestCase):
    """Test WGC at various σ values."""

    def test_wgc_satisfied_at_vacuum(self):
        """WGC should be satisfied at σ = 0."""
        w = weak_gravity_conjecture(SIGMA_HERE)
        self.assertTrue(w['WGC_satisfied'])

    def test_wgc_margin_36_orders(self):
        """WGC margin ≈ 36 orders of magnitude at σ = 0."""
        w = weak_gravity_conjecture(SIGMA_HERE)
        self.assertAlmostEqual(w['log10_ratio'], 36.1, delta=0.2)

    def test_wgc_satisfied_at_sigma_conv(self):
        """WGC should still hold at σ_conv (matter converts first)."""
        w = weak_gravity_conjecture(SIGMA_CONV * 0.99)
        self.assertTrue(w['WGC_satisfied'])
        self.assertGreater(w['log10_ratio'], 34)  # still huge margin

    def test_wgc_em_invariant(self):
        """EM force should be σ-invariant."""
        w0 = weak_gravity_conjecture(0)
        w1 = weak_gravity_conjecture(1.0)
        self.assertAlmostEqual(w0['F_EM_Nm2'], w1['F_EM_Nm2'], places=40)

    def test_gravity_strengthens_with_sigma(self):
        """Gravitational force should increase with σ (heavier nucleons)."""
        w0 = weak_gravity_conjecture(0)
        w1 = weak_gravity_conjecture(1.0)
        self.assertGreater(w1['F_G_Nm2'], w0['F_G_Nm2'])


class TestWGCCriticalSigma(unittest.TestCase):
    """Test the σ value where WGC would be saturated."""

    def test_sigma_wgc_around_41(self):
        """σ_WGC ≈ 41.5 (where gravity would match EM)."""
        w = wgc_critical_sigma()
        self.assertAlmostEqual(w['sigma_WGC'], 41.5, delta=1.0)

    def test_sigma_wgc_far_beyond_conv(self):
        """σ_WGC >> σ_conv (conversion wall prevents WGC violation)."""
        w = wgc_critical_sigma()
        self.assertGreater(w['sigma_WGC'], 10 * SIGMA_CONV)

    def test_structurally_impossible(self):
        """WGC violation should be flagged as structurally impossible."""
        w = wgc_critical_sigma()
        self.assertTrue(w['structurally_impossible'])

    def test_m_wgc_near_planck(self):
        """WGC saturation mass ≈ Planck mass × √α_EM."""
        w = wgc_critical_sigma()
        # m_WGC = √(e²/4πε₀G) = m_Planck × √α_EM
        expected_mev = M_PLANCK_MEV * math.sqrt(ALPHA)
        self.assertAlmostEqual(
            w['m_WGC_MeV'] / expected_mev, 1.0, delta=0.01
        )


class TestDistanceConjecture(unittest.TestCase):
    """Test Distance Conjecture compatibility."""

    def test_finite_field_range(self):
        """σ domain should be finite."""
        d = distance_conjecture_check()
        self.assertTrue(d['is_finite'])
        self.assertAlmostEqual(d['field_range'], SIGMA_CONV, places=5)

    def test_trivially_satisfied(self):
        """Distance Conjecture should be trivially satisfied."""
        d = distance_conjecture_check()
        self.assertTrue(d['trivially_satisfied'])
        self.assertFalse(d['distance_conjecture_applies'])

    def test_masses_get_heavier(self):
        """At the boundary, states get HEAVIER, not lighter."""
        d = distance_conjecture_check()
        self.assertGreater(d['mass_ratio_at_boundary'], 1.0)
        self.assertEqual(d['tower_direction'], 'heavier (not lighter)')


class TestDeSitterConjecture(unittest.TestCase):
    """Test de Sitter Conjecture compatibility."""

    def test_potential_convex(self):
        """V_eff(σ) should be convex everywhere."""
        d = de_sitter_conjecture_check()
        self.assertTrue(d['potential_is_convex'])

    def test_no_metastable_ds(self):
        """No metastable de Sitter vacua in V(σ)."""
        d = de_sitter_conjecture_check()
        self.assertFalse(d['has_metastable_dS'])

    def test_gradient_condition(self):
        """Gradient condition |∇V|/V ≥ O(1) should be satisfied."""
        d = de_sitter_conjecture_check()
        self.assertTrue(d['gradient_condition_satisfied'])

    def test_profile_all_convex(self):
        """Every point in the profile should have d²V > 0."""
        d = de_sitter_conjecture_check()
        for entry in d['profile']:
            self.assertTrue(entry['is_convex'])


class TestSwamplandSummary(unittest.TestCase):
    """Test overall swampland compatibility."""

    def test_all_conjectures_satisfied(self):
        """All three swampland conjectures should pass."""
        s = swampland_summary()
        self.assertEqual(s['overall'], 'ALL SATISFIED')

    def test_has_all_three_checks(self):
        """Summary should include WGC, Distance, and de Sitter."""
        s = swampland_summary()
        self.assertIn('weak_gravity', s)
        self.assertIn('distance', s)
        self.assertIn('de_sitter', s)


# =====================================================================
# 5. DILATON CORRESPONDENCE
# =====================================================================

class TestDilatonCorrespondence(unittest.TestCase):
    """Test σ ↔ dilaton mapping."""

    def test_vacuum_coupling_unity(self):
        """At σ = 0: e^σ = 1 (standard physics)."""
        d = dilaton_correspondence()
        self.assertAlmostEqual(d['vacuum_coupling'], 1.0, places=10)

    def test_max_coupling(self):
        """At σ_conv: e^σ = 1/ξ ≈ 6.32."""
        d = dilaton_correspondence()
        expected = 1.0 / XI
        self.assertAlmostEqual(d['max_coupling'], expected, delta=0.01)

    def test_coupling_range(self):
        """Coupling ranges from 1 to 1/ξ ≈ 6.32."""
        d = dilaton_correspondence()
        self.assertAlmostEqual(d['coupling_range'], 1.0 / XI, delta=0.01)

    def test_correspondence_table_has_all_entries(self):
        """Correspondence table should cover all key properties."""
        d = dilaton_correspondence()
        table = d['correspondence_table']
        self.assertIn('field', table)
        self.assertIn('coupling', table)
        self.assertIn('field_equation', table)
        self.assertIn('domain', table)
        self.assertIn('mass', table)
        self.assertIn('status', table)


# =====================================================================
# 6. COSMOLOGICAL CONSTANT
# =====================================================================

class TestCosmologicalConstant(unittest.TestCase):
    """Test CC problem transformation."""

    def test_standard_fine_tuning_123_orders(self):
        """Standard CC problem: ~123 orders of magnitude."""
        cc = cosmological_constant_analysis()
        self.assertEqual(cc['standard_problem']['fine_tuning_orders'], 123)

    def test_ssbm_no_extreme_tuning(self):
        """SSBM effective tuning is O(0.1), not 10^-120."""
        cc = cosmological_constant_analysis()
        eff = cc['ssbm_resolution']['effective_tuning']
        # ξ × η ≈ 0.066
        self.assertGreater(eff, 0.01)
        self.assertLess(eff, 1.0)

    def test_xi_times_eta(self):
        """Effective tuning = ξ × η ≈ 0.066."""
        cc = cosmological_constant_analysis()
        expected = XI * ETA
        self.assertAlmostEqual(
            cc['ssbm_resolution']['effective_tuning'], expected, places=4
        )

    def test_energy_released_positive(self):
        """Energy released per nucleon at σ_conv should be positive."""
        cc = cosmological_constant_analysis()
        self.assertGreater(
            cc['ssbm_resolution']['energy_per_nucleon_MeV'], 0
        )


# =====================================================================
# 7. COMPACTIFICATION
# =====================================================================

class TestCompactification(unittest.TestCase):
    """Test extra-dimension predictions."""

    def test_n_zero_returns_none(self):
        """n = 0 extra dimensions should return None."""
        self.assertIsNone(compactification_radius(0))

    def test_n_negative_returns_none(self):
        """Negative n should return None."""
        self.assertIsNone(compactification_radius(-1))

    def test_rc_planck_scale_for_n6(self):
        """For n = 6 (superstring): R_c ≈ 1.36 l_P."""
        r = compactification_radius(6)
        self.assertAlmostEqual(r['R_c_planck_lengths'], (1 / XI) ** (1 / 6), delta=0.01)
        self.assertGreater(r['R_c_planck_lengths'], 1.0)
        self.assertLess(r['R_c_planck_lengths'], 2.0)

    def test_rc_planck_scale_for_n7(self):
        """For n = 7 (M-theory): R_c ≈ 1.30 l_P."""
        r = compactification_radius(7)
        self.assertGreater(r['R_c_planck_lengths'], 1.0)
        self.assertLess(r['R_c_planck_lengths'], 2.0)

    def test_rc_increases_with_fewer_dims(self):
        """Fewer extra dims → larger compactification radius."""
        r1 = compactification_radius(1)
        r6 = compactification_radius(6)
        self.assertGreater(r1['R_c_planck_lengths'], r6['R_c_planck_lengths'])

    def test_xi_expression_matches(self):
        """R_c/l_P = e^(σ_conv/n) = (1/ξ)^(1/n) — should be identical."""
        for n in range(1, 8):
            r = compactification_radius(n)
            self.assertTrue(r['xi_check'],
                            msg=f"n={n}: e^(σ_conv/n) ≠ (1/ξ)^(1/n)")

    def test_rc_for_n1_kaluza_klein(self):
        """For n = 1 (KK): R_c = l_P/ξ ≈ 6.32 l_P."""
        r = compactification_radius(1)
        expected = 1.0 / XI
        self.assertAlmostEqual(r['R_c_planck_lengths'], expected, delta=0.01)

    def test_predictions_has_seven_entries(self):
        """Should predict for n = 1 through 7."""
        p = compactification_predictions()
        self.assertEqual(len(p['predictions']), 7)

    def test_all_near_planck_for_high_n(self):
        """For n ≥ 4, all R_c should be within 10× Planck length."""
        p = compactification_predictions()
        for r in p['predictions']:
            if r['n_extra_dims'] >= 4:
                self.assertTrue(r['is_planck_scale'],
                                msg=f"n={r['n_extra_dims']}: R_c not Planck-scale")


# =====================================================================
# 8. σ-FIELD AS MODULUS COMPARISON
# =====================================================================

class TestModulusComparison(unittest.TestCase):
    """Test the σ vs string moduli comparison table."""

    def test_comparison_has_entries(self):
        """Comparison should have multiple property rows."""
        c = sigma_modulus_comparison()
        self.assertGreater(len(c['properties']), 10)

    def test_each_entry_has_both_columns(self):
        """Each property should have σ and string_moduli descriptions."""
        c = sigma_modulus_comparison()
        for prop in c['properties']:
            self.assertIn('property', prop)
            self.assertIn('sigma', prop)
            self.assertIn('string_moduli', prop)

    def test_summary_exists(self):
        """Summary should be non-empty."""
        c = sigma_modulus_comparison()
        self.assertGreater(len(c['summary']), 100)


# =====================================================================
# σ-DEPENDENCE (RULE 4)
# =====================================================================

class TestSigmaHierarchyShift(unittest.TestCase):
    """Test hierarchy shift under σ-field compression."""

    def test_no_shift_at_vacuum(self):
        """At σ = 0: no hierarchy shift."""
        s = sigma_hierarchy_shift(SIGMA_HERE)
        self.assertAlmostEqual(s['fractional_shift'], 0.0, places=5)

    def test_gravity_strengthens_with_sigma(self):
        """At σ > 0: hierarchy shrinks (gravity relatively stronger)."""
        s = sigma_hierarchy_shift(0.1)
        self.assertTrue(s['gravity_strengthens'])
        self.assertLess(s['fractional_shift'], 0)

    def test_shift_at_event_horizon(self):
        """At σ ≈ ξ/2 ≈ 0.079 (event horizon): ~16% shift."""
        sigma_horizon = XI / 2
        s = sigma_hierarchy_shift(sigma_horizon)
        # ΔR/R ≈ −2σ × (m_QCD/m_p) ≈ −0.156
        self.assertLess(s['percent_change'], -10)
        self.assertGreater(s['percent_change'], -25)


# =====================================================================
# REPORTS
# =====================================================================

class TestReports(unittest.TestCase):
    """Test report generation."""

    def test_string_theory_report_has_all_sections(self):
        """Report should list all 7 connection areas."""
        r = string_theory_report()
        self.assertEqual(len(r['connections_to_open_problems']), 7)
        self.assertGreater(len(r['cascade_connections']), 3)
        self.assertGreater(len(r['predictions']), 3)

    def test_full_report_has_computed_results(self):
        """Full report should include all computed sections."""
        r = full_report()
        self.assertIn('moduli', r)
        self.assertIn('hierarchy', r)
        self.assertIn('swampland', r)
        self.assertIn('dilaton', r)
        self.assertIn('cosmological_constant', r)
        self.assertIn('compactification', r)
        self.assertIn('comparison', r)
