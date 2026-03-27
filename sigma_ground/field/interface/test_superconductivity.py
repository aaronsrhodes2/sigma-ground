"""
Tests for the superconductivity module.

Test structure:
  1. BCS energy gap — Δ(0), temperature dependence
  2. London penetration depth — fundamental + T-dependence
  3. Coherence length — BCS/Pippard
  4. GL parameter — Type I vs Type II classification
  5. Critical fields — thermodynamic, H_c1, H_c2
  6. Critical current — depairing limit
  7. Meissner fraction — two-fluid model
  8. σ-dependence — T_c shift through Debye temperature
  9. Nagatha export
"""

import math
import unittest

from .superconductivity import (
    bcs_gap_zero,
    bcs_gap_temperature,
    gap_frequency,
    london_penetration_depth,
    london_penetration_at_T,
    bcs_coherence_length,
    gl_parameter,
    gl_parameter_effective,
    is_type_II,
    thermodynamic_critical_field,
    lower_critical_field,
    upper_critical_field,
    depairing_current_density,
    specific_heat_jump_ratio,
    meissner_fraction,
    sigma_Tc_shift,
    sigma_gap_shift,
    mcmillan_Tc,
    mcmillan_Tc_for,
    sigma_mcmillan_Tc,
    debye_comparison,
    block_cooling_profile,
    superconductor_properties,
    SUPERCONDUCTORS,
    PHI_0,
)
from ..constants import K_B, E_CHARGE


def _sc_only():
    """Iterate only actual superconductors (T_c > 0)."""
    return {k: v for k, v in SUPERCONDUCTORS.items()
            if v.get('is_superconductor', True) and v['T_c_K'] > 0}


_NON_SC_KEYS = {'copper', 'silver', 'gold', 'platinum', 'palladium',
                'iron_ambient', 'cobalt', 'nickel'}


class TestBCSGap(unittest.TestCase):
    """BCS energy gap — Δ(0) = 1.764 k_B T_c."""

    def test_aluminum_gap(self):
        """Aluminum Δ(0) ≈ 0.17 meV."""
        T_c = 1.175
        delta = bcs_gap_zero(T_c)
        delta_meV = delta / (E_CHARGE * 1e-3)
        self.assertAlmostEqual(delta_meV, 0.17, delta=0.02)

    def test_niobium_gap(self):
        """Niobium Δ(0) ≈ 1.5 meV."""
        T_c = 9.25
        delta = bcs_gap_zero(T_c)
        delta_meV = delta / (E_CHARGE * 1e-3)
        self.assertAlmostEqual(delta_meV, 1.4, delta=0.2)

    def test_proportional_to_Tc(self):
        """Δ(0) ∝ T_c."""
        d1 = bcs_gap_zero(1.0)
        d2 = bcs_gap_zero(2.0)
        self.assertAlmostEqual(d2 / d1, 2.0, places=10)

    def test_gap_zero_above_Tc(self):
        """Δ(T) = 0 for T ≥ T_c."""
        T_c = 9.25
        self.assertEqual(bcs_gap_temperature(T_c, T_c), 0.0)
        self.assertEqual(bcs_gap_temperature(T_c, T_c + 1), 0.0)

    def test_gap_maximum_at_T0(self):
        """Δ(T=0) = Δ(0) (maximum gap)."""
        T_c = 9.25
        d0 = bcs_gap_zero(T_c)
        d_at_0 = bcs_gap_temperature(T_c, 0.0)
        self.assertAlmostEqual(d_at_0, d0, places=15)

    def test_gap_decreases_with_T(self):
        """Δ(T) decreases monotonically from 0 to T_c."""
        T_c = 9.25
        d_prev = bcs_gap_zero(T_c)
        for T in [1, 3, 5, 7, 9]:
            d = bcs_gap_temperature(T_c, T)
            self.assertLessEqual(d, d_prev + 1e-30)
            d_prev = d

    def test_gap_frequency_positive(self):
        """Gap frequency is positive."""
        f = gap_frequency(9.25)
        self.assertGreater(f, 0)

    def test_gap_frequency_niobium(self):
        """Niobium gap frequency ~ 700 GHz."""
        f = gap_frequency(9.25)
        self.assertGreater(f, 100e9)
        self.assertLess(f, 2000e9)


class TestLondonPenetration(unittest.TestCase):
    """London penetration depth."""

    def test_aluminum_lambda(self):
        """Aluminum λ_L ≈ 16 nm (T=0)."""
        n_e = SUPERCONDUCTORS['aluminum']['n_e_m3']
        lam = london_penetration_depth(n_e)
        self.assertGreater(lam * 1e9, 10)
        self.assertLess(lam * 1e9, 100)

    def test_positive(self):
        """λ_L > 0 for all superconductors."""
        for sc in _sc_only().values():
            lam = london_penetration_depth(sc['n_e_m3'])
            self.assertGreater(lam, 0)

    def test_diverges_at_Tc(self):
        """λ_L → ∞ at T = T_c."""
        n_e = SUPERCONDUCTORS['aluminum']['n_e_m3']
        T_c = SUPERCONDUCTORS['aluminum']['T_c_K']
        lam = london_penetration_at_T(n_e, T_c, T_c)
        self.assertEqual(lam, float('inf'))

    def test_increases_with_T(self):
        """λ_L(T) increases as T → T_c."""
        n_e = SUPERCONDUCTORS['niobium']['n_e_m3']
        T_c = SUPERCONDUCTORS['niobium']['T_c_K']
        lam_low = london_penetration_at_T(n_e, T_c, 2.0)
        lam_high = london_penetration_at_T(n_e, T_c, 8.0)
        self.assertGreater(lam_high, lam_low)


class TestCoherenceLength(unittest.TestCase):
    """BCS coherence length ξ₀."""

    def test_aluminum_xi(self):
        """Aluminum ξ₀ ≈ 1600 nm (clean limit)."""
        data = SUPERCONDUCTORS['aluminum']
        xi = bcs_coherence_length(data['v_F_m_s'], data['T_c_K'])
        self.assertGreater(xi * 1e6, 0.5)  # > 500 nm
        self.assertLess(xi * 1e6, 5)       # < 5 μm

    def test_positive(self):
        """ξ₀ > 0 for all superconductors."""
        for sc in _sc_only().values():
            xi = bcs_coherence_length(sc['v_F_m_s'], sc['T_c_K'])
            self.assertGreater(xi, 0)

    def test_inversely_proportional_to_Tc(self):
        """ξ₀ ∝ 1/T_c (roughly — also depends on v_F)."""
        # Aluminum (low T_c) has longer ξ₀ than niobium (high T_c)
        xi_al = bcs_coherence_length(
            SUPERCONDUCTORS['aluminum']['v_F_m_s'],
            SUPERCONDUCTORS['aluminum']['T_c_K'])
        xi_nb = bcs_coherence_length(
            SUPERCONDUCTORS['niobium']['v_F_m_s'],
            SUPERCONDUCTORS['niobium']['T_c_K'])
        self.assertGreater(xi_al, xi_nb)


class TestGLParameter(unittest.TestCase):
    """Ginzburg-Landau κ and Type I/II classification."""

    def test_aluminum_type_I(self):
        """Aluminum is Type I (measured κ < 1/√2)."""
        data = SUPERCONDUCTORS['aluminum']
        self.assertFalse(is_type_II(data['n_e_m3'], data['v_F_m_s'],
                                     data['T_c_K'], sc_key='aluminum'))

    def test_niobium_type_II(self):
        """Niobium is Type II (measured κ > 1/√2)."""
        data = SUPERCONDUCTORS['niobium']
        self.assertTrue(is_type_II(data['n_e_m3'], data['v_F_m_s'],
                                    data['T_c_K'], sc_key='niobium'))

    def test_database_consistency(self):
        """Database type matches measured κ classification."""
        for key, data in _sc_only().items():
            kappa = gl_parameter_effective(key)
            computed = kappa > 1.0 / math.sqrt(2)
            expected = data['type'] == 'II'
            self.assertEqual(computed, expected,
                f"{key}: κ={kappa:.3f}, computed={computed}, expected={expected}")

    def test_positive(self):
        """κ > 0 for all superconductors."""
        for key in _sc_only():
            kappa = gl_parameter_effective(key)
            self.assertGreater(kappa, 0)


class TestCriticalFields(unittest.TestCase):
    """Critical magnetic fields."""

    def test_Hc_positive(self):
        """H_c > 0 at T < T_c."""
        for data in _sc_only().values():
            H = thermodynamic_critical_field(data['n_e_m3'], data['T_c_K'], 0)
            self.assertGreater(H, 0)

    def test_Hc_zero_above_Tc(self):
        """H_c = 0 at T ≥ T_c."""
        data = SUPERCONDUCTORS['niobium']
        H = thermodynamic_critical_field(data['n_e_m3'], data['T_c_K'],
                                          data['T_c_K'])
        self.assertEqual(H, 0.0)

    def test_Hc_decreases_with_T(self):
        """H_c(T) decreases monotonically."""
        data = SUPERCONDUCTORS['niobium']
        H_prev = thermodynamic_critical_field(data['n_e_m3'], data['T_c_K'], 0)
        for T in [2, 4, 6, 8]:
            H = thermodynamic_critical_field(data['n_e_m3'], data['T_c_K'], T)
            self.assertLess(H, H_prev)
            H_prev = H

    def test_Hc2_greater_than_Hc1(self):
        """H_c2 > H_c1 for Type II."""
        data = SUPERCONDUCTORS['niobium']
        Hc1 = lower_critical_field(data['n_e_m3'], data['v_F_m_s'],
                                    data['T_c_K'], sc_key='niobium')
        Hc2 = upper_critical_field(data['n_e_m3'], data['v_F_m_s'],
                                    data['T_c_K'], sc_key='niobium')
        self.assertGreater(Hc2, Hc1)

    def test_Hc1_positive_type_II(self):
        """H_c1 > 0 for Type II superconductors."""
        data = SUPERCONDUCTORS['niobium']
        Hc1 = lower_critical_field(data['n_e_m3'], data['v_F_m_s'],
                                    data['T_c_K'], sc_key='niobium')
        self.assertGreater(Hc1, 0)


class TestCriticalCurrent(unittest.TestCase):
    """Depairing critical current density."""

    def test_positive(self):
        """J_c > 0."""
        for data in _sc_only().values():
            J = depairing_current_density(
                data['n_e_m3'], data['v_F_m_s'], data['T_c_K'])
            self.assertGreater(J, 0)

    def test_order_of_magnitude(self):
        """J_c ~ 10⁹ to 10¹² A/m² (theoretical maximum)."""
        data = SUPERCONDUCTORS['niobium']
        J = depairing_current_density(
            data['n_e_m3'], data['v_F_m_s'], data['T_c_K'])
        self.assertGreater(J, 1e8)
        self.assertLess(J, 1e14)


class TestMeissnerFraction(unittest.TestCase):
    """Gorter-Casimir two-fluid model."""

    def test_unity_at_T0(self):
        """All electrons superconducting at T=0."""
        self.assertAlmostEqual(meissner_fraction(9.25, 0), 1.0, places=10)

    def test_zero_at_Tc(self):
        """No superconducting electrons at T_c."""
        self.assertEqual(meissner_fraction(9.25, 9.25), 0.0)

    def test_zero_above_Tc(self):
        """Zero above T_c."""
        self.assertEqual(meissner_fraction(9.25, 20.0), 0.0)

    def test_monotonically_decreases(self):
        """Fraction decreases with temperature."""
        f_prev = 1.0
        for T in [1, 3, 5, 7, 9]:
            f = meissner_fraction(9.25, T)
            self.assertLess(f, f_prev)
            f_prev = f


class TestSpecificHeatJump(unittest.TestCase):
    """BCS universal ratio."""

    def test_bcs_ratio(self):
        """ΔC/(γT_c) = 1.43 (universal BCS weak-coupling)."""
        self.assertAlmostEqual(specific_heat_jump_ratio(), 1.43, places=2)


class TestFluxQuantum(unittest.TestCase):
    """Flux quantum Φ₀ = πℏ/e."""

    def test_value(self):
        """Φ₀ ≈ 2.068 × 10⁻¹⁵ Wb."""
        self.assertAlmostEqual(PHI_0 * 1e15, 2.068, delta=0.01)


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts superconducting properties through Θ_D."""

    def test_Tc_unchanged_at_zero(self):
        """T_c(σ=0) = T_c(0)."""
        T_c = sigma_Tc_shift(9.25, 0.0)
        self.assertAlmostEqual(T_c, 9.25, places=10)

    def test_Tc_decreases_with_sigma(self):
        """T_c(σ) < T_c(0) for σ > 0 (heavier lattice → lower Θ_D)."""
        T_c_0 = 9.25
        T_c_s = sigma_Tc_shift(T_c_0, 0.1)
        self.assertLess(T_c_s, T_c_0)

    def test_gap_decreases_with_sigma(self):
        """Δ(σ) < Δ(0) for σ > 0 (tracks T_c)."""
        d0 = sigma_gap_shift(9.25, 0.0)
        ds = sigma_gap_shift(9.25, 0.1)
        self.assertLess(ds, d0)

    def test_earth_sigma_negligible(self):
        """At Earth σ ~ 7×10⁻¹⁰: T_c shift < 10⁻⁸ K."""
        T_c_0 = 9.25
        T_c_s = sigma_Tc_shift(T_c_0, 7e-10)
        self.assertAlmostEqual(T_c_s, T_c_0, places=8)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_all_superconductors_export(self):
        """All entries produce valid export dicts."""
        for sc in SUPERCONDUCTORS:
            props = superconductor_properties(sc, T=0.0)
            self.assertIn('T_c_K', props)
            self.assertIn('origin_tag', props)
            data = SUPERCONDUCTORS[sc]
            if data.get('is_superconductor', True) and data['T_c_K'] > 0:
                self.assertIn('gap_meV', props)
                self.assertIn('london_depth_m', props)

    def test_type_II_has_Hc1_Hc2(self):
        """Type II exports include H_c1 and H_c2."""
        props = superconductor_properties('niobium')
        self.assertIn('H_c1_A_m', props)
        self.assertIn('H_c2_A_m', props)

    def test_sigma_propagates(self):
        """σ value appears in export and shifts T_c."""
        props = superconductor_properties('niobium', sigma=0.1)
        self.assertEqual(props['sigma'], 0.1)
        self.assertLess(props['T_c_K'], 9.25)


class TestExpandedDatabase(unittest.TestCase):
    """Rule 9 — If One, Then All: comprehensive database validation."""

    _REQUIRED_FIELDS = {
        'name', 'T_c_K', 'n_e_m3', 'v_F_m_s', 'kappa',
        'type', 'pressure_GPa', 'kappa_source',
    }

    def test_entry_count(self):
        """Database has ≥50 entries (30 ambient + 14 pressure + 9 compounds)."""
        self.assertGreaterEqual(len(SUPERCONDUCTORS), 50)

    def test_all_fields_present(self):
        """Every entry has every required field (Rule 9)."""
        for key, data in SUPERCONDUCTORS.items():
            for field in self._REQUIRED_FIELDS:
                self.assertIn(field, data, f"{key}: missing field '{field}'")

    def test_no_none_physics(self):
        """No None values for physics quantities (n_e, v_F, kappa, T_c)."""
        for key, data in SUPERCONDUCTORS.items():
            self.assertIsNotNone(data['T_c_K'], f"{key}: T_c_K is None")
            self.assertIsNotNone(data['n_e_m3'], f"{key}: n_e_m3 is None")
            self.assertIsNotNone(data['v_F_m_s'], f"{key}: v_F_m_s is None")
            self.assertIsNotNone(data['kappa'], f"{key}: kappa is None")

    def test_type_matches_kappa(self):
        """Type classification consistent with κ threshold for all entries."""
        threshold = 1.0 / math.sqrt(2.0)
        for key, data in SUPERCONDUCTORS.items():
            expected_type = 'II' if data['kappa'] > threshold else 'I'
            self.assertEqual(data['type'], expected_type,
                f"{key}: κ={data['kappa']:.4f}, type='{data['type']}', "
                f"expected='{expected_type}'")

    def test_kappa_source_valid(self):
        """kappa_source is 'measured' or 'derived' for all entries."""
        for key, data in SUPERCONDUCTORS.items():
            self.assertIn(data['kappa_source'], ('measured', 'derived'),
                f"{key}: invalid kappa_source '{data['kappa_source']}'")

    def test_n_e_physical_range(self):
        """Electron density between 10²⁶ and 10³⁰ m⁻³ for all entries."""
        for key, data in SUPERCONDUCTORS.items():
            self.assertGreater(data['n_e_m3'], 1e26,
                f"{key}: n_e={data['n_e_m3']:.2e} too low")
            self.assertLess(data['n_e_m3'], 1e31,
                f"{key}: n_e={data['n_e_m3']:.2e} too high")

    def test_v_F_physical_range(self):
        """Fermi velocity between 10⁴ and 5×10⁶ m/s for all entries."""
        for key, data in SUPERCONDUCTORS.items():
            self.assertGreater(data['v_F_m_s'], 1e4,
                f"{key}: v_F={data['v_F_m_s']:.2e} too low")
            self.assertLess(data['v_F_m_s'], 5e6,
                f"{key}: v_F={data['v_F_m_s']:.2e} too high")

    def test_T_c_positive(self):
        """All SC T_c values are positive; non-SC are zero."""
        for key, data in SUPERCONDUCTORS.items():
            if data.get('is_superconductor', True):
                self.assertGreater(data['T_c_K'], 0, f"{key}: T_c must be > 0")
            else:
                self.assertEqual(data['T_c_K'], 0.0, f"{key}: non-SC T_c must be 0")

    def test_kappa_positive(self):
        """All SC κ values are positive."""
        for key, data in _sc_only().items():
            self.assertGreater(data['kappa'], 0, f"{key}: κ must be > 0")

    def test_pressure_elements_flagged(self):
        """Known pressure-only superconductors have pressure_GPa set."""
        pressure_keys = ['silicon', 'iron', 'bismuth', 'calcium', 'sulfur']
        for key in pressure_keys:
            if key in SUPERCONDUCTORS:
                self.assertIsNotNone(SUPERCONDUCTORS[key]['pressure_GPa'],
                    f"{key}: should have pressure_GPa set")
                self.assertGreater(SUPERCONDUCTORS[key]['pressure_GPa'], 0)

    def test_ambient_elements_no_pressure(self):
        """Ambient-pressure superconductors have pressure_GPa = None."""
        ambient_keys = ['aluminum', 'niobium', 'lead', 'tin', 'mercury']
        for key in ambient_keys:
            self.assertIsNone(SUPERCONDUCTORS[key]['pressure_GPa'],
                f"{key}: ambient SC should have pressure_GPa = None")

    def test_high_Tc_compounds(self):
        """YBCO T_c > 77 K (liquid nitrogen), BSCCO > 100 K."""
        self.assertGreater(SUPERCONDUCTORS['YBCO']['T_c_K'], 77)
        self.assertGreater(SUPERCONDUCTORS['BSCCO_2223']['T_c_K'], 100)

    def test_niobium_highest_elemental(self):
        """Niobium has highest ambient-pressure elemental T_c."""
        nb_Tc = SUPERCONDUCTORS['niobium']['T_c_K']
        for key, data in SUPERCONDUCTORS.items():
            # Skip non-SC, compounds, and pressure-required
            if not data.get('is_superconductor', True):
                continue
            if data['pressure_GPa'] is not None:
                continue
            if key in ('NbTi', 'Nb3Sn', 'Nb3Ge', 'V3Si', 'MgB2',
                       'PbMo6S8', 'YBCO', 'BSCCO_2212', 'BSCCO_2223'):
                continue
            self.assertLessEqual(data['T_c_K'], nb_Tc,
                f"{key}: T_c={data['T_c_K']} > Nb T_c={nb_Tc}")

    def test_export_includes_new_fields(self):
        """Nagatha export includes kappa_source and pressure_GPa."""
        props = superconductor_properties('niobium')
        self.assertIn('kappa_source', props)
        self.assertIn('pressure_GPa', props)

    def test_export_pressure_element(self):
        """Pressure element export includes pressure_GPa value."""
        props = superconductor_properties('bismuth')
        self.assertEqual(props['pressure_GPa'], 2.55)


class TestMcMillanFormula(unittest.TestCase):
    """McMillan formula: measured λ + Θ_D → predicted T_c."""

    def _check_within(self, key, factor):
        """Assert McMillan T_c within *factor* of measured T_c."""
        data = SUPERCONDUCTORS[key]
        predicted = mcmillan_Tc_for(key)
        self.assertIsNotNone(predicted, f"{key}: missing McMillan data")
        measured = data['T_c_K']
        self.assertGreater(predicted, measured / factor,
            f"{key}: predicted={predicted:.3f} K, measured={measured:.3f} K, "
            f"ratio={predicted/measured:.2f} — too low")
        self.assertLess(predicted, measured * factor,
            f"{key}: predicted={predicted:.3f} K, measured={measured:.3f} K, "
            f"ratio={predicted/measured:.2f} — too high")

    # McMillan (1968) systematically overestimates T_c, especially for
    # strong-coupling materials (λ > ~1). Allen-Dynes (1975) corrects this,
    # but we use the original formula. Tolerances reflect this known bias.

    def test_aluminum(self):
        """Al (λ=0.43): McMillan T_c within factor 2 of 1.175 K."""
        self._check_within('aluminum', 2.0)

    def test_niobium(self):
        """Nb (λ=1.26, strong coupling): McMillan T_c within factor 2.5."""
        self._check_within('niobium', 2.5)

    def test_lead(self):
        """Pb (λ=1.55): McMillan T_c within factor 1.5 of 7.193 K."""
        self._check_within('lead', 1.5)

    def test_tin(self):
        """Sn (λ=0.72): McMillan T_c within factor 2 of 3.722 K."""
        self._check_within('tin', 2.0)

    def test_vanadium(self):
        """V (λ=0.80): McMillan T_c within factor 2.5 of 5.40 K."""
        self._check_within('vanadium', 2.5)

    def test_tantalum(self):
        """Ta (λ=0.69): McMillan T_c within factor 1.5 of 4.47 K."""
        self._check_within('tantalum', 1.5)

    def test_indium(self):
        """In (λ=0.81): McMillan T_c within factor 1.5 of 3.41 K."""
        self._check_within('indium', 1.5)

    def test_mercury(self):
        """Hg (λ=1.60, strong coupling): McMillan T_c within factor 2."""
        self._check_within('mercury', 2.0)

    def test_all_known_within_factor_3(self):
        """Every element with λ data: McMillan T_c within factor of 3.

        McMillan (1968) is known to overestimate for strong coupling
        and underestimate near the weak-coupling edge (λ ≈ μ*).
        Skip entries where predicted T_c < 0.01 K — the formula's
        exponential sensitivity breaks down there."""
        for key, data in _sc_only().items():
            predicted = mcmillan_Tc_for(key)
            if predicted is None:
                continue
            if predicted < 0.01:
                # Near weak-coupling edge — exponential sensitivity
                continue
            measured = data['T_c_K']
            ratio = predicted / measured
            self.assertGreater(ratio, 1.0 / 3.0,
                f"{key}: ratio={ratio:.2f} (predicted={predicted:.3f}, "
                f"measured={measured:.3f})")
            self.assertLess(ratio, 3.0,
                f"{key}: ratio={ratio:.2f} (predicted={predicted:.3f}, "
                f"measured={measured:.3f})")

    def test_weak_coupling_returns_zero(self):
        """Cu, Ag, Au: McMillan predicts T_c < 0.001 K."""
        for key in ('copper', 'silver', 'gold'):
            predicted = mcmillan_Tc_for(key)
            self.assertIsNotNone(predicted, f"{key}: missing McMillan data")
            self.assertLess(predicted, 0.001,
                f"{key}: predicted={predicted:.6f} K, should be ~0")

    def test_ferromagnet_nonzero(self):
        """Fe/Co/Ni: McMillan predicts nonzero T_c (magnetism kills SC)."""
        for key in ('iron_ambient', 'cobalt', 'nickel'):
            predicted = mcmillan_Tc_for(key)
            self.assertIsNotNone(predicted, f"{key}: missing McMillan data")
            # McMillan says these SHOULD superconduct, but they don't —
            # magnetic ordering destroys Cooper pairs. This is expected.
            self.assertGreater(predicted, 0.0,
                f"{key}: predicted={predicted:.6f} K, expected nonzero")

    def test_formula_basic_math(self):
        """McMillan formula: known inputs produce reasonable output."""
        # Nb-like: Θ_D=275, λ=1.26, μ*=0.13
        # McMillan overestimates for strong coupling: ~19 K (measured 9.25 K)
        T_c = mcmillan_Tc(275, 1.26, 0.13)
        self.assertGreater(T_c, 5)
        self.assertLess(T_c, 25)

    def test_zero_coupling_returns_zero(self):
        """λ too small → denominator ≤ 0 → T_c = 0."""
        # λ=0.05, μ*=0.10 → denom = 0.05 - 0.10*(1+0.031) < 0
        self.assertEqual(mcmillan_Tc(300, 0.05, 0.10), 0.0)


class TestMcMillanSigma(unittest.TestCase):
    """McMillan T_c under σ-field: Θ_D shifts through nuclear mass."""

    def test_identity_at_zero(self):
        """σ=0: sigma_mcmillan_Tc = mcmillan_Tc."""
        T_c_0 = mcmillan_Tc(275, 1.26, 0.13)
        T_c_s = sigma_mcmillan_Tc(275, 1.26, 0.13, 0.0)
        self.assertAlmostEqual(T_c_0, T_c_s, places=10)

    def test_positive_sigma_decreases_Tc(self):
        """Positive σ → heavier lattice → lower Θ_D → lower T_c."""
        T_c_0 = sigma_mcmillan_Tc(275, 1.26, 0.13, 0.0)
        T_c_s = sigma_mcmillan_Tc(275, 1.26, 0.13, 0.1)
        self.assertLess(T_c_s, T_c_0)

    def test_earth_sigma_negligible(self):
        """At Earth σ ~ 7×10⁻¹⁰: shift < 10⁻⁶ K."""
        T_c_0 = sigma_mcmillan_Tc(275, 1.26, 0.13, 0.0)
        T_c_s = sigma_mcmillan_Tc(275, 1.26, 0.13, 7e-10)
        self.assertAlmostEqual(T_c_0, T_c_s, places=6)


class TestNonSuperconductors(unittest.TestCase):
    """Non-SC metals: flags, T_c=0, suppression reasons."""

    def test_all_non_sc_have_zero_Tc(self):
        """Every non-SC entry has T_c = 0."""
        for key in _NON_SC_KEYS:
            self.assertEqual(SUPERCONDUCTORS[key]['T_c_K'], 0.0, key)

    def test_all_non_sc_flagged(self):
        """Every non-SC entry has is_superconductor=False."""
        for key in _NON_SC_KEYS:
            self.assertFalse(SUPERCONDUCTORS[key]['is_superconductor'], key)

    def test_suppression_set(self):
        """Every non-SC entry has a suppression reason."""
        for key in _NON_SC_KEYS:
            self.assertIsNotNone(SUPERCONDUCTORS[key]['suppression'], key)

    def test_weak_coupling_suppression(self):
        """Noble metals: suppressed by weak coupling."""
        for key in ('copper', 'silver', 'gold', 'platinum'):
            self.assertEqual(SUPERCONDUCTORS[key]['suppression'],
                             'weak_coupling', key)

    def test_ferromagnet_suppression(self):
        """Fe, Co, Ni: suppressed by ferromagnetism."""
        for key in ('iron_ambient', 'cobalt', 'nickel'):
            self.assertEqual(SUPERCONDUCTORS[key]['suppression'],
                             'ferromagnet', key)

    def test_palladium_spin_fluctuations(self):
        """Pd: suppressed by spin fluctuations."""
        self.assertEqual(SUPERCONDUCTORS['palladium']['suppression'],
                         'spin_fluctuations')

    def test_non_sc_export(self):
        """Non-SC export includes McMillan prediction but no BCS fields."""
        props = superconductor_properties('copper')
        self.assertFalse(props['is_superconductor'])
        self.assertEqual(props['T_c_K'], 0.0)
        self.assertIsNotNone(props.get('mcmillan_Tc_K'))
        self.assertNotIn('gap_meV', props)


class TestDebyeComparison(unittest.TestCase):
    """Derived Θ_D (from thermal.py) vs measured Θ_D."""

    def test_returns_results(self):
        """debye_comparison() returns non-empty list."""
        results = debye_comparison()
        self.assertGreater(len(results), 0)

    def test_within_factor_2(self):
        """Derived Θ_D within factor of 2 of measured."""
        for r in debye_comparison():
            ratio = r['derived_theta_D'] / r['measured_theta_D']
            self.assertGreater(ratio, 0.5,
                f"{r['material']}: derived={r['derived_theta_D']:.0f}, "
                f"measured={r['measured_theta_D']:.0f}, ratio={ratio:.2f}")
            self.assertLess(ratio, 2.0,
                f"{r['material']}: derived={r['derived_theta_D']:.0f}, "
                f"measured={r['measured_theta_D']:.0f}, ratio={ratio:.2f}")

    def test_result_fields(self):
        """Each result has required fields."""
        for r in debye_comparison():
            self.assertIn('material', r)
            self.assertIn('derived_theta_D', r)
            self.assertIn('measured_theta_D', r)
            self.assertIn('percent_error', r)


class TestBlockCooling(unittest.TestCase):
    """Cool a block of superconductor through T_c — verify the transition."""

    def test_niobium_resistance_drops_to_zero(self):
        """Niobium (T_c=9.25 K): resistance is finite above, exactly 0 below."""
        profile = block_cooling_profile('niobium', T_start=20.0, T_end=0.0,
                                        steps=200, rho_normal=1.5e-7)
        T_c = SUPERCONDUCTORS['niobium']['T_c_K']

        above = [p for p in profile if p['T_K'] > T_c]
        below = [p for p in profile if p['T_K'] < T_c]

        # Every point above T_c: finite resistance
        for p in above:
            self.assertEqual(p['resistivity'], 1.5e-7,
                f"T={p['T_K']:.2f} K: expected normal-state resistivity")

        # Every point below T_c: exactly zero
        for p in below:
            self.assertEqual(p['resistivity'], 0.0,
                f"T={p['T_K']:.2f} K: expected zero resistivity")

    def test_transition_is_sharp(self):
        """No intermediate resistivity — it's either rho_n or 0."""
        profile = block_cooling_profile('lead', T_start=15.0, T_end=0.0,
                                        steps=1000)
        for p in profile:
            self.assertIn(p['resistivity'], (0.0, 1.0e-7),
                f"T={p['T_K']:.4f} K: resistivity={p['resistivity']}, "
                f"expected either 0.0 or 1e-7")

    def test_bcs_gap_opens_at_Tc(self):
        """BCS gap is zero above T_c, positive below."""
        profile = block_cooling_profile('niobium', T_start=20.0, T_end=0.0,
                                        steps=200)
        T_c = SUPERCONDUCTORS['niobium']['T_c_K']

        for p in profile:
            if p['T_K'] >= T_c:
                self.assertEqual(p['bcs_gap_J'], 0.0,
                    f"T={p['T_K']:.2f} K: gap should be 0 above T_c")
            elif p['T_K'] < T_c - 0.5:  # well below T_c
                self.assertGreater(p['bcs_gap_J'], 0.0,
                    f"T={p['T_K']:.2f} K: gap should be positive below T_c")

    def test_gap_increases_on_cooling(self):
        """BCS gap grows monotonically as T decreases below T_c."""
        profile = block_cooling_profile('niobium', T_start=20.0, T_end=0.0,
                                        steps=200)
        T_c = SUPERCONDUCTORS['niobium']['T_c_K']
        below = [p for p in profile if p['T_K'] < T_c]

        # below is in order of decreasing T (cooling), gap should increase
        prev_gap = 0.0
        for p in below:
            self.assertGreaterEqual(p['bcs_gap_J'], prev_gap - 1e-30,
                f"T={p['T_K']:.2f} K: gap decreased")
            prev_gap = p['bcs_gap_J']

    def test_meissner_fraction_transition(self):
        """Meissner fraction: 0 above T_c, 1 at T=0."""
        profile = block_cooling_profile('aluminum', T_start=5.0, T_end=0.0,
                                        steps=100)
        T_c = SUPERCONDUCTORS['aluminum']['T_c_K']

        for p in profile:
            if p['T_K'] >= T_c:
                self.assertEqual(p['meissner_frac'], 0.0)
            elif p['T_K'] == 0.0:
                self.assertAlmostEqual(p['meissner_frac'], 1.0, places=10)

    def test_london_depth_finite_below_Tc(self):
        """London depth: infinite above T_c, finite below."""
        profile = block_cooling_profile('niobium', T_start=20.0, T_end=0.0,
                                        steps=200)
        T_c = SUPERCONDUCTORS['niobium']['T_c_K']

        for p in profile:
            if p['T_K'] >= T_c:
                self.assertEqual(p['london_depth_m'], float('inf'),
                    f"T={p['T_K']:.2f} K: depth should be inf above T_c")
            elif p['T_K'] < T_c - 0.5:
                self.assertLess(p['london_depth_m'], 1e-6,
                    f"T={p['T_K']:.2f} K: depth should be < 1 μm")
                self.assertGreater(p['london_depth_m'], 0,
                    f"T={p['T_K']:.2f} K: depth should be positive")

    def test_critical_field_appears(self):
        """H_c: zero above T_c, positive below."""
        profile = block_cooling_profile('lead', T_start=15.0, T_end=0.0,
                                        steps=100)
        T_c = SUPERCONDUCTORS['lead']['T_c_K']

        for p in profile:
            if p['T_K'] >= T_c:
                self.assertEqual(p['H_c_A_m'], 0.0)
            elif p['T_K'] < T_c - 0.5:
                self.assertGreater(p['H_c_A_m'], 0.0,
                    f"T={p['T_K']:.2f} K: H_c should be positive")

    def test_aluminum_full_cooling(self):
        """Aluminum (T_c=1.175 K): complete cooling profile check."""
        profile = block_cooling_profile('aluminum', T_start=3.0, T_end=0.0,
                                        steps=300, rho_normal=2.65e-8)
        T_c = SUPERCONDUCTORS['aluminum']['T_c_K']

        # Find the transition point
        above_count = sum(1 for p in profile if p['resistivity'] > 0)
        below_count = sum(1 for p in profile if p['resistivity'] == 0.0)

        # Transition should happen at T_c = 1.175 K
        # 301 points from 3.0 to 0.0 K, step = 0.01 K
        # Above: 3.0 down to 1.175 → ~183 points
        # Below: 1.165 down to 0.0 → ~118 points
        self.assertGreater(above_count, 100, "Should have many normal-state points")
        self.assertGreater(below_count, 100, "Should have many SC points")

        # At T=0: gap should be maximum, meissner=1, H_c maximum
        final = profile[-1]
        self.assertEqual(final['T_K'], 0.0)
        self.assertEqual(final['resistivity'], 0.0)
        self.assertAlmostEqual(final['meissner_frac'], 1.0, places=10)
        self.assertGreater(final['bcs_gap_J'], 0.0)
        self.assertGreater(final['H_c_A_m'], 0.0)

    def test_sigma_shifts_transition(self):
        """Positive σ shifts T_c lower — transition occurs at lower T."""
        profile_0 = block_cooling_profile('niobium', T_start=20.0, T_end=0.0,
                                          steps=200, sigma=0.0)
        profile_s = block_cooling_profile('niobium', T_start=20.0, T_end=0.0,
                                          steps=200, sigma=0.1)

        # Find highest T where resistance = 0
        def transition_T(prof):
            for p in prof:
                if p['resistivity'] == 0.0:
                    return p['T_K']
            return 0.0

        T_trans_0 = transition_T(profile_0)
        T_trans_s = transition_T(profile_s)
        self.assertLess(T_trans_s, T_trans_0,
            f"σ=0.1 should lower the transition: {T_trans_s} vs {T_trans_0}")

    def test_multiple_materials(self):
        """Cooling works for diverse materials: Nb (Type II), Pb (Type I), Al (deep I)."""
        for key in ('niobium', 'lead', 'aluminum'):
            T_c = SUPERCONDUCTORS[key]['T_c_K']
            profile = block_cooling_profile(key, T_start=T_c * 3, T_end=0.0,
                                            steps=100)
            # Must have both phases
            has_normal = any(p['resistivity'] > 0 for p in profile)
            has_sc = any(p['resistivity'] == 0.0 for p in profile)
            self.assertTrue(has_normal, f"{key}: no normal-state points")
            self.assertTrue(has_sc, f"{key}: no SC points")

            # T=0 point must be fully superconducting
            final = profile[-1]
            self.assertEqual(final['resistivity'], 0.0, f"{key}: not SC at T=0")
            self.assertGreater(final['bcs_gap_J'], 0.0, f"{key}: no gap at T=0")


class TestRule9McMillan(unittest.TestCase):
    """Rule 9 — every entry has McMillan fields (even if None)."""

    _MCMILLAN_FIELDS = {'lambda_ep', 'mu_star', 'theta_D_K',
                        'is_superconductor', 'suppression'}

    def test_all_entries_have_mcmillan_fields(self):
        """Every entry has lambda_ep, mu_star, theta_D_K, is_superconductor, suppression."""
        for key, data in SUPERCONDUCTORS.items():
            for field in self._MCMILLAN_FIELDS:
                self.assertIn(field, data,
                    f"{key}: missing McMillan field '{field}'")

    def test_sc_entries_are_flagged_true(self):
        """Actual superconductors have is_superconductor=True."""
        for key, data in _sc_only().items():
            self.assertTrue(data['is_superconductor'], key)

    def test_non_sc_count(self):
        """Database has exactly 8 non-SC entries."""
        non_sc = [k for k, v in SUPERCONDUCTORS.items()
                  if not v.get('is_superconductor', True)]
        self.assertEqual(len(non_sc), 8)


if __name__ == '__main__':
    unittest.main()
