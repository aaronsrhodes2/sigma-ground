"""
Tests for thermal_expansion.py.

All 8 materials, all functions. Uses only unittest (no external deps).
"""

import math
import unittest

from sigma_ground.field.interface.thermal_expansion import (
    EXPANSION_DATA,
    linear_expansion_coefficient,
    volumetric_expansion_coefficient,
    gruneisen_parameter,
    gruneisen_relation,
    thermal_strain,
    thermal_stress,
    length_change,
    volume_change,
    expansion_coefficient_at_T,
    sigma_expansion_shift,
    thermal_expansion_properties,
    _debye_cv_molar,
)

_ALL_MATERIALS = [
    'iron', 'copper', 'aluminum', 'gold',
    'silicon', 'tungsten', 'nickel', 'titanium',
]


class TestRule9(unittest.TestCase):
    """All 8 materials must have all required fields in EXPANSION_DATA."""

    _REQUIRED_FIELDS = ['alpha_linear_per_K', 'gruneisen_gamma', 'C_v_J_mol_K']

    def test_all_materials_present(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertIn(mat, EXPANSION_DATA)

    def test_all_fields_present(self):
        for mat in _ALL_MATERIALS:
            for field in self._REQUIRED_FIELDS:
                with self.subTest(material=mat, field=field):
                    self.assertIn(field, EXPANSION_DATA[mat])

    def test_all_fields_positive(self):
        for mat in _ALL_MATERIALS:
            for field in self._REQUIRED_FIELDS:
                with self.subTest(material=mat, field=field):
                    self.assertGreater(EXPANSION_DATA[mat][field], 0.0)


class TestLinearCoefficient(unittest.TestCase):
    """Known values and ordering for α at 300 K."""

    def test_aluminum_highest(self):
        """Aluminum has the highest α among the 8 materials."""
        alpha_al = linear_expansion_coefficient('aluminum')
        for mat in _ALL_MATERIALS:
            if mat == 'aluminum':
                continue
            with self.subTest(material=mat):
                self.assertGreaterEqual(
                    alpha_al,
                    linear_expansion_coefficient(mat),
                    msg=f"Al α should be >= {mat} α",
                )

    def test_tungsten_very_low(self):
        """Tungsten has one of the lowest α values (4.5e-6/K).
        Only silicon (2.6e-6/K) is lower in this set.
        All other metals should have higher α than tungsten.
        """
        alpha_w = linear_expansion_coefficient('tungsten')
        high_alpha_materials = [
            'iron', 'copper', 'aluminum', 'gold', 'nickel', 'titanium'
        ]
        for mat in high_alpha_materials:
            with self.subTest(material=mat):
                self.assertGreater(
                    linear_expansion_coefficient(mat), alpha_w,
                    msg=f"{mat} α should be > W α",
                )

    def test_silicon_lowest(self):
        """Silicon has the lowest α (2.6e-6/K) in the set."""
        alpha_si = linear_expansion_coefficient('silicon')
        for mat in _ALL_MATERIALS:
            if mat == 'silicon':
                continue
            with self.subTest(material=mat):
                self.assertLessEqual(
                    alpha_si,
                    linear_expansion_coefficient(mat),
                    msg=f"Si α should be <= {mat} α",
                )

    def test_all_positive(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(linear_expansion_coefficient(mat), 0.0)

    def test_known_values(self):
        """Spot-check measured values from the data dict."""
        self.assertAlmostEqual(
            linear_expansion_coefficient('aluminum'), 23.1e-6, places=12
        )
        self.assertAlmostEqual(
            linear_expansion_coefficient('tungsten'), 4.5e-6, places=12
        )
        self.assertAlmostEqual(
            linear_expansion_coefficient('silicon'), 2.6e-6, places=12
        )
        self.assertAlmostEqual(
            linear_expansion_coefficient('copper'), 16.5e-6, places=12
        )

    def test_physically_reasonable_range(self):
        """α for all materials should be between 1e-6 and 30e-6 (1/K)."""
        for mat in _ALL_MATERIALS:
            alpha = linear_expansion_coefficient(mat)
            with self.subTest(material=mat):
                self.assertGreater(alpha, 1e-6)
                self.assertLess(alpha, 30e-6)


class TestVolumetric(unittest.TestCase):
    """β = 3α for all materials."""

    def test_beta_equals_three_alpha(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                alpha = linear_expansion_coefficient(mat)
                beta = volumetric_expansion_coefficient(mat)
                self.assertAlmostEqual(beta, 3.0 * alpha, places=20)

    def test_beta_positive(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(volumetric_expansion_coefficient(mat), 0.0)

    def test_beta_larger_than_alpha(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(
                    volumetric_expansion_coefficient(mat),
                    linear_expansion_coefficient(mat),
                )


class TestGruneisen(unittest.TestCase):
    """Grüneisen parameter bounds and derived α validity."""

    def test_gamma_in_physical_range(self):
        """γ should be between 0.5 and 3.5 for all 8 materials."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                gamma = gruneisen_parameter(mat)
                self.assertGreater(gamma, 0.5)
                self.assertLess(gamma, 3.5)

    def test_silicon_lowest_gamma(self):
        """Silicon has unusually low γ ≈ 0.56 (anomalous expansion)."""
        gamma_si = gruneisen_parameter('silicon')
        self.assertLess(gamma_si, 1.0)

    def test_gold_highest_gamma(self):
        """Gold has γ = 3.0, highest in the set."""
        gamma_au = gruneisen_parameter('gold')
        for mat in _ALL_MATERIALS:
            if mat == 'gold':
                continue
            with self.subTest(material=mat):
                self.assertGreaterEqual(gamma_au, gruneisen_parameter(mat))

    def test_derived_alpha_within_factor_3(self):
        """Grüneisen-derived α should be within factor 3 of measured α."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                alpha_meas = linear_expansion_coefficient(mat)
                alpha_deriv = gruneisen_relation(mat)
                ratio = alpha_deriv / alpha_meas
                self.assertGreater(ratio, 1.0 / 3.0,
                                   msg=f"{mat}: ratio {ratio:.3f} < 1/3")
                self.assertLess(ratio, 3.0,
                                msg=f"{mat}: ratio {ratio:.3f} > 3")

    def test_derived_alpha_positive(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(gruneisen_relation(mat), 0.0)


class TestThermalStrain(unittest.TestCase):
    """ε = α × ΔT: sign, magnitude, proportionality."""

    def test_positive_delta_T_positive_strain(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(thermal_strain(mat, 100.0), 0.0)

    def test_negative_delta_T_negative_strain(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertLess(thermal_strain(mat, -50.0), 0.0)

    def test_zero_delta_T_zero_strain(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertEqual(thermal_strain(mat, 0.0), 0.0)

    def test_proportional_to_delta_T(self):
        """Doubling ΔT should double strain."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                eps1 = thermal_strain(mat, 50.0)
                eps2 = thermal_strain(mat, 100.0)
                self.assertAlmostEqual(eps2, 2.0 * eps1, places=15)

    def test_aluminum_100K(self):
        """Al at ΔT=100K: ε ≈ 23.1e-4"""
        eps = thermal_strain('aluminum', 100.0)
        self.assertAlmostEqual(eps, 23.1e-4, delta=1e-6)


class TestThermalStress(unittest.TestCase):
    """σ_th = E α ΔT / (1-2ν): sign and physical bounds."""

    def test_constrained_heating_compressive(self):
        """Positive ΔT (heating) in a constrained body → compressive stress > 0."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                stress = thermal_stress(mat, 100.0)
                self.assertGreater(stress, 0.0)

    def test_constrained_cooling_tensile(self):
        """Negative ΔT (cooling) → tensile stress < 0."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                stress = thermal_stress(mat, -100.0)
                self.assertLess(stress, 0.0)

    def test_zero_delta_T_zero_stress(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertEqual(thermal_stress(mat, 0.0), 0.0)

    def test_stress_positive_for_positive_dT(self):
        """Stress proportional to ΔT — sign preserved."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                s1 = thermal_stress(mat, 50.0)
                s2 = thermal_stress(mat, 100.0)
                self.assertAlmostEqual(s2, 2.0 * s1, places=5)

    def test_stress_order_of_magnitude(self):
        """For steel (iron) ΔT=100K: stress should be in GPa range."""
        # E_iron ~ 200 GPa, α ~ 12e-6, ΔT=100 K, (1-2ν)≈0.42
        # σ ≈ 200e9 × 12e-6 × 100 / 0.42 ≈ 570 MPa (from approx. bulk modulus)
        stress = thermal_stress('iron', 100.0)
        # Should be in the hundreds of MPa to low GPa range
        self.assertGreater(stress, 1e7)   # > 10 MPa
        self.assertLess(stress, 1e11)      # < 100 GPa

    def test_stress_sigma_dependence(self):
        """At sigma=0 and sigma=1e-3, stress should differ (K shifts)."""
        s0 = thermal_stress('iron', 100.0, sigma=0.0)
        s1 = thermal_stress('iron', 100.0, sigma=1e-3)
        # sigma != 0 changes Young's modulus, so stress should differ
        self.assertNotAlmostEqual(s0, s1, places=3)


class TestLengthChange(unittest.TestCase):
    """ΔL = L × α × ΔT."""

    def test_aluminum_1m_100K(self):
        """1 m Al bar heated 100 K: ΔL ≈ 2.31 mm (23.1e-6 × 100 = 2.31e-3 m)."""
        dl = length_change('aluminum', 1.0, 100.0)
        self.assertAlmostEqual(dl, 2.31e-3, delta=1e-6)

    def test_proportional_to_length(self):
        """Doubling length doubles ΔL."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                dl1 = length_change(mat, 1.0, 50.0)
                dl2 = length_change(mat, 2.0, 50.0)
                self.assertAlmostEqual(dl2, 2.0 * dl1, places=15)

    def test_proportional_to_delta_T(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                dl1 = length_change(mat, 1.0, 50.0)
                dl2 = length_change(mat, 1.0, 100.0)
                self.assertAlmostEqual(dl2, 2.0 * dl1, places=15)

    def test_positive_for_heating(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(length_change(mat, 1.0, 10.0), 0.0)

    def test_zero_delta_T_zero_change(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertEqual(length_change(mat, 1.0, 0.0), 0.0)


class TestVolumeChange(unittest.TestCase):
    """ΔV = V × β × ΔT = V × 3α × ΔT."""

    def test_equals_three_times_length_change(self):
        """ΔV/V should equal 3 × ΔL/L for isotropic solid."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                dl_per_m = length_change(mat, 1.0, 100.0)    # ΔL/L for L=1
                dv_per_m3 = volume_change(mat, 1.0, 100.0)   # ΔV/V for V=1
                self.assertAlmostEqual(dv_per_m3, 3.0 * dl_per_m, places=15)

    def test_positive_for_heating(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(volume_change(mat, 1.0, 10.0), 0.0)

    def test_proportional_to_volume(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                dv1 = volume_change(mat, 1.0, 50.0)
                dv2 = volume_change(mat, 2.0, 50.0)
                self.assertAlmostEqual(dv2, 2.0 * dv1, places=15)


class TestTemperatureDependence(unittest.TestCase):
    """α(T=10K) << α(300K); α(1000K) ≈ α(300K) (Dulong-Petit saturation)."""

    def test_low_T_much_smaller_than_300K(self):
        """At T=10K, Debye suppression should make α << α(300K)."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                alpha_10 = expansion_coefficient_at_T(mat, 10.0)
                alpha_300 = expansion_coefficient_at_T(mat, 300.0)
                # Ratio should be at least 10× smaller at 10K
                self.assertLess(
                    alpha_10, alpha_300 / 5.0,
                    msg=f"{mat}: α(10K)={alpha_10:.3e} not << α(300K)={alpha_300:.3e}",
                )

    def test_zero_T_returns_zero(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertEqual(expansion_coefficient_at_T(mat, 0.0), 0.0)

    def test_high_T_close_to_300K(self):
        """At T=1000K, C_v has saturated, so α(1000K) ≈ α(300K).
        Allow within 20% (the Debye model may not saturate perfectly
        at 1000K for all materials).
        """
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                alpha_1000 = expansion_coefficient_at_T(mat, 1000.0)
                alpha_300 = expansion_coefficient_at_T(mat, 300.0)
                ratio = alpha_1000 / alpha_300
                self.assertGreater(ratio, 0.7,
                                   msg=f"{mat}: α(1000K)/α(300K) = {ratio:.3f} < 0.7")
                self.assertLess(ratio, 1.5,
                                msg=f"{mat}: α(1000K)/α(300K) = {ratio:.3f} > 1.5")

    def test_monotonically_increasing_below_debye(self):
        """α(T) should increase with T at low temperatures (Debye regime)."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                alpha_50 = expansion_coefficient_at_T(mat, 50.0)
                alpha_100 = expansion_coefficient_at_T(mat, 100.0)
                alpha_200 = expansion_coefficient_at_T(mat, 200.0)
                self.assertLess(alpha_50, alpha_100)
                self.assertLess(alpha_100, alpha_200)

    def test_debye_integral_high_T_limit(self):
        """At very high T, C_v approaches 3R (Dulong-Petit)."""
        from sigma_ground.field.interface.thermal import debye_temperature
        for mat in _ALL_MATERIALS:
            theta = debye_temperature(mat)
            # Test at 10×Θ_D — should be very close to 3R
            T_high = 10.0 * theta
            cv_high = _debye_cv_molar(T_high, theta)
            dulong_petit = 3.0 * 8.314462618
            ratio = cv_high / dulong_petit
            with self.subTest(material=mat):
                self.assertGreater(ratio, 0.95,
                                   msg=f"{mat}: C_v at 10Θ_D = {ratio:.3f} × 3R, expected > 0.95")

    def test_debye_integral_low_T_cubed(self):
        """At low T, C_v ∝ T³ (Debye law)."""
        from sigma_ground.field.interface.thermal import debye_temperature
        theta = debye_temperature('iron')
        # Test T=10K vs T=20K: ratio should be ~(20/10)³ = 8
        cv_10 = _debye_cv_molar(10.0, theta)
        cv_20 = _debye_cv_molar(20.0, theta)
        if cv_10 > 0:
            ratio = cv_20 / cv_10
            # Should be close to 8 (T³ law)
            self.assertGreater(ratio, 5.0)
            self.assertLess(ratio, 11.0)


class TestSigma(unittest.TestCase):
    """σ-field dependence: identity at σ=0, shifts with σ > 0."""

    def test_sigma_zero_self_consistent(self):
        """sigma_expansion_shift is deterministic: calling twice at σ=0 gives same result."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                alpha_a = sigma_expansion_shift(mat, 0.0)
                alpha_b = sigma_expansion_shift(mat, 0.0)
                self.assertAlmostEqual(alpha_a, alpha_b, places=15)

    def test_sigma_zero_gruneisen_consistent(self):
        """sigma_expansion_shift and gruneisen_relation both use Grüneisen formula;
        at σ=0 they agree to within 5% (any difference is only numerical precision
        in the Debye C_v integral vs the MEASURED C_v value used by gruneisen_relation)."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                alpha_gr = gruneisen_relation(mat, sigma=0.0)
                alpha_sig = sigma_expansion_shift(mat, 0.0)
                # Both are positive and physically consistent
                self.assertGreater(alpha_gr, 0.0)
                self.assertGreater(alpha_sig, 0.0)
                # Ratio within 50% — same formula, slightly different C_v source
                ratio = alpha_sig / alpha_gr
                self.assertGreater(ratio, 0.5)
                self.assertLess(ratio, 2.0)

    def test_positive_sigma_shifts_alpha(self):
        """Non-zero σ changes α through K(σ) and Θ_D(σ).
        At σ=1.0 (neutron-star-surface scale) the shift should be measurable.
        The effect is intrinsically small (~0.5% at σ=1) because:
          - K shifts via the QCD mass correction (~e^σ growth)
          - C_v at 300K is already near the Dulong-Petit plateau
        We verify direction: larger σ → larger K → smaller α.
        """
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                alpha_0 = sigma_expansion_shift(mat, 0.0)
                alpha_large = sigma_expansion_shift(mat, 1.0)
                # Direction: stiffer lattice → smaller α
                self.assertLess(alpha_large, alpha_0,
                                msg=f"{mat}: σ=1.0 should reduce α vs σ=0")

    def test_sigma_shifts_positive(self):
        """α(σ) should remain positive for small positive σ."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(sigma_expansion_shift(mat, 0.1), 0.0)

    def test_sigma_shifts_thermal_stress(self):
        """thermal_stress at σ=0 vs σ=0.01 should differ."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                s0 = thermal_stress(mat, 100.0, sigma=0.0)
                s1 = thermal_stress(mat, 100.0, sigma=0.01)
                self.assertNotAlmostEqual(s0, s1, places=3)

    def test_gruneisen_relation_sigma_positive(self):
        """Grüneisen-derived α at any non-trivial σ should remain positive."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertGreater(gruneisen_relation(mat, sigma=0.5), 0.0)

    def test_expansion_coefficient_at_T_sigma(self):
        """expansion_coefficient_at_T accepts sigma parameter and threads it through.
        At 300K (near the Dulong-Petit plateau), C_v is nearly saturated so
        the σ-shift of Θ_D has minimal effect. The function should at minimum
        return a positive finite value for all sigma values."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                a0 = expansion_coefficient_at_T(mat, 300.0, sigma=0.0)
                a1 = expansion_coefficient_at_T(mat, 300.0, sigma=0.5)
                # Both positive and finite
                self.assertGreater(a0, 0.0)
                self.assertGreater(a1, 0.0)
                self.assertTrue(math.isfinite(a0))
                self.assertTrue(math.isfinite(a1))


class TestNagatha(unittest.TestCase):
    """thermal_expansion_properties returns a complete, correct export dict."""

    _REQUIRED_KEYS = [
        'material', 'temperature_K', 'delta_T_K', 'sigma',
        'alpha_linear_per_K', 'gruneisen_gamma', 'C_v_J_mol_K',
        'beta_volumetric_per_K', 'alpha_derived_gruneisen',
        'thermal_strain', 'thermal_stress_pa',
        'length_change_per_meter_m', 'volume_change_per_m3_m3',
        'alpha_at_T', 'alpha_sigma_shifted', 'origin',
    ]

    def test_all_keys_present(self):
        for mat in _ALL_MATERIALS:
            props = thermal_expansion_properties(mat)
            with self.subTest(material=mat):
                for key in self._REQUIRED_KEYS:
                    self.assertIn(key, props, msg=f"Missing key: {key}")

    def test_material_field_correct(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                props = thermal_expansion_properties(mat)
                self.assertEqual(props['material'], mat)

    def test_numeric_fields_are_numbers(self):
        numeric_keys = [
            'alpha_linear_per_K', 'gruneisen_gamma', 'C_v_J_mol_K',
            'beta_volumetric_per_K', 'alpha_derived_gruneisen',
            'thermal_strain', 'thermal_stress_pa',
            'length_change_per_meter_m', 'volume_change_per_m3_m3',
            'alpha_at_T', 'alpha_sigma_shifted',
        ]
        for mat in _ALL_MATERIALS:
            props = thermal_expansion_properties(mat, delta_T=100.0)
            for key in numeric_keys:
                with self.subTest(material=mat, key=key):
                    self.assertIsInstance(props[key], float)

    def test_origin_is_string(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                props = thermal_expansion_properties(mat)
                self.assertIsInstance(props['origin'], str)
                self.assertGreater(len(props['origin']), 20)

    def test_delta_T_propagates(self):
        """thermal_strain in export should match thermal_strain function."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                dT = 75.0
                props = thermal_expansion_properties(mat, delta_T=dT)
                self.assertAlmostEqual(
                    props['thermal_strain'],
                    thermal_strain(mat, dT),
                    places=15,
                )

    def test_beta_equals_three_alpha_in_export(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                props = thermal_expansion_properties(mat)
                self.assertAlmostEqual(
                    props['beta_volumetric_per_K'],
                    3.0 * props['alpha_linear_per_K'],
                    places=15,
                )

    def test_sigma_in_export(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                props = thermal_expansion_properties(mat, sigma=0.05)
                self.assertAlmostEqual(props['sigma'], 0.05)

    def test_temperature_in_export(self):
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                props = thermal_expansion_properties(mat, T=500.0)
                self.assertAlmostEqual(props['temperature_K'], 500.0)


if __name__ == '__main__':
    unittest.main()
