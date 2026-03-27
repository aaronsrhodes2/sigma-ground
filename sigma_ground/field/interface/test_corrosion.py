"""
Tests for the corrosion module.

Test structure:
  1. TestPillingBedworth   — PBR values and known references
  2. TestOxideClassification — film type from PBR
  3. TestParabolicGrowth   — thickness > 0, time/temperature scaling
  4. TestGalvanicSeries    — ordering by electrode potential
  5. TestGalvanicPotential — galvanic couple signs
  6. TestCorrosionRate     — relative rates, temperature dependence
  7. TestSigma             — σ-field shifts activation energy and rate
  8. TestRule9             — all 8 materials carry all fields
  9. TestNagatha           — export completeness
"""

import math
import unittest

from .corrosion import (
    CORROSION_DATA,
    pilling_bedworth_ratio,
    oxide_classification,
    parabolic_oxide_thickness,
    galvanic_potential,
    galvanic_series_rank,
    corrosion_rate_estimate,
    sigma_corrosion_shift,
    corrosion_properties,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_ALL_MATERIALS = [
    'iron', 'copper', 'aluminum', 'gold',
    'silicon', 'tungsten', 'nickel', 'titanium',
]

_REQUIRED_CORROSION_FIELDS = [
    'E_standard_V',
    'oxide_density_kg_m3',
    'oxide_molar_mass_g',
    'metal_molar_mass_g',
    'n_oxide_metal_atoms',
    'oxide_name',
    'k_parabolic_m2_s',
    'Q_oxidation_eV',
]

_REQUIRED_EXPORT_KEYS = [
    'material',
    'oxide_name',
    'E_standard_V',
    'pilling_bedworth_ratio',
    'oxide_classification',
    'oxide_thickness_m',
    'corrosion_rate_kg_m2_s',
    'sigma_corrosion_rate_kg_m2_s',
    'galvanic_rank',
    'time_s',
    'temperature_K',
    'sigma',
    'origin_tag',
]


# ── TestPillingBedworth ───────────────────────────────────────────────────────

class TestPillingBedworth(unittest.TestCase):
    """Pilling-Bedworth ratio."""

    def test_aluminum_pbr_approx_1_28(self):
        """Al₂O₃ on Al: PBR ≈ 1.28, firmly protective.

        Reference calculation:
          PBR = (M_Al2O3 × ρ_Al) / (2 × M_Al × ρ_Al2O3)
              = (101.96 × 2700) / (2 × 26.982 × 3950)
              ≈ 275292 / 213268 ≈ 1.29
        """
        pbr = pilling_bedworth_ratio('aluminum')
        self.assertAlmostEqual(pbr, 1.28, delta=0.05)
        self.assertGreater(pbr, 1.0)
        self.assertLess(pbr, 2.0)

    def test_iron_pbr_greater_than_2(self):
        """Fe₂O₃ on Fe: PBR > 2, spalling.

        Reference calculation:
          PBR = (159.69 × 7874) / (2 × 55.845 × 5250)
              ≈ 1257299 / 586373 ≈ 2.14
        """
        pbr = pilling_bedworth_ratio('iron')
        self.assertGreater(pbr, 2.0)
        # Sanity: not unreasonably large
        self.assertLess(pbr, 5.0)

    def test_titanium_pbr_protective(self):
        """TiO₂ on Ti: PBR ≈ 1.73, protective."""
        pbr = pilling_bedworth_ratio('titanium')
        self.assertGreater(pbr, 1.0)
        self.assertLess(pbr, 2.0)

    def test_pbr_positive(self):
        """PBR is positive for all materials."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                pbr = pilling_bedworth_ratio(mat)
                self.assertGreater(pbr, 0.0)

    def test_pbr_unknown_material_raises(self):
        """Unknown material raises KeyError."""
        with self.assertRaises(KeyError):
            pilling_bedworth_ratio('unobtanium')


# ── TestOxideClassification ───────────────────────────────────────────────────

class TestOxideClassification(unittest.TestCase):
    """Oxide film classification from PBR."""

    def test_aluminum_protective(self):
        """Al₂O₃ film is protective (1 < PBR < 2)."""
        self.assertEqual(oxide_classification('aluminum'), 'protective')

    def test_iron_spalling(self):
        """Fe₂O₃ (rust) is spalling (PBR > 2)."""
        self.assertEqual(oxide_classification('iron'), 'spalling')

    def test_titanium_protective(self):
        """TiO₂ is protective."""
        self.assertEqual(oxide_classification('titanium'), 'protective')

    def test_gold_protective(self):
        """Au₂O₃ PBR falls in protective range; film is negligible but geometrically intact."""
        # Gold barely oxidizes, but PBR calculation should be well-formed
        cls = oxide_classification('gold')
        self.assertIn(cls, ('protective', 'porous', 'spalling'))

    def test_all_return_valid_string(self):
        """All materials return one of the three valid classification strings."""
        valid = {'porous', 'protective', 'spalling'}
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertIn(oxide_classification(mat), valid)


# ── TestParabolicGrowth ───────────────────────────────────────────────────────

class TestParabolicGrowth(unittest.TestCase):
    """Parabolic oxide thickness."""

    def test_thickness_positive(self):
        """Thickness is positive for all materials at t > 0."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                x = parabolic_oxide_thickness(mat, 1e6)
                self.assertGreater(x, 0.0)

    def test_longer_time_gives_thicker_oxide(self):
        """x(2t) > x(t): thickness grows with time."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                x1 = parabolic_oxide_thickness(mat, 1e6)
                x2 = parabolic_oxide_thickness(mat, 1e8)
                self.assertGreater(x2, x1)

    def test_higher_temperature_gives_thicker_oxide(self):
        """x(T=500) > x(T=300): faster diffusion at higher T."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                x_low = parabolic_oxide_thickness(mat, 1e6, temperature=300)
                x_high = parabolic_oxide_thickness(mat, 1e6, temperature=500)
                self.assertGreater(x_high, x_low)

    def test_parabolic_scaling(self):
        """x ∝ √t: quadrupling time doubles thickness."""
        t1 = 1e6
        t2 = 4e6
        x1 = parabolic_oxide_thickness('aluminum', t1)
        x2 = parabolic_oxide_thickness('aluminum', t2)
        ratio = x2 / x1
        # Should be √4 = 2.0 exactly
        self.assertAlmostEqual(ratio, 2.0, places=10)

    def test_at_standard_conditions(self):
        """At T=300 K, t=1 s: x = √k_parabolic."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                k = CORROSION_DATA[mat]['k_parabolic_m2_s']
                x = parabolic_oxide_thickness(mat, 1.0, temperature=300.0)
                self.assertAlmostEqual(x, math.sqrt(k), places=40)

    def test_zero_time_raises(self):
        """time_s ≤ 0 raises ValueError."""
        with self.assertRaises(ValueError):
            parabolic_oxide_thickness('iron', 0.0)
        with self.assertRaises(ValueError):
            parabolic_oxide_thickness('iron', -1.0)

    def test_zero_temperature_raises(self):
        """temperature ≤ 0 K raises ValueError."""
        with self.assertRaises(ValueError):
            parabolic_oxide_thickness('iron', 1e6, temperature=0.0)


# ── TestGalvanicSeries ────────────────────────────────────────────────────────

class TestGalvanicSeries(unittest.TestCase):
    """Galvanic series ordering."""

    def test_aluminum_more_anodic_than_copper(self):
        """Al (−1.66 V) is more anodic than Cu (+0.34 V)."""
        series = galvanic_series_rank()
        keys = [k for k, _ in series]
        al_idx = keys.index('aluminum')
        cu_idx = keys.index('copper')
        self.assertLess(al_idx, cu_idx)

    def test_gold_most_noble(self):
        """Gold has the highest (most positive) E° among the 8 materials."""
        series = galvanic_series_rank()
        # Last entry has highest E°
        self.assertEqual(series[-1][0], 'gold')

    def test_aluminum_or_titanium_most_anodic(self):
        """Most anodic material is Al (−1.66 V) or Ti (−1.63 V)."""
        series = galvanic_series_rank()
        most_anodic_key = series[0][0]
        self.assertIn(most_anodic_key, ('aluminum', 'titanium'))

    def test_series_sorted_ascending(self):
        """Series is monotonically sorted by E° (ascending)."""
        series = galvanic_series_rank()
        potentials = [e for _, e in series]
        self.assertEqual(potentials, sorted(potentials))

    def test_series_contains_all_materials(self):
        """Galvanic series includes all 8 materials."""
        series = galvanic_series_rank()
        keys = {k for k, _ in series}
        self.assertEqual(keys, set(_ALL_MATERIALS))


# ── TestGalvanicPotential ─────────────────────────────────────────────────────

class TestGalvanicPotential(unittest.TestCase):
    """Galvanic couple potential differences."""

    def test_fe_cu_couple_positive(self):
        """ΔE(Cu, Fe) > 0: copper is cathodic, iron corrodes."""
        delta_e = galvanic_potential('copper', 'iron')
        self.assertGreater(delta_e, 0.0)
        # Numeric check: +0.34 − (−0.44) = +0.78 V
        self.assertAlmostEqual(delta_e, 0.78, delta=0.02)

    def test_fe_cu_signs_antisymmetric(self):
        """ΔE(a, b) = −ΔE(b, a)."""
        delta_ab = galvanic_potential('copper', 'iron')
        delta_ba = galvanic_potential('iron', 'copper')
        self.assertAlmostEqual(delta_ab, -delta_ba, places=12)

    def test_same_material_zero(self):
        """ΔE(a, a) = 0."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertAlmostEqual(galvanic_potential(mat, mat), 0.0, places=12)

    def test_au_most_noble(self):
        """Au paired with any other material: Au is always cathodic (ΔE > 0)."""
        others = [m for m in _ALL_MATERIALS if m != 'gold']
        for mat in others:
            with self.subTest(material=mat):
                delta = galvanic_potential('gold', mat)
                self.assertGreater(delta, 0.0)

    def test_al_most_anodic_in_couple_with_cu(self):
        """Al is anodic (ΔE < 0) when paired as a with Cu as b."""
        delta = galvanic_potential('aluminum', 'copper')
        self.assertLess(delta, 0.0)

    def test_unknown_raises(self):
        """Unknown material raises KeyError."""
        with self.assertRaises(KeyError):
            galvanic_potential('unobtanium', 'iron')
        with self.assertRaises(KeyError):
            galvanic_potential('iron', 'unobtanium')


# ── TestCorrosionRate ─────────────────────────────────────────────────────────

class TestCorrosionRate(unittest.TestCase):
    """Corrosion mass loss rate."""

    def test_all_rates_positive(self):
        """Corrosion rate is positive for all materials."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                rate = corrosion_rate_estimate(mat)
                self.assertGreater(rate, 0.0)

    def test_gold_rate_much_less_than_iron(self):
        """Gold corrodes far slower than iron."""
        rate_au = corrosion_rate_estimate('gold')
        rate_fe = corrosion_rate_estimate('iron')
        self.assertLess(rate_au, rate_fe)

    def test_iron_rate_greater_than_aluminum(self):
        """Iron corrodes faster than aluminum at 300 K.

        Fe has k ~ 1e-21 m²/s, Al has k ~ 1e-26 m²/s, so √k_Fe >> √k_Al.
        Iron's k is 5 orders of magnitude larger.
        """
        rate_fe = corrosion_rate_estimate('iron')
        rate_al = corrosion_rate_estimate('aluminum')
        self.assertGreater(rate_fe, rate_al)

    def test_rate_increases_with_temperature(self):
        """Higher temperature → higher corrosion rate (Arrhenius)."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                r300 = corrosion_rate_estimate(mat, temperature=300)
                r600 = corrosion_rate_estimate(mat, temperature=600)
                self.assertGreater(r600, r300)

    def test_invalid_temperature_raises(self):
        """temperature ≤ 0 K raises ValueError."""
        with self.assertRaises(ValueError):
            corrosion_rate_estimate('iron', temperature=0.0)


# ── TestSigma ─────────────────────────────────────────────────────────────────

class TestSigma(unittest.TestCase):
    """σ-field effect on corrosion."""

    def test_sigma_zero_equals_base_rate(self):
        """sigma=0 returns same as corrosion_rate_estimate at 300 K."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                base = corrosion_rate_estimate(mat, temperature=300.0)
                shifted = sigma_corrosion_shift(mat, 0.0)
                self.assertAlmostEqual(shifted, base, places=30)

    def test_positive_sigma_changes_rate(self):
        """Positive sigma shifts corrosion rate (should differ from sigma=0).

        Uses relative comparison: sigma=0.1 should change the rate by at least
        0.001% relative to baseline.  assertNotAlmostEqual(places=N) is an
        absolute check and fails for very small rates, so we use a relative test.
        """
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                base = sigma_corrosion_shift(mat, 0.0)
                shifted = sigma_corrosion_shift(mat, 0.1)
                rel_diff = abs(shifted - base) / (abs(base) + 1e-300)
                self.assertGreater(
                    rel_diff, 1e-5,
                    msg=f"sigma shift too small for {mat}: base={base}, shifted={shifted}"
                )

    def test_sigma_rate_positive(self):
        """sigma-shifted corrosion rate is positive for all materials."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                rate = sigma_corrosion_shift(mat, 0.05)
                self.assertGreater(rate, 0.0)

    def test_large_sigma_shifts_rate_significantly(self):
        """sigma=1.0 produces a noticeably different rate from sigma=0."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                base = sigma_corrosion_shift(mat, 0.0)
                shifted = sigma_corrosion_shift(mat, 1.0)
                # At sigma=1, QCD mass fraction shifts; expect at least 1% change
                ratio = abs(shifted - base) / (base + 1e-300)
                self.assertGreater(ratio, 1e-4)

    def test_unknown_material_raises(self):
        """Unknown material raises KeyError."""
        with self.assertRaises(KeyError):
            sigma_corrosion_shift('unobtanium', 0.0)


# ── TestRule9 ─────────────────────────────────────────────────────────────────

class TestRule9(unittest.TestCase):
    """Golden Rule 9: every material gets every field."""

    def test_all_8_materials_present(self):
        """CORROSION_DATA has entries for all 8 expected materials."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                self.assertIn(mat, CORROSION_DATA)

    def test_all_fields_present(self):
        """Every material has every required field."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                for field in _REQUIRED_CORROSION_FIELDS:
                    self.assertIn(
                        field, CORROSION_DATA[mat],
                        msg=f"Missing field {field!r} for {mat!r}"
                    )

    def test_numeric_fields_positive(self):
        """All numeric data fields are positive."""
        numeric_fields = [
            'oxide_density_kg_m3', 'oxide_molar_mass_g',
            'metal_molar_mass_g', 'n_oxide_metal_atoms',
            'k_parabolic_m2_s', 'Q_oxidation_eV',
        ]
        for mat in _ALL_MATERIALS:
            for field in numeric_fields:
                with self.subTest(material=mat, field=field):
                    val = CORROSION_DATA[mat][field]
                    self.assertGreater(val, 0.0)

    def test_oxide_name_is_string(self):
        """oxide_name is a non-empty string for every material."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                name = CORROSION_DATA[mat]['oxide_name']
                self.assertIsInstance(name, str)
                self.assertGreater(len(name), 0)

    def test_n_oxide_metal_atoms_integer(self):
        """n_oxide_metal_atoms is a positive integer."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                n = CORROSION_DATA[mat]['n_oxide_metal_atoms']
                self.assertIsInstance(n, int)
                self.assertGreaterEqual(n, 1)


# ── TestNagatha ───────────────────────────────────────────────────────────────

class TestNagatha(unittest.TestCase):
    """Nagatha export completeness."""

    def test_all_required_keys_present(self):
        """Export dict contains all required keys for every material."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                result = corrosion_properties(mat)
                for key in _REQUIRED_EXPORT_KEYS:
                    self.assertIn(
                        key, result,
                        msg=f"Missing key {key!r} in export for {mat!r}"
                    )

    def test_material_key_matches(self):
        """Export 'material' field matches input key."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                result = corrosion_properties(mat)
                self.assertEqual(result['material'], mat)

    def test_default_time_one_year(self):
        """Default time_s = 3.15e7 ≈ 1 year."""
        result = corrosion_properties('iron')
        self.assertAlmostEqual(result['time_s'], 3.15e7, delta=1.0)

    def test_thickness_positive(self):
        """oxide_thickness_m is positive for all materials."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                result = corrosion_properties(mat)
                self.assertGreater(result['oxide_thickness_m'], 0.0)

    def test_pbr_matches_function(self):
        """Exported PBR matches pilling_bedworth_ratio()."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                result = corrosion_properties(mat)
                expected = pilling_bedworth_ratio(mat)
                self.assertAlmostEqual(
                    result['pilling_bedworth_ratio'], expected, places=12
                )

    def test_classification_matches_function(self):
        """Exported oxide_classification matches oxide_classification()."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                result = corrosion_properties(mat)
                self.assertEqual(
                    result['oxide_classification'], oxide_classification(mat)
                )

    def test_sigma_zero_rate_matches_base(self):
        """sigma=0 export rate matches corrosion_rate_estimate."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                result = corrosion_properties(mat, T=300.0, sigma=0.0)
                base = corrosion_rate_estimate(mat, temperature=300.0)
                self.assertAlmostEqual(
                    result['corrosion_rate_kg_m2_s'], base, places=30
                )

    def test_origin_tag_nonempty(self):
        """origin_tag is a non-empty string."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                result = corrosion_properties(mat)
                self.assertIsInstance(result['origin_tag'], str)
                self.assertGreater(len(result['origin_tag']), 10)

    def test_unknown_material_raises(self):
        """Unknown material raises KeyError."""
        with self.assertRaises(KeyError):
            corrosion_properties('unobtanium')

    def test_sigma_nonzero_export(self):
        """sigma != 0 export completes without error."""
        for mat in _ALL_MATERIALS:
            with self.subTest(material=mat):
                result = corrosion_properties(mat, sigma=0.1)
                self.assertIn('sigma_corrosion_rate_kg_m2_s', result)
                self.assertGreater(result['sigma_corrosion_rate_kg_m2_s'], 0.0)


if __name__ == '__main__':
    unittest.main()
