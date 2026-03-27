"""Tests for wear physics — Archard's law and material removal."""

import math
import unittest

from .wear import (
    archard_wear_volume,
    specific_wear_rate,
    wear_depth,
    sliding_wear_rate,
    wear_mass_loss,
    wear_regime,
    sliding_distance_to_depth,
    relative_wear_resistance,
    sigma_wear_shift,
    wear_profile,
    wear_properties,
    WEAR_DATA,
)
from .friction import _hardness
from .surface import MATERIALS


# ── Rule 9: completeness ────────────────────────────────────────

class TestRule9WearData(unittest.TestCase):
    """Every material in MATERIALS must have wear data."""

    def test_all_materials_have_wear_data(self):
        for key in MATERIALS:
            self.assertIn(key, WEAR_DATA, f"{key} missing from WEAR_DATA")

    def test_all_wear_entries_have_all_fields(self):
        required = {'K_adhesive', 'K_abrasive', 'K_lubricated'}
        for key, data in WEAR_DATA.items():
            for field in required:
                self.assertIn(field, data, f"{key} missing {field}")

    def test_wear_coefficients_positive(self):
        for key, data in WEAR_DATA.items():
            for field in ('K_adhesive', 'K_abrasive', 'K_lubricated'):
                self.assertGreater(data[field], 0, f"{key}.{field} must be > 0")

    def test_wear_coefficient_ordering(self):
        """Lubricated K should be smallest, abrasive largest."""
        for key, data in WEAR_DATA.items():
            self.assertLess(data['K_lubricated'], data['K_adhesive'],
                            f"{key}: lubricated should be less than adhesive")
            # Abrasive >= adhesive for most materials
            self.assertGreaterEqual(data['K_abrasive'], data['K_adhesive'] * 0.5,
                                    f"{key}: abrasive implausibly low")


# ── Archard wear volume ─────────────────────────────────────────

class TestArchardWearVolume(unittest.TestCase):

    def test_positive_for_all_materials(self):
        for key in WEAR_DATA:
            V = archard_wear_volume(key, 100.0, 1000.0)
            self.assertGreater(V, 0, f"{key}: wear volume should be > 0")

    def test_proportional_to_force(self):
        V1 = archard_wear_volume('iron', 10.0, 100.0)
        V2 = archard_wear_volume('iron', 20.0, 100.0)
        self.assertAlmostEqual(V2 / V1, 2.0, places=6)

    def test_proportional_to_distance(self):
        V1 = archard_wear_volume('iron', 10.0, 100.0)
        V2 = archard_wear_volume('iron', 10.0, 300.0)
        self.assertAlmostEqual(V2 / V1, 3.0, places=6)

    def test_zero_force_zero_wear(self):
        self.assertEqual(archard_wear_volume('iron', 0.0, 100.0), 0.0)

    def test_zero_distance_zero_wear(self):
        self.assertEqual(archard_wear_volume('iron', 10.0, 0.0), 0.0)

    def test_negative_inputs_zero(self):
        self.assertEqual(archard_wear_volume('iron', -5.0, 100.0), 0.0)
        self.assertEqual(archard_wear_volume('iron', 10.0, -50.0), 0.0)

    def test_harder_material_wears_less(self):
        """Tungsten (hard) should wear less than aluminum (soft) under same load."""
        V_W = archard_wear_volume('tungsten', 100.0, 1000.0)
        V_Al = archard_wear_volume('aluminum', 100.0, 1000.0)
        self.assertLess(V_W, V_Al)

    def test_lubricated_less_than_dry(self):
        V_dry = archard_wear_volume('iron', 100.0, 1000.0, 'adhesive')
        V_lub = archard_wear_volume('iron', 100.0, 1000.0, 'lubricated')
        self.assertLess(V_lub, V_dry)

    def test_abrasive_more_than_adhesive(self):
        V_adh = archard_wear_volume('iron', 100.0, 1000.0, 'adhesive')
        V_abr = archard_wear_volume('iron', 100.0, 1000.0, 'abrasive')
        self.assertGreater(V_abr, V_adh)

    def test_invalid_wear_mode_raises(self):
        with self.assertRaises(ValueError):
            archard_wear_volume('iron', 100.0, 1000.0, 'magical')


# ── Specific wear rate ───────────────────────────────────────────

class TestSpecificWearRate(unittest.TestCase):

    def test_positive_all_materials(self):
        for key in WEAR_DATA:
            k = specific_wear_rate(key)
            self.assertGreater(k, 0, f"{key}: specific wear rate should be > 0")

    def test_units_order_of_magnitude(self):
        """Typical specific wear rates: 10⁻¹² to 10⁻⁶ m²/N."""
        for key in WEAR_DATA:
            k = specific_wear_rate(key, 'adhesive')
            self.assertGreater(k, 1e-16, f"{key}: implausibly low wear rate")
            self.assertLess(k, 1e-3, f"{key}: implausibly high wear rate")

    def test_lubricated_lower(self):
        for key in WEAR_DATA:
            k_adh = specific_wear_rate(key, 'adhesive')
            k_lub = specific_wear_rate(key, 'lubricated')
            self.assertLess(k_lub, k_adh, f"{key}: lubricated should be lower")

    def test_consistent_with_archard(self):
        """k = V / (F × d), check consistency."""
        F, d = 50.0, 200.0
        for key in WEAR_DATA:
            k = specific_wear_rate(key)
            V = archard_wear_volume(key, F, d)
            self.assertAlmostEqual(V / (F * d), k, places=20)


# ── Wear depth ───────────────────────────────────────────────────

class TestWearDepth(unittest.TestCase):

    def test_positive(self):
        h = wear_depth('iron', 1e7, 1000.0)
        self.assertGreater(h, 0)

    def test_proportional_to_pressure(self):
        h1 = wear_depth('iron', 1e7, 1000.0)
        h2 = wear_depth('iron', 2e7, 1000.0)
        self.assertAlmostEqual(h2 / h1, 2.0, places=6)

    def test_zero_inputs(self):
        self.assertEqual(wear_depth('iron', 0.0, 1000.0), 0.0)
        self.assertEqual(wear_depth('iron', 1e7, 0.0), 0.0)


# ── Sliding wear rate (time-based) ──────────────────────────────

class TestSlidingWearRate(unittest.TestCase):

    def test_positive(self):
        rate = sliding_wear_rate('copper', 10.0, 1.0)
        self.assertGreater(rate, 0)

    def test_proportional_to_velocity(self):
        r1 = sliding_wear_rate('copper', 10.0, 1.0)
        r2 = sliding_wear_rate('copper', 10.0, 3.0)
        self.assertAlmostEqual(r2 / r1, 3.0, places=6)

    def test_zero_inputs(self):
        self.assertEqual(sliding_wear_rate('copper', 0.0, 1.0), 0.0)
        self.assertEqual(sliding_wear_rate('copper', 10.0, 0.0), 0.0)


# ── Mass loss ────────────────────────────────────────────────────

class TestWearMassLoss(unittest.TestCase):

    def test_positive(self):
        m = wear_mass_loss('gold', 50.0, 500.0)
        self.assertGreater(m, 0)

    def test_consistent_with_volume(self):
        V = archard_wear_volume('gold', 50.0, 500.0)
        m = wear_mass_loss('gold', 50.0, 500.0)
        rho = MATERIALS['gold']['density_kg_m3']
        self.assertAlmostEqual(m, rho * V, places=20)

    def test_denser_material_loses_more_mass_per_volume(self):
        """Gold (dense) loses more mass than aluminum (light) for same volume."""
        # Same K and H would give same volume; different K/H but we just
        # check that mass = rho * V relationship holds.
        V_Au = archard_wear_volume('gold', 50.0, 500.0)
        V_Al = archard_wear_volume('aluminum', 50.0, 500.0)
        m_Au = wear_mass_loss('gold', 50.0, 500.0)
        m_Al = wear_mass_loss('aluminum', 50.0, 500.0)
        # mass/volume = density
        self.assertAlmostEqual(m_Au / V_Au, MATERIALS['gold']['density_kg_m3'])
        self.assertAlmostEqual(m_Al / V_Al, MATERIALS['aluminum']['density_kg_m3'])


# ── Wear regime classification ──────────────────────────────────

class TestWearRegime(unittest.TestCase):

    def test_self_contact_adhesive(self):
        """Same material should give adhesive regime."""
        self.assertEqual(wear_regime('iron', 'iron'), 'adhesive')
        self.assertEqual(wear_regime('copper', 'copper'), 'adhesive')

    def test_hard_on_soft(self):
        """Tungsten on aluminum — very different hardness."""
        H_W = _hardness('tungsten')
        H_Al = _hardness('aluminum')
        if H_W / H_Al > 1.2:
            self.assertEqual(wear_regime('aluminum', 'tungsten'), 'abrasive')

    def test_soft_on_hard(self):
        """Aluminum on tungsten — counter-body wears."""
        H_W = _hardness('tungsten')
        H_Al = _hardness('aluminum')
        if H_Al / H_W < 0.8:
            self.assertEqual(wear_regime('tungsten', 'aluminum'),
                             'counter-body-wears')


# ── Distance to failure ─────────────────────────────────────────

class TestSlidingDistanceToDepth(unittest.TestCase):

    def test_positive(self):
        d = sliding_distance_to_depth('iron', 1e-3, 1e7)
        self.assertGreater(d, 0)

    def test_deeper_tolerance_longer_life(self):
        d1 = sliding_distance_to_depth('iron', 1e-3, 1e7)
        d2 = sliding_distance_to_depth('iron', 2e-3, 1e7)
        self.assertAlmostEqual(d2 / d1, 2.0, places=6)

    def test_higher_pressure_shorter_life(self):
        d1 = sliding_distance_to_depth('iron', 1e-3, 1e7)
        d2 = sliding_distance_to_depth('iron', 1e-3, 2e7)
        self.assertAlmostEqual(d1 / d2, 2.0, places=6)

    def test_round_trip_with_wear_depth(self):
        """wear_depth(d) should equal target_depth when d = distance_to_depth."""
        target = 5e-4  # 0.5 mm
        p = 1e7
        d = sliding_distance_to_depth('copper', target, p)
        h = wear_depth('copper', p, d)
        self.assertAlmostEqual(h, target, places=10)


# ── Relative wear resistance ────────────────────────────────────

class TestRelativeWearResistance(unittest.TestCase):

    def test_iron_reference_is_unity(self):
        R = relative_wear_resistance('iron', 'iron')
        self.assertAlmostEqual(R, 1.0, places=6)

    def test_tungsten_more_resistant_than_aluminum(self):
        R_W = relative_wear_resistance('tungsten', 'iron')
        R_Al = relative_wear_resistance('aluminum', 'iron')
        self.assertGreater(R_W, R_Al)


# ── σ-field coupling ────────────────────────────────────────────

class TestSigmaWearShift(unittest.TestCase):

    def test_identity_at_zero(self):
        for key in WEAR_DATA:
            self.assertAlmostEqual(sigma_wear_shift(key, 0.0), 1.0)

    def test_positive_sigma_reduces_wear(self):
        """Positive σ strengthens bonds → higher H → less wear."""
        ratio = sigma_wear_shift('iron', 0.01)
        self.assertLess(ratio, 1.0)

    def test_negative_sigma_increases_wear(self):
        ratio = sigma_wear_shift('iron', -0.01)
        self.assertGreater(ratio, 1.0)

    def test_sigma_shifts_archard_volume(self):
        """Wear volume should change with σ consistent with shift ratio."""
        V_0 = archard_wear_volume('copper', 100.0, 1000.0, 'adhesive', 0.0)
        V_s = archard_wear_volume('copper', 100.0, 1000.0, 'adhesive', 0.01)
        ratio = sigma_wear_shift('copper', 0.01)
        self.assertAlmostEqual(V_s / V_0, ratio, places=6)

    def test_earth_sigma_negligible(self):
        """Earth's σ ≈ 10⁻³⁹, should barely shift wear."""
        ratio = sigma_wear_shift('iron', 1e-39)
        self.assertAlmostEqual(ratio, 1.0, places=10)


# ── Wear profile simulation ─────────────────────────────────────

class TestWearProfile(unittest.TestCase):

    def test_returns_correct_length(self):
        profile = wear_profile('iron', 100.0, 1.0, 100.0, steps=10)
        self.assertEqual(len(profile), 11)  # 0..10 inclusive

    def test_starts_at_zero(self):
        profile = wear_profile('iron', 100.0, 1.0, 100.0)
        self.assertEqual(profile[0]['time_s'], 0.0)
        self.assertEqual(profile[0]['volume_m3'], 0.0)
        self.assertEqual(profile[0]['mass_kg'], 0.0)

    def test_monotonically_increasing(self):
        profile = wear_profile('copper', 50.0, 2.0, 200.0, steps=20)
        for i in range(1, len(profile)):
            self.assertGreater(profile[i]['volume_m3'],
                               profile[i - 1]['volume_m3'])

    def test_final_volume_matches_archard(self):
        F, v, t = 100.0, 1.0, 500.0
        profile = wear_profile('aluminum', F, v, t, steps=50)
        V_profile = profile[-1]['volume_m3']
        V_archard = archard_wear_volume('aluminum', F, v * t)
        self.assertAlmostEqual(V_profile, V_archard, places=15)


# ── Nagatha export ──────────────────────────────────────────────

class TestWearProperties(unittest.TestCase):

    def test_all_materials_export(self):
        for key in WEAR_DATA:
            props = wear_properties(key)
            self.assertEqual(props['material'], key)
            self.assertGreater(props['hardness_pa'], 0)
            self.assertIn('origin', props)

    def test_expected_keys(self):
        props = wear_properties('iron')
        expected = {
            'material', 'sigma', 'hardness_pa',
            'K_adhesive', 'K_abrasive', 'K_lubricated',
            'specific_wear_rate_adhesive_m2_N',
            'specific_wear_rate_abrasive_m2_N',
            'specific_wear_rate_lubricated_m2_N',
            'sigma_wear_ratio', 'relative_wear_resistance_vs_iron',
            'origin',
        }
        for k in expected:
            self.assertIn(k, props, f"Missing key: {k}")

    def test_sigma_export(self):
        props = wear_properties('iron', sigma=0.01)
        self.assertAlmostEqual(props['sigma'], 0.01)
        self.assertLess(props['sigma_wear_ratio'], 1.0)


if __name__ == '__main__':
    unittest.main()
