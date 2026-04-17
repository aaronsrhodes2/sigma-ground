"""
Tests for sigma_ground.field.interface.sgr_a_star_rate.

Validates that:
  - Null-Poisson rate bound formula is correct at 90 % CL.
  - Sgr A* alone is 8–10 orders of magnitude too weak for baseline B3.
  - Population-level XRB bound is tighter but still insufficient.
  - Object-years needed to reach baseline is ~3e10 (astronomically large).
"""

import math
import unittest

from sigma_ground.field.constants import XI
from sigma_ground.field.interface.sgr_a_star_rate import (
    BASELINE_B3_RATE_PER_YEAR,
    GRAVITY_OBSERVATION_YEARS,
    SGR_A_STAR_MASS_FRACTIONAL_STABILITY,
    SGR_A_STAR_MASS_SOLAR,
    TAU_HUBBLE_YEARS,
    XRAY_BINARY_BH_OBJECT_YEARS,
    baseline_rate_test,
    minimum_object_years_to_test_baseline,
    rate_bound_null_observation,
    sgr_a_star_rate_bound,
    summary,
    xray_binary_population_rate_bound,
)


class TestRateBoundFormula(unittest.TestCase):

    def test_ninety_percent_CL_uses_ln_10(self):
        """λ_max at 90 % CL is −ln(0.1) = 2.303."""
        R = rate_bound_null_observation(1.0, confidence_level=0.90)
        self.assertAlmostEqual(R, -math.log(0.10), places=10)

    def test_scales_inversely_with_object_years(self):
        R1 = rate_bound_null_observation(10.0)
        R100 = rate_bound_null_observation(1000.0)
        self.assertAlmostEqual(R1 / R100, 100.0, places=10)

    def test_higher_CL_gives_higher_rate_bound(self):
        R90 = rate_bound_null_observation(100.0, confidence_level=0.90)
        R99 = rate_bound_null_observation(100.0, confidence_level=0.99)
        self.assertGreater(R99, R90)

    def test_zero_object_years_rejected(self):
        with self.assertRaises(ValueError):
            rate_bound_null_observation(0.0)
        with self.assertRaises(ValueError):
            rate_bound_null_observation(-10.0)

    def test_invalid_CL_rejected(self):
        with self.assertRaises(ValueError):
            rate_bound_null_observation(100.0, confidence_level=0.0)
        with self.assertRaises(ValueError):
            rate_bound_null_observation(100.0, confidence_level=1.0)


class TestConstants(unittest.TestCase):

    def test_baseline_rate_is_inverse_hubble(self):
        self.assertAlmostEqual(
            BASELINE_B3_RATE_PER_YEAR * TAU_HUBBLE_YEARS, 1.0, places=10
        )

    def test_hubble_time_is_roughly_14_gyr(self):
        self.assertGreater(TAU_HUBBLE_YEARS, 1.3e10)
        self.assertLess(TAU_HUBBLE_YEARS, 1.5e10)

    def test_sgr_a_star_mass_is_about_4_million_solar(self):
        self.assertGreater(SGR_A_STAR_MASS_SOLAR, 4.0e6)
        self.assertLess(SGR_A_STAR_MASS_SOLAR, 4.5e6)


class TestSgrAStarBound(unittest.TestCase):

    def test_single_object_rate_bound_is_roughly_0p08(self):
        """Sgr A* alone: R_max = 2.303/30 ≈ 0.077 /yr."""
        R = sgr_a_star_rate_bound()
        self.assertAlmostEqual(R, 2.303 / 30.0, delta=0.01)

    def test_rate_bound_far_exceeds_baseline(self):
        R = sgr_a_star_rate_bound()
        self.assertGreater(R / BASELINE_B3_RATE_PER_YEAR, 1e8)

    def test_not_sensitive_to_baseline(self):
        R = sgr_a_star_rate_bound()
        result = baseline_rate_test(R)
        self.assertFalse(result['sensitive'])
        self.assertGreater(result['ratio'], 1e8)


class TestXRBPopulationBound(unittest.TestCase):

    def test_population_bound_tighter_than_sgr_a_star(self):
        R_sgr = sgr_a_star_rate_bound()
        R_xrb = xray_binary_population_rate_bound()
        self.assertLess(R_xrb, R_sgr)

    def test_xrb_still_fails_baseline(self):
        R = xray_binary_population_rate_bound()
        result = baseline_rate_test(R)
        self.assertFalse(result['sensitive'])
        self.assertGreater(result['ratio'], 1e6)

    def test_bound_scales_with_object_years(self):
        R_600 = xray_binary_population_rate_bound(object_years=600.0)
        R_6000 = xray_binary_population_rate_bound(object_years=6000.0)
        self.assertAlmostEqual(R_600 / R_6000, 10.0, places=10)


class TestMinimumObjectYears(unittest.TestCase):

    def test_baseline_test_requires_enormous_observation(self):
        """Reaching R_max = 1/τ_Hubble needs ~3e10 object-years."""
        N = minimum_object_years_to_test_baseline()
        self.assertGreater(N, 1e10)
        self.assertLess(N, 1e11)

    def test_tighter_CL_demands_more(self):
        N90 = minimum_object_years_to_test_baseline(confidence_level=0.90)
        N99 = minimum_object_years_to_test_baseline(confidence_level=0.99)
        self.assertGreater(N99, N90)


class TestSummary(unittest.TestCase):

    def test_summary_has_expected_keys(self):
        s = summary()
        expected = {
            'sgr_a_star',
            'xrb_population',
            'baseline_rate_per_year',
            'tau_Hubble_years',
            'minimum_object_years_for_baseline_test',
            'xi_conversion_fraction',
        }
        self.assertEqual(set(s.keys()), expected)

    def test_summary_xi_matches_constant(self):
        s = summary()
        self.assertEqual(s['xi_conversion_fraction'], XI)

    def test_summary_sgr_not_sensitive(self):
        s = summary()
        self.assertFalse(s['sgr_a_star']['sensitive'])


if __name__ == '__main__':
    unittest.main()
