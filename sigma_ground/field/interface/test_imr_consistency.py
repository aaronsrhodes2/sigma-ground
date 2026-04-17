"""
Tests for sigma_ground.field.interface.imr_consistency.

Validates that:
  - Hypothesis A predicts ε_M = 0 exactly.
  - Hypothesis B1 predicts ε_M in [0.17, 0.20] for plausible f_rad.
  - Published LIGO bounds load and have expected structure.
  - GW150914 alone excludes B1 at > 2σ.
  - Combined GWTC-3 catalog excludes B1 at > 4σ.
"""

import math
import unittest

from sigma_ground.field.constants import XI
from sigma_ground.field.interface.imr_consistency import (
    LIGO_IMR_BOUNDS,
    NOMINAL_F_RAD,
    exclusion_report,
    exclusion_sigma,
    predict_epsilon_M,
    published_bound,
)


class TestPredictEpsilonM(unittest.TestCase):

    def test_hypothesis_A_is_zero(self):
        """GR / mass-conservation predicts ε_M = 0 identically."""
        self.assertEqual(predict_epsilon_M('A'), 0.0)
        self.assertEqual(predict_epsilon_M('A', f_rad=0.10), 0.0)

    def test_hypothesis_B1_at_nominal_f_rad(self):
        """B1 at f_rad = 0.05 predicts ε_M ≈ 0.181."""
        eps = predict_epsilon_M('B1')
        self.assertAlmostEqual(eps, 2.0 * XI / (2.0 * 0.95 - XI), places=12)
        self.assertGreater(eps, 0.17)
        self.assertLess(eps, 0.20)

    def test_B1_weakly_depends_on_f_rad(self):
        """B1 prediction stays in [0.17, 0.20] across plausible f_rad range."""
        for f_rad in (0.03, 0.05, 0.08, 0.10):
            eps = predict_epsilon_M('B1', f_rad=f_rad)
            self.assertGreater(eps, 0.17)
            self.assertLess(eps, 0.20)

    def test_B1_is_positive(self):
        """B1 predicts a POSITIVE deviation (inspiral overestimates M_f)."""
        self.assertGreater(predict_epsilon_M('B1'), 0.0)

    def test_unknown_hypothesis_rejected(self):
        with self.assertRaises(ValueError):
            predict_epsilon_M('B99')

    def test_unphysical_f_rad_rejected(self):
        with self.assertRaises(ValueError):
            predict_epsilon_M('B1', f_rad=0.99)


class TestPublishedBound(unittest.TestCase):

    def test_gw150914_bound_loads(self):
        median, lower, upper, source = published_bound('GW150914')
        self.assertAlmostEqual(median, -0.04, places=2)
        self.assertLess(lower, median)
        self.assertGreater(upper, median)
        self.assertIn('Abbott', source)

    def test_combined_bound_loads(self):
        median, lower, upper, source = published_bound('COMBINED_GWTC3')
        self.assertAlmostEqual(median, -0.01, places=2)
        self.assertLess(upper - lower, 0.12)  # tighter than any single event

    def test_unknown_event_rejected(self):
        with self.assertRaises(ValueError):
            published_bound('GW999999')

    def test_all_bounds_have_4_fields(self):
        for name, bound in LIGO_IMR_BOUNDS.items():
            self.assertEqual(
                len(bound), 4,
                f"bound for {name} has {len(bound)} fields, expected 4"
            )


class TestExclusionSigma(unittest.TestCase):

    def test_hypothesis_A_exclusion_is_zero(self):
        """A predicts ε_M = 0, which trivially sits inside any bound."""
        for event in LIGO_IMR_BOUNDS:
            self.assertEqual(exclusion_sigma(event, hypothesis='A'), 0.0)

    def test_gw150914_excludes_B1_at_over_2_sigma(self):
        """GW150914 alone places B1 in tension at > 2σ (approx 3σ)."""
        sigma = exclusion_sigma('GW150914', hypothesis='B1')
        self.assertGreater(sigma, 2.0,
                           f"expected >2σ exclusion, got {sigma:.2f}σ")

    def test_combined_catalog_excludes_B1_at_over_4_sigma(self):
        """Combined GWTC-3 strongly excludes B1."""
        sigma = exclusion_sigma('COMBINED_GWTC3', hypothesis='B1')
        self.assertGreater(sigma, 4.0,
                           f"expected >4σ exclusion, got {sigma:.2f}σ")

    def test_gw190521_weakly_excludes_B1(self):
        """GW190521 has wide posterior; B1 sits inside or near 1σ."""
        sigma = exclusion_sigma('GW190521', hypothesis='B1')
        # Wide posterior → B1 not strongly excluded by this event alone.
        self.assertLess(sigma, 2.0)

    def test_exclusion_monotonic_in_f_rad(self):
        """Higher f_rad marginally increases ε_M_B1 → slightly higher σ."""
        lo = exclusion_sigma('GW150914', hypothesis='B1', f_rad=0.03)
        hi = exclusion_sigma('GW150914', hypothesis='B1', f_rad=0.10)
        self.assertGreater(hi, lo)


class TestExclusionReport(unittest.TestCase):

    def test_report_returns_row_per_event(self):
        rows = exclusion_report(hypothesis='B1')
        self.assertEqual(len(rows), len(LIGO_IMR_BOUNDS))

    def test_report_is_sorted_by_name(self):
        rows = exclusion_report(hypothesis='B1')
        names = [r[0] for r in rows]
        self.assertEqual(names, sorted(names))

    def test_report_predicted_eps_is_consistent(self):
        """All rows share the same predicted ε_M."""
        rows = exclusion_report(hypothesis='B1')
        predicted = rows[0][1]
        for row in rows:
            self.assertEqual(row[1], predicted)

    def test_report_combined_has_highest_sigma(self):
        """COMBINED_GWTC3 should yield the highest exclusion σ."""
        rows = exclusion_report(hypothesis='B1')
        combined_sigma = next(r[4] for r in rows if r[0] == 'COMBINED_GWTC3')
        for name, _, _, _, sigma, _ in rows:
            if name == 'COMBINED_GWTC3':
                continue
            self.assertGreaterEqual(combined_sigma, sigma,
                                    f"combined σ should exceed {name}")


if __name__ == '__main__':
    unittest.main()
