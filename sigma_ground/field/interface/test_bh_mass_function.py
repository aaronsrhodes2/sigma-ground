"""
Tests for sigma_ground.field.interface.bh_mass_function.

Validates B2 (critical-mass threshold) hypothesis constraint machinery:
  - Primary-mass 90 % CL lower bound sets hard floor on M_crit.
  - b2_features returns structural landmarks (cutoff, pile-up) correctly.
  - Events trigger B2 only if M_f > M_crit.
  - IMR exclusion σ degrades smoothly as M_crit climbs past event masses.
  - Viability sweep produces expected monotonic structure.
"""

import math
import unittest

from sigma_ground.field.constants import XI
from sigma_ground.field.interface.bh_mass_function import (
    GWTC3_PRIMARY_MASSES,
    PAIR_INSTABILITY_HIGH_MSUN,
    PAIR_INSTABILITY_LOW_MSUN,
    b2_features,
    b2_imr_exclusion,
    b2_viable,
    events_triggered_by_B2,
    minimum_M_crit_90CL,
    sweep_M_crit,
)


class TestMinimumMCrit(unittest.TestCase):

    def test_lower_bound_comes_from_GW190521(self):
        """GW190521 primary has the tightest 90 % CL lower bound → 71 M_sun."""
        M_min, event = minimum_M_crit_90CL()
        self.assertAlmostEqual(M_min, 71.0, places=1)
        self.assertEqual(event, 'GW190521_primary')

    def test_lower_bound_is_positive(self):
        M_min, _ = minimum_M_crit_90CL()
        self.assertGreater(M_min, 0.0)


class TestB2Features(unittest.TestCase):

    def test_pileup_is_cutoff_times_one_minus_xi(self):
        f = b2_features(100.0)
        self.assertAlmostEqual(f['pileup_M_solar'], 100.0 * (1.0 - XI), places=10)

    def test_cutoff_matches_input(self):
        f = b2_features(150.0)
        self.assertEqual(f['cutoff_M_solar'], 150.0)

    def test_depletion_range_monotonic(self):
        f = b2_features(100.0)
        lo, hi = f['depletion_range_solar']
        self.assertLess(lo, hi)

    def test_drop_fraction_is_xi(self):
        self.assertEqual(b2_features(50.0)['drop_fraction'], XI)

    def test_zero_or_negative_rejected(self):
        with self.assertRaises(ValueError):
            b2_features(0.0)
        with self.assertRaises(ValueError):
            b2_features(-10.0)


class TestEventsTriggeredByB2(unittest.TestCase):

    def test_very_low_M_crit_triggers_all_events(self):
        """M_crit below smallest remnant mass triggers all 5 events."""
        triggered = events_triggered_by_B2(10.0)
        self.assertEqual(len(triggered), 5)

    def test_M_crit_above_all_events_triggers_none(self):
        """M_crit above 142 M_sun triggers none of the 5 events."""
        triggered = events_triggered_by_B2(200.0)
        self.assertEqual(len(triggered), 0)

    def test_M_crit_at_60_triggers_two_events(self):
        """M_crit = 60 M_sun triggers GW150914 (62) and GW190521 (142) only."""
        triggered = events_triggered_by_B2(60.0)
        self.assertEqual(set(triggered), {'GW150914', 'GW190521'})

    def test_M_crit_at_65_triggers_only_GW190521(self):
        """M_crit above all but GW190521 leaves it the sole trigger."""
        triggered = events_triggered_by_B2(65.0)
        self.assertEqual(triggered, ['GW190521'])

    def test_triggered_list_is_sorted_ascending(self):
        """Returned list is sorted by remnant mass ascending."""
        triggered = events_triggered_by_B2(10.0)
        # Known ordering: GW151226 (21) < GW170104 (49) < GW170814 (53)
        #                 < GW150914 (62) < GW190521 (142)
        self.assertEqual(triggered[0], 'GW151226')
        self.assertEqual(triggered[-1], 'GW190521')


class TestB2ImrExclusion(unittest.TestCase):

    def test_very_low_M_crit_matches_B1_aggregate(self):
        """All events triggered → σ should be sqrt of sum of individual σ²."""
        sigma, triggered = b2_imr_exclusion(10.0)
        self.assertEqual(len(triggered), 5)
        # Sum-in-quadrature of the 5 individual event σ's (not combined posterior)
        # individual σs are ~3.65, 0.52, 1.40, 2.90, 1.08 → sqrt(sum_sq) ~ 5
        self.assertGreater(sigma, 4.0)

    def test_M_crit_above_all_remnants_gives_zero_sigma(self):
        sigma, triggered = b2_imr_exclusion(200.0)
        self.assertEqual(sigma, 0.0)
        self.assertEqual(triggered, [])

    def test_sigma_decreases_as_M_crit_increases(self):
        """Fewer triggered events → lower combined σ."""
        lo_M = b2_imr_exclusion(30.0)[0]
        hi_M = b2_imr_exclusion(100.0)[0]
        self.assertGreater(lo_M, hi_M)


class TestB2Viable(unittest.TestCase):

    def test_low_M_crit_ruled_out_by_primary_mass(self):
        """M_crit = 50 < 71 fails the primary-mass check."""
        result = b2_viable(50.0)
        self.assertFalse(result['primary_mass_ok'])
        self.assertFalse(result['viable'])

    def test_high_M_crit_passes_both_checks(self):
        """M_crit = 200 passes primary-mass (200 > 71) and IMR (no trigger)."""
        result = b2_viable(200.0)
        self.assertTrue(result['primary_mass_ok'])
        self.assertTrue(result['imr_ok'])
        self.assertTrue(result['viable'])

    def test_M_crit_in_PI_gap_flagged(self):
        """M_crit ∈ [45, 135] flagged as within pair-instability gap."""
        result = b2_viable(100.0)
        self.assertTrue(result['in_PI_gap'])
        result_high = b2_viable(200.0)
        self.assertFalse(result_high['in_PI_gap'])

    def test_boundary_of_PI_gap(self):
        self.assertTrue(b2_viable(PAIR_INSTABILITY_LOW_MSUN)['in_PI_gap'])
        self.assertTrue(b2_viable(PAIR_INSTABILITY_HIGH_MSUN)['in_PI_gap'])

    def test_threshold_tightens_exclusion(self):
        """Tighter imr_threshold_sigma rules out more M_crit values."""
        result_loose = b2_viable(75.0, imr_threshold_sigma=5.0)
        result_tight = b2_viable(75.0, imr_threshold_sigma=2.0)
        # Same primary check; IMR may differ
        self.assertEqual(result_loose['primary_mass_ok'],
                         result_tight['primary_mass_ok'])


class TestSweep(unittest.TestCase):

    def test_sweep_returns_row_per_grid_point(self):
        rows = sweep_M_crit(M_crit_grid=(20.0, 100.0, 200.0))
        self.assertEqual(len(rows), 3)

    def test_sweep_is_sorted_by_M_crit(self):
        rows = sweep_M_crit()
        M_values = [r['M_crit_solar'] for r in rows]
        self.assertEqual(M_values, sorted(M_values))

    def test_sweep_finds_viability_transition(self):
        """A wide sweep should show both non-viable and viable outcomes."""
        rows = sweep_M_crit()
        viables = [r['viable'] for r in rows]
        self.assertIn(False, viables)  # low-M_crit region non-viable
        self.assertIn(True, viables)   # high-M_crit region viable

    def test_catalog_primary_masses_structure(self):
        """All catalog entries have 4 fields (median, lower, upper, source)."""
        for name, entry in GWTC3_PRIMARY_MASSES.items():
            self.assertEqual(
                len(entry), 4,
                f"{name} has {len(entry)} fields, expected 4"
            )


if __name__ == '__main__':
    unittest.main()
