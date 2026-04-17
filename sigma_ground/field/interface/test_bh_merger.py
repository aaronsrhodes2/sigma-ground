"""
Tests for sigma_ground.field.interface.bh_merger.

Validates analytic BH-merger predictions:
  - Schwarzschild radius for solar mass matches published value (~2953 m).
  - Shell radii form a geometric sequence in ξ.
  - Inward echo delays grow linearly in n at spacing 2·r_s·σ_conv/c.
  - Outward echo delays grow geometrically as ξ⁻ⁿ − 1.
  - γ-at-merger matches Phase G terminal values per mode.
  - predict_named_event round-trips against manual computation.
"""

import math
import unittest

from sigma_ground.field.constants import (
    C, ETA, G, M_SUN_KG, SIGMA_CONV, THETA_ENTANGLEMENT_PER_DIM, XI,
)
from sigma_ground.field.interface.bh_merger import (
    NAMED_EVENTS,
    echo_delays,
    gamma_at_merger,
    predict_event,
    predict_named_event,
    ringdown_amplitude_suppression,
    schwarzschild_radius,
    shell_radii,
)


class TestSchwarzschildRadius(unittest.TestCase):

    def test_solar_mass_matches_canonical_value(self):
        """r_s(M_sun) ≈ 2953 m (published Schwarzschild radius of the Sun).

        Tolerance 3 m reflects rounding in M_SUN_KG = 1.989e30 kg (true
        solar mass is 1.98892e30 kg, a 40-ppm offset).
        """
        r_s = schwarzschild_radius(M_SUN_KG)
        self.assertAlmostEqual(r_s, 2953.0, delta=3.0)

    def test_60_solar_mass_remnant(self):
        """GW150914-class remnant: M ≈ 62.2 M_sun → r_s ≈ 183 km."""
        r_s = schwarzschild_radius(62.2 * M_SUN_KG)
        self.assertAlmostEqual(r_s / 1000.0, 183.0, delta=2.0)

    def test_zero_mass_rejected(self):
        with self.assertRaises(ValueError):
            schwarzschild_radius(0.0)
        with self.assertRaises(ValueError):
            schwarzschild_radius(-1.0)


class TestShellRadii(unittest.TestCase):

    def test_inward_geometric_sequence(self):
        """r_n / r_{n-1} = ξ for inward shells."""
        shells = shell_radii(1.0e30, n_max=5, direction='inward')
        self.assertEqual(len(shells), 5)
        for i in range(1, len(shells)):
            self.assertAlmostEqual(shells[i] / shells[i - 1], XI, places=10)

    def test_outward_geometric_sequence(self):
        """r_n / r_{n-1} = 1/ξ for outward shells."""
        shells = shell_radii(1.0e30, n_max=5, direction='outward')
        self.assertEqual(len(shells), 5)
        for i in range(1, len(shells)):
            self.assertAlmostEqual(shells[i] / shells[i - 1], 1.0 / XI, places=8)

    def test_first_inward_shell_is_r_s_times_xi(self):
        M = 62.2 * M_SUN_KG
        r_s = schwarzschild_radius(M)
        shells = shell_radii(M, n_max=1, direction='inward')
        self.assertAlmostEqual(shells[0], r_s * XI, places=6)

    def test_n_max_zero_returns_empty(self):
        self.assertEqual(shell_radii(1.0e30, n_max=0), [])

    def test_unknown_direction_rejected(self):
        with self.assertRaises(ValueError):
            shell_radii(1.0e30, direction='sideways')


class TestEchoDelays(unittest.TestCase):

    def test_inward_linear_growth(self):
        """Δt_n = 2·r_s·n·σ_conv/c — linear in n."""
        M = 62.2 * M_SUN_KG
        r_s = schwarzschild_radius(M)
        delays = echo_delays(M, n_max=5, direction='inward')
        for n, dt in enumerate(delays, start=1):
            expected = 2.0 * r_s * n * SIGMA_CONV / C
            self.assertAlmostEqual(dt, expected, places=12)

    def test_inward_first_delay_gw150914_scale(self):
        """GW150914 remnant (62.2 M_sun) should give Δt_1 ≈ 2.25 ms."""
        delays = echo_delays(62.2 * M_SUN_KG, n_max=1, direction='inward')
        dt_ms = delays[0] * 1000.0
        self.assertAlmostEqual(dt_ms, 2.25, delta=0.05)

    def test_inward_spacing_is_uniform(self):
        """Δt_{n+1} − Δt_n = 2·r_s·σ_conv/c — constant spacing."""
        M = 100.0 * M_SUN_KG
        r_s = schwarzschild_radius(M)
        delays = echo_delays(M, n_max=5, direction='inward')
        step = 2.0 * r_s * SIGMA_CONV / C
        for i in range(1, len(delays)):
            self.assertAlmostEqual(delays[i] - delays[i - 1], step, places=12)

    def test_outward_geometric_growth(self):
        """Δt_n = 2·r_s·(ξ⁻ⁿ − 1)/c — geometrically growing in n."""
        M = 62.2 * M_SUN_KG
        r_s = schwarzschild_radius(M)
        delays = echo_delays(M, n_max=3, direction='outward')
        for n, dt in enumerate(delays, start=1):
            expected = 2.0 * r_s * ((XI ** (-n)) - 1.0) / C
            self.assertAlmostEqual(dt, expected, places=10)

    def test_outward_first_delay_much_larger_than_inward(self):
        """Outward Δt_1 ≈ 6.5 ms for GW150914, vs inward Δt_1 ≈ 2.25 ms."""
        d_in = echo_delays(62.2 * M_SUN_KG, n_max=1, direction='inward')[0]
        d_out = echo_delays(62.2 * M_SUN_KG, n_max=1, direction='outward')[0]
        self.assertGreater(d_out, d_in)

    def test_delays_scale_linearly_with_mass(self):
        """Δt_n ∝ M (since r_s ∝ M and formula is linear in r_s)."""
        d1 = echo_delays(10.0 * M_SUN_KG, n_max=3, direction='inward')
        d2 = echo_delays(100.0 * M_SUN_KG, n_max=3, direction='inward')
        for a, b in zip(d1, d2):
            self.assertAlmostEqual(b / a, 10.0, places=10)

    def test_n_max_zero_returns_empty(self):
        self.assertEqual(echo_delays(1.0e30, n_max=0), [])


class TestGammaAtMerger(unittest.TestCase):

    def test_sigma_coh_matches_one_minus_eta_over_two(self):
        g = gamma_at_merger(mode='sigma_coh')
        self.assertAlmostEqual(g, 1.0 - ETA / 2.0, places=12)

    def test_linear_and_cbrt_converge_on_theta(self):
        gL = gamma_at_merger(mode='linear')
        gC = gamma_at_merger(mode='cbrt')
        self.assertAlmostEqual(gL, THETA_ENTANGLEMENT_PER_DIM, places=12)
        self.assertAlmostEqual(gC, THETA_ENTANGLEMENT_PER_DIM, places=12)

    def test_exp_terminates_at_theta_plus_tail(self):
        gE = gamma_at_merger(mode='exp')
        expected = THETA_ENTANGLEMENT_PER_DIM + (1.0 - THETA_ENTANGLEMENT_PER_DIM) / math.e
        self.assertAlmostEqual(gE, expected, places=12)

    def test_ordering(self):
        """linear == cbrt (Θ) < sigma_coh (1 − η/2) < exp (Θ + (1−Θ)/e)."""
        gL = gamma_at_merger(mode='linear')
        gC = gamma_at_merger(mode='cbrt')
        gS = gamma_at_merger(mode='sigma_coh')
        gE = gamma_at_merger(mode='exp')
        self.assertAlmostEqual(gL, gC, places=12)
        self.assertLess(gC, gS)
        self.assertLess(gS, gE)

    def test_ringdown_suppression_alias(self):
        """ringdown_amplitude_suppression is an alias for gamma_at_merger."""
        for m in ('linear', 'exp', 'cbrt', 'sigma_coh'):
            self.assertEqual(
                ringdown_amplitude_suppression(mode=m),
                gamma_at_merger(mode=m),
            )


class TestPredictEvent(unittest.TestCase):

    def test_gw150914_named_event(self):
        """Round-trip: predict_named_event('GW150914') matches manual calc."""
        out = predict_named_event('GW150914', n_max=3)
        self.assertEqual(out['name'], 'GW150914')
        self.assertAlmostEqual(out['M_remnant_solar'], 62.2, places=6)
        self.assertAlmostEqual(out['r_s_km'], 183.0, delta=2.0)
        self.assertEqual(len(out['echo_delays_ms']), 3)
        self.assertAlmostEqual(out['echo_delays_ms'][0], 2.25, delta=0.05)
        # Inward spacing is uniform.
        step = out['echo_delays_ms'][1] - out['echo_delays_ms'][0]
        self.assertAlmostEqual(
            out['echo_delays_ms'][2] - out['echo_delays_ms'][1],
            step,
            places=8,
        )

    def test_gw151226_lighter_remnant_shorter_delays(self):
        """~21 M_sun remnant gives shorter echo delays than 62 M_sun."""
        big = predict_named_event('GW150914', n_max=1)
        small = predict_named_event('GW151226', n_max=1)
        self.assertLess(small['echo_delays_ms'][0], big['echo_delays_ms'][0])

    def test_gw190521_heaviest_longest_delays(self):
        """~142 M_sun remnant gives longer echo delays than 62 M_sun."""
        med = predict_named_event('GW150914', n_max=1)
        big = predict_named_event('GW190521', n_max=1)
        self.assertGreater(big['echo_delays_ms'][0], med['echo_delays_ms'][0])

    def test_all_named_events_load(self):
        for name in NAMED_EVENTS:
            out = predict_named_event(name, n_max=3)
            self.assertEqual(out['name'], name)
            self.assertIn('f_QNM_Hz', out)
            self.assertIn('tau_QNM_ms', out)

    def test_gamma_by_mode_has_four_modes(self):
        out = predict_event(62.2, n_max=1)
        self.assertEqual(
            set(out['gamma_by_mode'].keys()),
            {'linear', 'exp', 'cbrt', 'sigma_coh'},
        )

    def test_unknown_named_event_rejected(self):
        with self.assertRaises(ValueError):
            predict_named_event('GW999999')

    def test_zero_mass_rejected(self):
        with self.assertRaises(ValueError):
            predict_event(0.0)
        with self.assertRaises(ValueError):
            predict_event(-1.0)


class TestCrossCheckWithPhysicalScales(unittest.TestCase):
    """Sanity: predicted echo delays must fall within LIGO time resolution
    and within the typical BH ringdown damping time."""

    def test_echo_delays_exceed_ligo_sample_period(self):
        """LIGO samples at 16384 Hz (period ≈ 61 μs).  Δt_1 must exceed
        this comfortably for echoes to be temporally resolvable.
        """
        ligo_sample_period_s = 1.0 / 16384.0
        for name in NAMED_EVENTS:
            out = predict_named_event(name, n_max=1)
            self.assertGreater(out['echo_delays_s'][0], 10.0 * ligo_sample_period_s)

    def test_echo_delays_within_few_ringdown_times(self):
        """Echoes must arrive during the ringdown's damping envelope to be
        detectable.  Require Δt_1 < ~3 · τ_QNM so at least the first echo
        falls inside the detectable ringdown amplitude.
        """
        for name in NAMED_EVENTS:
            out = predict_named_event(name, n_max=1)
            tau_s = out['tau_QNM_ms'] / 1000.0
            self.assertLess(out['echo_delays_s'][0], 3.0 * tau_s)


if __name__ == '__main__':
    unittest.main()
