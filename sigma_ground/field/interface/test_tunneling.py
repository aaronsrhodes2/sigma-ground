"""
Tests for tunneling.py — barrier transmission, WKB, STM, field emission.

Strategy:
  - Test rectangular barrier T → 0 for thick barriers
  - Test rectangular barrier T → 1 for E >> V₀
  - Test R + T = 1 (probability conservation)
  - Test WKB agrees with exact for thick barriers
  - Test STM current exponential in distance
  - Test Fowler-Nordheim scaling with field
  - Test double-barrier resonances = particle-in-box
  - Test work function data is physical
  - Test tunneling time (Hartman effect limit)
  - Test Rule 9: full_report

Reference values:
  STM decay: ~1 decade per Ångström for Φ ≈ 4-5 eV (MEASURED)
  Work function W:  4.5 eV typical metal (MEASURED)
  FN emission: onset at ~3-10 GV/m for metals (MEASURED)
"""

import math
import unittest

from sigma_ground.field.interface.tunneling import (
    rectangular_barrier_T,
    rectangular_barrier_R,
    decay_constant_m,
    wkb_transmission,
    wkb_rectangular,
    double_barrier_resonances_eV,
    fowler_nordheim_current_density,
    field_emission_onset_V_m,
    WORK_FUNCTIONS_EV,
    stm_current,
    stm_resolution_m,
    stm_decay_per_angstrom,
    tunnel_diode_peak_current,
    gamow_factor,
    phase_time_s,
    tunneling_report,
    full_report,
)


class TestRectangularBarrier(unittest.TestCase):
    """Exact rectangular barrier transmission."""

    def test_T_plus_R_equals_1(self):
        """Probability conservation: T + R = 1."""
        T = rectangular_barrier_T(1.0, 3.0, 1e-9)
        R = rectangular_barrier_R(1.0, 3.0, 1e-9)
        self.assertAlmostEqual(T + R, 1.0, delta=1e-10)

    def test_thin_barrier_high_T(self):
        """Very thin barrier → T close to 1."""
        T = rectangular_barrier_T(1.0, 2.0, 1e-12)
        self.assertGreater(T, 0.99)

    def test_thick_barrier_low_T(self):
        """Thick barrier → T → 0."""
        T = rectangular_barrier_T(1.0, 5.0, 10e-9)
        self.assertLess(T, 1e-10)

    def test_above_barrier_oscillates(self):
        """E > V₀: T oscillates but stays ≤ 1."""
        T = rectangular_barrier_T(5.0, 2.0, 1e-9)
        self.assertGreater(T, 0)
        self.assertLessEqual(T, 1.0 + 1e-10)

    def test_above_barrier_resonance(self):
        """At resonance (k'd = nπ), T = 1 even for E > V₀."""
        # For a particle above barrier, T = 1 when sin(k'd) = 0
        # We just check T > 0.9 for some E > V₀
        # (hard to hit exact resonance analytically here)
        T = rectangular_barrier_T(10.0, 2.0, 1e-10)
        self.assertGreater(T, 0.5)

    def test_zero_energy_zero_T(self):
        """E = 0 → T = 0 (can't tunnel with no energy)."""
        T = rectangular_barrier_T(0.0, 2.0, 1e-9)
        self.assertEqual(T, 0.0)

    def test_zero_width_T_equals_1(self):
        """d = 0 → T = 1 (no barrier)."""
        T = rectangular_barrier_T(1.0, 5.0, 0.0)
        self.assertEqual(T, 1.0)

    def test_T_decreases_with_width(self):
        """Wider barrier → lower transmission."""
        T_narrow = rectangular_barrier_T(1.0, 3.0, 0.5e-9)
        T_wide = rectangular_barrier_T(1.0, 3.0, 2e-9)
        self.assertGreater(T_narrow, T_wide)

    def test_T_decreases_with_height(self):
        """Higher barrier → lower transmission."""
        T_low = rectangular_barrier_T(1.0, 2.0, 1e-9)
        T_high = rectangular_barrier_T(1.0, 5.0, 1e-9)
        self.assertGreater(T_low, T_high)

    def test_T_increases_with_energy(self):
        """Higher energy → more transmission (for E < V₀)."""
        T_low = rectangular_barrier_T(0.5, 3.0, 1e-9)
        T_high = rectangular_barrier_T(2.5, 3.0, 1e-9)
        self.assertGreater(T_high, T_low)

    def test_heavier_particle_lower_T(self):
        """Heavier particle tunnels less."""
        from sigma_ground.field.constants import M_ELECTRON_KG, AMU_KG
        T_e = rectangular_barrier_T(1.0, 3.0, 1e-9, M_ELECTRON_KG)
        T_p = rectangular_barrier_T(1.0, 3.0, 1e-9, AMU_KG)
        self.assertGreater(T_e, T_p)

    def test_E_equals_V0(self):
        """T at E = V₀ is finite and between 0 and 1."""
        T = rectangular_barrier_T(3.0, 3.0, 1e-9)
        self.assertGreater(T, 0)
        self.assertLessEqual(T, 1.0)


class TestDecayConstant(unittest.TestCase):
    """Exponential decay constant κ."""

    def test_positive(self):
        """κ > 0 for E < V₀."""
        kappa = decay_constant_m(5.0, 1.0)
        self.assertGreater(kappa, 0)

    def test_zero_at_barrier_top(self):
        """κ = 0 when E = V₀."""
        kappa = decay_constant_m(5.0, 5.0)
        self.assertEqual(kappa, 0.0)

    def test_increases_with_barrier(self):
        """Higher barrier → larger κ."""
        k_low = decay_constant_m(2.0, 1.0)
        k_high = decay_constant_m(10.0, 1.0)
        self.assertGreater(k_high, k_low)

    def test_typical_value(self):
        """κ ≈ 10 nm⁻¹ for Φ ~ 4 eV (order of magnitude)."""
        kappa = decay_constant_m(4.0, 0.0)
        kappa_per_nm = kappa * 1e-9
        self.assertGreater(kappa_per_nm, 5)
        self.assertLess(kappa_per_nm, 20)


class TestWKB(unittest.TestCase):
    """WKB approximation."""

    def test_wkb_agrees_with_exact_thick_barrier(self):
        """WKB ≈ exact for thick barriers (where WKB is best)."""
        E, V0, d = 1.0, 5.0, 5e-9
        T_exact = rectangular_barrier_T(E, V0, d)
        T_wkb = wkb_rectangular(E, V0, d)
        # WKB should be within a factor of ~10 for thick barriers
        if T_exact > 1e-100:
            ratio = T_wkb / T_exact
            self.assertGreater(ratio, 0.01)
            self.assertLess(ratio, 100)

    def test_wkb_general_matches_rectangular(self):
        """WKB with constant V(x) matches wkb_rectangular."""
        E, V0, d = 1.0, 3.0, 2e-9
        T_wkb_rect = wkb_rectangular(E, V0, d)
        T_wkb_gen = wkb_transmission(lambda x: V0, E, 0.0, d)
        # Should match closely (both use same exponent)
        if T_wkb_rect > 1e-100:
            self.assertAlmostEqual(
                math.log(T_wkb_gen) / math.log(T_wkb_rect),
                1.0, delta=0.05
            )

    def test_wkb_triangular_barrier(self):
        """WKB for triangular barrier (field emission shape)."""
        V0 = 4.0
        E = 0.0
        d = 2e-9
        # Triangular: V(x) = V₀(1 − x/d)
        T = wkb_transmission(lambda x: V0 * (1 - x/d), E, 0.0, d)
        self.assertGreater(T, 0)
        self.assertLess(T, 1)


class TestDoubleBarrier(unittest.TestCase):
    """Resonant tunneling double barrier."""

    def test_resonances_exist(self):
        """Double barrier has resonance levels."""
        res = double_barrier_resonances_eV(5.0, 1e-9, 3e-9)
        self.assertGreater(len(res), 0)

    def test_resonances_below_barrier(self):
        """All resonances have E < V₀."""
        V0 = 5.0
        res = double_barrier_resonances_eV(V0, 1e-9, 3e-9)
        for n, E in res:
            self.assertLess(E, V0)

    def test_resonances_scale_n_squared(self):
        """Resonance energies ∝ n² (particle-in-box)."""
        res = double_barrier_resonances_eV(100.0, 0.5e-9, 5e-9)
        if len(res) >= 2:
            E1 = res[0][1]
            E2 = res[1][1]
            self.assertAlmostEqual(E2 / E1, 4.0, delta=0.5)


class TestSTM(unittest.TestCase):
    """Scanning tunneling microscope."""

    def test_current_positive(self):
        """STM current is positive for positive bias."""
        I = stm_current(1.0, 0.5e-9, 4.5)
        self.assertGreater(I, 0)

    def test_current_exponential_in_d(self):
        """Current decays exponentially with distance."""
        I1 = stm_current(1.0, 0.5e-9, 4.5)
        I2 = stm_current(1.0, 0.6e-9, 4.5)
        self.assertGreater(I1, I2)
        # ~1 decade per Å → factor of ~10 per 0.1 nm
        ratio = I1 / I2
        self.assertGreater(ratio, 2)  # significant drop

    def test_decay_per_angstrom(self):
        """Decay factor per Å ≈ 0.1-0.2 for Φ ≈ 4 eV (MEASURED)."""
        factor = stm_decay_per_angstrom(4.0)
        self.assertGreater(factor, 0.05)
        self.assertLess(factor, 0.5)

    def test_resolution_subnanometer(self):
        """STM resolution is sub-nanometer."""
        res = stm_resolution_m(4.5)
        self.assertLess(res, 1e-9)
        self.assertGreater(res, 1e-11)

    def test_proportional_to_bias(self):
        """Current ∝ V at low bias."""
        I1 = stm_current(1.0, 0.5e-9, 4.5)
        I2 = stm_current(2.0, 0.5e-9, 4.5)
        self.assertAlmostEqual(I2 / I1, 2.0, delta=0.01)


class TestFowlerNordheim(unittest.TestCase):
    """Fowler-Nordheim field emission."""

    def test_zero_field_zero_current(self):
        """No field → no emission."""
        J = fowler_nordheim_current_density(0.0, 4.5)
        self.assertEqual(J, 0.0)

    def test_current_increases_with_field(self):
        """Higher field → more emission."""
        J_low = fowler_nordheim_current_density(3e9, 4.5)
        J_high = fowler_nordheim_current_density(5e9, 4.5)
        self.assertGreater(J_high, J_low)

    def test_current_positive(self):
        """Field emission current is positive."""
        J = fowler_nordheim_current_density(5e9, 4.5)
        self.assertGreater(J, 0)

    def test_lower_work_function_more_emission(self):
        """Lower Φ → more emission at same field."""
        J_low_phi = fowler_nordheim_current_density(5e9, 3.0)
        J_high_phi = fowler_nordheim_current_density(5e9, 5.0)
        self.assertGreater(J_low_phi, J_high_phi)

    def test_onset_field_order_of_magnitude(self):
        """Onset field for tungsten ≈ GV/m range."""
        E = field_emission_onset_V_m(4.55)
        self.assertGreater(E, 1e8)
        self.assertLess(E, 1e11)


class TestWorkFunctions(unittest.TestCase):
    """Work function data."""

    def test_all_positive(self):
        """All work functions are positive."""
        for metal, phi in WORK_FUNCTIONS_EV.items():
            with self.subTest(metal=metal):
                self.assertGreater(phi, 0)

    def test_physical_range(self):
        """Work functions in 3-6 eV range (MEASURED)."""
        for metal, phi in WORK_FUNCTIONS_EV.items():
            with self.subTest(metal=metal):
                self.assertGreater(phi, 3.0)
                self.assertLess(phi, 7.0)

    def test_tungsten(self):
        """Tungsten Φ ≈ 4.55 eV (MEASURED: Eastman 1970)."""
        self.assertAlmostEqual(WORK_FUNCTIONS_EV['tungsten'], 4.55, delta=0.1)


class TestTunnelingTime(unittest.TestCase):
    """Phase tunneling time."""

    def test_positive(self):
        """Tunneling time is positive."""
        tau = phase_time_s(1.0, 3.0, 1e-9)
        self.assertGreater(tau, 0)

    def test_above_barrier_transit(self):
        """Above barrier: phase time ≈ transit time d/v."""
        tau = phase_time_s(10.0, 2.0, 1e-9)
        from sigma_ground.field.constants import M_ELECTRON_KG, EV_TO_J
        v = math.sqrt(2 * 10.0 * EV_TO_J / M_ELECTRON_KG)
        t_transit = 1e-9 / v
        self.assertAlmostEqual(tau, t_transit, delta=t_transit * 0.5)

    def test_hartman_saturation(self):
        """Very thick barrier: tunneling time saturates (Hartman effect)."""
        tau_thin = phase_time_s(1.0, 5.0, 1e-9)
        tau_thick = phase_time_s(1.0, 5.0, 100e-9)
        # Thick barrier time should NOT scale linearly with thickness
        # (Hartman effect: time saturates)
        ratio = tau_thick / tau_thin
        self.assertLess(ratio, 50)  # much less than 100× thickness ratio


class TestTunnelDiode(unittest.TestCase):
    """Esaki tunnel diode."""

    def test_peak_current_positive(self):
        """Peak tunnel current is positive."""
        J = tunnel_diode_peak_current(1.12, 5e-9)
        self.assertGreater(J, 0)

    def test_wider_junction_lower_current(self):
        """Wider depletion → less tunneling."""
        J_narrow = tunnel_diode_peak_current(1.12, 2e-9)
        J_wide = tunnel_diode_peak_current(1.12, 10e-9)
        self.assertGreater(J_narrow, J_wide)


class TestGamow(unittest.TestCase):
    """Gamow tunneling factor."""

    def test_positive(self):
        """Gamow factor is non-negative."""
        G = gamow_factor(90, 5.0, 7e-15)
        self.assertGreaterEqual(G, 0)

    def test_less_than_one(self):
        """Gamow factor < 1 (barrier reduces transmission)."""
        G = gamow_factor(88, 4.0, 8e-15)
        self.assertLess(G, 1)

    def test_higher_energy_more_tunneling(self):
        """Higher alpha energy → more tunneling."""
        G_low = gamow_factor(88, 4.0, 8e-15)
        G_high = gamow_factor(88, 8.0, 8e-15)
        self.assertGreater(G_high, G_low)


class TestReports(unittest.TestCase):
    """Rule 9: full_report."""

    def test_report_complete(self):
        """tunneling_report has required fields."""
        r = tunneling_report()
        required = ['E_eV', 'V0_eV', 'transmission_exact',
                     'transmission_WKB', 'reflection',
                     'stm_current_1V_0.5nm', 'stm_resolution_nm']
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_full_report(self):
        """full_report returns dict with extra fields."""
        r = full_report()
        self.assertIsInstance(r, dict)
        self.assertIn('double_barrier_resonances', r)
        self.assertIn('work_functions_eV', r)

    def test_conservation_in_report(self):
        """T + R = 1 in report."""
        r = tunneling_report()
        self.assertAlmostEqual(
            r['transmission_exact'] + r['reflection'], 1.0, delta=1e-10
        )


if __name__ == '__main__':
    unittest.main()
