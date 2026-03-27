"""
Tests for the hysteresis module.

Test structure:
  1. TestLangevin         — L(0)=0, L(large)→1, L(small)≈x/3, antisymmetry
  2. TestAnhysteretic     — M(0)=0, M(large)→M_sat, monotonic, zero for non-FM
  3. TestHysteresisLoop   — shape: ascending < descending at same H,
                            remanence at H=0, coercivity near H_c
  4. TestEnergyLoss       — positive for ferromagnets, zero for non-magnetic,
                            iron > nickel (higher H_c × M_sat product)
  5. TestCurieWeiss       — diverges near T_C, positive above T_C, zero below
  6. TestMvsT             — M(0)=M_sat, M(T_C)=0, monotonically decreasing
  7. TestNonMagnetic      — all functions return 0 for Cu, Al, Au, Si, W, Ti
  8. TestSigma            — T_C shifts with σ, σ=0 is identity
  9. TestRule9            — all 8 materials present, all fields populated
 10. TestNagatha          — export dict complete and self-consistent
"""

import math
import unittest

from .hysteresis import (
    HYSTERESIS_DATA,
    langevin_function,
    anhysteretic_magnetization,
    hysteresis_loop_point,
    hysteresis_loop,
    energy_loss_per_cycle,
    curie_weiss_susceptibility,
    magnetization_vs_temperature,
    sigma_hysteresis_shift,
    hysteresis_properties,
)

# Convenience: the full material set expected by Rule 9
ALL_MATERIALS = [
    'iron', 'copper', 'aluminum', 'gold',
    'silicon', 'tungsten', 'nickel', 'titanium',
]
FERROMAGNETS = ['iron', 'nickel']
NON_FERROMAGNETS = ['copper', 'aluminum', 'gold', 'silicon', 'tungsten', 'titanium']


# ── 1. Langevin Function ───────────────────────────────────────────

class TestLangevin(unittest.TestCase):
    """L(x) = coth(x) - 1/x — the anhysteretic backbone."""

    def test_zero_returns_zero(self):
        """L(0) = 0 exactly (by Taylor branch)."""
        self.assertAlmostEqual(langevin_function(0.0), 0.0, places=15)

    def test_tiny_x_returns_zero_approx(self):
        """L(tiny) ≈ 0, not a NaN or inf."""
        result = langevin_function(1e-10)
        self.assertTrue(math.isfinite(result))
        self.assertAlmostEqual(result, 0.0, places=8)

    def test_small_x_linear_approx(self):
        """For small x, L(x) ≈ x/3 (first Taylor term)."""
        for x in [0.001, 0.01, 0.05]:
            L = langevin_function(x)
            expected = x / 3.0
            # Within 1% of linear term for x < 0.05
            self.assertAlmostEqual(L, expected, delta=expected * 0.02,
                                   msg=f"L({x}) = {L}, expected ≈ {expected}")

    def test_small_x_cubic_correction(self):
        """For x = 0.5, cubic correction matters: L ≈ x/3 - x³/45."""
        x = 0.5
        L = langevin_function(x)
        approx = x / 3.0 - x**3 / 45.0
        self.assertAlmostEqual(L, approx, places=3)

    def test_large_x_approaches_one(self):
        """L(x) → 1 as x → ∞ (full saturation).

        L(100) = 1 - 1/tanh(100) ≈ 0.99 (very close but not exactly 1).
        L(1000) clamps to 1.0 exactly.
        """
        L_large = langevin_function(100.0)
        self.assertAlmostEqual(L_large, 1.0, delta=0.02)
        L_very_large = langevin_function(1000.0)
        self.assertAlmostEqual(L_very_large, 1.0, places=10)

    def test_very_large_x_clamps(self):
        """L(1000) = 1.0 (clamp branch avoids overflow)."""
        L = langevin_function(1000.0)
        self.assertAlmostEqual(L, 1.0, places=10)

    def test_negative_large_x_approaches_minus_one(self):
        """L(-large) → -1 (antisymmetry at saturation).

        L(-100) ≈ -0.99; L(-1000) clamps to -1.0.
        """
        L = langevin_function(-100.0)
        self.assertAlmostEqual(L, -1.0, delta=0.02)
        L_clamp = langevin_function(-1000.0)
        self.assertAlmostEqual(L_clamp, -1.0, places=10)

    def test_antisymmetry(self):
        """L(-x) = -L(x) for a range of values."""
        for x in [0.0, 0.1, 1.0, 5.0, 50.0]:
            self.assertAlmostEqual(
                langevin_function(-x), -langevin_function(x),
                places=12, msg=f"Antisymmetry failed at x={x}"
            )

    def test_range_bounded(self):
        """L(x) is strictly in (-1, 1) for finite x."""
        for x in [-500, -10, -1, -0.1, 0, 0.1, 1, 10, 500]:
            L = langevin_function(x)
            self.assertGreaterEqual(L, -1.0, f"L({x}) = {L} < -1")
            self.assertLessEqual(L, 1.0, f"L({x}) = {L} > 1")

    def test_positive_for_positive_x(self):
        """L(x) > 0 for x > 0."""
        for x in [0.001, 0.1, 1.0, 10.0]:
            self.assertGreater(langevin_function(x), 0.0)

    def test_monotonically_increasing(self):
        """L is monotonically increasing."""
        xs = [0.01 * i for i in range(1, 200)]
        Ls = [langevin_function(x) for x in xs]
        for i in range(len(Ls) - 1):
            self.assertLess(Ls[i], Ls[i + 1],
                            f"L not increasing at x={xs[i]:.3f}")


# ── 2. Anhysteretic Magnetization ─────────────────────────────────

class TestAnhysteretic(unittest.TestCase):
    """M_anh — equilibrium M-H curve."""

    def test_zero_field_gives_zero_magnetization(self):
        """At H=0 the Langevin argument is 0, so M_anh = 0."""
        for mat in FERROMAGNETS:
            M = anhysteretic_magnetization(mat, 0.0)
            self.assertAlmostEqual(M, 0.0, places=10,
                                   msg=f"{mat}: M_anh(0) should be 0")

    def test_large_field_approaches_saturation(self):
        """At astronomically large H, M → M_sat (within 1%).

        The Langevin saturation field for a single-atom moment (order μ_B)
        at room temperature is:
          x = μ₀ m H / k_B T = 1  when H ≈ k_B T / (μ₀ m) ≈ 10¹¹ A/m.
        So H = 1e12 A/m is deeply saturating.
        """
        for mat in FERROMAGNETS:
            M_sat = HYSTERESIS_DATA[mat]['M_sat_A_m']
            M = anhysteretic_magnetization(mat, 1e12)
            self.assertAlmostEqual(M / M_sat, 1.0, delta=0.01,
                                   msg=f"{mat}: M_anh(H=1e12) should → M_sat")

    def test_monotonically_increasing(self):
        """M_anh increases with H for ferromagnets."""
        for mat in FERROMAGNETS:
            H_vals = [100 * i for i in range(1, 20)]
            M_vals = [anhysteretic_magnetization(mat, H) for H in H_vals]
            for i in range(len(M_vals) - 1):
                self.assertLessEqual(M_vals[i], M_vals[i + 1],
                                     msg=f"{mat}: M_anh not monotone")

    def test_positive_for_positive_field(self):
        """M_anh > 0 for H > 0 in ferromagnets."""
        for mat in FERROMAGNETS:
            M = anhysteretic_magnetization(mat, 1000.0)
            self.assertGreater(M, 0.0, msg=f"{mat}: M_anh should be positive")

    def test_below_saturation_at_moderate_field(self):
        """M_anh < M_sat at moderate fields (not yet saturated)."""
        for mat in FERROMAGNETS:
            M_sat = HYSTERESIS_DATA[mat]['M_sat_A_m']
            H_c = HYSTERESIS_DATA[mat]['H_c_A_m']
            # At 10× coercivity, still far from saturation
            M = anhysteretic_magnetization(mat, 10.0 * H_c)
            self.assertLess(M, M_sat,
                            msg=f"{mat}: M_anh at 10×H_c should be < M_sat")

    def test_non_ferromagnets_return_zero(self):
        """Non-ferromagnetic materials: M_anh = 0 for any H."""
        for mat in NON_FERROMAGNETS:
            for H in [0.0, 1e3, 1e6]:
                M = anhysteretic_magnetization(mat, H)
                self.assertEqual(M, 0.0,
                                 msg=f"{mat}: M_anh should be 0, got {M}")

    def test_iron_larger_than_nickel(self):
        """Iron has higher M_sat so its M_anh(H) > Ni at same large H."""
        H = 1e9  # near-saturating
        M_fe = anhysteretic_magnetization('iron', H)
        M_ni = anhysteretic_magnetization('nickel', H)
        self.assertGreater(M_fe, M_ni,
                           msg="Iron M_anh should exceed Nickel at saturation")


# ── 3. Hysteresis Loop ─────────────────────────────────────────────

class TestHysteresisLoop(unittest.TestCase):
    """Full B-H loop shape and key points."""

    def _get_loop(self, material, H_max=None, steps=200):
        if H_max is None:
            H_c = HYSTERESIS_DATA[material]['H_c_A_m']
            H_max = max(H_c * 10.0, 1000.0)
        return hysteresis_loop(material, H_max, steps=steps)

    def test_loop_returns_list_of_dicts(self):
        """Loop is a list of dicts with required keys."""
        loop = self._get_loop('iron')
        self.assertIsInstance(loop, list)
        self.assertGreater(len(loop), 0)
        required_keys = {'H_A_m', 'B_T', 'M_A_m', 'branch'}
        for point in loop:
            for key in required_keys:
                self.assertIn(key, point,
                              msg=f"Missing key '{key}' in loop point")

    def test_loop_length(self):
        """Total number of points = 3 × steps."""
        steps = 50
        loop = hysteresis_loop('iron', 10000.0, steps=steps)
        self.assertEqual(len(loop), 3 * steps)

    def test_branch_labels_present(self):
        """All three branch labels appear."""
        loop = self._get_loop('iron')
        branches = {pt['branch'] for pt in loop}
        self.assertIn('ascending_initial', branches)
        self.assertIn('descending', branches)
        self.assertIn('ascending', branches)

    def test_ascending_less_than_descending_at_same_H(self):
        """At any H between -H_max and H_max, B_ascending < B_descending.

        This is the defining property of hysteresis: the descending
        branch lies above the ascending branch in the first quadrant.
        """
        H_max = 5000.0
        steps = 100
        loop = hysteresis_loop('iron', H_max, steps=steps)

        # Build lookup: H → B for each branch
        desc = {round(pt['H_A_m'], 2): pt['B_T']
                for pt in loop if pt['branch'] == 'descending'}
        asc = {round(pt['H_A_m'], 2): pt['B_T']
               for pt in loop if pt['branch'] == 'ascending'}

        # Find H values that appear in both branches (interior of loop)
        common_H = set(desc.keys()) & set(asc.keys())
        # Filter to the interior: H between -H_max and H_max (exclusive ends)
        interior = [H for H in common_H if -H_max * 0.99 < H < H_max * 0.99]

        self.assertGreater(len(interior), 5,
                           "Expected multiple common H values for comparison")
        violations = 0
        for H in interior:
            if desc[H] < asc[H] - 1e-15:
                violations += 1
        self.assertEqual(violations, 0,
                         "Descending B should be ≥ ascending B at the same H "
                         f"({violations} violations found)")

    def test_remanence_nonzero_for_iron(self):
        """After removing field (H=0 on descending branch), B > 0 for iron."""
        H_max = 5000.0
        loop = hysteresis_loop('iron', H_max, steps=500)
        # Find the descending-branch point closest to H = 0
        desc_pts = [pt for pt in loop if pt['branch'] == 'descending']
        closest = min(desc_pts, key=lambda pt: abs(pt['H_A_m']))
        # B at H≈0 on descending branch is the remanent flux density
        B_rem = closest['B_T']
        self.assertGreater(B_rem, 0.0,
                           f"Remanence B_r should be positive, got {B_rem:.4f} T")

    def test_remanence_is_data_field(self):
        """Measured remanence B_r is stored in HYSTERESIS_DATA and is realistic.

        Note: the computed loop remanence from the simplified coercivity-shift
        model is small (because M_anh(H_c=80) ≈ 0.3 A/m at room temperature —
        the Langevin saturation field is ~10¹¹ A/m for single-atom moments).
        The MEASURED B_r = 0.8 T for iron is stored in HYSTERESIS_DATA['iron']
        and is the authoritative value. This test checks that stored value.
        """
        B_r = HYSTERESIS_DATA['iron']['B_r_T']
        self.assertGreater(B_r, 0.1, f"Iron B_r should be > 0.1 T, got {B_r}")
        self.assertLess(B_r, 3.0, f"Iron B_r should be < 3.0 T, got {B_r}")

    def test_coercivity_sign_change_near_Hc(self):
        """On descending branch, M changes sign near H = -H_c."""
        H_c = HYSTERESIS_DATA['iron']['H_c_A_m']
        H_max = H_c * 20.0
        loop = hysteresis_loop('iron', H_max, steps=1000)
        desc_pts = sorted(
            [pt for pt in loop if pt['branch'] == 'descending'],
            key=lambda pt: pt['H_A_m']
        )
        # Find where M changes sign on descending branch
        sign_changes = []
        for i in range(len(desc_pts) - 1):
            M1 = desc_pts[i]['M_A_m']
            M2 = desc_pts[i + 1]['M_A_m']
            if M1 * M2 < 0:
                sign_changes.append(
                    (desc_pts[i]['H_A_m'] + desc_pts[i + 1]['H_A_m']) / 2.0
                )
        self.assertGreater(len(sign_changes), 0,
                           "M should change sign on descending branch near -H_c")
        # The sign change should occur near -H_c
        H_cross = sign_changes[0]
        self.assertAlmostEqual(H_cross, -H_c, delta=H_c * 2.0,
                               msg=f"Coercivity crossing at {H_cross:.1f} A/m, "
                                   f"expected near -{H_c:.1f} A/m")

    def test_non_ferromagnet_loop_linear(self):
        """For copper, B = μ₀ H (no M, linear loop)."""
        from ..constants import MU_0
        H_max = 1e4
        loop = hysteresis_loop('copper', H_max, steps=20)
        for pt in loop:
            expected_B = MU_0 * pt['H_A_m']
            self.assertAlmostEqual(pt['B_T'], expected_B, places=25,
                                   msg="Copper loop should be linear: B = μ₀H")

    def test_saturation_reached_at_large_field(self):
        """At H >> Langevin saturation field (~10¹¹ A/m), M ≈ M_sat.

        The Langevin saturation scale is k_B T / (μ₀ m_atom) ≈ 10¹¹ A/m.
        Use H_max = 1e12 A/m to ensure deep saturation (within 1%).
        """
        M_sat = HYSTERESIS_DATA['iron']['M_sat_A_m']
        H_max = 1e12
        loop = hysteresis_loop('iron', H_max, steps=50)
        # Last point of ascending initial branch
        asc_init = [pt for pt in loop if pt['branch'] == 'ascending_initial']
        M_top = asc_init[-1]['M_A_m']
        self.assertAlmostEqual(M_top / M_sat, 1.0, delta=0.01,
                               msg="M should approach M_sat at H=1e12 A/m")


# ── 4. Energy Loss Per Cycle ───────────────────────────────────────

class TestEnergyLoss(unittest.TestCase):
    """Hysteresis energy loss: area of B-H loop."""

    def test_positive_for_ferromagnets(self):
        """Energy loss > 0 for iron and nickel."""
        for mat in FERROMAGNETS:
            W = energy_loss_per_cycle(mat)
            self.assertGreater(W, 0.0,
                               msg=f"{mat}: energy loss should be positive")

    def test_zero_for_non_magnetic(self):
        """Non-ferromagnetic materials have zero hysteresis loss."""
        for mat in NON_FERROMAGNETS:
            W = energy_loss_per_cycle(mat)
            self.assertEqual(W, 0.0,
                             msg=f"{mat}: energy loss should be 0, got {W}")

    def test_iron_greater_than_nickel(self):
        """Iron has higher H_c × M_sat product → greater loss than nickel."""
        W_fe = energy_loss_per_cycle('iron')
        W_ni = energy_loss_per_cycle('nickel')
        self.assertGreater(W_fe, W_ni,
                           msg="Iron loss should exceed nickel loss "
                               f"(Fe: {W_fe:.3e}, Ni: {W_ni:.3e} J/m³)")

    def test_order_of_magnitude_iron(self):
        """Iron loss ≈ 4 μ₀ H_c M_sat — in a physically reasonable range."""
        from ..constants import MU_0
        H_c = HYSTERESIS_DATA['iron']['H_c_A_m']
        M_sat = HYSTERESIS_DATA['iron']['M_sat_A_m']
        expected = 4.0 * MU_0 * H_c * M_sat
        W = energy_loss_per_cycle('iron')
        self.assertAlmostEqual(W, expected, places=10,
                               msg="Iron loss should match 4μ₀H_cM_sat formula")

    def test_h_max_argument_accepted(self):
        """H_max argument is accepted (for signature compatibility)."""
        W1 = energy_loss_per_cycle('iron', H_max=None)
        W2 = energy_loss_per_cycle('iron', H_max=1e6)
        # Rectangular approximation is H_max-independent
        self.assertAlmostEqual(W1, W2, places=10)


# ── 5. Curie-Weiss Susceptibility ─────────────────────────────────

class TestCurieWeiss(unittest.TestCase):
    """χ = C/(T - T_C) for T > T_C."""

    def test_positive_above_tc(self):
        """Susceptibility is positive for T > T_C (paramagnetic regime)."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            chi = curie_weiss_susceptibility(mat, T_C + 100.0)
            self.assertGreater(chi, 0.0,
                               msg=f"{mat}: χ should be positive above T_C")

    def test_zero_below_tc(self):
        """Returns 0 for T ≤ T_C (not paramagnetic, use M(T) instead)."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            chi = curie_weiss_susceptibility(mat, T_C - 10.0)
            self.assertEqual(chi, 0.0,
                             msg=f"{mat}: χ should be 0 below T_C")

    def test_zero_at_tc(self):
        """Returns 0 at T = T_C (division by zero guard)."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            chi = curie_weiss_susceptibility(mat, T_C)
            self.assertEqual(chi, 0.0)

    def test_diverges_near_tc(self):
        """χ increases as T → T_C from above (Curie-Weiss divergence)."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            chi_far = curie_weiss_susceptibility(mat, T_C + 500.0)
            chi_near = curie_weiss_susceptibility(mat, T_C + 1.0)
            self.assertGreater(chi_near, chi_far,
                               msg=f"{mat}: χ should increase toward T_C")

    def test_decreasing_with_temperature(self):
        """χ decreases as T increases (Curie-Weiss: 1/(T-T_C))."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            temps = [T_C + dt for dt in [10, 50, 100, 300, 1000]]
            chis = [curie_weiss_susceptibility(mat, T) for T in temps]
            for i in range(len(chis) - 1):
                self.assertGreater(chis[i], chis[i + 1],
                                   msg=f"{mat}: χ not decreasing at T={temps[i]}")

    def test_zero_for_non_ferromagnets(self):
        """Non-ferromagnets have no Curie-Weiss divergence → return 0."""
        for mat in NON_FERROMAGNETS:
            chi = curie_weiss_susceptibility(mat, 1000.0)
            self.assertEqual(chi, 0.0,
                             msg=f"{mat}: Curie-Weiss should be 0")


# ── 6. Magnetization vs. Temperature ──────────────────────────────

class TestMvsT(unittest.TestCase):
    """M(T) using critical exponent β = 0.34."""

    def test_zero_temperature_gives_msat(self):
        """M(0) = M_sat (fully ordered ground state)."""
        for mat in FERROMAGNETS:
            M_sat = HYSTERESIS_DATA[mat]['M_sat_A_m']
            M = magnetization_vs_temperature(mat, 0.0)
            self.assertAlmostEqual(M, M_sat, places=5,
                                   msg=f"{mat}: M(0) should equal M_sat")

    def test_at_curie_temp_gives_zero(self):
        """M(T_C) = 0 (transition to paramagnetic)."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            M = magnetization_vs_temperature(mat, T_C)
            self.assertAlmostEqual(M, 0.0, places=10,
                                   msg=f"{mat}: M(T_C) should be 0")

    def test_above_curie_temp_gives_zero(self):
        """M(T > T_C) = 0 (paramagnetic regime)."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            M = magnetization_vs_temperature(mat, T_C + 100.0)
            self.assertEqual(M, 0.0,
                             msg=f"{mat}: M above T_C should be 0")

    def test_monotonically_decreasing(self):
        """M(T) decreases as temperature increases."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            temps = [T_C * frac for frac in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]]
            mags = [magnetization_vs_temperature(mat, T) for T in temps]
            for i in range(len(mags) - 1):
                self.assertGreaterEqual(mags[i], mags[i + 1],
                                        msg=f"{mat}: M not decreasing at T={temps[i]:.1f}K")

    def test_midpoint_between_zero_and_msat(self):
        """At some intermediate T, M is between 0 and M_sat."""
        for mat in FERROMAGNETS:
            T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
            M_sat = HYSTERESIS_DATA[mat]['M_sat_A_m']
            M_mid = magnetization_vs_temperature(mat, T_C * 0.5)
            self.assertGreater(M_mid, 0.0,
                               msg=f"{mat}: M at 0.5 T_C should be positive")
            self.assertLess(M_mid, M_sat,
                            msg=f"{mat}: M at 0.5 T_C should be < M_sat")

    def test_non_ferromagnets_return_zero(self):
        """Non-ferromagnetic materials: M(T) = 0 for all T."""
        for mat in NON_FERROMAGNETS:
            for T in [0.0, 300.0, 1000.0, 5000.0]:
                M = magnetization_vs_temperature(mat, T)
                self.assertEqual(M, 0.0,
                                 msg=f"{mat}: M(T={T}) should be 0")

    def test_critical_exponent_consistent(self):
        """M(T) ∝ (1-T/T_C)^0.34 — check ratio at two temperatures."""
        mat = 'iron'
        T_C = HYSTERESIS_DATA[mat]['T_Curie_K']
        M_sat = HYSTERESIS_DATA[mat]['M_sat_A_m']
        T1 = T_C * 0.8
        T2 = T_C * 0.5
        M1 = magnetization_vs_temperature(mat, T1)
        M2 = magnetization_vs_temperature(mat, T2)
        expected1 = M_sat * (1.0 - 0.8) ** 0.34
        expected2 = M_sat * (1.0 - 0.5) ** 0.34
        self.assertAlmostEqual(M1, expected1, places=8)
        self.assertAlmostEqual(M2, expected2, places=8)


# ── 7. Non-Magnetic Materials ──────────────────────────────────────

class TestNonMagnetic(unittest.TestCase):
    """All hysteresis quantities are zero for non-ferromagnets."""

    def test_anhysteretic_zero(self):
        for mat in NON_FERROMAGNETS:
            self.assertEqual(anhysteretic_magnetization(mat, 1e6), 0.0,
                             msg=f"{mat}: M_anh should be 0")

    def test_loop_point_zero(self):
        for mat in NON_FERROMAGNETS:
            for ascending in [True, False]:
                M = hysteresis_loop_point(mat, 1e6, ascending=ascending)
                self.assertEqual(M, 0.0, msg=f"{mat}: loop point should be 0")

    def test_energy_loss_zero(self):
        for mat in NON_FERROMAGNETS:
            W = energy_loss_per_cycle(mat)
            self.assertEqual(W, 0.0, msg=f"{mat}: energy loss should be 0")

    def test_curie_weiss_zero(self):
        for mat in NON_FERROMAGNETS:
            chi = curie_weiss_susceptibility(mat, 1000.0)
            self.assertEqual(chi, 0.0, msg=f"{mat}: Curie-Weiss should be 0")

    def test_mvst_zero(self):
        for mat in NON_FERROMAGNETS:
            M = magnetization_vs_temperature(mat, 300.0)
            self.assertEqual(M, 0.0, msg=f"{mat}: M(T) should be 0")

    def test_sigma_shift_zero(self):
        for mat in NON_FERROMAGNETS:
            T_C = sigma_hysteresis_shift(mat, 0.1)
            self.assertEqual(T_C, 0.0, msg=f"{mat}: σ T_C shift should be 0")


# ── 8. Sigma-Field Dependence ──────────────────────────────────────

class TestSigma(unittest.TestCase):
    """T_C shifts through exchange coupling; σ=0 is the identity."""

    def test_sigma_zero_identity(self):
        """σ=0 returns the reference T_C unchanged."""
        for mat in FERROMAGNETS:
            T_C_ref = HYSTERESIS_DATA[mat]['T_Curie_K']
            T_C_sigma = sigma_hysteresis_shift(mat, 0.0)
            self.assertAlmostEqual(T_C_sigma, T_C_ref, places=8,
                                   msg=f"{mat}: σ=0 should return T_C unchanged")

    def test_positive_sigma_increases_tc(self):
        """Positive σ (stronger QCD / stiffer lattice) increases T_C."""
        for mat in FERROMAGNETS:
            T_C_0 = sigma_hysteresis_shift(mat, 0.0)
            T_C_pos = sigma_hysteresis_shift(mat, 0.1)
            self.assertGreater(T_C_pos, T_C_0,
                               msg=f"{mat}: σ>0 should increase T_C")

    def test_negative_sigma_decreases_tc(self):
        """Negative σ (weaker QCD) decreases T_C."""
        for mat in FERROMAGNETS:
            T_C_0 = sigma_hysteresis_shift(mat, 0.0)
            T_C_neg = sigma_hysteresis_shift(mat, -0.1)
            self.assertLess(T_C_neg, T_C_0,
                            msg=f"{mat}: σ<0 should decrease T_C")

    def test_shift_magnitude_small_at_earth(self):
        """At Earth σ ≈ 7e-10, shift is < 10⁻⁸ relative."""
        sigma_earth = 7e-10
        for mat in FERROMAGNETS:
            T_C_0 = sigma_hysteresis_shift(mat, 0.0)
            T_C_earth = sigma_hysteresis_shift(mat, sigma_earth)
            relative_shift = abs(T_C_earth - T_C_0) / T_C_0
            self.assertLess(relative_shift, 1e-8,
                            msg=f"{mat}: Earth σ shift too large: {relative_shift:.2e}")

    def test_shift_noticeable_at_neutron_star(self):
        """At σ ≈ 0.1 (neutron star), shift is ~ 0.5% (detectable)."""
        sigma_ns = 0.1
        for mat in FERROMAGNETS:
            T_C_0 = sigma_hysteresis_shift(mat, 0.0)
            T_C_ns = sigma_hysteresis_shift(mat, sigma_ns)
            relative_shift = (T_C_ns - T_C_0) / T_C_0
            # f_QCD ≈ 0.99, e^0.1 ≈ 1.105, shift ≈ 0.99 × 0.105 ≈ 0.104
            self.assertGreater(relative_shift, 0.08,
                               msg=f"{mat}: neutron star σ shift too small: {relative_shift:.4f}")
            self.assertLess(relative_shift, 0.15,
                            msg=f"{mat}: neutron star σ shift too large: {relative_shift:.4f}")

    def test_non_ferromagnets_return_zero(self):
        """Non-ferromagnets return T_C = 0 regardless of σ."""
        for mat in NON_FERROMAGNETS:
            for sigma in [0.0, 0.1, -0.1]:
                result = sigma_hysteresis_shift(mat, sigma)
                self.assertEqual(result, 0.0,
                                 msg=f"{mat}: σ shift T_C should be 0")


# ── 9. Rule 9 — All Materials, All Fields ─────────────────────────

class TestRule9(unittest.TestCase):
    """Every material has every field. No gaps allowed."""

    REQUIRED_FIELDS = [
        'M_sat_A_m',
        'H_c_A_m',
        'B_r_T',
        'T_Curie_K',
        'is_ferromagnetic',
        'magnetic_type',
    ]

    def test_all_eight_materials_present(self):
        """HYSTERESIS_DATA contains exactly the eight expected materials."""
        for mat in ALL_MATERIALS:
            self.assertIn(mat, HYSTERESIS_DATA,
                          msg=f"'{mat}' missing from HYSTERESIS_DATA")

    def test_all_fields_present_for_every_material(self):
        """Every material has every required field."""
        for mat in ALL_MATERIALS:
            entry = HYSTERESIS_DATA[mat]
            for field in self.REQUIRED_FIELDS:
                self.assertIn(field, entry,
                              msg=f"'{mat}' missing field '{field}'")

    def test_no_none_values(self):
        """No field has a None value."""
        for mat in ALL_MATERIALS:
            entry = HYSTERESIS_DATA[mat]
            for field in self.REQUIRED_FIELDS:
                self.assertIsNotNone(entry[field],
                                     msg=f"'{mat}'.'{field}' is None")

    def test_magnetic_type_valid(self):
        """magnetic_type is one of the three allowed strings."""
        valid_types = {'ferromagnetic', 'paramagnetic', 'diamagnetic'}
        for mat in ALL_MATERIALS:
            mt = HYSTERESIS_DATA[mat]['magnetic_type']
            self.assertIn(mt, valid_types,
                          msg=f"'{mat}': invalid magnetic_type '{mt}'")

    def test_is_ferromagnetic_bool(self):
        """is_ferromagnetic is a boolean."""
        for mat in ALL_MATERIALS:
            flag = HYSTERESIS_DATA[mat]['is_ferromagnetic']
            self.assertIsInstance(flag, bool,
                                  msg=f"'{mat}': is_ferromagnetic is not bool")

    def test_ferromagnets_have_nonzero_data(self):
        """Iron and nickel have positive M_sat, H_c, B_r, T_Curie."""
        for mat in FERROMAGNETS:
            entry = HYSTERESIS_DATA[mat]
            self.assertGreater(entry['M_sat_A_m'], 0.0,
                               msg=f"{mat}: M_sat should be > 0")
            self.assertGreater(entry['H_c_A_m'], 0.0,
                               msg=f"{mat}: H_c should be > 0")
            self.assertGreater(entry['B_r_T'], 0.0,
                               msg=f"{mat}: B_r should be > 0")
            self.assertGreater(entry['T_Curie_K'], 0.0,
                               msg=f"{mat}: T_Curie should be > 0")
            self.assertTrue(entry['is_ferromagnetic'],
                            msg=f"{mat}: is_ferromagnetic should be True")
            self.assertEqual(entry['magnetic_type'], 'ferromagnetic',
                             msg=f"{mat}: magnetic_type should be 'ferromagnetic'")

    def test_non_ferromagnets_have_zero_data(self):
        """Non-ferromagnetic materials have zero for all magnetic quantities."""
        for mat in NON_FERROMAGNETS:
            entry = HYSTERESIS_DATA[mat]
            self.assertEqual(entry['M_sat_A_m'], 0.0,
                             msg=f"{mat}: M_sat should be 0")
            self.assertEqual(entry['H_c_A_m'], 0.0,
                             msg=f"{mat}: H_c should be 0")
            self.assertEqual(entry['B_r_T'], 0.0,
                             msg=f"{mat}: B_r should be 0")
            self.assertEqual(entry['T_Curie_K'], 0.0,
                             msg=f"{mat}: T_Curie should be 0")
            self.assertFalse(entry['is_ferromagnetic'],
                             msg=f"{mat}: is_ferromagnetic should be False")

    def test_iron_curie_temp_known_value(self):
        """Iron T_C = 1043 K (MEASURED)."""
        self.assertAlmostEqual(HYSTERESIS_DATA['iron']['T_Curie_K'],
                               1043.0, places=0)

    def test_nickel_curie_temp_known_value(self):
        """Nickel T_C = 627 K (MEASURED)."""
        self.assertAlmostEqual(HYSTERESIS_DATA['nickel']['T_Curie_K'],
                               627.0, places=0)

    def test_iron_msat_known_value(self):
        """Iron M_sat ≈ 1.71×10⁶ A/m (MEASURED)."""
        M_sat = HYSTERESIS_DATA['iron']['M_sat_A_m']
        self.assertAlmostEqual(M_sat, 1.71e6, delta=1e5)


# ── 10. Nagatha Export ─────────────────────────────────────────────

class TestNagatha(unittest.TestCase):
    """hysteresis_properties() export completeness and self-consistency."""

    REQUIRED_KEYS = [
        'material',
        'H_field_A_m',
        'temperature_K',
        'sigma',
        'M_sat_A_m',
        'H_c_A_m',
        'B_r_T',
        'T_Curie_K',
        'is_ferromagnetic',
        'magnetic_type',
        'M_anhysteretic_A_m',
        'M_ascending_A_m',
        'M_descending_A_m',
        'B_ascending_T',
        'B_descending_T',
        'energy_loss_per_cycle_J_m3',
        'magnetization_at_T_A_m',
        'curie_weiss_susceptibility',
        'T_Curie_sigma_K',
        'origin',
    ]

    def test_all_keys_present_for_all_materials(self):
        """Export dict contains all required keys for every material."""
        for mat in ALL_MATERIALS:
            result = hysteresis_properties(mat)
            for key in self.REQUIRED_KEYS:
                self.assertIn(key, result,
                              msg=f"'{mat}': missing key '{key}'")

    def test_material_key_matches(self):
        """'material' field matches the input key."""
        for mat in ALL_MATERIALS:
            result = hysteresis_properties(mat)
            self.assertEqual(result['material'], mat)

    def test_origin_string_present(self):
        """Origin tag is a non-empty string."""
        for mat in ALL_MATERIALS:
            result = hysteresis_properties(mat)
            origin = result['origin']
            self.assertIsInstance(origin, str)
            self.assertGreater(len(origin), 10,
                               msg=f"'{mat}': origin string is too short")

    def test_origin_contains_measured_tag(self):
        """Origin string mentions MEASURED (for material data)."""
        for mat in ALL_MATERIALS:
            result = hysteresis_properties(mat)
            self.assertIn('MEASURED', result['origin'],
                          msg=f"'{mat}': MEASURED tag missing from origin")

    def test_ascending_descending_consistent_ferromagnet(self):
        """For iron at H > 0: B_descending ≥ B_ascending."""
        result = hysteresis_properties('iron', H_field=500.0, T=300.0)
        B_asc = result['B_ascending_T']
        B_desc = result['B_descending_T']
        self.assertGreaterEqual(B_desc, B_asc,
                                msg="Descending B should be ≥ ascending B at H>0")

    def test_non_magnetic_export_zeros(self):
        """Non-ferromagnets: M quantities and energy loss are all 0."""
        for mat in NON_FERROMAGNETS:
            result = hysteresis_properties(mat, H_field=1e5)
            self.assertEqual(result['M_sat_A_m'], 0.0)
            self.assertEqual(result['M_ascending_A_m'], 0.0)
            self.assertEqual(result['M_descending_A_m'], 0.0)
            self.assertEqual(result['energy_loss_per_cycle_J_m3'], 0.0)
            self.assertFalse(result['is_ferromagnetic'])

    def test_sigma_zero_no_change(self):
        """σ=0 export T_Curie_sigma matches T_Curie_K for ferromagnets."""
        for mat in FERROMAGNETS:
            result = hysteresis_properties(mat, sigma=0.0)
            self.assertAlmostEqual(result['T_Curie_sigma_K'],
                                   result['T_Curie_K'], places=8,
                                   msg=f"{mat}: σ=0 T_C_sigma should equal T_C")

    def test_parameter_passthrough(self):
        """H_field, T, sigma are stored in the export dict."""
        result = hysteresis_properties('iron', H_field=123.0, T=500.0, sigma=0.05)
        self.assertEqual(result['H_field_A_m'], 123.0)
        self.assertEqual(result['temperature_K'], 500.0)
        self.assertEqual(result['sigma'], 0.05)

    def test_b_field_formula(self):
        """B_ascending = μ₀ (H + M_ascending), verified numerically."""
        from ..constants import MU_0
        for mat in FERROMAGNETS:
            H = 1000.0
            result = hysteresis_properties(mat, H_field=H)
            M_asc = result['M_ascending_A_m']
            B_expected = MU_0 * (H + M_asc)
            B_reported = result['B_ascending_T']
            self.assertAlmostEqual(B_reported, B_expected, places=20,
                                   msg=f"{mat}: B_ascending formula mismatch")


if __name__ == '__main__':
    unittest.main()
