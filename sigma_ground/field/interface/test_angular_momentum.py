"""
Tests for angular_momentum.py — quantum numbers, CG coefficients, term symbols.

Strategy:
  - Test angular momentum magnitude |J| = ℏ√(j(j+1))
  - Test m_j values range from −j to +j
  - Test angular momentum addition (triangle rule)
  - Test state counting conservation
  - Test Clebsch-Gordan: known exact values
  - Test Hund's rules: known ground state terms
  - Test spin-orbit: Landé interval rule
  - Test Landé g-factor special cases
  - Test Pauli matrices properties
  - Test Rule 9: full_report

Reference values (MEASURED / exact):
  CG: ⟨½,½;½,½|1,1⟩ = 1
  CG: ⟨½,½;½,−½|0,0⟩ = 1/√2
  CG: ⟨½,½;½,−½|1,0⟩ = 1/√2
  d² ground state: ³F₂ (Ti²⁺, V³⁺)
  d⁵ ground state: ⁶S₅/₂ (Fe³⁺, Mn²⁺)
  d⁸ ground state: ³F₄ (Ni²⁺)
  p² ground state: ³P₀ (C, Si ground state)
"""

import math
import unittest

from sigma_ground.field.interface.angular_momentum import (
    angular_momentum_magnitude,
    angular_momentum_z_values,
    multiplicity,
    allowed_J_values,
    total_states,
    verify_state_count,
    clebsch_gordan,
    term_symbol,
    all_term_symbols,
    hund_ground_state,
    spin_orbit_energy_eV,
    spin_orbit_splitting_eV,
    lande_interval_check,
    hydrogen_spin_orbit_constant_eV,
    lande_g_factor,
    magnetic_moment_bohr_magnetons,
    pauli_matrices,
    spin_expectation,
    angular_momentum_report,
    full_report,
)
from sigma_ground.field.constants import HBAR


class TestAngularMomentumBasics(unittest.TestCase):
    """Basic angular momentum properties."""

    def test_magnitude_spin_half(self):
        """|J| = ℏ√(3/4) for j=½."""
        mag = angular_momentum_magnitude(0.5)
        self.assertAlmostEqual(mag, HBAR * math.sqrt(0.75), delta=1e-40)

    def test_magnitude_j1(self):
        """|J| = ℏ√2 for j=1."""
        mag = angular_momentum_magnitude(1)
        self.assertAlmostEqual(mag, HBAR * math.sqrt(2), delta=1e-40)

    def test_z_values_spin_half(self):
        """j=½ → m = −½, +½."""
        vals = angular_momentum_z_values(0.5)
        self.assertEqual(len(vals), 2)
        self.assertAlmostEqual(vals[0], -0.5)
        self.assertAlmostEqual(vals[1], 0.5)

    def test_z_values_j2(self):
        """j=2 → m = −2, −1, 0, 1, 2."""
        vals = angular_momentum_z_values(2)
        self.assertEqual(len(vals), 5)

    def test_multiplicity(self):
        """2j+1 substates."""
        self.assertEqual(multiplicity(0), 1)
        self.assertEqual(multiplicity(0.5), 2)
        self.assertEqual(multiplicity(1), 3)
        self.assertEqual(multiplicity(2), 5)


class TestAngularMomentumAddition(unittest.TestCase):
    """Adding two angular momenta."""

    def test_half_half(self):
        """½ + ½ → J = 0 or 1."""
        Js = allowed_J_values(0.5, 0.5)
        self.assertAlmostEqual(Js[0], 0)
        self.assertAlmostEqual(Js[1], 1)

    def test_1_half(self):
        """1 + ½ → J = ½ or 3/2."""
        Js = allowed_J_values(1, 0.5)
        self.assertAlmostEqual(Js[0], 0.5)
        self.assertAlmostEqual(Js[1], 1.5)

    def test_1_1(self):
        """1 + 1 → J = 0, 1, 2."""
        Js = allowed_J_values(1, 1)
        self.assertEqual(len(Js), 3)
        self.assertAlmostEqual(Js[0], 0)
        self.assertAlmostEqual(Js[2], 2)

    def test_state_count_conservation(self):
        """Total state count is conserved."""
        self.assertTrue(verify_state_count(0.5, 0.5))
        self.assertTrue(verify_state_count(1, 0.5))
        self.assertTrue(verify_state_count(1, 1))
        self.assertTrue(verify_state_count(2, 1))
        self.assertTrue(verify_state_count(1.5, 0.5))

    def test_total_states(self):
        """(2j₁+1)(2j₂+1) product."""
        self.assertEqual(total_states(0.5, 0.5), 4)
        self.assertEqual(total_states(1, 1), 9)
        self.assertEqual(total_states(2, 1), 15)


class TestClebschGordan(unittest.TestCase):
    """Clebsch-Gordan coefficients."""

    def test_stretched_state(self):
        """⟨½,½;½,½|1,1⟩ = 1 (both aligned → maximum J, M)."""
        cg = clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1, 1)
        self.assertAlmostEqual(cg, 1.0, delta=0.01)

    def test_singlet(self):
        """⟨½,½;½,−½|0,0⟩ = 1/√2."""
        cg = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0, 0)
        self.assertAlmostEqual(abs(cg), 1.0 / math.sqrt(2), delta=0.01)

    def test_triplet_m0(self):
        """⟨½,½;½,−½|1,0⟩ = 1/√2."""
        cg = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1, 0)
        self.assertAlmostEqual(abs(cg), 1.0 / math.sqrt(2), delta=0.01)

    def test_selection_M(self):
        """M ≠ m₁+m₂ → CG = 0."""
        cg = clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1, 0)
        self.assertAlmostEqual(cg, 0.0, delta=1e-10)

    def test_triangle_violation(self):
        """J outside triangle → CG = 0."""
        cg = clebsch_gordan(0.5, 0.5, 0.5, 0.5, 2, 1)
        self.assertAlmostEqual(cg, 0.0, delta=1e-10)

    def test_orthogonality(self):
        """Σ_{m1,m2} |CG|² = 1 for each (J, M)."""
        j1, j2, J, M = 1, 0.5, 1.5, 0.5
        total = 0.0
        for m1 in [-1, 0, 1]:
            m2 = M - m1
            if abs(m2) <= j2:
                cg = clebsch_gordan(j1, m1, j2, m2, J, M)
                total += cg**2
        self.assertAlmostEqual(total, 1.0, delta=0.05)

    def test_1_1_coupling(self):
        """⟨1,1;1,−1|0,0⟩ = 1/√3."""
        cg = clebsch_gordan(1, 1, 1, -1, 0, 0)
        self.assertAlmostEqual(abs(cg), 1.0 / math.sqrt(3), delta=0.02)


class TestTermSymbols(unittest.TestCase):
    """Spectroscopic term symbols."""

    def test_hydrogen_ground(self):
        """Hydrogen ground state: ²S₁/₂."""
        ts = term_symbol(0, 0.5, 0.5)
        self.assertEqual(ts, '2S1/2')

    def test_triplet_P(self):
        """³P₂ format."""
        ts = term_symbol(1, 1, 2)
        self.assertEqual(ts, '3P2')

    def test_all_terms(self):
        """L=2, S=1 gives ³D₁, ³D₂, ³D₃."""
        terms = all_term_symbols(2, 1)
        self.assertEqual(len(terms), 3)
        J_vals = [J for J, ts in terms]
        self.assertAlmostEqual(min(J_vals), 1)
        self.assertAlmostEqual(max(J_vals), 3)


class TestHundRules(unittest.TestCase):
    """Hund's rules for ground state terms."""

    def test_d1(self):
        """d¹ (Ti³⁺): ²D₃/₂."""
        S, L, J, term = hund_ground_state(1, 2)
        self.assertAlmostEqual(S, 0.5)
        self.assertEqual(L, 2)
        self.assertAlmostEqual(J, 1.5)

    def test_d2(self):
        """d² (Ti²⁺, V³⁺): ³F₂."""
        S, L, J, term = hund_ground_state(2, 2)
        self.assertAlmostEqual(S, 1.0)
        self.assertEqual(L, 3)
        self.assertAlmostEqual(J, 2.0)
        self.assertIn('3F', term)

    def test_d5(self):
        """d⁵ (Fe³⁺, Mn²⁺): ⁶S₅/₂ — half-filled, L=0."""
        S, L, J, term = hund_ground_state(5, 2)
        self.assertAlmostEqual(S, 2.5)
        self.assertEqual(L, 0)
        self.assertAlmostEqual(J, 2.5)
        self.assertIn('6S', term)

    def test_d8(self):
        """d⁸ (Ni²⁺): ³F₄ — more than half, J = L+S."""
        S, L, J, term = hund_ground_state(8, 2)
        self.assertAlmostEqual(S, 1.0)
        self.assertEqual(L, 3)
        self.assertAlmostEqual(J, 4.0)
        self.assertIn('3F', term)

    def test_d10(self):
        """d¹⁰: ¹S₀ — closed shell."""
        S, L, J, term = hund_ground_state(10, 2)
        self.assertAlmostEqual(S, 0.0)
        self.assertEqual(L, 0)
        self.assertAlmostEqual(J, 0.0)
        self.assertIn('1S', term)

    def test_p2(self):
        """p² (C, Si ground state): ³P₀."""
        S, L, J, term = hund_ground_state(2, 1)
        self.assertAlmostEqual(S, 1.0)
        self.assertEqual(L, 1)
        self.assertAlmostEqual(J, 0.0)
        self.assertIn('3P', term)

    def test_p3(self):
        """p³ (N ground state): ⁴S₃/₂."""
        S, L, J, term = hund_ground_state(3, 1)
        self.assertAlmostEqual(S, 1.5)
        self.assertEqual(L, 0)
        self.assertAlmostEqual(J, 1.5)
        self.assertIn('4S', term)

    def test_f7(self):
        """f⁷ (Gd³⁺, Eu²⁺): ⁸S₇/₂ — half-filled f-shell."""
        S, L, J, term = hund_ground_state(7, 3)
        self.assertAlmostEqual(S, 3.5)
        self.assertEqual(L, 0)
        self.assertAlmostEqual(J, 3.5)
        self.assertIn('8S', term)

    def test_invalid_electron_count(self):
        """Too many electrons → error."""
        with self.assertRaises(ValueError):
            hund_ground_state(11, 2)  # max d electrons = 10


class TestSpinOrbit(unittest.TestCase):
    """Spin-orbit coupling."""

    def test_splitting_positive(self):
        """Spin-orbit splitting is positive for A > 0."""
        dE = spin_orbit_splitting_eV(0.01, 2, 1)  # ³D term
        self.assertGreater(dE, 0)

    def test_lande_interval_rule(self):
        """ΔE(J, J-1) ∝ J — Landé interval rule."""
        A = 0.01
        L, S = 2, 1
        intervals = lande_interval_check(A, L, S)
        for J, dE, ratio in intervals:
            self.assertAlmostEqual(ratio, J, delta=0.01)

    def test_s_orbital_no_SO(self):
        """L=0 → no spin-orbit splitting."""
        dE = spin_orbit_splitting_eV(0.01, 0, 2.5)  # ⁶S
        self.assertEqual(dE, 0.0)

    def test_SO_energy_formula(self):
        """E_SO = (A/2)[J(J+1) − L(L+1) − S(S+1)]."""
        A = 0.05
        L, S, J = 1, 1, 2
        E = spin_orbit_energy_eV(A, L, S, J)
        expected = 0.5 * A * (J*(J+1) - L*(L+1) - S*(S+1))
        self.assertAlmostEqual(E, expected, delta=1e-10)


class TestSpinOrbitConstant(unittest.TestCase):
    """Spin-orbit coupling constant estimates."""

    def test_hydrogen_2p(self):
        """Hydrogen 2p SO constant ≈ 1.5×10⁻⁵ eV."""
        A = hydrogen_spin_orbit_constant_eV(1, 2, 1)
        self.assertGreater(A, 5e-6)
        self.assertLess(A, 5e-5)

    def test_s_orbital_zero(self):
        """l=0 → A = 0 (no orbital angular momentum)."""
        A = hydrogen_spin_orbit_constant_eV(1, 1, 0)
        self.assertEqual(A, 0.0)

    def test_Z_scaling(self):
        """A ∝ Z⁴ for hydrogen-like."""
        A1 = hydrogen_spin_orbit_constant_eV(1, 2, 1)
        A2 = hydrogen_spin_orbit_constant_eV(2, 2, 1)
        self.assertAlmostEqual(A2 / A1, 16.0, delta=1.0)


class TestLandeGFactor(unittest.TestCase):
    """Landé g-factor."""

    def test_pure_spin(self):
        """L=0, S=½, J=½ → g=2 (free electron)."""
        g = lande_g_factor(0, 0.5, 0.5)
        self.assertAlmostEqual(g, 2.0, delta=0.01)

    def test_pure_orbital(self):
        """S=0, L=1, J=1 → g=1."""
        g = lande_g_factor(1, 0, 1)
        self.assertAlmostEqual(g, 1.0, delta=0.01)

    def test_J_zero(self):
        """J=0 → g=0."""
        g = lande_g_factor(1, 1, 0)
        self.assertEqual(g, 0.0)

    def test_3P2(self):
        """³P₂ (L=1, S=1, J=2): g = 3/2."""
        g = lande_g_factor(1, 1, 2)
        self.assertAlmostEqual(g, 1.5, delta=0.01)


class TestMagneticMoment(unittest.TestCase):
    """Magnetic moment in Bohr magnetons."""

    def test_free_electron(self):
        """Free electron: μ_eff = 2√(3/4) ≈ 1.73 μ_B."""
        mu = magnetic_moment_bohr_magnetons(0, 0.5, 0.5)
        self.assertAlmostEqual(mu, 1.73, delta=0.02)

    def test_d5_high_spin(self):
        """d⁵ ⁶S₅/₂: μ_eff = 2√(35/4) ≈ 5.92 μ_B (MEASURED for Fe³⁺)."""
        mu = magnetic_moment_bohr_magnetons(0, 2.5, 2.5)
        self.assertAlmostEqual(mu, 5.92, delta=0.1)


class TestPauliMatrices(unittest.TestCase):
    """Pauli spin matrices."""

    def test_sigma_z_eigenvalues(self):
        """σ_z diagonal: +1, −1."""
        p = pauli_matrices()
        self.assertEqual(p['z'][0][0], 1)
        self.assertEqual(p['z'][1][1], -1)

    def test_sigma_x_off_diagonal(self):
        """σ_x off-diagonal: 1, 1."""
        p = pauli_matrices()
        self.assertEqual(p['x'][0][1], 1)
        self.assertEqual(p['x'][1][0], 1)


class TestSpinExpectation(unittest.TestCase):
    """Spin-½ expectation values."""

    def test_spin_up(self):
        """|↑⟩ → ⟨S_z⟩ = +½."""
        sz = spin_expectation(1, 0, 'z')
        self.assertAlmostEqual(sz, 0.5, delta=0.01)

    def test_spin_down(self):
        """|↓⟩ → ⟨S_z⟩ = −½."""
        sz = spin_expectation(0, 1, 'z')
        self.assertAlmostEqual(sz, -0.5, delta=0.01)

    def test_superposition(self):
        """(|↑⟩+|↓⟩)/√2 → ⟨S_z⟩ = 0, ⟨S_x⟩ = ½."""
        sz = spin_expectation(1, 1, 'z')
        sx = spin_expectation(1, 1, 'x')
        self.assertAlmostEqual(sz, 0.0, delta=0.01)
        self.assertAlmostEqual(sx, 0.5, delta=0.01)


class TestReports(unittest.TestCase):
    """Rule 9: full_report."""

    def test_report_complete(self):
        """angular_momentum_report has required fields."""
        r = angular_momentum_report()
        required = ['n_electrons', 'l', 'ground_state_term',
                     'ground_state_S', 'ground_state_L', 'ground_state_J',
                     'lande_g_factor', 'magnetic_moment_muB']
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_full_report(self):
        """full_report returns dict with extra fields."""
        r = full_report()
        self.assertIsInstance(r, dict)
        self.assertIn('cg_example_1/2_1/2_to_1', r)
        self.assertIn('state_count_verified', r)
        self.assertTrue(r['state_count_verified'])

    def test_full_report_f_shell(self):
        """full_report works for f-shell."""
        r = full_report(n_electrons=7, l=3)
        self.assertIn('8S', r['ground_state_term'])


if __name__ == '__main__':
    unittest.main()
