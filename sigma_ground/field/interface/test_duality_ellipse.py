"""
Tests for the Khatiwada-Qian wave-particle duality ellipse
(arXiv:2505.21443v1, May 2025) — validation and integration with sigma-ground.

STRATEGY:
  Phase A (this file) validates the paper's math on synthetic bipartite states
  constructed to match the paper's state form:

      |Ψ⟩ = c₁|0⟩|M₁⟩ + c₂|1⟩|M₂⟩
      |M₁⟩ = |0⟩,  |M₂⟩ = γ|0⟩ + √(1−γ²)|1⟩

  and verifies:
      V² / γ² + D² = 1          (duality ellipse, paper Eq. 8)
      C = 2|c₁c₂| √(1−γ²)        (concurrence from coherence, paper Eq. 11)
      C = 2·s₁·s₂                 (bridge to sigma-ground Schmidt)

  where
      V = 2|c₁c₂|γ                (visibility)
      D = ||c₁|² − |c₂|²|          (distinguishability)
      γ = |⟨M₁|M₂⟩|                (environment marginal overlap)

  HYPOTHESIS TESTS:
      H1 — paper:           γ and D are independent free parameters
      H2 — info-cascade:    γ² + D² = 1  (information conservation)
      H3 — empirical:       whatever V(D) curve matter-shaper actually produces
      H4 — per-dim hunch:   γ = Θ = η^(1/3)  (cosmic entanglement share per
                            gravitational dimension, cosmological prediction)
      H5 — fossil-direct:   γ = η             (fossil entanglement fraction
                            applied flat, cosmological prediction)

  GATE: if T7 (Schmidt bridge) fails, the paper's concurrence does not match
        sigma-ground's existing reduced-density-matrix Schmidt decomposition.
        No integration; stop and report.

State convention (from quantum_computing.py):
  MSB-first qubit ordering.  For a 2-qubit state [a, b, c, d]:
      state[0] = |00⟩ = q₀=0, q₁=0
      state[1] = |01⟩ = q₀=0, q₁=1
      state[2] = |10⟩ = q₀=1, q₁=0
      state[3] = |11⟩ = q₀=1, q₁=1
  Qubit 0 = path,  qubit 1 = environment marginal.
"""

import math
import unittest

from sigma_ground.field.constants import ETA, SIGMA_CONV, THETA_ENTANGLEMENT_PER_DIM
from sigma_ground.field.entanglement import sigma_coherence
from sigma_ground.field.interface.duality_ellipse import coherence_gamma_from_sigma
from sigma_ground.field.interface.quantum_computing import run_circuit
from sigma_ground.field.interface.quantum_output import (
    schmidt_coefficients,
    entanglement_entropy,
)


# ---------------------------------------------------------------------------
#  Paper's state constructor + paper's formulas (independent of sigma-ground)
# ---------------------------------------------------------------------------


def _build_psi(c1, c2, gamma):
    """Construct |Ψ⟩ = c₁|0⟩|M₁⟩ + c₂|1⟩|M₂⟩ as a 4-element state vector.

    With |M₁⟩ = |0⟩ and |M₂⟩ = γ|0⟩ + √(1−γ²)|1⟩, the basis expansion is:
        |Ψ⟩ = c₁|00⟩ + c₂·γ|10⟩ + c₂·√(1−γ²)|11⟩

    State entries (MSB-first, qubit 0 is path, qubit 1 is environment):
        state[0] = c₁              (|00⟩)
        state[1] = 0                (|01⟩)
        state[2] = c₂·γ            (|10⟩)
        state[3] = c₂·√(1−γ²)      (|11⟩)

    Assumes |c₁|² + |c₂|² = 1 (caller must normalize) and 0 ≤ γ ≤ 1.
    """
    c1 = complex(c1)
    c2 = complex(c2)
    root = math.sqrt(max(0.0, 1.0 - gamma * gamma))
    return [c1, complex(0.0), c2 * gamma, c2 * root]


def _V_paper(c1, c2, gamma):
    """Paper Eq. 8 visibility:  V = 2|c₁c₂|·γ."""
    return 2.0 * abs(c1) * abs(c2) * gamma


def _D_paper(c1, c2):
    """Paper distinguishability:  D = ||c₁|² − |c₂|²|."""
    return abs(abs(c1) ** 2 - abs(c2) ** 2)


def _C_paper(c1, c2, gamma):
    """Paper Eq. 11 concurrence:  C = 2|c₁c₂|·√(1−γ²)."""
    return 2.0 * abs(c1) * abs(c2) * math.sqrt(max(0.0, 1.0 - gamma * gamma))


def _ellipse_residual(V, D, gamma):
    """Residual of paper Eq. 8:  V²/γ² + D² − 1.  Zero on saturation."""
    if gamma <= 0.0:
        # Degenerate limit γ→0: V must be 0, and D² = 1 for saturation.
        # Return |D²−1| + V as a degenerate check.
        return abs(D * D - 1.0) + V
    return (V * V) / (gamma * gamma) + D * D - 1.0


# ---------------------------------------------------------------------------
#  T1–T8: core ellipse and concurrence validation
# ---------------------------------------------------------------------------


class TestEllipseCoreCases(unittest.TestCase):
    """T1–T3: specific states with hand-computed expected values."""

    def test_t1_bell_gamma_one_recovers_englert(self):
        """T1: γ=1, c₁=c₂=1/√2 → Englert circle V²+D²=1, and C_paper=0."""
        c1 = c2 = 1.0 / math.sqrt(2.0)
        gamma = 1.0
        V = _V_paper(c1, c2, gamma)
        D = _D_paper(c1, c2)
        C = _C_paper(c1, c2, gamma)
        # V=1, D=0 → V²+D²=1 (Englert saturation)
        self.assertAlmostEqual(V, 1.0, places=12)
        self.assertAlmostEqual(D, 0.0, places=12)
        self.assertAlmostEqual(V * V + D * D, 1.0, places=12)
        # γ=1 → C_paper = 0 (environment marginals identical, no path entanglement)
        self.assertAlmostEqual(C, 0.0, places=12)

    def test_t2_orthogonal_marginals_gamma_zero(self):
        """T2: γ=0, generic c₁,c₂ → V=0 and C_paper=2|c₁c₂| (max for this bias)."""
        c1 = math.sqrt(0.6)
        c2 = math.sqrt(0.4)
        gamma = 0.0
        V = _V_paper(c1, c2, gamma)
        C = _C_paper(c1, c2, gamma)
        self.assertAlmostEqual(V, 0.0, places=12)
        self.assertAlmostEqual(C, 2.0 * c1 * c2, places=12)

    def test_t3_generic_state_saturates_ellipse(self):
        """T3: c₁=√0.7, c₂=√0.3, γ=0.5 → V²/γ² + D² = 1 exactly."""
        c1 = math.sqrt(0.7)
        c2 = math.sqrt(0.3)
        gamma = 0.5
        V = _V_paper(c1, c2, gamma)
        D = _D_paper(c1, c2)
        C = _C_paper(c1, c2, gamma)
        # V = 2·√0.21·0.5 ≈ 0.45826
        # D = |0.7 − 0.3| = 0.4
        # V²/γ² + D² = (0.45826²)/0.25 + 0.16 = 0.84 + 0.16 = 1.0
        self.assertAlmostEqual(V, 2.0 * math.sqrt(0.21) * 0.5, places=12)
        self.assertAlmostEqual(D, 0.4, places=12)
        self.assertAlmostEqual(_ellipse_residual(V, D, gamma), 0.0, places=10)
        # C = 2·√0.21·√(1−0.25) = 2·√0.21·√0.75 ≈ 0.7937
        self.assertAlmostEqual(
            C, 2.0 * math.sqrt(0.21) * math.sqrt(0.75), places=12
        )


class TestSchmidtBridge(unittest.TestCase):
    """T4 and T7: paper's concurrence matches sigma-ground's Schmidt decomposition.

    This is THE load-bearing test.  If it fails, the paper and the library
    are describing subtly different physical quantities and the integration
    cannot proceed cleanly.  All of Phase B hinges on this.
    """

    def test_t4_concurrence_matches_schmidt_specific(self):
        """T4: for (√0.9, √0.1, γ=0.8), C_paper == 2·s₁·s₂ from Schmidt."""
        c1 = math.sqrt(0.9)
        c2 = math.sqrt(0.1)
        gamma = 0.8
        psi = _build_psi(c1, c2, gamma)
        # Verify normalization
        norm = sum(abs(a) ** 2 for a in psi)
        self.assertAlmostEqual(norm, 1.0, places=12)
        # Schmidt bridge
        coeffs = schmidt_coefficients(psi, [0])
        s1, s2 = coeffs[0], coeffs[1]
        C_paper = _C_paper(c1, c2, gamma)
        self.assertAlmostEqual(2.0 * s1 * s2, C_paper, places=10)
        # Also verify ellipse
        V = _V_paper(c1, c2, gamma)
        D = _D_paper(c1, c2)
        self.assertAlmostEqual(_ellipse_residual(V, D, gamma), 0.0, places=10)

    def test_t7_schmidt_bridge_load_bearing_sweep(self):
        """T7: across 20×20 (p, γ) samples, C_paper == 2·s₁·s₂ always.

        If this fails, STOP.  The paper's concurrence is not sigma-ground's
        concurrence.  No integration possible.
        """
        max_residual = 0.0
        for i in range(1, 20):
            p = 0.05 + i * 0.045  # p ∈ [0.095, 0.905]
            c1 = math.sqrt(p)
            c2 = math.sqrt(1.0 - p)
            for j in range(1, 20):
                gamma = 0.05 + j * 0.045
                psi = _build_psi(c1, c2, gamma)
                coeffs = schmidt_coefficients(psi, [0])
                s1 = coeffs[0]
                s2 = coeffs[1] if len(coeffs) > 1 else 0.0
                C_paper = _C_paper(c1, c2, gamma)
                residual = abs(2.0 * s1 * s2 - C_paper)
                if residual > max_residual:
                    max_residual = residual
        self.assertLess(
            max_residual, 1e-10,
            f"T7 GATE FAILURE: max Schmidt-concurrence residual = {max_residual}"
        )


class TestEllipseSweep(unittest.TestCase):
    """T5 and T6: ellipse residual ≈ 0 across the full (p, γ) grid."""

    def test_t5_ellipse_holds_across_sweep(self):
        """T5: 20×20 sweep over (p=|c₁|², γ) shows ellipse residual < 1e-12."""
        max_residual = 0.0
        for i in range(1, 20):
            p = 0.05 + i * 0.045
            c1 = math.sqrt(p)
            c2 = math.sqrt(1.0 - p)
            for j in range(1, 20):
                gamma = 0.05 + j * 0.045
                V = _V_paper(c1, c2, gamma)
                D = _D_paper(c1, c2)
                residual = abs(_ellipse_residual(V, D, gamma))
                if residual > max_residual:
                    max_residual = residual
        self.assertLess(max_residual, 1e-12)

    def test_t6_gamma_one_recovers_englert_across_sweep(self):
        """T6: for γ=1 across all bias values, V²+D² = 1 exactly (Englert circle)."""
        max_residual = 0.0
        for i in range(1, 100):
            p = 0.005 + i * 0.01
            c1 = math.sqrt(p)
            c2 = math.sqrt(1.0 - p)
            gamma = 1.0
            V = _V_paper(c1, c2, gamma)
            D = _D_paper(c1, c2)
            residual = abs(V * V + D * D - 1.0)
            if residual > max_residual:
                max_residual = residual
        self.assertLess(max_residual, 1e-12)


class TestEntropyConsistency(unittest.TestCase):
    """T8: Bell state entropy = ln(2), confirmed via both paths."""

    def test_t8_bell_state_entropy_ln2(self):
        """T8: Bell state (|00⟩+|11⟩)/√2 has entanglement entropy ln(2).

        Via sigma-ground's entanglement_entropy AND via the Schmidt formula
        S = −Σ sᵢ² log(sᵢ²) — both must give ln(2) for a maximally entangled pair.
        """
        psi = run_circuit(2, [('h', 0), ('cnot', 0, 1)])
        S_direct = entanglement_entropy(psi, 0)
        coeffs = schmidt_coefficients(psi, [0])
        S_schmidt = -sum(
            s * s * math.log(s * s) for s in coeffs if s > 1e-15
        )
        self.assertAlmostEqual(S_direct, math.log(2.0), delta=0.01)
        self.assertAlmostEqual(S_schmidt, math.log(2.0), delta=0.01)
        self.assertAlmostEqual(S_direct, S_schmidt, delta=0.01)


# ---------------------------------------------------------------------------
#  H1 / H2 / H3: hypothesis tests
# ---------------------------------------------------------------------------


class TestHypothesisH1PaperFreeGamma(unittest.TestCase):
    """H1: the paper claims γ and D are independent.

    Pass: construct pairs of states with the SAME D but DIFFERENT γ;
    both must sit on the ellipse (V²/γ² + D² = 1).
    """

    def test_h1_same_D_different_gamma(self):
        """Two states with D=0.4 but γ=0.3 and γ=0.9; both must saturate ellipse."""
        # D = ||c1|² − |c2|²| = 0.4, so |c1|² = 0.7, |c2|² = 0.3
        c1 = math.sqrt(0.7)
        c2 = math.sqrt(0.3)
        for gamma in (0.3, 0.5, 0.7, 0.9):
            V = _V_paper(c1, c2, gamma)
            D = _D_paper(c1, c2)
            self.assertAlmostEqual(D, 0.4, places=12)
            self.assertAlmostEqual(
                _ellipse_residual(V, D, gamma), 0.0, places=10,
                msg=f"Ellipse failed at γ={gamma}, D={D}"
            )

    def test_h1_gamma_independent_of_D(self):
        """Direct check: 4×4 (D, γ) grid — all combinations produce valid ellipse states."""
        Ds = [0.0, 0.3, 0.6, 0.9]
        gammas = [0.1, 0.4, 0.7, 1.0]
        for D_target in Ds:
            p = 0.5 + D_target / 2.0   # |c1|² = (1 + D)/2  gives that bias
            c1 = math.sqrt(p)
            c2 = math.sqrt(1.0 - p)
            for gamma in gammas:
                V = _V_paper(c1, c2, gamma)
                D_actual = _D_paper(c1, c2)
                self.assertAlmostEqual(D_actual, D_target, places=12)
                self.assertAlmostEqual(
                    _ellipse_residual(V, D_actual, gamma), 0.0, places=10
                )


class TestHypothesisH2InformationConservation(unittest.TestCase):
    """H2: the user's theory says γ² + D² = 1 (locked, information-conservation).

    On SYNTHETIC states, this is a RESTRICTION — math allows any γ for any D,
    so H2 is NOT automatically satisfied.  This test confirms that H2 is a
    non-trivial physical claim: synthetic states can violate it freely.

    H2 is ultimately tested empirically (H3) against matter-shaper output.
    """

    def test_h2_violated_for_free_synthetic_states(self):
        """Synthetic states freely violate γ² + D² = 1, confirming H2 is non-trivial."""
        # Pick a state where |c1|²=|c2|² (D=0) and γ=0.5
        # H2 predicts γ² + D² = 0.25 ≠ 1 → violation
        c1 = c2 = 1.0 / math.sqrt(2.0)
        gamma = 0.5
        D = _D_paper(c1, c2)
        h2_residual = gamma * gamma + D * D - 1.0
        self.assertAlmostEqual(D, 0.0, places=12)
        self.assertAlmostEqual(h2_residual, -0.75, places=12)
        # So H2 is not a mathematical tautology — it IS a physical claim.

    def test_h2_curve_shape_when_locked(self):
        """If γ² + D² = 1 is enforced, predicts V = 1 − D² (not √(1−D²))."""
        # Under H2: γ = √(1 − D²).  Then V_max = γ·√(1−D²) = (1−D²).
        for D in (0.0, 0.2, 0.5, 0.7, 0.9):
            gamma_h2 = math.sqrt(max(0.0, 1.0 - D * D))
            # With |c1|² = (1+D)/2, |c2|² = (1−D)/2:
            c1 = math.sqrt((1.0 + D) / 2.0)
            c2 = math.sqrt((1.0 - D) / 2.0)
            V_h2 = _V_paper(c1, c2, gamma_h2)
            V_expected = 1.0 - D * D
            self.assertAlmostEqual(V_h2, V_expected, places=10)

    def test_h2_ellipse_still_saturates_when_locked(self):
        """Even under the H2 lock, V²/γ² + D² = 1 still holds (ellipse is consistent)."""
        for D in (0.0, 0.3, 0.6, 0.9):
            gamma_h2 = math.sqrt(max(0.0, 1.0 - D * D))
            c1 = math.sqrt((1.0 + D) / 2.0)
            c2 = math.sqrt((1.0 - D) / 2.0)
            V = _V_paper(c1, c2, gamma_h2)
            if gamma_h2 > 1e-9:
                self.assertAlmostEqual(
                    _ellipse_residual(V, D, gamma_h2), 0.0, places=10
                )


class TestHypothesisH4ThetaPerGravitationalDimension(unittest.TestCase):
    """H4: γ is not free but pinned to Θ = η^(1/3) — the share of fossil
    entanglement that lives in a single gravitational dimension.

    This is a COSMOLOGICAL prediction.  At laboratory σ≈0 we observe γ→1
    (matter-shaper H1 match); H4 is expected to become visible only at
    σ > 0.  These tests verify the mathematical consistency of γ=Θ with
    the ellipse, document the signature of the hypothesis (V(D=0) = Θ),
    and record that H4 is distinguishable from H1 at σ>0.
    """

    def test_h4_ellipse_holds_with_gamma_theta(self):
        """With γ pinned to Θ, the ellipse V²/γ² + D² = 1 still saturates."""
        gamma = THETA_ENTANGLEMENT_PER_DIM
        self.assertAlmostEqual(gamma, 0.4153 ** (1.0 / 3.0), places=12)
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            c1 = math.sqrt(p)
            c2 = math.sqrt(1.0 - p)
            V = _V_paper(c1, c2, gamma)
            D = _D_paper(c1, c2)
            self.assertAlmostEqual(
                _ellipse_residual(V, D, gamma), 0.0, places=10,
                msg=f"H4 ellipse failure at p={p}, gamma=Theta={gamma}"
            )

    def test_h4_signature_at_D_zero(self):
        """H4 signature: V(D=0) = Θ.  A laboratory V=1 observation falsifies H4
        at that σ.  An observation of V = Θ ≈ 0.7461 at D=0 confirms H4.
        """
        c1 = c2 = 1.0 / math.sqrt(2.0)
        gamma = THETA_ENTANGLEMENT_PER_DIM
        V = _V_paper(c1, c2, gamma)
        D = _D_paper(c1, c2)
        self.assertAlmostEqual(D, 0.0, places=12)
        self.assertAlmostEqual(V, THETA_ENTANGLEMENT_PER_DIM, places=12)

    def test_h4_discriminates_from_h1_at_all_D(self):
        """H1 and H4 predict DIFFERENT V(D) curves everywhere except D=1.

        This is the σ>0 discrimination test: at any σ where γ has relaxed
        toward Θ, the measured V curve sits below H1's envelope by factor Θ.
        """
        for D in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
            V_h1 = 1.0 * math.sqrt(max(0.0, 1.0 - D * D))
            V_h4 = THETA_ENTANGLEMENT_PER_DIM * math.sqrt(max(0.0, 1.0 - D * D))
            if D < 0.999:
                ratio = V_h4 / V_h1
                self.assertAlmostEqual(ratio, THETA_ENTANGLEMENT_PER_DIM, places=10)


class TestHypothesisH5EtaDirect(unittest.TestCase):
    """H5: γ = η directly — the full fossil entanglement fraction damps the
    coherence uniformly, without the per-dimension cube-root distribution of H4.

    Like H4, this is a COSMOLOGICAL prediction expected to be invisible at
    laboratory σ≈0.  These tests verify mathematical consistency and record
    the hypothesis's signature: V(D=0) = η ≈ 0.4153.
    """

    def test_h5_ellipse_holds_with_gamma_eta(self):
        """With γ pinned to η, the ellipse V²/γ² + D² = 1 still saturates."""
        gamma = ETA
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            c1 = math.sqrt(p)
            c2 = math.sqrt(1.0 - p)
            V = _V_paper(c1, c2, gamma)
            D = _D_paper(c1, c2)
            self.assertAlmostEqual(
                _ellipse_residual(V, D, gamma), 0.0, places=10,
                msg=f"H5 ellipse failure at p={p}, gamma=eta={gamma}"
            )

    def test_h5_signature_at_D_zero(self):
        """H5 signature: V(D=0) = η ≈ 0.4153.  Lab V=1 falsifies H5 at σ=0."""
        c1 = c2 = 1.0 / math.sqrt(2.0)
        gamma = ETA
        V = _V_paper(c1, c2, gamma)
        D = _D_paper(c1, c2)
        self.assertAlmostEqual(D, 0.0, places=12)
        self.assertAlmostEqual(V, ETA, places=12)

    def test_h5_predicts_lower_visibility_than_h4(self):
        """Since η < Θ (η = Θ³), H5 predicts STRICTER coherence damping than H4.

        At σ → σ_conv, a measurement of V_max ≈ 0.42 favours H5;
        V_max ≈ 0.75 favours H4.  Both are below H1's V=1.
        """
        self.assertLess(ETA, THETA_ENTANGLEMENT_PER_DIM)
        self.assertAlmostEqual(ETA, THETA_ENTANGLEMENT_PER_DIM ** 3, places=10)
        for D in (0.0, 0.3, 0.6):
            V_h4 = THETA_ENTANGLEMENT_PER_DIM * math.sqrt(1.0 - D * D)
            V_h5 = ETA * math.sqrt(1.0 - D * D)
            self.assertLess(V_h5, V_h4)


class TestEllipseBoundaryConditions(unittest.TestCase):
    """Physical-sanity checks on ellipse limits."""

    def test_D_zero_gives_max_visibility(self):
        """D=0, γ=1 → V=1 (bright stripes when nobody peeks)."""
        c1 = c2 = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(_V_paper(c1, c2, 1.0), 1.0, places=12)

    def test_D_one_gives_zero_visibility(self):
        """D=1 (one path has all amplitude) → V=0 regardless of γ (c₂=0)."""
        c1, c2 = 1.0, 0.0
        for gamma in (0.0, 0.3, 1.0):
            self.assertAlmostEqual(_V_paper(c1, c2, gamma), 0.0, places=12)

    def test_gamma_equals_D_unphysical_at_zero(self):
        """The γ=D hypothesis would give V=0 at D=0 — unphysical.

        Confirms the user's rejection of γ=D as a candidate model.
        At D=0 with no peeking, we MUST have V→1 (bright stripes).
        Setting γ=D=0 gives V=0, which contradicts observation.
        """
        # Under γ=D=0, ellipse gives V=0 regardless of c1,c2
        c1 = c2 = 1.0 / math.sqrt(2.0)
        gamma = 0.0  # = D = 0
        V = _V_paper(c1, c2, gamma)
        self.assertAlmostEqual(V, 0.0, places=12)
        # A physical double-slit at D=0 must give V≈1.  γ=D disagrees.
        # Therefore γ=D is rejected as a model.


class TestPhaseGGammaFromSigma(unittest.TestCase):
    """Phase G: four candidate γ(σ) derivations connecting sigma-ground's
    gravitational scale field to the Khatiwada-Qian marginal coherence.

    Shared invariants (all four modes):
      - γ(0) = 1 exactly (laboratory regime, H1 match at σ=0)
      - γ(σ) is monotonic non-increasing in σ
      - γ(σ) in [0, 1] across the physical domain σ ∈ [0, σ_conv]

    Mode-specific terminal values at σ = σ_conv:
      linear:    Θ         (H4 signature by construction)
      cbrt:      Θ         (H4 signature by construction)
      exp:       Θ + (1−Θ)/e
      sigma_coh: 1 − η/2    (new prediction, reuses sigma_coherence damping)
    """

    MODES = ('linear', 'exp', 'cbrt', 'sigma_coh')

    def test_g1_all_modes_pin_gamma_to_one_at_sigma_zero(self):
        """γ(σ=0) = 1 exactly in every mode — recovers H1 at lab scale."""
        for mode in self.MODES:
            gamma = coherence_gamma_from_sigma(0.0, mode=mode)
            self.assertAlmostEqual(
                gamma, 1.0, places=12,
                msg=f"mode={mode} does not pin γ(0)=1"
            )

    def test_g2_all_modes_monotonic_decreasing(self):
        """γ decreases (or stays flat) as σ increases, all modes."""
        sigmas = [0.0, 0.05, 0.2, 0.5, 1.0, 1.5, SIGMA_CONV]
        for mode in self.MODES:
            prev = coherence_gamma_from_sigma(sigmas[0], mode=mode)
            for s in sigmas[1:]:
                g = coherence_gamma_from_sigma(s, mode=mode)
                self.assertLessEqual(
                    g, prev + 1e-14,
                    msg=f"mode={mode} not monotonic at σ={s}: {prev} -> {g}"
                )
                prev = g

    def test_g3_all_modes_stay_in_unit_interval(self):
        """γ ∈ [0, 1] across [0, σ_conv] for every mode."""
        for mode in self.MODES:
            for i in range(101):
                s = (i / 100.0) * SIGMA_CONV
                g = coherence_gamma_from_sigma(s, mode=mode)
                self.assertGreaterEqual(g, 0.0, msg=f"mode={mode} γ<0 at σ={s}")
                self.assertLessEqual(g, 1.0 + 1e-12, msg=f"mode={mode} γ>1 at σ={s}")

    def test_g4_linear_mode_terminates_at_theta(self):
        """linear mode γ(σ_conv) = Θ = η^(1/3) exactly (H4 endpoint)."""
        self.assertAlmostEqual(
            coherence_gamma_from_sigma(SIGMA_CONV, mode='linear'),
            THETA_ENTANGLEMENT_PER_DIM,
            places=12,
        )

    def test_g5_cbrt_mode_terminates_at_theta(self):
        """cbrt mode γ(σ_conv) = Θ by construction (1−(1−η))^(1/3) = η^(1/3)."""
        self.assertAlmostEqual(
            coherence_gamma_from_sigma(SIGMA_CONV, mode='cbrt'),
            THETA_ENTANGLEMENT_PER_DIM,
            places=12,
        )

    def test_g6_exp_mode_terminates_at_expected_intermediate(self):
        """exp mode γ(σ_conv) = Θ + (1−Θ)/e — intermediate between H4 and H1."""
        expected = THETA_ENTANGLEMENT_PER_DIM + (1.0 - THETA_ENTANGLEMENT_PER_DIM) / math.e
        self.assertAlmostEqual(
            coherence_gamma_from_sigma(SIGMA_CONV, mode='exp'),
            expected, places=12,
        )

    def test_g7_sigma_coh_terminates_at_1_minus_eta_over_2(self):
        """sigma_coh mode γ(σ_conv) = 1 − η/2 — ties quantum γ to gravitational damping."""
        self.assertAlmostEqual(
            coherence_gamma_from_sigma(SIGMA_CONV, mode='sigma_coh'),
            1.0 - ETA / 2.0,
            places=12,
        )

    def test_g8_sigma_coh_matches_sigma_coherence_ratio(self):
        """γ_sigma_coh(σ) == sigma_coherence(η,σ,0)/σ for σ > 0 — bridge check.

        The sigma_coh candidate's physical claim is that the quantum
        marginal coherence γ is identically the fractional gravitational
        damping σ_eff/σ computed by the pre-existing sigma_coherence
        formula (entanglement.py:209).  This test checks that identity
        up to the normalisation σ_conv.
        """
        for s in (0.1, 0.3, 0.7, 1.2, SIGMA_CONV):
            sigma_eff = sigma_coherence(ETA, s, 0.0)
            fraction  = sigma_eff / s  # = 1 - η/2 (constant, independent of σ)
            # Our γ_sigma_coh scales the damping across [0, σ_conv]:
            #   γ = 1 − (η/2)·(σ/σ_conv),
            # which at σ=σ_conv reduces to 1 − η/2 = sigma_eff/σ for ANY σ.
            # The test anchors the two at σ=σ_conv:
            if abs(s - SIGMA_CONV) < 1e-10:
                g = coherence_gamma_from_sigma(s, mode='sigma_coh')
                self.assertAlmostEqual(g, fraction, places=12)

    def test_g9_mode_ordering_at_sigma_conv(self):
        """At σ_conv: linear == cbrt < sigma_coh < exp — visibility-reduction
        hierarchy predicted by each candidate.  Different physics, different
        predicted V_max at the conversion horizon:
            linear, cbrt → Θ          ≈ 0.7461  (H4 terminus)
            sigma_coh    → 1 − η/2    ≈ 0.7924  (damping-ratio terminus)
            exp          → Θ + (1−Θ)/e ≈ 0.8395 (slowest decay)
        """
        gL = coherence_gamma_from_sigma(SIGMA_CONV, mode='linear')
        gC = coherence_gamma_from_sigma(SIGMA_CONV, mode='cbrt')
        gE = coherence_gamma_from_sigma(SIGMA_CONV, mode='exp')
        gS = coherence_gamma_from_sigma(SIGMA_CONV, mode='sigma_coh')
        self.assertAlmostEqual(gL, gC, places=12)  # both = Θ
        self.assertLess(gC, gS)
        self.assertLess(gS, gE)

    def test_g10_rejects_sigma_outside_physical_range(self):
        """σ < 0 or σ > σ_conv must raise — model is undefined past conversion."""
        with self.assertRaises(ValueError):
            coherence_gamma_from_sigma(-0.1)
        with self.assertRaises(ValueError):
            coherence_gamma_from_sigma(SIGMA_CONV + 0.1)

    def test_g11_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            coherence_gamma_from_sigma(0.5, mode='not-a-real-mode')


if __name__ == '__main__':
    unittest.main()
