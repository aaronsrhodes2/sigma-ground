"""Consistency tests for η — the cosmic entanglement fraction.

Two routes give η in the SSBM framework:

  1) ETA = 0.4153
     The "working value", derived heuristically from the dark-energy
     density constraint. Tagged DERIVED in constants.py but the
     derivation is a coherence-ratio argument (see docstring there).

  2) ETA_FORMULA = exp(-φ/σ_conv) ≈ 0.4158
     A candidate first-principles expression connecting η to ξ alone
     (since σ_conv = -ln(ξ)). Currently tagged SPECULATIVE in
     constants.py while the framework decides whether to adopt it.

If (2) is correct, then η is not actually a free input of SSBM -- it's
derived from ξ (the only true free input) via golden-ratio mediation.
That would reduce SSBM's free-parameter count from 1 to 0.

The current numerical gap between the two values is 0.125%. These tests
pin that consistency to a tolerance, so any future drift (whether from
tightening ξ measurements, refining the formula, or replacing one with
the other) gets caught.

This test is intentionally LOOSE today (0.5% tolerance) -- the goal is
to document the open question, not to assert one route is correct.
Tightening the tolerance (or replacing ETA with ETA_FORMULA outright)
is a future judgment call requiring more theoretical input.
"""

from __future__ import annotations

import math
import unittest

from sigma_ground.field.constants import (
    ETA, ETA_FORMULA, PHI, SIGMA_CONV, XI,
)


class TestEtaConsistency(unittest.TestCase):

    def test_eta_is_in_unit_interval(self):
        """η must be a probability/fraction in [0,1]."""
        self.assertGreater(ETA, 0.0)
        self.assertLess(ETA, 1.0)

    def test_eta_formula_is_in_unit_interval(self):
        self.assertGreater(ETA_FORMULA, 0.0)
        self.assertLess(ETA_FORMULA, 1.0)

    def test_sigma_conv_derives_from_xi(self):
        """σ_conv = -ln(ξ) -- a non-negotiable consistency relation."""
        self.assertAlmostEqual(SIGMA_CONV, -math.log(XI), places=14)

    def test_eta_formula_uses_only_xi_and_phi(self):
        """ETA_FORMULA = exp(-φ/σ_conv) -- recompute and verify.

        This pins the formula. If anyone changes ETA_FORMULA's definition
        in constants.py, this test reminds them what it should be.
        """
        expected = math.exp(-PHI / SIGMA_CONV)
        self.assertAlmostEqual(ETA_FORMULA, expected, places=14)

    def test_eta_formula_uses_only_xi(self):
        """The chain ETA_FORMULA -> exp(-φ/σ_conv) -> exp(-φ × (-1/ln ξ))
        means ETA_FORMULA derives from ξ alone (φ being mathematical).
        """
        expected = math.exp(-PHI / (-math.log(XI)))
        self.assertAlmostEqual(ETA_FORMULA, expected, places=14)

    def test_two_eta_routes_agree_within_half_pct(self):
        """The two derivations of η (heuristic vs golden-ratio formula)
        must agree at the half-percent level today. If this test ever
        fails, either:
          - the formula is incorrect and should be revisited
          - the heuristic ETA value has been refined and the formula
            should track it
          - constants.py was edited inconsistently
        """
        rel_err = abs(ETA - ETA_FORMULA) / ETA
        self.assertLess(rel_err, 0.005,  # 0.5% -- generous to allow refinement
                          f"ETA = {ETA:.6f}, ETA_FORMULA = {ETA_FORMULA:.6f}, "
                          f"relative gap = {rel_err*100:.4f}%, exceeds 0.5% bound")

    def test_two_eta_routes_currently_agree_at_0_125_pct(self):
        """Pin the CURRENT numerical gap so it doesn't regress silently.

        Today the gap is 0.125% (ETA_FORMULA gives 0.4158 vs measured 0.4153).
        If this test starts failing, it's because:
          - someone tightened ETA or ETA_FORMULA without updating the other
          - someone changed XI (the only free input)
          - someone refined PHI or SIGMA_CONV (shouldn't happen -- both
            derive from fixed inputs)
        """
        rel_err_pct = abs(ETA - ETA_FORMULA) / ETA * 100
        # Today: 0.125% +/- 0.005% bound
        self.assertAlmostEqual(rel_err_pct, 0.125, delta=0.005,
                                msg=f"ETA / ETA_FORMULA relative gap drifted "
                                    f"from 0.125% to {rel_err_pct:.4f}%")


class TestEtaProvenanceMetadata(unittest.TestCase):
    """Document what reducing SSBM's free-input count to zero requires."""

    def test_xi_is_the_only_planned_free_input(self):
        """ξ from Planck 2018 is the irreducible empirical input.
        η, σ_conv, and downstream quantities derive from it.
        """
        # σ_conv from ξ
        self.assertAlmostEqual(SIGMA_CONV, -math.log(XI), places=14)
        # ETA_FORMULA from ξ (via σ_conv) and pure math (φ)
        expected_eta = math.exp(-PHI / -math.log(XI))
        self.assertAlmostEqual(ETA_FORMULA, expected_eta, places=14)

    def test_phi_is_mathematical_not_empirical(self):
        """φ is golden ratio (1+√5)/2 -- pure math, universe-independent.
        Including it in ETA_FORMULA does NOT introduce a new free parameter.
        """
        self.assertAlmostEqual(PHI, (1.0 + math.sqrt(5.0)) / 2.0, places=14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
