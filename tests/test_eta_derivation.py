"""Tests pinning η as an [EMPIRICAL-INPUT] anchored at DESI Union3 c².

Pre-2026-05-15 this file pinned the consistency of two competing routes:
  1) ETA = 0.4153 (working) from a heuristic dark-energy match
  2) ETA_FORMULA = exp(-φ/σ_conv) ≈ 0.4158 from a 2026-04-17 formula search

A 2026-05-15 audit found that route (2) was numerology by construction
(formula search over a small constant basis hitting a near-match by chance,
with the φ-in-target / φ-in-formula loop being circular) and that route (1)
was itself partly circular (the ρ_released side wasn't independently built).

Resolution: adopt ETA = ETA_HDE_UNION3 = 0.642² ≈ 0.4122 as an empirical
input alongside ξ. The "perfect physics library" loses one slot of false
derivation in exchange for one honest empirical anchor.

These tests pin the new contract:
  - ETA is the DESI Union3 c² value (literally, by code identity).
  - ETA falls inside the DESI Union3 1-σ band by construction.
  - ETA_FORMULA is None (the rejected candidate has been demoted).
  - Removing ETA_FORMULA is non-reversible without theoretical input.

See misc/eta_empirical_verdict_2026-05-15.md for the rejection rationale
and misc/bh_phase_xi_eta_candidates_results.md (now SUPERSEDED) for the
candidate-formula history.
"""

from __future__ import annotations

import math
import unittest

from sigma_ground.field.constants import (
    ETA, ETA_FORMULA, ETA_HDE_UNION3, ETA_UNCERTAINTY_1SIGMA,
    C_HDE_UNION3, PHI, SIGMA_CONV, XI,
)


class TestEtaIsEmpiricalInput(unittest.TestCase):
    """η is now an EMPIRICAL-INPUT alongside ξ, anchored at DESI Union3."""

    def test_eta_is_in_unit_interval(self):
        """η must be a probability/fraction in [0,1]."""
        self.assertGreater(ETA, 0.0)
        self.assertLess(ETA, 1.0)

    def test_eta_equals_desi_union3_c_squared_by_identity(self):
        """ETA is defined as ETA_HDE_UNION3 in constants.py -- the empirical
        anchor must hold by code identity, not by approximation.
        """
        self.assertEqual(ETA, ETA_HDE_UNION3)
        self.assertEqual(ETA, C_HDE_UNION3 ** 2)

    def test_eta_value_matches_published_desi_central(self):
        """DESI 2024 Union3 central c = 0.642 → c² = 0.412164."""
        self.assertAlmostEqual(ETA, 0.412164, places=6)

    def test_eta_in_desi_union3_one_sigma_band(self):
        """Adopted as the central value, ETA sits at the middle of the 1σ band."""
        c_low  = C_HDE_UNION3 - 0.028
        c_high = C_HDE_UNION3 + 0.028
        eta_low, eta_high = c_low ** 2, c_high ** 2
        self.assertGreaterEqual(ETA, eta_low)
        self.assertLessEqual(ETA, eta_high)

    def test_eta_uncertainty_is_two_c_sigma_c(self):
        """σ_η = 2 c σ_c via first-order error propagation on c² → η."""
        expected = 2.0 * C_HDE_UNION3 * 0.028
        self.assertAlmostEqual(ETA_UNCERTAINTY_1SIGMA, expected, places=12)


class TestEtaFormulaRejected(unittest.TestCase):
    """ETA_FORMULA was REJECTED by the 2026-05-15 audit.

    The constant is retained (as None) so existing imports don't silently
    return a stale value. Any code that previously took ETA_FORMULA's
    numeric value will now fail loudly with TypeError, which is what we
    want -- silent acceptance would mean the rejection didn't propagate.
    """

    def test_eta_formula_is_none_after_rejection(self):
        self.assertIsNone(ETA_FORMULA)

    def test_eta_formula_cannot_be_used_in_arithmetic(self):
        """Loud failure if any callsite still expects a number."""
        with self.assertRaises(TypeError):
            _ = ETA_FORMULA + 0.0  # type: ignore[operator]

    def test_exp_neg_phi_over_sigma_conv_still_computes_to_0_4158(self):
        """The numerical formula evaluates the same as it always did -- the
        rejection is epistemic, not arithmetic. This test guards against
        anyone "fixing" the rejection by re-adopting the formula without
        a derivation. If you find yourself wanting to delete this test,
        you need a physical mechanism first, not a recalculation.
        """
        evaluated = math.exp(-PHI / SIGMA_CONV)
        self.assertAlmostEqual(evaluated, 0.4158, places=4)

    def test_evaluated_formula_no_longer_agrees_with_adopted_eta(self):
        """After adopting ETA = c² ≈ 0.4122, the (numerologically rejected)
        formula value 0.4158 disagrees by ~0.87%, NOT 0.125%.

        This is the diagnostic that the old "near-miss" was tracking the
        wrong target. The relabelling forces honesty: 0.87% gap means the
        formula is not even a near-derivation under the new empirical
        anchor; it just happened to track an earlier heuristic value.
        """
        evaluated = math.exp(-PHI / SIGMA_CONV)
        gap_pct = abs(evaluated - ETA) / ETA * 100.0
        # Old gap: 0.125% (vs 0.4153). New gap: ~0.87% (vs 0.4122).
        self.assertGreater(gap_pct, 0.5)
        self.assertLess(gap_pct, 1.5)


class TestEtaProvenanceMetadata(unittest.TestCase):
    """Document the new free-input count and provenance chain."""

    def test_xi_and_eta_are_the_two_empirical_inputs(self):
        """ξ from Planck 2018 and η from DESI 2024 Union3 c² are the two
        irreducible empirical inputs. σ_conv derives from ξ; nothing in
        the model derives η (the previously-claimed dark-energy match
        was retracted as partly circular).
        """
        self.assertAlmostEqual(SIGMA_CONV, -math.log(XI), places=14)
        self.assertEqual(ETA, ETA_HDE_UNION3)

    def test_phi_is_mathematical_not_empirical(self):
        """φ remains pure math (golden ratio). It is no longer claimed to
        appear in any η-derivation; the Phase XI formula was rejected.
        """
        self.assertAlmostEqual(PHI, (1.0 + math.sqrt(5.0)) / 2.0, places=14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
