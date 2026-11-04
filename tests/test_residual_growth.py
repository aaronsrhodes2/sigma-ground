"""The growth fitter (stage 2, phase 4): the three models on synthetic nightly means, the BIC choice, the nominations; and the RTN helper's geometry."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


class TestRTN(unittest.TestCase):
    def test_frame_and_projection(self):
        from sigma_ground.field.interface.rtn import rtn_frame, to_rtn
        r = np.array([1.0, 0.0, 0.0]); v = np.array([0.0, 1.0, 0.0])
        R, T, N = rtn_frame(r, v)
        np.testing.assert_allclose([R, T, N], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-15)
        np.testing.assert_allclose(to_rtn(np.array([0.5, -2.0, 3.0]), r, v), [0.5, -2.0, 3.0])
        np.testing.assert_allclose(to_rtn(np.array([[0.5, -2.0, 3.0]]), np.array([r]), np.array([v])), [[0.5, -2.0, 3.0]])
        with self.assertRaises(ValueError):
            rtn_frame(r, r)


class TestGrowth(unittest.TestCase):
    def test_quadratic_growth_is_found_and_nominated(self):
        import residual_growth as rg
        rng = np.random.default_rng(3)
        lead = np.arange(5, 365, 10.0); t = lead / 365.25
        y = 0.02 + 0.0 * t + 1.5 * t ** 2 + rng.normal(0, 0.02, len(t)); sem = np.full(len(t), 0.02)
        f = rg.fit_growth(lead, y, sem)
        self.assertEqual(f["chosen"], "quadratic")
        self.assertAlmostEqual(f["models"]["quadratic"]["coef"][2], 1.5, delta=0.15)
        c = rg.nominate(f, "nea", 5.0)
        self.assertEqual(c["class"], "unmodelled transverse acceleration"); self.assertGreater(c["nsig"], 5.0)

    def test_flat_residuals_nominate_nothing(self):
        import residual_growth as rg
        rng = np.random.default_rng(4)
        lead = np.arange(5, 200, 7.0); y = 0.1 + rng.normal(0, 0.03, len(lead)); sem = np.full(len(lead), 0.03)
        f = rg.fit_growth(lead, y, sem)
        self.assertEqual(f["chosen"], "constant"); self.assertIsNone(rg.nominate(f, "mba", 5.0))

    def test_linear_growth_is_the_start_class(self):
        import residual_growth as rg
        lead = np.arange(5, 300, 10.0); y = 0.4 * lead / 365.25 + 0.01; sem = np.full(len(lead), 0.02)
        f = rg.fit_growth(lead, y, sem)
        self.assertEqual(f["chosen"], "linear"); self.assertEqual(rg.nominate(f, "mba", 5.0)["class"], "start/constant")


if __name__ == "__main__":
    unittest.main()
