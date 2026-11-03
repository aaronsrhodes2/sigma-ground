"""Convergence order of NBodySystem.yoshida_step (C26 Phase 3, 2026-11-02): Yoshida's 6th- and 8th-order compositions of the Verlet stage.

Ground truth is independent of the engine, as in test_nbody_integrator_order.py: the textbook right-hand side integrated by scipy's DOP853 at rtol 1e-13. The order is measured by halving
dt on the compact binary of that file (Newtonian and with the 1PN test-particle law, which exercises the implicit kick). A wrong digit in a Yoshida coefficient drops the order to 2, so
this test is also the check on the transcription of his Table 1.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.interface import nbody
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles
from sigma_ground.field.interface.test_nbody_integrator_order import _HAVE_SCIPY, _binary, _reference_separation


def _yoshida_separation(bodies, mode: str, order: int, t_end: float, n_steps: int) -> np.ndarray:
    tog = PhysicsToggles() if mode == "newton" else PhysicsToggles(**{mode: True})
    s = NBodySystem([b.replace() for b in bodies], toggles=tog)
    h = t_end / n_steps
    for _ in range(n_steps):
        s.yoshida_step(h, order)
    return s.bodies[1].position_m - s.bodies[0].position_m


def _orders(bodies, period, mode, order, ns):
    ref = _reference_separation(bodies, mode, period)
    errs = [float(np.linalg.norm(_yoshida_separation(bodies, mode, order, period, n) - ref) / np.linalg.norm(ref)) for n in ns]
    return [math.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)], errs


class TestCoefficients(unittest.TestCase):
    def test_weights_sum_to_one(self):                                              # BY CONSTRUCTION: w0 + 2 sum(w1..wm) = 1 is the consistency condition
        for order, w in nbody._YOSHIDA_W.items():
            self.assertAlmostEqual(w[0] + 2.0 * sum(w[1:]), 1.0, places=14, msg=str(order))
        self.assertEqual(len(nbody._YOSHIDA_W[6]), 4)
        self.assertEqual(len(nbody._YOSHIDA_W[8]), 8)

    def test_force_evaluations_per_step(self):                                      # BUG: m stages cost m + 1 evaluations (adjacent half-kicks merged), 8 and 16 here
        bodies, _ = _binary(2.8e30, 1.0, 2.0e6, 0.0)
        for order, expected in ((6, 8), (8, 16)):
            s = NBodySystem([b.replace() for b in bodies], toggles=PhysicsToggles())
            calls = [0]
            real = s.compute_accelerations

            def counted():
                calls[0] += 1
                return real()
            s.compute_accelerations = counted
            s.yoshida_step(100.0, order)
            self.assertEqual(calls[0], expected, order)


@unittest.skipUnless(_HAVE_SCIPY, "needs scipy (pip install sigma-ground[mcp])")
class TestOrder(unittest.TestCase):
    MILD = _binary(2.8e30, 1.0, 2.0e7, 0.3)          # GM/(r c^2) = 1e-4: the 1PN term is present but the Newtonian truncation dominates, so the order of the scheme itself is what is measured

    def _assert(self, mode, order, ns, expected, delta=0.35):
        orders, errs = _orders(*self.MILD, mode, order, ns)
        for p in orders:
            self.assertAlmostEqual(p, expected, delta=delta, msg=f"yoshida {order} with {mode}: orders {orders}, errors {errs}")

    def test_sixth_order_newton(self):
        self._assert("newton", 6, (40, 80, 160), 6.0)

    def test_eighth_order_newton(self):                                              # below 80 steps per orbit the 8th-order scheme is pre-asymptotic (measured 7.6 at 40 -> 80, 7.9 at 80 -> 160)
        self._assert("newton", 8, (80, 160, 320), 8.0)

    def test_sixth_order_with_gr_1pn(self):
        self._assert("gr_1pn", 6, (40, 80, 160), 6.0)

    def test_eighth_order_with_gr_1pn(self):
        self._assert("gr_1pn", 8, (80, 160, 320), 8.0)


if __name__ == "__main__":
    unittest.main()
