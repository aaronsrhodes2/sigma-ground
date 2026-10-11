"""kernel.ode: every claim is checked against a closed form, never against the integrator's own output."""
from __future__ import annotations

import math
import unittest

from sigma_ground.kernel.ode import Event, dopri5


def _lane_emden(n):
    def f(x, y):
        th, dth = y
        return [dth, -(max(th, 0.0) ** n) - 2.0 * dth / x]
    return f


def _series(x0, n):                                            # regular-singular centre: theta = 1 - x^2/6 + n x^4/120
    return [1 - x0 ** 2 / 6 + n * x0 ** 4 / 120, -x0 / 3 + n * x0 ** 3 / 30]


class TestDopri5(unittest.TestCase):
    def test_exponential_decay(self):                          # BUG
        r = dopri5(lambda t, y: [-y[0]], (0.0, 1.0), [1.0], rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(r.y[-1][0], math.exp(-1.0), delta=1e-10)
        self.assertEqual(r.status, "finished")

    def test_backward_integration(self):                       # BUG
        r = dopri5(lambda t, y: [y[0]], (1.0, 0.0), [math.e], rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(r.y[-1][0], 1.0, delta=1e-10)

    def test_harmonic_oscillator_energy_over_100_periods(self):   # BUG: bound scaled to the tolerance, not to the code
        r = dopri5(lambda t, y: [y[1], -y[0]], (0.0, 200 * math.pi), [1.0, 0.0], rtol=1e-11, atol=1e-13)
        x, v = r.y[-1]
        self.assertLess(abs(0.5 * (x * x + v * v) - 0.5), 1e-6)
        self.assertAlmostEqual(x, 1.0, delta=1e-6)

    def test_lane_emden_n0_n1_n5_closed_forms(self):           # BUG
        x0 = 1e-4
        cases = {0: lambda x: 1 - x * x / 6, 1: lambda x: math.sin(x) / x, 5: lambda x: (1 + x * x / 3) ** -0.5}
        for n, exact in cases.items():
            r = dopri5(_lane_emden(n), (x0, 1.0), _series(x0, n), rtol=1e-12, atol=1e-14)
            self.assertAlmostEqual(r.y[-1][0], exact(1.0), delta=1e-10, msg=f"n={n}")

    def test_terminal_event_locates_first_zero_of_sinc(self):  # BUG: the n=1 surface is at xi_1 = pi exactly
        x0 = 1e-4
        ev = Event(lambda x, y: y[0], terminal=True, direction=-1)
        r = dopri5(_lane_emden(1), (x0, 10.0), _series(x0, 1), rtol=1e-12, atol=1e-14, events=[ev])
        self.assertEqual(r.status, "terminal_event")
        self.assertAlmostEqual(r.t[-1], math.pi, delta=1e-9)
        self.assertAlmostEqual(r.events[0][1], math.pi, delta=1e-9)

    def test_dense_output_between_steps(self):                 # BUG: checks the continuous-extension coefficients
        r = dopri5(lambda t, y: [-y[0]], (0.0, 3.0), [1.0], rtol=1e-10, atol=1e-13)
        for t in (0.0123, 0.5, 1.2345, 2.999):
            self.assertAlmostEqual(r(t)[0], math.exp(-t), delta=2e-9, msg=f"t={t}")

    def test_step_count_is_adaptive(self):                     # BY CONSTRUCTION: a stiffer stretch costs more steps
        smooth = dopri5(lambda t, y: [-y[0]], (0.0, 1.0), [1.0])
        wiggly = dopri5(lambda t, y: [math.cos(50 * t)], (0.0, 1.0), [0.0])
        self.assertGreater(len(wiggly.t), len(smooth.t))


if __name__ == "__main__":
    unittest.main()
