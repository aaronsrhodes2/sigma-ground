"""NBodySystem._t_eval: the time of the positions at which a force is evaluated inside a Forest-Ruth step.

Before 2026-10-13 the engine advanced its clock only at the end of a step, so every force evaluation in the step saw the step's START time. A time-dependent force
(the Moon's rotating figure) was then evaluated up to a step behind, which acts like a constant lag and produced a spurious constant along-track acceleration.
Ground truth: the Forest-Ruth scheme itself. With theta = 1/(2 - 2^(1/3)) the force evaluations sit at t + theta/2 dt, t + dt/2, t + dt - theta/2 dt (the positions after
drifts theta/2, theta/2 + (1 - theta)/2 = 1/2, and 1/2 + (1 - theta)/2), whatever the velocity-dependent layers add (their extra evaluations are at the same positions).
"""
from __future__ import annotations

import unittest

import numpy as np

from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

THETA = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))


class _Recorder(NBodySystem):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen: list[float] = []

    def compute_accelerations(self):
        self.seen.append(self._t_eval)
        return super().compute_accelerations()


def _two_bodies():
    return [CelestialBody(1.989e30, np.zeros(3), np.zeros(3), 6.96e8, 0.0),
            CelestialBody(5.972e24, np.array([1.496e11, 0.0, 0.0]), np.array([0.0, 2.978e4, 0.0]), 6.371e6, 0.0)]


class TestSubstepTime(unittest.TestCase):
    def test_newtonian_step_evaluates_at_the_three_scheme_times(self):          # BUG
        s = _Recorder(_two_bodies(), toggles=PhysicsToggles())
        dt = 3600.0
        s.forest_ruth_step(dt)
        want = [THETA / 2 * dt, dt / 2, dt - THETA / 2 * dt]
        self.assertEqual(len(s.seen), 3)
        for got, w in zip(s.seen, want):
            self.assertAlmostEqual(got, w, delta=1e-9)

    def test_times_continue_from_the_clock_on_the_next_step(self):             # BUG
        s = _Recorder(_two_bodies(), toggles=PhysicsToggles())
        dt = 600.0
        s.forest_ruth_step(dt)
        s.seen.clear()
        s.forest_ruth_step(dt)
        self.assertAlmostEqual(s.seen[1], dt + dt / 2, delta=1e-9)
        self.assertAlmostEqual(s._time, 2 * dt, delta=1e-12)
        self.assertAlmostEqual(s._t_eval, s._time, delta=1e-12)                 # outside a step it equals the clock

    def test_velocity_dependent_layers_add_evaluations_but_not_new_times(self):  # BUG
        s = _Recorder(_two_bodies(), toggles=PhysicsToggles(eih_cross=True))
        dt = 3600.0
        s.forest_ruth_step(dt)
        want = {THETA / 2 * dt, dt / 2, dt - THETA / 2 * dt}
        got = {round(t, 6) for t in s.seen}
        self.assertEqual(got, {round(w, 6) for w in want})
        self.assertGreater(len(s.seen), 3)                                       # the implicit kicks re-evaluate at the same positions

    def test_the_trajectory_is_unchanged(self):                                  # BY CONSTRUCTION: tracking time must not move a single bit of the orbit
        a, b = _two_bodies(), _two_bodies()
        s1, s2 = NBodySystem(a, toggles=PhysicsToggles(eih_cross=True)), _Recorder(b, toggles=PhysicsToggles(eih_cross=True))
        for _ in range(20):
            s1.forest_ruth_step(1800.0)
            s2.forest_ruth_step(1800.0)
        for x, y in zip(s1.bodies, s2.bodies):
            self.assertTrue(np.array_equal(x.position_m, y.position_m))


if __name__ == "__main__":
    unittest.main()
