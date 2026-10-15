"""PrescribedSources and the engine layer toggles.prescribed_sources.

Ground truth that does not come from the code under test: (1) cubic Hermite interpolation reproduces any cubic polynomial path exactly, (2) the closed-form circular orbit
bounds the interpolation error, (3) Newton's law summed by hand, (4) the live N-body system itself: a body integrated as a member of the system and the same body replayed as a
prescribed source must pull the others identically (this is the property that makes the layer a faithful replacement for full membership).
"""
from __future__ import annotations

import unittest

import numpy as np

from sigma_ground.field.constants import G
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles
from sigma_ground.field.interface.prescribed_sources import PrescribedSources


class TestInterpolation(unittest.TestCase):
    def test_cubic_polynomial_paths_are_reproduced_exactly(self):                       # BUG
        c = np.array([[1.0, -2.0, 0.5, 0.01], [0.0, 3.0, -0.2, 0.002], [5.0, 0.0, 0.1, -0.003]])      # x(t) = c0 + c1 t + c2 t^2 + c3 t^3 per axis
        h = 7.0
        t = np.arange(6) * h
        pos = np.stack([c[:, 0] + c[:, 1] * tt + c[:, 2] * tt ** 2 + c[:, 3] * tt ** 3 for tt in t])[:, None, :]
        vel = np.stack([c[:, 1] + 2 * c[:, 2] * tt + 3 * c[:, 3] * tt ** 2 for tt in t])[:, None, :]
        src = PrescribedSources([1.0], 0.0, h, pos, vel)
        for tt in (0.0, 3.3, 7.0, 20.1, 34.9, 35.0):
            want = c[:, 0] + c[:, 1] * tt + c[:, 2] * tt ** 2 + c[:, 3] * tt ** 3
            self.assertLess(np.linalg.norm(src.positions_at(tt)[0] - want), 1e-9 * (1 + np.linalg.norm(want)))

    def test_circular_orbit_interpolation_error_is_small(self):                         # BUG: bound (h^4/384) a (w h)^4 -> relative (w h)^4 / 384
        a, w = 3.0e11, 2 * np.pi / (4.6 * 365.25 * 86400.0)                              # a 4.6-yr orbit at ~2.8 AU
        h = 86400.0
        t = np.arange(40) * h
        pos = np.stack([a * np.array([np.cos(w * tt), np.sin(w * tt), 0.0]) for tt in t])[:, None, :]
        vel = np.stack([a * w * np.array([-np.sin(w * tt), np.cos(w * tt), 0.0]) for tt in t])[:, None, :]
        src = PrescribedSources([1.0], 0.0, h, pos, vel)
        worst = max(np.linalg.norm(src.positions_at(tt)[0] - a * np.array([np.cos(w * tt), np.sin(w * tt), 0.0])) for tt in np.linspace(0, 38 * h, 77))
        self.assertLess(worst / a, (w * h) ** 4 / 384 * 1.5)

    def test_a_time_outside_the_table_is_refused(self):                                  # BUG: no silent extrapolation
        src = PrescribedSources([1.0], 0.0, 1.0, np.zeros((3, 1, 3)), np.zeros((3, 1, 3)))
        with self.assertRaises(ValueError):
            src.positions_at(2.5)
        with self.assertRaises(ValueError):
            src.positions_at(-0.5)

    def test_inconsistent_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            PrescribedSources([1.0, 2.0], 0.0, 1.0, np.zeros((3, 1, 3)), np.zeros((3, 1, 3)))


class TestForce(unittest.TestCase):
    def test_acceleration_is_newtons_law_summed_by_hand(self):                           # BUG
        gm = np.array([2.0e10, 5.0e9])
        pos = np.array([[[1e9, 0.0, 0.0], [0.0, -3e9, 1e8]]] * 2)
        src = PrescribedSources(gm, 0.0, 10.0, pos, np.zeros_like(pos))
        pts = np.array([[0.0, 0.0, 0.0], [2e9, 1e9, 0.0]])
        got = src.accelerations_on(pts, 5.0)
        for k, p in enumerate(pts):
            want = sum(gm[n] * (pos[0, n] - p) / np.linalg.norm(pos[0, n] - p) ** 3 for n in range(2))
            self.assertLess(np.linalg.norm(got[k] - want), 1e-12 * np.linalg.norm(want))


def _three_bodies(m_ast=2.0e-6):
    msun, mpl = 1.98847e30, 5.97e24
    a_pl, a_as = 1.496e11, 4.2e11
    sun = CelestialBody(msun, np.zeros(3), np.zeros(3), 6.96e8, 0.0)
    pl = CelestialBody(mpl, np.array([a_pl, 0.0, 0.0]), np.array([0.0, np.sqrt(G * msun / a_pl), 0.0]), 6.4e6, 0.0)
    ast = CelestialBody(m_ast * msun, np.array([-a_as * 0.6, a_as * 0.8, 0.0]), np.array([-0.8, -0.6, 0.0]) * np.sqrt(G * msun / a_as), 1e5, 0.0)
    return [sun, pl, ast]


class TestEngineLayer(unittest.TestCase):
    def test_replayed_member_pulls_like_the_live_member(self):                           # BUG: the layer must be a faithful replacement for membership
        dt, steps = 6 * 3600.0, 600
        live = NBodySystem(_three_bodies(), toggles=PhysicsToggles())
        pos, vel = [live.bodies[2].position_m.copy()], [live.bodies[2].velocity_m_s.copy()]
        for _ in range(steps):
            live.forest_ruth_step(dt)
            pos.append(live.bodies[2].position_m.copy())
            vel.append(live.bodies[2].velocity_m_s.copy())
        ast_gm = G * _three_bodies()[2].mass_kg
        src = PrescribedSources([ast_gm], 0.0, dt, np.array(pos)[:, None, :], np.array(vel)[:, None, :])
        replay = NBodySystem(_three_bodies()[:2], toggles=PhysicsToggles(prescribed_sources=True), prescribed_sources=src)
        without = NBodySystem(_three_bodies()[:2], toggles=PhysicsToggles())
        for _ in range(steps):
            replay.forest_ruth_step(dt)
            without.forest_ruth_step(dt)
        effect = np.linalg.norm(live.bodies[1].position_m - without.bodies[1].position_m)       # what the asteroid does to the planet
        miss = np.linalg.norm(live.bodies[1].position_m - replay.bodies[1].position_m)           # what the replay gets wrong
        self.assertGreater(effect, 1e3)                                                          # the test would be vacuous if the asteroid did nothing
        self.assertLess(miss, 2e-4 * effect)

    def test_toggle_off_ignores_the_sources_and_is_bit_identical(self):                  # BY CONSTRUCTION
        src = PrescribedSources([1e12], 0.0, 100.0, np.full((5, 1, 3), 1e11), np.zeros((5, 1, 3)))
        a = NBodySystem(_three_bodies()[:2], toggles=PhysicsToggles()).compute_accelerations()
        b = NBodySystem(_three_bodies()[:2], toggles=PhysicsToggles(), prescribed_sources=src).compute_accelerations()
        self.assertTrue(np.array_equal(a, b))

    def test_toggle_on_without_sources_is_rejected(self):                                # BY CONSTRUCTION
        with self.assertRaises(ValueError):
            NBodySystem(_three_bodies()[:2], toggles=PhysicsToggles(prescribed_sources=True))

    def test_the_sources_are_evaluated_at_the_substep_times(self):                       # BUG: same clock as the rotating figure
        seen = []

        class Rec(PrescribedSources):
            def accelerations_on(self, points_m, t_s):
                seen.append(t_s)
                return super().accelerations_on(points_m, t_s)
        src = Rec([1e10], 0.0, 100.0, np.full((20, 1, 3), 3e11), np.zeros((20, 1, 3)))
        s = NBodySystem(_three_bodies()[:2], toggles=PhysicsToggles(prescribed_sources=True), prescribed_sources=src)
        dt = 600.0
        s.forest_ruth_step(dt)
        th = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        self.assertEqual(len(seen), 3)
        for got, want in zip(seen, (th / 2 * dt, dt / 2, dt - th / 2 * dt)):
            self.assertAlmostEqual(got, want, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
