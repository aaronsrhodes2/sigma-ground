"""NBodySystem sectoral-figure layer (PhysicsToggles.sectoral_c22): the engine contract.

Ground truth that does not come from the engine code: the shared force function (itself tested against the engine's J2, the spherical-harmonic derivatives and momentum conservation in
test_lunar_figure.py), the Forest-Ruth sub-step times (test_nbody_substep_time.py), and plain consistency requirements (off = bit-identical, setup errors are rejected).
"""
from __future__ import annotations

import unittest

import numpy as np

from sigma_ground.field.constants import G
from sigma_ground.field.interface import figure_forces as ff
from sigma_ground.field.interface import lunar_figure as lf
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

M_MOON = 7.342e22
R = 1.738e6
JD0 = 2455197.5


def _bodies(c22=lf.C22_MOON):
    earth = CelestialBody(5.972e24, np.zeros(3), np.zeros(3), 6.371e6, 0.0)
    moon = CelestialBody(M_MOON, np.array([3.844e8, 0.0, 0.0]), np.array([0.0, 1.022e3, 0.0]), 1.7374e6, 0.0, pole_axis_unit=np.array([0.1, 0.2, 0.97]),
                         c22=c22, figure_ref_radius_m=R)
    sun = CelestialBody(1.989e30, np.array([1.5e11, 2e10, 3e9]), np.zeros(3), 6.96e8, 0.0)
    return [earth, moon, sun]


def _system(bodies, c22_on=True):
    return NBodySystem(bodies, toggles=PhysicsToggles(sectoral_c22=c22_on), epoch_jd_tdb=JD0, rotation_models={1: lf.moon_axes})


class TestLayerContract(unittest.TestCase):
    def test_toggle_off_is_bit_identical_to_a_system_without_the_feature(self):        # BY CONSTRUCTION
        a = NBodySystem(_bodies(), toggles=PhysicsToggles()).compute_accelerations()
        b = NBodySystem(_bodies(), toggles=PhysicsToggles(sectoral_c22=False), epoch_jd_tdb=JD0, rotation_models={1: lf.moon_axes}).compute_accelerations()
        self.assertTrue(np.array_equal(a, b))

    def test_zero_c22_with_the_toggle_on_changes_nothing(self):                         # BY CONSTRUCTION
        a = NBodySystem(_bodies(c22=0.0), toggles=PhysicsToggles()).compute_accelerations()
        b = _system(_bodies(c22=0.0)).compute_accelerations()
        self.assertTrue(np.array_equal(a, b))

    def test_the_layer_adds_exactly_the_shared_force_function(self):                    # BUG (tolerance: round-off of the large Newtonian term, ~1e-9 of the extra)
        bodies = _bodies()
        on = _system(bodies).compute_accelerations()
        off = NBodySystem(bodies, toggles=PhysicsToggles()).compute_accelerations()
        x, y, _ = lf.moon_axes(JD0)
        T = ff.c22_T(M_MOON, R, lf.C22_MOON, x, y)
        extra = ff.sectoral_extra_accelerations(np.array([b.position_m for b in bodies]), np.array([b.mass_kg for b in bodies]), 1, T)
        self.assertLess(np.linalg.norm(on - off - extra), 1e-6 * np.linalg.norm(extra[0]) + 1e-18)

    def test_it_uses_the_body_reference_radius_not_the_physical_radius(self):           # BUG: figure_ref_radius_m = 1738 km, radius_m = 1737.4 km
        bodies = _bodies()
        extra = _system(bodies).compute_accelerations() - NBodySystem(bodies, toggles=PhysicsToggles()).compute_accelerations()
        bodies2 = [b if i != 1 else CelestialBody(b.mass_kg, b.position_m, b.velocity_m_s, b.radius_m, 0.0, pole_axis_unit=b.pole_axis_unit, c22=b.c22) for i, b in enumerate(bodies)]
        extra2 = _system(bodies2).compute_accelerations() - NBodySystem(bodies2, toggles=PhysicsToggles()).compute_accelerations()
        ratio = np.linalg.norm(extra2[0]) / np.linalg.norm(extra[0])
        self.assertAlmostEqual(ratio, (1.7374e6 / R) ** 2, delta=1e-4)

    def test_setup_errors_are_rejected_not_silent(self):
        with self.assertRaises(ValueError):                                              # c22 set, toggle on, no rotation model
            NBodySystem(_bodies(), toggles=PhysicsToggles(sectoral_c22=True), epoch_jd_tdb=JD0)
        with self.assertRaises(ValueError):                                              # no epoch
            NBodySystem(_bodies(), toggles=PhysicsToggles(sectoral_c22=True), rotation_models={1: lf.moon_axes})


class TestTimeDependence(unittest.TestCase):
    def test_the_rotation_model_is_called_at_the_substep_times(self):                    # BUG: epoch + _t_eval, not the step's start
        calls = []

        def model(jd):
            calls.append(jd)
            return lf.moon_axes(jd)
        s = NBodySystem(_bodies(), toggles=PhysicsToggles(sectoral_c22=True), epoch_jd_tdb=JD0, rotation_models={1: model})
        dt = 3600.0
        s.forest_ruth_step(dt)
        theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        want = [JD0 + (theta / 2 * dt) / 86400.0, JD0 + dt / 2 / 86400.0, JD0 + (dt - theta / 2 * dt) / 86400.0]
        self.assertEqual(len(calls), 3)
        for got, w in zip(calls, want):
            self.assertAlmostEqual(got, w, delta=1e-12)

    def test_total_momentum_is_conserved_over_a_run(self):                              # BUG: Newton III on the whole layer, integrated
        s = _system(_bodies())
        p0 = sum(b.momentum() for b in s.bodies)
        for _ in range(200):
            s.forest_ruth_step(600.0)
        p1 = sum(b.momentum() for b in s.bodies)
        scale = sum(np.linalg.norm(b.momentum()) for b in s.bodies)
        self.assertLess(float(np.linalg.norm(p1 - p0)) / scale, 1e-12)


if __name__ == "__main__":
    unittest.main()
