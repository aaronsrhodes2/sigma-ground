"""step_adequacy: steps per orbit of every body against the integration step (the step rule of misc/CANDIDATE_LEDGER.md C14, the Io/Europa/Ganymede convergence of C19).

Independent ground truth: Kepler's third law with published masses and semi-major axes (Io 1.769138 d about Jupiter, the Moon 27.32 d sidereal)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.interface.nbody import CelestialBody
from sigma_ground.field.interface.step_rule import step_adequacy, STEPS_PER_ORBIT_TARGET

DAY = 86400.0


def _body(m, x, vy=0.0):
    return CelestialBody(mass_kg=m, position_m=np.array([x, 0.0, 0.0]), velocity_m_s=np.array([0.0, vy, 0.0]), radius_m=1.0, love_number_k2=0.0)


class TestStepAdequacy(unittest.TestCase):
    def setUp(self):
        # Sun, Earth, Moon (Moon about the Earth), and Jupiter with Io
        self.sun, self.earth, self.moon = _body(1.98847e30, 0.0), _body(5.9722e24, 1.495979e11), _body(7.342e22, 1.495979e11 + 3.844e8)
        self.jup, self.io = _body(1.89813e27, 7.7857e11), _body(8.9319e22, 7.7857e11 + 4.2182e8)

    def test_io_period_is_kepler_third_law(self):
        rows = step_adequacy([self.jup, self.io], 0.025 * DAY, names=["Jupiter", "Io"])
        self.assertEqual(len(rows), 1)                              # only Io has a more massive partner
        self.assertEqual(rows[0]["body"], "Io")
        self.assertAlmostEqual(rows[0]["period_s"] / DAY, 1.769138, delta=0.002)

    def test_io_is_inadequate_at_the_shootout_step_and_the_halved_step_doubles_the_count(self):
        a = step_adequacy([self.jup, self.io], 0.025 * DAY)[0]
        b = step_adequacy([self.jup, self.io], 0.0125 * DAY)[0]
        self.assertAlmostEqual(a["steps_per_orbit"], 70.8, delta=0.3)            # C19: "Io at 71 steps per orbit"
        self.assertAlmostEqual(b["steps_per_orbit"] / a["steps_per_orbit"], 2.0, places=9)
        self.assertFalse(a["adequate"])
        self.assertFalse(b["adequate"])                                          # 142 < 300: still not converged (C14 needs 300)
        self.assertTrue(step_adequacy([self.jup, self.io], 1.769138 * DAY / STEPS_PER_ORBIT_TARGET * 0.99)[0]["adequate"])

    def test_the_moon_is_taken_about_the_earth_not_the_sun(self):
        rows = step_adequacy([self.sun, self.earth, self.moon], 0.025 * DAY, names=["Sun", "Earth", "Moon"])
        moon = [r for r in rows if r["body"] == "Moon"][0]
        self.assertEqual(moon["partner"], 1)                                     # the Earth, though the Sun's pull on the Moon is larger
        self.assertAlmostEqual(moon["period_s"] / DAY, 27.32, delta=0.15)
        self.assertTrue(moon["adequate"])                                        # 1092 steps per orbit
        self.assertEqual([r["body"] for r in rows], ["Moon", "Earth"])           # shortest period first; the Sun has no heavier partner

    def test_equal_masses_have_no_partner(self):
        self.assertEqual(step_adequacy([_body(1e24, 0.0), _body(1e24, 1e9)], 100.0), [])


if __name__ == "__main__":
    unittest.main()
