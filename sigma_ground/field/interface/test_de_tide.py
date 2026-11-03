"""C26 levers L3 and L4 (2026-11-02): DE430's time-delayed Earth tide and the Sun's linear GM rate.

Ground truth that does not come from the code under test: with zero delays and one Love number in all three bands, the three band terms of Folkner et al. 2014 eq. (32) must sum
to the isotropic degree-2 result, -3 k2 G m_T R^5 r / r^7 times (mE + mM)/mE, for ANY spin-axis direction (that is what checks the transcription of the equation); a nonzero
rotational delay must move the acceleration off the Earth-Moon line in the sense of the Earth's spin (the bulge leads); and GM(t) = GM(0)(1 + rate t).
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.constants import G
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

M_E, M_M, R_E = 5.9722e24, 7.342e22, 6378.1366e3


def _system(pole, k2=(0.3, 0.3, 0.3), tau_o=(0.0, 0.0, 0.0), tau_r=(0.0, 0.0, 0.0), moon_pos=(384400e3, 0.0, 0.0), moon_vel=(0.0, 1022.0, 0.0)):
    pole = np.asarray(pole, float) / np.linalg.norm(pole)
    earth = CelestialBody(mass_kg=M_E, position_m=np.zeros(3), velocity_m_s=np.zeros(3), radius_m=6371e3, love_number_k2=0.0, pole_axis_unit=pole)
    moon = CelestialBody(mass_kg=M_M, position_m=np.array(moon_pos, float), velocity_m_s=np.array(moon_vel, float), radius_m=1737e3, love_number_k2=0.0)
    de = {"earth": 0, "moon": 1, "sun": None, "k2": k2, "tau_orbit_s": tau_o, "tau_rot_s": tau_r, "radius_m": R_E, "theta_dot": 7.292115e-5}
    return NBodySystem([earth, moon], toggles=PhysicsToggles(earth_tide_de=True), de_tide=de)


def _tide_only(sys_):
    a = sys_.compute_accelerations()
    sys_.toggles = PhysicsToggles()
    return a - sys_.compute_accelerations()


class TestDeTide(unittest.TestCase):
    def test_zero_delay_equal_bands_is_the_isotropic_love_result_for_any_pole(self):     # BUG: the transcription of eq. (32)
        r = np.array([384400e3, 0.0, 0.0])
        for pole in ([0, 0, 1], [0, 1, 0], [1, 0, 0], [0.3, -0.5, 0.81], [0.1, 0.9, 0.2]):
            a = _tide_only(_system(pole))
            expected_rel = -3.0 * 0.3 * G * M_M * R_E ** 5 * r / np.linalg.norm(r) ** 8 * (M_E + M_M) / M_E      # -3 k2 G m_T R^5 r_hat / r^7
            # rtol 2e-5: the tide (4e-14 m/s^2) is read by subtracting the Newtonian 2.7e-3 m/s^2, whose round-off is 3e-19 (1e-5 of the tide)
            np.testing.assert_allclose(a[1], expected_rel * M_E / (M_E + M_M), rtol=2e-5, atol=1e-25, err_msg=str(pole))
            np.testing.assert_allclose(a[0], -expected_rel * M_M / (M_E + M_M), rtol=2e-5, atol=1e-25, err_msg=str(pole))

    def test_barycentre_is_untouched(self):                                                # BY CONSTRUCTION: the memo's statement
        a = _tide_only(_system([0.2, 0.1, 0.97], k2=(0.335, 0.277, 0.283), tau_o=(0.077 * 86400, -0.01 * 86400, -0.2098 * 86400), tau_r=(0.0, 0.0119 * 86400, -0.00074 * 86400)))
        self.assertLess(float(np.linalg.norm(M_E * a[0] + M_M * a[1])), 1e-4 * float(np.linalg.norm(M_M * a[1])))   # the Newtonian subtraction leaves ~2e4 N of round-off; the tide is ~3e9 N

    def test_rotational_delay_makes_the_bulge_lead_the_moon(self):                           # the along-track component points along the spin (prograde) for a positive delay
        a0 = _tide_only(_system([0, 0, 1], k2=(0.0, 0.0, 0.3)))
        a1 = _tide_only(_system([0, 0, 1], k2=(0.0, 0.0, 0.3), tau_r=(0.0, 0.0, 600.0)))
        self.assertAlmostEqual(float(a0[1][1]), 0.0, places=20)
        self.assertGreater(float(a1[1][1]), 0.0)                                              # +y is the direction of the Earth's spin at +x
        self.assertLess(abs(a1[1][1]), abs(a1[1][0]))

    def test_orbital_delay_uses_the_earlier_position(self):                                  # r_T(t - tau) lies behind the Moon along its motion (semi-diurnal band: the zonal band is axisymmetric and cannot pull along-track)
        a0 = _tide_only(_system([0, 0, 1], k2=(0.0, 0.0, 0.3)))
        a1 = _tide_only(_system([0, 0, 1], k2=(0.0, 0.0, 0.3), tau_o=(0.0, 0.0, 600.0)))
        self.assertAlmostEqual(float(a0[1][1]), 0.0, places=20)
        self.assertLess(float(a1[1][1]), 0.0)                                                 # the bulge sits behind the Moon: a retarding pull


class TestGmRate(unittest.TestCase):
    def test_mass_follows_the_rate_at_the_evaluation_time(self):
        sun = CelestialBody(mass_kg=1.989e30, position_m=np.zeros(3), velocity_m_s=np.zeros(3), radius_m=7e8, love_number_k2=0.0)
        earth = CelestialBody(mass_kg=M_E, position_m=np.array([1.496e11, 0.0, 0.0]), velocity_m_s=np.array([0.0, 29780.0, 0.0]), radius_m=6371e3, love_number_k2=0.0)
        s = NBodySystem([sun, earth], toggles=PhysicsToggles(gm_rate=True), gm_rates_per_yr={0: -9e-14})
        a0 = s.compute_accelerations()[1]
        s._t_eval = 10 * 31557600.0
        a1 = s.compute_accelerations()[1]
        self.assertAlmostEqual(float(np.linalg.norm(a1) / np.linalg.norm(a0)), 1.0 - 9e-13, places=15)
        self.assertEqual(s.bodies[0].mass_kg, 1.989e30 * (1.0 - 9e-13))

    def test_toggle_off_leaves_masses_alone(self):
        sun = CelestialBody(mass_kg=1.989e30, position_m=np.zeros(3), velocity_m_s=np.zeros(3), radius_m=7e8, love_number_k2=0.0)
        s = NBodySystem([sun], toggles=PhysicsToggles(), gm_rates_per_yr={0: -9e-14})
        s._t_eval = 1e9
        s.compute_accelerations()
        self.assertEqual(s.bodies[0].mass_kg, 1.989e30)


if __name__ == "__main__":
    unittest.main()
