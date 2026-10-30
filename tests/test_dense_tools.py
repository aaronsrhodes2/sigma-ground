"""chandrasekhar_limit and white_dwarf: checked against computations that do not use the library's own routines.

* The Chandrasekhar mass from a Lane-Emden n = 3 solution integrated HERE with scipy, and CODATA constants from scipy.constants, not from sigma-ground.
* The non-relativistic white-dwarf mass-radius law R ~ M^(-1/3) (the n = 3/2 polytrope), measured as a slope between two low-density stars.
* The two limits together: a dense star approaches the limit from below, mass rises and radius falls with central density."""
from __future__ import annotations

import json
import math
import unittest

import numpy as np


def _lane_emden_omega3() -> float:
    """-xi^2 theta'(xi) at the first zero of theta, for the n = 3 Lane-Emden equation theta'' + (2/xi) theta' + theta^3 = 0."""
    from scipy.integrate import solve_ivp

    def f(xi, y):
        return [y[1], -2.0 * y[1] / xi - max(y[0], 0.0) ** 3]
    xi0 = 1e-6
    sol = solve_ivp(f, (xi0, 10.0), [1.0 - xi0 ** 2 / 6.0, -xi0 / 3.0], method="DOP853", rtol=1e-12, atol=1e-14, events=lambda xi, y: y[0])
    xi1, (_, dth) = sol.t_events[0][0], sol.y_events[0][0]
    return float(-xi1 ** 2 * dth)


class TestChandrasekhar(unittest.TestCase):
    def test_the_limit_equals_the_closed_form_with_an_independently_integrated_lane_emden_constant(self):
        from scipy import constants as sc
        from sigma_ground.mcp.tools import dense_tools as T
        w3 = _lane_emden_omega3()
        self.assertAlmostEqual(w3, 2.01824, places=4)                                      # the textbook constant, reproduced here from the ODE
        hbar, c, G, m_u = sc.hbar, sc.c, sc.G, sc.physical_constants["atomic mass constant"][0]
        for mu_e in (1.0, 2.0, 2.15):
            expected = (math.sqrt(3 * math.pi) / 2) * w3 * (hbar * c / G) ** 1.5 / (mu_e * m_u) ** 2
            got = T.chandrasekhar_limit(mu_e).to_dict()["value"]["mass_kg"]
            self.assertAlmostEqual(got / expected, 1.0, delta=2e-5, msg=f"mu_e={mu_e}")      # CODATA G differs between editions in the 5th digit
        self.assertAlmostEqual(T.chandrasekhar_limit(2.0).to_dict()["value"]["mass_msun"], 1.4558, delta=0.003)   # about 1.46 solar masses for mu_e = 2

    def test_it_scales_as_one_over_mu_e_squared(self):
        from sigma_ground.mcp.tools import dense_tools as T
        a = T.chandrasekhar_limit(2.0).to_dict()["value"]["mass_kg"]
        b = T.chandrasekhar_limit(4.0).to_dict()["value"]["mass_kg"]
        self.assertAlmostEqual(a / b, 4.0, places=9)


class TestWhiteDwarf(unittest.TestCase):
    def test_low_density_stars_follow_the_non_relativistic_mass_radius_law(self):
        from sigma_ground.mcp.tools import dense_tools as T
        a = T.white_dwarf(2e7).to_dict()["value"]
        b = T.white_dwarf(1e8).to_dict()["value"]
        slope = math.log(b["radius_m"] / a["radius_m"]) / math.log(b["mass_kg"] / a["mass_kg"])
        self.assertAlmostEqual(slope, -1.0 / 3.0, delta=0.04)                              # R ~ M^(-1/3) for the n = 3/2 polytrope, up to the first relativistic correction at x ~ 0.3

    def test_mass_rises_and_radius_falls_with_density_and_the_limit_is_not_exceeded(self):
        from sigma_ground.mcp.tools import dense_tools as T
        rows = [T.white_dwarf(r).to_dict()["value"] for r in (1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13)]
        for lo, hi in zip(rows, rows[1:]):
            self.assertGreater(hi["mass_kg"], lo["mass_kg"])
            self.assertLess(hi["radius_m"], lo["radius_m"])
        self.assertTrue(all(r["fraction_of_chandrasekhar_mass"] < 1.0 for r in rows))
        self.assertGreater(rows[-1]["fraction_of_chandrasekhar_mass"], 0.97)                # the densest approaches the limit from below

    def test_a_one_solar_mass_star_is_earth_sized(self):
        from sigma_ground.mcp.tools import dense_tools as T
        row = next(v for v in (T.white_dwarf(r).to_dict()["value"] for r in np.geomspace(1e7, 1e11, 40)) if v["mass_msun"] >= 1.0)
        self.assertTrue(4000 < row["radius_km"] < 12000, row["radius_km"])                  # a 1 solar mass white dwarf is roughly Earth-sized (6,371 km), observed ones 5,000 to 10,000 km

    def test_the_regime_is_stated_and_bad_input_is_declined(self):
        from sigma_ground.mcp.tools import dense_tools as T
        self.assertEqual(T.white_dwarf(1e5).to_dict()["value"]["regime"], "EDGE")
        self.assertEqual(T.white_dwarf(1e9).to_dict()["value"]["regime"], "SAFE")
        self.assertEqual(T.white_dwarf(1e15).to_dict()["value"]["regime"], "WALL")
        for args in ((-1.0,), (0.0,), (1e9, -2.0)):
            r = T.white_dwarf(*args).to_dict()
            self.assertIsNone(r["value"])
            self.assertTrue(r["notes"])
        self.assertIsNone(T.chandrasekhar_limit(0.0).to_dict()["value"])
        json.dumps(T.white_dwarf(1e9).to_dict())

    def test_the_two_tools_agree_with_each_other(self):
        from sigma_ground.mcp.tools import dense_tools as T
        s = T.white_dwarf(1e13).to_dict()["value"]
        self.assertAlmostEqual(s["mass_kg"] / T.chandrasekhar_limit(2.0).to_dict()["value"]["mass_kg"], s["fraction_of_chandrasekhar_mass"], places=12)


if __name__ == "__main__":
    unittest.main()
