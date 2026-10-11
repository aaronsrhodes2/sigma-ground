"""Regression tests for field.asteroids Bennu and Ryugu shapes (fix queue C4, A48).

Ground truth is the mission literature:
  Bennu: Barnouin et al. 2019, Nat. Geosci. 12, 247 (Table 1): best-fit
    ellipsoid semi-axes 252.78 x 246.20 x 228.69 m; shape-model volume
    0.0615 km^3; bulk density 1190 +- 13 kg/m^3; mass 7.329e10 kg.
  Ryugu: Watanabe et al. 2019, Science 364, 268: equatorial radius 502 m,
    polar radius 438 m, volume 0.377 +- 0.005 km^3, density 1190 kg/m^3.
"""

import math
import unittest

from sigma_ground.field import asteroids as A


class TestBennuRyuguShapes(unittest.TestCase):

    def test_bennu_semi_axes_are_the_best_fit_ellipsoid(self):
        """Before: 131.25 x 129.25 x 124.65 m (halved)."""
        b = A.BENNU
        self.assertEqual((b["a_m"], b["b_m"], b["c_m"]), (252.78, 246.20, 228.69))

    def test_bennu_mass_volume_density_consistent(self):
        """M / ((4/3) pi abc) within 5 % of the measured 1190 kg/m^3
        (ellipsoid volume is 0.97x the shape model's); before 6.95x."""
        b = A.BENNU
        V = A.ellipsoid_volume(b["a_m"], b["b_m"], b["c_m"])
        self.assertAlmostEqual(V / 0.0615e9, 1.0, delta=0.05)
        self.assertAlmostEqual(b["mass_kg"] / V / 1190.0, 1.0, delta=0.05)
        # volume-equivalent radius of 0.0615 km^3 is 245.0 m
        R = A.mean_radius(b["a_m"], b["b_m"], b["c_m"])
        self.assertAlmostEqual(R / 245.0, 1.0, delta=0.02)

    def test_ryugu_semi_axes_are_the_published_radii(self):
        """Before: 252 x 251 x 237 m (halved)."""
        r = A.RYUGU
        self.assertEqual((r["a_m"], r["b_m"], r["c_m"]), (502.0, 502.0, 438.0))

    def test_ryugu_volume_within_spinning_top_bound(self):
        """The spheroid through the ridge and poles holds 1.23x the 0.377 km^3
        shape-model volume (a spinning top is smaller than its spheroid);
        before: 0.0628 km^3, 6x too small."""
        r = A.RYUGU
        V = A.ellipsoid_volume(r["a_m"], r["b_m"], r["c_m"])
        self.assertTrue(1.0 < V / 0.377e9 < 1.30, V / 0.377e9)
        self.assertTrue(0.75 < r["mass_kg"] / V / 1190.0 < 1.0)

    def test_bennu_escape_velocity_scale(self):
        """The volume-equivalent sphere of Barnouin's 0.0615 km^3 (R = 245.0 m)
        with M = 7.329e10 kg has sqrt(2GM/R) = 0.1998 m/s; the a-tip value
        must be within 3 % of it (before: 0.273 m/s, +37 %)."""
        from sigma_ground.field.constants import G
        R_eq = (3.0 * 0.0615e9 / (4.0 * math.pi)) ** (1.0 / 3.0)
        v_ref = math.sqrt(2.0 * G * 7.329e10 / R_eq)
        v = A.escape_velocity(A.BENNU)["v_a"]
        self.assertAlmostEqual(v / v_ref, 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
