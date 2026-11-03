"""C26's maths levers L1 and L2 (2026-11-02): the zonal reference radius is its own field, the pole epoch is a predictor choice, and every stored predictor is untouched.

Ground truth not from the code under test: DE440's header radii (RE 6378.1366, AM 1738.0, ASUN 696000 km), the J2 force law's R^2 scaling, and the rule that a layer that is off leaves a
stored predictor bit for bit as it was.
"""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

import sigma_ground.field.interface.rolling_shootout as rs
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles


def _pair(field_radius_m: float = 0.0):
    earth = CelestialBody(mass_kg=5.972e24, position_m=np.zeros(3), velocity_m_s=np.zeros(3), radius_m=6371e3, love_number_k2=0.0, j2=1.0826e-3,
                          pole_axis_unit=np.array([0.0, 0.0, 1.0]), field_radius_m=field_radius_m)
    sat = CelestialBody(mass_kg=1.0, position_m=np.array([7000e3, 0.0, 3000e3]), velocity_m_s=np.zeros(3), radius_m=1.0, love_number_k2=0.0)
    return NBodySystem([earth, sat], toggles=PhysicsToggles(j2_zonal=True))


class TestFieldRadius(unittest.TestCase):
    def test_zero_means_the_mean_radius_bit_for_bit(self):                          # BUG: a stored predictor (field_radius 0) must compute exactly what it did
        a0 = _pair().compute_accelerations()[1]
        a1 = _pair(6371e3).compute_accelerations()[1]
        np.testing.assert_array_equal(a0, a1)

    def test_j2_scales_with_the_field_radius_squared(self):                         # BUG: the J2 term reads field_radius_m, not radius_m
        newton = _pair()
        newton.toggles = PhysicsToggles()
        n = newton.compute_accelerations()[1]
        a0 = _pair().compute_accelerations()[1] - n
        a1 = _pair(6378.1366e3).compute_accelerations()[1] - n
        self.assertAlmostEqual(float(np.linalg.norm(a1) / np.linalg.norm(a0)), (6378.1366 / 6371.0) ** 2, places=9)

    def test_body_params_carry_it_into_the_snapshot_bodies(self):                   # BUG: _build_bodies_at_snapshot maps field_radius_km -> field_radius_m
        p = rs.BodyParams(6371.0, 0.3, 1e-3, field_radius_km=6378.1366)
        self.assertEqual(dataclasses.replace(p).field_radius_km, 6378.1366)
        self.assertEqual(rs.BodyParams(6371.0, 0.3, 1e-3).field_radius_km, 0.0)


class TestPrescribedSourceEdges(unittest.TestCase):
    """The Yoshida steps evaluate forces a fraction of a step outside [start, end]; the prescribed-source table continues its end cubic for up to one node spacing."""

    def _src(self):
        from sigma_ground.field.interface.prescribed_sources import PrescribedSources
        t = np.arange(4) * 86400.0
        pos = np.stack([np.stack([t * 1000.0, np.zeros(4), np.zeros(4)], axis=1)], axis=1)      # one source moving at 1000 m/s along x
        vel = np.zeros_like(pos); vel[:, :, 0] = 1000.0
        return PrescribedSources(np.array([1.0]), 0.0, 86400.0, pos, vel)

    def test_inside_is_exact_for_uniform_motion(self):
        s = self._src()
        np.testing.assert_allclose(s.positions_at(43200.0)[0, 0], 43200.0 * 1000.0)

    def test_one_spacing_outside_continues_the_cubic(self):                           # BUG: a Yoshida sub-step at t < 0 raised; now it continues the end segment
        s = self._src()
        np.testing.assert_allclose(s.positions_at(-680.0)[0, 0], -680.0 * 1000.0)
        np.testing.assert_allclose(s.positions_at(3 * 86400.0 + 5000.0)[0, 0], (3 * 86400.0 + 5000.0) * 1000.0)

    def test_further_out_is_an_error(self):
        with self.assertRaises(ValueError):
            self._src().positions_at(-2 * 86400.0)


class TestPredictors(unittest.TestCase):
    def names(self):
        return {p.name: p for p in rs.PREDICTORS}

    def test_the_six_new_labels_exist_and_build_on_restated_zon(self):
        ps = self.names()
        zon = ps["restated_zon"]
        for n in ("restated_hdr_sun", "restated_hdr_earth", "restated_hdr_moon", "restated_geo", "restated_geo_hdr", "restated_geo_hdr_em"):
            p = ps[n]
            self.assertEqual(p.toggles, zon.toggles, n)
            self.assertEqual(p.perturber_sets, zon.perturber_sets, n)
            self.assertEqual(p.lunar_c22, zon.lunar_c22, n)
            self.assertTrue(set(rs._MEASURED_ZONALS) <= set(p.zonal_overrides), n)

    def test_stored_predictors_are_untouched(self):                                 # BUG: new fields default to what the stored labels computed with
        ps = self.names()
        self.assertEqual(ps["restated_zon"].pole_epoch, "snapshot")
        self.assertEqual(ps["restated_zon"].zonal_overrides, rs._MEASURED_ZONALS)
        self.assertTrue(all(a != "field_radius_km" for _, a, _ in ps["restated_hdr"].zonal_overrides))

    def test_header_arms_touch_one_body_on_the_field_radius(self):                  # BUG: L1 arms change the zonal radius alone, never radius_km (tide, SRP disc)
        ps = self.names()
        for one in ("Sun", "Earth", "Moon"):
            extra = [o for o in ps[f"restated_hdr_{one.lower()}"].zonal_overrides if o not in rs._MEASURED_ZONALS]
            self.assertTrue(extra and all(n == one for n, _, _ in extra), one)
            self.assertTrue(all(a != "radius_km" for _, a, _ in extra), one)
            self.assertIn((one, "field_radius_km", {"Sun": 696000.0, "Earth": 6378.1366, "Moon": 1738.0}[one]), extra)

    def test_geo_keeps_the_table_zonals_and_moves_the_pole_epoch(self):
        ps = self.names()
        geo = ps["restated_geo"]
        self.assertEqual(geo.pole_epoch, "start")
        extra = [o for o in geo.zonal_overrides if o not in rs._MEASURED_ZONALS]
        self.assertEqual(sorted(extra), sorted([("Sun", "field_radius_km", 696000.0), ("Earth", "field_radius_km", 6378.1366), ("Moon", "field_radius_km", 1738.0)]))
        em = [o for o in ps["restated_geo_hdr_em"].zonal_overrides if o not in rs._MEASURED_ZONALS]
        self.assertNotIn(("Sun", "j2", 2.1961391516529825e-07), em)
        self.assertIn(("Sun", "field_radius_km", 696000.0), em)

    def test_pole_at_start_differs_from_the_j2010_pole(self):                       # the Earth's pole moves about 0.85 deg per century; 2000 vs 2010 is ~0.085 deg
        p2000 = rs._iau_pole_unit("Earth", 2451544.5)
        p2010 = rs._iau_pole_unit("Earth", 2455197.5)
        ang = np.degrees(np.arccos(float(np.clip(p2000 @ p2010, -1, 1))))
        self.assertTrue(0.05 < ang < 0.2, ang)


if __name__ == "__main__":
    unittest.main()
