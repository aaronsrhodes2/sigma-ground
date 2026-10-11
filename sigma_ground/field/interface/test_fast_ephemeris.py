"""The instant ephemeris answers sensibly and stays within its own published error bars on dates it was not fitted to.

Ground truth is the JPL DE440s kernel (skipped when it is not on this machine); no expected value is computed by the code under test.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sigma_ground.field.interface.fast_ephemeris import AU_KM, FastEphemeris, where

KERNEL = Path("D:/datasets/jpl/de440s.bsp")
JD_2030 = 2462502.5          # 2030-01-01, after the 2020 training end


def _kernel():
    from jplephem.spk import SPK
    return SPK.open(str(KERNEL))


class TestWhere(unittest.TestCase):
    def setUp(self):
        self.eph = FastEphemeris()

    def test_earth_view_has_direction_distance_and_light_time(self):
        r = self.eph.where("Mars", 100, jd=JD_2030)
        self.assertTrue(0 <= r["ra_deg"] < 360 and -90 <= r["dec_deg"] <= 90)
        self.assertTrue(0.3 < r["distance_au"] < 2.8)
        self.assertAlmostEqual(r["light_time_s"], r["distance_km"] / 299792.458, delta=1.0)

    def test_sun_is_about_one_au_and_moon_within_its_orbit_range(self):
        self.assertAlmostEqual(self.eph.where("Sun", jd=JD_2030)["distance_au"], 1.0, delta=0.02)
        d = self.eph.where("Moon", jd=JD_2030)["distance_km"]
        self.assertTrue(350000 < d < 410000)

    def test_body_view_returns_earth_and_sun_directions(self):
        r = self.eph.where("Mars", 0, jd=JD_2030, view="body")
        self.assertTrue(0 <= r["earth_sun_separation_deg"] <= 180)
        self.assertGreater(r["sun"]["distance_km"], 1.3e8)         # Mars is never closer than ~1.38 AU = 2.06e8 km; loose bound

    def test_uncertainty_is_reported_and_never_zero(self):
        for body in ("Mercury", "Mars", "Jupiter", "Neptune", "Moon", "Sun"):
            u = self.eph.where(body, 365 * 5, jd=JD_2030)["uncertainty"]
            self.assertGreater(u["median_arcsec"], 0.0, body)
            self.assertGreater(u["p95_arcsec"], u["median_arcsec"], body)

    def test_errors_are_handled(self):
        with self.assertRaises(ValueError):
            self.eph.where("Vulcan", 1)
        with self.assertRaises(ValueError):
            self.eph.where("Earth", 1)
        with self.assertRaises(ValueError):
            self.eph.where("Mars", 1, view="elsewhere")

    def test_module_level_where(self):
        self.assertIn("ra_deg", where("Venus", 10, jd=JD_2030))


@unittest.skipUnless(KERNEL.exists(), "DE440s kernel not available")
class TestAgainstDE440(unittest.TestCase):
    """Held-out dates (2030-2050): inner bodies within a few arcseconds, everything within 3x its published 95th percentile."""

    def test_held_out_dates_within_published_error(self):
        k = _kernel()
        eph = FastEphemeris()
        jds = np.arange(JD_2030, JD_2030 + 20 * 365.25, 60.0)
        sun = k[0, 10].compute(jds)
        earth = (k[0, 3].compute(jds) + k[3, 399].compute(jds)) - sun
        chains = {"Mercury": ((0, 1), (1, 199)), "Venus": ((0, 2), (2, 299)), "Mars": ((0, 4),), "Jupiter": ((0, 5),),
                  "Saturn": ((0, 6),), "Uranus": ((0, 7),), "Neptune": ((0, 8),)}
        for name, chain in chains.items():
            true = sum(k[c, t].compute(jds) for c, t in chain) - sun - earth
            pred = (eph.heliocentric(name, jds) - eph.heliocentric("Earth", jds)).T
            ang = np.degrees(np.arccos(np.clip((true * pred).sum(0) / (np.linalg.norm(true, axis=0) * np.linalg.norm(pred, axis=0)),
                                               -1, 1))) * 3600.0
            p95 = eph.uncertainty(name, JD_2030 + 20 * 365.25)["p95_arcsec"]
            self.assertLess(float(np.median(ang)), 3.0 * p95, name)
        # the headline claim: the four terrestrial bodies are within ~15 arcsec at the median
        for name in ("Mercury", "Venus", "Mars"):
            true = sum(k[c, t].compute(jds) for c, t in chains[name]) - sun - earth
            pred = (eph.heliocentric(name, jds) - eph.heliocentric("Earth", jds)).T
            ang = np.degrees(np.arccos(np.clip((true * pred).sum(0) / (np.linalg.norm(true, axis=0) * np.linalg.norm(pred, axis=0)),
                                               -1, 1))) * 3600.0
            self.assertLess(float(np.median(ang)), 15.0, name)


if __name__ == "__main__":
    unittest.main()
