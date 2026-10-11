"""Regression tests for orbital.fit_orbit and the GM tables (fix queue C3, A47).

Ground truth is independent of orbital.py:
  - synthetic two-body states from the classical-elements-to-state formulas
    written here (own Kepler solver), generated with the physical
    mu = G(M + m) at the 27 fixture epochs;
  - DE440 GMs typed from JPL SSD Astrodynamic Parameters
    (https://ssd.jpl.nasa.gov/astro_par.html; Park et al. 2021, AJ 161, 105)
    and the JPL SSD satellite physical parameters
    (https://ssd.jpl.nasa.gov/sats/phys_par/);
  - the DE440 fixture snapshots (leave-one-out) and the de440s kernel.
"""

from __future__ import annotations

import copy
import json
import math
import unittest
import warnings
from pathlib import Path

import numpy as np

from sigma_ground.field.interface import orbital as O

_FIXTURE = Path(__file__).parent / "fixtures" / "de440_state_vectors.json"
_OBL = math.radians(23.439291111)

# DE440 [LIT], km^3/s^2 (astro_par; Park et al. 2021)
_DE440 = {
    "Sun": 132_712_440_041.279419, "Mercury": 22_031.868551,
    "Venus": 324_858.592, "Earth": 398_600.435507, "Moon": 4_902.800118,
    "Mars": 42_828.375816, "Jupiter": 126_712_764.1,
    "Saturn": 37_940_584.8418, "Uranus": 5_794_556.4,
    "Neptune": 6_836_527.10058, "Pluto": 975.5,
}
# JPL SSD satellite physical parameters [LIT] (PLU060)
_PLUTO_PLANET, _CHARON = 869.3, 106.1


def _kepler_E(M: float, e: float) -> float:
    E = M if e < 0.8 else math.pi
    for _ in range(100):
        d = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= d
        if abs(d) < 1e-15:
            break
    return E


def _coe2rv(a, e, i, Om, w, M, mu):
    """Classical elements -> (r, v), perifocal rotation R3(-Om) R1(-i) R3(-w)."""
    E = _kepler_E(M % (2 * math.pi), e)
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                        math.sqrt(1 - e) * math.cos(E / 2))
    p = a * (1 - e * e)
    r = p / (1 + e * math.cos(nu))
    rp = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    vp = np.array([-math.sqrt(mu / p) * math.sin(nu),
                   math.sqrt(mu / p) * (e + math.cos(nu)), 0.0])
    cO, sO, ci, si = math.cos(Om), math.sin(Om), math.cos(i), math.sin(i)
    cw, sw = math.cos(w), math.sin(w)
    R = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    return R @ rp, R @ vp


def _ecl2eq(v):
    x, y, z = v
    return np.array([x, math.cos(_OBL) * y - math.sin(_OBL) * z,
                     math.sin(_OBL) * y + math.cos(_OBL) * z])


class TestFitOrbit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(_FIXTURE) as f:
            cls.fix = json.load(f)

    def _synthetic(self, body, anchor, el, mu):
        fx = copy.deepcopy(self.fix)
        a, e, i, Om, w, M0 = el
        n = math.sqrt(mu / a ** 3)
        for snap in fx["snapshots"].values():
            t = (snap["epoch"]["jd_tdb"] - 2451545.0) * 86400.0
            r, v = _coe2rv(a, e, i, Om, w, M0 + n * t, mu)
            r, v = _ecl2eq(r), _ecl2eq(v)
            anc = next(b for b in snap["bodies"] if b["name"] == anchor)["state_vector"]
            sv = next(b for b in snap["bodies"] if b["name"] == body)["state_vector"]
            sv.update(x_km=anc["x_km"] + r[0], y_km=anc["y_km"] + r[1],
                      z_km=anc["z_km"] + r[2], vx_km_s=anc["vx_km_s"] + v[0],
                      vy_km_s=anc["vy_km_s"] + v[1], vz_km_s=anc["vz_km_s"] + v[2])
        return fx, n

    def _check_recovery(self, body, anchor, el, mu):
        fx, n = self._synthetic(body, anchor, el, mu)
        orbit = O.fit_orbit(body, fx)
        self.assertAlmostEqual(orbit.a_km / el[0], 1.0, delta=1e-9)
        self.assertAlmostEqual(orbit.eccentricity, el[1], delta=1e-8)
        for jd in np.linspace(2451700.3, 2460900.7, 25):   # off the fit epochs
            t = (jd - 2451545.0) * 86400.0
            r, _ = _coe2rv(el[0], el[1], el[2], el[3], el[4], el[5] + n * t, mu)
            err = np.linalg.norm(np.array(orbit.predict_relative_ecliptic(jd)) - r)
            self.assertLess(err, 1.0, f"{body} at JD {jd}: {err:.3g} km")

    def test_synthetic_moon_recovered(self):
        """Noise-free two-body Moon (mu = GM_E + GM_M): before, da/a = +1.19 %
        and out-of-sample errors of 5.6e5 km (aliased start, wrong mu)."""
        self._check_recovery(
            "Moon", "Earth",
            (384_400.0, 0.0549, math.radians(5.145), 1.2, 2.3, 0.7),
            _DE440["Earth"] + _DE440["Moon"])

    def test_synthetic_charon_recovered(self):
        """Noise-free two-body Charon (mu = GM_Pluto + GM_Charon): before,
        da/a = +13.9 %."""
        self._check_recovery(
            "Charon", "Pluto",
            (19_596.0, 0.0002, math.radians(96.0), 3.9, 1.0, 2.0),
            _PLUTO_PLANET + _CHARON)

    def test_two_body_mu(self):
        """mu = G(M + m): Earth-Moon from DE440; Pluto + Charon equals the
        DE440 Pluto system GM to the small moons' share (< 0.2 km^3/s^2)."""
        self.assertAlmostEqual(O._two_body_mu("Moon", "Earth"),
                               _DE440["Earth"] + _DE440["Moon"], delta=1e-6)
        self.assertAlmostEqual(O._two_body_mu("Charon", "Pluto"),
                               _DE440["Pluto"], delta=0.2)
        self.assertAlmostEqual(O._two_body_mu("Jupiter", "Sun"),
                               _DE440["Sun"] + _DE440["Jupiter"], delta=1e-3)

    def test_real_moon_leave_one_out(self):
        """Real DE440 Moon, held-out epochs: before, 7.7e5 km median (worse
        than placing the Moon at Earth's centre, 3.9e5 km)."""
        errs = []
        for ex in ("j2003", "j2013", "j2023"):
            orbit = O.fit_orbit("Moon", self.fix, exclude_epoch=ex)
            snap = self.fix["snapshots"][ex]
            pos = {b["name"]: np.array([b["state_vector"][c] for c in ("x_km", "y_km", "z_km")])
                   for b in snap["bodies"]}
            pred = _ecl2eq(orbit.predict_relative_ecliptic(snap["epoch"]["jd_tdb"]))
            errs.append(np.linalg.norm(pred - (pos["Moon"] - pos["Earth"])))
        self.assertLess(float(np.median(errs)), 1.0e5, errs)

    def test_real_moon_vs_kernel(self):
        """Out-of-sample vs de440s at 200 random epochs 2000-2026: the fit
        must beat the trivial predictor 'at Earth's centre' by > 4x."""
        try:
            from jplephem.spk import SPK
            from sigma_ground.field.interface.adapters._jplephem_bridge import _KERNEL_DIR
        except ImportError:
            self.skipTest("jplephem not installed")
        path = _KERNEL_DIR / "de440s.bsp"
        if not path.exists():
            self.skipTest("de440s kernel not installed")
        kernel = SPK.open(str(path))
        orbit = O.fit_orbit("Moon", self.fix)
        jds = np.random.default_rng(7).uniform(2451545.0, 2461041.5, 200)
        errs, trivial = [], []
        for jd in jds:
            truth = np.asarray(kernel[3, 301].compute(jd)) - np.asarray(kernel[3, 399].compute(jd))
            errs.append(np.linalg.norm(_ecl2eq(orbit.predict_relative_ecliptic(jd)) - truth))
            trivial.append(np.linalg.norm(truth))
        self.assertLess(np.median(errs), 0.25 * np.median(trivial))

    def test_unusable_fit_warns(self):
        """Samples no Kepler orbit can describe must not pass silently."""
        fx = copy.deepcopy(self.fix)
        rng = np.random.default_rng(3)
        for snap in fx["snapshots"].values():
            earth = next(b for b in snap["bodies"] if b["name"] == "Earth")["state_vector"]
            moon = next(b for b in snap["bodies"] if b["name"] == "Moon")["state_vector"]
            d = rng.normal(size=3)
            d *= 384_400.0 / np.linalg.norm(d)
            moon.update(x_km=earth["x_km"] + d[0], y_km=earth["y_km"] + d[1],
                        z_km=earth["z_km"] + d[2])
        with self.assertWarns(RuntimeWarning):
            O.fit_orbit("Moon", fx)


class TestDE440GM(unittest.TestCase):
    """ANCHOR_GM and sg_gm carry DE440 values under a DE440 label."""

    def test_anchor_gm_is_de440(self):
        """Before: Sun/Earth were DE430, Mars..Pluto planet-only (Pluto 869.326)."""
        for name, ref in _DE440.items():
            if name in O.ANCHOR_GM:
                self.assertEqual(O.ANCHOR_GM[name], ref, name)

    def test_sg_gm_is_de440(self):
        """Before: Mercury 22032.09 (+10 ppm), no Moon, 'sun' a KeyError."""
        from sigma_ground.field.interface.adapters import SGPhysicsMixin
        m = SGPhysicsMixin()
        for name, ref in _DE440.items():
            self.assertAlmostEqual(m.sg_gm(name) / 1e9, ref, delta=1e-9 * ref, msg=name)
        self.assertEqual(m.sg_gm("sun"), m.sg_gm("Sun"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
