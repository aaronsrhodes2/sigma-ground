"""Regression tests for the rolling DE440 shoot-out (rolling_shootout.py).

Each test checks the shoot-out against ground truth that does not come from
the shoot-out itself: the JPL DE440 state vectors in the fixture, closed
forms, or (where installed) the IAU 2006/2000A precession-nutation model in
erfa. Fast by default; the one long integration is gated behind
MATERIA_RUN_PROPAGATION=1, like the other propagation tests.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from sigma_ground.field.interface import rolling_shootout as rs
from sigma_ground.field.interface.orbital import J2000_OBLIQUITY_DEG


_DE440 = rs._load_de440()
_KEYS = sorted(_DE440["snapshots"])
_WINDOWS = ("j2003", "j2015", "j2024")

_SLOW = os.environ.get("MATERIA_RUN_PROPAGATION", "").strip() == "1"


def _state(snap_key: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    for b in _DE440["snapshots"][snap_key]["bodies"]:
        if b["name"] == name:
            sv = b["state_vector"]
            return (np.array([sv["x_km"], sv["y_km"], sv["z_km"]]),
                    np.array([sv["vx_km_s"], sv["vy_km_s"], sv["vz_km_s"]]))
    raise KeyError(name)


def _orbit_normal(snap_key: str, body: str, parent: str) -> np.ndarray:
    r1, v1 = _state(snap_key, body)
    r0, v0 = _state(snap_key, parent)
    h = np.cross(r1 - r0, v1 - v0)
    return h / np.linalg.norm(h)


def _mean_orbit_normal(body: str, parent: str) -> np.ndarray:
    """Orbit normal averaged over the 27 annual DE440 snapshots.

    For a moon whose node precesses about its Laplace plane, the average over
    many node phases is the Laplace-plane pole, which for a close-in moon is
    the planet's spin axis.
    """
    m = sum(_orbit_normal(k, body, parent) for k in _KEYS)
    return m / np.linalg.norm(m)


def _angle_deg(a: np.ndarray, b: np.ndarray, sign_free: bool = False) -> float:
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sign_free:
        c = abs(c)
    return math.degrees(math.acos(c))


def _poles_at(snap_key: str) -> dict[str, np.ndarray]:
    bodies, names, _, _ = rs._build_bodies_at_snapshot(
        snap_key, _DE440, rs.PREDICTORS[0])
    return {n: b.pole_axis_unit for n, b in zip(names, bodies)}


# ── F1: zonal harmonics about each body's own spin axis ──────────────────

class TestZonalPoles:
    """The shoot-out's J2/J3/J4 must act about each body's IAU spin axis.

    Ground truth comes from the DE440 fixture's own dynamics: close-in moons
    orbit in their planet's equator (Io 0.04 deg, Enceladus 0.01 deg, Phobos
    1.1 deg with a 2.3-yr node period that the 27-snapshot mean averages
    out), synchronous moons and Mercury sit in Cassini states with spin axis
    within a fraction of a degree of their orbit normal, Venus is 2.64 deg
    from its orbit normal (obliquity 177.36 deg), and the Moon's axis is
    1.54 deg from the ecliptic pole, coplanar with and opposite its orbit
    normal (Cassini's laws).
    """

    @pytest.mark.parametrize("snap", _WINDOWS)
    @pytest.mark.parametrize("planet,moon,tol", [
        ("Jupiter", "Io", 0.05),
        ("Saturn", "Enceladus", 0.05),
        ("Saturn", "Dione", 0.05),
        ("Mars", "Phobos", 0.05),
        ("Uranus", "Umbriel", 0.3),
        ("Uranus", "Oberon", 0.3),
    ])
    def test_planet_pole_is_equatorial_moon_normal(self, snap, planet, moon, tol):
        pole = _poles_at(snap)[planet]
        ang = _angle_deg(pole, _mean_orbit_normal(moon, planet), sign_free=True)
        assert ang < tol, (f"{planet} pole is {ang:.3f} deg from the mean "
                           f"orbit normal of {moon} (DE440)")

    @pytest.mark.parametrize("snap", _WINDOWS)
    @pytest.mark.parametrize("body,parent,tol", [
        ("Io", "Jupiter", 0.05),
        ("Europa", "Jupiter", 0.05),
        ("Ganymede", "Jupiter", 0.05),
        ("Callisto", "Jupiter", 0.05),
        ("Enceladus", "Saturn", 0.1),
        ("Titan", "Saturn", 0.5),     # Titan's obliquity is ~0.3 deg
        ("Mercury", "Sun", 0.06),     # Mercury's obliquity is 2.04 arcmin
    ])
    def test_cassini_state_bodies(self, snap, body, parent, tol):
        pole = _poles_at(snap)[body]
        ang = _angle_deg(pole, _orbit_normal(snap, body, parent))
        assert ang < tol, f"{body} pole is {ang:.3f} deg from its orbit normal"

    @pytest.mark.parametrize("snap", _WINDOWS)
    def test_venus_obliquity(self, snap):
        pole = _poles_at(snap)["Venus"]
        ang = _angle_deg(pole, _orbit_normal(snap, "Venus", "Sun"), sign_free=True)
        assert abs(ang - (180.0 - 177.36)) < 0.2, f"Venus axis {ang:.3f} deg from normal"

    @pytest.mark.parametrize("snap", _WINDOWS)
    def test_moon_cassini_state(self, snap):
        pole = _poles_at(snap)["Moon"]
        eps = math.radians(J2000_OBLIQUITY_DEG)
        ecl = np.array([0.0, -math.sin(eps), math.cos(eps)])
        n = _orbit_normal(snap, "Moon", "Earth")
        a_ecl = _angle_deg(pole, ecl)
        assert abs(a_ecl - 1.543) < 0.05, f"Moon axis {a_ecl:.3f} deg from ecliptic pole"
        # coplanar with the ecliptic pole and the orbit normal ...
        assert abs(float(np.dot(pole, np.cross(ecl, n)))) < 1e-3
        # ... and on the far side of the ecliptic pole from the orbit normal
        assert abs(_angle_deg(pole, n) - (a_ecl + _angle_deg(ecl, n))) < 0.05

    @pytest.mark.parametrize("snap", _WINDOWS)
    def test_earth_pole_matches_iau2006_cip(self, snap):
        erfa = pytest.importorskip("erfa")
        jd = _DE440["snapshots"][snap]["epoch"]["jd_tdb"]
        x, y = erfa.xy06(2400000.5, jd - 2400000.5)
        cip = np.array([x, y, math.sqrt(1.0 - x * x - y * y)])
        ang = _angle_deg(_poles_at(snap)["Earth"], cip)
        assert ang < 0.01, f"Earth pole {ang:.4f} deg from the IAU 2006/2000A CIP"

    def test_neptune_pole_carries_its_periodic_term(self):
        """IAU 2015: RA = 299.36 + 0.70 sin N, Dec = 43.46 - 0.51 cos N,
        N = 357.85 + 52.316 T (Archinal et al. 2018, Table 1)."""
        for snap in _WINDOWS:
            jd = _DE440["snapshots"][snap]["epoch"]["jd_tdb"]
            T = (jd - 2451545.0) / 36525.0
            N = math.radians(357.85 + 52.316 * T)
            want = rs._pole_unit_from_radec(299.36 + 0.70 * math.sin(N),
                                            43.46 - 0.51 * math.cos(N))
            assert _angle_deg(_poles_at(snap)["Neptune"], want) < 1e-9

    def test_every_zonal_body_has_an_iau_pole(self):
        """Golden Rule 9: no body with a zonal coefficient falls back to +z."""
        for name, p in rs._BODY_PARAMS.items():
            if p.j2 or p.j3 or p.j4:
                assert name in rs._IAU_POLE_MODELS, name

    @pytest.mark.skipif(not _SLOW, reason="1-yr Jupiter-system integration "
                        "(~2.5 min); set MATERIA_RUN_PROPAGATION=1")
    def test_galileans_one_year_with_zonals(self):
        _galileans_one_year_with_zonals()


def _galileans_one_year_with_zonals() -> None:
    """Sun, planets, Moon and the Galileans, FR4 Newton + J2 + J4 at
    dt = 0.05 d from j2015: Ganymede's and Callisto's Jupiter-relative
    positions at j2016 against DE440. With Jupiter's J2/J4 about ICRS +z
    they were 20,815 and 3,440 km off; about the IAU pole, 1,054 and 25 km.
    (Io and Europa are integrator-limited at this dt, so they are not used.)
    """
    from sigma_ground.field.interface.nbody import NBodySystem, PhysicsToggles
    keep = {"Sun", "Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter",
            "Saturn", "Uranus", "Neptune", "Io", "Europa", "Ganymede",
            "Callisto"}
    bodies, names, _, _ = rs._build_bodies_at_snapshot(
        "j2015", _DE440, rs.PREDICTORS[0])
    sel = [i for i, n in enumerate(names) if n in keep]
    bodies = [bodies[i] for i in sel]
    names = [names[i] for i in sel]
    system = NBodySystem(bodies, toggles=PhysicsToggles(j2_zonal=True, j4_zonal=True))
    span = (_DE440["snapshots"]["j2016"]["epoch"]["jd_tdb"]
            - _DE440["snapshots"]["j2015"]["epoch"]["jd_tdb"])
    n = int(round(span / 0.05))
    for _ in range(n):
        system.forest_ruth_step(span * 86400.0 / n)
    ij = names.index("Jupiter")
    for moon, tol_km in (("Ganymede", 3000.0), ("Callisto", 200.0)):
        im = names.index(moon)
        pred = (system.bodies[im].position_m - system.bodies[ij].position_m) / 1e3
        truth = _state("j2016", moon)[0] - _state("j2016", "Jupiter")[0]
        err = float(np.linalg.norm(pred - truth))
        assert err < tol_km, f"{moon} {err:.0f} km from DE440 after 1 yr"
