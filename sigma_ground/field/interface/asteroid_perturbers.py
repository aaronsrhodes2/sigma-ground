"""Asteroid perturbers: initial states for the heaviest main-belt asteroids, from JPL SBDB osculating elements and measured masses.

The Mars, Earth and Moon forecasts of an N-body model that carries only the Sun, the planets and the moons are limited by the asteroids it leaves out. This module turns the
SBDB fixture (``fixtures/sbdb_asteroids_54.json``: elements at one epoch and, for the seven heaviest, a measured GM) into barycentric states at any earlier or later epoch.

Method (Golden Rule 1; derivation chain):
  MEASURED      : SBDB osculating elements (a, e, i, Omega, omega, M at epoch, heliocentric ecliptic J2000) and GM (JPL SBDB, public domain);
                  the planet and Sun positions and GMs come from the caller (DE440 kernel and fixture), never from this module.
  FIRST_PRINCIPLES: two-body Kepler conversion elements -> heliocentric state at the element epoch (exact); then the asteroid is integrated as a MASSLESS test particle
                  in the barycentric frame under the Sun and the eight planet-system barycentres, a(r) = -sum GM_k (r - r_k)/|r - r_k|^3, with the one adaptive integrator
                  (kernel/ode.py). The ecliptic-to-equatorial rotation uses the IAU 2006 obliquity of field/interface/fast_ephemeris.py.
Golden Rule 2: ``regime`` of a returned state is SAFE inside +/- 40 yr of the element epoch, EDGE beyond (the test-particle model neglects the asteroids' mutual pull
(about 1e3 km after 25 yr for Ceres, irrelevant to the force it exerts) and non-gravitational forces), BEYOND past 100 yr.
PHYSICS_GAP: asteroids do not perturb each other here; Yarkovsky and radiation pressure are ignored; relativistic terms on the asteroid are ignored (shifts of order 10 m).
PHYSICS_GAP: only objects with a measured GM are returned as massive perturbers; the other ~340 asteroids of the DE440 model are not included.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from sigma_ground.field.interface.fast_ephemeris import _ECL_TO_EQ
from sigma_ground.kernel.ode import dopri5

FIXTURE = Path(__file__).parent / "fixtures" / "sbdb_asteroids_54.json"
AU_KM = 149597870.7                      # IAU 2012 exact
DAY_S = 86400.0
DEFAULT_PERTURBERS = ("Ceres", "Vesta", "Pallas", "Hygiea", "Interamnia", "Psyche", "Kalliope")


@dataclass(frozen=True)
class Perturber:
    name: str
    gm_km3_s2: float
    radius_km: float
    epoch_jd: float
    a_au: float
    e: float
    i_deg: float
    om_deg: float
    w_deg: float
    ma_deg: float


def load_perturbers(names=DEFAULT_PERTURBERS, fixture: Path = FIXTURE) -> list[Perturber]:
    """The named asteroids that carry a measured GM in the SBDB fixture."""
    doc = json.loads(Path(fixture).read_text(encoding="utf-8"))
    by_name = {b.get("name"): b for b in doc["bodies"]}
    out = []
    for n in names:
        b = by_name.get(n)
        if b is None or not b.get("GM_km3_s2"):
            raise ValueError(f"{n!r} is not in the fixture with a measured GM")
        out.append(Perturber(n, float(b["GM_km3_s2"]), 0.5 * float(b["diameter_km"] or 0.0), float(b["epoch_jd_tdb"]), float(b["a_au"]), float(b["e"]),
                             float(b["i_deg"]), float(b["om_deg"]), float(b["w_deg"]), float(b["ma_deg"])))
    return out


def kepler_state_ecliptic(p: Perturber, gm_sun_km3_s2: float) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric ecliptic J2000 position (km) and velocity (km/s) at the element epoch from the two-body solution (exact)."""
    a = p.a_au * AU_KM
    e = p.e
    n = math.sqrt(gm_sun_km3_s2 / a ** 3)
    M = math.radians(p.ma_deg) % (2.0 * math.pi)
    E = M if e < 0.8 else math.pi
    for _ in range(60):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-15:
            break
    cE, sE = math.cos(E), math.sin(E)
    xp, yp = a * (cE - e), a * math.sqrt(1.0 - e * e) * sE
    vxp, vyp = -a * n * sE / (1.0 - e * cE), a * n * math.sqrt(1.0 - e * e) * cE / (1.0 - e * cE)
    Om, w, inc = math.radians(p.om_deg), math.radians(p.w_deg), math.radians(p.i_deg)
    cO, sO, cw, sw, ci, si = math.cos(Om), math.sin(Om), math.cos(w), math.sin(w), math.cos(inc), math.sin(inc)
    R = np.array([[cw * cO - sw * sO * ci, -sw * cO - cw * sO * ci],
                  [cw * sO + sw * cO * ci, -sw * sO + cw * cO * ci],
                  [sw * si, cw * si]])
    return R @ np.array([xp, yp]), R @ np.array([vxp, vyp])


def propagate_to(p: Perturber, target_jds: list[float], attractors: Callable[[float], list[tuple[float, np.ndarray]]],
                 sun_ssb_state: Callable[[float], tuple[np.ndarray, np.ndarray]], gm_sun_km3_s2: float, rtol: float = 1e-11) -> dict:
    """Barycentric equatorial states (km, km/day) of one asteroid at each target JD.

    ``attractors(jd)`` returns [(GM km^3/s^2, position km in the barycentric equatorial frame), ...] for the Sun and the planet systems.
    ``sun_ssb_state(jd)`` returns the Sun's barycentric (position km, velocity km/day) to place the heliocentric elements in that frame.
    """
    r_ecl, v_ecl = kepler_state_ecliptic(p, gm_sun_km3_s2)
    r_eq, v_eq = _ECL_TO_EQ @ r_ecl, _ECL_TO_EQ @ v_ecl                   # km, km/s heliocentric equatorial
    sp, sv = sun_ssb_state(p.epoch_jd)
    y0 = list(r_eq + sp) + list(v_eq * DAY_S + sv)                      # km, km/day, barycentric

    def f(t, y):
        r = np.array(y[:3])
        acc = np.zeros(3)
        for gm, pos in attractors(t):
            d = r - pos
            acc -= gm * DAY_S ** 2 * d / (np.linalg.norm(d) ** 3)         # km/day^2
        return [y[3], y[4], y[5], acc[0], acc[1], acc[2]]

    out = {}
    lo, hi = min(target_jds + [p.epoch_jd]), max(target_jds + [p.epoch_jd])
    # integrate outward from the element epoch in each direction that has targets
    for direction, (t0, t1) in (("back", (p.epoch_jd, lo)), ("fwd", (p.epoch_jd, hi))):
        if t0 == t1:
            continue
        res = dopri5(f, (t0, t1), y0, rtol=rtol, atol=1e-6, dense=True)
        for jd in target_jds:
            if (direction == "back" and jd <= p.epoch_jd) or (direction == "fwd" and jd >= p.epoch_jd):
                out[jd] = res(jd)
    for jd in target_jds:
        if jd == p.epoch_jd:
            out[jd] = list(y0)
    return out
