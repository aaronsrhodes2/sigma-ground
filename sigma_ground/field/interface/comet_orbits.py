"""Comets as massless test particles: Newtonian pull of the Sun and the planet systems, the Sun's 1PN term, and the Marsden non-gravitational acceleration.

A comet's mass is 1e-9 of the Sun's at most, so it cannot change any planet's motion; the only reason to carry comets is to PREDICT THEM. This module is the propagation model. Truth for
scoring is JPL's own solution of each comet (Horizons state vectors), a fit to astrometry that includes the non-gravitational parameters (SBDB A1, A2, A3).

Derivation chain (Golden Rule 1):
  MEASURED / catalogued : the DE440s kernel positions of the Sun and the nine planet-system barycentres; their GM (DE440 fixture); SBDB non-gravitational parameters A1, A2, A3 (au/day^2).
  FIRST_PRINCIPLES      : Newtonian pull a = -sum_k GM_k (r - r_k)/|r - r_k|^3 in the barycentric frame; the Sun's 1PN test-particle term (isotropic PPN, gamma = beta = 1):
                          a = GM/(c^2 r^3) [ (4 GM/r - v^2) r + 4 (r.v) v ]  with r, v heliocentric (Will 1993; verified by its perihelion precession 6 pi GM/(c^2 a (1-e^2)) per orbit).
  BORROWED              : the Marsden et al. (1973) non-gravitational law  a = g(r) (A1 R + A2 T + A3 N), R the unit radial vector, N = R x v / |R x v|, T = N x R, and
                          g(r) = alpha (r/r0)^-m (1 + (r/r0)^n)^-k with the water-ice constants alpha = 0.1112620426, r0 = 2.808 au, m = 2.15, n = 5.093, k = 4.6142.
                          (BORROWED because the constants are a fit to cometary activity, not derived; the candidate ladder tests whether they help.)
Golden Rule 2: valid for a body that stays outside the planets' spheres of influence by a wide margin; a close planetary encounter is integrated honestly (adaptive step) but its
  sensitivity to the initial state is real. Not included: the asteroids' pull, the time delay DT of the Marsden law, outgassing outbursts, mass loss, fragmentation.
NUMERICAL: adaptive Dormand-Prince at rtol 1e-13: 0.12 km after three periods of an e = 0.85, a = 3 au orbit (error falls ~10x per decade of rtol: 43 km at 1e-10).
PHYSICS_GAP: the Sun's 1PN term only (the planets' relativistic terms on a comet are ~1e-9 of the Sun's).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from sigma_ground.kernel.ode import dopri5

AU_KM = 149597870.7
DAY_S = 86400.0
C_KM_S = 299792.458
# water-ice Marsden law (Marsden, Sekanina & Yeomans 1973)
MARSDEN_WATER = dict(alpha=0.1112620426, r0=2.808, m=2.15, n=5.093, k=4.6142)
SEGMENTS = [(0, 10), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)]       # Sun, nine planet-system barycentres


def marsden_g(r_au: float, alpha: float = 0.1112620426, r0: float = 2.808, m: float = 2.15, n: float = 5.093, k: float = 4.6142) -> float:
    x = r_au / r0
    return alpha * x ** (-m) * (1.0 + x ** n) ** (-k)


def nongrav_acceleration(r_helio_km: np.ndarray, v_helio_km_day: np.ndarray, A: Sequence[float], law: dict | None = None) -> np.ndarray:
    """Marsden acceleration in km/day^2 for A = (A1, A2, A3) in au/day^2 and heliocentric state in km, km/day."""
    r = float(np.linalg.norm(r_helio_km))
    R = r_helio_km / r
    h = np.cross(r_helio_km, v_helio_km_day)
    N = h / np.linalg.norm(h)
    T = np.cross(N, R)
    g = marsden_g(r / AU_KM, **(law or MARSDEN_WATER))
    return g * AU_KM * (A[0] * R + A[1] * T + A[2] * N)


def sun_1pn_acceleration(r_helio_km: np.ndarray, v_helio_km_day: np.ndarray, gm_sun_km3_s2: float) -> np.ndarray:
    """Isotropic-PPN 1PN acceleration of a test particle in the Sun's field, km/day^2 (heliocentric r, v)."""
    r = float(np.linalg.norm(r_helio_km))
    v = v_helio_km_day / DAY_S                                              # km/s
    gm = gm_sun_km3_s2
    a = gm / (C_KM_S ** 2 * r ** 3) * ((4.0 * gm / r - float(v @ v)) * r_helio_km + 4.0 * float(r_helio_km @ v) * v)    # km/s^2 (r in km)
    return a * DAY_S ** 2


def propagate(y0: Sequence[float], jd0: float, jd_targets: Sequence[float], kernel, gm_planets_km3_s2: Sequence[float], *, nongrav: Sequence[float] | None = None,
              law: dict | None = None, gr: bool = True, rtol: float = 1e-13, atol: float = 1e-6,
              segments: Sequence[tuple[int, int]] = SEGMENTS) -> dict[float, np.ndarray]:
    """Barycentric states (km, km/day) of one comet at each target JD, from the state y0 (km, km/day, barycentric ICRF equatorial) at jd0.

    gm_planets_km3_s2: GM of the Sun then the nine planet-system barycentres (Mercury, Venus, Earth-Moon, Mars, Jupiter ... Pluto), in the order of ``segments``.
    """
    gm = np.asarray(gm_planets_km3_s2, dtype=float) * DAY_S ** 2             # km^3/day^2
    gm_sun_s = float(gm_planets_km3_s2[0])

    def pos_vel_sun(jd):
        p, v = kernel[0, 10].compute_and_differentiate(jd)
        return np.asarray(p, dtype=float), np.asarray(v, dtype=float)

    def f(t, y):
        r = np.array(y[:3])
        v = np.array(y[3:])
        P = np.array([np.asarray(kernel[c, tt].compute(t), dtype=float) for c, tt in segments])         # (10, 3) km
        d = r[None, :] - P
        r3 = np.einsum("ki,ki->k", d, d) ** 1.5
        acc = -np.einsum("k,ki->i", gm / r3, d)
        sp, sv = pos_vel_sun(t)
        rh, vh = r - sp, v - sv
        if gr:
            acc = acc + sun_1pn_acceleration(rh, vh, gm_sun_s)
        if nongrav is not None and any(nongrav):
            acc = acc + nongrav_acceleration(rh, vh, nongrav, law)
        return [y[3], y[4], y[5], acc[0], acc[1], acc[2]]

    out: dict[float, np.ndarray] = {}
    lo, hi = min(list(jd_targets) + [jd0]), max(list(jd_targets) + [jd0])
    for t0, t1 in ((jd0, lo), (jd0, hi)):
        if t0 == t1:
            continue
        res = dopri5(f, (t0, t1), list(y0), rtol=rtol, atol=atol, dense=True)
        for jd in jd_targets:
            if (t1 == lo and jd <= jd0) or (t1 == hi and jd >= jd0):
                out[jd] = np.array(res(jd))
    for jd in jd_targets:
        if jd == jd0:
            out[jd] = np.array(y0, dtype=float)
    return out
