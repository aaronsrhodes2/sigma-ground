"""Paths of the DE440 perturbers (asteroids and the Kuiper-belt ring) over one shoot-out window, as a PrescribedSources table for the N-body engine.

What DE440 carries beyond the planets (Park et al. 2021, AJ 161:105, section 2.2): 343 asteroids (the DE430 set, ~90 % of the belt mass), 30 Kuiper-belt objects, and a circular
Kuiper-belt ring of 36 equal point masses in the ecliptic plane at a = 44 au. Their masses are the constants MAnnnn of the DE440 ASCII header (``header.440``, JPL).

Method (Golden Rule 1; derivation chain):
  MEASURED         : SBDB osculating elements of each asteroid at its element epoch (heliocentric ecliptic J2000); DE440's GM of each perturber; the planets' positions from
                     the DE440s kernel.
  FIRST_PRINCIPLES : two-body elements -> state at the element epoch (``asteroid_perturbers.kepler_state_ecliptic``, exact); then each asteroid is a MASSLESS test particle in the
                     barycentric frame under the Sun and the planet-system barycentres, integrated with fourth-order Runge-Kutta in steps of at most one day (all asteroids at
                     once). The ring is analytic: 36 points on a heliocentric circular Kepler orbit of radius a with the Sun's barycentric motion added.
  NUMERICAL        : RK4 with a step of 1 d on a 2.8 au circular orbit is off by 11 m after 4 yr (measured against the exact solution; falls 16-fold per halving of the step, grows
                     linearly in time): ~0.15 km after 50 yr, against a source distance of 4e8 km. Its pull on a planet changes by < 1e-9 of itself.
Golden Rule 2: valid where the test-particle picture holds: the asteroids' mutual pull and the pull of the integrated bodies on them are neglected (the planets are DE440's own, so the
  asteroid path is the one DE440's planets would give it); non-gravitational forces and relativity on the asteroid are ignored.
PHYSICS_GAP: asteroids not in the element catalogue (24 of the 343 in the local file, 4 % of the mass) are absent until their elements are added; the 30 Kuiper-belt objects are absent
  until their elements are added; the ring's phases are equally spaced from an arbitrary origin.
"""
from __future__ import annotations

import math

import numpy as np

from sigma_ground.field.interface.asteroid_perturbers import AU_KM, DAY_S, Perturber, kepler_state_ecliptic
from sigma_ground.field.interface.fast_ephemeris import _ECL_TO_EQ
from sigma_ground.field.interface.prescribed_sources import PrescribedSources

# kernel segments for the Sun and the nine planet-system barycentres (Mercury, Venus, Earth-Moon, Mars, Jupiter ... Pluto), SSB-centred
PLANET_SEGMENTS = [(0, 10), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)]
KM_TO_M = 1.0e3


def planet_table(kernel, jds: np.ndarray) -> np.ndarray:
    """(len(jds), 10, 3) barycentric positions (km) of the Sun and the nine planet-system barycentres."""
    out = np.empty((len(jds), len(PLANET_SEGMENTS), 3))
    for k, (c, t) in enumerate(PLANET_SEGMENTS):
        out[:, k, :] = np.asarray(kernel[c, t].compute(jds), dtype=float).T
    return out


def _acc(r: np.ndarray, planets: np.ndarray, gm_day2: np.ndarray) -> np.ndarray:
    d = r[:, None, :] - planets[None, :, :]                                 # (N, 10, 3) km
    r2 = np.einsum("nki,nki->nk", d, d)
    return -np.einsum("nk,nki->ni", gm_day2[None, :] / (r2 * np.sqrt(r2)), d)       # km / day^2


def rk4_segment(y0: np.ndarray, t0: float, t1: float, n_steps: int, kernel, gm_planets_km3_s2: np.ndarray, record: bool = False):
    """Integrate N test particles (state km, km/day; rows) from JD t0 to t1 in n_steps RK4 steps. Returns the final state, and with record=True also the (n_steps+1, N, 6) node states."""
    gm_day2 = np.asarray(gm_planets_km3_s2, dtype=float) * DAY_S ** 2
    ts = t0 + (t1 - t0) * np.arange(2 * n_steps + 1) / (2.0 * n_steps)
    P = planet_table(kernel, ts)
    h = (t1 - t0) / n_steps
    y = np.array(y0, dtype=float)
    nodes = [y.copy()] if record else None

    def f(yy, Pk):
        return np.concatenate([yy[:, 3:], _acc(yy[:, :3], Pk, gm_day2)], axis=1)

    for i in range(n_steps):
        k1 = f(y, P[2 * i])
        k2 = f(y + 0.5 * h * k1, P[2 * i + 1])
        k3 = f(y + 0.5 * h * k2, P[2 * i + 1])
        k4 = f(y + h * k3, P[2 * i + 2])
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if record:
            nodes.append(y.copy())
    return (y, np.array(nodes)) if record else y


def barycentric_state_at_epoch(p: Perturber, kernel, gm_sun_km3_s2: float) -> tuple[float, np.ndarray]:
    """(element epoch JD, barycentric equatorial state (km, km/day)) of one asteroid: its Kepler state placed on the Sun's barycentric position and velocity."""
    r_ecl, v_ecl = kepler_state_ecliptic(p, gm_sun_km3_s2)
    r_eq, v_eq = _ECL_TO_EQ @ r_ecl, _ECL_TO_EQ @ v_ecl                    # km, km/s heliocentric
    sp, sv = kernel[0, 10].compute_and_differentiate(p.epoch_jd)
    return p.epoch_jd, np.concatenate([r_eq + np.asarray(sp, dtype=float), v_eq * DAY_S + np.asarray(sv, dtype=float)])


def asteroid_window_paths(perturbers: list[Perturber], kernel, gm_planets_km3_s2: np.ndarray, window_jd: float, horizon_days: float, node_days: float = 1.0):
    """Node states (T, N, 6) of the asteroids, km and km/day, at window_jd + j * node_days for j = 0 .. horizon/node_days.

    Each asteroid is first carried from its element epoch to window_jd (steps of at most one day, grouped by element epoch), then recorded forward in steps of node_days.
    """
    n_nodes = int(math.ceil(horizon_days / node_days - 1e-9)) + 1               # the table must cover the whole horizon, with one node of margin
    y_start = np.empty((len(perturbers), 6))
    by_epoch: dict[float, list[int]] = {}
    gm_sun = float(gm_planets_km3_s2[0])
    states = []
    for i, p in enumerate(perturbers):
        ep, y = barycentric_state_at_epoch(p, kernel, gm_sun)
        states.append(y)
        by_epoch.setdefault(ep, []).append(i)
    for ep, idx in by_epoch.items():
        y0 = np.array([states[i] for i in idx])
        if abs(ep - window_jd) < 1e-9:
            y_start[idx] = y0
        else:
            n = max(1, int(math.ceil(abs(window_jd - ep) / 1.0)))
            y_start[idx] = rk4_segment(y0, ep, window_jd, n, kernel, gm_planets_km3_s2)
    _, nodes = rk4_segment(y_start, window_jd, window_jd + n_nodes * node_days, n_nodes, kernel, gm_planets_km3_s2, record=True)
    return nodes


def ring_window_paths(kernel, gm_sun_km3_s2: float, window_jd: float, horizon_days: float, node_days: float = 1.0, n_points: int = 36, a_au: float = 44.0):
    """Node states (T, n_points, 6), km and km/day, of the Kuiper-belt ring: equally spaced points on a heliocentric circular orbit in the ecliptic plane, carried with the Sun."""
    n_nodes = int(math.ceil(horizon_days / node_days - 1e-9)) + 1               # the table must cover the whole horizon, with one node of margin
    jds = window_jd + node_days * np.arange(n_nodes + 1)
    a = a_au * AU_KM
    n_rad = math.sqrt(gm_sun_km3_s2 / a ** 3)                               # rad / s
    phase0 = 2.0 * math.pi * np.arange(n_points) / n_points
    sp, sv = kernel[0, 10].compute_and_differentiate(jds)                    # (3, T) km and km/day
    sun_p, sv = np.asarray(sp, dtype=float).T, np.asarray(sv, dtype=float).T
    ang = phase0[None, :] + n_rad * (jds - window_jd)[:, None] * DAY_S         # (T, P)
    r_ecl = np.stack([a * np.cos(ang), a * np.sin(ang), np.zeros_like(ang)], axis=-1)
    v_ecl = np.stack([-a * n_rad * np.sin(ang), a * n_rad * np.cos(ang), np.zeros_like(ang)], axis=-1) * DAY_S        # km / day
    r_eq = r_ecl @ _ECL_TO_EQ.T
    v_eq = v_ecl @ _ECL_TO_EQ.T
    return np.concatenate([r_eq + sun_p[:, None, :], v_eq + sv[:, None, :]], axis=-1)


def to_prescribed_sources(nodes_km_day: np.ndarray, gm_km3_s2: np.ndarray, node_days: float = 1.0) -> PrescribedSources:
    """Wrap node states (T, N, 6) in km and km/day as engine sources in SI units, time zero at the first node."""
    pos = nodes_km_day[:, :, :3] * KM_TO_M
    vel = nodes_km_day[:, :, 3:] * KM_TO_M / DAY_S
    return PrescribedSources(np.asarray(gm_km3_s2, dtype=float) * 1.0e9, 0.0, node_days * DAY_S, pos, vel)


def concat_sources(*parts: PrescribedSources) -> PrescribedSources:
    """Merge source tables with identical time axes."""
    p0 = parts[0]
    for q in parts[1:]:
        if q.h != p0.h or q.t0 != p0.t0 or q.n_nodes != p0.n_nodes:
            raise ValueError("concat_sources: time axes differ")
    return PrescribedSources(np.concatenate([q.gm for q in parts]), p0.t0, p0.h, np.concatenate([q.pos for q in parts], axis=1), np.concatenate([q.vel for q in parts], axis=1))
