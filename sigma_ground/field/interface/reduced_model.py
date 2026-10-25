"""The reduced solar-system model as a library: Sun, Mercury, Venus, Earth, Moon, Mars, and the giants and Pluto as SYSTEM barycentres, with the shoot-out's force layers.

This is the model scripts/fair_jpl_contest.py (run_one) has used since 2026-10; it lives here so that Mentat and the forecast builder import it instead of copying a script. It was validated against the
full 33-body shoot-out (inner planets and the Moon agree to three digits), and tests/test_reduced_model.py checks that this module reproduces run_one's numbers for the same start and layers.

Layers come from a named predictor of rolling_shootout (e.g. ``restated_zon``): asteroids, Kuiper-belt ring and objects as prescribed sources, the Moon's C22 figure, measured zonal harmonics. The start state is
DE440's own state (a kernel) at the start epoch: this is a propagation from JPL's state, not an independent measurement of the sky, and every answer built on it says so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KM = 1e3
DAY = 86400.0
CHAINS = {"Sun": ((0, 10),), "Mercury": ((0, 1), (1, 199)), "Venus": ((0, 2), (2, 299)), "Earth": ((0, 3), (3, 399)), "Moon": ((0, 3), (3, 301)),
          "Mars": ((0, 4),), "Jupiter": ((0, 5),), "Saturn": ((0, 6),), "Uranus": ((0, 7),), "Neptune": ((0, 8),), "Pluto": ((0, 9),)}
ORDER = list(CHAINS)
SYSTEMS = ("Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")            # barycentres: planet + its moons
_FIXTURE = Path(__file__).parent / "fixtures" / "de440_state_vectors_v2physics.json"


def kernel_state(kernel, name: str, jd: float) -> tuple[np.ndarray, np.ndarray]:
    """SSB-frame ICRF position (km) and velocity (km/s) of a reduced-model body in a JPL kernel at TDB Julian date jd."""
    p, v = 0.0, 0.0
    for c, t in CHAINS[name]:
        pp, vv = kernel[c, t].compute_and_differentiate(jd)
        p, v = p + pp, v + vv
    return np.asarray(p, dtype=float), np.asarray(v, dtype=float) / DAY


def reduced_masses(snapshot: dict) -> dict[str, float]:
    """GM (m^3/s^2) per reduced body from a fixture snapshot, the giants and Pluto summed over the moons that are members of their system (gravitational_anchor)."""
    gm = {b["name"]: b["gm_km3_s2"] * 1e9 for b in snapshot["bodies"]}
    anchor = {b["name"]: b.get("gravitational_anchor") for b in snapshot["bodies"]}
    out = {}
    for n in ORDER:
        out[n] = gm[n] + sum(g for m, g in gm.items() if anchor.get(m) == n) if n in SYSTEMS else gm[n]
    return out


def build_system(kernel, jd: float, predictor_name: str = "restated_zon", dt_days: float = 0.25, horizon_days: float = 365.25, fixture_key: str = "j2010",
                 fixture_file: Path | None = None):
    """(NBodySystem, ORDER) started from the kernel's state at jd with the layers of `predictor_name`.

    fixture_key only selects the snapshot the masses and body parameters (radii, zonals, Love numbers) are read from; they are not part of the initial state. horizon_days sizes the perturber paths.
    """
    import dataclasses
    from sigma_ground.field.constants import G
    from sigma_ground.field.interface import rolling_shootout as rs
    from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem
    de = json.loads((fixture_file or _FIXTURE).read_text())
    pred = next(p for p in rs.PREDICTORS if p.name == predictor_name)
    pred = dataclasses.replace(pred, dt_days=dt_days)
    gm = reduced_masses(de["snapshots"][fixture_key])
    full, names, _, _ = rs._build_bodies_at_snapshot(fixture_key, de, pred)
    byname = dict(zip(names, full))
    bodies = []
    for n in ORDER:
        p, v = kernel_state(kernel, n, jd)
        base = byname[n]
        if n in SYSTEMS:                                       # a barycentre carries no oblateness or tides of its own
            bodies.append(CelestialBody(mass_kg=gm[n] / G, position_m=p * KM, velocity_m_s=v * KM, radius_m=base.radius_m, love_number_k2=0.0, area_m2=base.area_m2,
                                        reflectivity=base.reflectivity, j2=0.0, j3=0.0, j4=0.0, pole_axis_unit=base.pole_axis_unit))
        else:
            bodies.append(CelestialBody(mass_kg=gm[n] / G, position_m=p * KM, velocity_m_s=v * KM, radius_m=base.radius_m, love_number_k2=base.love_number_k2, area_m2=base.area_m2,
                                        reflectivity=base.reflectivity, j2=base.j2, j3=base.j3, j4=base.j4, pole_axis_unit=base.pole_axis_unit))
    if pred.lunar_c22 or pred.perturber_sets:
        bodies, toggles, kw = rs._candidate_layers(pred, bodies, list(ORDER), None, jd, horizon_days)
    else:
        toggles, kw = pred.toggles, {}
    system = NBodySystem(bodies, toggles=toggles, solar_luminosity_W=rs._L_SUN_W if toggles.srp else 0.0, parent_attractor_indices=None, **kw)
    return system, list(ORDER)


def propagate_dense(system, jd_start: float, jd_end: float, dt_days: float, every: int = 1) -> dict:
    """Integrate with fixed-step Forest-Ruth from jd_start to jd_end and store the state every `every` steps. Returns {jd, pos_km (T,N,3), vel_km_s (T,N,3)}; row 0 is the start."""
    n_steps = max(1, int(round((jd_end - jd_start) / dt_days)))
    dts = (jd_end - jd_start) * DAY / n_steps
    jd = np.empty(n_steps // every + 2)
    pos = np.empty((len(jd), len(system.bodies), 3))
    vel = np.empty_like(pos)

    def store(k: int, t: float) -> None:
        jd[k] = t
        pos[k] = [b.position_m / KM for b in system.bodies]
        vel[k] = [b.velocity_m_s / KM for b in system.bodies]

    k = 0
    store(k, jd_start)
    for s in range(1, n_steps + 1):
        system.forest_ruth_step(dts)
        if s % every == 0 or s == n_steps:
            k += 1
            store(k, jd_start + s * dts / DAY)
    return {"jd": jd[:k + 1], "pos_km": pos[:k + 1], "vel_km_s": vel[:k + 1]}


def hermite(t: float, jd: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation of every body's position (km) and velocity (km/s) at jd value t from a dense table (jd ascending)."""
    i = int(np.searchsorted(jd, t, side="right") - 1)
    i = min(max(i, 0), len(jd) - 2)
    h = (jd[i + 1] - jd[i]) * DAY
    s = (t - jd[i]) * DAY / h
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    p = h00 * pos[i] + h10 * h * vel[i] + h01 * pos[i + 1] + h11 * h * vel[i + 1]
    d00, d10, d01, d11 = (6 * s ** 2 - 6 * s) / h, (3 * s ** 2 - 4 * s + 1), (-6 * s ** 2 + 6 * s) / h, (3 * s ** 2 - 2 * s)
    v = d00 * pos[i] + d10 * vel[i] + d01 * pos[i + 1] + d11 * vel[i + 1]
    return p, v
