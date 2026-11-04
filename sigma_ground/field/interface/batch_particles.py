"""Many test particles at once: the whole-sky engine (misc/CANDIDATE_LEDGER.md C35).

The served small-body engine (``comet_orbits.propagate``) carries ONE body through an adaptive Dormand-Prince integration with ten kernel lookups per force evaluation; it costs
20-30 s a body and the sky has 1.4 million. This module carries N bodies in one fixed-step fourth-order Runge-Kutta pass over an (N, 6) array, the planets looked up once per
segment at the stage times, with the same physics as the served engine: Newtonian Sun + nine planet-system barycentres, the Sun's 1PN test-particle term, the prescribed asteroid
sources (a body that is itself a source does not pull on itself), and the Marsden/Yarkovsky non-gravitational term for the rows that carry one.

Derivation chain (Golden Rule 1):
  MEASURED / catalogued : the kernel positions of the Sun and the planet-system barycentres and their GM; the prescribed-source table; SBDB A1, A2, A3.
  FIRST_PRINCIPLES      : the same accelerations as comet_orbits.py, written for arrays: a_i = -sum_k GM_k (r_i - r_k)/|r_i - r_k|^3 ; the isotropic-PPN 1PN term
                          GM/(c^2 r^3)[(4GM/r - v^2) r + 4 (r.v) v] with heliocentric r, v ; the source pull GM_s (r_s - r_i)/|r_s - r_i|^3 ; Marsden a = g(r)(A1 R + A2 T + A3 N).
  NUMERICAL             : classic RK4, fixed step h (a day for the main belt, a quarter day for near-Earth orbits), the planets at the 2n+1 stage times from the kernel.
                          perturber_paths.py measured RK4 at h = 1 d on a 2.8 au circular orbit: 11 m after 4 yr, falling 16-fold per halving of h. The whole-sky acceptance
                          (S-A, pre-registered 2026-11-04) is agreement with the served dopri5 engine within 0.1 km over a year on the tracked bodies; the test measures it.
  Positions between nodes come from cubic Hermite interpolation of the recorded nodes (error h^4/384 max|x''''|: below a metre for a main-belt body at h = 1 d), so the light-time
  iteration for an observation re-interpolates instead of re-integrating.
Golden Rule 2: massless bodies outside the planets' spheres of influence; a body inside 0.02 au of the Earth belongs to the adaptive per-body path (a fixed step is worst at a
  close approach). No mutual pull between the integrated bodies. The Marsden time delay DT is not carried here (zero for every asteroid; comets with DT stay on the per-body path).
PHYSICS_GAP: the planets' own relativistic terms on the particle (~1e-9 of the Sun's); planetary oblateness; the Moon as a separate body (the Earth-Moon barycentre is used).
"""
from __future__ import annotations

import math

import numpy as np

from sigma_ground.field.interface.comet_orbits import AU_KM, C_KM_S, DAY_S, _law_kwargs, marsden_g
from sigma_ground.field.interface.perturber_paths import PLANET_SEGMENTS, planet_table

SOURCE_CHUNK = 4096                                                      # bodies per chunk of the (N, n_sources, 3) source sum (memory: 4096 x 343 x 3 x 8 B = 34 MB)


def _sun_table(kernel, jds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 3) barycentric position (km) and velocity (km/day) of the Sun at the times."""
    p, v = kernel[0, 10].compute_and_differentiate(jds)
    return np.asarray(p, dtype=float).T, np.asarray(v, dtype=float).T


def newtonian(r: np.ndarray, planets: np.ndarray, gm_day2: np.ndarray) -> np.ndarray:
    """(N, 3) km/day^2 from the Sun and planet-system barycentres at one time: planets (10, 3) km."""
    d = r[:, None, :] - planets[None, :, :]
    r2 = np.einsum("nki,nki->nk", d, d)
    return -np.einsum("nk,nki->ni", gm_day2[None, :] / (r2 * np.sqrt(r2)), d)


def sun_1pn(r_h: np.ndarray, v_h_day: np.ndarray, gm_sun_km3_s2: float) -> np.ndarray:
    """(N, 3) km/day^2: the isotropic-PPN 1PN term of the Sun for heliocentric r (km) and v (km/day); the array form of comet_orbits.sun_1pn_acceleration."""
    v = v_h_day / DAY_S
    r = np.sqrt(np.einsum("ni,ni->n", r_h, r_h))
    v2 = np.einsum("ni,ni->n", v, v)
    rv = np.einsum("ni,ni->n", r_h, v)
    gm = gm_sun_km3_s2
    a = (gm / (C_KM_S ** 2 * r ** 3))[:, None] * ((4.0 * gm / r - v2)[:, None] * r_h + 4.0 * rv[:, None] * v)
    return a * DAY_S ** 2


try:                                                                     # the (N x 343) source sum is the whole cost of a whole-sky night (S-B measured 2,523 body-years/min in numpy,
    import numba                                                         # 76,000 without sources, 2026-11-04): a compiled parallel loop; the numpy path below is the fallback and the reference

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _source_pull_nb(r, src, gm, self_index, out):
        n_b, n_s = r.shape[0], src.shape[0]
        for i in numba.prange(n_b):
            ax = ay = az = 0.0
            si = self_index[i]
            for s in range(n_s):
                if s == si:
                    continue
                dx = src[s, 0] - r[i, 0]; dy = src[s, 1] - r[i, 1]; dz = src[s, 2] - r[i, 2]
                r2 = dx * dx + dy * dy + dz * dz
                w = gm[s] / (r2 * math.sqrt(r2))
                ax += w * dx; ay += w * dy; az += w * dz
            out[i, 0] = ax; out[i, 1] = ay; out[i, 2] = az
    HAVE_NUMBA = True
except Exception:                                                        # noqa: BLE001
    HAVE_NUMBA = False


def source_pull(r: np.ndarray, sources, t_s: float, self_index: np.ndarray | None) -> np.ndarray:
    """(N, 3) km/day^2 from the prescribed sources at one time; self_index[i] is body i's own column in the table (or -1), whose pull is excluded."""
    src = sources.positions_at(t_s) * 1e-3                                # (n, 3) km
    gm = sources.gm * 1e-9 * DAY_S ** 2                                    # km^3/day^2
    out = np.empty_like(r)
    if HAVE_NUMBA:
        si = np.full(len(r), -1, dtype=np.int64) if self_index is None else np.asarray(self_index, dtype=np.int64)
        _source_pull_nb(np.ascontiguousarray(r), np.ascontiguousarray(src), np.ascontiguousarray(gm), si, out)
        return out
    for a in range(0, len(r), SOURCE_CHUNK):
        b = min(a + SOURCE_CHUNK, len(r))
        d = src[None, :, :] - r[a:b, None, :]
        r2 = np.einsum("kni,kni->kn", d, d)
        w = gm[None, :] / (r2 * np.sqrt(r2))
        if self_index is not None:
            rows = np.nonzero(self_index[a:b] >= 0)[0]
            w[rows, self_index[a:b][rows]] = 0.0
        out[a:b] = np.einsum("kn,kni->ki", w, d)
    return out


def marsden_batch(r_h: np.ndarray, v_h_day: np.ndarray, A_au_day2: np.ndarray, law: dict | None) -> np.ndarray:
    """(N, 3) km/day^2 Marsden acceleration for A (N, 3) in au/day^2; rows with A = 0 cost nothing."""
    out = np.zeros_like(r_h)
    rows = np.nonzero(np.any(A_au_day2 != 0.0, axis=1))[0]
    if len(rows) == 0:
        return out
    r = r_h[rows]; v = v_h_day[rows]; A = A_au_day2[rows]
    rn = np.sqrt(np.einsum("ni,ni->n", r, r))
    R = r / rn[:, None]
    h = np.cross(r, v)
    N = h / np.sqrt(np.einsum("ni,ni->n", h, h))[:, None]
    T = np.cross(N, R)
    kw = _law_kwargs(law)
    g = np.array([marsden_g(x, **kw) for x in rn / AU_KM]) if len(rows) < 64 else _marsden_g_array(rn / AU_KM, **kw)
    out[rows] = (g * AU_KM)[:, None] * (A[:, 0:1] * R + A[:, 1:2] * T + A[:, 2:3] * N)
    return out


def _marsden_g_array(r_au: np.ndarray, alpha: float, r0: float, m: float, n: float, k: float) -> np.ndarray:
    x = r_au / r0
    return alpha * x ** (-m) * (1.0 + x ** n) ** (-k)


def propagate_batch(states: np.ndarray, jd0: float, jd_end: float, kernel, gm_planets_km3_s2, *, sources=None, sources_epoch_jd: float | None = None,
                    self_index: np.ndarray | None = None, gr: bool = True, step_days: float = 1.0, nongrav: np.ndarray | None = None, law: dict | None = None,
                    node_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Carry N test particles from jd0 to jd_end (either direction) in RK4 steps of at most step_days.

    states: (N, 6) barycentric ICRF km and km/day. Returns (t_nodes (T,), nodes (T, N, 6)) with a node every `node_every` steps, the first at jd0 and the last at jd_end.
    sources / sources_epoch_jd / self_index: the prescribed-source table (SI, time zero at sources_epoch_jd) and each body's own column in it (-1 when it is not a source).
    nongrav: (N, 3) A1, A2, A3 in au/day^2 (zeros for bodies without a term), law: the g(r) constants shared by the rows that carry one (the Yarkovsky law for asteroids).
    """
    y = np.array(states, dtype=float)
    if y.ndim != 2 or y.shape[1] != 6:
        raise ValueError("propagate_batch: states must be (N, 6)")
    span = jd_end - jd0
    n = max(1, int(math.ceil(abs(span) / step_days - 1e-9)))
    h = span / n
    ts = jd0 + h * np.arange(2 * n + 1) / 2.0
    P = planet_table(kernel, ts)                                         # (2n+1, 10, 3) km
    S_p, S_v = _sun_table(kernel, ts)                                      # (2n+1, 3)
    gm_day2 = np.asarray(gm_planets_km3_s2, dtype=float) * DAY_S ** 2
    gm_sun = float(gm_planets_km3_s2[0])
    A = None if nongrav is None or not np.any(nongrav) else np.asarray(nongrav, dtype=float)
    t_nodes = [jd0]; nodes = [y.copy()]

    def f(yy, k):
        r, v = yy[:, :3], yy[:, 3:]
        acc = newtonian(r, P[k], gm_day2)
        rh, vh = r - S_p[k], v - S_v[k]
        if gr:
            acc += sun_1pn(rh, vh, gm_sun)
        if sources is not None:
            acc += source_pull(r, sources, (ts[k] - sources_epoch_jd) * DAY_S, self_index)
        if A is not None:
            acc += marsden_batch(rh, vh, A, law)
        return np.concatenate([v, acc], axis=1)

    for i in range(n):
        k1 = f(y, 2 * i)
        k2 = f(y + 0.5 * h * k1, 2 * i + 1)
        k3 = f(y + 0.5 * h * k2, 2 * i + 1)
        k4 = f(y + h * k3, 2 * i + 2)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) % node_every == 0 or i == n - 1:
            t_nodes.append(jd0 + h * (i + 1)); nodes.append(y.copy())
    return np.array(t_nodes), np.array(nodes)


def positions_at(t_nodes: np.ndarray, nodes: np.ndarray, jd: np.ndarray, body: np.ndarray, velocity: bool = False) -> np.ndarray:
    """(M, 3) positions (km) [or (M, 6) states] of body[m] at jd[m] by cubic Hermite between the bracketing nodes; t_nodes ascending or descending."""
    t = np.asarray(t_nodes, dtype=float); jd = np.asarray(jd, dtype=float); body = np.asarray(body)
    sign = 1.0 if t[-1] >= t[0] else -1.0
    tt = sign * t
    i = np.clip(np.searchsorted(tt, sign * jd, side="right") - 1, 0, len(t) - 2)
    hseg = (t[i + 1] - t[i])
    s = ((jd - t[i]) / hseg)[:, None]
    p0 = nodes[i, body, :3]; p1 = nodes[i + 1, body, :3]
    m0 = nodes[i, body, 3:] * hseg[:, None]; m1 = nodes[i + 1, body, 3:] * hseg[:, None]
    s2, s3 = s * s, s * s * s
    pos = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1
    if not velocity:
        return pos
    dpos_ds = (6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * m0 + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * m1
    return np.concatenate([pos, dpos_ds / hseg[:, None]], axis=1)


def light_time_positions(t_nodes: np.ndarray, nodes: np.ndarray, jd_tdb: np.ndarray, body: np.ndarray, x_obs_km: np.ndarray, iters: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(M, 3) body positions at the light-time-corrected emission times for observations at jd_tdb from observers at x_obs_km, and the light times (s); interpolation only."""
    tau = np.zeros(len(jd_tdb))
    for _ in range(iters):
        pos = positions_at(t_nodes, nodes, jd_tdb - tau / DAY_S, body)
        tau = np.sqrt(np.einsum("ni,ni->n", pos - x_obs_km, pos - x_obs_km)) / C_KM_S
    return positions_at(t_nodes, nodes, jd_tdb - tau / DAY_S, body), tau
