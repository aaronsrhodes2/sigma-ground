"""Prescribed massive sources: bodies whose motion is given in advance and which pull on the integrated bodies (the asteroid perturbers).

Why: the DE440 model carries 343 asteroids. As full N-body members they would multiply the cost of every force evaluation (the 1PN cross terms are O(N^2) and run several times per
step); as prescribed sources they cost one (n_bodies x n_sources) Newtonian sum per evaluation. The sources are NOT integrated here: the caller supplies each source's position and
velocity at uniform nodes and the pull is evaluated at the sub-step time with cubic Hermite interpolation.

Derivation chain (Golden Rule 1):
  FIRST_PRINCIPLES: the Newtonian acceleration of body i from source s is  a_i = GM_s (r_s - r_i) / |r_s - r_i|^3  (no reaction is applied: the source's own motion
                    already contains its response to the planets, which is how the caller obtained it).
  NUMERICAL       : cubic Hermite interpolation between nodes h apart has position error (h^4 / 384) max|x''''| ; for a main-belt asteroid (period >= 3.5 yr) with h = 1 d that
                    is below 1 km, a relative error of 1e-8 of the source distance, which changes the pull on a planet by less than 1e-8 of a pull that is itself ~1e-9 of the Sun's.
Golden Rule 2: Newtonian only. The 1PN corrections to the pull of a body that carries 1e-10 of the Sun's mass are of order 1e-18 of the Sun's pull and are dropped.
PHYSICS_GAP: no source-source interaction, no pull from the integrated bodies is fed back to the sources (the caller's pre-pass decides how the sources moved).
"""
from __future__ import annotations

import numpy as np


class PrescribedSources:
    """Massive point sources with tabulated motion.

    gm_m3_s2   : (n,) gravitational parameter of each source, m^3/s^2
    t0_s, h_s  : time of the first node (s since the system epoch) and the uniform node spacing (s)
    pos_m      : (T, n, 3) positions at the nodes, m
    vel_m_s    : (T, n, 3) velocities at the nodes, m/s
    """

    def __init__(self, gm_m3_s2: np.ndarray, t0_s: float, h_s: float, pos_m: np.ndarray, vel_m_s: np.ndarray) -> None:
        self.gm = np.asarray(gm_m3_s2, dtype=float)
        self.t0 = float(t0_s)
        self.h = float(h_s)
        self.pos = np.asarray(pos_m, dtype=float)
        self.vel = np.asarray(vel_m_s, dtype=float)
        if self.pos.ndim != 3 or self.pos.shape[2] != 3 or self.pos.shape != self.vel.shape or self.pos.shape[1] != self.gm.shape[0] or self.h <= 0.0:
            raise ValueError("PrescribedSources: inconsistent shapes or non-positive spacing")
        self.n_nodes = self.pos.shape[0]

    @property
    def t_end(self) -> float:
        return self.t0 + (self.n_nodes - 1) * self.h

    def positions_at(self, t_s: float) -> np.ndarray:
        """(n, 3) source positions at time t (cubic Hermite between the bracketing nodes). A time outside the table is an error, not an extrapolation."""
        x = (t_s - self.t0) / self.h
        if x < -1e-9 or x > self.n_nodes - 1 + 1e-9:
            raise ValueError(f"PrescribedSources: time {t_s} s is outside the table [{self.t0}, {self.t_end}] s")
        k = min(max(int(np.floor(x)), 0), self.n_nodes - 2)
        s = x - k
        p0, p1 = self.pos[k], self.pos[k + 1]
        m0, m1 = self.vel[k] * self.h, self.vel[k + 1] * self.h
        s2, s3 = s * s, s * s * s
        return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1

    def accelerations_on(self, points_m: np.ndarray, t_s: float) -> np.ndarray:
        """(k, 3) Newtonian acceleration of k points at time t from all sources."""
        src = self.positions_at(t_s)                                      # (n, 3)
        d = src[None, :, :] - points_m[:, None, :]                         # (k, n, 3)
        r2 = np.einsum("kni,kni->kn", d, d)
        w = self.gm[None, :] / (r2 * np.sqrt(r2))
        return np.einsum("kn,kni->ki", w, d)
