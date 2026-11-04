"""The radial / along-track / cross-track frame of an orbit: the split that tells a timing error (along) from a plane error (cross) from a size error (radial).

Extracted 2026-11-04 from the inline block in scripts/fair_jpl_contest.py (the Moon's gap about the Earth, C27): R = r/|r|, N = (r x v)/|r x v|, T = N x R, with r, v the body's
position and velocity about its centre. Pure geometry; the only numerical point is a degenerate (r x v = 0) orbit, which raises.
"""
from __future__ import annotations

import numpy as np


def rtn_frame(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, T, N) unit vectors for r, v (3,) or (N, 3)."""
    r = np.asarray(r, dtype=float); v = np.asarray(v, dtype=float)
    one = r.ndim == 1
    r2 = np.atleast_2d(r); v2 = np.atleast_2d(v)
    rh = r2 / np.linalg.norm(r2, axis=1)[:, None]
    h = np.cross(r2, v2); hn = np.linalg.norm(h, axis=1)
    if np.any(hn == 0.0):
        raise ValueError("rtn_frame: r and v are parallel (no orbit plane)")
    nh = h / hn[:, None]
    th = np.cross(nh, rh)
    return (rh[0], th[0], nh[0]) if one else (rh, th, nh)


def to_rtn(d: np.ndarray, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The vector(s) d expressed in the RTN frame of (r, v): (3,) or (N, 3) of (radial, along-track, cross-track)."""
    rh, th, nh = rtn_frame(r, v)
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        return np.array([d @ rh, d @ th, d @ nh])
    return np.stack([np.einsum("ni,ni->n", d, rh), np.einsum("ni,ni->n", d, th), np.einsum("ni,ni->n", d, nh)], axis=1)
