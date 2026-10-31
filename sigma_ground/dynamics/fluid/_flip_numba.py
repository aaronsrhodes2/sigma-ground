"""Compiled particle↔grid transfers for ``flip.Flip`` (used when numba is installed; flip.py has the NumPy fallback).

Same trilinear stencil as the fallback: the 8 faces of one velocity component around a position, past the last face the
edge face. ``off`` is that component's face offset in cells.
"""
import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def p2g(pos, vel, h, off, sx, sy, sz):
    """Each particle's velocity component, spread to the faces with trilinear weights and normalised by them."""
    num = np.zeros((sx, sy, sz))
    den = np.zeros((sx, sy, sz))
    for p in range(pos.shape[0]):
        gx, gy, gz = pos[p, 0] / h - off[0], pos[p, 1] / h - off[1], pos[p, 2] / h - off[2]
        x0, y0, z0 = math.floor(gx), math.floor(gy), math.floor(gz)
        fx, fy, fz = gx - x0, gy - y0, gz - z0
        for dx in range(2):
            i = min(max(x0 + dx, 0), sx - 1)
            wx = fx if dx else 1.0 - fx
            for dy in range(2):
                j = min(max(y0 + dy, 0), sy - 1)
                wy = fy if dy else 1.0 - fy
                for dz in range(2):
                    k = min(max(z0 + dz, 0), sz - 1)
                    w = wx * wy * (fz if dz else 1.0 - fz)
                    num[i, j, k] += w * vel[p]
                    den[i, j, k] += w
    out = np.zeros((sx, sy, sz))
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                if den[i, j, k] > 0:
                    out[i, j, k] = num[i, j, k] / den[i, j, k]
    return out


@njit(cache=True, parallel=True)
def g2p(pos, u, h, off):
    """The face field ``u`` sampled at each position."""
    sx, sy, sz = u.shape
    out = np.empty(pos.shape[0])
    for p in prange(pos.shape[0]):
        gx, gy, gz = pos[p, 0] / h - off[0], pos[p, 1] / h - off[1], pos[p, 2] / h - off[2]
        x0, y0, z0 = math.floor(gx), math.floor(gy), math.floor(gz)
        fx, fy, fz = gx - x0, gy - y0, gz - z0
        s = 0.0
        for dx in range(2):
            i = min(max(x0 + dx, 0), sx - 1)
            wx = fx if dx else 1.0 - fx
            for dy in range(2):
                j = min(max(y0 + dy, 0), sy - 1)
                wy = fy if dy else 1.0 - fy
                for dz in range(2):
                    k = min(max(z0 + dz, 0), sz - 1)
                    s += wx * wy * (fz if dz else 1.0 - fz) * u[i, j, k]
        out[p] = s
    return out


@njit(cache=True)
def extrapolate(u, valid, layers):
    """Fill ``layers`` rings of faces beyond the valid ones with the mean of their valid neighbours (in place)."""
    sx, sy, sz = u.shape
    known = valid.copy()
    for _ in range(layers):
        grow = np.zeros((sx, sy, sz), np.bool_)
        for i in range(sx):
            for j in range(sy):
                for k in range(sz):
                    if known[i, j, k]:
                        continue
                    s, n = 0.0, 0
                    if i > 0 and known[i - 1, j, k]:
                        s += u[i - 1, j, k]; n += 1
                    if i < sx - 1 and known[i + 1, j, k]:
                        s += u[i + 1, j, k]; n += 1
                    if j > 0 and known[i, j - 1, k]:
                        s += u[i, j - 1, k]; n += 1
                    if j < sy - 1 and known[i, j + 1, k]:
                        s += u[i, j + 1, k]; n += 1
                    if k > 0 and known[i, j, k - 1]:
                        s += u[i, j, k - 1]; n += 1
                    if k < sz - 1 and known[i, j, k + 1]:
                        s += u[i, j, k + 1]; n += 1
                    if n:
                        u[i, j, k] = s / n
                        grow[i, j, k] = True
        for i in range(sx):
            for j in range(sy):
                for k in range(sz):
                    if grow[i, j, k]:
                        known[i, j, k] = True
    return u


@njit(cache=True)
def _apply_A(cell, diag, x, out):
    """out = A·x for the pressure Laplacian: diag on the cell, −1 to each liquid neighbour."""
    sx, sy, sz = x.shape
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                if not cell[i, j, k]:
                    out[i, j, k] = 0.0
                    continue
                s = diag[i, j, k] * x[i, j, k]
                if i > 0 and cell[i - 1, j, k]:
                    s -= x[i - 1, j, k]
                if i < sx - 1 and cell[i + 1, j, k]:
                    s -= x[i + 1, j, k]
                if j > 0 and cell[i, j - 1, k]:
                    s -= x[i, j - 1, k]
                if j < sy - 1 and cell[i, j + 1, k]:
                    s -= x[i, j + 1, k]
                if k > 0 and cell[i, j, k - 1]:
                    s -= x[i, j, k - 1]
                if k < sz - 1 and cell[i, j, k + 1]:
                    s -= x[i, j, k + 1]
                out[i, j, k] = s


@njit(cache=True)
def _mic0(cell, diag):
    """Modified incomplete Cholesky, level 0 (Bridson, Fluid Simulation for Computer Graphics, §5.2): tau 0.97,
    safety 0.25."""
    sx, sy, sz = diag.shape
    pre = np.zeros((sx, sy, sz))
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                if not cell[i, j, k]:
                    continue
                e = diag[i, j, k]
                if i > 0 and cell[i - 1, j, k]:
                    p = pre[i - 1, j, k]
                    a = 1.0 if (j < sy - 1 and cell[i - 1, j + 1, k]) else 0.0
                    b = 1.0 if (k < sz - 1 and cell[i - 1, j, k + 1]) else 0.0
                    e -= p * p + 0.97 * (a + b) * p * p
                if j > 0 and cell[i, j - 1, k]:
                    p = pre[i, j - 1, k]
                    a = 1.0 if (i < sx - 1 and cell[i + 1, j - 1, k]) else 0.0
                    b = 1.0 if (k < sz - 1 and cell[i, j - 1, k + 1]) else 0.0
                    e -= p * p + 0.97 * (a + b) * p * p
                if k > 0 and cell[i, j, k - 1]:
                    p = pre[i, j, k - 1]
                    a = 1.0 if (i < sx - 1 and cell[i + 1, j, k - 1]) else 0.0
                    b = 1.0 if (j < sy - 1 and cell[i, j + 1, k - 1]) else 0.0
                    e -= p * p + 0.97 * (a + b) * p * p
                if e < 0.25 * diag[i, j, k]:
                    e = diag[i, j, k]
                pre[i, j, k] = 1.0 / np.sqrt(e)
    return pre


@njit(cache=True)
def _precondition(cell, pre, r, z):
    sx, sy, sz = r.shape
    q = np.zeros((sx, sy, sz))
    for i in range(sx):                                   # L q = r
        for j in range(sy):
            for k in range(sz):
                if not cell[i, j, k]:
                    continue
                t = r[i, j, k]
                if i > 0 and cell[i - 1, j, k]:
                    t += q[i - 1, j, k] * pre[i - 1, j, k]
                if j > 0 and cell[i, j - 1, k]:
                    t += q[i, j - 1, k] * pre[i, j - 1, k]
                if k > 0 and cell[i, j, k - 1]:
                    t += q[i, j, k - 1] * pre[i, j, k - 1]
                q[i, j, k] = t * pre[i, j, k]
    for i in range(sx - 1, -1, -1):                       # Lᵀ z = q
        for j in range(sy - 1, -1, -1):
            for k in range(sz - 1, -1, -1):
                if not cell[i, j, k]:
                    z[i, j, k] = 0.0
                    continue
                t = q[i, j, k]
                p = pre[i, j, k]
                if i < sx - 1 and cell[i + 1, j, k]:
                    t += z[i + 1, j, k] * p
                if j < sy - 1 and cell[i, j + 1, k]:
                    t += z[i, j + 1, k] * p
                if k < sz - 1 and cell[i, j, k + 1]:
                    t += z[i, j, k + 1] * p
                z[i, j, k] = t * p


@njit(cache=True)
def pcg(cell, diag, b, x, rtol, maxiter):
    """Solve A x = b on the liquid cells (``cell``), MIC(0)-preconditioned conjugate gradients, from ``x``.
    Returns (x, iterations)."""
    pre = _mic0(cell, diag)
    Ax = np.zeros_like(x)
    _apply_A(cell, diag, x, Ax)
    r = np.where(cell, b - Ax, 0.0)
    bnorm = np.sqrt((b * b).sum())
    if bnorm == 0.0:
        return x * 0.0, 0
    z = np.zeros_like(x)
    _precondition(cell, pre, r, z)
    s = z.copy()
    rz = (r * z).sum()
    As = np.zeros_like(x)
    for it in range(maxiter):
        if np.sqrt((r * r).sum()) <= rtol * bnorm:
            return x, it
        _apply_A(cell, diag, s, As)
        alpha = rz / (s * As).sum()
        x += alpha * s
        r -= alpha * As
        _precondition(cell, pre, r, z)
        rz_new = (r * z).sum()
        s = z + (rz_new / rz) * s
        rz = rz_new
    return x, maxiter
