"""Small numeric helpers for grading engine experiments. Pure Python, no numpy, written independently of the Lua."""
from __future__ import annotations

import math


def solve3(A, b):
    """Solve a 3x3 linear system by Gaussian elimination with partial pivoting."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    n = len(M)
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("singular system")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def quad_fit(ts, ys):
    """Least-squares y = c0 + c1 t + c2 t^2. Returns (c0, c1, c2, max_abs_residual)."""
    S = [sum(t ** k for t in ts) for k in range(5)]
    A = [[S[i + j] for j in range(3)] for i in range(3)]
    b = [sum(y * t ** i for t, y in zip(ts, ys)) for i in range(3)]
    c = solve3(A, b)
    res = max(abs(y - (c[0] + c[1] * t + c[2] * t * t)) for t, y in zip(ts, ys))
    return c[0], c[1], c[2], res


def quad_through(ts, ys):
    """Exact quadratic through three points (c0, c1, c2)."""
    return solve3([[1, t, t * t] for t in ts], list(ys))[:3]


def poly_roots_in(c, level, lo, hi):
    """Roots of c0 + c1 t + c2 t^2 = level inside [lo, hi]."""
    a, b, d = c[2], c[1], c[0] - level
    if abs(a) < 1e-14:
        return [(-d / b)] if abs(b) > 1e-14 and lo <= -d / b <= hi else []
    disc = b * b - 4 * a * d
    if disc < 0:
        return []
    s = math.sqrt(disc)
    return sorted(r for r in ((-b - s) / (2 * a), (-b + s) / (2 * a)) if lo - 1e-12 <= r <= hi + 1e-12)


def crossing(ts, ys, level, direction=-1):
    """Time at which y first crosses `level` going in `direction` (-1 falling, +1 rising), refined with the exact
    quadratic through the three nearest samples (the engine's samples lie on a parabola, so this is exact)."""
    for i in range(1, len(ts)):
        a, b = ys[i - 1] - level, ys[i] - level
        if (direction < 0 and a > 0 >= b) or (direction > 0 and a < 0 <= b):
            j0 = max(0, i - 2)
            j1 = min(len(ts), j0 + 3)
            j0 = max(0, j1 - 3)
            c = quad_through(ts[j0:j1], ys[j0:j1])
            roots = poly_roots_in(c, level, ts[i - 1], ts[i])
            if roots:
                return roots[0]
            return ts[i - 1] + (ts[i] - ts[i - 1]) * a / (a - b)       # linear fallback
    return None


def slope_at(ts, ys, t, half_window=3):
    """Derivative of y at time t from a local quadratic fit over +/- half_window samples."""
    i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
    lo, hi = max(0, i - half_window), min(len(ts), i + half_window + 1)
    c0, c1, c2, _ = quad_fit(ts[lo:hi], ys[lo:hi])
    return c1 + 2 * c2 * t
