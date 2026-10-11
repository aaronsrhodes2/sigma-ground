"""The one adaptive ODE integrator: Dormand-Prince 5(4) with dense output and terminal events.

Tier 0, pure stdlib (no numpy), so every higher tier can use it. Written for the dense-matter work (Lane-Emden, Tolman-Oppenheimer-
Volkoff): smooth right-hand sides, a regular-singular centre handled by the caller's series start, and a terminal event where a
profile ends (a surface where the pressure reaches zero, the first zero of a Lane-Emden function).

Derivation chain (Golden Rule 1):
  MEASURED / textbook   : the Dormand-Prince coefficients (Dormand & Prince 1980, J. Comput. Appl. Math. 6, 19) and the 4th-order
                          continuous extension (Hairer, Norsett & Wanner, Solving ODEs I, sec. II.6, ``dopri5.f``).
  FIRST_PRINCIPLES      : step-size control from the embedded 4th-order error estimate, h_new = 0.9 h (1/err)^(1/5).
  Tests (kernel/test_ode.py) check every coefficient-dependent claim against closed forms; none uses the code's own output as truth.

PHYSICS_GAP: no stiff solver (an implicit method is a separate item); no vector-valued events; step control is the plain
elementary controller, not PI. ``sg_solve_ivp(method="dopri5")`` delegation is pending decision D2 in misc/NUCLEAR_MATTER_PLAN.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

# ---- Dormand-Prince 5(4) tableau ------------------------------------------------------------------------------------
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_B5 = _A[6]                                                   # FSAL: the 5th-order weights are the last stage row
_E = (71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)   # b5 - b4
# dense-output coefficients (Hairer dopri5.f)
_D1 = -12715105075 / 11282082432
_D3 = 87487479700 / 32700410799
_D4 = -10690763975 / 1880347072
_D5 = 701980252875 / 199316789632
_D6 = -1453857185 / 822651844
_D7 = 69997945 / 29380423


@dataclass
class Event:
    """A scalar function g(t, y); the integration notes (and optionally stops at) its sign changes."""
    g: Callable[[float, Sequence[float]], float]
    terminal: bool = False
    direction: int = 0                                        # +1 rising only, -1 falling only, 0 either


@dataclass
class ODEResult:
    t: list = field(default_factory=list)
    y: list = field(default_factory=list)                     # list of state lists
    events: list = field(default_factory=list)                # (event index, t, y)
    status: str = "running"                                   # "finished" | "terminal_event" | "max_steps" | "step_underflow"
    nfev: int = 0
    _segments: list = field(default_factory=list)             # (t0, h, rcont1..rcont5) per accepted step

    def __call__(self, t: float) -> list:
        """Dense output: the state at any t inside the integrated range (4th-order accurate)."""
        if not self._segments:
            raise ValueError("no steps taken")
        t0 = self._segments[0][0]
        sign = 1.0 if self._segments[-1][0] + self._segments[-1][1] >= t0 else -1.0
        lo, hi = 0, len(self._segments) - 1
        while lo < hi:                                        # binary search for the segment containing t
            mid = (lo + hi) // 2
            s = self._segments[mid]
            if sign * (t - (s[0] + s[1])) > 0:
                lo = mid + 1
            else:
                hi = mid
        return _eval_segment(self._segments[lo], t)


def _eval_segment(seg: tuple, t: float) -> list:
    t0, h, r1, r2, r3, r4, r5 = seg
    th = (t - t0) / h
    th1 = 1.0 - th
    return [r1[i] + th * (r2[i] + th1 * (r3[i] + th * (r4[i] + th1 * r5[i]))) for i in range(len(r1))]


def _brent(f: Callable[[float], float], a: float, b: float, fa: float, fb: float, xtol: float = 1e-15, maxit: int = 100) -> float:
    """Root of f in [a, b] (f(a) and f(b) of opposite sign), Brent's method."""
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxit):
        if (fb > 0) == (fc > 0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2 * 2.220446049250313e-16 * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b
        if abs(e) < tol or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                p, q = 2 * m * s, 1 - s
            else:
                q0, r = fa / fc, fb / fc
                p = s * (2 * m * q0 * (q0 - r) - (b - a) * (r - 1))
                q = (q0 - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
    return b


def _initial_step(f, t0, y0, f0, direction, rtol, atol) -> float:
    n = len(y0)
    sc = [atol + rtol * abs(y0[i]) for i in range(n)]
    d0 = math.sqrt(sum((y0[i] / sc[i]) ** 2 for i in range(n)) / n)
    d1 = math.sqrt(sum((f0[i] / sc[i]) ** 2 for i in range(n)) / n)
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    y1 = [y0[i] + direction * h0 * f0[i] for i in range(n)]
    f1 = f(t0 + direction * h0, y1)
    d2 = math.sqrt(sum(((f1[i] - f0[i]) / sc[i]) ** 2 for i in range(n)) / n) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** 0.2
    return min(100 * h0, h1)


def dopri5(f: Callable[[float, Sequence[float]], Sequence[float]], t_span: tuple, y0: Sequence[float], *,
           rtol: float = 1e-9, atol: float = 1e-12, events: Sequence[Event] = (), dense: bool = True,
           first_step: float | None = None, max_step: float = math.inf, max_steps: int = 1_000_000) -> ODEResult:
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] (either direction).

    Returns an :class:`ODEResult` with the accepted points, located events and (when ``dense``) a callable interpolant.
    A terminal event ends the integration at the located event time, which is appended as the last point.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    direction = 1.0 if tf >= t0 else -1.0
    y = [float(v) for v in y0]
    n = len(y)
    res = ODEResult()
    res.t.append(t0)
    res.y.append(list(y))
    k1 = list(f(t0, y))
    res.nfev = 1
    h = abs(first_step) if first_step else _initial_step(f, t0, y, k1, direction, rtol, atol)
    h = min(h, max_step)
    t = t0
    ev_prev = [ev.g(t, y) for ev in events]
    steps = 0
    while direction * (tf - t) > 0:
        if steps >= max_steps:
            res.status = "max_steps"
            return res
        h = min(h, abs(tf - t), max_step)
        if h < 1e-14 * max(1.0, abs(t)):
            res.status = "step_underflow"
            return res
        hs = direction * h
        ks = [k1]
        for s in range(1, 7):
            ys = [y[i] + hs * sum(_A[s][j] * ks[j][i] for j in range(s)) for i in range(n)]
            ks.append(list(f(t + _C[s] * hs, ys)))
        res.nfev += 6
        y_new = ys                                           # stage 7 input is the 5th-order solution (FSAL)
        k7 = ks[6]
        err2 = 0.0
        for i in range(n):
            e = hs * sum(_E[j] * ks[j][i] for j in range(7))
            sc = atol + rtol * max(abs(y[i]), abs(y_new[i]))
            err2 += (e / sc) ** 2
        err = math.sqrt(err2 / n)
        if err > 1.0:
            h *= max(0.2, 0.9 * err ** -0.2)
            continue
        t_new = t + hs
        steps += 1
        ydiff = [y_new[i] - y[i] for i in range(n)]
        r3 = [hs * ks[0][i] - ydiff[i] for i in range(n)]
        r4 = [ydiff[i] - hs * k7[i] - r3[i] for i in range(n)]
        r5 = [hs * (_D1 * ks[0][i] + _D3 * ks[2][i] + _D4 * ks[3][i] + _D5 * ks[4][i] + _D6 * ks[5][i] + _D7 * k7[i]) for i in range(n)]
        seg = (t, hs, list(y), ydiff, r3, r4, r5)
        stop = False
        for idx, ev in enumerate(events):
            g_new = ev.g(t_new, y_new)
            g_old = ev_prev[idx]
            crossed = (g_old < 0 <= g_new and ev.direction >= 0) or (g_old > 0 >= g_new and ev.direction <= 0)
            if crossed and g_old != g_new:
                te = _brent(lambda tt: ev.g(tt, _eval_segment(seg, tt)), t, t_new, g_old, g_new)
                ye = _eval_segment(seg, te)
                res.events.append((idx, te, ye))
                if ev.terminal:
                    seg_t = (t, hs, list(y), ydiff, r3, r4, r5)
                    if dense:
                        res._segments.append(seg_t)
                    res.t.append(te)
                    res.y.append(ye)
                    res.status = "terminal_event"
                    stop = True
                    break
            ev_prev[idx] = g_new
        if stop:
            return res
        if dense:
            res._segments.append(seg)
        res.t.append(t_new)
        res.y.append(list(y_new))
        t, y, k1 = t_new, y_new, k7
        h *= min(10.0, max(0.2, 0.9 * (err ** -0.2 if err > 0 else 10.0)))
    res.status = "finished"
    return res
