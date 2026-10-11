"""SGIntegrator -- Forest-Ruth behind a scipy.integrate-like interface.

Provides sg_solve_ivp() and sg_odeint() with call signatures modelled on
scipy.integrate.solve_ivp / odeint, but FIXED-STEP: there is no error
control, no rtol/atol and no dense_output.  The default method is the
Forest-Ruth 4th-order symplectic splitting for separable Hamiltonian systems
(orbits, oscillators): its energy error stays bounded (no secular drift),
it is not zero.

The symplectic methods ('forest_ruth', 'verlet') need the state split as
y = [q_1..q_N, p_1..p_N] with dq/dt depending only on (p, t) and dp/dt only
on (q, t).  For anything else -- odd-length or first-order states, damping,
velocity-dependent forces -- use method='rk4'.  An odd-length state is
rejected; a probe of f at the start warns when the split is not separable.

Convention
----------
All public functions are prefixed sg_ (sigma-ground).  The scipy functions
remain available and unmodified via their normal import path.

Usage
-----
    # Before: scipy (RK45, not symplectic -- energy drifts over time)
    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, t_span, y0, method='RK45', t_eval=t_eval)

    # After: Forest-Ruth, symplectic -- bounded energy error, fixed step dt
    from sigma_ground.field.interface.adapters.integrator import sg_solve_ivp
    sol = sg_solve_ivp(f, t_span, y0, method='forest_ruth', t_eval=t_eval, dt=dt)

    # N-body shortcut (most common case):
    from sigma_ground.field.interface.adapters.integrator import sg_nbody
    final_bodies = sg_nbody(bodies, dt_s=3600.0, t_total_s=365.25*86400)
"""

from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


_METHODS = ("forest_ruth", "verlet", "rk4")
_SYMPLECTIC = ("forest_ruth", "verlet")

# Forest-Ruth coefficients (exact -- Forest & Ruth 1990, Physica D 43, 105)
_FR4_THETA = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_FR4_C = [_FR4_THETA / 2, (1 - _FR4_THETA) / 2, (1 - _FR4_THETA) / 2, _FR4_THETA / 2]
_FR4_D = [_FR4_THETA, 1 - 2 * _FR4_THETA, _FR4_THETA]  # d[1] is negative

# A t_eval time within this fraction of a step of a grid node is served by
# that node's state (no side step).
_NODE_SNAP_FRACTION = 1e-9


@dataclass
class SGSolution:
    """Return type for sg_solve_ivp -- modelled on scipy.integrate.OdeResult.

    status is 0 when the integration reached the end of t_span with a finite
    state, -1 when the state overflowed to inf/NaN (integration stops there;
    t and y hold the points recorded before the failure).  nfev counts the
    right-hand-side evaluations.
    """
    t:       NDArray[np.float64]   # time points
    y:       NDArray[np.float64]   # state (n_states × n_time)
    method:  str
    success: bool
    message: str
    status:  int = 0
    nfev:    int = 0


def sg_solve_ivp(
    fun:     Callable[[float, NDArray], NDArray],
    t_span:  tuple[float, float],
    y0:      NDArray[np.float64],
    method:  str = "forest_ruth",
    t_eval:  NDArray[np.float64] | None = None,
    dt:      float | None = None,
    include_gr: bool = False,
    args:    tuple | None = None,
) -> SGSolution:
    """Integrate an ODE with a fixed step (symplectic by default).

    The state vector y must be split as [q, p] (positions then momenta)
    for the symplectic methods, with dq/dt a function of (p, t) only and
    dp/dt a function of (q, t) only.  For any other system use method='rk4'.

    Parameters
    ----------
    fun     : callable f(t, y[, *args]) returning dy/dt (array or list), shape (n,)
    t_span  : (t0, tf) integration interval; tf < t0 integrates backward
    y0      : initial state; for the symplectic methods [q_1..q_N, p_1..p_N]
    method  : 'forest_ruth' (4th-order symplectic, default)
              'verlet'      (2nd-order symplectic, faster)
              'rk4'         (4th-order non-symplectic, for non-Hamiltonian)
    t_eval  : times at which to record the solution (optional).  They must
              lie inside t_span and be sorted in the direction of
              integration.  The returned sol.t equals t_eval.
    dt      : step magnitude (positive); if None, |tf - t0| / 1000.  The
              step is shortened so that a whole number of steps spans t_span.
    include_gr : must be False.  This solver integrates the f you pass and
              has no bodies or masses to add a relativistic term to; put the
              1PN acceleration in f, or use sg_nbody(..., include_gr=True).
    args    : extra positional arguments passed to fun, as in scipy.

    Returns
    -------
    SGSolution with fields t, y (states), method, success, message, status, nfev

    Notes
    -----
    Forest-Ruth requires a FIXED dt for symplecticity, so the main trajectory
    always steps on the grid t0 + k*dt.  A t_eval time between two grid nodes
    is served by one extra step of the same method from the preceding node
    (a side step that does not alter the main trajectory), so its local
    error is that of one step of length < dt.

    For an explicitly time-dependent f, the drifts of the splitting advance
    time (extended phase space, H = T(p) + p_t + V(q, t)), so the order is 4
    for driven Hamiltonian systems too.
    """
    if include_gr:
        raise ValueError(
            "sg_solve_ivp integrates the f(t, y) you pass and has no bodies "
            "or masses to add a GR term to, so include_gr=True is not "
            "supported. Put the 1PN acceleration in f, or use "
            "sg_nbody(..., include_gr=True) / NBodySystem(include_gr=True)."
        )
    if method not in _METHODS:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Choose 'forest_ruth', 'verlet', or 'rk4'."
        )

    counter = [0]
    user_fun = fun
    if args is not None:
        if not isinstance(args, tuple):
            args = (args,)
        extra = args

        def fun(t, y):  # noqa: F811 -- bind the extra arguments
            return user_fun(t, y, *extra)
    rhs_inner = fun

    def rhs(t, y):
        counter[0] += 1
        return np.asarray(rhs_inner(t, y), dtype=np.float64)

    t0, tf = float(t_span[0]), float(t_span[1])
    if not (math.isfinite(t0) and math.isfinite(tf)):
        raise ValueError(f"t_span must be finite, got {t_span!r}")
    span = tf - t0

    y = np.array(y0, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"y0 must be 1-D, got shape {y.shape}")
    n_state = y.size
    N = n_state // 2  # number of degrees of freedom
    if method in _SYMPLECTIC and (n_state == 0 or n_state % 2):
        raise ValueError(
            f"method={method!r} is a symplectic [q, p] splitting and needs an "
            f"even-length state [q_1..q_N, p_1..p_N]; got {n_state} "
            f"component(s). Use method='rk4' for first-order or odd-length "
            f"systems."
        )

    if dt is None:
        dt = abs(span) / 1_000
    else:
        dt = float(dt)
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(
                f"dt must be a positive finite step magnitude (the direction "
                f"comes from t_span), got {dt}"
            )

    # Requested output times
    if t_eval is not None:
        te = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        if te.ndim != 1:
            raise ValueError("t_eval must be 1-D")
        if not np.all(np.isfinite(te)):
            raise ValueError("t_eval must be finite")
        lo, hi = min(t0, tf), max(t0, tf)
        tol = 4 * np.finfo(float).eps * max(abs(t0), abs(tf), 1.0)
        if np.any(te < lo - tol) or np.any(te > hi + tol):
            raise ValueError(
                f"Values in t_eval are not within t_span {t_span!r}")
        if te.size > 1 and np.any(np.diff(te) * (1.0 if span >= 0 else -1.0) < 0):
            raise ValueError(
                "Values in t_eval are not sorted in the direction of "
                "integration")
    else:
        te = None

    if method in _SYMPLECTIC:
        _warn_if_not_separable(rhs, t0, y, N, method)

    # Zero-length span: the solution is y0.
    if span == 0.0:
        times = [t0] if te is None else list(te)
        return SGSolution(
            t=np.array(times, dtype=np.float64),
            y=np.column_stack([y.copy() for _ in times]),
            method=method, success=True, message="OK", status=0,
            nfev=counter[0],
        )

    n_steps = int(math.ceil(abs(span) / dt))
    dt_real = span / n_steps  # signed; lands exactly on tf

    def _step(t_now, y_now, h):
        if method == "forest_ruth":
            return _fr4_step(rhs, t_now, y_now, h, N, _FR4_C, _FR4_D)
        if method == "verlet":
            return _verlet_step(rhs, t_now, y_now, h, N)
        return _rk4_step(rhs, t_now, y_now, h)

    # Determine recording: grid nodes (t_eval None) or exact t_eval times
    pending: dict[int, list[tuple[float, float]]] = {}
    if te is None:
        record_steps = set(range(0, n_steps + 1, max(1, n_steps // 200)))
        record_steps.add(n_steps)
    else:
        record_steps = set()
        for t_req in te:
            u = (t_req - t0) / dt_real
            k = int(math.floor(u))
            frac = u - k
            if frac >= 1.0 - _NODE_SNAP_FRACTION:
                k, frac = k + 1, 0.0
            k = min(max(k, 0), n_steps)
            if frac <= _NODE_SNAP_FRACTION or k == n_steps:
                frac = 0.0
            pending.setdefault(k, []).append((float(t_req), frac))

    t_out: list[float]             = []
    y_out: list[NDArray[np.float64]] = []

    t = t0
    status, message = 0, "OK"

    for step in range(n_steps + 1):
        if step in record_steps:
            t_out.append(t)
            y_out.append(y.copy())
        for t_req, frac in pending.get(step, ()):
            t_out.append(t_req)
            y_out.append(y.copy() if frac == 0.0 else _step(t, y, t_req - t))

        if step == n_steps:
            break

        y = _step(t, y, dt_real)
        t += dt_real
        if not np.all(np.isfinite(y)):
            status = -1
            message = (
                f"State became non-finite (inf/NaN) at t = {t:.6g} after "
                f"{step + 1} of {n_steps} steps; the solution is returned up "
                f"to the last finite step."
            )
            break

    if status == 0 and y_out and not all(np.all(np.isfinite(v)) for v in y_out):
        status = -1
        message = "A recorded state is non-finite (inf/NaN)."

    return SGSolution(
        t=np.array(t_out),
        y=np.column_stack(y_out) if y_out else np.zeros((n_state, 0)),
        method=method,
        success=(status == 0),
        message=message,
        status=status,
        nfev=counter[0],
    )


def _warn_if_not_separable(
    rhs: Callable, t: float, y: NDArray, N: int, method: str,
) -> None:
    """Probe f at the start: warn if the [q, p] split is not separable.

    The symplectic splittings integrate dq/dt = F(p, t), dp/dt = G(q, t)
    exactly piece by piece; if dq/dt depends on q or dp/dt on p (damping,
    first-order systems written as pairs, velocity-dependent forces) the
    method silently drops to first order.  The probe nudges q and p by a
    relative 1e-6 and flags a response above 1e-9 of the derivative's size
    (far above rounding, far below any coupling that matters at order 1e-3).
    """
    f0 = rhs(t, y)
    if f0.shape != y.shape:
        raise ValueError(
            f"f(t, y) returned shape {f0.shape}; expected {y.shape}")

    def _nudged(block: slice) -> NDArray:
        x = y[block]
        scale = float(np.max(np.abs(x))) if x.size else 0.0
        h = 1e-6 * (scale if scale > 0.0 else 1.0)
        y1 = y.copy()
        y1[block] = x + h
        return rhs(t, y1)

    def _responds(a: NDArray, b: NDArray) -> bool:
        size = float(np.max(np.abs(a))) + float(np.max(np.abs(b)))
        diff = float(np.max(np.abs(a - b)))
        return math.isfinite(diff) and diff > 1e-9 * size and diff > 0.0

    q_part, p_part = slice(0, N), slice(N, 2 * N)
    couplings = []
    fq = _nudged(q_part)
    if _responds(f0[q_part], fq[q_part]):
        couplings.append("dq/dt depends on q")
    fp = _nudged(p_part)
    if _responds(f0[p_part], fp[p_part]):
        couplings.append("dp/dt depends on p")
    if couplings:
        warnings.warn(
            f"sg_solve_ivp(method={method!r}): {' and '.join(couplings)}, so "
            f"f is not a separable [q, p] Hamiltonian split "
            f"(dq/dt = F(p, t), dp/dt = G(q, t)) and the symplectic method "
            f"drops to first order. Use method='rk4' for damped, first-order "
            f"or velocity-dependent systems.",
            RuntimeWarning, stacklevel=3,
        )


def _fr4_step(
    fun: Callable, t: float, y: NDArray, dt: float,
    N: int, c: list, d: list,
) -> NDArray:
    """One Forest-Ruth step: 4 drifts + 3 kicks.

    Time is a coordinate of the drift part (extended phase space,
    H = T(p) + p_t + V(q, t)): each drift advances t by c_i*dt and each kick
    is evaluated at the time reached, so a driven system keeps 4th order.
    """
    q, p = y[:N].copy(), y[N:].copy()
    tau = [t]  # current time inside the step

    def _drift(frac: float) -> None:
        # Compute dq/dt = dp/dt evaluated at momentum; for simple Hamiltonian
        # H = T(p) + V(q), dq/dt = dH/dp = p/m.  We call fun and use the
        # first N components (position derivatives) as the drift.
        dqdt = fun(tau[0], np.concatenate([q, p]))[:N]
        q[:] += frac * dt * dqdt
        tau[0] += frac * dt

    def _kick(frac: float) -> None:
        dpdt = fun(tau[0], np.concatenate([q, p]))[N:]
        p[:] += frac * dt * dpdt

    _drift(c[0]); _kick(d[0])
    _drift(c[1]); _kick(d[1])
    _drift(c[2]); _kick(d[2])
    _drift(c[3])

    return np.concatenate([q, p])


def _verlet_step(
    fun: Callable, t: float, y: NDArray, dt: float, N: int,
) -> NDArray:
    """One velocity-Verlet (leapfrog) step."""
    q, p = y[:N].copy(), y[N:].copy()
    dydt = fun(t, y)
    p_half = p + 0.5 * dt * dydt[N:]
    q_new  = q + dt * fun(t, np.concatenate([q, p_half]))[:N]
    dydt2  = fun(t + dt, np.concatenate([q_new, p_half]))
    p_new  = p_half + 0.5 * dt * dydt2[N:]
    return np.concatenate([q_new, p_new])


def _rk4_step(
    fun: Callable, t: float, y: NDArray, dt: float,
) -> NDArray:
    """Standard 4th-order Runge-Kutta (not symplectic)."""
    k1 = fun(t,            y)
    k2 = fun(t + dt / 2,  y + dt / 2 * k1)
    k3 = fun(t + dt / 2,  y + dt / 2 * k2)
    k4 = fun(t + dt,       y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ── N-body convenience wrapper ────────────────────────────────────────────

def sg_nbody(
    bodies: list,
    dt_s:   float,
    t_total_s: float,
    method: str = "forest_ruth",
    include_gr: bool = False,
    record_every: int = 1,
) -> tuple[list, list[float]]:
    """Propagate CelestialBody list using Forest-Ruth (or Verlet).

    Convenience wrapper around NBodySystem; no scipy dependency.

    Parameters
    ----------
    bodies     : list of CelestialBody (from sigma_ground.field.interface.nbody)
    dt_s       : fixed timestep (seconds) -- keep fixed for symplecticity
    t_total_s  : total integration time (seconds)
    method     : 'forest_ruth' or 'verlet' (anything else raises ValueError)
    include_gr : opt in to the 1PN Schwarzschild correction (nbody gr_1pn; a
                 single-body form, valid in the attractor's rest frame).
                 Default False since 2026-10-10: plain Newtonian, like the
                 scipy propagators this replaces.
    record_every : record the simulation TIME every N steps (1 = every
                 step); states are not stored -- use NBodySystem directly
                 for a trajectory.

    Returns
    -------
    final_bodies : list of CelestialBody at t = t_total_s
    times_s      : list of recorded simulation times

    Notes
    -----
    Whole steps of dt_s are taken, then one shorter step to land exactly on
    t_total_s when dt_s does not divide it (FIXED 2026-10-10: ceil() whole
    steps overshot, e.g. dt = 3 d, t_total = 10 d ended at 12 d).
    """
    from sigma_ground.field.interface.nbody import NBodySystem

    if method not in ("forest_ruth", "verlet"):
        raise ValueError(
            f"Unknown method {method!r}. Choose 'forest_ruth' or 'verlet'."
        )
    dt_s, t_total_s = float(dt_s), float(t_total_s)
    if not (math.isfinite(dt_s) and dt_s > 0):
        raise ValueError(f"dt_s must be positive and finite, got {dt_s}")
    if not (math.isfinite(t_total_s) and t_total_s >= 0):
        raise ValueError(f"t_total_s must be >= 0 and finite, got {t_total_s}")
    if record_every != int(record_every) or record_every < 1:
        raise ValueError(f"record_every must be a positive integer, got {record_every}")

    system  = NBodySystem(bodies, include_gr=include_gr)
    step_fn = system.forest_ruth_step if method == "forest_ruth" else system.step
    ratio   = t_total_s / dt_s
    n_full  = int(math.floor(ratio))
    if ratio - n_full > 1.0 - 1e-9:   # e.g. 7 d / 0.7 d = 9.999...
        n_full += 1
    remainder = t_total_s - n_full * dt_s
    if abs(remainder) <= 1e-9 * dt_s:
        remainder = 0.0
    steps = [dt_s] * n_full + ([remainder] if remainder > 0.0 else [])
    times   = []

    for i, h in enumerate(steps):
        if i % record_every == 0:
            times.append(system.time)
        step_fn(h)

    times.append(system.time)
    return system.bodies, times


_TIME_PARAM_NAMES = {"t", "time", "tau", "tt"}


def _first_param_is_time(func: Callable) -> bool:
    """Guess f(t, y) vs f(y, t) from the first parameter's name."""
    try:
        params = list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return False
    if not params:
        return False
    first = params[0].lower()
    return (first in _TIME_PARAM_NAMES or first.startswith("t_")
            or first.startswith("time"))


def sg_odeint(
    func: Callable,
    y0:   NDArray,
    t:    NDArray,
    method: str = "forest_ruth",
    dt:   float | None = None,
    args: tuple = (),
    tfirst: bool | None = None,
) -> NDArray:
    """scipy.integrate.odeint-like interface on the fixed-step integrators.

    Parameters
    ----------
    func   : callable f(y, t, *args) (odeint convention) or f(t, y, *args)
    y0     : initial state (for the symplectic methods [q_1..q_N, p_1..p_N])
    t      : output times, monotonic (increasing or decreasing), any spacing;
             t[0] is the initial time.  Row k of the result is y(t[k]).
    method : 'forest_ruth', 'verlet', or 'rk4' (see sg_solve_ivp)
    dt     : integration step magnitude.  It is independent of the output
             spacing.  If None, the smaller of |t[-1] - t[0]| / 1000 and the
             mean output spacing |t[-1] - t[0]| / (len(t) - 1), rounded
             down to a whole number of steps: a dense grid is stepped at
             its own spacing (as before 2026-10-10), a coarse grid still at
             span / 1000, and the cost stays below max(1000, len(t) - 1)
             steps.  There is no error control: pass a smaller dt for more
             accuracy.
    args   : extra arguments passed to func, as in odeint.
    tfirst : True for f(t, y), False for f(y, t) (odeint's default).  If
             None, a first parameter named t, time, tau, t_* or time* means
             f(t, y); anything else means f(y, t).  Pass it explicitly when
             in doubt.

    Returns
    -------
    y : NDArray shape (len(t), len(y0)).  If the state overflows, the rows
        from the failure on are NaN and a RuntimeWarning is issued.
    """
    if not isinstance(args, tuple):
        args = (args,)
    if tfirst is None:
        tfirst = _first_param_is_time(func)
    if tfirst:
        def fun(t_, y_):
            return func(t_, y_, *args)
    else:
        def fun(t_, y_):
            return func(y_, t_, *args)

    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if t.ndim != 1 or t.size == 0:
        raise ValueError("t must be a non-empty 1-D sequence of times")
    y0 = np.asarray(y0, dtype=np.float64)
    if t.size < 2:
        return np.array([y0])
    steps = np.diff(t)
    if not (np.all(steps >= 0) or np.all(steps <= 0)):
        raise ValueError("t must be monotonic (increasing or decreasing)")

    if dt is None:
        # FIXED 2026-10-10: a fixed span/1000 default stepped dense output
        # grids more coarsely than the grid (q''=-q on linspace(0,300,30001):
        # error 0.16, against 2e-7 at the grid spacing).
        span = abs(float(t[-1]) - float(t[0]))
        if span > 0.0:
            h = min(span / 1000.0, span / (t.size - 1))
            n = max(1, math.ceil(span / h - 1e-9))  # rounding in span / h
            dt = span / n
            while math.ceil(span / dt) > n:  # sg_solve_ivp takes ceil(span/dt)
                dt = float(np.nextafter(dt, np.inf))

    sol = sg_solve_ivp(fun, (t[0], t[-1]), y0, method=method,
                       t_eval=t, dt=dt)
    out = sol.y.T
    if out.shape[0] < t.size:
        warnings.warn(f"sg_odeint: {sol.message}", RuntimeWarning, stacklevel=2)
        pad = np.full((t.size - out.shape[0], y0.size), np.nan)
        out = np.vstack([out, pad])

    # Return (n_time, n_state) like scipy.integrate.odeint
    return out
