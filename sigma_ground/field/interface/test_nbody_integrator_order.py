"""Convergence order of the N-body integrators with velocity-dependent forces (fixed 2026-10-10, A12/A1).

Before the fix, the 1PN layers (gr_1pn, eih_cross) read the velocity inside compute_accelerations, but every kick was
explicit in v: step() evaluated its closing half-kick at a(x_{n+1}, v_n), and forest_ruth_step and respa_step did the
same. All three fell to FIRST order (measured p -> 1.0-1.3). The fix makes each kick time-symmetric (an implicit
velocity update, solved by fixed-point iteration), which restores order 2 for step() and respa_step and order 4 for
forest_ruth_step (Hairer, Lubich & Wanner 2006, Geometric Numerical Integration, II.4 and VI.3).

Ground truth here is independent of the engine: the right-hand side is the textbook 1PN law written out below
(PPN test-particle equation, Iorio arXiv:gr-qc/0405053 eq. 1; EIH equations, Will 1993) and the reference solution is
scipy's DOP853 at rtol 1e-13. The order is measured by halving dt.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.constants import C as _C, G as _G
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

try:
    from scipy.integrate import solve_ivp
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy ships with the mcp extra
    _HAVE_SCIPY = False

_C2 = _C * _C


def _textbook_acc(gms, xs, vs, mode: str) -> np.ndarray:
    """Newton plus the textbook 1PN correction: mode 'newton', 'gr_1pn' (test-particle law with each body's own
    velocity) or 'eih_cross' (Einstein-Infeld-Hoffmann)."""
    n = len(gms)
    a_n = np.zeros((n, 3)); phi = np.zeros(n)
    for i in range(n):
        for k in range(n):
            if k != i:
                d = xs[k] - xs[i]; r = float(np.linalg.norm(d))
                a_n[i] += gms[k] * d / r**3
                phi[i] += gms[k] / r
    out = a_n.copy()
    if mode == "newton":
        return out
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            xij = xs[i] - xs[j]; r = float(np.linalg.norm(xij))
            if mode == "gr_1pn":
                nn = xij / r; v = vs[i]
                out[i] += gms[j] / (_C2 * r**2) * ((4 * gms[j] / r - v @ v) * nn + 4 * (nn @ v) * v)
            else:
                A = (-4 * phi[i] / _C2 - phi[j] / _C2 + vs[i] @ vs[i] / _C2 + 2 * vs[j] @ vs[j] / _C2
                     - 4 * vs[i] @ vs[j] / _C2 - 1.5 * (xij @ vs[j] / r) ** 2 / _C2 + 0.5 * ((-xij) @ a_n[j]) / _C2)
                out[i] += gms[j] * (-xij) / r**3 * A
                out[i] += gms[j] / (_C2 * r**3) * (xij @ (4 * vs[i] - 3 * vs[j])) * (vs[i] - vs[j])
                out[i] += 3.5 / _C2 * gms[j] * a_n[j] / r
    return out


def _binary(m1: float, m2: float, r_peri: float, e: float):
    """Two bodies in their centre-of-mass frame, starting at periapsis; returns (bodies, period)."""
    M = m1 + m2; mu = _G * M
    vp = math.sqrt(mu * (1 + e) / r_peri)
    x = np.array([r_peri, 0.0, 0.0]); v = np.array([0.0, vp, 0.0])
    bodies = [CelestialBody(m1, -m2 / M * x, -m2 / M * v, 1.0, 0.0),
              CelestialBody(m2, m1 / M * x, m1 / M * v, 1.0, 0.0)]
    return bodies, 2 * math.pi * math.sqrt((r_peri / (1 - e)) ** 3 / mu)


def _reference_separation(bodies, mode: str, t_end: float) -> np.ndarray:
    gms = [b.gm_m3_s2 for b in bodies]

    def rhs(_t, y):
        xs = [y[0:3], y[6:9]]; vs = [y[3:6], y[9:12]]
        a = _textbook_acc(gms, xs, vs, mode)
        return np.concatenate([vs[0], a[0], vs[1], a[1]])

    y0 = np.concatenate([bodies[0].position_m, bodies[0].velocity_m_s, bodies[1].position_m, bodies[1].velocity_m_s])
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", rtol=1e-13, atol=1e-9)
    return sol.y[6:9, -1] - sol.y[0:3, -1]


def _engine_separation(bodies, mode: str, method: str, t_end: float, n_steps: int) -> np.ndarray:
    tog = PhysicsToggles() if mode == "newton" else PhysicsToggles(**{mode: True})
    s = NBodySystem([b.replace() for b in bodies], toggles=tog)
    h = t_end / n_steps
    for _ in range(n_steps):
        if method == "respa_step":
            s.respa_step(h, 4)          # no parents: every force is "slow", including the 1PN layer
        else:
            getattr(s, method)(h)
    return s.bodies[1].position_m - s.bodies[0].position_m


def _observed_orders(bodies, period, mode, method, ns):
    ref = _reference_separation(bodies, mode, period)
    errs = [float(np.linalg.norm(_engine_separation(bodies, mode, method, period, n) - ref) / np.linalg.norm(ref))
            for n in ns]
    return [math.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)], errs


@unittest.skipUnless(_HAVE_SCIPY, "needs scipy (pip install sigma-ground[mcp])")
class TestOrderWithVelocityDependentForces(unittest.TestCase):
    """Compact orbit, GM/(r c^2) = 1e-3 at periapsis, e = 0.3: the 1PN force is large enough to dominate the error
    when the integrator mishandles it (before the fix: step p = 1.46, 1.30; forest_ruth_step p = 0.99, 1.00)."""

    TEST_MASS = _binary(2.8e30, 1.0, 2.0e6, 0.3)
    EQUAL_MASS = _binary(1.4e30, 1.4e30, 2.0e6, 0.3)

    def _assert_order(self, system, mode, method, ns, expected):
        orders, errs = _observed_orders(*system, mode, method, ns)
        for p in orders:
            self.assertAlmostEqual(p, expected, delta=0.15,
                                   msg=f"{method} with {mode}: orders {orders}, errors {errs}")

    def test_step_is_second_order_with_gr_1pn(self):
        self._assert_order(self.TEST_MASS, "gr_1pn", "step", (200, 400, 800), 2.0)

    def test_step_is_second_order_with_eih_equal_masses(self):
        self._assert_order(self.EQUAL_MASS, "eih_cross", "step", (200, 400, 800), 2.0)

    def test_forest_ruth_is_fourth_order_with_gr_1pn(self):
        self._assert_order(self.TEST_MASS, "gr_1pn", "forest_ruth_step", (100, 200, 400), 4.0)

    def test_forest_ruth_is_fourth_order_with_eih_equal_masses(self):
        self._assert_order(self.EQUAL_MASS, "eih_cross", "forest_ruth_step", (100, 200, 400), 4.0)

    def test_respa_is_second_order_with_gr_in_the_slow_forces(self):
        self._assert_order(self.TEST_MASS, "gr_1pn", "respa_step", (200, 400, 800), 2.0)

    def test_newtonian_orders_unchanged(self):
        self._assert_order(self.TEST_MASS, "newton", "step", (200, 400, 800), 2.0)
        self._assert_order(self.TEST_MASS, "newton", "forest_ruth_step", (100, 200, 400), 4.0)


@unittest.skipUnless(_HAVE_SCIPY, "needs scipy (pip install sigma-ground[mcp])")
class TestMercuryWithGRAtShootoutStep(unittest.TestCase):
    """Sun-Mercury, Forest-Ruth at the shoot-out step dt = 0.1 d, one year, against DOP853 on the textbook law.
    With GR on, the error must be the 4th-order truncation error of the Newtonian run, not more. Before the fix it was
    205 km against 6.2 km for Newton (the first-order kick error dominated); after, 6.2 km for both."""

    def test_gr_error_equals_newtonian_truncation_error(self):
        m_sun, m_mer, a, e = 1.98892e30, 3.3011e23, 5.7909e10, 0.2056
        bodies, _ = _binary(m_sun, m_mer, a * (1 - e), e)
        year = 365.25 * 86400.0
        n = int(round(365.25 / 0.1))
        err = {}
        for mode in ("newton", "eih_cross"):
            ref = _reference_separation(bodies, mode, year)
            err[mode] = float(np.linalg.norm(_engine_separation(bodies, mode, "forest_ruth_step", year, n) - ref))
        self.assertLess(err["eih_cross"], 10e3, msg=str(err))
        self.assertAlmostEqual(err["eih_cross"] / err["newton"], 1.0, delta=0.05, msg=str(err))


class TestNewtonianPathHasNoIteration(unittest.TestCase):
    """Position-only forces keep the original explicit schemes: 2 force evaluations per Verlet step, 3 per Forest-Ruth
    step. The implicit kick runs only when gr_1pn or eih_cross is on."""

    def _count(self, toggles, method):
        bodies, period = _binary(2.0e30, 6.0e24, 1.5e11, 0.1)
        s = NBodySystem(bodies, toggles=toggles)
        calls = [0]
        inner = s.compute_accelerations

        def counted():
            calls[0] += 1
            return inner()

        s.compute_accelerations = counted
        for _ in range(10):
            getattr(s, method)(period / 1000)
        return calls[0] / 10

    def test_evaluation_counts(self):
        position_only = PhysicsToggles(gr_2pn=True, j2_zonal=True, tidal_force=True)
        self.assertEqual(self._count(PhysicsToggles(), "step"), 2.0)
        self.assertEqual(self._count(position_only, "step"), 2.0)
        self.assertEqual(self._count(PhysicsToggles(), "forest_ruth_step"), 3.0)
        self.assertEqual(self._count(position_only, "forest_ruth_step"), 3.0)
        self.assertGreater(self._count(PhysicsToggles(gr_1pn=True), "step"), 2.0)
        self.assertGreater(self._count(PhysicsToggles(eih_cross=True), "forest_ruth_step"), 3.0)

    def test_implicit_kick_converges_without_warning_at_solar_system_steps(self):
        import warnings as _w
        bodies, period = _binary(2.0e30, 6.0e24, 1.5e11, 0.1)
        s = NBodySystem(bodies, toggles=PhysicsToggles(eih_cross=True))
        with _w.catch_warnings():
            _w.simplefilter("error")
            for _ in range(20):
                s.forest_ruth_step(period / 100)


if __name__ == "__main__":
    unittest.main()
