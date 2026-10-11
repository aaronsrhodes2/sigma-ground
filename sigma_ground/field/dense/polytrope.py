"""Polytropes: the Lane-Emden equation and the structure of an n-polytrope.

For P = K rho^(1 + 1/n) in Newtonian hydrostatic balance, theta(xi) = (rho/rho_c)^(1/n) obeys the Lane-Emden equation
    theta'' + (2/xi) theta' + theta^n = 0,   theta(0) = 1, theta'(0) = 0,
with r = a xi, a^2 = (n + 1) K rho_c^(1/n - 1) / (4 pi G). The first zero xi_1 is the surface; omega_n = -xi_1^2 theta'(xi_1) fixes the mass,
M = 4 pi a^3 rho_c omega_n.

Derivation chain (Golden Rule 1): FIRST_PRINCIPLES (Newtonian hydrostatics; Chandrasekhar 1939, ch. IV) + MEASURED (G). The centre is a regular
singular point, so integration starts from the series theta = 1 - xi^2/6 + n xi^4/120 (error O(xi^6) at xi_0 = 1e-4: 1e-24), using the one
adaptive integrator (kernel/ode.py) with a terminal event at theta = 0.

Golden Rule 2: n must lie in [0, 5): at n >= 5 the radius is infinite (``BEYOND``). PHYSICS_GAP: Newtonian only (GR is tov.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sigma_ground.field.constants import G
from sigma_ground.kernel.ode import Event, dopri5


@dataclass(frozen=True)
class LaneEmden:
    n: float
    xi1: float                  # first zero of theta
    omega: float                # -xi1^2 theta'(xi1)
    regime: str = "SAFE"


_CACHE: dict = {}


def lane_emden(n: float, rtol: float = 1e-12) -> LaneEmden:
    """Surface radius xi_1 and mass parameter omega_n of the n-polytrope, 0 <= n < 5."""
    if not (0.0 <= n < 5.0):
        raise ValueError("Lane-Emden surface is finite only for 0 <= n < 5")
    key = (n, rtol)
    if key in _CACHE:
        return _CACHE[key]
    x0 = 1e-4

    def f(x, y):
        th, dth = y
        return [dth, -(max(th, 0.0) ** n) - 2.0 * dth / x]

    y0 = [1.0 - x0 ** 2 / 6.0 + n * x0 ** 4 / 120.0, -x0 / 3.0 + n * x0 ** 3 / 30.0]
    res = dopri5(f, (x0, 100.0), y0, rtol=rtol, atol=1e-16,
                 events=[Event(lambda x, y: y[0], terminal=True, direction=-1)], dense=False)
    if res.status != "terminal_event":
        raise RuntimeError("Lane-Emden surface not found")
    xi1 = res.t[-1]
    out = LaneEmden(n, xi1, -xi1 ** 2 * res.y[-1][1])
    _CACHE[key] = out
    return out


@dataclass(frozen=True)
class PolytropeStar:
    n: float
    K: float
    central_density_kg_m3: float
    radius_m: float
    mass_kg: float
    central_pressure_pa: float


def polytrope_structure(n: float, K: float, rho_c: float) -> PolytropeStar:
    """Radius and mass of the n-polytrope with constant K (SI: P = K rho^(1+1/n)) and central density rho_c."""
    if K <= 0 or rho_c <= 0:
        raise ValueError("K and rho_c must be positive")
    le = lane_emden(n)
    a = math.sqrt((n + 1.0) * K * rho_c ** (1.0 / n - 1.0) / (4.0 * math.pi * G))
    return PolytropeStar(n, K, rho_c, a * le.xi1, 4.0 * math.pi * a ** 3 * rho_c * le.omega, K * rho_c ** (1.0 + 1.0 / n))
