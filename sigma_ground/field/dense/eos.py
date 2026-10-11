"""Equation-of-state objects for the structure solvers (TOV, white dwarfs).

An EOS here is a barotropic relation: energy density eps and number density n as functions of pressure P (SI: J/m^3, 1/m^3, Pa).
``TabulatedEOS`` interpolates log(eps) and log(n) against log(P) with monotone-safe cubic Hermite splines whose slopes are the *exact*
thermodynamic derivatives (so the table obeys the first law to interpolation accuracy):
    d ln eps / d ln P = P / (eps c_s^2)            (c_s^2 = dP/d eps, in units of c^2)
    d ln n   / d ln P = P / (c_s^2 (eps + P))      (from d eps / d n = (eps + P)/n)

Derivation chain (Golden Rule 1): the slopes are identities of thermodynamics (T = 0, one conserved particle number); the tables are built
from exact analytic gases (fermi_gas.py) or, later, from fitted nuclear EOS. Golden Rule 2: ``regime`` is SAFE inside the tabulated range,
EDGE in the power-law tail below it, BEYOND above it.

PHYSICS_GAP: single-component barotropic fluids only (no composition gradients, no finite temperature).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Callable, Sequence

from sigma_ground.field.constants import AMU_KG, C
from sigma_ground.field.dense import fermi_gas as fg


class EOS:
    """Protocol: barotropic EOS in SI units."""
    baryon_mass_kg: float = 0.0

    def eps(self, pressure_pa: float) -> float:            # energy density, J/m^3 (rest mass included)
        raise NotImplementedError

    def n(self, pressure_pa: float) -> float:              # baryon number density, 1/m^3
        raise NotImplementedError

    def regime(self, pressure_pa: float) -> str:
        return "SAFE"

    surface_pressure_pa: float = 0.0                       # the pressure at which a star ends


class UniformEOS(EOS):
    """Incompressible fluid: eps = const for every P >= 0 (the Schwarzschild interior solution's EOS). Not causal; a test fixture."""

    def __init__(self, eps_jm3: float, baryon_mass_kg: float = AMU_KG):
        self._eps = float(eps_jm3)
        self.baryon_mass_kg = baryon_mass_kg

    def eps(self, p):
        return self._eps

    def n(self, p):
        return self._eps / (self.baryon_mass_kg * C * C)


@dataclass
class _Table:
    logp: list
    logeps: list
    logn: list
    slope_eps: list
    slope_n: list


class TabulatedEOS(EOS):
    """Log-log cubic-Hermite EOS built from arrays of (P, eps, n, c_s^2/c^2), strictly increasing in P."""

    def __init__(self, name: str, P: Sequence[float], eps: Sequence[float], n: Sequence[float], cs2: Sequence[float],
                 baryon_mass_kg: float, p_max_ok: float | None = None):
        if not (len(P) == len(eps) == len(n) == len(cs2) >= 4):
            raise ValueError("table arrays must have equal length >= 4")
        if any(P[i + 1] <= P[i] for i in range(len(P) - 1)):
            raise ValueError("pressure must be strictly increasing")
        self.name = name
        self.baryon_mass_kg = baryon_mass_kg
        self._p_min, self._p_max = P[0], P[-1]
        lp = [math.log(p) for p in P]
        self._t = _Table(
            lp, [math.log(e) for e in eps], [math.log(x) for x in n],
            [p / (e * c) for p, e, c in zip(P, eps, cs2)],
            [p / (c * (e + p)) for p, e, c in zip(P, eps, cs2)],
        )
        # power-law tail below the table: eps ~ P^a, n ~ P^b with the table's lowest slopes (exact for a polytropic low-density end)
        self._tail = (self._t.slope_eps[0], self._t.slope_n[0])
        self.surface_pressure_pa = 0.0

    def _hermite(self, lp: float, ys: list, ms: list) -> float:
        lps = self._t.logp
        i = bisect.bisect_right(lps, lp) - 1
        i = max(0, min(i, len(lps) - 2))
        h = lps[i + 1] - lps[i]
        t = (lp - lps[i]) / h
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * ms[i]
                + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * ms[i + 1])

    def _eval(self, p: float, ys: list, ms: list, tail_slope: float) -> float:
        if p <= 0.0:
            return 0.0
        lp = math.log(p)
        if lp < self._t.logp[0]:
            return math.exp(ys[0] + tail_slope * (lp - self._t.logp[0]))
        if lp > self._t.logp[-1] * (1 + 1e-12) and lp - self._t.logp[-1] > 1e-9:
            raise ValueError(f"pressure {p:.3e} Pa above the table maximum {self._p_max:.3e} Pa (regime BEYOND)")
        return math.exp(self._hermite(lp, ys, ms))

    def eps(self, p):
        return self._eval(p, self._t.logeps, self._t.slope_eps, self._tail[0])

    def n(self, p):
        return self._eval(p, self._t.logn, self._t.slope_n, self._tail[1])

    def regime(self, p):
        return "EDGE" if p < self._p_min else ("BEYOND" if p > self._p_max else "SAFE")

    @property
    def pressure_range_pa(self) -> tuple:
        return self._p_min, self._p_max


def _log_grid(lo: float, hi: float, n: int) -> list:
    return [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (n - 1)) for i in range(n)]


def free_fermi_eos(species: fg.FermionSpecies = fg.NEUTRON, x_min: float = 1e-3, x_max: float = 60.0, points: int = 1500) -> TabulatedEOS:
    """T = 0 gas of free fermions of one species (``free_neutron``: the Oppenheimer-Volkoff 1939 star)."""
    P, E, N, CS = [], [], [], []
    for x in _log_grid(x_min, x_max, points):
        P.append(fg.chandrasekhar_pressure(x, species))
        E.append(fg.chandrasekhar_energy(x, species))
        N.append(species.n0_m3 * x ** 3)
        CS.append(fg.sound_speed_sq_over_c2(x))
    return TabulatedEOS(f"free_{species.name}", P, E, N, CS, species.mass_kg)


def electron_ion_eos(mu_e: float = 2.0, x_min: float = 1e-2, x_max: float = 400.0, points: int = 1500) -> TabulatedEOS:
    """Degenerate electrons with ions of mean molecular weight per electron mu_e (the white-dwarf EOS)."""
    P, E, N, CS = [], [], [], []
    n0 = fg.ELECTRON.n0_m3
    for x in _log_grid(x_min, x_max, points):
        n_e = n0 * x ** 3
        rho = mu_e * AMU_KG * n_e
        s = fg.electron_gas(rho, mu_e)
        P.append(s.pressure_pa)
        E.append(s.energy_density_jm3)
        N.append(n_e / 1.0 * mu_e)             # baryon number density = rho / m_u = mu_e n_e (ions carry the baryons)
        CS.append(s.sound_speed_sq_over_c2)
    return TabulatedEOS(f"electron_ion_mue{mu_e:g}", P, E, N, CS, AMU_KG)


class PolytropeEOS(EOS):
    """Relativistic polytrope P = K rho^gamma with eps = rho c^2 + P/(gamma - 1): the n = 1/(gamma-1) polytrope, for the Newtonian-limit test."""

    def __init__(self, K: float, gamma: float, baryon_mass_kg: float = AMU_KG):
        if gamma <= 1.0:
            raise ValueError("gamma must exceed 1")
        self.K, self.gamma = K, gamma
        self.baryon_mass_kg = baryon_mass_kg
        self.name = f"polytrope_g{gamma:g}"

    def rho(self, p):
        return (max(p, 0.0) / self.K) ** (1.0 / self.gamma)

    def eps(self, p):
        return self.rho(p) * C * C + max(p, 0.0) / (self.gamma - 1.0)

    def n(self, p):
        return self.rho(p) / self.baryon_mass_kg


def first_law_error(eos: TabulatedEOS, p: float, h: float = 1e-4) -> float:
    """|d eps/dn * n / (eps + P) - 1| by finite difference along the table: a self-check of the interpolation."""
    p1, p2 = p * (1 - h), p * (1 + h)
    d_eps = eos.eps(p2) - eos.eps(p1)
    d_n = eos.n(p2) - eos.n(p1)
    return abs(d_eps / d_n * eos.n(p) / (eos.eps(p) + p) - 1.0)
