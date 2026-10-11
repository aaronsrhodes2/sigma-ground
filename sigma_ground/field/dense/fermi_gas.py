"""Degenerate Fermi gas: exact zero-temperature pressure, energy and trace, plus a finite-temperature non-relativistic gas.

The equation of state of a cold gas of spin-1/2 fermions (electrons in a white dwarf, free neutrons in the simplest neutron star),
written in the dimensionless Fermi momentum x = p_F / (m c).

Derivation chain (Golden Rule 1)
  MEASURED       : hbar, c (field.constants); the fermion masses (field.constants, Mentat lookup_constant, CODATA 2022).
  FIRST_PRINCIPLES (Chandrasekhar 1935; Landau & Lifshitz, Statistical Physics, sec. 61): counting states, n = g p_F^3 / (6 pi^2 hbar^3)
      for g = 2, then in units eps0 = m^4 c^5 / (pi^2 hbar^3):
        P        = eps0 * (1/3) * int_0^x t^4 / sqrt(1+t^2) dt            = eps0/24 [x(2x^2-3) sqrt(1+x^2) + 3 asinh x]
        eps      = eps0 * int_0^x t^2 sqrt(1+t^2) dt  (incl. rest mass)    = eps0/8  [x(2x^2+1) sqrt(1+x^2) - asinh x]
        eps - 3P = eps0 * int_0^x t^2 / sqrt(1+t^2) dt                     = eps0/2  [x sqrt(1+x^2) - asinh x]
      Near x -> 0 the closed forms lose digits to cancellation, so below ``_SERIES_X`` the integrals are summed as binomial series
      (convergent for x < 1); both forms are tested against direct numerical quadrature.
  Finite temperature (non-relativistic only): Fermi-Dirac integrals evaluated by quadrature, chemical potential by bisection.

Golden Rule 2: each public EOS function returns a ``regime`` label. SAFE: inside the model's validity. EDGE: approaching its breakdown
(here: x < 0.1 where Coulomb and ion-lattice corrections, not modelled here, matter for an electron gas; T/T_F > 0.1 where the T = 0 form
is no longer adequate). WALL: the breakdown itself. BEYOND: outside validity.

PHYSICS_GAP: free particles only (no Coulomb corrections here: Salpeter corrections live in white_dwarf.py; no nuclear interaction here: that
is nuclear_matter.py). Fully relativistic finite-temperature is not implemented.
PHYSICS_GAP: g = 2 spin states for every species.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sigma_ground.field.constants import (
    AMU_KG, C, HBAR, K_B, M_ELECTRON_KG, M_MUON_KG, M_NEUTRON_KG, M_PROTON_KG,
)

_SERIES_X = 0.3        # below this, sum the binomial series instead of the cancelling closed forms
_SERIES_TERMS = 16


@dataclass(frozen=True)
class FermionSpecies:
    name: str
    mass_kg: float
    g: int = 2                                   # spin degeneracy

    @property
    def mc2_j(self) -> float:
        return self.mass_kg * C * C

    @property
    def eps0_pa(self) -> float:
        """Energy-density scale m^4 c^5 / (pi^2 hbar^3), in J/m^3 (= Pa)."""
        return self.mass_kg ** 4 * C ** 5 / (math.pi ** 2 * HBAR ** 3)

    @property
    def n0_m3(self) -> float:
        """Number-density scale: the density at x = 1, (m c / hbar)^3 / (3 pi^2) for g = 2."""
        return (self.g / 2.0) * (self.mass_kg * C / HBAR) ** 3 / (3.0 * math.pi ** 2)


ELECTRON = FermionSpecies("electron", M_ELECTRON_KG)
MUON = FermionSpecies("muon", M_MUON_KG)
PROTON = FermionSpecies("proton", M_PROTON_KG)
NEUTRON = FermionSpecies("neutron", M_NEUTRON_KG)


def _binom(a: float, k: int) -> float:
    out = 1.0
    for j in range(k):
        out *= (a - j) / (j + 1)
    return out


# ---- the three dimensionless integrals, each exact -------------------------------------------------------------------
def _i_pressure(x: float) -> float:
    """(1/3) int_0^x t^4 / sqrt(1+t^2) dt  = P / eps0."""
    if x < _SERIES_X:
        return sum(_binom(-0.5, k) * x ** (5 + 2 * k) / (5 + 2 * k) for k in range(_SERIES_TERMS)) / 3.0
    return (x * (2 * x * x - 3) * math.sqrt(1 + x * x) + 3 * math.asinh(x)) / 24.0


def _i_energy_kinetic(x: float) -> float:
    """int_0^x t^2 (sqrt(1+t^2) - 1) dt = (eps - n m c^2) / eps0, the kinetic part."""
    if x < _SERIES_X:
        return sum(_binom(0.5, k) * x ** (3 + 2 * k) / (3 + 2 * k) for k in range(1, _SERIES_TERMS))
    return (x * (2 * x * x + 1) * math.sqrt(1 + x * x) - math.asinh(x)) / 8.0 - x ** 3 / 3.0


def _i_trace(x: float) -> float:
    """int_0^x t^2 / sqrt(1+t^2) dt = (eps - 3P) / eps0."""
    if x < _SERIES_X:
        return sum(_binom(-0.5, k) * x ** (3 + 2 * k) / (3 + 2 * k) for k in range(_SERIES_TERMS))
    return (x * math.sqrt(1 + x * x) - math.asinh(x)) / 2.0


# ---- public API (SI) --------------------------------------------------------------------------------------------------
def fermi_momentum(n_m3: float, species: FermionSpecies = NEUTRON) -> float:
    """p_F in kg m/s for number density n (m^-3): p_F = hbar (6 pi^2 n / g)^(1/3)."""
    if n_m3 < 0:
        raise ValueError("number density must be non-negative")
    return HBAR * (6.0 * math.pi ** 2 * n_m3 / species.g) ** (1.0 / 3.0)


def relativity_parameter(n_m3: float, species: FermionSpecies = NEUTRON) -> float:
    """x = p_F / (m c): ~0 non-relativistic, >> 1 ultra-relativistic."""
    return fermi_momentum(n_m3, species) / (species.mass_kg * C)


def chandrasekhar_pressure(x: float, species: FermionSpecies = ELECTRON) -> float:
    """Degenerate pressure in Pa at Fermi parameter x."""
    return species.eps0_pa * _i_pressure(x)


def chandrasekhar_energy(x: float, species: FermionSpecies = ELECTRON, include_rest: bool = True) -> float:
    """Energy density in J/m^3 (rest mass of the gas particles included unless ``include_rest`` is False)."""
    kin = species.eps0_pa * _i_energy_kinetic(x)
    return kin + (species.eps0_pa * x ** 3 / 3.0 if include_rest else 0.0)


def trace(x: float, species: FermionSpecies = ELECTRON) -> float:
    """eps - 3P in J/m^3, always positive for a free gas (theorem: the integrand is positive)."""
    return species.eps0_pa * _i_trace(x)


def sound_speed_sq_over_c2(x: float) -> float:
    """c_s^2 / c^2 = dP/d(eps) = x^2 / (3 (1 + x^2)): 0 at x = 0, 1/3 as x -> infinity (causal by construction)."""
    return x * x / (3.0 * (1.0 + x * x))


@dataclass(frozen=True)
class GasState:
    species: str
    n_m3: float
    x: float
    pressure_pa: float
    energy_density_jm3: float              # includes rest mass
    kinetic_energy_density_jm3: float      # excludes the rest mass of the gas particles (no cancellation at small x)
    mu_j: float                            # chemical potential per particle, rest mass included
    sound_speed_sq_over_c2: float
    three_p_over_eps: float
    regime: str
    temperature_k: float = 0.0


def eos_T0(n_m3: float, species: FermionSpecies = NEUTRON) -> GasState:
    """Exact T = 0 EOS of a free Fermi gas of one species at number density n."""
    x = relativity_parameter(n_m3, species)
    p = chandrasekhar_pressure(x, species)
    eps = chandrasekhar_energy(x, species)
    mu = species.mc2_j * math.sqrt(1.0 + x * x)
    regime = "EDGE" if (species is ELECTRON and 0 < x < 0.1) else "SAFE"
    kin = species.eps0_pa * _i_energy_kinetic(x)
    return GasState(species.name, n_m3, x, p, eps, kin, mu, sound_speed_sq_over_c2(x), 3.0 * p / eps if eps > 0 else 0.0, regime)


def electron_gas(rho_kg_m3: float, mu_e: float = 2.0) -> GasState:
    """Degenerate electrons neutralised by non-degenerate ions: n_e = rho / (mu_e m_u); energy density = rho c^2 + kinetic electron energy.

    mu_e is the mean molecular weight per electron (2 for fully ionised He, C, O; 1 for hydrogen; 2.15-2.2 for iron-group matter).
    """
    if rho_kg_m3 <= 0 or mu_e <= 0:
        raise ValueError("rho and mu_e must be positive")
    n_e = rho_kg_m3 / (mu_e * AMU_KG)
    x = relativity_parameter(n_e, ELECTRON)
    p = chandrasekhar_pressure(x)
    eps = rho_kg_m3 * C * C + ELECTRON.eps0_pa * _i_energy_kinetic(x)
    # sound speed of the combined fluid: dP / d eps with the rest mass of the ions carried along:  dP/d(rho c^2 + eps_kin)
    dp_dx = ELECTRON.eps0_pa * x ** 4 / (3.0 * math.sqrt(1 + x * x))
    de_dx = (rho_kg_m3 * C * C) * 3.0 / x + ELECTRON.eps0_pa * x * x * (math.sqrt(1 + x * x) - 1.0)
    cs2 = dp_dx / de_dx
    regime = "EDGE" if x < 0.1 else "SAFE"
    return GasState("electron+ions", n_e, x, p, eps, ELECTRON.eps0_pa * _i_energy_kinetic(x), ELECTRON.mc2_j * math.sqrt(1 + x * x), cs2,
                    3.0 * p / eps, regime)


# ---- finite temperature, non-relativistic ---------------------------------------------------------------------------
def _fd_integral(k: float, eta: float, n: int = 6000) -> float:
    """F_k(eta) = int_0^inf t^k / (exp(t - eta) + 1) dt for k = 1/2, 3/2, by Simpson on t = u^2 (removes the sqrt endpoint)."""
    umax = math.sqrt(max(eta, 0.0) + 45.0)
    h = umax / n
    total = 0.0
    for i in range(n + 1):
        u = i * h
        t = u * u
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        z = t - eta
        occ = 0.0 if z > 700 else 1.0 / (math.exp(z) + 1.0)
        total += w * 2.0 * u ** (2 * k + 1) * occ
    return total * h / 3.0


def eos_finite_T_nonrel(n_m3: float, temperature_k: float, species: FermionSpecies = ELECTRON) -> GasState:
    """Non-relativistic ideal Fermi gas at temperature T: P = (2/3) E_kin density, from Fermi-Dirac integrals.

    Spans the degenerate (T << T_F, Sommerfeld) and classical (T >> T_F, P = n k T) limits. Valid only for x << 1 (``BEYOND`` otherwise).
    """
    if temperature_k <= 0:
        return eos_T0(n_m3, species)
    kT = K_B * temperature_k
    m = species.mass_kg
    pref = species.g / (4.0 * math.pi ** 2) * (2.0 * m * kT / HBAR ** 2) ** 1.5      # n = pref * F_{1/2}(eta)
    target = n_m3 / pref
    lo, hi = -60.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _fd_integral(0.5, mid) < target:
            lo = mid
        else:
            hi = mid
    eta = 0.5 * (lo + hi)
    p = pref * kT * (2.0 / 3.0) * _fd_integral(1.5, eta)
    x = relativity_parameter(n_m3, species)
    e_f = (fermi_momentum(n_m3, species) ** 2) / (2.0 * m)
    t_ratio = kT / e_f
    regime = "BEYOND" if x > 0.3 else ("EDGE" if t_ratio > 0.1 and t_ratio < 10 else "SAFE")
    eps = n_m3 * species.mc2_j + 1.5 * p
    return GasState(species.name, n_m3, x, p, eps, 1.5 * p, species.mc2_j + kT * eta, 0.0, 3.0 * p / eps, regime, temperature_k)
