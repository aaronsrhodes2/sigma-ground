"""White dwarfs: Chandrasekhar's mass limit and the structure of a cold electron-degenerate star.

Newtonian hydrostatics with the exact T = 0 electron EOS (fermi_gas.py), in the variable u = x^2 (x = p_F/m_e c), which makes the equations regular
at the surface:  dm/dr = 4 pi r^2 rho,   du/dr = -6 G m A sqrt(1 + u) / (r^2 eps0)   with rho = A x^3, A = mu_e m_u n0, eps0 = m_e^4 c^5/(pi^2 hbar^3).
(Derivation: dP/dr = -G m rho/r^2 and dP/dx = eps0 x^4 / (3 sqrt(1 + x^2)).) The surface is the terminal event u = 0.

Chandrasekhar's limit (1931, 1935) is the ultra-relativistic n = 3 polytrope:
    M_Ch = (sqrt(3 pi)/2) omega_3 (hbar c / G)^(3/2) / (mu_e m_B)^2,    omega_3 = 2.01824 (Lane-Emden, polytrope.py).

Derivation chain (Golden Rule 1): FIRST_PRINCIPLES (the two relations above) + MEASURED (hbar, c, G, m_e, u). ``gr=True`` solves the full
Tolman-Oppenheimer-Volkoff equations on the same EOS (tov.py), which turns the sequence over below the Newtonian limit (a theorem of GR).

Golden Rule 2: ``regime`` is EDGE for rho_c < 1e6 kg/m^3 (x < 0.07: Coulomb and lattice corrections, not modelled, matter), WALL once the
central density exceeds the neutron-drip density (the electron-ion EOS no longer describes the matter), SAFE between.

PHYSICS_GAP: no Coulomb (Salpeter) corrections yet; ions treated as non-degenerate with zero pressure; zero temperature.
PHYSICS_GAP: ``RHO_DRIP_KG_M3`` is a literature value (~4.3e11 g/cm^3), flagged LIT, not derived here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sigma_ground.field.constants import AMU_KG, C, G, HBAR, M_SUN_KG
from sigma_ground.field.dense import eos as eosmod
from sigma_ground.field.dense import fermi_gas as fg
from sigma_ground.field.dense.polytrope import lane_emden
from sigma_ground.field.dense.tov import maximum_mass, solve_tov
from sigma_ground.kernel.ode import Event, dopri5

RHO_DRIP_KG_M3 = 4.3e14      # [LIT] neutron-drip density, about 4.3e11 g/cm^3 (Baym, Pethick & Sutherland 1971)


def chandrasekhar_mass(mu_e: float = 2.0, baryon_mass_kg: float = AMU_KG) -> float:
    """Chandrasekhar's limiting mass in kg."""
    w3 = lane_emden(3.0).omega
    return (math.sqrt(3.0 * math.pi) / 2.0) * w3 * (HBAR * C / G) ** 1.5 / (mu_e * baryon_mass_kg) ** 2


@dataclass(frozen=True)
class WhiteDwarf:
    mu_e: float
    central_density_kg_m3: float
    radius_m: float
    mass_kg: float
    gr: bool
    regime: str

    @property
    def mass_msun(self) -> float:
        return self.mass_kg / M_SUN_KG


def _regime(rho_c: float) -> str:
    return "EDGE" if rho_c < 1e6 else ("WALL" if rho_c > RHO_DRIP_KG_M3 else "SAFE")


def white_dwarf_structure(rho_c: float, mu_e: float = 2.0, gr: bool = False, rtol: float = 1e-10) -> WhiteDwarf:
    """Mass and radius of a cold white dwarf with central mass density rho_c (kg/m^3)."""
    if rho_c <= 0 or mu_e <= 0:
        raise ValueError("rho_c and mu_e must be positive")
    reg = _regime(rho_c)
    gas = fg.electron_gas(rho_c, mu_e)
    if gr:
        star = solve_tov(eosmod.electron_ion_eos(mu_e), gas.pressure_pa, rtol=rtol)
        return WhiteDwarf(mu_e, rho_c, star.radius_m, star.mass_kg, True, reg)
    e0 = fg.ELECTRON.eps0_pa
    A = mu_e * AMU_KG * fg.ELECTRON.n0_m3
    L = 1e6                                               # length unit (m); mass in solar masses
    uc = gas.x ** 2
    k = 6.0 * G * A / e0 * M_SUN_KG / L                   # du/d(r_hat) = -k m_hat sqrt(1+u) / r_hat^2   (r in units of L, m in M_sun)
    mass_coef = 4.0 * math.pi * A * L ** 3 / M_SUN_KG     # dm_hat/dr_hat = mass_coef r_hat^2 x^3

    def f(r, y):
        u, m = y
        uu = max(u, 0.0)
        return [-k * m * math.sqrt(1.0 + uu) / (r * r), mass_coef * r * r * uu ** 1.5]

    r0 = 1e-5
    m0 = mass_coef * uc ** 1.5 * r0 ** 3 / 3.0
    y0 = [uc - (k / 2.0) * (mass_coef * uc ** 1.5 / 3.0) * math.sqrt(1.0 + uc) * r0 * r0, m0]
    res = dopri5(f, (r0, 100.0), y0, rtol=rtol, atol=1e-16,
                 events=[Event(lambda r, y: y[0], terminal=True, direction=-1)], dense=False)
    if res.status != "terminal_event":
        raise RuntimeError("white-dwarf surface not found")
    return WhiteDwarf(mu_e, rho_c, res.t[-1] * L, res.y[-1][1] * M_SUN_KG, False, reg)


def white_dwarf_mass_radius(rho_c_list, mu_e: float = 2.0, gr: bool = False) -> list:
    return [white_dwarf_structure(r, mu_e, gr) for r in rho_c_list]


def white_dwarf_max_mass_gr(mu_e: float = 2.0) -> float:
    """Maximum mass (kg) of the general-relativistic white-dwarf sequence on the ideal electron-ion EOS."""
    eos = eosmod.electron_ion_eos(mu_e)
    p_lo = fg.electron_gas(1e11, mu_e).pressure_pa
    p_hi = fg.electron_gas(3e16, mu_e).pressure_pa
    return maximum_mass(eos, p_lo, p_hi, tol=1e-5, rtol=1e-9).mass_kg
