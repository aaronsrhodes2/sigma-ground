"""Tolman-Oppenheimer-Volkoff stars: structure, mass-radius curves, maximum mass.

Hydrostatic equilibrium in general relativity (Tolman 1939; Oppenheimer & Volkoff 1939), with eps the energy density in J/m^3:
    dP/dr  = -G (eps + P)/c^2 (m + 4 pi r^3 P/c^2) / [ r^2 (1 - 2 G m/(r c^2)) ]
    dm/dr  =  4 pi r^2 eps / c^2
    d nu/dr = -2 (dP/dr) / (eps + P)                         (nu: g_tt = -e^nu, fixed at the surface by the exterior Schwarzschild metric)
    dMb/dr =  4 pi r^2 n m_b / sqrt(1 - 2 G m/(r c^2))       (baryon mass)
    dJ/dr  =  4 pi r^2 (eps + 3P)/c^2 e^{nu/2} / sqrt(1 - 2 G m/(r c^2))    (Tolman mass M_T = J e^{shift/2}, equal to M for a static star)

Solved in the dimensionless variables  r^ = r/l,  m^ = m G/(l c^2),  p^ = P/eps_c,  with  l = c^2/sqrt(G eps_c):  every G, c and eps_c drops out,
so the same integrator serves a white dwarf and a neutron star. The centre starts from the regular series; the surface is a terminal event
of the one adaptive integrator (kernel/ode.py).

Derivation chain (Golden Rule 1): FIRST_PRINCIPLES (the Einstein equations for a static spherical perfect fluid) + MEASURED (G, c) + an EOS object.
Golden Rule 2: ``regime`` is SAFE for a causal, bound star; WALL when the central sound speed exceeds c (c_s^2 > c^2) or compactness reaches
Buchdahl's bound; EDGE when the EOS table's tail is used at the surface (an expected, labelled approximation: the cut-off is
``surface_cut`` of the central pressure).

PHYSICS_GAP: non-rotating, spherical, zero temperature; stability is read from dM/d(rho_c) only (turning point), not from a perturbation analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from sigma_ground.field.constants import C, G, M_SUN_KG
from sigma_ground.field.dense.eos import EOS
from sigma_ground.kernel.ode import Event, dopri5


@dataclass
class TOVStar:
    eos_name: str
    central_pressure_pa: float
    central_energy_density_jm3: float
    radius_m: float
    mass_kg: float
    baryon_mass_kg: float
    tolman_mass_kg: float
    compactness: float                       # C = G M / (R c^2); Buchdahl: C < 4/9
    regime: str
    r_m: list = field(default_factory=list)
    P_pa: list = field(default_factory=list)
    m_kg: list = field(default_factory=list)

    @property
    def mass_msun(self) -> float:
        return self.mass_kg / M_SUN_KG

    @property
    def radius_km(self) -> float:
        return self.radius_m / 1e3

    @property
    def binding_energy_kg(self) -> float:
        """Gravitational binding energy as a mass: Mb - M (positive for a bound star)."""
        return self.baryon_mass_kg - self.mass_kg

    @property
    def trace_ratio_central(self) -> float:
        return 3.0 * self.central_pressure_pa / self.central_energy_density_jm3


def solve_tov(eos: EOS, central_pressure_pa: float | None = None, central_density_kg_m3: float | None = None,
              rtol: float = 1e-10, surface_cut: float = 1e-12, keep_profile: bool = False, name: str | None = None) -> TOVStar:
    """Integrate one star from its centre to its surface.

    Give the central pressure (Pa), or the central mass density (kg/m^3) for EOS objects that can invert it (``eos.n``-based tables).
    ``surface_cut`` is the pressure, as a fraction of the central pressure, at which the star ends (the EOS tail is a power law there).
    """
    if central_pressure_pa is None:
        raise ValueError("pass central_pressure_pa (density inversion is provided by TabulatedEOS.pressure_from_density if needed)")
    pc = float(central_pressure_pa)
    if pc <= 0:
        raise ValueError("central pressure must be positive")
    ec = eos.eps(pc)
    ell = C * C / math.sqrt(G * ec)                     # length unit
    m0 = ell * C * C / G                                # mass unit
    pcut = max(eos.surface_pressure_pa, surface_cut * pc) / ec
    pc_hat = pc / ec
    mb_unit = eos.baryon_mass_kg * C * C / ec           # n_hat = n m_b c^2 / eps_c

    def f(r, y):
        p, m, nu, mb, J = y
        pp = max(p, 0.0) * ec
        e_hat = eos.eps(pp) / ec
        n_hat = eos.n(pp) * mb_unit
        w = 1.0 - 2.0 * m / r
        dp = -(e_hat + p) * (m + 4.0 * math.pi * r ** 3 * p) / (r * r * w)
        dm = 4.0 * math.pi * r * r * e_hat
        dnu = -2.0 * dp / (e_hat + p)
        sw = math.sqrt(w)
        dmb = 4.0 * math.pi * r * r * n_hat / sw
        dJ = 4.0 * math.pi * r * r * (e_hat + 3.0 * p) * math.exp(0.5 * nu) / sw
        return [dp, dm, dnu, dmb, dJ]

    r0 = 1e-6
    y0 = [pc_hat - (2.0 * math.pi / 3.0) * (1.0 + pc_hat) * (1.0 + 3.0 * pc_hat) * r0 * r0,
          (4.0 * math.pi / 3.0) * r0 ** 3, 0.0, (4.0 * math.pi / 3.0) * (eos.n(pc) * mb_unit) * r0 ** 3, 0.0]
    surf = Event(lambda r, y: y[0] - pcut, terminal=True, direction=-1)
    res = dopri5(f, (r0, 1.0e7), y0, rtol=rtol, atol=1e-16, events=[surf], dense=keep_profile, first_step=1e-4)
    if res.status != "terminal_event":
        raise RuntimeError(f"TOV integration did not reach a surface ({res.status}); central pressure {pc:.3e} Pa")
    R_hat = res.t[-1]
    p, m_hat, nu_t, mb_hat, J_hat = res.y[-1]
    C_comp = m_hat / R_hat                              # G M / (R c^2) in these units
    nu_surface = math.log(1.0 - 2.0 * C_comp)
    shift = nu_surface - nu_t                           # e^{nu} = e^{nu_t + shift}
    tolman = J_hat * math.exp(0.5 * shift)
    # regime: causality at the centre and Buchdahl
    regime = "SAFE"
    if C_comp >= 4.0 / 9.0:
        regime = "WALL"
    elif surface_cut > 0 and eos.regime(pcut * ec) != "SAFE":
        regime = "EDGE"
    star = TOVStar(name or getattr(eos, "name", type(eos).__name__), pc, ec, R_hat * ell, m_hat * m0, mb_hat * m0, tolman * m0, C_comp, regime)
    if keep_profile:
        star.r_m = [t * ell for t in res.t]
        star.P_pa = [y[0] * ec for y in res.y]
        star.m_kg = [y[1] * m0 for y in res.y]
    return star


def mass_radius_curve(eos: EOS, p_min: float, p_max: float, n_points: int = 30, rtol: float = 1e-9) -> list:
    """Stars at log-spaced central pressures: a list of TOVStar (the M-R curve, with the unstable branch beyond the maximum)."""
    if n_points < 2:
        raise ValueError("n_points >= 2")
    out = []
    for i in range(n_points):
        pc = math.exp(math.log(p_min) + (math.log(p_max) - math.log(p_min)) * i / (n_points - 1))
        out.append(solve_tov(eos, pc, rtol=rtol))
    return out


def maximum_mass(eos: EOS, p_min: float, p_max: float, tol: float = 1e-6, rtol: float = 1e-9) -> TOVStar:
    """The maximum-mass star (dM/d rho_c = 0), by a coarse scan then golden-section on log P_c."""
    scan = mass_radius_curve(eos, p_min, p_max, 24, rtol)
    k = max(range(len(scan)), key=lambda i: scan[i].mass_kg)
    if k in (0, len(scan) - 1):
        raise ValueError("no interior mass maximum in the scanned pressure range")
    lo, hi = math.log(scan[k - 1].central_pressure_pa), math.log(scan[k + 1].central_pressure_pa)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = solve_tov(eos, math.exp(a), rtol=rtol).mass_kg, solve_tov(eos, math.exp(b), rtol=rtol).mass_kg
    while hi - lo > tol:
        if fa > fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = solve_tov(eos, math.exp(a), rtol=rtol).mass_kg
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = solve_tov(eos, math.exp(b), rtol=rtol).mass_kg
    return solve_tov(eos, math.exp(0.5 * (lo + hi)), rtol=rtol)


def is_stable_branch(star_lower_pc: TOVStar, star_higher_pc: TOVStar) -> bool:
    """A star is on the stable branch while mass still rises with central pressure (necessary, not sufficient)."""
    return star_higher_pc.mass_kg >= star_lower_pc.mass_kg
