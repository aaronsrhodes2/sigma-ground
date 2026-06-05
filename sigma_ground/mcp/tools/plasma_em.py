"""Plasma & electromagnetism analysis tools (standard physics).

Composite tools cascading through field.interface.plasma and
field.electrodynamics.
"""
from __future__ import annotations

import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field (interface.plasma, electrodynamics)"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def _mag(v):
    if v is None:
        return None
    try:
        return math.sqrt(sum(float(c) ** 2 for c in v))
    except TypeError:
        return abs(float(v))


def plasma_parameters_analysis(electron_density_m3: float = 1.0e19,
                               electron_temperature_k: float = 1.0e6,
                               magnetic_field_t: float = 1.0,
                               ion_mass_kg: float = 1.673e-27,
                               perp_velocity_m_s: float = 1.0e5,
                               z_eff: float = 1.0) -> dict[str, Any]:
    """Core plasma parameters: Debye length, the number of particles in a Debye
    sphere, the Coulomb logarithm ln(Lambda), the ion Larmor (cyclotron) radius,
    and the Spitzer parallel resistivity eta_parallel. Defaults are a hot
    tenuous lab/fusion plasma. e.g. plasma_parameters_analysis(1e19, 1e6, 1.0).

    eta_parallel is the NRL/Spitzer parallel resistivity (validated to within
    ~1.2% of eta_par = 5.2e-5 Z lnLambda / T_eV^1.5 Ohm.m); eta_perp ~ 1.96x it.
    ln(Lambda) for the resistivity uses electron_density_m3, matching the
    coulomb_logarithm reported here."""
    from sigma_ground.field.interface import plasma as P
    results = {
        "debye_length_m": _safe(P.debye_length, electron_density_m3, electron_temperature_k),
        "debye_number": _safe(P.debye_number, electron_density_m3, electron_temperature_k),
        "coulomb_logarithm": _safe(P.coulomb_logarithm, electron_density_m3, electron_temperature_k),
        "cyclotron_radius_m": _safe(P.cyclotron_radius, ion_mass_kg, perp_velocity_m_s, magnetic_field_t),
        "spitzer_resistivity_ohm_m": _safe(P.spitzer_resistivity, electron_temperature_k, z_eff, electron_density_m3),
    }
    return ToolResult(value=results, units="m, dimensionless, dimensionless, m, ohm.m", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="lambda_D=sqrt(eps0 kT/n e^2); N_D=(4/3)pi n lambda_D^3; ln Lambda; "
                              "r_c=mv/qB; eta_par=0.51*(4 sqrt(2 pi)/3) sqrt(m_e) Z e^2 lnLambda/((4 pi eps0)^2 (k_B T_e)^1.5)",
                      inputs={"electron_density_m3": electron_density_m3,
                              "electron_temperature_k": electron_temperature_k,
                              "magnetic_field_t": magnetic_field_t,
                              "z_eff": z_eff}).to_dict()


def electromagnetic_force_analysis(charge_c: float = 1.602176634e-19,
                                   charge2_c: float = 1.602176634e-19,
                                   separation_m: float = 1.0e-9,
                                   e_field_v_m: float = 1.0e5,
                                   velocity_m_s: float = 1.0e6,
                                   b_field_t: float = 0.5) -> dict[str, Any]:
    """Electromagnetic forces and wave energetics: Coulomb force between two
    charges, magnetic (qv x B) and full Lorentz force magnitudes, and the
    time-averaged energy density and intensity of a plane EM wave.
    e.g. electromagnetic_force_analysis(1.6e-19, 1.6e-19, 1e-9)."""
    from sigma_ground.field import electrodynamics as E
    v_vec = (velocity_m_s, 0.0, 0.0)
    b_vec = (0.0, 0.0, b_field_t)
    e_vec = (e_field_v_m, 0.0, 0.0)
    results = {
        "coulomb_force_N": _safe(E.coulomb_force, charge_c, charge2_c, separation_m),
        "magnetic_force_N": _mag(_safe(E.magnetic_force, charge_c, v_vec, b_vec)),
        "lorentz_force_N": _mag(_safe(E.lorentz_force, charge_c, e_vec, v_vec, b_vec)),
        "em_energy_density_J_m3": _safe(E.em_wave_energy_density, e_field_v_m),
        "em_intensity_W_m2": _safe(E.em_wave_intensity, e_field_v_m),
    }
    return ToolResult(value=results, units="N, J/m^3, W/m^2", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="F=k q1 q2/r^2; F=q v x B; F=q(E+v x B); <u>=eps0 E0^2/2; I=eps0 c E0^2/2",
                      inputs={"charge_c": charge_c, "charge2_c": charge2_c,
                              "separation_m": separation_m, "e_field_v_m": e_field_v_m,
                              "velocity_m_s": velocity_m_s, "b_field_t": b_field_t}).to_dict()
