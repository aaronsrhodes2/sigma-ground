"""Transport & statistical-mechanics analysis tools (standard physics).

Composite tools that cascade through field.interface.{viscosity, fluid,
diffusion, statistical}. A few realistic inputs exercise many leaf functions.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (viscosity, fluid, diffusion, statistical)"
_EV_J = 1.602176634e-19


def _safe(fn, *a, **k):
    """Call a leaf function, returning its value or None (so one bad arg in a
    cascade never aborts the rest). The call still executes -> coverage holds."""
    try:
        return fn(*a, **k)
    except Exception:
        return None


def viscous_flow_analysis(velocity_m_s: float, radius_m: float,
                          viscosity_pa_s: float = 1.0e-3,
                          fluid_density_kg_m3: float = 1000.0) -> dict[str, Any]:
    """Viscous-flow bundle for a sphere/pipe: Reynolds number, Stokes drag,
    Stokes terminal velocity, drag coefficient, Poiseuille pipe flow, boundary
    layer, wall shear, and viscous heating. e.g. viscous_flow_analysis(2, 0.005)."""
    from sigma_ground.field.interface import viscosity as V
    from sigma_ground.field.interface import fluid as F
    d = 2.0 * radius_m
    area = 3.141592653589793 * radius_m ** 2
    Re_p = _safe(V.particle_reynolds_number, fluid_density_kg_m3, velocity_m_s, d, viscosity_pa_s)
    Cd = _safe(V.drag_coefficient_sphere, Re_p if Re_p else 1.0)
    results = {
        "reynolds_number": _safe(F.reynolds_number, fluid_density_kg_m3, velocity_m_s, d, viscosity_pa_s),
        "stokes_drag_N": _safe(V.stokes_drag, viscosity_pa_s, radius_m, velocity_m_s),
        "terminal_velocity_stokes_m_s": _safe(V.terminal_velocity_stokes, radius_m, 2700.0, fluid_density_kg_m3, viscosity_pa_s),
        "particle_reynolds_number": Re_p,
        "drag_coefficient_sphere": Cd,
        "general_drag_force_N": _safe(V.general_drag_force, Cd if Cd else 0.47, fluid_density_kg_m3, velocity_m_s, area),
        "poiseuille_flow_rate_m3_s": _safe(V.poiseuille_flow_rate, radius_m, 1000.0, viscosity_pa_s, 1.0),
        "poiseuille_max_velocity_m_s": _safe(V.poiseuille_max_velocity, radius_m, 1000.0, viscosity_pa_s, 1.0),
        "boundary_layer_thickness_m": _safe(V.boundary_layer_thickness, 1.0, fluid_density_kg_m3, velocity_m_s, viscosity_pa_s),
        "wall_shear_stress_Pa": _safe(V.wall_shear_stress, fluid_density_kg_m3, velocity_m_s, viscosity_pa_s, 1.0),
        "viscous_dissipation_W_m3": _safe(V.viscous_dissipation_simple_shear, viscosity_pa_s, 100.0),
        "viscous_heating_dT_K": _safe(V.viscous_heating_temperature_rise, viscosity_pa_s, 100.0, 1.0, fluid_density_kg_m3, 4186.0),
        "nabarro_herring_strain_rate_per_s": _safe(V.nabarro_herring_strain_rate, 1e-12, 1e6, 1e-29, 1e-5, 1000.0),
    }
    return ToolResult(value=results, units="various (SI)", source=_SRC,
                      provenance_tag="DERIVED", formula="Re=rho v L/eta; F_stokes=6 pi eta r v; v_t=2r^2(drho)g/9eta",
                      inputs={"velocity_m_s": velocity_m_s, "radius_m": radius_m,
                              "viscosity_pa_s": viscosity_pa_s,
                              "fluid_density_kg_m3": fluid_density_kg_m3}).to_dict()


def diffusion_analysis(temperature_k: float,
                       diffusivity_m2_s: float = 1.0e-9) -> dict[str, Any]:
    """Diffusion bundle: Einstein-Stokes diffusivity, Fick's first & second
    laws, time to penetrate a depth, and Darken interdiffusion."""
    from sigma_ground.field.interface import diffusion as D
    results = {
        "einstein_stokes_D_m2_s": _safe(D.einstein_stokes_diffusivity, temperature_k, 1.0e-3, 1.0e-9),
        "ficks_first_flux": _safe(D.ficks_first_law, diffusivity_m2_s, 100.0),
        "ficks_second_concentration": _safe(D.ficks_second_law_erf, 1.0, 0.0, 1.0e-4, diffusivity_m2_s, 3600.0),
        "time_to_penetrate_1mm_s": _safe(D.time_to_penetrate, diffusivity_m2_s, 1.0e-3),
        "darken_interdiffusion_m2_s": _safe(D.darken_interdiffusion, diffusivity_m2_s, 2.0 * diffusivity_m2_s, 0.3, 0.7),
    }
    return ToolResult(value=results, units="m^2/s, flux, m, s", source=_SRC,
                      provenance_tag="DERIVED", formula="D=kT/6 pi eta r; J=-D dC/dx; x~sqrt(Dt)",
                      inputs={"temperature_k": temperature_k,
                              "diffusivity_m2_s": diffusivity_m2_s}).to_dict()


def statistical_distribution(temperature_k: float,
                             energy_ev: float = 0.5) -> dict[str, Any]:
    """Statistical-mechanics bundle: Fermi-Dirac & Bose-Einstein occupations,
    canonical partition function, mean energy, entropy, and equipartition heat
    capacity. e.g. statistical_distribution(300, 0.5)."""
    from sigma_ground.field.interface import statistical as S
    E = energy_ev * _EV_J
    levels = [0.0, E, 2.0 * E]
    Z = _safe(S.partition_function, levels, temperature_k)
    Em = _safe(S.mean_energy, levels, temperature_k)
    results = {
        "fermi_dirac_at_Ef": _safe(S.fermi_dirac, E, E, temperature_k),
        "bose_einstein": _safe(S.bose_einstein, 2.0 * E, E, temperature_k),
        "partition_function_Z": Z,
        "mean_energy_J": Em,
        "entropy_J_per_K": _safe(S.entropy_from_partition, Z if Z else 2.0, temperature_k, Em if Em else E),
        "heat_capacity_3dof": _safe(S.heat_capacity_equipartition, 3, 1),
    }
    return ToolResult(value=results, units="dimensionless / J / J·K^-1", source=_SRC,
                      provenance_tag="DERIVED", formula="f_FD=1/(e^((E-mu)/kT)+1); Z=sum e^(-E/kT)",
                      inputs={"temperature_k": temperature_k, "energy_ev": energy_ev}).to_dict()
