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


def _safe_call(fn, *a, **k):
    """Call a leaf function, returning (value, error_note) instead of a bare
    value-or-None. On success: (value, None). On failure: (None, note) where
    `note` is the exception's type and message (e.g. "KeyError:
    'unobtainium'") -- deliberately NOT fn-name-specific, so that two
    different leaf calls failing on the SAME underlying bad input (e.g. the
    same bad material_key) produce byte-identical notes and can be
    deduplicated by _notes_from_errors() below.

    Like _safe() above, this never raises -- one bad sub-call only ever
    poisons its own (key, value) pair in the caller's results dict, so the
    caller always goes on to make the rest of the cascade's calls. The
    difference from _safe() is purely that the reason for a failure is
    handed back instead of swallowed, so it can be threaded into
    ToolResult(notes=...) rather than surfacing a bare `None` with nothing
    explaining why."""
    try:
        return fn(*a, **k), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _notes_from_errors(errors: dict[str, str]) -> str:
    """Fold a {result_key: error_note} map (only the FAILED keys) into one
    ToolResult.notes string.

    Groups result keys whose error_note is byte-identical -- the signature
    of a shared root cause, e.g. N calls in one cascade that all take the
    same bad material_key -- so the explanation is stated once per DISTINCT
    cause (naming every result key it took down), not once per affected
    key. Keys with distinct causes each keep their own clause. Returns ""
    when `errors` is empty (the all-success case), matching ToolResult's
    default notes="".
    """
    if not errors:
        return ""
    groups: dict[str, list[str]] = {}
    for key, note in errors.items():
        groups.setdefault(note, []).append(key)
    return " | ".join(f"{', '.join(keys)}: {note}" for note, keys in groups.items())


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
    Re_p, err_Re_p = _safe_call(V.particle_reynolds_number, fluid_density_kg_m3, velocity_m_s, d, viscosity_pa_s)
    Cd, err_Cd = _safe_call(V.drag_coefficient_sphere, Re_p if Re_p else 1.0)
    _raw = {
        "reynolds_number": _safe_call(F.reynolds_number, fluid_density_kg_m3, velocity_m_s, d, viscosity_pa_s),
        "stokes_drag_N": _safe_call(V.stokes_drag, viscosity_pa_s, radius_m, velocity_m_s),
        "terminal_velocity_stokes_m_s": _safe_call(V.terminal_velocity_stokes, radius_m, 2700.0, fluid_density_kg_m3, viscosity_pa_s),
        "particle_reynolds_number": (Re_p, err_Re_p),
        "drag_coefficient_sphere": (Cd, err_Cd),
        "general_drag_force_N": _safe_call(V.general_drag_force, Cd if Cd else 0.47, fluid_density_kg_m3, velocity_m_s, area),
        "poiseuille_flow_rate_m3_s": _safe_call(V.poiseuille_flow_rate, radius_m, 1000.0, viscosity_pa_s, 1.0),
        "poiseuille_max_velocity_m_s": _safe_call(V.poiseuille_max_velocity, radius_m, 1000.0, viscosity_pa_s, 1.0),
        "boundary_layer_thickness_m": _safe_call(V.boundary_layer_thickness, 1.0, fluid_density_kg_m3, velocity_m_s, viscosity_pa_s),
        "wall_shear_stress_Pa": _safe_call(V.wall_shear_stress, fluid_density_kg_m3, velocity_m_s, viscosity_pa_s, 1.0),
        "viscous_dissipation_W_m3": _safe_call(V.viscous_dissipation_simple_shear, viscosity_pa_s, 100.0),
        "viscous_heating_dT_K": _safe_call(V.viscous_heating_temperature_rise, viscosity_pa_s, 100.0, 1.0, fluid_density_kg_m3, 4186.0),
        "nabarro_herring_strain_rate_per_s": _safe_call(V.nabarro_herring_strain_rate, 1e-12, 1e6, 1e-29, 1e-5, 1000.0),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="various (SI)", source=_SRC,
                      provenance_tag="DERIVED", formula="Re=rho v L/eta; F_stokes=6 pi eta r v; v_t=2r^2(drho)g/9eta",
                      notes=_notes_from_errors(errors),
                      inputs={"velocity_m_s": velocity_m_s, "radius_m": radius_m,
                              "viscosity_pa_s": viscosity_pa_s,
                              "fluid_density_kg_m3": fluid_density_kg_m3}).to_dict()


def diffusion_analysis(temperature_k: float,
                       diffusivity_m2_s: float = 1.0e-9) -> dict[str, Any]:
    """Diffusion bundle: Einstein-Stokes diffusivity, Fick's first & second
    laws, time to penetrate a depth, and Darken interdiffusion."""
    from sigma_ground.field.interface import diffusion as D
    _raw = {
        "einstein_stokes_D_m2_s": _safe_call(D.einstein_stokes_diffusivity, temperature_k, 1.0e-3, 1.0e-9),
        "ficks_first_flux": _safe_call(D.ficks_first_law, diffusivity_m2_s, 100.0),
        "ficks_second_concentration": _safe_call(D.ficks_second_law_erf, 1.0, 0.0, 1.0e-4, diffusivity_m2_s, 3600.0),
        "time_to_penetrate_1mm_s": _safe_call(D.time_to_penetrate, diffusivity_m2_s, 1.0e-3),
        "darken_interdiffusion_m2_s": _safe_call(D.darken_interdiffusion, diffusivity_m2_s, 2.0 * diffusivity_m2_s, 0.3, 0.7),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="m^2/s, flux, m, s", source=_SRC,
                      provenance_tag="DERIVED", formula="D=kT/6 pi eta r; J=-D dC/dx; x~sqrt(Dt)",
                      notes=_notes_from_errors(errors),
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
    Z, err_Z = _safe_call(S.partition_function, levels, temperature_k)
    Em, err_Em = _safe_call(S.mean_energy, levels, temperature_k)
    _raw = {
        "fermi_dirac_at_Ef": _safe_call(S.fermi_dirac, E, E, temperature_k),
        "bose_einstein": _safe_call(S.bose_einstein, 2.0 * E, E, temperature_k),
        "partition_function_Z": (Z, err_Z),
        "mean_energy_J": (Em, err_Em),
        "entropy_J_per_K": _safe_call(S.entropy_from_partition, Z if Z else 2.0, temperature_k, Em if Em else E),
        "heat_capacity_3dof": _safe_call(S.heat_capacity_equipartition, 3, 1),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="dimensionless / J / J·K^-1", source=_SRC,
                      provenance_tag="DERIVED", formula="f_FD=1/(e^((E-mu)/kT)+1); Z=sum e^(-E/kT)",
                      notes=_notes_from_errors(errors),
                      inputs={"temperature_k": temperature_k, "energy_ev": energy_ev}).to_dict()
