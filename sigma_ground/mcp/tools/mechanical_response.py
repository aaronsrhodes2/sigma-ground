"""Mechanical-response analysis tools (standard physics).

Composite tools cascading through field.interface.{viscoelasticity, acoustics}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (viscoelasticity, acoustics)"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def viscoelastic_creep_analysis(material_key: str = "copper",
                                time_s: float = 3600.0,
                                applied_stress: float = 5.0e7,
                                temperature_k: float = 400.0,
                                initial_strain: float = 0.001) -> dict[str, Any]:
    """Viscoelastic creep & relaxation under load: Maxwell creep strain,
    Kelvin-Voigt retarded strain, standard-linear-solid (SLS) creep, and SLS
    stress relaxation from an initial strain.
    e.g. viscoelastic_creep_analysis('copper', 3600, 5e7, 400)."""
    from sigma_ground.field.interface import viscoelasticity as VE
    results = {
        "maxwell_creep_strain": _safe(VE.maxwell_creep_strain, material_key,
                                     time_s, applied_stress, temperature_k),
        "kelvin_voigt_strain": _safe(VE.kelvin_voigt_creep, material_key,
                                    time_s, applied_stress, temperature_k),
        "sls_creep_strain": _safe(VE.sls_creep, material_key, time_s,
                                 applied_stress, temperature_k),
        "sls_relaxed_stress_Pa": _safe(VE.sls_stress_relaxation, material_key,
                                      time_s, initial_strain, temperature_k),
    }
    return ToolResult(value=results, units="dimensionless, Pa", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="Maxwell/Kelvin-Voigt/SLS creep; SLS stress relaxation",
                      inputs={"material_key": material_key, "time_s": time_s,
                              "applied_stress": applied_stress,
                              "temperature_k": temperature_k,
                              "initial_strain": initial_strain}).to_dict()


def acoustic_interface_analysis(material_key_1: str = "lead",
                                material_key_2: str = "aluminum",
                                incidence_angle_deg: float = 10.0) -> dict[str, Any]:
    """Sound at a planar interface between two media: energy reflection &
    transmission coefficients (normal incidence), Snell refraction angle, and
    the critical angle for total internal reflection (when v1 < v2).
    e.g. acoustic_interface_analysis('lead', 'aluminum', 10)."""
    from sigma_ground.field.interface import acoustics as AC
    results = {
        "reflection_coefficient": _safe(AC.reflection_coefficient, material_key_1, material_key_2),
        "transmission_coefficient": _safe(AC.transmission_coefficient, material_key_1, material_key_2),
        "refraction_angle_deg": _safe(AC.snell_refraction_angle, material_key_1,
                                     material_key_2, incidence_angle_deg),
        "critical_angle_deg": _safe(AC.critical_angle, material_key_1, material_key_2),
    }
    return ToolResult(value=results, units="dimensionless, deg", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="R=(Z2-Z1)^2/(Z2+Z1)^2; T=1-R; Snell sin th/v; th_c=arcsin(v1/v2)",
                      inputs={"material_key_1": material_key_1,
                              "material_key_2": material_key_2,
                              "incidence_angle_deg": incidence_angle_deg}).to_dict()
