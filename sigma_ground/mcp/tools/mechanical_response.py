"""Mechanical-response analysis tools (standard physics).

Composite tools cascading through field.interface.{viscoelasticity, acoustics}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (viscoelasticity, acoustics)"


def _safe_call(fn, *a, **k):
    """Call a leaf function, returning (value, error_note) instead of a bare
    value-or-None. On success: (value, None). On failure: (None, note) where
    `note` is the exception's type and message (e.g. "KeyError:
    'unobtainium'") -- deliberately NOT fn-name-specific, so that two
    different leaf calls failing on the SAME underlying bad input (e.g. the
    same bad material_key) produce byte-identical notes and can be
    deduplicated by _notes_from_errors() below.

    Like the old _safe() this replaces, this never raises -- one bad
    sub-call only ever poisons its own (key, value) pair in the caller's
    results dict, so the caller always goes on to make the rest of the
    cascade's calls. The difference is purely that the reason for a failure
    is handed back instead of swallowed, so it can be threaded into
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
    _raw = {
        "maxwell_creep_strain": _safe_call(VE.maxwell_creep_strain, material_key,
                                     time_s, applied_stress, temperature_k),
        "kelvin_voigt_strain": _safe_call(VE.kelvin_voigt_creep, material_key,
                                    time_s, applied_stress, temperature_k),
        "sls_creep_strain": _safe_call(VE.sls_creep, material_key, time_s,
                                 applied_stress, temperature_k),
        "sls_relaxed_stress_Pa": _safe_call(VE.sls_stress_relaxation, material_key,
                                      time_s, initial_strain, temperature_k),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="dimensionless, Pa", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="Maxwell/Kelvin-Voigt/SLS creep; SLS stress relaxation",
                      notes=_notes_from_errors(errors),
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
    _raw = {
        "reflection_coefficient": _safe_call(AC.reflection_coefficient, material_key_1, material_key_2),
        "transmission_coefficient": _safe_call(AC.transmission_coefficient, material_key_1, material_key_2),
        "refraction_angle_deg": _safe_call(AC.snell_refraction_angle, material_key_1,
                                     material_key_2, incidence_angle_deg),
        "critical_angle_deg": _safe_call(AC.critical_angle, material_key_1, material_key_2),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="dimensionless, deg", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="R=(Z2-Z1)^2/(Z2+Z1)^2; T=1-R; Snell sin th/v; th_c=arcsin(v1/v2)",
                      notes=_notes_from_errors(errors),
                      inputs={"material_key_1": material_key_1,
                              "material_key_2": material_key_2,
                              "incidence_angle_deg": incidence_angle_deg}).to_dict()
