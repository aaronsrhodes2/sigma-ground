"""Materials strength & composites analysis tools (standard physics).

Composite tools cascading through field.interface.{elasticity, stress,
plasticity, composites}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (elasticity, stress, plasticity, composites)"


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


def elastic_analysis(material_key: str, strain: float = 0.001) -> dict[str, Any]:
    """Elastic response of a material under a strain: uniaxial / shear /
    hydrostatic stress, the three strain-energy densities, transverse strain,
    volume change, a von Mises yield check, and isotropic moduli from Lame
    parameters. e.g. elastic_analysis('iron', 0.001)."""
    from sigma_ground.field.interface import elasticity as E
    _raw = {
        "uniaxial_stress_Pa": _safe_call(E.uniaxial_stress, material_key, strain),
        "shear_stress_Pa": _safe_call(E.shear_stress, material_key, strain),
        "hydrostatic_stress_Pa": _safe_call(E.hydrostatic_stress, material_key, strain),
        "transverse_strain": _safe_call(E.transverse_strain, material_key, strain),
        "volume_change": _safe_call(E.volume_change_uniaxial, material_key, strain),
        "strain_energy_uniaxial_J_m3": _safe_call(E.strain_energy_density_uniaxial, material_key, strain),
        "strain_energy_shear_J_m3": _safe_call(E.strain_energy_density_shear, material_key, strain),
        "strain_energy_hydrostatic_J_m3": _safe_call(E.strain_energy_density_hydrostatic, material_key, strain),
        # PHYSICS_GAP (2026-10-10, unification round 2, U8; KNOWN_GAPS.md): the next two
        # entries ignore material_key — fixed Lamé parameters and a fixed 250 MPa yield, so
        # every material reports K = 70 GPa, E = 78.75 GPa, G = 30 GPa. Flagged, not fixed.
        "is_yielded_at_200MPa": _safe_call(E.is_yielded, 200.0e6, 0.0, 0.0, 250.0e6),
        "moduli_from_lame": _safe_call(E.moduli_from_lame, 50.0e9, 30.0e9),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="Pa, J/m^3", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="sigma=E eps; tau=G gamma; U=0.5 sigma eps; von Mises yield",
                      notes=_notes_from_errors(errors),
                      inputs={"material_key": material_key, "strain": strain}).to_dict()


def stress_failure_analysis(material_key: str,
                            applied_stress: float = 1.0e8) -> dict[str, Any]:
    """Fracture / fatigue / creep: Mode-I stress-intensity factor, critical
    crack length, fatigue life (Basquin S-N), Paris remaining life, and creep
    strain rate + rupture time at 800 K. e.g. stress_failure_analysis('iron', 1e8)."""
    from sigma_ground.field.interface import stress as S
    _raw = {
        "stress_intensity_Pa_sqrt_m": _safe_call(S.stress_intensity, applied_stress, 0.001),
        "critical_crack_length_m": _safe_call(S.critical_crack_length, material_key, applied_stress),
        "fatigue_life_cycles": _safe_call(S.fatigue_life, material_key, applied_stress),
        "paris_remaining_life_cycles": _safe_call(S.paris_remaining_life, material_key, 1.0e-4, 1.0e-3, applied_stress),
        "creep_strain_rate_per_s": _safe_call(S.creep_strain_rate, material_key, applied_stress, 800.0),
        "creep_rupture_time_s": _safe_call(S.creep_rupture_time, material_key, applied_stress, 800.0),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="Pa.m^0.5, m, cycles, 1/s, s", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="K_I=sigma sqrt(pi a); Basquin S-N; Paris da/dN; creep",
                      notes=_notes_from_errors(errors),
                      inputs={"material_key": material_key,
                              "applied_stress": applied_stress}).to_dict()


def plasticity_analysis(material_key: str,
                        plastic_strain: float = 0.05) -> dict[str, Any]:
    """Plastic flow stress at a given plastic strain: Johnson-Cook, Ludwik
    hardening law, and the instantaneous work-hardening rate.
    e.g. plasticity_analysis('aluminum', 0.05)."""
    from sigma_ground.field.interface import plasticity as P
    _raw = {
        "johnson_cook_stress_Pa": _safe_call(P.johnson_cook_stress, material_key, plastic_strain),
        "ludwik_stress_Pa": _safe_call(P.ludwik_stress, material_key, plastic_strain),
        "work_hardening_rate_Pa": _safe_call(P.work_hardening_rate, material_key, plastic_strain),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="Pa", source=_SRC, provenance_tag="DERIVED",
                      formula="Johnson-Cook & Ludwik flow stress; dsigma/d eps_p",
                      notes=_notes_from_errors(errors),
                      inputs={"material_key": material_key,
                              "plastic_strain": plastic_strain}).to_dict()


def composite_bounds_analysis(bulk_modulus1_pa: float = 100.0e9,
                              bulk_modulus2_pa: float = 200.0e9,
                              fraction1: float = 0.5) -> dict[str, Any]:
    """Two-phase composite property bounds: Voigt-Reuss-Hill average,
    Hashin-Shtrikman bounds, thermal-conductivity bounds, and Gibson-Ashby
    foam strength. e.g. composite_bounds_analysis(100e9, 200e9, 0.5)."""
    from sigma_ground.field.interface import composites as C
    f2 = 1.0 - fraction1
    _raw = {
        "voigt_reuss_hill_Pa": _safe_call(C.voigt_reuss_hill, [bulk_modulus1_pa, bulk_modulus2_pa], [fraction1, f2]),
        "hashin_shtrikman_bounds": _safe_call(C.hashin_shtrikman_bounds, bulk_modulus1_pa, 0.4 * bulk_modulus1_pa, fraction1, bulk_modulus2_pa, 0.4 * bulk_modulus2_pa),
        "thermal_conductivity_bounds": _safe_call(C.thermal_conductivity_bounds, [100.0, 400.0], [fraction1, f2]),
        "gibson_ashby_foam_strength_Pa": _safe_call(C.gibson_ashby_strength, 200.0e6, 0.3),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="Pa, W/(m.K)", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="Voigt/Reuss/Hill; Hashin-Shtrikman; Gibson-Ashby foam",
                      notes=_notes_from_errors(errors),
                      inputs={"bulk_modulus1_pa": bulk_modulus1_pa,
                              "bulk_modulus2_pa": bulk_modulus2_pa,
                              "fraction1": fraction1}).to_dict()
