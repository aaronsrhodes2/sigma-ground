"""Assorted physics analysis tools (standard physics).

Composite tools for the remaining leaf functions across field.asteroids and
field.interface.{mobius, impact, cosmology}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult


def _safe_call(fn, *a, **k):
    """Call a leaf function, returning (value, error_note) instead of a bare
    value-or-None. On success: (value, None). On failure: (None, note) where
    `note` is the exception's type and message (e.g. "KeyError:
    'unobtainium'") -- deliberately NOT fn-name-specific, so that two
    different leaf calls failing on the SAME underlying bad input (e.g. the
    same bad material_key) produce byte-identical notes and can be
    deduplicated by _notes_from_errors() below.

    One bad sub-call only ever poisons its own (key, value) pair in the
    caller's results dict, so the caller always goes on to make the rest of
    the cascade's calls -- it never raises."""
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


def asteroid_analysis(body_key: str = "ceres") -> dict[str, Any]:
    """Small-body geophysics from a tri-axial ellipsoid model: surface gravity
    (per axis + mean), escape velocity, and shape (axis ratios, oblateness,
    near-spherical / contact-binary flags). Bodies: bennu, ryugu, itokawa, eros,
    vesta, ceres. e.g. asteroid_analysis('ceres')."""
    from sigma_ground.field import asteroids as A
    body = A.ALL_ASTEROIDS.get(body_key)
    g, err_g = _safe_call(A.surface_gravity, body)
    v, err_v = _safe_call(A.escape_velocity, body)
    shape, err_shape = _safe_call(A.axis_ratios, body)
    g = g or {}
    v = v or {}
    shape = shape or {}
    results = {
        "surface_gravity_mean_m_s2": g.get("g_mean") if isinstance(g, dict) else None,
        "escape_velocity_m_s": v.get("v_a") if isinstance(v, dict) else None,
        "c_over_a": shape.get("c_over_a") if isinstance(shape, dict) else None,
        "oblateness": shape.get("oblateness") if isinstance(shape, dict) else None,
        "is_nearly_spherical": shape.get("is_nearly_spherical") if isinstance(shape, dict) else None,
    }
    # Each derived key inherits the error of the single leaf call it comes
    # from, so three keys sharing the same failed axis_ratios() call dedupe
    # into one _notes_from_errors() clause naming all three.
    errors = {k: e for k, e in {
        "surface_gravity_mean_m_s2": err_g,
        "escape_velocity_m_s": err_v,
        "c_over_a": err_shape,
        "oblateness": err_shape,
        "is_nearly_spherical": err_shape,
    }.items() if e}
    return ToolResult(value=results, units="m/s^2, m/s, dimensionless",
                      source="sigma_ground.field.asteroids",
                      provenance_tag="DERIVED",
                      formula="g=GM/r^2 per axis; v_esc=sqrt(2GM/r); axis ratios from a,b,c",
                      notes=_notes_from_errors(errors),
                      inputs={"body_key": body_key}).to_dict()


def mobius_bimetallic_analysis(mat_a: str = "copper", mat_b: str = "iron",
                               loop_length_m: float = 0.1, width_m: float = 0.01,
                               thickness_m: float = 0.001,
                               t_hot: float = 400.0, t_cold: float = 300.0) -> dict[str, Any]:
    """Bimetallic strip / Mobius loop: total series resistance of the two-metal
    loop and the thermoelectric (Seebeck) voltage across a hot-cold gradient.
    e.g. mobius_bimetallic_analysis('copper', 'iron', 0.1, 0.01, 0.001)."""
    from sigma_ground.field.interface import mobius as MO
    _raw = {
        "total_resistance_ohm": _safe_call(MO.mobius_total_resistance, mat_a, mat_b,
                                           loop_length_m, width_m, thickness_m),
        "bimetallic_seebeck_voltage_V": _safe_call(MO.bimetallic_seebeck_voltage, mat_a, mat_b,
                                                   t_hot, t_cold),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="ohm, V",
                      source="sigma_ground.field.interface.mobius",
                      provenance_tag="DERIVED",
                      formula="R=rho L/A (series); V=(S_A-S_B)(T_hot-T_cold)",
                      notes=_notes_from_errors(errors),
                      inputs={"mat_a": mat_a, "mat_b": mat_b,
                              "loop_length_m": loop_length_m, "width_m": width_m,
                              "thickness_m": thickness_m,
                              "t_hot": t_hot, "t_cold": t_cold}).to_dict()


def hertzian_impact_analysis(e1_pa: float = 200.0e9, nu1: float = 0.3,
                             yield1_pa: float = 250.0e6, density1: float = 7850.0,
                             e2_pa: float = 200.0e9, nu2: float = 0.3,
                             yield2_pa: float = 250.0e6, density2: float = 7850.0,
                             velocity_m_s: float = 1.0) -> dict[str, Any]:
    """Hertzian contact impact of two bodies: the reduced (effective) elastic
    modulus and the coefficient of restitution for the pair (velocity-dependent,
    elastic-plastic). Default is steel-on-steel at 1 m/s.
    e.g. hertzian_impact_analysis(200e9, 0.3, 250e6, 7850, ...)."""
    from sigma_ground.field.interface import impact as IM
    _raw = {
        "reduced_modulus_Pa": _safe_call(IM.reduced_modulus_pair, e1_pa, nu1, e2_pa, nu2),
        "coefficient_of_restitution": _safe_call(IM.cor_pair, e1_pa, nu1, yield1_pa, density1,
                                                 e2_pa, nu2, yield2_pa, density2, velocity_m_s),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="Pa, dimensionless",
                      source="sigma_ground.field.interface.impact",
                      provenance_tag="DERIVED",
                      formula="1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2; COR(v) elastic-plastic",
                      notes=" | ".join(filter(None, [
                          "Coefficient of restitution decreases with impact velocity.",
                          _notes_from_errors(errors)])),
                      inputs={"e1_pa": e1_pa, "nu1": nu1, "yield1_pa": yield1_pa,
                              "e2_pa": e2_pa, "nu2": nu2, "yield2_pa": yield2_pa,
                              "velocity_m_s": velocity_m_s}).to_dict()


def holographic_dark_energy_analysis(rho_de_J_m3: float = 6.0e-10) -> dict[str, Any]:
    """Holographic dark-energy parameter c^2 implied by an observed dark-energy
    density with the Hubble-radius IR cutoff (inverts rho_DE = 3 c^2 C^4 /
    (8 pi G L^2)). rho_DE ~ 6e-10 J/m^3 -> c^2 ~ 0.7-0.8 (DESI-consistent).
    e.g. holographic_dark_energy_analysis(6e-10)."""
    from sigma_ground.field.interface import cosmology as CO
    c2, err = _safe_call(CO.hde_c_squared_given_rho_de, rho_de_J_m3)
    results = {"hde_c_squared": c2}
    errors = {"hde_c_squared": err} if err else {}
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface.cosmology",
                      provenance_tag="DERIVED",
                      formula="c^2 = rho_DE / (3 C^4 / (8 pi G L^2)),  L = Hubble radius",
                      notes=_notes_from_errors(errors),
                      inputs={"rho_de_J_m3": rho_de_J_m3}).to_dict()
