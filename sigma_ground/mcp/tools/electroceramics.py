"""Electroceramics analysis tools (standard dielectric & piezoelectric physics).

Composite tools cascading through field.interface.{piezoelectricity, dielectric}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (piezoelectricity, dielectric)"


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


def piezoelectric_actuator_analysis(material_key: str = "PZT5A",
                                    e_field_v_m: float = 1.0e6,
                                    length_m: float = 0.02) -> dict[str, Any]:
    """Converse piezoelectric effect of an actuator under an applied field:
    induced strain (eps=d E) and tip displacement (delta=d E L).
    Keys: quartz, PZT4, PZT5A, BaTiO3, LiNbO3, AlN, PVDF.
    e.g. piezoelectric_actuator_analysis('PZT5A', 1e6, 0.02)."""
    from sigma_ground.field.interface import piezoelectricity as P
    _raw = {
        "strain": _safe_call(P.piezoelectric_strain, material_key, e_field_v_m),
        "displacement_m": _safe_call(P.piezoelectric_displacement, material_key, e_field_v_m, length_m),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="dimensionless, m", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="strain=d*E; displacement=d*E*L (converse piezoelectric effect)",
                      notes=_notes_from_errors(errors),
                      inputs={"material_key": material_key,
                              "e_field_v_m": e_field_v_m,
                              "length_m": length_m}).to_dict()


def dielectric_polarization_analysis(polarizability_fm2: float = 1.6e-40,
                                     number_density_m3: float = 3.0e28) -> dict[str, Any]:
    """Relative permittivity of a non-polar dielectric from the Clausius-
    Mossotti relation, given molecular polarizability (SI, F.m^2) and number
    density. e.g. dielectric_polarization_analysis(1.6e-40, 3e28)."""
    from sigma_ground.field.interface import dielectric as D
    eps_r, err = _safe_call(D.clausius_mossotti, polarizability_fm2, number_density_m3)
    results = {"relative_permittivity": eps_r}
    errors = {"relative_permittivity": err} if err else {}
    return ToolResult(value=results, units="dimensionless", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="(eps_r-1)/(eps_r+2) = N alpha / (3 eps0)  (Clausius-Mossotti)",
                      notes=_notes_from_errors(errors),
                      inputs={"polarizability_fm2": polarizability_fm2,
                              "number_density_m3": number_density_m3}).to_dict()
