"""Special-relativity & atomic-spectra analysis tools (standard physics).

Composite tools cascading through field.relativity and
field.interface.atomic_spectra.
"""
from __future__ import annotations

import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_EV_J = 1.602176634e-19


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


def relativistic_energy_analysis(rest_mass_kg: float = 9.1093837015e-31,
                                 velocity_m_s: float = 2.6e8) -> dict[str, Any]:
    """Relativistic energy of a moving particle: rest energy (E0=m0 c^2),
    relativistic kinetic energy ((gamma-1) m0 c^2), and the energy-momentum
    invariant (m0 c^2)^2. Defaults: an electron at 0.867 c.
    e.g. relativistic_energy_analysis(9.109e-31, 2.6e8)."""
    from sigma_ground.field import relativity as R
    E0, err_E0 = _safe_call(R.rest_energy, rest_mass_kg)
    KE, err_KE = _safe_call(R.kinetic_energy_rel, rest_mass_kg, velocity_m_s)
    inv, err_inv = _safe_call(R.energy_momentum_invariant, rest_mass_kg)
    results = {
        "rest_energy_J": E0,
        "rest_energy_MeV": (E0 / _EV_J / 1.0e6) if E0 is not None else None,
        "kinetic_energy_J": KE,
        "kinetic_energy_MeV": (KE / _EV_J / 1.0e6) if KE is not None else None,
        "energy_momentum_invariant_J2": inv,
    }
    # rest_energy_MeV/kinetic_energy_MeV are pure arithmetic derivations of
    # E0/KE (not their own leaf call), so they share their source's error.
    errors = {k: e for k, e in {
        "rest_energy_J": err_E0,
        "rest_energy_MeV": err_E0,
        "kinetic_energy_J": err_KE,
        "kinetic_energy_MeV": err_KE,
        "energy_momentum_invariant_J2": err_inv,
    }.items() if e}
    return ToolResult(value=results, units="J, MeV, J^2",
                      source="sigma_ground.field.relativity",
                      provenance_tag="DERIVED",
                      formula="E0=m0 c^2; KE=(gamma-1) m0 c^2; E^2-(pc)^2=(m0 c^2)^2",
                      notes=_notes_from_errors(errors),
                      inputs={"rest_mass_kg": rest_mass_kg,
                              "velocity_m_s": velocity_m_s}).to_dict()


def zeeman_effect_analysis(total_angular_momentum_j: float = 1.0) -> dict[str, Any]:
    """Number of Zeeman sublevels a state of total angular momentum j splits
    into in a magnetic field (2j+1 values of m_j).
    e.g. zeeman_effect_analysis(1.0) -> 3 sublevels."""
    from sigma_ground.field.interface import atomic_spectra as A
    j = total_angular_momentum_j
    n, err = _safe_call(A.zeeman_splitting_count, j)
    try:
        echo = j if math.isfinite(float(j)) else repr(j)     # nan/inf aren't JSON
    except (TypeError, ValueError):
        echo = repr(j)
    if err:
        # j must be 0, 1/2, 1, 3/2, ...; it used to be int(2j + 1) of anything
        # (j = 1.3 -> "3 sublevels", j = -2 -> "-3")
        return ToolResult(value=None, units="dimensionless",
                          source="sigma_ground.field.interface.atomic_spectra",
                          provenance_tag="DERIVED",
                          notes=f"total_angular_momentum_j must be a non-negative integer "
                                f"or half-integer (0, 1/2, 1, 3/2, ...): {err}",
                          inputs={"total_angular_momentum_j": echo}).to_dict()
    return ToolResult(value={"zeeman_sublevels": n}, units="dimensionless",
                      source="sigma_ground.field.interface.atomic_spectra",
                      provenance_tag="DERIVED",
                      formula="number of m_j values = 2j + 1",
                      inputs={"total_angular_momentum_j": echo}).to_dict()
