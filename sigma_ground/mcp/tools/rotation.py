"""Rotational dynamics + atomic angular-momentum tools (standard physics).

Composite tools cascading through field.interface.{rotational,
angular_momentum}. Rolling-from-geometry uses a real Shape instance.
"""
from __future__ import annotations

import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (rotational, angular_momentum)"


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


def rotational_dynamics(mass_kg: float, radius_m: float, angle_deg: float = 30.0,
                        angular_velocity_rad_s: float = 10.0) -> dict[str, Any]:
    """Rotational dynamics bundle: rod moment of inertia, parallel-axis shift,
    angular momentum, torque & angular acceleration, plus rolling-down-a-ramp
    speed/distance/time and shape-geometry moment of inertia + rolling
    acceleration. e.g. rotational_dynamics(2, 0.5)."""
    from sigma_ground.field.interface import rotational as R
    try:
        from sigma_ground.shapes import Sphere
        sph = Sphere(radius_m)
    except Exception:
        sph = None
    ang = math.radians(angle_deg)
    I_rod, err_I_rod = _safe_call(R.moment_of_inertia_rod, mass_kg, 2.0 * radius_m)
    tau, err_tau = _safe_call(R.torque, 10.0, radius_m)
    _raw = {
        "moment_of_inertia_rod_kg_m2": (I_rod, err_I_rod),
        "parallel_axis_I_kg_m2": _safe_call(R.parallel_axis, I_rod if I_rod else 1.0, mass_kg, radius_m),
        "angular_momentum_kg_m2_s": _safe_call(R.angular_momentum, I_rod if I_rod else 1.0, angular_velocity_rad_s),
        "torque_Nm": (tau, err_tau),
        "angular_acceleration_rad_s2": _safe_call(R.angular_acceleration, tau if tau else 5.0, I_rod if I_rod else 1.0),
        "rolling_speed_from_1m_drop_m_s": _safe_call(R.rolling_speed_from_height, 1.0),
        "rolling_distance_on_flat_m": _safe_call(R.rolling_distance_on_flat, 5.0, 0.05),
        "rolling_time_on_flat_s": _safe_call(R.rolling_time_on_flat, 5.0, 0.05),
        "ramp_to_flat_distance_m": _safe_call(R.ramp_to_flat_distance, 1.0, ang, 0.05),
        "shape_moment_of_inertia_kg_m2": _safe_call(R.shape_moment_of_inertia, sph, mass_kg) if sph else (None, None),
        "shape_rolling_acceleration_m_s2": _safe_call(R.shape_rolling_acceleration, sph, ang) if sph else (None, None),
        "shape_rolling_speed_m_s": _safe_call(R.shape_rolling_speed_from_height, sph, 1.0) if sph else (None, None),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="kg.m^2, N.m, rad/s^2, m, m/s", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="I_rod=mL^2/12; L=Iw; tau=rF; alpha=tau/I; a_roll=g sinth/(1+I/mr^2)",
                      notes=_notes_from_errors(errors),
                      inputs={"mass_kg": mass_kg, "radius_m": radius_m,
                              "angle_deg": angle_deg,
                              "angular_velocity_rad_s": angular_velocity_rad_s}).to_dict()


# m_j is listed up to j = 50 (101 values); beyond that only the count and the
# range are reported. j = 3e8 used to enumerate 6e8 values and OOM the server.
_MAX_LISTED_MJ = 101


def _echo(x):
    """An input as JSON can carry it: nan/inf become their repr."""
    try:
        return x if math.isfinite(float(x)) else repr(x)
    except (TypeError, ValueError):
        return repr(x)


def atomic_angular_momentum(total_j: float = 1.5,
                            spin_orbit_constant_ev: float = 0.05) -> dict[str, Any]:
    """Atomic angular momentum: |J| magnitude and allowed m_j values, plus
    spin-orbit coupling energy/splitting and the Lande interval check for a
    term (L=2, S=1/2). e.g. atomic_angular_momentum(1.5).

    total_j must be a non-negative integer or half-integer. The spin-orbit
    energy exists only for the term's levels (J = 3/2 or 5/2); for any other J
    it is left out with a note rather than computed for a level that isn't
    there.
    """
    from sigma_ground.field.interface import angular_momentum as AM
    L, S = 2, 0.5
    inputs = {"total_j": _echo(total_j), "spin_orbit_constant_ev": _echo(spin_orbit_constant_ev)}

    def decline(note):
        return ToolResult(value=None, units="J.s / eV", source=_SRC,
                          provenance_tag="DERIVED", notes=note, inputs=inputs).to_dict()

    try:
        j = AM.check_quantum_number(total_j)
    except (TypeError, ValueError) as e:
        return decline(f"total_j must be a non-negative integer or half-integer "
                       f"(0, 1/2, 1, 3/2, ...): {e}")
    try:
        a_ev = float(spin_orbit_constant_ev)
    except (TypeError, ValueError):
        a_ev = float("nan")
    if not math.isfinite(a_ev):
        return decline(f"spin_orbit_constant_ev must be a finite number; got {spin_orbit_constant_ev!r}")

    n_states = AM.multiplicity(j)
    levels = AM.allowed_J_values(L, S)
    is_level = any(abs(j - J) < 1e-9 for J in levels)
    _raw = {
        "magnitude_J_s": _safe_call(AM.angular_momentum_magnitude, j),
        "m_j_values": (_safe_call(AM.angular_momentum_z_values, j)
                       if n_states <= _MAX_LISTED_MJ else (None, None)),
        "spin_orbit_energy_eV": (_safe_call(AM.spin_orbit_energy_eV, a_ev, L, S, j)
                                 if is_level else (None, None)),
        "spin_orbit_splitting_eV": _safe_call(AM.spin_orbit_splitting_eV, a_ev, L, S),
        "lande_interval_check": _safe_call(AM.lande_interval_check, a_ev, L, S),
        "spin_expectation_z": _safe_call(AM.spin_expectation, 0.6, 0.8),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    results["multiplicity"] = n_states
    errors = {k: e for k, (v, e) in _raw.items() if e}
    notes = [n for n in [_notes_from_errors(errors)] if n]
    if n_states > _MAX_LISTED_MJ:
        notes.append(f"m_j runs from {-j:g} to {j:g} in steps of 1: {n_states} values, "
                     f"not listed above {_MAX_LISTED_MJ}")
    if not is_level:
        notes.append(f"J = {j:g} is not a level of the term (L=2, S=1/2 has J = "
                     f"{' or '.join(f'{J:g}' for J in levels)}), so spin_orbit_energy_eV "
                     f"is not computed")
    return ToolResult(value=results, units="J.s / eV", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="|J|=hbar sqrt(j(j+1)); E_SO=(A/2)[J(J+1)-L(L+1)-S(S+1)]",
                      notes=" | ".join(notes), inputs=inputs).to_dict()
