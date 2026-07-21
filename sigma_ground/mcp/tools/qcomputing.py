"""Quantum-computing analysis tools (standard quantum information).

Composite tools cascading through field.interface.{quantum_algorithms,
quantum_output, quantum_computing, quantum}. High-level algorithms and
measurement capabilities only -- individual gate-matrix primitives
(gate_rx/rz/s/t/y/cz/iswap/fredkin/phase) are NOT exposed (building blocks,
covered indirectly by the algorithms). See misc/COVERAGE_LEDGER.md.
"""
from __future__ import annotations

import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_R = 2.0 ** -0.5  # 1/sqrt(2)


def _safe_call(fn, *a, **k):
    """Call a leaf function, returning (value, error_note) instead of a bare
    value-or-None. On success: (value, None). On failure: (None, note) where
    `note` is the exception's type and message (e.g. "KeyError:
    'unobtainium'") -- deliberately NOT fn-name-specific, so that two
    different leaf calls failing on the SAME underlying bad input produce
    byte-identical notes and can be deduplicated by _notes_from_errors()
    below.

    Never raises -- one bad sub-call only ever poisons its own (key, value)
    pair in the caller's results dict, so the caller always goes on to make
    the rest of the cascade's calls. The reason for a failure is handed back
    instead of swallowed, so it can be threaded into ToolResult(notes=...)
    rather than surfacing a bare `None` with nothing explaining why."""
    try:
        return fn(*a, **k), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _notes_from_errors(errors: dict[str, str]) -> str:
    """Fold a {result_key: error_note} map (only the FAILED keys) into one
    ToolResult.notes string.

    Groups result keys whose error_note is byte-identical -- the signature
    of a shared root cause -- so the explanation is stated once per DISTINCT
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


def quantum_algorithm_analysis(grover_n_qubits: int = 3,
                               grover_marked_item: int = 5,
                               qaoa_edges: list[list[int]] | None = None,
                               qaoa_n_nodes: int = 3,
                               simon_hidden_string: str = "11") -> dict[str, Any]:
    """Run three canonical quantum algorithms: Grover search (find a marked item
    in 2^n), QAOA Max-Cut (combinatorial optimisation), and Simon's algorithm
    (recover a hidden period). e.g. quantum_algorithm_analysis(3, 5)."""
    from sigma_ground.field.interface import quantum_output as QO
    from sigma_ground.field.interface import quantum_algorithms as QA
    if qaoa_edges is None:
        qaoa_edges = [[0, 1], [1, 2], [2, 0]]  # triangle
    g, err_g = _safe_call(QO.grover_search, grover_n_qubits, grover_marked_item)
    q, err_q = _safe_call(QA.qaoa_maxcut, qaoa_edges, qaoa_n_nodes)
    s, err_s = _safe_call(QA.simon_algorithm, simon_hidden_string)
    g = g or {}
    q = q or {}
    s = s or {}
    results = {
        "grover_answer": g.get("answer") if isinstance(g, dict) else None,
        "qaoa_best_cut_value": q.get("best_cut_value") if isinstance(q, dict) else None,
        "qaoa_max_possible_cut": q.get("max_possible_cut") if isinstance(q, dict) else None,
        "qaoa_best_bitstring": q.get("best_bitstring") if isinstance(q, dict) else None,
        "simon_found_string": s.get("found_string") if isinstance(s, dict) else None,
        "simon_success": s.get("success") if isinstance(s, dict) else None,
    }
    errors: dict[str, str] = {}
    if err_g:
        errors["grover_answer"] = err_g
    if err_q:
        for k in ("qaoa_best_cut_value", "qaoa_max_possible_cut", "qaoa_best_bitstring"):
            errors[k] = err_q
    if err_s:
        for k in ("simon_found_string", "simon_success"):
            errors[k] = err_s
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface (quantum_output, quantum_algorithms)",
                      provenance_tag="DERIVED",
                      formula="Grover O(sqrt N); QAOA p-layer Max-Cut; Simon period via linear system",
                      notes=_notes_from_errors(errors),
                      inputs={"grover_n_qubits": grover_n_qubits,
                              "grover_marked_item": grover_marked_item,
                              "qaoa_n_nodes": qaoa_n_nodes,
                              "simon_hidden_string": simon_hidden_string}).to_dict()


def quantum_state_analysis() -> dict[str, Any]:
    """Single- and two-qubit state diagnostics on canonical states: the
    expectation value of Pauli-Z on |+>, the Bloch-sphere angles of |+>, and the
    Schmidt coefficients + entanglement entropy of a Bell state. Includes one
    stochastic projective measurement of |+> (Born rule)."""
    from sigma_ground.field.interface import quantum_output as QO
    plus = [_R, _R]
    bell = [_R, 0.0, 0.0, _R]
    theta_phi, err_theta_phi = _safe_call(QO.state_to_bloch, plus)
    schmidt, err_schmidt = _safe_call(QO.schmidt_coefficients, bell, [0])
    # entanglement entropy S = -sum lambda^2 log2 lambda^2
    ent = None
    if schmidt:
        ent = -sum((c * c) * math.log2(c * c) for c in schmidt if c > 1e-15)
    meas, err_meas = _safe_call(QO.measure_all, plus)
    expZ, err_expZ = _safe_call(QO.expectation_observable, plus, [1, -1])
    results = {
        "expectation_Z_on_plus": expZ,
        "bloch_theta_phi_of_plus": list(theta_phi) if theta_phi else None,
        "bell_schmidt_coefficients": schmidt,
        "bell_entanglement_entropy_bits": ent,
        "sampled_measurement_bitstring": (meas[1] if isinstance(meas, (tuple, list)) and len(meas) > 1 else None),
    }
    errors: dict[str, str] = {}
    if err_expZ:
        errors["expectation_Z_on_plus"] = err_expZ
    if err_theta_phi:
        errors["bloch_theta_phi_of_plus"] = err_theta_phi
    if err_schmidt:
        errors["bell_schmidt_coefficients"] = err_schmidt
    if err_meas:
        errors["sampled_measurement_bitstring"] = err_meas
    return ToolResult(value=results, units="dimensionless, rad, bits",
                      source="sigma_ground.field.interface.quantum_output",
                      provenance_tag="DERIVED",
                      formula="<O>=sum o|amp|^2; Bloch (theta,phi); Schmidt SVD; S=-sum p log2 p",
                      notes=" | ".join(filter(None, [
                          "sampled_measurement_bitstring is one stochastic Born-rule outcome.",
                          _notes_from_errors(errors)])),
                      inputs={"state_plus": "[0.707,0.707]", "state_bell": "[0.707,0,0,0.707]"}).to_dict()


def qubit_hardware_analysis(qubit_type: str = "transmon",
                            material_key: str = "aluminum",
                            b_tesla: float = 1.0,
                            radius_m: float = 5.0e-9) -> dict[str, Any]:
    """Operating parameters of a physical qubit: frequency (from cascade
    physics) plus typical coherence times (T1/T2) and gate fidelity. Types:
    transmon (material_key), spin (b_tesla), quantum_dot (radius_m), nv_center.
    e.g. qubit_hardware_analysis('transmon', 'aluminum')."""
    from sigma_ground.field.interface import quantum_computing as QC
    kw: dict[str, Any] = {}
    if qubit_type == "transmon":
        kw["material_key"] = material_key
    elif qubit_type == "spin":
        kw["B_tesla"] = b_tesla
    elif qubit_type == "quantum_dot":
        kw["radius_m"] = radius_m
    summary, err = _safe_call(QC.qubit_summary, qubit_type, **kw)
    results = summary if isinstance(summary, dict) else {"summary": None}
    errors = {"summary": err} if err else {}
    return ToolResult(value=results, units="GHz, us, dimensionless",
                      source="sigma_ground.field.interface.quantum_computing",
                      provenance_tag="DERIVED",
                      formula="frequency from cascade; T1/T2/fidelity typical measured ranges",
                      notes=_notes_from_errors(errors),
                      inputs={"qubit_type": qubit_type, "material_key": material_key}).to_dict()


def interference_visibility_analysis(intensity_max: float = 1.0,
                                     intensity_min: float = 0.2) -> dict[str, Any]:
    """Fringe visibility (contrast) of an interference pattern,
    V = (I_max - I_min)/(I_max + I_min). e.g.
    interference_visibility_analysis(1.0, 0.2) -> 0.667."""
    from sigma_ground.field.interface import quantum as QM
    V, err = _safe_call(QM.fringe_visibility, intensity_max, intensity_min)
    results = {"fringe_visibility": V}
    errors = {"fringe_visibility": err} if err else {}
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface.quantum",
                      provenance_tag="DERIVED",
                      formula="V = (I_max - I_min)/(I_max + I_min)",
                      notes=_notes_from_errors(errors),
                      inputs={"intensity_max": intensity_max,
                              "intensity_min": intensity_min}).to_dict()
