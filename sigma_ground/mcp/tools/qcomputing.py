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


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


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
    g = _safe(QO.grover_search, grover_n_qubits, grover_marked_item) or {}
    q = _safe(QA.qaoa_maxcut, qaoa_edges, qaoa_n_nodes) or {}
    s = _safe(QA.simon_algorithm, simon_hidden_string) or {}
    results = {
        "grover_answer": g.get("answer") if isinstance(g, dict) else None,
        "qaoa_best_cut_value": q.get("best_cut_value") if isinstance(q, dict) else None,
        "qaoa_max_possible_cut": q.get("max_possible_cut") if isinstance(q, dict) else None,
        "qaoa_best_bitstring": q.get("best_bitstring") if isinstance(q, dict) else None,
        "simon_found_string": s.get("found_string") if isinstance(s, dict) else None,
        "simon_success": s.get("success") if isinstance(s, dict) else None,
    }
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface (quantum_output, quantum_algorithms)",
                      provenance_tag="DERIVED",
                      formula="Grover O(sqrt N); QAOA p-layer Max-Cut; Simon period via linear system",
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
    theta_phi = _safe(QO.state_to_bloch, plus)
    schmidt = _safe(QO.schmidt_coefficients, bell, [0])
    # entanglement entropy S = -sum lambda^2 log2 lambda^2
    ent = None
    if schmidt:
        ent = -sum((c * c) * math.log2(c * c) for c in schmidt if c > 1e-15)
    meas = _safe(QO.measure_all, plus)
    results = {
        "expectation_Z_on_plus": _safe(QO.expectation_observable, plus, [1, -1]),
        "bloch_theta_phi_of_plus": list(theta_phi) if theta_phi else None,
        "bell_schmidt_coefficients": schmidt,
        "bell_entanglement_entropy_bits": ent,
        "sampled_measurement_bitstring": (meas[1] if isinstance(meas, (tuple, list)) and len(meas) > 1 else None),
    }
    return ToolResult(value=results, units="dimensionless, rad, bits",
                      source="sigma_ground.field.interface.quantum_output",
                      provenance_tag="DERIVED",
                      formula="<O>=sum o|amp|^2; Bloch (theta,phi); Schmidt SVD; S=-sum p log2 p",
                      notes="sampled_measurement_bitstring is one stochastic Born-rule outcome.",
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
    summary = _safe(QC.qubit_summary, qubit_type, **kw)
    results = summary if isinstance(summary, dict) else {"summary": None}
    return ToolResult(value=results, units="GHz, us, dimensionless",
                      source="sigma_ground.field.interface.quantum_computing",
                      provenance_tag="DERIVED",
                      formula="frequency from cascade; T1/T2/fidelity typical measured ranges",
                      inputs={"qubit_type": qubit_type, "material_key": material_key}).to_dict()


def interference_visibility_analysis(intensity_max: float = 1.0,
                                     intensity_min: float = 0.2) -> dict[str, Any]:
    """Fringe visibility (contrast) of an interference pattern,
    V = (I_max - I_min)/(I_max + I_min). e.g.
    interference_visibility_analysis(1.0, 0.2) -> 0.667."""
    from sigma_ground.field.interface import quantum as QM
    V = _safe(QM.fringe_visibility, intensity_max, intensity_min)
    results = {"fringe_visibility": V}
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface.quantum",
                      provenance_tag="DERIVED",
                      formula="V = (I_max - I_min)/(I_max + I_min)",
                      inputs={"intensity_max": intensity_max,
                              "intensity_min": intensity_min}).to_dict()
