"""Quantum-solids analysis tools (standard physics).

Composite tools cascading through field.interface.{superconductivity, tunneling,
quantum_wells, band_structure, quantum_matter}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (superconductivity, tunneling, quantum_wells, band_structure, quantum_matter)"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def superconducting_gap_analysis(critical_temp_k: float = 9.2) -> dict[str, Any]:
    """BCS spectroscopic gap frequency f = 2*Delta/h of a superconductor from
    its critical temperature (Delta = 1.764 k_B Tc). Defaults to niobium
    (Tc=9.2 K -> ~677 GHz). e.g. superconducting_gap_analysis(9.2).

    (The GL-parameter and Hc1/Hc2 critical-field functions are intentionally
    NOT wired -- their current model is ~12 orders off and inverts Hc1/Hc2;
    deferred for review. See misc/COVERAGE_LEDGER.md.)"""
    from sigma_ground.field.interface import superconductivity as SC
    results = {"gap_frequency_Hz": _safe(SC.gap_frequency, critical_temp_k)}
    return ToolResult(value=results, units="Hz",
                      source="sigma_ground.field.interface.superconductivity",
                      provenance_tag="DERIVED",
                      formula="f = 2*Delta/h, Delta = 1.764 k_B Tc (BCS weak coupling)",
                      inputs={"critical_temp_k": critical_temp_k}).to_dict()


def quantum_tunneling_analysis(barrier_height_eV: float = 1.0,
                               particle_energy_eV: float = 0.5,
                               barrier_width_nm: float = 1.0) -> dict[str, Any]:
    """WKB transmission probability of a particle through a rectangular
    potential barrier. e.g. quantum_tunneling_analysis(1.0, 0.5, 1.0) -> a
    0.5 eV electron through a 1 eV, 1 nm barrier."""
    from sigma_ground.field.interface import tunneling as T
    height = barrier_height_eV

    def _barrier(x):
        return height

    width_m = barrier_width_nm * 1.0e-9
    prob = _safe(T.wkb_transmission, _barrier, particle_energy_eV, 0.0, width_m)
    results = {"transmission_probability": prob}
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface.tunneling",
                      provenance_tag="DERIVED",
                      formula="T = exp(-2 int kappa dx), kappa=sqrt(2m(V-E))/hbar (WKB)",
                      inputs={"barrier_height_eV": barrier_height_eV,
                              "particle_energy_eV": particle_energy_eV,
                              "barrier_width_nm": barrier_width_nm}).to_dict()


def quantum_box_energy_analysis(n1: int = 1, n2: int = 1, n3: int = 1,
                                box_size_nm: float = 1.0) -> dict[str, Any]:
    """Energy of a quantum state (n1,n2,n3) for a particle in a 3D cubic box
    (infinite well). e.g. quantum_box_energy_analysis(1, 1, 1, 1.0) -> ground
    state of an electron in a 1 nm box."""
    from sigma_ground.field.interface import quantum_wells as QW
    L = box_size_nm * 1.0e-9
    E = _safe(QW.box_energy_3d_eV, n1, n2, n3, L)
    results = {"energy_eV": E}
    return ToolResult(value=results, units="eV",
                      source="sigma_ground.field.interface.quantum_wells",
                      provenance_tag="DERIVED",
                      formula="E = (hbar^2 pi^2 / 2m)(n1^2/L1^2 + n2^2/L2^2 + n3^2/L3^2)",
                      inputs={"n1": n1, "n2": n2, "n3": n3,
                              "box_size_nm": box_size_nm}).to_dict()


def band_dos_shape_analysis(structure: str = "bcc",
                            d_electron_count: int = 5) -> dict[str, Any]:
    """Tight-binding density-of-states shape factor at the Fermi level for a
    transition metal (van Hove peak > 1, pseudogap < 1). structure in
    bcc/fcc/hcp, d-count 1-9. e.g. band_dos_shape_analysis('bcc', 5)."""
    from sigma_ground.field.interface import band_structure as BS
    g = _safe(BS.dos_shape_factor, structure, d_electron_count)
    results = {"dos_shape_factor": g}
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface.band_structure",
                      provenance_tag="DERIVED",
                      formula="g_dos = TB DOS(E_F) / rectangular DOS (avg-normalized)",
                      inputs={"structure": structure,
                              "d_electron_count": d_electron_count}).to_dict()


def magnetic_exchange_analysis(atomic_number: int = 24, oxidation_state: int = 3,
                               coord_key: str = "oxide_oct") -> dict[str, Any]:
    """Two-site Heisenberg model for a magnetic ion: exchange coupling J from
    crystal-field / Goodenough-Kanamori, VQE vs exact ground-state energy, and
    the resulting spin state (singlet/triplet). Defaults: Cr3+ octahedral oxide.
    e.g. magnetic_exchange_analysis(24, 3, 'oxide_oct')."""
    from sigma_ground.field.interface import quantum_matter as QM
    r = _safe(QM.two_site_spin_hamiltonian_from_crystal_field,
              atomic_number, oxidation_state, coord_key)
    if isinstance(r, dict):
        results = {
            "J_exchange_eV": r.get("J_exchange_eV"),
            "vqe_ground_energy_eV": r.get("E_vqe_eV"),
            "exact_ground_energy_eV": r.get("E_exact_eV"),
            "spin_state": r.get("spin_state"),
            "is_high_spin": r.get("is_high_spin"),
        }
    else:
        results = {"J_exchange_eV": None, "vqe_ground_energy_eV": None,
                   "exact_ground_energy_eV": None, "spin_state": None,
                   "is_high_spin": None}
    return ToolResult(value=results, units="eV", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="H = J(X1X2+Y1Y2+Z1Z2); J from crystal field; VQE vs exact",
                      inputs={"atomic_number": atomic_number,
                              "oxidation_state": oxidation_state,
                              "coord_key": coord_key}).to_dict()
