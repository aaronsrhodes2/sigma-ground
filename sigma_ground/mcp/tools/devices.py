"""Electronic-device analysis tools (standard physics).

Composite tools cascading through field.interface.electronics — capacitor
geometries, Hall effect, and p-n junction device quantities. These take fully
defaulted, realistic scenarios so a single call exercises a whole topic (the
per-quantity wrappers in electronics.py / circuits.py require explicit inputs).
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface.electronics"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def capacitor_analysis(area_m2: float = 1.0e-4, separation_m: float = 1.0e-3,
                       epsilon_r: float = 1.0, length_m: float = 1.0,
                       r_inner_m: float = 1.0e-3, r_outer_m: float = 5.0e-3,
                       sphere_inner_m: float = 0.05, sphere_outer_m: float = 0.10,
                       voltage_v: float = 100.0) -> dict[str, Any]:
    """Capacitance of three canonical geometries (parallel-plate, coaxial,
    concentric-sphere) and the energy stored on the parallel-plate cap at a
    given voltage. e.g. capacitor_analysis(1e-4, 1e-3, 1.0)."""
    from sigma_ground.field.interface import electronics as E
    C_pp = _safe(E.parallel_plate_capacitance, area_m2, separation_m, epsilon_r)
    results = {
        "parallel_plate_F": C_pp,
        "coaxial_F": _safe(E.coaxial_capacitance, length_m, r_inner_m, r_outer_m, epsilon_r),
        "spherical_F": _safe(E.spherical_capacitance, sphere_inner_m, sphere_outer_m, epsilon_r),
        "energy_stored_J": (_safe(E.energy_stored, C_pp, voltage_v) if C_pp else None),
    }
    return ToolResult(value=results, units="F, J", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="C_pp=eps0 eps_r A/d; C_coax=2pi eps L/ln(ro/ri); C_sph=4pi eps/(1/ri-1/ro); U=CV^2/2",
                      inputs={"area_m2": area_m2, "separation_m": separation_m,
                              "epsilon_r": epsilon_r, "voltage_v": voltage_v}).to_dict()


def hall_effect_analysis(material_key: str = "copper", current_a: float = 1.0,
                         b_field_t: float = 0.5,
                         thickness_m: float = 1.0e-4) -> dict[str, Any]:
    """Hall voltage of a current-carrying conductor in a transverse magnetic
    field (negative for electron carriers). e.g. hall_effect_analysis('copper',
    1.0, 0.5, 1e-4)."""
    from sigma_ground.field.interface import electronics as E
    results = {
        "hall_voltage_V": _safe(E.hall_voltage, material_key, current_a, b_field_t, thickness_m),
    }
    return ToolResult(value=results, units="V", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="V_H = R_H I B / t",
                      inputs={"material_key": material_key, "current_a": current_a,
                              "b_field_t": b_field_t, "thickness_m": thickness_m}).to_dict()


def semiconductor_junction_analysis(sc_key: str = "silicon",
                                    donor_density: float = 1.0e22,
                                    acceptor_density: float = 1.0e22,
                                    area_m2: float = 1.0e-6,
                                    applied_voltage: float = 0.0) -> dict[str, Any]:
    """p-n junction device: depletion (junction) capacitance and reverse
    saturation current. Keys: silicon, germanium, gallium_arsenide,
    silicon_carbide, gallium_nitride, indium_phosphide.
    e.g. semiconductor_junction_analysis('silicon', 1e22, 1e22, 1e-6)."""
    from sigma_ground.field.interface import electronics as E
    results = {
        "junction_capacitance_F": _safe(E.junction_capacitance, sc_key,
                                        donor_density, acceptor_density,
                                        area_m2, applied_voltage),
        "saturation_current_A": _safe(E.diode_saturation_current, sc_key,
                                      donor_density, acceptor_density, area_m2),
    }
    return ToolResult(value=results, units="F, A", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="C_j=eps A/W(V); I_0 from n_i^2 (diffusion)",
                      inputs={"sc_key": sc_key, "donor_density": donor_density,
                              "acceptor_density": acceptor_density,
                              "area_m2": area_m2,
                              "applied_voltage": applied_voltage}).to_dict()
