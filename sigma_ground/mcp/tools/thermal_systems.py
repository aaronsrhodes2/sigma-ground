"""Thermal-systems analysis tools (standard physics).

Composite tools cascading through field.interface.{thermoelectric, gas}.

(thermal.contact_conductance is intentionally NOT wired — its current model
returns a near-ballistic ~1e9 W/m^2K, orders above engineering joint
conductance; deferred for review. See misc/COVERAGE_LEDGER.md.)
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (thermoelectric, thermal, gas)"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def thermoelectric_generator_analysis(hot_temperature_k: float = 600.0,
                                      cold_temperature_k: float = 300.0,
                                      material_key: str = "silicon",
                                      mat_p: str = "iron",
                                      mat_n: str = "copper") -> dict[str, Any]:
    """Thermoelectric generator (TEG): Carnot limit, Seebeck thermocouple
    voltage, leg resistance, maximum power, Ioffe efficiency (ZT), Fourier
    heat flow through a leg, and a full-system simulation.
    e.g. thermoelectric_generator_analysis(600, 300, 'silicon', 'iron', 'copper')."""
    from sigma_ground.field.interface import thermoelectric as TE
    pmax = _safe(TE.thermoelectric_power_max, mat_p, mat_n,
                 hot_temperature_k, cold_temperature_k) or {}
    teg = _safe(TE.simulate_teg_system, material_key, mat_p, mat_n,
                hot_temperature_k, cold_temperature_k)
    results = {
        "carnot_efficiency": _safe(TE.carnot_efficiency, hot_temperature_k, cold_temperature_k),
        "thermocouple_voltage_V": _safe(TE.thermocouple_voltage, mat_p, mat_n,
                                        hot_temperature_k, cold_temperature_k),
        "leg_resistance_ohm": _safe(TE.leg_resistance, material_key, 0.01, 1.0e-4),
        "max_power_W": pmax.get("power_W") if isinstance(pmax, dict) else None,
        "thermoelectric_efficiency": _safe(TE.thermoelectric_efficiency, material_key,
                                          hot_temperature_k, cold_temperature_k),
        "heat_flow_through_leg_W": _safe(TE.heat_flow_through_leg, material_key,
                                        hot_temperature_k, cold_temperature_k),
        "teg_system_power_W": (teg.get("power_max_W") if isinstance(teg, dict) else None),
        "teg_system_efficiency": (teg.get("efficiency") if isinstance(teg, dict) else None),
    }
    return ToolResult(value=results, units="dimensionless, V, ohm, W", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="eta_C=1-Tc/Th; V=|dS| dT; P=Voc^2/4R; Ioffe ZT efficiency; Q=kA dT/L",
                      inputs={"hot_temperature_k": hot_temperature_k,
                              "cold_temperature_k": cold_temperature_k,
                              "material_key": material_key,
                              "mat_p": mat_p, "mat_n": mat_n}).to_dict()


def natural_convection_analysis(hot_temperature_k: float = 350.0,
                                ambient_temperature_k: float = 300.0,
                                length_m: float = 0.01,
                                gas_key: str = "N2",
                                gas_key_2: str = "O2") -> dict[str, Any]:
    """Buoyancy-driven natural convection of a gas: characteristic buoyancy
    velocity, Grashof number (laminar/turbulent), and the binary gas
    diffusivity (Chapman-Enskog). e.g. natural_convection_analysis(350, 300, 0.01, 'N2')."""
    from sigma_ground.field.interface import gas as G
    results = {
        "buoyancy_velocity_m_s": _safe(G.buoyancy_velocity, hot_temperature_k,
                                      ambient_temperature_k, length_m),
        "grashof_number": _safe(G.grashof_number, hot_temperature_k,
                               ambient_temperature_k, length_m, gas_key),
        "binary_diffusivity_m2_s": _safe(G.gas_diffusivity, gas_key, gas_key_2,
                                        ambient_temperature_k),
    }
    return ToolResult(value=results, units="m/s, dimensionless, m^2/s",
                      source="sigma_ground.field.interface.gas",
                      provenance_tag="DERIVED",
                      formula="v=sqrt(g L dT/T); Gr=g beta dT L^3/nu^2; D Chapman-Enskog",
                      inputs={"hot_temperature_k": hot_temperature_k,
                              "ambient_temperature_k": ambient_temperature_k,
                              "length_m": length_m, "gas_key": gas_key}).to_dict()
