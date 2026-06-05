"""Thermal-systems analysis tools (standard physics).

Composite tools cascading through field.interface.{thermoelectric, thermal, gas}.

thermal_contact_analysis wires thermal.contact_conductance, which was rebuilt
on the Cooper-Mikic-Yovanovich plastic joint-conductance correlation (validated
against the textbook closed form). The earlier atomic-gap model — which returned
a near-ballistic ~1e9 W/m^2K and was deferred — has been retired.
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


def thermal_contact_analysis(material_1: str = "copper",
                             material_2: str = "aluminum",
                             pressure_pa: float = 1.0e6,
                             temperature_k: float = 300.0,
                             roughness_m: float = 2.0e-6,
                             asperity_slope: float = 0.1) -> dict[str, Any]:
    """Engineering thermal contact (joint) conductance between two pressed
    metal surfaces, via the Cooper-Mikic-Yovanovich plastic correlation
    h_c = 1.25 k_s (m/sigma)(P/H_c)^0.95. Also reports the contact
    (thermal) resistance 1/h_c, the harmonic-mean conductivity k_s, the
    contact microhardness H_c of the softer metal, and the Bowden-Tabor
    real-contact fraction P/H. Roughness and asperity slope are ENGINEERING
    surface-finish inputs, not material properties.
    e.g. thermal_contact_analysis('copper','aluminum', 1e6, 300, 2e-6, 0.1)."""
    from sigma_ground.field.interface import thermal as TH
    from sigma_ground.field.interface.friction import _hardness, real_contact_fraction

    h_c = _safe(TH.contact_conductance, material_1, material_2, pressure_pa,
                temperature_k, 0.0, roughness_m, asperity_slope)
    k1 = _safe(TH.thermal_conductivity, material_1, temperature_k)
    k2 = _safe(TH.thermal_conductivity, material_2, temperature_k)
    k_s = (2.0 * k1 * k2 / (k1 + k2)) if (k1 and k2) else None

    h1 = _safe(_hardness, material_1)
    h2 = _safe(_hardness, material_2)
    microhardness = min(h1, h2) if (h1 and h2) else (h1 or h2)

    f1 = _safe(real_contact_fraction, material_1, pressure_pa)
    f2 = _safe(real_contact_fraction, material_2, pressure_pa)
    # Softer material yields more → larger real-contact fraction.
    f_real = max(f1, f2) if (f1 is not None and f2 is not None) else (f1 or f2)

    results = {
        "contact_conductance_W_m2K": h_c,
        "contact_resistance_m2K_W": (1.0 / h_c if h_c else None),
        "harmonic_mean_conductivity_W_mK": k_s,
        "contact_microhardness_Pa": microhardness,
        "real_contact_fraction": f_real,
    }
    return ToolResult(value=results,
                      units="W/(m^2.K), m^2.K/W, W/(m.K), Pa, dimensionless",
                      source="sigma_ground.field.interface.thermal "
                             "(Cooper-Mikic-Yovanovich plastic model)",
                      provenance_tag="DERIVED",
                      formula="h_c = 1.25 k_s (m/sigma)(P/H_c)^0.95  [CMY 1969 / Mikic 1974]",
                      notes="Solid-spot joint conductance for two conforming rough "
                            "metal surfaces in vacuum (no interstitial gas or radiation). "
                            "roughness_m (RMS) and asperity_slope are ENGINEERING "
                            "surface-finish inputs, not material constants; defaults "
                            "describe a typical machined pair. Cu-Al at 1 MPa ~ 6e4 "
                            "W/(m^2.K); rougher finishes / lower pressure trend toward "
                            "the 1e3 floor. Valid for P/H_c <~ 0.1.",
                      inputs={"material_1": material_1, "material_2": material_2,
                              "pressure_pa": pressure_pa,
                              "temperature_k": temperature_k,
                              "roughness_m": roughness_m,
                              "asperity_slope": asperity_slope}).to_dict()
