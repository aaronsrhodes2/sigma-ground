"""Tribology analysis tools (standard physics).

Composite tools cascading through field.interface.{friction, wear, adhesion}.

wetting_analysis uses the Owens-Wendt wetting model (adhesion.WETTING_SOLIDS /
WETTING_LIQUIDS), which DOES give physical contact angles (water/PTFE ~108 deg,
mercury/glass ~133-140 deg, water/clean-glass ~0 deg). The older
adhesion.contact_angle stays unwired: it draws both phases from the solids
broken-bond DB, so for metal pairs cos(theta)=W/gamma_LV-1 >= 1 and it reports
0 deg (complete wetting) for systems that really bead. See misc/COVERAGE_LEDGER.md.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (friction, wear, adhesion)"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def friction_analysis(material_key_1: str = "copper",
                      material_key_2: str = "steel_mild",
                      normal_force_n: float = 10.0) -> dict[str, Any]:
    """Dry sliding friction between two materials: interfacial shear strength,
    adhesive friction coefficient, ploughing-term coefficient, and the friction
    force at a given normal load. e.g. friction_analysis('copper', 'steel_mild', 10)."""
    from sigma_ground.field.interface import friction as F
    # ploughing is harder-on-softer; try both orders and take the real (nonzero) one
    plough = max(_safe(F.ploughing_friction, material_key_1, material_key_2) or 0.0,
                 _safe(F.ploughing_friction, material_key_2, material_key_1) or 0.0)
    results = {
        "interfacial_shear_strength_Pa": _safe(F.interfacial_shear_strength, material_key_1, material_key_2),
        "adhesive_friction_coefficient": _safe(F.friction_coefficient, material_key_1, material_key_2),
        "ploughing_coefficient": plough,
        "friction_force_N": _safe(F.friction_force, material_key_1, material_key_2, normal_force_n),
    }
    return ToolResult(value=results, units="Pa, dimensionless, N", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="mu_adh = tau_i / H_soft; F = (mu_adh + mu_plough) N (Bowden-Tabor)",
                      inputs={"material_key_1": material_key_1,
                              "material_key_2": material_key_2,
                              "normal_force_n": normal_force_n}).to_dict()


def wear_analysis(material_key: str = "copper",
                  normal_force_n: float = 10.0,
                  sliding_distance_m: float = 100.0,
                  velocity_m_s: float = 1.0,
                  counter_material: str = "steel_mild") -> dict[str, Any]:
    """Sliding wear (Archard model): worn volume, mass loss over a sliding
    distance, sliding wear rate, and the wear regime (mild/severe, adhesive/
    abrasive). e.g. wear_analysis('copper', 10, 100, 1.0, 'steel_mild')."""
    from sigma_ground.field.interface import wear as W
    results = {
        "archard_wear_volume_m3": _safe(W.archard_wear_volume, material_key, normal_force_n, sliding_distance_m),
        "wear_mass_loss_kg": _safe(W.wear_mass_loss, material_key, normal_force_n, sliding_distance_m),
        "sliding_wear_rate": _safe(W.sliding_wear_rate, material_key, normal_force_n, velocity_m_s),
        "wear_regime": _safe(W.wear_regime, material_key, counter_material),
    }
    return ToolResult(value=results, units="m^3, kg, varies, label", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="V = K F d / H (Archard); m = rho V; regime from K, pressure",
                      inputs={"material_key": material_key, "normal_force_n": normal_force_n,
                              "sliding_distance_m": sliding_distance_m,
                              "velocity_m_s": velocity_m_s,
                              "counter_material": counter_material}).to_dict()


def wetting_analysis(solid_key: str = "glass",
                     liquid_key: str = "water") -> dict[str, Any]:
    """Liquid wetting on a solid (Young-Dupre + Owens-Wendt): equilibrium
    contact angle, work of adhesion, spreading coefficient, and wetting regime.
    e.g. wetting_analysis('ptfe', 'water') -> ~108 deg (hydrophobic);
    wetting_analysis('glass', 'mercury') -> ~133 deg (beads, non-wetting)."""
    from sigma_ground.field.interface import adhesion as A

    solid = A.WETTING_SOLIDS.get(solid_key)
    liquid = A.WETTING_LIQUIDS.get(liquid_key)
    if solid is None or liquid is None:
        return ToolResult(
            value={"contact_angle_deg": None}, units="deg", source=_SRC,
            provenance_tag="DERIVED",
            notes=("unknown material key. available solids: "
                   f"{sorted(A.WETTING_SOLIDS)}; "
                   f"available liquids: {sorted(A.WETTING_LIQUIDS)}"),
            inputs={"solid_key": solid_key, "liquid_key": liquid_key}).to_dict()

    theta = A.wetting_contact_angle(solid_key, liquid_key)
    W = A.work_of_solid_liquid_adhesion(solid_key, liquid_key)
    S = A.spreading_coefficient(solid_key, liquid_key)
    gamma_lv = liquid["gamma_lv"]

    if theta is None:
        regime = "undefined"
    elif theta <= 1.0:
        regime = "complete wetting (spreads)"
    elif theta < 90.0:
        regime = "wetting (hydrophilic)"
    elif theta < 180.0:
        regime = "non-wetting (hydrophobic)"
    else:
        regime = "complete non-wetting (beads)"

    results = {
        "contact_angle_deg": theta,
        "work_of_adhesion_J_m2": W,
        "spreading_coefficient_J_m2": S,
        "gamma_lv_N_m": gamma_lv,
        "wetting_regime": regime,
        "solid": solid["name"],
        "liquid": liquid["name"],
    }
    return ToolResult(
        value=results, units="deg, J/m^2, J/m^2, N/m, label", source=_SRC,
        provenance_tag="DERIVED",
        formula=("cos(theta) = W_SL/gamma_LV - 1 (Young-Dupre); "
                 "W_SL = 2 sqrt(gd_s gd_l) + 2 sqrt(gp_s gp_l) (Owens-Wendt)"),
        notes=("Owens-Wendt geometric-mean combining rule; metallic liquids "
               "(mercury, molten solder, gallium) use the dispersive (Fowkes) "
               "term only on a dielectric solid. Real contact angles are "
               "sensitive to surface cleanliness/roughness -- treat as +/-10 deg."),
        inputs={"solid_key": solid_key, "liquid_key": liquid_key}).to_dict()
