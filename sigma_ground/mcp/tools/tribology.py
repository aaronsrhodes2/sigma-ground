"""Tribology analysis tools (standard physics).

Composite tools cascading through field.interface.{friction, wear}.

(adhesion.contact_angle is intentionally NOT wired -- with only a solids surface-
energy DB the work-of-adhesion model overestimates wetting and returns 0 deg
(complete wetting) for metal pairs that are really non-wetting; deferred for
review. See misc/COVERAGE_LEDGER.md.)
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
