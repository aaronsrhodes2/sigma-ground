"""Electroceramics analysis tools (standard dielectric & piezoelectric physics).

Composite tools cascading through field.interface.{piezoelectricity, dielectric}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (piezoelectricity, dielectric)"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def piezoelectric_actuator_analysis(material_key: str = "PZT5A",
                                    e_field_v_m: float = 1.0e6,
                                    length_m: float = 0.02) -> dict[str, Any]:
    """Converse piezoelectric effect of an actuator under an applied field:
    induced strain (eps=d E) and tip displacement (delta=d E L).
    Keys: quartz, PZT4, PZT5A, BaTiO3, LiNbO3, AlN, PVDF.
    e.g. piezoelectric_actuator_analysis('PZT5A', 1e6, 0.02)."""
    from sigma_ground.field.interface import piezoelectricity as P
    results = {
        "strain": _safe(P.piezoelectric_strain, material_key, e_field_v_m),
        "displacement_m": _safe(P.piezoelectric_displacement, material_key, e_field_v_m, length_m),
    }
    return ToolResult(value=results, units="dimensionless, m", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="strain=d*E; displacement=d*E*L (converse piezoelectric effect)",
                      inputs={"material_key": material_key,
                              "e_field_v_m": e_field_v_m,
                              "length_m": length_m}).to_dict()


def dielectric_polarization_analysis(polarizability_fm2: float = 1.6e-40,
                                     number_density_m3: float = 3.0e28) -> dict[str, Any]:
    """Relative permittivity of a non-polar dielectric from the Clausius-
    Mossotti relation, given molecular polarizability (SI, F.m^2) and number
    density. e.g. dielectric_polarization_analysis(1.6e-40, 3e28)."""
    from sigma_ground.field.interface import dielectric as D
    eps_r = _safe(D.clausius_mossotti, polarizability_fm2, number_density_m3)
    results = {"relative_permittivity": eps_r}
    return ToolResult(value=results, units="dimensionless", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="(eps_r-1)/(eps_r+2) = N alpha / (3 eps0)  (Clausius-Mossotti)",
                      inputs={"polarizability_fm2": polarizability_fm2,
                              "number_density_m3": number_density_m3}).to_dict()
