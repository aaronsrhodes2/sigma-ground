"""Compact stars from the dense-matter library: Chandrasekhar's mass limit and the structure of a cold white dwarf (field/dense/white_dwarf.py: exact T = 0 electron equation of state with Newtonian
hydrostatics, the Lane-Emden n = 3 limit for the mass). This file only asks and words the answer; a question the model cannot answer is declined with the reason."""
from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult

_SOURCE = "sigma-ground field.dense.white_dwarf: Newtonian hydrostatics on the exact T = 0 electron EOS (Chandrasekhar 1931, 1935); Chandrasekhar mass from the Lane-Emden n = 3 polytrope; constants CODATA"
_GAPS = ("Not modelled: Coulomb (Salpeter) corrections, ions' own pressure and finite temperature; the model is exact for the ideal degenerate electron gas, so a real star is somewhat smaller than this at a given mass. "
         "Valid between about 1e6 kg/m^3 (below it Coulomb and lattice corrections matter) and the neutron-drip density 4.3e14 kg/m^3 (above it the electron-ion matter is no longer the right equation of state).")


def chandrasekhar_limit(mu_e: float = 2.0) -> ToolResult:
    """Chandrasekhar's limiting mass of a white dwarf: the maximum mass a cold electron-degenerate star can have, from the Lane-Emden n = 3 polytrope; mu_e is the mean molecular weight per electron (2 for helium, carbon, oxygen)."""
    from sigma_ground.field.constants import M_SUN_KG
    from sigma_ground.field.dense import white_dwarf as wd
    inputs = {"mu_e": mu_e}
    if not mu_e > 0:
        return ToolResult(value=None, source=_SOURCE, notes="mu_e, the mean molecular weight per electron, must be positive (2 for helium, carbon and oxygen; 1 for hydrogen)", inputs=inputs)
    m = wd.chandrasekhar_mass(mu_e)
    return ToolResult(value={"mass_kg": m, "mass_msun": m / M_SUN_KG, "mu_e": mu_e}, units="kg, solar masses", source=_SOURCE, provenance_tag="DERIVED",
                      notes="M_Ch = (sqrt(3 pi) / 2) omega_3 (hbar c / G)^(3/2) / (mu_e m_u)^2 with omega_3 = 2.01824 from the Lane-Emden n = 3 solution; it scales as 1/mu_e^2. "
                            "Real white dwarfs approach it only in the ideal limit; the general-relativistic correction lowers the maximum slightly.", inputs=inputs)


def white_dwarf(central_density_kg_m3: float, mu_e: float = 2.0) -> ToolResult:
    """Mass and radius of a cold white dwarf with a given central mass density, from the exact degenerate-electron equation of state, e.g. white_dwarf(1e9)."""
    from sigma_ground.field.constants import M_SUN_KG
    from sigma_ground.field.dense import white_dwarf as wd
    inputs = {"central_density_kg_m3": central_density_kg_m3, "mu_e": mu_e}
    if not (central_density_kg_m3 > 0 and mu_e > 0):
        return ToolResult(value=None, source=_SOURCE, notes="the central density and mu_e must be positive", inputs=inputs)
    try:
        s = wd.white_dwarf_structure(central_density_kg_m3, mu_e)
    except Exception as e:                                           # the integrator's own limits: say so, do not guess
        return ToolResult(value=None, source=_SOURCE, notes=f"the structure could not be solved at this density ({type(e).__name__}: {e})", inputs=inputs)
    m_ch = wd.chandrasekhar_mass(mu_e)
    regime = {"EDGE": "below 1e6 kg/m^3: Coulomb and lattice corrections, not modelled, matter", "SAFE": "inside the range where the ideal degenerate electron gas is the right equation of state",
              "WALL": "above the neutron-drip density: the electron-ion matter model no longer applies"}[s.regime]
    return ToolResult(value={"central_density_kg_m3": central_density_kg_m3, "mu_e": mu_e, "mass_kg": s.mass_kg, "mass_msun": s.mass_kg / M_SUN_KG, "radius_m": s.radius_m, "radius_km": s.radius_m / 1e3,
                             "fraction_of_chandrasekhar_mass": s.mass_kg / m_ch, "regime": s.regime, "regime_meaning": regime},
                      units="kg, solar masses, m, km", source=_SOURCE, provenance_tag="DERIVED", notes=_GAPS, inputs=inputs)
