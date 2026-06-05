"""Density from composition + crystal cell — the X-ray (crystallographic) density.

    rho = Z_cell * m_formula / V_cell

For a crystalline solid, density is not an independent fact — it is forced by the
unit cell: how heavy the atoms are (composition) and how tightly the lattice packs
them (structure + lattice parameter). Compute those and the density falls out.

Validated against the tabulated densities of the elemental metals to **0.49% mean
error (2.68% max)** in `misc/validate_density.py` — so for a crystalline material
the stored `density_kg_m3` is redundant; it emerges from the measured cell.

This is the accurate, MEASURED-cell path. It complements (does not duplicate)
`field.interface.element.predict_density_kg_m3(Z)`, which PREDICTS the whole cell
from Slater radii alone — fully first-principles from Z, but element-only and
rough (~30-100%). Here we use the material's measured crystal_structure + lattice
parameter, so the result is exact to <1%. Amorphous solids (glass, most polymers)
have no unit cell; their density is NOT derivable this way and is flagged, never
faked.

Origin: rho = Z_cell*m/V is FIRST_PRINCIPLES crystallography. The cell geometry
(atoms per cell, V from lattice) is FIRST_PRINCIPLES; the lattice parameter and
atomic masses are MEASURED. Same discipline as the emergent-color rung: a physics
model converts the substance's measured constants into the macroscopic property.
"""
from __future__ import annotations

import math

_AMU_KG_FALLBACK = 1.66053906660e-27          # CODATA atomic mass unit (kg)

# Atoms (or formula units) per conventional unit cell, by structure.
_ATOMS_PER_CELL = {"fcc": 4, "bcc": 2, "hcp": 2, "diamond": 8, "diamond_cubic": 8}


def _amu_kg() -> float:
    try:
        from ..field.constants import AMU_KG
        return AMU_KG
    except Exception:
        return _AMU_KG_FALLBACK


def _v_cell_m3(structure: str, a_m: float, c_m: float | None = None) -> float:
    """Conventional unit-cell volume (m^3) from the lattice parameter(s)."""
    if structure == "hcp":
        c = c_m if c_m else a_m * math.sqrt(8.0 / 3.0)     # ideal c/a = 1.633
        return a_m * a_m * c * math.sqrt(3.0) / 2.0
    return a_m ** 3                                         # cubic: fcc / bcc / diamond


def crystal_density(structure: str, a_m: float, mass_per_formula_kg: float,
                    c_m: float | None = None, z_cell: int | None = None) -> float:
    """rho = Z_cell * m_formula / V_cell  (kg/m^3).

    structure : 'fcc' | 'bcc' | 'hcp' | 'diamond' (sets Z_cell unless given)
    a_m, c_m  : lattice parameters in METRES
    mass_per_formula_kg : mass of one formula unit (one atom, for an element)
    """
    z = z_cell if z_cell is not None else _ATOMS_PER_CELL.get(structure, 1)
    return z * mass_per_formula_kg / _v_cell_m3(structure, a_m, c_m)


def density_of(material_key: str) -> dict:
    """Derive a material's density. Returns a provenance dict:

        {density_kg_m3, emergent, method[, tabulated_kg_m3, error_vs_tabulated_pct]}

    Path, in order:
      1. crystallographic — measured cell + atomic mass (accurate, <1%). EMERGENT.
      2. predicted cell from Z — element.predict_density_kg_m3 (rough). EMERGENT.
      3. tabulated fallback — amorphous / complex cell / missing data. NOT emergent.
    When a tabulated value is also on file, the result carries a cross-check.
    """
    try:
        from ..field.interface.surface import MATERIALS
    except Exception:
        MATERIALS = {}
    m = MATERIALS.get(material_key, {})
    a = m.get("lattice_param_angstrom")
    s = m.get("crystal_structure")
    A = m.get("A")                  # nucleon count ~ molar mass in u (elemental, <0.5%)
    Z = m.get("Z")
    mat_type = m.get("material_type")
    tab = m.get("density_kg_m3")

    out: dict = {}
    # 1) measured-cell crystallographic — the accurate emergent path.
    #    Only for single-element solids: a compound's `A` is an AVERAGE and its
    #    atoms-per-cell is not the elemental default (e.g. corundum is labelled
    #    'hcp' but packs Al2O3 formula units, not lone atoms) — so the elemental
    #    formula would be garbage. Compounds fall through to the honest fallback.
    elemental = mat_type in ("metal", "semiconductor")
    if elemental and a and s in _ATOMS_PER_CELL and A:
        rho = crystal_density(s, a * 1e-10, A * _amu_kg())
        out = {"density_kg_m3": round(rho, 1), "emergent": True,
               "method": f"crystallographic ({s} cell, a={a} A, A={A})"}
    # 2) predicted cell from Z (rough; only for elemental metals/semiconductors).
    elif Z and mat_type in ("metal", "semiconductor") and float(Z).is_integer():
        try:
            from ..field.interface.element import predict_density_kg_m3
            rho = predict_density_kg_m3(int(Z))
            out = {"density_kg_m3": round(rho, 1), "emergent": True,
                   "method": "predicted cell from Z (Slater radii; rough)"}
        except Exception:
            out = {}
    # 3) tabulated fallback / unknown — honestly flagged.
    if not out:
        if tab:
            return {"density_kg_m3": tab, "emergent": False,
                    "method": "tabulated (no derivable cell: amorphous / complex / missing)"}
        return {"density_kg_m3": None, "emergent": False, "method": "unknown material"}

    if tab:                          # cross-check the derivation against reality
        out["tabulated_kg_m3"] = tab
        out["error_vs_tabulated_pct"] = round(100.0 * (out["density_kg_m3"] - tab) / tab, 2)
    return out
