"""Microstructure & molecular materials analysis tools (standard physics).

Composite tools cascading through field.interface.{grain_structure, alloys,
molecular_bonds, organic_materials}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def _safe_call(fn, *a, **k):
    """Call a leaf function, returning (value, error_note) instead of a bare
    value-or-None. On success: (value, None). On failure: (None, note) where
    `note` is the exception's type and message (e.g. "KeyError:
    'unobtainium'") -- deliberately NOT fn-name-specific, so that two
    different leaf calls failing on the SAME underlying bad input (e.g. the
    same bad material_key) produce byte-identical notes and can be
    deduplicated by _notes_from_errors() below.

    Like _safe() above, this never raises -- one bad sub-call only ever
    poisons its own (key, value) pair in the caller's results dict, so the
    caller always goes on to make the rest of the cascade's calls. The
    difference from _safe() is purely that the reason for a failure is
    handed back instead of swallowed, so it can be threaded into
    ToolResult(notes=...) rather than surfacing a bare `None` with nothing
    explaining why."""
    try:
        return fn(*a, **k), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _notes_from_errors(errors: dict[str, str]) -> str:
    """Fold a {result_key: error_note} map (only the FAILED keys) into one
    ToolResult.notes string.

    Groups result keys whose error_note is byte-identical -- the signature
    of a shared root cause, e.g. N calls in one cascade that all take the
    same bad material_key -- so the explanation is stated once per DISTINCT
    cause (naming every result key it took down), not once per affected
    key. Keys with distinct causes each keep their own clause. Returns ""
    when `errors` is empty (the all-success case), matching ToolResult's
    default notes="".
    """
    if not errors:
        return ""
    groups: dict[str, list[str]] = {}
    for key, note in errors.items():
        groups.setdefault(note, []).append(key)
    return " | ".join(f"{', '.join(keys)}: {note}" for note, keys in groups.items())


# fuel -> (bonds_broken, bonds_formed, n_O2) for complete combustion.
# bonds_broken excludes O=O (n_O2 accounts for that separately, per
# combustion_enthalpy_kJ_mol's signature).
_FUELS = {
    "methane": ({"C-H": 4}, {"C=O": 2, "O-H": 4}, 2),
    "propane": ({"C-H": 8, "C-C": 2}, {"C=O": 6, "O-H": 8}, 5),
    # Cross-checked against field.interface.chemical_reactions.REACTIONS
    # ['ethanol_combustion'] (measured dH_comb = -1367.0 kJ/mol) -- same
    # underlying data, just re-shaped to this function's (broken, formed,
    # n_O2) convention rather than duplicated by hand.
    "ethanol": ({"C-C": 1, "C-H": 5, "C-O": 1, "O-H": 1},
                {"C=O": 4, "O-H": 6}, 3),
}


def dislocation_strengthening_analysis(material_key: str = "copper",
                                       dislocation_density: float = 1.0e14) -> dict[str, Any]:
    """Taylor work-hardening: the shear flow stress contributed by a forest of
    dislocations, tau = alpha G b sqrt(rho). e.g.
    dislocation_strengthening_analysis('copper', 1e14)."""
    from sigma_ground.field.interface import grain_structure as GS
    tau, err = _safe_call(GS.taylor_hardening_stress, material_key, dislocation_density)
    results = {"taylor_flow_stress_Pa": tau}
    errors = {"taylor_flow_stress_Pa": err} if err else {}
    return ToolResult(value=results, units="Pa",
                      source="sigma_ground.field.interface.grain_structure",
                      provenance_tag="DERIVED",
                      formula="tau = alpha G b sqrt(rho_dislocation) (Taylor)",
                      notes=_notes_from_errors(errors),
                      inputs={"material_key": material_key,
                              "dislocation_density": dislocation_density}).to_dict()


def alloy_resistivity_analysis(metal_a: str = "copper", metal_b: str = "nickel",
                               fraction_b: float = 0.3) -> dict[str, Any]:
    """Residual resistivity of a binary solid-solution alloy from Nordheim's
    rule (delta-rho proportional to x(1-x)). e.g. alloy_resistivity_analysis(
    'copper', 'nickel', 0.3) -> Cu-30%Ni."""
    from sigma_ground.field.interface import alloys as AL
    comp = {metal_a: 1.0 - fraction_b, metal_b: fraction_b}
    r, err = _safe_call(AL.alloy_Nordheim_resistivity, comp)
    rho = r.get("rho_residual_uOhm_cm") if isinstance(r, dict) else None
    results = {"residual_resistivity_uOhm_cm": rho}
    errors = {"residual_resistivity_uOhm_cm": err} if err else {}
    return ToolResult(value=results, units="uOhm.cm",
                      source="sigma_ground.field.interface.alloys",
                      provenance_tag="DERIVED",
                      formula="delta-rho = A_Nordheim x(1-x)",
                      notes=_notes_from_errors(errors),
                      inputs={"metal_a": metal_a, "metal_b": metal_b,
                              "fraction_b": fraction_b}).to_dict()


def molecular_dipole_analysis(bond_dipoles_debye: list[float] | None = None,
                              bond_angles_deg: list[float] | None = None) -> dict[str, Any]:
    """Net molecular dipole moment from the vector sum of bond dipoles
    (Debye). Default is a water-like molecule (two 1.5 D O-H bonds at +/-52.25
    deg -> ~1.84 D). e.g. molecular_dipole_analysis([1.5,1.5], [52.25,-52.25])."""
    from sigma_ground.field.interface import molecular_bonds as MB
    if bond_dipoles_debye is None:
        bond_dipoles_debye = [1.5, 1.5]
    if bond_angles_deg is None:
        bond_angles_deg = [52.25, -52.25]
    mu, err = _safe_call(MB.molecular_dipole_moment, bond_dipoles_debye, bond_angles_deg)
    results = {"dipole_moment_debye": mu}
    errors = {"dipole_moment_debye": err} if err else {}
    return ToolResult(value=results, units="D (debye)",
                      source="sigma_ground.field.interface.molecular_bonds",
                      provenance_tag="DERIVED",
                      formula="mu = |sum of bond-dipole vectors|",
                      notes=_notes_from_errors(errors),
                      inputs={"bond_dipoles_debye": bond_dipoles_debye,
                              "bond_angles_deg": bond_angles_deg}).to_dict()


def combustion_enthalpy_analysis(fuel: str = "methane") -> dict[str, Any]:
    """Combustion enthalpy of a hydrocarbon fuel from a bond-energy inventory
    (Hess's law). Fuels: methane, propane, ethanol. NOTE: the bond-energy
    method is approximate (typically 10-25% from experiment, which is -890
    kJ/mol for methane). e.g. combustion_enthalpy_analysis('methane')."""
    from sigma_ground.field.interface import organic_materials as OM
    key = fuel.strip().lower()
    if key not in _FUELS:
        return ToolResult(
            value=None,
            source="sigma_ground.mcp.tools.materials_micro",
            notes=(f"'{fuel}' not in lookup. Available: {sorted(_FUELS.keys())}."),
            inputs={"fuel": fuel},
        ).to_dict()
    broken, formed, n_o2 = _FUELS[key]
    released = _safe(OM.combustion_enthalpy_kJ_mol, broken, formed, n_o2)
    results = {
        "energy_released_kJ_mol": released,
        "delta_H_combustion_kJ_mol": (-released if released is not None else None),
    }
    return ToolResult(value=results, units="kJ/mol",
                      source="sigma_ground.field.interface.organic_materials",
                      provenance_tag="DERIVED",
                      formula="dH = E_bonds_broken - E_bonds_formed (Hess's law, bond-energy estimate)",
                      notes="Bond-energy estimate; experimental methane dH_comb = -890 kJ/mol.",
                      inputs={"fuel": fuel}).to_dict()
