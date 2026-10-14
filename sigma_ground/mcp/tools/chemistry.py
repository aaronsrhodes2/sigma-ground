"""Chemistry tools for the Sigma Ground MCP.

Thin wrappers over the library's already-implemented chemistry
(`sigma_ground.field.interface.{molecular_bonds, chemical_reactions,
acid_base, electrochemistry, solution}`), surfaced as MCP tools so the
switchboard can reach them. Every value traces to a library function; key-based
tools decline gracefully and list valid keys rather than crash. Verified
against textbook values before exposure (0.1 M acetic acid pH 2.88; Daniell
cell 1.10 V; AgCl solubility 1.33e-5 M; etc.).
"""
from __future__ import annotations

import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_EV_KJMOL = 96.485   # eV/particle → kJ/mol


def _decline(notes: str, **inputs) -> ToolResult:
    return ToolResult(value=None, source="sigma_ground.field.interface",
                      provenance_tag="SPECULATIVE-PENDING", notes=notes,
                      inputs=inputs)


# ── Bonds & molecular geometry ──────────────────────────────────────────
def bond_energy(atom_a: str, atom_b: str) -> dict[str, Any]:
    """Single-bond dissociation energy A–B (eV) via Pauling's equation.

    Supported atoms: H, C, N, O, F, S, Cl.
    """
    from sigma_ground.field.interface.molecular_bonds import (
        pauling_bond_energy, ATOMS)
    a, b = str(atom_a).strip().title(), str(atom_b).strip().title()
    if a not in ATOMS or b not in ATOMS:
        return _decline(f"bond_energy supports atoms {sorted(ATOMS)}; "
                        f"got {atom_a!r}, {atom_b!r}", atom_a=atom_a, atom_b=atom_b).to_dict()
    try:
        ev = pauling_bond_energy(a, b)
    except Exception as e:
        return _decline(f"bond_energy failed: {e}", atom_a=a, atom_b=b).to_dict()
    return ToolResult(
        value=ev, units="eV",
        source="sigma_ground.field.interface.molecular_bonds (Pauling)",
        provenance_tag="DERIVED", formula="D(A-B)=½(D_AA+D_BB)+(χ_A−χ_B)²",
        notes=f"≈{ev*_EV_KJMOL:.0f} kJ/mol. Pauling estimate (~15%).",
        inputs={"atom_a": a, "atom_b": b}).to_dict()


def bond_angle(electron_domains: int, lone_pairs: int = 0) -> dict[str, Any]:
    """VSEPR bond angle (degrees) for n electron domains and lone pairs.

    e.g. (4, 0)=109.47° (CH4); (4, 1)=~107° (NH3); (2, 0)=180° (CO2).
    """
    from sigma_ground.field.interface.molecular_bonds import vsepr_bond_angle
    try:
        ang = vsepr_bond_angle(int(electron_domains), int(lone_pairs))
    except Exception as e:
        return _decline(f"bond_angle failed: {e}",
                        electron_domains=electron_domains, lone_pairs=lone_pairs).to_dict()
    return ToolResult(
        value=ang, units="degrees",
        source="sigma_ground.field.interface.molecular_bonds (VSEPR)",
        provenance_tag="DERIVED", formula="ideal geometry − lone-pair compression",
        inputs={"electron_domains": electron_domains, "lone_pairs": lone_pairs}).to_dict()


# ── Thermochemistry ─────────────────────────────────────────────────────
def reaction_enthalpy(reaction_key: str) -> dict[str, Any]:
    """ΔH of a named reaction (kJ/mol) from bond energies (Hess's law).

    ESTIMATE (~15%): a bond-energy approximation, not a calorimetric value.
    """
    from sigma_ground.field.interface.chemical_reactions import (
        reaction_enthalpy_by_key, REACTIONS)
    if reaction_key not in REACTIONS:
        return _decline(f"Unknown reaction {reaction_key!r}. "
                        f"Available: {sorted(REACTIONS)}", reaction_key=reaction_key).to_dict()
    try:
        dh = reaction_enthalpy_by_key(reaction_key)
    except Exception as e:
        return _decline(f"reaction_enthalpy failed: {e}", reaction_key=reaction_key).to_dict()
    return ToolResult(
        value=dh, units="kJ/mol",
        source="sigma_ground.field.interface.chemical_reactions (bond-energy Hess's law)",
        provenance_tag="DERIVED", formula="ΔH = Σ(bonds broken) − Σ(bonds formed)",
        notes="Bond-energy ESTIMATE (~15%); not a calorimetric ΔH.",
        inputs={"reaction_key": reaction_key}).to_dict()


# ── Acid–base ───────────────────────────────────────────────────────────
# How pH_solution treats each species type in ACID_BASE_DATA.
_PH_LAW = {"strong_acid": "strong acid: [H+] = C (+ water)",
           "weak_acid": "weak acid: Ka equilibrium + charge balance",
           "strong_base": "strong base: [OH-] = n*C (+ water)",
           "weak_base": "weak base: Kb equilibrium + charge balance"}


def weak_acid_ph(acid_key: str, concentration_mol_l: float) -> dict[str, Any]:
    """pH of a solution of any species in the acid-base table at a given
    molar concentration -- weak or strong, acid or base.

    Dispatches on the species' type. It used to run every key, bases
    included, through the weak-ACID equation: 1.45 mM KOH came back as
    pH 15.0 and 0.1 M ammonia as an acidic 5.12.
    """
    from sigma_ground.field.interface.acid_base import pH_solution, ACID_BASE_DATA
    if acid_key not in ACID_BASE_DATA:
        return _decline(f"Unknown acid {acid_key!r}. Available: {sorted(ACID_BASE_DATA)}",
                        acid_key=acid_key).to_dict()
    conc = float(concentration_mol_l)
    if not (conc >= 0 and math.isfinite(conc)):
        return _decline(f"concentration must be a finite value >= 0 mol/L; got {conc}",
                        acid_key=acid_key,
                        concentration_mol_l=repr(conc)).to_dict()   # nan/inf aren't JSON
    kind = ACID_BASE_DATA[acid_key]["type"]
    try:
        ph = pH_solution(acid_key, conc)
    except Exception as e:
        return _decline(f"weak_acid_ph failed: {e}", acid_key=acid_key).to_dict()
    return ToolResult(
        value=ph, units="pH",
        source="sigma_ground.field.interface.acid_base",
        provenance_tag="DERIVED", formula=_PH_LAW.get(kind, kind),
        notes=f"{acid_key} is a {kind.replace('_', ' ')}.",
        inputs={"acid_key": acid_key, "concentration_mol_l": concentration_mol_l}).to_dict()


def buffer_ph(acid_key: str, ratio_base_over_acid: float = 1.0) -> dict[str, Any]:
    """Buffer pH via Henderson–Hasselbalch: pH = pKa + log10([A⁻]/[HA]).

    A buffer is a WEAK acid or base with its conjugate. For a weak base the
    pair is base / conjugate acid (NH3/NH4+) and pKa is the conjugate acid's.
    Strong acids and bases dissociate completely and form no buffer; they
    used to come back as pKa + log(ratio) anyway (HCl "pH -6.30", NaOH
    "pH 14.55"), and a ratio <= 0 came back as -inf, which isn't JSON.
    """
    from sigma_ground.field.interface.acid_base import (
        henderson_hasselbalch, pKa, water_ion_product, ACID_BASE_DATA)
    inputs = {"acid_key": acid_key, "ratio_base_over_acid": ratio_base_over_acid}
    if acid_key not in ACID_BASE_DATA:
        return _decline(f"Unknown acid {acid_key!r}. Available: {sorted(ACID_BASE_DATA)}",
                        acid_key=acid_key).to_dict()
    kind = ACID_BASE_DATA[acid_key]["type"]
    if kind.startswith("strong"):
        weak = sorted(k for k, d in ACID_BASE_DATA.items() if d["type"].startswith("weak"))
        return _decline(f"{acid_key} is a {kind.replace('_', ' ')}: it dissociates "
                        f"completely, so it forms no buffer (weak_acid_ph gives its pH). "
                        f"Buffers need a weak acid or base: {weak}", **inputs).to_dict()
    ratio = float(ratio_base_over_acid)
    if not (ratio > 0 and math.isfinite(ratio)):
        return _decline(f"ratio_base_over_acid must be a positive, finite "
                        f"[base]/[acid] ratio; got {ratio}", acid_key=acid_key,
                        ratio_base_over_acid=repr(ratio)).to_dict()   # nan/inf aren't JSON
    try:
        ph = henderson_hasselbalch(acid_key, ratio)
        pka = pKa(acid_key)
        pkw = -math.log10(water_ion_product())
    except Exception as e:
        return _decline(f"buffer_ph failed: {e}", acid_key=acid_key).to_dict()
    if not 0.0 <= ph <= pkw:
        return _decline(f"Henderson–Hasselbalch gives pH {ph:.2f} at a ratio of "
                        f"{ratio:g}, outside 0–{pkw:.1f}: this far from pKa ± 1 the "
                        f"mixture is not a buffer and its pH depends on the "
                        f"concentrations, which this tool doesn't take", **inputs).to_dict()
    pair = (f"{acid_key} / its conjugate acid, pKa {pka:.2f} of the conjugate acid"
            if kind == "weak_base" else f"{acid_key} / its conjugate base, pKa {pka:.2f}")
    notes = pair + "."
    if not 0.1 <= ratio <= 10:
        notes += (" Ratio outside 0.1–10 (pKa ± 1): Henderson–Hasselbalch is only "
                  "approximate here, and the true pH also depends on the concentrations.")
    return ToolResult(
        value=ph, units="pH",
        source="sigma_ground.field.interface.acid_base (Henderson–Hasselbalch)",
        provenance_tag="DERIVED", formula="pH = pKa + log10([A⁻]/[HA])",
        notes=notes, inputs=inputs).to_dict()


# ── Electrochemistry ────────────────────────────────────────────────────
def cell_potential(cathode: str, anode: str,
                   reaction_quotient: float = 1.0) -> dict[str, Any]:
    """Galvanic cell EMF (V): E = E°cell − (RT/nF)lnQ (Nernst-corrected)."""
    from sigma_ground.field.interface.electrochemistry import (
        cell_potential as _cell, STANDARD_POTENTIALS)
    if cathode not in STANDARD_POTENTIALS or anode not in STANDARD_POTENTIALS:
        return _decline(f"Electrodes must be in {sorted(STANDARD_POTENTIALS)}; "
                        f"got {cathode!r}/{anode!r}", cathode=cathode, anode=anode).to_dict()
    try:
        e = _cell(cathode, anode, float(reaction_quotient))
    except Exception as ex:
        return _decline(f"cell_potential failed: {ex}", cathode=cathode, anode=anode).to_dict()
    return ToolResult(
        value=e, units="V",
        source="sigma_ground.field.interface.electrochemistry (standard potentials + Nernst)",
        provenance_tag="DERIVED", formula="E = E°(cathode) − E°(anode) − (RT/nF)lnQ",
        notes="Spontaneous if E > 0." ,
        inputs={"cathode": cathode, "anode": anode, "Q": reaction_quotient}).to_dict()


def electrolysis_mass(molar_mass_kg: float, current_a: float, time_s: float,
                      electrons: int) -> dict[str, Any]:
    """Mass deposited by electrolysis (kg) — Faraday's first law."""
    from sigma_ground.field.interface.electrochemistry import faraday_mass_deposited
    try:
        m = faraday_mass_deposited(float(molar_mass_kg), float(current_a),
                                   float(time_s), int(electrons))
    except Exception as e:
        return _decline(f"electrolysis_mass failed: {e}").to_dict()
    return ToolResult(
        value=m, units="kg",
        source="sigma_ground.field.interface.electrochemistry (Faraday)",
        provenance_tag="DERIVED", formula="m = M·I·t / (n·F)",
        inputs={"molar_mass_kg": molar_mass_kg, "current_a": current_a,
                "time_s": time_s, "electrons": electrons}).to_dict()


# ── Solutions: colligative + solubility ─────────────────────────────────
def boiling_point_elevation(molality_mol_kg: float,
                            van_t_hoff_i: float = 1.0) -> dict[str, Any]:
    """Boiling-point elevation ΔTb (K) = i·Kb·molality."""
    from sigma_ground.field.interface.solution import boiling_point_elevation as _bpe
    try:
        dt = _bpe(float(molality_mol_kg), float(van_t_hoff_i))
    except Exception as e:
        return _decline(f"boiling_point_elevation failed: {e}").to_dict()
    return ToolResult(value=dt, units="K",
                      source="sigma_ground.field.interface.solution",
                      provenance_tag="DERIVED", formula="ΔTb = i·Kb·m",
                      inputs={"molality_mol_kg": molality_mol_kg, "van_t_hoff_i": van_t_hoff_i}).to_dict()


def freezing_point_depression(molality_mol_kg: float,
                              van_t_hoff_i: float = 1.0) -> dict[str, Any]:
    """Freezing-point depression ΔTf (K) = i·Kf·molality."""
    from sigma_ground.field.interface.solution import freezing_point_depression as _fpd
    try:
        dt = _fpd(float(molality_mol_kg), float(van_t_hoff_i))
    except Exception as e:
        return _decline(f"freezing_point_depression failed: {e}").to_dict()
    return ToolResult(value=dt, units="K",
                      source="sigma_ground.field.interface.solution",
                      provenance_tag="DERIVED", formula="ΔTf = i·Kf·m",
                      inputs={"molality_mol_kg": molality_mol_kg, "van_t_hoff_i": van_t_hoff_i}).to_dict()


def osmotic_pressure(molarity_mol_l: float,
                     van_t_hoff_i: float = 1.0) -> dict[str, Any]:
    """Osmotic pressure π (Pa) = i·M·R·T (van't Hoff)."""
    from sigma_ground.field.interface.solution import osmotic_pressure as _op
    try:
        p = _op(float(molarity_mol_l), float(van_t_hoff_i))
    except Exception as e:
        return _decline(f"osmotic_pressure failed: {e}").to_dict()
    return ToolResult(value=p, units="Pa",
                      source="sigma_ground.field.interface.solution",
                      provenance_tag="DERIVED", formula="π = i·M·R·T",
                      inputs={"molarity_mol_l": molarity_mol_l, "van_t_hoff_i": van_t_hoff_i}).to_dict()


def molar_solubility(salt_key: str) -> dict[str, Any]:
    """Molar solubility (mol/L) of a sparingly-soluble salt from its Ksp."""
    from sigma_ground.field.interface.solution import molar_solubility as _ms, SOLUBILITY_DATA
    if salt_key not in SOLUBILITY_DATA:
        return _decline(f"Unknown salt {salt_key!r}. Available: {sorted(SOLUBILITY_DATA)}",
                        salt_key=salt_key).to_dict()
    try:
        s = _ms(salt_key)
    except Exception as e:
        return _decline(f"molar_solubility failed: {e}", salt_key=salt_key).to_dict()
    return ToolResult(value=s, units="mol/L",
                      source="sigma_ground.field.interface.solution (from measured Ksp)",
                      provenance_tag="DERIVED", formula="s = (Ksp / Π νᵢ^νᵢ)^(1/Σν)",
                      inputs={"salt_key": salt_key}).to_dict()
