"""Quarksum inventory analysis tools (particle bookkeeping & mass closure).

Composite tools cascading through the inventory engine: build a structure from a
named spec, then resolve materials -> molecules -> atoms -> particles -> quarks,
count the particles, compute the mass budget, and read off constituent behaviors.

The env-resolver / setter cascade (apply_env, resolve_*_env) and the low-level
checksum walkers are NOT exposed -- they are state-mutation (Materia's lane) and
verification plumbing. See misc/COVERAGE_LEDGER.md.
"""
from __future__ import annotations

import contextlib
import io
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.inventory"


def _safe(fn, *a, **k):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
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
        with contextlib.redirect_stdout(io.StringIO()):
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


def _build(structure_name: str):
    from sigma_ground.inventory.builder import load_structure_spec, build_structure_from_spec
    spec = _safe(load_structure_spec, structure_name)
    if spec is None:
        return None
    return _safe(build_structure_from_spec, spec)


def _find_constituents(node, found=None):
    """Depth-first: return (molecule, particle, quark) found anywhere in the tree."""
    if found is None:
        found = [None, None, None]
    for m in (getattr(node, "molecules", None) or []):
        found[0] = found[0] or m
        for atom in (getattr(m, "atoms", None) or []):
            for p in (getattr(atom, "protons", None) or []) + (getattr(atom, "neutrons", None) or []):
                found[1] = found[1] or p
                qs = getattr(p, "quarks", None) or []
                if qs:
                    found[2] = found[2] or qs[0]
    for ch in (getattr(node, "children", None) or []):
        _find_constituents(ch, found)
    return found


def material_inventory_analysis(structure_name: str = "water_molecule") -> dict[str, Any]:
    """Quarksum particle inventory & mass closure of a structure: counts of
    protons / neutrons / electrons (and total particles / baryons), the total
    mass, the mass defect (nuclear + QCD binding), and GM. Structures include
    water_molecule, hydrogen_atom, bronze_cube, gold_ring, earths_layers,
    universe, etc. e.g. material_inventory_analysis('water_molecule')."""
    from sigma_ground.inventory.checksum.particle_inventory import compute_particle_inventory
    from sigma_ground.inventory.physics import compute_physics, compute_tangle
    st = _build(structure_name)
    if st is None:
        return ToolResult(value=None, source=_SRC, provenance_tag="SPECULATIVE-PENDING",
                          notes=f"unknown structure {structure_name!r}",
                          inputs={"structure_name": structure_name}).to_dict()
    inv, err_inv = _safe_call(compute_particle_inventory, st)
    inv = inv or {}
    phys, err_phys = _safe_call(compute_physics, st)
    phys = phys or {}
    tangle, err_tangle = _safe_call(compute_tangle, st)
    results = {
        "protons": inv.get("protons"),
        "neutrons": inv.get("neutrons"),
        "electrons": inv.get("electrons"),
        "total_particles": phys.get("N_particles_total"),
        "baryons": phys.get("N_baryons"),
        "total_mass_kg": phys.get("total_mass_kg"),
        "mass_defect_kg": phys.get("mass_defect_kg"),
        "GM_m3_s2": phys.get("GM_m3_s2"),
        "tangle_present": tangle is not None,
    }
    errors = {}
    if err_inv:
        errors.update({k: err_inv for k in ("protons", "neutrons", "electrons")})
    if err_phys:
        errors.update({k: err_phys for k in
                       ("total_particles", "baryons", "total_mass_kg", "mass_defect_kg", "GM_m3_s2")})
    if err_tangle:
        errors["tangle_present"] = err_tangle
    return ToolResult(value=results, units="counts, kg, m^3/s^2", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="materials -> molecules -> atoms -> particles -> quarks; sum masses; defect = constituents - total",
                      notes=_notes_from_errors(errors),
                      inputs={"structure_name": structure_name}).to_dict()


def constituent_behaviors_analysis(structure_name: str = "water_molecule") -> dict[str, Any]:
    """Physical behaviors of a structure's constituents: the QCD behaviors of a
    representative quark (flavor, color, bare vs constituent mass), a subatomic
    particle (type, rest mass, charge), and a molecule (formula, bonds).
    e.g. constituent_behaviors_analysis('water_molecule')."""
    from sigma_ground.inventory.behaviors import behaviors
    st = _build(structure_name)
    if st is None:
        return ToolResult(value=None, source=_SRC, provenance_tag="SPECULATIVE-PENDING",
                          notes=f"unknown structure {structure_name!r}",
                          inputs={"structure_name": structure_name}).to_dict()
    mol, part, quark = _find_constituents(st)
    bm, err_bm = _safe_call(behaviors, mol) if mol else (None, None)
    bp, err_bp = _safe_call(behaviors, part) if part else (None, None)
    bq, err_bq = _safe_call(behaviors, quark) if quark else (None, None)
    results = {
        "quark": ({k: bq.get(k) for k in ("flavor", "color", "bare_mass_mev", "constituent_mass_mev")}
                  if isinstance(bq, dict) else None),
        "particle": ({k: bp.get(k) for k in ("particle_type", "symbol", "rest_mass_kg", "charge_e")}
                     if isinstance(bp, dict) else None),
        "molecule": ({k: bm.get(k) for k in ("formula", "bond_summary")}
                     if isinstance(bm, dict) else None),
    }
    errors = {k: e for k, e in
             (("quark", err_bq), ("particle", err_bp), ("molecule", err_bm)) if e}
    return ToolResult(value=results, units="MeV, kg, e", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="behaviors(entity): intrinsic + operable physics per scope",
                      notes=_notes_from_errors(errors),
                      inputs={"structure_name": structure_name}).to_dict()


def planet_moment_of_inertia_analysis(structure_name: str = "earths_layers",
                                      planet_radius_km: float = 6371.0) -> dict[str, Any]:
    """Moment-of-inertia factor C/MR^2 of a layered planet, derived from the
    inventory composition of its shells (densities x geometry). Earth ~ 0.331.
    e.g. planet_moment_of_inertia_analysis('earths_layers', 6371)."""
    from sigma_ground.inventory.derived import derive_moi_factor
    r, err = _safe_call(derive_moi_factor, structure_name, planet_radius_km)
    results = {
        "c_over_mr2": getattr(r, "c_over_mr2", None),
        "total_mass_kg": getattr(r, "total_mass_kg", None),
        "moi_C_kgm2": getattr(r, "moi_C_kgm2", None),
    }
    errors = {k: err for k in results} if err else {}
    return ToolResult(value=results, units="dimensionless, kg, kg.m^2", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="C/MR^2 = sum shell MoI / (M R^2), shells from inventory densities",
                      notes=_notes_from_errors(errors),
                      inputs={"structure_name": structure_name,
                              "planet_radius_km": planet_radius_km}).to_dict()
