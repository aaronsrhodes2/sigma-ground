"""Simulation-playground tools — the STATEFUL (conversation-mode) surface.

Unlike the stateless Q&A tools, these hold a *live scene* across calls: a piece
of matter (an inventory Structure) is loaded into a module-level session store,
then iteratively mutated. The FastMCP server is a long-running process and the
benchmark runner keeps one ClientSession for the whole conversation, so this
in-process store persists across turns.

Seam: Mentat CONSUMES inventory / Materia / radiance; it never edits them. The
mutators are inventory.behaviors.apply_env (the in-place setter cascade that was
excluded from the Q&A coverage surface precisely because it mutates state).

Loop:  playground_load -> playground_inspect / playground_apply (mutate) ->
       playground_simulate (Materia one-shot) / playground_render -> reset.
"""
from __future__ import annotations

import contextlib
import io
import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.inventory (+ materia, radiance)"

# Macroscopic environment knobs apply_env understands (atom/molecule level).
# (quark/particle also accept energy_gev / momentum_gev / color_field.)
_KNOBS = ("temperature_k", "pressure_pa", "electric_field_vm",
          "magnetic_field_t", "energy_ev")

# Module-level live-scene store. handle -> {"material","structure","history"}.
_SCENES: dict[str, dict[str, Any]] = {}


def _safe(fn, *a, **k):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)
    except Exception:
        return None


def _build(material: str):
    """material name -> a freshly resolved inventory Structure (or None)."""
    from sigma_ground.inventory.builder import load_structure_spec, build_structure_from_spec
    spec = _safe(load_structure_spec, material)
    if spec is None:
        return None
    return _safe(build_structure_from_spec, spec)


def _molecules(node, out=None):
    """Depth-first list of every Molecule in a Structure tree."""
    if out is None:
        out = []
    for m in (getattr(node, "molecules", None) or []):
        out.append(m)
    for ch in (getattr(node, "children", None) or []):
        _molecules(ch, out)
    return out


def _find_constituents(structure):
    """First (molecule, particle, quark) anywhere in the tree."""
    mol = part = quark = None
    for m in _molecules(structure):
        mol = mol or m
        for atom in (getattr(m, "atoms", None) or []):
            for p in (getattr(atom, "protons", None) or []) + (getattr(atom, "neutrons", None) or []):
                part = part or p
                qs = getattr(p, "quarks", None) or []
                if qs:
                    quark = quark or qs[0]
    return mol, part, quark


def _snapshot(structure) -> dict[str, Any]:
    """Flat map of the operable (mutable) fields, for before/after diffing.

    Captures the fields apply_env actually moves: bond lengths (thermal
    expansion / pressure compression), atom charge_state + electron count
    (ionization), and an aggregate particle spin (field-driven spin flips).
    """
    snap: dict[str, Any] = {}
    net_spin = 0.0
    for mi, m in enumerate(_molecules(structure)):
        for bi, b in enumerate(getattr(m, "bonds", None) or []):
            ln = getattr(b, "length", None)
            if ln is not None:
                snap[f"mol{mi}.bond{bi}.length_A"] = round(float(ln), 5)
        for ai, atom in enumerate(getattr(m, "atoms", None) or []):
            cs = getattr(atom, "charge_state", None)
            if cs is not None:
                snap[f"mol{mi}.atom{ai}.charge_state"] = cs
            parts = ((getattr(atom, "protons", None) or [])
                     + (getattr(atom, "neutrons", None) or [])
                     + (getattr(atom, "electrons", None) or []))
            snap[f"mol{mi}.atom{ai}.electrons"] = len(getattr(atom, "electrons", None) or [])
            for p in parts:
                s = getattr(p, "spin_projection", None)
                if s is not None:
                    net_spin += float(s)
    snap["net_spin_projection"] = round(net_spin, 4)
    return snap


def _diff(before: dict, after: dict) -> dict[str, Any]:
    changed = {}
    for k in sorted(set(before) | set(after)):
        b, a = before.get(k), after.get(k)
        if b != a:
            changed[k] = {"before": b, "after": a}
    return changed


def _summary(structure) -> dict[str, Any]:
    """Particle counts + mass budget of the current state."""
    from sigma_ground.inventory.checksum.particle_inventory import compute_particle_inventory
    from sigma_ground.inventory.physics import compute_physics
    inv = _safe(compute_particle_inventory, structure) or {}
    phys = _safe(compute_physics, structure) or {}
    return {
        "protons": inv.get("protons"),
        "neutrons": inv.get("neutrons"),
        "electrons": inv.get("electrons"),
        "total_particles": phys.get("N_particles_total"),
        "total_mass_kg": phys.get("total_mass_kg"),
        "mass_defect_kg": phys.get("mass_defect_kg"),
    }


def _result(value, handle, *, units="", notes="", formula="") -> dict[str, Any]:
    return ToolResult(value=value, units=units, source=_SRC,
                      provenance_tag="DERIVED", notes=notes, formula=formula,
                      inputs={"handle": handle}).to_dict()


def _missing(handle: str) -> dict[str, Any]:
    return ToolResult(
        value=None, source=_SRC, provenance_tag="SPECULATIVE-PENDING",
        notes=(f"no scene loaded at handle {handle!r}. "
               f"call playground_load first. open scenes: {sorted(_SCENES)}"),
        inputs={"handle": handle}).to_dict()


# ── tools ──────────────────────────────────────────────────────────────────

def playground_load(material: str = "water_molecule",
                    handle: str = "scene") -> dict[str, Any]:
    """Load a piece of matter into a LIVE scene (persists across turns). Returns
    its formula, proton/neutron/electron counts, mass, and the tunable knobs.
    Materials: water_molecule, hydrogen_atom, bronze_cube, gold_ring,
    earths_layers, tungsten_cube, seawater_liter, universe, ...
    e.g. playground_load('bronze_cube')."""
    st = _build(material)
    if st is None:
        from sigma_ground.inventory import list_structures
        ids = [s.get("id") for s in (_safe(list_structures) or [])]
        return ToolResult(
            value=None, source=_SRC, provenance_tag="SPECULATIVE-PENDING",
            notes=f"unknown material {material!r}. available: {ids}",
            inputs={"material": material, "handle": handle}).to_dict()
    _SCENES[handle] = {"material": material, "structure": st, "history": []}
    mols = _molecules(st)
    val = {
        "handle": handle,
        "material": material,
        "formula": (getattr(mols[0], "formula", None) if mols else None),
        **_summary(st),
        "tunable_knobs": list(_KNOBS),
    }
    return _result(val, handle, units="counts, kg",
                   notes=f"loaded {material!r} into scene {handle!r}; "
                         f"tune it with playground_apply.")


def playground_inspect(handle: str = "scene",
                       scope: str = "summary") -> dict[str, Any]:
    """Query the CURRENT (possibly mutated) state of a loaded scene. scope:
    'summary' (counts + mass), 'matter' (bond lengths + atom charge states),
    'constituents' (behaviors of a representative quark / particle / molecule).
    e.g. playground_inspect('scene', 'matter')."""
    sc = _SCENES.get(handle)
    if sc is None:
        return _missing(handle)
    st = sc["structure"]
    if scope == "matter":
        val = {"operable_fields": _snapshot(st),
               "applied_history": sc["history"]}
    elif scope == "constituents":
        from sigma_ground.inventory.behaviors import behaviors
        mol, part, quark = _find_constituents(st)
        bm = _safe(behaviors, mol) if mol else None
        bp = _safe(behaviors, part) if part else None
        bq = _safe(behaviors, quark) if quark else None
        val = {
            "quark": ({k: bq.get(k) for k in ("flavor", "color", "spin_projection",
                                              "constituent_mass_mev")}
                      if isinstance(bq, dict) else None),
            "particle": ({k: bp.get(k) for k in ("particle_type", "charge_e",
                                                 "spin_projection")}
                         if isinstance(bp, dict) else None),
            "molecule": ({k: bm.get(k) for k in ("formula", "bond_summary")}
                         if isinstance(bm, dict) else None),
        }
    else:
        val = {**_summary(st), "applied_history": sc["history"]}
    return _result(val, handle, notes=f"scene {handle!r} ({sc['material']}), scope={scope}")


def playground_apply(handle: str = "scene",
                     environment: dict[str, float] | None = None,
                     mode: str = "update") -> dict[str, Any]:
    """Apply an environment to the live scene and MUTATE it in place (cascades
    to molecules/atoms/particles/quarks). Knobs: temperature_k, pressure_pa,
    electric_field_vm, magnetic_field_t, energy_ev. mode 'update' (absolute) or
    'delta' (relative). Returns a DIFF of every field that changed.
    e.g. playground_apply('scene', {'temperature_k': 1500})."""
    sc = _SCENES.get(handle)
    if sc is None:
        return _missing(handle)
    environment = environment or {}
    if not environment:
        return ToolResult(
            value=None, source=_SRC, provenance_tag="SPECULATIVE-PENDING",
            notes=f"specify an environment, e.g. {{'temperature_k': 1500}}. knobs: {list(_KNOBS)}",
            inputs={"handle": handle}).to_dict()
    from sigma_ground.inventory.behaviors import apply_env
    st = sc["structure"]
    before = _snapshot(st)
    log = _safe(apply_env, st, environment, mode)
    after = _snapshot(st)
    diff = _diff(before, after)
    sc["history"].append({"environment": environment, "mode": mode,
                          "changed": list(diff)})
    val = {"applied": environment, "mode": mode, "changed": diff,
           "n_fields_changed": len(diff)}
    note = (f"applied {environment} ({mode}) to {handle!r}; {len(diff)} field(s) "
            f"changed and persist.")
    if log is None:
        note = "[apply_env raised] " + note
    return _result(val, handle, notes=note,
                   formula="apply_env cascade: thermal/pressure bond change, "
                           "ionization, spin flips")


def playground_simulate(handle: str = "scene",
                        scenario: str = "drop it from 10 km altitude") -> dict[str, Any]:
    """Run a Materia one-shot dynamics scenario on the loaded body (drop, launch,
    heat, orbit, ...). Bridges the macroscopic simulator to the loaded matter by
    its material identity. e.g. playground_simulate('scene', 'drop it from 10 km')."""
    sc = _SCENES.get(handle)
    if sc is None:
        return _missing(handle)
    from sigma_ground.mcp.tools.simulation import simulate as _sim
    res = _sim(scenario)
    out = res.to_dict() if hasattr(res, "to_dict") else res
    # annotate with the scene it ran against
    if isinstance(out, dict):
        out.setdefault("inputs", {})
        out["inputs"]["scene_handle"] = handle
        out["inputs"]["scene_material"] = sc["material"]
        note = out.get("notes") or ""
        out["notes"] = (f"[scene {handle!r}: {sc['material']}] " + note).strip()
    return out


def _material_key(structure) -> str:
    """Resolve a radiance material key from a Structure (best-effort)."""
    _KNOWN = {"copper", "water", "gold", "iron", "silicon", "aluminum"}
    _MAP = {"h2o": "water", "cu": "copper", "fe": "iron", "au": "gold",
            "si": "silicon", "al": "aluminum", "ice": "water",
            "bronze": "copper", "metal": "copper", "seawater": "water"}
    cands = [getattr(structure, "formula", "") or "",
             getattr(structure, "material_class", "") or ""]
    mols = _molecules(structure)
    if mols:
        cands.append(getattr(mols[0], "formula", "") or "")
    for c in cands:
        cl = str(c).strip().lower()
        if cl in _KNOWN:
            return cl
        if cl in _MAP:
            return _MAP[cl]
    return "copper"  # known-good albedo fallback


def playground_render(handle: str = "scene", mode: str = "ascii",
                      size: int = 48) -> dict[str, Any]:
    """Render a quick visual of the current scene as a material-colored,
    surface-lit sphere (bulk-shape approximation, NOT atomic geometry). mode
    'ascii' returns terminal art; 'png' writes a file and returns its path.
    e.g. playground_render('scene')."""
    sc = _SCENES.get(handle)
    if sc is None:
        return _missing(handle)
    st = sc["structure"]
    from sigma_ground.radiance import RadianceScene, render_ascii, render_to_png, Camera, orbit_eye
    from sigma_ground.radiance.camera import Vec3
    from sigma_ground.shapes import Sphere

    mass = getattr(st, "resolved_mass_kg", None) or getattr(st, "mass_kg", None) or 0.0
    rho = getattr(st, "standard_density", None) or 0.0
    if mass > 0 and rho > 0:
        radius = (3.0 * mass / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    else:
        radius = 1.0  # illustrative unit sphere (e.g. single molecule)
    key = _material_key(st)
    scene = _safe(RadianceScene.from_shape, Sphere(radius), key)
    if scene is None:
        return ToolResult(value=None, source="sigma_ground.radiance",
                          provenance_tag="SPECULATIVE-PENDING",
                          notes=f"could not build a scene for {handle!r}",
                          inputs={"handle": handle}).to_dict()
    target = Vec3(0.0, 0.0, 0.0)
    eye = orbit_eye(target, 3.0 * radius, 35.0, 20.0)
    w = max(16, int(size)); h = max(8, int(size // 2))
    note = (f"bulk-shape render of {sc['material']} as a {key}-colored sphere "
            f"(r={radius:.3g} m from mass/density); illustrative, not atomic geometry.")
    if mode == "png":
        cam = Camera(eye, target, width=4 * w, height=4 * h)
        path = _safe(render_to_png, scene, cam, f".mentat_render_{handle}.png")
        return _result(path, handle, units="png path", notes=note)
    cam = Camera(eye, target, width=w, height=h)
    art = _safe(render_ascii, scene, cam) or ""
    return _result(art, handle, units="ascii", notes=note)


def playground_reset(handle: str = "scene") -> dict[str, Any]:
    """Rebuild the scene's matter to its pristine (un-mutated) state, clearing
    the applied-environment history. e.g. playground_reset('scene')."""
    sc = _SCENES.get(handle)
    if sc is None:
        return _missing(handle)
    st = _build(sc["material"])
    if st is None:
        return _missing(handle)
    _SCENES[handle] = {"material": sc["material"], "structure": st, "history": []}
    return _result({"handle": handle, "material": sc["material"], **_summary(st)},
                   handle, notes=f"scene {handle!r} reset to pristine {sc['material']!r}.")


def playground_status() -> dict[str, Any]:
    """List the live scenes in this session, their materials, and applied
    history. e.g. playground_status()."""
    scenes = {h: {"material": sc["material"],
                  "mutations_applied": len(sc["history"]),
                  "history": sc["history"]}
              for h, sc in _SCENES.items()}
    return ToolResult(value={"open_scenes": list(_SCENES), "scenes": scenes},
                      source=_SRC, provenance_tag="DERIVED",
                      notes=f"{len(_SCENES)} live scene(s).",
                      inputs={}).to_dict()


def playground_clear(handle: str | None = None) -> dict[str, Any]:
    """Drop a scene (or all scenes if handle is omitted) from the session.
    e.g. playground_clear('scene')."""
    if handle is None:
        n = len(_SCENES)
        _SCENES.clear()
        return ToolResult(value={"cleared": n}, source=_SRC,
                          provenance_tag="DERIVED", notes="cleared all scenes.",
                          inputs={}).to_dict()
    existed = _SCENES.pop(handle, None) is not None
    return ToolResult(value={"cleared": handle, "existed": existed}, source=_SRC,
                      provenance_tag="DERIVED",
                      notes=(f"dropped scene {handle!r}." if existed
                             else f"no scene {handle!r}."),
                      inputs={"handle": handle}).to_dict()
