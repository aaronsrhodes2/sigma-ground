"""Deckard — the shape researcher (the matter compiler).

Named for the Diablo NPC who identifies unidentified items: Deckard turns a
*name* ("coffee cup") into grounded *matter*. Two beats:

  1. RESEARCH  — resolve a named object to a concrete, cited ConstructSpec
                 (research.py): a frozen markdown catalog hit, else the
                 Researcher (LLM grounded by our data + free factual APIs),
                 else a flagged best-guess. Never a confident fake.
  2. COMPILE   — fit the primitive kit and build the layered CSG/SDF construct,
                 then integrate mass / centre-of-mass / inertia (construct.py),
                 cross-checking the SDF integrator against the closed-form
                 volumes so the matter is provably right.

Deckard's compiled `Construct` is consumed by TWO layers — Materia (to move it)
and Radiance (to render it) — so it lives in the matter core, below both, and
reuses the geometry kernel (`sigma_ground.kernel`).

It researches *form-facts* (dimensions, proportions, topology) and fits
primitives to them; it never ingests a foreign mesh.
"""
from __future__ import annotations

from . import catalog
from .schema import (ConstructSpec, Fact, SpecLayer, Part,
                     emit_markdown, parse_markdown)
from .research import research, voxel_plan, CATALOG
from .construct import compile, compile_vessel, Construct, Layer
from .selfaudit import audit

# Back-compat: ItemSpec is now the generalised ConstructSpec.
ItemSpec = ConstructSpec


def _auto_pitch(longest_m: float) -> float:
    """Voxel cell so the longest axis is ~45 cells — recognizable shape, compact
    grid (keeps the serialized 3-D texture small). Clamped to a sane range."""
    return min(0.025, max(0.0035, longest_m / 45.0))


def _voxel_identify(name, *, material_hint=None, exclude=None, pitch=None):
    """Try the REAL-MESH voxel path: resolve the name to a PartNet model, load its
    actual per-part meshes, voxelize, and build a drop-in voxel Construct. Returns
    None (caller falls back to primitives) when the name isn't a real category, the
    mesh data isn't on disk, or the opt-in [shapes] deps are missing."""
    plan = voxel_plan(name, material_hint=material_hint, exclude=exclude)
    if plan is None:
        return None
    try:
        from .sources import mesh as meshsrc
        from .voxelize import voxelize, construct_from_field
    except Exception:
        return None                                   # [shapes] extra not installed
    anno = plan["anno_id"]
    if not meshsrc.data_available(anno):
        return None                                   # local 242 GB not present
    parts = meshsrc.load_parts(anno, scale_to_m=plan["scale_to_m"])
    if not parts:
        return None
    pitch = pitch or _auto_pitch(plan["scale_to_m"])
    field = voxelize([(m, plan["material"]) for m, _ in parts], pitch=pitch)
    src = (f"PartNet anno_id {anno} — real mesh voxelized @ {pitch*1000:.0f}mm "
           f"(material {plan['material']}, cited {plan.get('license','')})")
    return construct_from_field(name, field, source=src, identified=True)


def identify(name: str, resolution: int = 64, *, allow_llm: bool = True,
             fidelity: str = "primitive", material_hint: str | None = None,
             exclude=None) -> Construct:
    """Full Deckard pipeline: a name → researched, compiled, validated matter.

    ``fidelity``:
      • ``"primitive"`` (default) — fit the analytic primitive kit (clean,
        parametric, refractive; the no-mesh path). Unchanged behaviour.
      • ``"voxel"`` / ``"auto"`` — load the REAL ShapeNet/PartNet mesh and
        voxelize it (a real chair, not a rod+box), falling back to primitives
        when no mesh/extra is available. Opt-in so existing callers are untouched.

    ``allow_llm=False`` forces the deterministic path (catalog hit else flagged
    best-guess) — no network. ``material_hint`` SELECTS the material (never
    invents); ``exclude`` skips already-shown anno_ids so "a different chair"
    grabs a fresh real model.
    """
    if fidelity in ("voxel", "auto"):
        vc = _voxel_identify(name, material_hint=material_hint, exclude=exclude)
        if vc is not None:
            return vc                                 # real mesh, voxelized
        # else: fall through to the primitive path (honest fallback)
    return compile(research(name, allow_llm=allow_llm, material_hint=material_hint,
                            exclude=exclude), resolution=resolution)


__all__ = [
    "identify", "research", "catalog", "CATALOG",
    "ConstructSpec", "ItemSpec", "Fact", "SpecLayer", "Part",
    "emit_markdown", "parse_markdown",
    "compile", "compile_vessel", "Construct", "Layer", "audit",
]
