"""Deckard — multi-OBJECT scene composition (stacking independently-researched
Constructs into ONE renderable assembly).

``ask_deckard.request_shape_scene()`` (radiance/ask_deckard.py) turns ONE name
into ONE Construct. Real asks are compound: "a doughnut ... sitting on a
plate, which is flat on top of a ... table" is THREE independently-researched
objects with an explicit spatial relationship (bottom-of-one on top-of-the-
next). This module is the multi-object counterpart: research each object on
its own (exactly as ``ask_deckard`` does for one), then STACK the compiled
Constructs and merge them into a single Construct. Deckard stops at matter:
the bake to a render-ready SceneSpec is Radiance's side of the seam —
``radiance.ask_deckard.compose_scene()`` calls ``stack_constructs`` here and
hands the merged Construct to ``scene_export.construct_to_scene()`` (same
tier direction as the single-object ``request_shape_scene``).

Stacking reuses construct.py's OWN whole-object anchor semantics rather than
inventing a second positioning system:

  * ``Part``-level attach (``construct.py``'s ``_local_anchor``/
    ``_resolve_positions``) already anchors "top"/"bottom" of ONE part as
    "centred in x/y, at the extreme +/-z of that part's own dims" — see
    ``_local_anchor``'s box/cylinder/sphere cases.
  * A compiled ``Construct`` already carries the SAME shape of information
    for the WHOLE object: ``construct.bbox`` is centred-in-x/y, extremal-in-z
    over every part, by construction (``_compile_parts``/``_compile_vessel``
    build it that way). So ``_object_anchor`` below computes a whole-object
    top/bottom anchor with the IDENTICAL semantics as ``_local_anchor``,
    just generalised from one part's half-extents to the compiled object's
    full bbox — no bounding box is hand-derived from scratch here.

Positioning itself translates the WHOLE compiled Construct (every leaf shape,
recursively, by one uniform offset) rather than re-deriving a spec and
recompiling — this works uniformly for BOTH of Deckard's construct kinds
(the ``layered_vessel`` cup kit, which has no ``Part``/attach concept at all,
and the general ``parts`` primitive kit), since translation happens after
compile, on the shapes construct.py already produced.

Infallibility (the same discipline as ``ask_deckard.py``): if ANY object in
the chain fails to ground (``ConstructSpec.identified`` is False, or research/
compile raises), the whole stack refuses rather than rendering a partial
fake — one ungroundable object should not silently vanish from "a doughnut on
a plate on a table".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .construct import Construct, Layer, _LayerStack, compile as _compile_construct
from .research import research as _research
from ..kernel.csg import ComposedSDF


@dataclass
class SceneItem:
    """One object in a bottom-to-top stack.

    ``name`` is researched via ``deckard.research()`` exactly as a single-
    object ``ask_deckard.request_shape_scene()`` call would. ``material_hint``
    SELECTS a cited material variant (see ``research.research``'s own
    contract) — it never invents one. ``stack=True`` (the default for every
    item after the first) means "translate this object so its bottom anchor
    sits flush on the PREVIOUS item's top anchor"; the first item in the
    list is always the base of the stack and keeps its own researched
    placement (``stack`` is meaningless for index 0). ``dx``/``dy`` are an
    optional horizontal nudge (metres) applied on top of the automatic
    centre-on-centre alignment — e.g. an off-centre doughnut on a plate.
    """
    name: str
    material_hint: str | None = None
    stack: bool = True
    dx: float = 0.0
    dy: float = 0.0


# ── "1700s table" — a documented STYLE convention, not period-accurate design.
# research()'s material_hint parameter (see research._HINT_MATERIAL) only
# understands plain material words ("oak", "steel", "wood") — it has NO
# notion of era or style. So a period/style word in an object's name is
# mapped here to the closest honest material_hint it already knows how to
# SELECT (never invents a wood species/finish it can't cite). This is a
# convention scoped to multi-object scene composition, applied ONLY when the
# caller didn't already supply an explicit material_hint (an explicit hint
# always wins).
_STYLE_MATERIAL_HINT = {
    "1700s": "oak", "1600s": "oak", "1800s": "oak",
    "17th century": "oak", "18th century": "oak", "19th century": "oak",
    "antique": "oak", "colonial": "oak", "victorian": "oak",
    "georgian": "oak", "period": "oak",
}


def _style_hint(name: str) -> str | None:
    """A material_hint implied by a period/style word in ``name``, or None.
    See ``_STYLE_MATERIAL_HINT`` — an honest-approximation convention, not a
    claim of period-accurate construction."""
    low = name.lower()
    for word, hint in _STYLE_MATERIAL_HINT.items():
        if word in low:
            return hint
    return None


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "obj"


def _object_anchor(bbox, which: str):
    """Whole-object anchor point, at the top or bottom of a compiled
    Construct's bbox — SAME semantics as construct.py's ``_local_anchor``
    top/bottom (centred in x/y, at the extreme z), generalised from one
    part's half-extents to the compiled object's full bbox."""
    (x0, x1), (y0, y1), (z0, z1) = bbox
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return (cx, cy, z1 if which == "top" else z0)


# ── translate a WHOLE compiled Construct by one uniform offset ─────────────
def _translate_shape(shape, offset, seen: set) -> None:
    """Shift ``shape.center`` (and anything it wraps) by ``offset``, IN
    PLACE — recurses into ``_base``/``_cut`` (construct.py's ``_Rotated``/
    ``_Clipped``/``_Subtracted`` wrappers) so a rotated or fill-clipped part
    moves as one rigid piece, and also shifts a ``_Clipped`` fill-plane's
    absolute ``_level`` along its axis so a liquid fill keeps its fraction.
    ``seen`` (by object identity) guards against double-shifting a shape
    object that TWO leaves both reference — the liquid/gas halves of one
    fluid fill share the same cavity ``_base`` instance (see
    ``construct._expand_fill``)."""
    if id(shape) in seen:
        return
    seen.add(id(shape))
    dx, dy, dz = offset
    cx, cy, cz = shape.center
    shape.center = (cx + dx, cy + dy, cz + dz)
    base = getattr(shape, "_base", None)
    if base is not None:
        _translate_shape(base, offset, seen)
    cut = getattr(shape, "_cut", None)
    if cut is not None:
        _translate_shape(cut, offset, seen)
    if hasattr(shape, "_level") and hasattr(shape, "_axis"):
        shape._level = shape._level + offset[shape._axis]


def _translate_construct(construct: Construct, offset) -> Construct:
    """Translate every leaf shape of a compiled Construct by ``offset``,
    in place, and keep its cached ``bbox``/``com_m`` consistent. Pure
    translation of a rigid body leaves its inertia about its OWN centre of
    mass unchanged, so ``inertia_kgm2`` needs no adjustment here."""
    seen: set = set()
    for leaf, _op in construct.composed._leaves:
        _translate_shape(leaf.shape, offset, seen)
    dx, dy, dz = offset
    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    construct.bbox = ((x0 + dx, x1 + dx), (y0 + dy, y1 + dy), (z0 + dz, z1 + dz))
    cx, cy, cz = construct.com_m
    construct.com_m = (cx + dx, cy + dy, cz + dz)
    return construct


def _normalize_items(items) -> list:
    out = []
    for it in items:
        if isinstance(it, SceneItem):
            out.append(it)
        elif isinstance(it, str):
            out.append(SceneItem(name=it))
        else:
            vals = list(it)
            name = vals[0]
            hint = vals[1] if len(vals) > 1 else None
            stack = vals[2] if len(vals) > 2 else True
            out.append(SceneItem(name=name, material_hint=hint, stack=stack))
    return out


def _merge_constructs(built: list) -> Construct:
    """Fold a bottom-to-top list of (Construct, name_prefix) — each ALREADY
    translated into its final world position — into ONE synthetic Construct
    that ``scene_export.construct_to_scene`` can export as a single SceneSpec.

    Every part/layer NAME is namespaced with its object's prefix so two
    independently-researched objects that happen to both name a part "body"
    (or a vessel's "glaze"/"ceramic"/"water"/"air" layer labels) never
    collide once folded into one ``ComposedSDF``. Mass sums exactly (rigid
    translation doesn't change it); centre of mass is the mass-weighted
    average of each object's own (already-shifted) CoM; inertia about the
    MERGED CoM is the standard parallel-axis sum over each object's own
    (unchanged-by-translation) inertia about ITS OWN CoM.
    """
    if not built:
        raise ValueError("scene._merge_constructs: nothing to merge")

    stack = _LayerStack()
    density_by_label: dict = {}
    layers: list = []
    mass = 0.0
    Sx = Sy = Sz = 0.0
    xs: list = []
    ys: list = []
    zs: list = []
    all_identified = True
    sources_note = []

    for construct, prefix in built:
        for leaf, op in construct.composed._leaves:
            stack.add(leaf.shape, f"{prefix}{leaf.material}", op)
        for label, rho in construct.density_by_label.items():
            density_by_label[f"{prefix}{label}"] = rho
        for L in construct.layers:
            layers.append(Layer(f"{prefix}{L.name}", L.material, L.density_kg_m3,
                                L.volume_m3, L.mass_kg, L.source))
        m = construct.mass_kg
        cx, cy, cz = construct.com_m
        mass += m
        Sx += m * cx; Sy += m * cy; Sz += m * cz
        (x0, x1), (y0, y1), (z0, z1) = construct.bbox
        xs += [x0, x1]; ys += [y0, y1]; zs += [z0, z1]
        all_identified = all_identified and construct.identified
        sources_note.append(f"{construct.name}: {construct.source}")

    if mass <= 0.0:
        raise ValueError("scene._merge_constructs: merged assembly has zero mass")
    com = (Sx / mass, Sy / mass, Sz / mass)

    Ixx = Iyy = Izz = 0.0
    for construct, _prefix in built:
        ix, iy, iz = construct.inertia_kgm2
        cx, cy, cz = construct.com_m
        ddx, ddy, ddz = cx - com[0], cy - com[1], cz - com[2]
        m = construct.mass_kg
        Ixx += ix + m * (ddy * ddy + ddz * ddz)
        Iyy += iy + m * (ddx * ddx + ddz * ddz)
        Izz += iz + m * (ddx * ddx + ddy * ddy)

    bbox = ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))
    composed = ComposedSDF(stack)
    name = " on ".join(c.name for c, _p in reversed(built))
    validation = {
        "mode": "multi_object_stack", "passed": all_identified,
        "objects": [{"name": c.name, "prefix": p, "bbox": c.bbox,
                     "identified": c.identified} for c, p in built],
    }
    return Construct(
        name=name, composed=composed, density_by_label=density_by_label,
        layers=layers, mass_kg=mass, com_m=com, inertia_kgm2=(Ixx, Iyy, Izz),
        bbox=bbox, validation=validation, identified=all_identified,
        source="; ".join(sources_note))


def stack_constructs(items, *, allow_llm: bool = True, resolution: int = 64) -> Construct:
    """Research + compile each item independently, stack each one's bottom
    anchor flush on the previous item's top anchor, and merge into ONE
    Construct.

    ``items``: an ordered, BOTTOM-TO-TOP list — ``[SceneItem, ...]``, plain
    name strings, or ``(name, material_hint, stack)`` tuples (any tail may be
    omitted). The first item is the base of the stack and keeps its own
    researched placement.

    Honest refusal, not a fabricated stand-in: raises ``ValueError`` if any
    item's ``ConstructSpec`` doesn't ground (``identified`` is False) — the
    SAME infallibility contract as ``ask_deckard.request_shape_scene``
    (ungroundable -> refuse, never render a partial fake).
    """
    norm = _normalize_items(items)
    if not norm:
        raise ValueError("stack_constructs: no items given")

    built: list = []
    prev_bbox = None
    for i, item in enumerate(norm):
        hint = item.material_hint or _style_hint(item.name)
        spec = _research(item.name, allow_llm=allow_llm, material_hint=hint)
        if not spec.identified:
            names = ", ".join(s.get("name", "?") for s in spec.sources)
            raise ValueError(
                f"stack_constructs: '{item.name}' did not ground to a real "
                f"object (best-guess only) — refusing rather than faking the "
                f"stack. Sources tried: {names}")
        construct = _compile_construct(spec, resolution=resolution)
        if not construct.identified:
            raise ValueError(
                f"stack_constructs: '{item.name}' compiled but lost its "
                "identified flag — refusing")

        if i > 0 and item.stack and prev_bbox is not None:
            top = _object_anchor(prev_bbox, "top")
            bottom = _object_anchor(construct.bbox, "bottom")
            offset = (top[0] - bottom[0] + item.dx,
                     top[1] - bottom[1] + item.dy,
                     top[2] - bottom[2])
            _translate_construct(construct, offset)

        built.append((construct, f"{_slug(item.name)}{i}__"))
        prev_bbox = construct.bbox

    return _merge_constructs(built)


__all__ = ["SceneItem", "stack_constructs"]
