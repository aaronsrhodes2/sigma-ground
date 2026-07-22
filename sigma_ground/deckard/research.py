"""Deckard's shape research — a name → a concrete, cited ConstructSpec.

Order of resolution:
  1. a frozen catalog hit (``catalog/<slug>.md``) — deterministic, offline;
  2. on a miss, the Researcher (``researcher.py``: local qwen, prompt grounded by
     a free Wikipedia extract + our own data/Wikidata) synthesises and freezes one;
  3. if that is unavailable or fails, a flagged best-guess (identified=False).

The Deckard–Cain discipline: a partial ID is allowed, a confident fake is not.
"""
from __future__ import annotations

from . import catalog
from .schema import ConstructSpec, Fact, SpecLayer

# Back-compat: the old in-code CATALOG (name -> builder) is now the markdown
# catalog; expose the alias map for introspection.
CATALOG = catalog.ALIASES


def _generic_vessel(name: str) -> ConstructSpec:
    """A flagged best-guess small vessel — used when nothing identifies ``name``."""
    def est(v, c=0.3):
        return Fact(v, "estimated", "", c)
    return ConstructSpec(
        name=name, kind="layered_vessel", identified=False,
        geometry={
            "outer_radius_m": est(0.040), "height_m": est(0.095),
            "wall_m": est(0.005), "glaze_m": est(0.0003),
            "base_m": est(0.007), "fill_fraction": est(0.80),
        },
        layers=[
            SpecLayer("glaze", "glaze (glassy)", est(2400.0), est(0.0003),
                      ["air", "ceramic"]),
            SpecLayer("ceramic", "stoneware", est(2300.0), est(0.005),
                      ["glaze", "air", "water"]),
            SpecLayer("water", "liquid water", est(998.0), est(0.030),
                      ["ceramic", "air"]),
        ],
        sources=[{"name": "no catalog/DB entry — defaulted to a generic small vessel",
                  "license": ""}],
        notes="Unidentified: best-guess proportions of a generic small vessel.",
    )


def _scaffold_from_composition(name: str) -> ConstructSpec | None:
    """A flagged multi-part placeholder from a KNOWN part decomposition, used when
    the model could not shape the object. With a geometry-ENRICHED prior (PartNet
    medians: size_frac/z_frac/r_frac/count) the placeholder gets REAL proportions
    and placement — measured census shape, still identified=False because THIS
    object's specifics are unverified. Without enrichment, the old disjoint stack.
    Honest either way: parts known, the rest census-or-guess — never a confident
    fake."""
    import math
    from . import sources
    from .schema import Part
    got = sources.composition_of(name)
    if not got:
        return None
    part_list, source, lic = got

    # material prior: what the census says this object is made of (cited density)
    material, dens = "plastic", None
    mats = sources.shapenetsem.materials_of(name)
    if mats:
        for cand in mats[0]:
            d = sources.density_of(cand, allow_web=False)
            if d is not None:
                material, dens = cand, d
                break
    if dens is None:
        dens = sources.density_of("plastic", allow_web=False) \
            or Fact(950.0, "estimated", "", 0.2)

    enriched = [p for p in part_list if isinstance(p.get("size_frac"), list)]
    if enriched:
        # overall extents: Sem real medians, else typical size as a cube, else 25 cm
        sem = sources.shapenetsem.dims_of(name)
        if sem:
            W, D, H = sem[0]
        else:
            got_sz = sources.typical_size_of(name)
            W = D = H = (got_sz[0] if got_sz else 0.25)
        parts = []
        zs = []
        for pp in enriched:
            sf = pp["size_frac"]
            tx, ty, tz = (max(sf[i] * (W, D, H)[i], 0.002) for i in range(3))
            shape = pp.get("shape") or "box"
            if shape == "box":
                dims_raw = {"x_m": tx, "y_m": ty, "z_m": tz}
            elif shape in ("cylinder", "cone"):
                dims_raw = {"radius_m": (tx + ty) / 4.0, "height_m": tz}
            elif shape == "sphere":
                dims_raw = {"radius_m": (tx + ty + tz) / 6.0}
            elif shape == "ellipsoid":
                dims_raw = {"rx_m": tx / 2, "ry_m": ty / 2, "rz_m": tz / 2}
            else:                                  # outline/blank -> slab placeholder
                dims_raw = {"x_m": tx, "y_m": ty, "z_m": max(tz, 0.004)}
                shape = "box"
            dims = {k: Fact(round(v, 5), "estimated", "", 0.35)
                    for k, v in dims_raw.items()}
            count = max(1, int(pp.get("count", 1)))
            zc = pp.get("z_frac", 0.0) * H
            r = max(0.0, pp.get("r_frac", 0.0)) * (W + D) / 4.0
            for k in range(count):
                ang = 2.0 * math.pi * (k + 0.5) / count
                cx, cy = (r * math.cos(ang), r * math.sin(ang)) if (
                    count > 1 or pp.get("r_frac", 0.0) >= 0.25) else (0.0, 0.0)
                nm = pp["name"] if count == 1 else f"{pp['name']}_{k+1}"
                parts.append(Part(nm, shape, dict(dims) if k else dims, material,
                                  dens, (round(cx, 5), round(cy, 5), round(zc, 5))))
                zs.append(zc - tz / 2)
        if parts:
            floor = min(zs)                        # sit the construct on z≈0
            for p in parts:
                p.center_m = (p.center_m[0], p.center_m[1],
                              round(p.center_m[2] - floor, 5))
            return ConstructSpec(
                name=name, kind="composite", identified=False, parts=parts,
                sources=[{"name": source or "part decomposition", "license": lic},
                         {"name": "scaffold: proportions & placement from PartNet "
                                  "census medians — still a flagged placeholder",
                          "license": ""}],
                notes="Unidentified shape: scaffolded from the measured part "
                      "census (median proportions/placement; this object's own "
                      "specifics unverified).",
            )

    _DEF = {
        "sphere":    ({"radius_m": 0.03}, 0.03),
        "cylinder":  ({"radius_m": 0.02, "height_m": 0.08}, 0.04),
        "cone":      ({"radius_m": 0.02, "height_m": 0.06}, 0.03),
        "box":       ({"x_m": 0.06, "y_m": 0.04, "z_m": 0.03}, 0.015),
        "ellipsoid": ({"rx_m": 0.03, "ry_m": 0.02, "rz_m": 0.02}, 0.02),
        "torus":     ({"major_radius_m": 0.02, "minor_radius_m": 0.006}, 0.006),
    }
    parts, z = [], 0.0
    for pp in part_list:
        shape = pp.get("shape") or "box"
        if shape not in _DEF:                 # outline / blank -> a flat box placeholder
            shape = "box"
        dims_raw, hh = _DEF[shape]
        dims = {k: Fact(v, "estimated", "", 0.2) for k, v in dims_raw.items()}
        z += hh
        parts.append(Part(pp["name"], shape, dims, material, dens, (0.0, 0.0, z)))
        z += hh + 0.02                        # gap -> a disjoint stack
    if not parts:
        return None
    return ConstructSpec(
        name=name, kind="composite", identified=False, parts=parts,
        sources=[{"name": source or "part decomposition", "license": lic},
                 {"name": "scaffold: known parts, estimated proportions & layout", "license": ""}],
        notes="Unidentified shape: scaffolded from the known part decomposition "
              "(proportions and layout are placeholders).",
    )


# ── legged-furniture archetype ──────────────────────────────────────────────
# A chair/table/stool is a STRUCTURED assembly: a horizontal surface, legs that
# rise from a common floor to that surface's corners, and (for seating) a back
# standing vertically at the rear edge. Marginal part statistics ("the seat sits
# at z≈0.5 on average") can't express those JOINTS — independently placed parts
# float apart. So furniture gets a parametric assembler, the structural analogue
# of the layered_vessel cup kit: the GRAMMAR is archetypal, every DIMENSION is
# grounded (overall size from ShapeNetSem medians, leg count from the PartNet
# census). The LLM still identifies the object; Deckard knows how it's built.
#
#   category-or-alias -> (surface-height fraction of total height, has_back, default legs)
_FURNITURE = {
    "armchair": (0.45, True, 4), "rocking chair": (0.45, True, 4),
    "office chair": (0.50, True, 4), "chair": (0.50, True, 4),
    "bar stool": (0.92, False, 4), "stool": (0.92, False, 4),
    "bench": (0.55, True, 4), "loveseat": (0.42, True, 4),
    "sofa": (0.40, True, 4), "couch": (0.40, True, 4), "settee": (0.42, True, 4),
    "coffee table": (1.0, False, 4), "dining table": (1.0, False, 4),
    "picnic table": (1.0, False, 4), "desk": (1.0, False, 4), "table": (1.0, False, 4),
}


def _furniture_archetype(name: str):
    """(surface_frac, has_back, n_legs, matched-category) for a legged-furniture
    name, by longest whole-word match — or None."""
    qw = {w for w in __import__("re").split(r"[^a-z0-9]+", name.lower()) if w}
    best, best_len, cat = None, 0, None
    for key, arch in _FURNITURE.items():
        kw = set(key.split())
        if kw <= qw and len(kw) > best_len:
            best, best_len, cat = arch, len(kw), key
    return (*best, cat) if best else None


# ── Per-part material: a documented ROLE-CONVENTION ──────────────────────────
# PartNet carries NO per-part material, so we do NOT claim to measure it.
# Instead each part's material is assigned by ROLE — cited to ShapeNetSem's
# category material ratios. Structural parts (frame/legs/back/support) take the
# frame material; soft parts (seat/cushion/pad) take the upholstery material
# when the category composition includes one. This is a convention, flagged as
# such; the SHAPE remains 100% the real exemplar's layout.
_SOFT_ROLE = ("seat", "cushion", "pad", "mattress", "pillow", "headrest", "armrest", "sole")
_STRUCT_ROLE = ("support", "frame", "bar", "leg", "stretcher", "runner", "post", "base", "rail")
_SOFT_MATERIALS = ("fabric", "leather", "carpet", "foam", "cloth", "textile")

# A material adjective ("a wooden chair") biases the FRAME material — it selects,
# it does not invent. Maps the adjective to a canonical material the resolver
# knows; resolve.material_profile() turns that into a cited density.
_HINT_MATERIAL = {
    "wood": "wood", "wooden": "wood", "oak": "wood_oak", "pine": "wood_pine",
    "balsa": "wood_balsa", "maple": "wood_maple", "bamboo": "wood_bamboo",
    "metal": "metal", "metallic": "metal", "steel": "steel", "stainless": "stainless_steel",
    "iron": "iron", "aluminum": "aluminum", "aluminium": "aluminum",
    "brass": "brass", "bronze": "bronze", "copper": "copper", "gold": "gold",
    "plastic": "plastic", "acrylic": "plastic_acrylic", "abs": "plastic_abs",
    "glass": "glass", "leather": "leather",
}


def _hint_to_material(material_hint):
    """Adjective/hint → a canonical material name the resolver knows, or None."""
    if not material_hint:
        return None
    h = material_hint.strip().lower()
    if h in _HINT_MATERIAL:
        return _HINT_MATERIAL[h]
    for w in reversed(h.split()):                 # "a wooden chair" → "wooden"
        if w in _HINT_MATERIAL:
            return _HINT_MATERIAL[w]
    return None


def _role_material(part_name, structural, soft):
    """Which material a part takes by ROLE (structural vs upholstery)."""
    nm = (part_name or "").lower()
    if soft and any(k in nm for k in _SOFT_ROLE) and not any(k in nm for k in _STRUCT_ROLE):
        return soft
    return structural


def _density_fact(material_name):
    """Cited density Fact for a material name via the combined resolver
    (field.interface.resolve.material_profile — derived/cited where possible,
    a flagged estimate otherwise). Tier-OK: deckard(2) → field(1)."""
    try:
        from sigma_ground.field.interface.resolve import material_profile
        d = material_profile(material_name).get("density")
        if d is not None:
            return Fact(round(float(d.value), 1), d.source or "estimated", d.license, d.confidence)
    except Exception:
        pass
    return Fact(700.0, "estimated", "", 0.3)


# ── Containers: carve an open-top interior so a vessel can actually hold a fill.
# The exemplar body is a solid block; a real container is hollow. We make the
# largest body part a round wall and carve an interior cavity (named 'interior')
# that a scenario's fill floods bottom-up (construct.py op="subtract" + fill).
# A documented convention (carved interior at a flagged wall fraction), NOT
# measured geometry — the part LAYOUT still comes from the real exemplar.
_CONTAINER_WORDS = (
    "mug", "cup", "glass", "bowl", "bottle", "vase", "jar", "bucket",
    "pitcher", "tumbler", "flask", "jug", "tankard", "goblet", "beaker",
)


def _is_container(name):
    """True if the head noun (last word) names a vessel — so 'a wine glass' is a
    container but 'a glass table' is not."""
    words = name.lower().replace("-", " ").replace("_", " ").split()
    return bool(words) and words[-1] in _CONTAINER_WORDS


def _carve_interior(parts):
    """Hollow the container: round the largest body part into a wall and add an
    open-top interior cavity (op='subtract'). Returns the cavity name, or None."""
    from .schema import Part

    def vol(p):
        d = p.dims
        if p.shape == "cylinder":
            return d["radius_m"].value ** 2 * d["height_m"].value
        if p.shape == "box":
            return d["x_m"].value * d["y_m"].value * d["z_m"].value
        return 0.0

    bodies = [p for p in parts if p.shape in ("box", "cylinder")]
    if not bodies:
        return None
    body = max(bodies, key=vol)
    d = body.dims
    if body.shape == "box":                          # containers are round
        r_out = min(d["x_m"].value, d["y_m"].value) / 2.0
        h = d["z_m"].value
        body.shape = "cylinder"
        body.dims = {"radius_m": Fact(round(r_out, 5), "estimated", "container convention", 0.4),
                     "height_m": d["z_m"]}
    else:
        r_out, h = d["radius_m"].value, d["height_m"].value
    wall = max(0.002, 0.08 * r_out)                  # ~8% wall, ≥2 mm
    base = max(0.002, 0.10 * h)                       # ~10% floor
    r_in = max(r_out - wall, 0.001)
    cx, cy, cz = body.center_m
    interior = Part(
        "interior", "cylinder",
        {"radius_m": Fact(round(r_in, 5), "estimated", "carved cavity", 0.4),
         "height_m": Fact(round(h, 5), "estimated", "carved cavity", 0.4)},
        "air", Fact(1.2, "air (cavity)", "", 0.5),
        (cx, cy, round(cz + base, 5)),               # raised → floor below, open at top
    )
    interior.op = "subtract"
    parts.append(interior)
    return "interior"


def _exemplar_spec(name: str, material_hint: str | None = None,
                   exclude=None) -> ConstructSpec | None:
    """Assemble a known PartNet category from a REAL representative model's part
    layout (legs/back/seat at their actual measured positions), scaled to the
    object's median real-world size, with per-part material by role-convention
    (cited to ShapeNetSem ratios). The shape is learned from real objects, not
    hand-authored — Deckard inherently knows what a chair (mug, lamp, scissors…)
    is. ``material_hint`` (e.g. "wooden") biases the frame material AND prefers a
    tagged shape variant; ``exclude`` (anno_ids) skips already-used models so "a
    different chair" grabs a fresh one. None if the name isn't a PartNet
    category."""
    from . import sources
    from .schema import Part

    got = sources.exemplar.exemplar_of(name, exclude=exclude, hint=material_hint)
    if got is None:
        return None
    layout, src, lic, anno = got
    reused = bool(exclude) and anno in {str(x) for x in exclude}

    # overall real-world extents (W x D x H) — ShapeNetSem median, else a guess
    dim = sources.shapenetsem.dims_of(name)
    if dim:
        H, W, D = sorted(dim[0], reverse=True)       # tallest axis is height
        size_src = {"name": f"{dim[2]} — overall size (median of {dim[1]} models)",
                    "license": dim[3]}
    else:
        sz = sources.typical_size_of(name)
        H = sz[0] if sz else 0.3
        W = D = 0.62 * H
        size_src = {"name": "typical overall size (scale estimate)", "license": ""}

    # per-part material by ROLE-CONVENTION (PartNet has no per-part material).
    # Frame material = the densest cited SOLID in the ShapeNetSem category
    # composition, OR the material_hint override; soft parts take the upholstery
    # material when the composition has one. Densities are resolved AND cited
    # through material_profile (the combined resolver).
    comp = sources.shapenetsem.materials_of(name)
    ratios = comp[0] if comp else {}
    structural = _hint_to_material(material_hint)
    hint_used = structural is not None
    if structural is None:
        # the DOMINANT non-soft material is the frame (highest ratio), not the
        # first one the dict happens to list — a 40%-wood / 4%-plastic chair is
        # wood-framed, not plastic.
        non_soft = {c: r for c, r in ratios.items() if c not in _SOFT_MATERIALS}
        structural = (max(non_soft, key=non_soft.get) if non_soft else None) or "wood"
    soft_r = {c: r for c, r in ratios.items() if c in _SOFT_MATERIALS}
    soft = max(soft_r, key=soft_r.get) if soft_r else None
    struct_dens = _density_fact(structural)
    soft_dens = _density_fact(soft) if soft else None

    WDH = (W, D, H)
    est = lambda v: Fact(round(max(v, 0.003), 5), "estimated", "PartNet exemplar", 0.5)
    parts, zs = [], []
    for i, pp in enumerate(layout):
        cf, sf = pp["center_frac"], pp["size_frac"]
        ext = [max(sf[k] * WDH[k], 0.004) for k in range(3)]
        ctr = [cf[k] * WDH[k] for k in range(3)]
        shape = pp.get("shape", "box")
        if shape == "cylinder":
            dims = {"radius_m": est((ext[0] + ext[1]) / 4.0), "height_m": est(ext[2])}
        elif shape == "sphere":
            dims = {"radius_m": est((ext[0] + ext[1] + ext[2]) / 6.0)}
        else:
            dims, shape = {"x_m": est(ext[0]), "y_m": est(ext[1]), "z_m": est(ext[2])}, "box"
        pmat = _role_material(pp.get("name"), structural, soft)
        pdens = soft_dens if (soft is not None and pmat == soft) else struct_dens
        parts.append(Part(f"{pp['name'] or 'part'}_{i}", shape, dims, pmat, pdens,
                          tuple(round(x, 5) for x in ctr)))
        zs.append(ctr[2] - ext[2] / 2.0)
    if not parts:
        return None
    floor = min(zs)                                  # sit on the floor (z≈0)
    for p in parts:
        p.center_m = (p.center_m[0], p.center_m[1], round(p.center_m[2] - floor, 5))

    cavity = _carve_interior(parts) if _is_container(name) else None

    comp_str = (", ".join(f"{m} {r:.0%}" for m, r in list(ratios.items())[:4])
                if ratios else structural)
    mat_note = (f"per-part material by role-convention (frame={structural}"
                + (f", upholstery={soft}" if soft else "") + ")"
                + (f"; material_hint '{material_hint}' applied" if hint_used else "")
                + ("; hollow vessel (carved open-top interior, fillable)" if cavity else ""))
    return ConstructSpec(
        name=name, kind="composite", identified=True, parts=parts,
        sources=[{"name": "assembled from a representative real model — its actual "
                          "part layout (no hand-authored template)", "license": lic},
                 {"name": (f"representative real model — PartNet anno_id {anno}"
                           + (" (no distinct variant available yet — distill more "
                              "exemplars for this category to vary the shape)"
                              if reused else "")),
                  "license": lic, "anno_id": anno},
                 {"name": src, "license": lic}, size_src,
                 {"name": f"material composition (ShapeNetSem): {comp_str}",
                  "license": comp[2] if comp else ""},
                 {"name": mat_note + " — PartNet carries no per-part material; "
                          "materials cited to ShapeNetSem category ratios", "license": ""}],
        notes=f"A representative {name}: {len(parts)} parts at their measured "
              f"relative positions, scaled to the median real size; {mat_note}.",
    )


def _furniture_spec(name: str) -> ConstructSpec | None:
    """Build a structurally-correct legged-furniture construct, or None if the
    name isn't furniture. Grammar archetypal; size from ShapeNetSem, legs from
    PartNet — so the geometry assembles like a real chair, not floating parts."""
    import math
    from . import sources
    from .schema import Part

    arch = _furniture_archetype(name)
    if arch is None:
        return None
    surface_frac, has_back, n_legs, cat = arch

    # overall extents (W x D x H), real-world: Sem medians else a sane default
    dim = sources.shapenetsem.dims_of(name)
    if dim:
        a, b, c = sorted(dim[0], reverse=True)
        H, W, D = a, b, c                        # tallest axis is height
        size_src = [{"name": f"{dim[2]} — overall size (median of {dim[1]} models)",
                     "license": dim[3]}]
    else:
        H, W, D = {"table": (0.75, 1.2, 0.75), "desk": (0.75, 1.2, 0.7)}.get(
            cat, (0.95, 0.5, 0.5))
        size_src = [{"name": "typical furniture proportions (archetype default)",
                     "license": ""}]
    # the census knows how many legs this category really has
    comp = sources.composition_of(name)
    if comp:
        leg = next((p for p in comp[0] if p["name"] in ("leg", "foot", "post")), None)
        if leg and 3 <= int(leg.get("count", 0)) <= 6:
            n_legs = int(leg["count"])

    wood = "oak"
    dens = sources.density_of(wood, allow_web=False) or Fact(700.0, "estimated", "", 0.3)
    est = lambda v: Fact(round(v, 5), "estimated", "structural archetype", 0.5)

    slab_t = min(0.06, max(0.03, 0.06 * H))
    leg_w = min(0.06, max(0.025, 0.09 * min(W, D)))
    back_t = slab_t
    surf_z = surface_frac * H                    # top of the seat/tabletop
    seat_bottom = max(surf_z - slab_t, 0.02)

    parts = [Part("seat" if has_back else "top", "box",
                  {"x_m": est(W), "y_m": est(D), "z_m": est(slab_t)},
                  wood, dens, (0.0, 0.0, round(surf_z - slab_t / 2, 5)))]

    inset = leg_w / 2 + 0.012
    corners = [(sx * (W / 2 - inset), sy * (D / 2 - inset))
               for sx in (-1, 1) for sy in (-1, 1)]
    if n_legs == 4:
        spots = corners
    else:                                        # 3 / 5 / 6 legs -> even ring
        rr = min(W, D) / 2 - inset
        spots = [(rr * math.cos(2 * math.pi * k / n_legs),
                  rr * math.sin(2 * math.pi * k / n_legs)) for k in range(n_legs)]
    for i, (cx, cy) in enumerate(spots):
        parts.append(Part(f"leg_{i+1}", "box",
                          {"x_m": est(leg_w), "y_m": est(leg_w), "z_m": est(seat_bottom)},
                          wood, dens, (round(cx, 5), round(cy, 5),
                                       round(seat_bottom / 2, 5))))

    if has_back:                                 # vertical panel at the rear (−y) edge
        back_h = max(H - surf_z, 0.05)
        parts.append(Part("back", "box",
                          {"x_m": est(0.9 * W), "y_m": est(back_t), "z_m": est(back_h)},
                          wood, dens, (0.0, round(-(D / 2 - back_t / 2), 5),
                                       round(surf_z + back_h / 2, 5))))

    return ConstructSpec(
        name=name, kind="composite", identified=True, parts=parts,
        sources=[{"name": "structural archetype — legged furniture "
                          "(seat/top + corner legs + vertical back)", "license": ""},
                 *size_src,
                 *([{"name": (comp[1] + " — leg count"), "license": comp[2]}]
                   if comp else [])],
        notes=f"Assembled as legged furniture ({cat}): a horizontal "
              f"{'seat' if has_back else 'top'}, {n_legs} legs to the floor"
              + (", and a vertical back." if has_back else "."),
    )


# ── generative tree archetype ───────────────────────────────────────────────
# A tree has no attach-anchor "outline" mate to lean on (construct.py's
# _local_anchor has no case for it) -- so, like furniture, it's a flat
# world-space parts list: one Cylinder trunk (radius = DBH/2, cited height)
# plus one Outline canopy (revolve profile, cited crown radius). DBH is the
# ONE free parameter; everything else derives from it via cited equations
# (sources.allometry) or a documented shape convention (flagged estimate).

_CROWN_BASE_FRAC = 0.55   # trunk clear-bole fraction of total height
_CROWN_BASE_FRAC_NOTE = (
    "crown-base height = 55% of total tree height (clear-bole fraction) -- "
    "an ESTIMATED typical stand-grown loblolly pine live-crown ratio "
    "(~0.35-0.55 per general USFS silvics descriptions); NEITHER verified "
    "source paper (Bechtold 2003 crown-diameter, Bragg 2008 height) "
    "publishes a crown-base/live-crown-ratio model -- flagged, not cited."
)

_CANOPY_FOLIAGE_DENSITY = Fact(
    0.12,
    "estimated -- stand-in for canopy bulk density (CBD), the fire-science "
    "term for foliage+twig mass per unit CROWN VOLUME (mostly air): real "
    "literature reports CBD ~0.05-0.30 kg/m3 for planted pine stands (e.g. "
    "the Scott & Reinhardt 2001 standard fuel models); the loblolly-specific "
    "figure is NOT fetched/verified this session -- do not treat as cited.",
    "", 0.2,
)

# alias -> canonical species; matched by whole-word containment against name
_TREE_SPECIES_ALIASES = {
    "loblolly pine": "loblolly pine",
    "loblolly": "loblolly pine",
    "pinus taeda": "loblolly pine",
    "loblolly pine tree": "loblolly pine",
}
# bare trigger words -> the ONE curated species, as a flagged (identified=
# False) stand-in. NOTE (scope decision, flag for the Captain if it should
# change): this makes "pine"/"tree"/"conifer" alone ALWAYS build a full
# timber-scale generative tree, pre-empting the old LLM/ShapeNetSem path --
# including ShapeNetSem's own "pine"/"pine tree" entries in
# shapenetsem_sizes.json, which are small POTTED/decorative plants (median
# ~1.5x2.8x1.5m, n=14), not timber trees. If "a potted pine" should still
# resolve to that small decorative object, narrow this set (e.g. drop
# "pine"/"tree"/"conifer", require "loblolly"/"pinus taeda" explicitly) --
# one-line change, no other code affected.
_TREE_TRIGGER_WORDS = {"tree", "conifer", "evergreen", "pine"}


def _tree_archetype(name: str):
    """(species, identified) for a tree-shaped name, or None if the name
    isn't tree-shaped at all. A curated species-alias match is
    identified=True; a bare tree/conifer/pine trigger word with no specific
    species match falls back to the ONE curated species (loblolly pine) as
    the nearest available real allometry, flagged identified=False (a
    stand-in, not a claim that THIS tree IS a loblolly pine) -- the
    Deckard-Cain discipline: a partial ID is allowed, a confident fake is
    not."""
    qw = {w for w in __import__("re").split(r"[^a-z0-9]+", name.lower()) if w}
    best, best_len = None, 0
    for alias, species in _TREE_SPECIES_ALIASES.items():
        kw = set(alias.split())
        if kw <= qw and len(kw) > best_len:
            best, best_len = species, len(kw)
    if best:
        return best, True
    if qw & _TREE_TRIGGER_WORDS:
        return "loblolly pine", False
    return None


def _tree_spec(name: str, dbh_cm: float | None = None) -> ConstructSpec | None:
    """Build a generative tree construct: an allometrically-sized trunk
    cylinder + a canopy revolve-profile, from ONE free parameter (DBH). See
    sources.allometry for the cited equations (loblolly pine, the only
    curated species). v1: no branch topology -- a single trunk + one canopy
    volume; see the module notes on the returned spec for the v2 roadmap."""
    from . import sources
    from .schema import Part
    from ..dynamics.mechanisms.choice import Choice

    arch = _tree_archetype(name)
    if arch is None:
        return None
    species, identified = arch

    sources_list = []
    dbh_was_default = dbh_cm is None
    if dbh_was_default:
        default_cm = sources.allometry.species_default_dbh_cm(species)
        if default_cm is None:
            return None
        dbh_choice = Choice(
            f"trunk diameter-at-breast-height (DBH, cm) for {species} -- no "
            "caller-supplied size",
            default_cm,
            checked="sigma_ground/deckard/sources/allometry.py species table "
                    "(fitted DBH ranges: 0.3-108.2cm height / 12.7-87.9cm crown)",
            reason="midpoint chosen to sit inside BOTH papers' fitted ranges "
                  "(interpolation, not extrapolation) -- NOT a measurement "
                  "of any specific tree",
            alternatives=["pass dbh_cm explicitly to size a specific tree"],
        )
        dbh_cm = default_cm
        sources_list.append({"name": dbh_choice.cite(), "license": ""})
    dbh_cm = max(0.5, float(dbh_cm))

    height = sources.allometry.height_of(species, dbh_cm)
    crown_r = sources.allometry.crown_radius_of(species, dbh_cm)
    if height is None or crown_r is None:
        return None
    sources_list.append({"name": height.source, "license": height.license})
    sources_list.append({"name": crown_r.source, "license": crown_r.license})

    H, R = height.value, crown_r.value
    dbh_m = dbh_cm / 100.0
    trunk_r_m = dbh_m / 2.0

    wood = _density_fact("wood_pine")             # cited: 420 kg/m3, WoodHandbook
    sources_list.append({"name": wood.source, "license": wood.license})

    trunk_r_source = ("estimated (defaulted DBH -- see Choice note above)"
                      if dbh_was_default else "caller-specified DBH")
    trunk_r_conf = 0.35 if dbh_was_default else 0.9
    trunk = Part(
        "trunk", "cylinder",
        {"radius_m": Fact(round(trunk_r_m, 5), trunk_r_source, "", trunk_r_conf),
         "height_m": Fact(round(H, 3), height.source, height.license, height.confidence)},
        "wood_pine", wood, (0.0, 0.0, round(H / 2.0, 5)),
    )

    crown_base_h = _CROWN_BASE_FRAC * H
    crown_depth = max(H - crown_base_h, 0.5)
    crown_mid_z = crown_base_h + crown_depth / 2.0
    d = crown_depth
    # half-profile (z, r), z ascending, LOCAL to the canopy part's center:
    # a flat-bottom disk (0.35R) where it meets the bole, widest (R) low-
    # third, tapering to a point (r=0) at the treetop -- a documented SHAPE
    # CONVENTION (crown WIDTH is cited; crown vertical PROFILE is not).
    profile = [
        (-d / 2.0,               0.35 * R),
        (-d / 2.0 + 0.30 * d,    R),
        ( d / 2.0 - 0.30 * d,    0.5 * R),
        ( d / 2.0,               0.0),
    ]
    canopy = Part(
        "canopy", "outline", {}, "pine_foliage", _CANOPY_FOLIAGE_DENSITY,
        (0.0, 0.0, round(crown_mid_z, 5)), (0.0, 0.0, 0.0), "add", None,
        outline={"profile": [[round(z, 5), round(r, 5)] for z, r in profile],
                "mode": "revolve"},
    )
    sources_list.append({"name": _CANOPY_FOLIAGE_DENSITY.source, "license": ""})
    sources_list.append({"name": _CROWN_BASE_FRAC_NOTE, "license": ""})
    sources_list.append({
        "name": "canopy vertical silhouette (widest-point/base/apex taper "
                "fractions: 0.35/0.30/0.30 of crown radius/depth) is a "
                "documented shape CONVENTION, not measured -- crown WIDTH "
                "is cited (Bechtold), crown vertical PROFILE is not.",
        "license": "",
    })

    id_note = (f"identified as {species.title()} (Pinus taeda)" if identified
              else f"species not identified; using {species}'s curated "
                   "allometry as the nearest available real stand-in")
    return ConstructSpec(
        name=name, kind="composite", identified=identified,
        parts=[trunk, canopy], sources=sources_list,
        notes=(f"Generative tree ({id_note}): DBH={dbh_cm:.1f} cm -> "
              f"height={H:.2f} m (Chapman-Richards, Bragg 2008), crown "
              f"radius={R:.2f} m (linear model, Bechtold 2003). v1: a "
              f"single untapered trunk cylinder (radius=DBH/2, full height "
              f"-- a geometric UPPER BOUND on real tapered stem volume) "
              f"plus one canopy revolve-volume (foliage bulk density, "
              f"estimated, not measured); no branch topology. See "
              f"sources.allometry.validate_tree_mass for a self-consistency "
              f"check (NOT an independent biomass citation -- see its own "
              f"note). v2 roadmap: a tapered trunk (swap Cylinder for a "
              f"Cone/frustum once a taper coefficient is cited), a "
              f"scaffold-branch graph (3-6 primary branches placed by a "
              f"pipe-model-theory or per-species branch allometry -- a "
              f"THIRD citable source, not yet researched), and a cited "
              f"canopy bulk-density figure in place of the current "
              f"estimate."),
    )


# ── decorated doughnut archetype ────────────────────────────────────────────
# The LLM researcher (researcher.py) has no vocabulary for the SCATTER
# convention (schema.Part.scatter / construct._expand_scatter, added this
# session) -- it can only emit sphere/cylinder/box/cone/torus/ellipsoid/
# outline parts, so even a fresh "doughnut with sprinkles" research call
# would come back a plain torus. This is a hand-curated archetype (same
# tier as _tree_archetype/_furniture_archetype: a documented, deterministic
# recipe, not an LLM guess) so "doughnut" reliably gets its decoration.
# v2 roadmap: teach researcher.py's system prompt the scatter/conform-shell
# grammar generally, so ANY object with a coating + scattered decoration
# (not just doughnuts) can be LLM-composed -- not built this session.
_DOUGHNUT_TRIGGER_WORDS = {"doughnut", "donut"}

_SPRINKLE_SIZE_NOTE = (
    "estimated -- no citable confectionery-sprinkle dimension exists in "
    "sources/ this session; ~3mm length x ~1mm diameter is a commonly-cited "
    "'jimmies' size (candy-supplier descriptions), NOT independently "
    "verified against a primary source here -- treat as an approximation."
)
_ICING_DENSITY_NOTE = (
    "approximated as cited crystalline sucrose density (inventory/data/"
    "materials.json 'Sucrose', 1550 kg/m3) -- icing/glaze is mostly sucrose "
    "in a thin water paste; real icing bulk density is somewhat lower "
    "(entrained water/air), not independently measured this session."
)
_RAINBOW_SPRINKLE_MATERIALS = ["sprinkle_red", "sprinkle_orange", "sprinkle_yellow",
                              "sprinkle_green", "sprinkle_blue", "sprinkle_violet"]


def _doughnut_archetype(name: str) -> bool:
    qw = {w for w in __import__("re").split(r"[^a-z0-9]+", name.lower()) if w}
    return bool(qw & _DOUGHNUT_TRIGGER_WORDS)


def _decorated_doughnut_spec(name: str, *, sprinkles: bool = True,
                             sprinkle_count: int = 24) -> ConstructSpec | None:
    """A torus dough body + a conforming frosting shell + a scatter of
    rainbow sprinkles. Dimensions are ESTIMATED (no cited doughnut-geometry
    source) -- flagged throughout, not claimed as measured. Simplification,
    flagged: the frosting coats the WHOLE torus (not just the top, as a real
    dipped doughnut would) -- a directional partial coat needs a
    _Clipped-style half-space cut, a v2 extension not built this session."""
    from . import sources
    from .schema import Part, ESTIMATED

    if not _doughnut_archetype(name):
        return None

    major_r, minor_r = 0.03763, 0.007526   # ESTIMATED typical doughnut proportions
    dough = Part(
        "dough", "torus",
        {"major_radius_m": Fact(major_r, ESTIMATED, "", 0.55),
         "minor_radius_m": Fact(minor_r, ESTIMATED, "", 0.55)},
        "leavened dough", Fact(1000.0, ESTIMATED, "", 0.2),
    )
    parts = [dough]
    sources_list = [
        {"name": _SPRINKLE_SIZE_NOTE, "license": ""},
        {"name": _ICING_DENSITY_NOTE, "license": ""},
    ]

    coat_m = 0.0015   # 1.5mm icing coat -- ESTIMATE, no cited source
    icing_dens = sources.density_of("sucrose") or Fact(1550.0, ESTIMATED, _ICING_DENSITY_NOTE, 0.3)
    frosting = Part(
        "frosting", "torus",
        {"major_radius_m": Fact(major_r, ESTIMATED, "", 0.55),
         "minor_radius_m": Fact(minor_r + coat_m, ESTIMATED, "1.5mm coat, ESTIMATE", 0.3)},
        "icing", icing_dens, conform="dough",
    )
    parts.append(frosting)

    if sprinkles:
        sprinkle = Part(
            "sprinkle", "ellipsoid",
            {"rx_m": Fact(0.0005, ESTIMATED, _SPRINKLE_SIZE_NOTE, 0.2),
             "ry_m": Fact(0.0005, ESTIMATED, _SPRINKLE_SIZE_NOTE, 0.2),
             "rz_m": Fact(0.0015, ESTIMATED, _SPRINKLE_SIZE_NOTE, 0.2)},
            _RAINBOW_SPRINKLE_MATERIALS[0], sources.density_of("sucrose"),
            scatter=dict(on="frosting", count=sprinkle_count,
                        materials=_RAINBOW_SPRINKLE_MATERIALS,
                        seed=7, region="top", embed_frac=0.4, size_jitter=0.15),
        )
        parts.append(sprinkle)

    return ConstructSpec(
        name=name, kind="composite", identified=True, parts=parts,
        sources=sources_list,
        notes=("Frosting is a full uniform coat (all the way around the "
              "torus, not just the top) -- a documented simplification. "
              "Sprinkles are a SCATTER of one template ellipsoid across "
              "the frosting's top-facing region, materials cycling a "
              "6-colour rainbow palette. See schema.Part.scatter's "
              "convention doc for the placement mechanism."),
    )


def research(name: str, *, allow_llm: bool = True,
             material_hint: str | None = None, exclude=None,
             dbh_cm: float | None = None) -> ConstructSpec:
    """Resolve a named object to a cited ConstructSpec; flag if unidentified.

    A catalog hit is returned verbatim (deterministic). A PartNet category gets
    the real-exemplar assembler with per-part materials (``material_hint`` such
    as "wooden" biases the frame material — it SELECTS, it does not invent).
    ``exclude`` (anno_ids) skips already-used models so a "different" request
    grabs a fresh real model from the category's variant pool. A tree name gets
    the generative allometry-driven trunk+canopy assembler (``dbh_cm`` sizes it;
    unset defaults via a cited ``Choice``, see ``_tree_spec``). On a miss the
    Researcher is consulted if available; failing that, a flagged best-guess is
    returned — never a confident fake.
    """
    hit = catalog.lookup(name)
    if hit is not None:
        return hit

    structured = (_exemplar_spec(name, material_hint=material_hint, exclude=exclude)
                  or _furniture_spec(name)          # real layout, then archetypes
                  or _tree_spec(name, dbh_cm=dbh_cm)
                  or _decorated_doughnut_spec(name))
    if structured is not None:
        return structured

    if allow_llm:
        try:
            from .researcher import research_spec   # lazy: optional deps / network
            spec = research_spec(name)
            if spec is not None:
                try:
                    catalog.save_for(name, spec)   # freeze → next lookup is a hit
                except Exception:
                    pass
                return spec
        except Exception:
            pass   # fall through to the flagged default — never a fake

    scaffold = _scaffold_from_composition(name)   # known parts -> a flagged scaffold
    if scaffold is not None:
        return scaffold
    return _generic_vessel(name)


def voxel_plan(name: str, *, material_hint: str | None = None, exclude=None):
    """Resolve a name to the recipe for VOXELIZING its real mesh, or None if the
    name isn't a real PartNet category.

    Returns ``{"anno_id", "scale_to_m", "material", "source", "license"}`` — the
    SAME exemplar selection, ShapeNetSem median size, and dominant role-convention
    material that ``_exemplar_spec`` uses, so the voxel path and the primitive
    path resolve identically: "a wooden chair" picks a real chair model + wood,
    and ``exclude`` makes "a different chair" grab a fresh anno_id. The structural
    (dominant) material is used for the whole solid in this stage; the per-part
    label grid is the eventual per-cell refinement.
    """
    from . import sources

    got = sources.exemplar.exemplar_of(name, exclude=exclude, hint=material_hint)
    if got is None:
        return None
    _layout, src, lic, anno = got
    dim = sources.shapenetsem.dims_of(name)
    if dim:
        longest = max(dim[0])                            # tallest/longest real edge
    else:
        sz = sources.typical_size_of(name)
        longest = sz[0] if sz else 0.3
    comp = sources.shapenetsem.materials_of(name)
    ratios = comp[0] if comp else {}
    structural = _hint_to_material(material_hint)
    if structural is None:
        non_soft = {c: r for c, r in ratios.items() if c not in _SOFT_MATERIALS}
        structural = (max(non_soft, key=non_soft.get) if non_soft else None) or "wood"
    return {"anno_id": str(anno), "scale_to_m": float(longest),
            "material": structural, "source": src, "license": lic}
