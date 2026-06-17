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


def _exemplar_spec(name: str, material_hint: str | None = None) -> ConstructSpec | None:
    """Assemble a known PartNet category from a REAL representative model's part
    layout (legs/back/seat at their actual measured positions), scaled to the
    object's median real-world size, with per-part material by role-convention
    (cited to ShapeNetSem ratios). The shape is learned from real objects, not
    hand-authored — Deckard inherently knows what a chair (mug, lamp, scissors…)
    is. ``material_hint`` (e.g. "wooden") biases the frame material. None if the
    name isn't a PartNet category."""
    from . import sources
    from .schema import Part

    got = sources.exemplar.exemplar_of(name)
    if got is None:
        return None
    layout, src, lic = got

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
        structural = next((c for c in ratios if c not in _SOFT_MATERIALS), None) or "wood"
    soft = next((c for c in ratios if c in _SOFT_MATERIALS), None)
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

    comp_str = (", ".join(f"{m} {r:.0%}" for m, r in list(ratios.items())[:4])
                if ratios else structural)
    mat_note = (f"per-part material by role-convention (frame={structural}"
                + (f", upholstery={soft}" if soft else "") + ")"
                + (f"; material_hint '{material_hint}' applied" if hint_used else ""))
    return ConstructSpec(
        name=name, kind="composite", identified=True, parts=parts,
        sources=[{"name": "assembled from a representative real model — its actual "
                          "part layout (no hand-authored template)", "license": lic},
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


def research(name: str, *, allow_llm: bool = True,
             material_hint: str | None = None) -> ConstructSpec:
    """Resolve a named object to a cited ConstructSpec; flag if unidentified.

    A catalog hit is returned verbatim (deterministic). A PartNet category gets
    the real-exemplar assembler with per-part materials (``material_hint`` such
    as "wooden" biases the frame material — it SELECTS, it does not invent). On
    a miss the Researcher is consulted if available; failing that, a flagged
    best-guess is returned — never a confident fake.
    """
    hit = catalog.lookup(name)
    if hit is not None:
        return hit

    structured = (_exemplar_spec(name, material_hint=material_hint)
                  or _furniture_spec(name))         # real layout, else archetype
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
