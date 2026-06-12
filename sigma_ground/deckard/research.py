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


def research(name: str, *, allow_llm: bool = True) -> ConstructSpec:
    """Resolve a named object to a cited ConstructSpec; flag if unidentified.

    A catalog hit is returned verbatim (deterministic). On a miss the Researcher
    is consulted if available; failing that, a flagged best-guess is returned —
    never a confident fake.
    """
    hit = catalog.lookup(name)
    if hit is not None:
        return hit

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
