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
    the model could not shape the object. We know its parts (the composition
    prior), just not its proportions — so each part becomes a default primitive at
    a default human-scale size, stacked disjoint, identified=False. Honest: parts
    known, proportions guessed — never a confident fake."""
    from . import sources
    from .schema import Part
    got = sources.composition_of(name)
    if not got:
        return None
    part_list, source, lic = got
    _DEF = {
        "sphere":    ({"radius_m": 0.03}, 0.03),
        "cylinder":  ({"radius_m": 0.02, "height_m": 0.08}, 0.04),
        "cone":      ({"radius_m": 0.02, "height_m": 0.06}, 0.03),
        "box":       ({"x_m": 0.06, "y_m": 0.04, "z_m": 0.03}, 0.015),
        "ellipsoid": ({"rx_m": 0.03, "ry_m": 0.02, "rz_m": 0.02}, 0.02),
        "torus":     ({"major_radius_m": 0.02, "minor_radius_m": 0.006}, 0.006),
    }
    dens = sources.density_of("plastic", allow_web=False) or Fact(950.0, "estimated", "", 0.2)
    parts, z = [], 0.0
    for pp in part_list:
        shape = pp.get("shape") or "box"
        if shape not in _DEF:                 # outline / blank -> a flat box placeholder
            shape = "box"
        dims_raw, hh = _DEF[shape]
        dims = {k: Fact(v, "estimated", "", 0.2) for k, v in dims_raw.items()}
        z += hh
        parts.append(Part(pp["name"], shape, dims, "plastic", dens, (0.0, 0.0, z)))
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
