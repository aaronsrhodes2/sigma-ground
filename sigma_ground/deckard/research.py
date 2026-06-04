"""Deckard's shape research — a name → a concrete, cited ConstructSpec.

Order of resolution:
  1. a frozen catalog hit (``catalog/<slug>.md``) — deterministic, offline;
  2. on a miss, the Researcher (``researcher.py``: Gemini-free → local qwen,
     grounded by our own data + free factual APIs) synthesises and freezes one;
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
                return spec
        except Exception:
            pass   # fall through to the flagged default — never a fake

    return _generic_vessel(name)
