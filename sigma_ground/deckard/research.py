"""Deckard's shape research — a name → concrete, cited spatial dimensions.

v1 is a small grounded catalog of common items with *typical* dimensions and
material densities, each carrying its source. This is the deterministic core;
the residual path (dimension databases, 3D-model references mined for
proportions, general web/LLM) plugs in later behind the same `research()`
front door. Unknown items return a flagged best-guess rather than a fake — the
Deckard-Cain discipline: a partial ID is allowed, a confident fake is not.

A `layered_vessel` is described by physical thicknesses (geometry-driven), so
the layer mass-ratios fall out exactly — geometry and ratio are two readings of
one dial; we author the dial we can ground (thickness), and the ratio derives.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ItemSpec:
    name: str
    kind: str                      # 'layered_vessel'
    geometry: dict                 # outer_radius_m, height_m, wall_m, glaze_m, base_m, fill_fraction
    layers: list                   # [{name, material, density_kg_m3, source}], outer→inner
    source: str = ""               # provenance of the dimensions
    identified: bool = True        # False → best-guess, flagged

    def note(self) -> str:
        tag = "" if self.identified else "  [UNIDENTIFIED — best-guess proportions]"
        return f"{self.name} ({self.kind}) — {self.source}{tag}"


# ── Grounded catalog (typical dimensions; cited) ────────────────────────
def _coffee_cup() -> ItemSpec:
    return ItemSpec(
        name="coffee cup",
        kind="layered_vessel",
        geometry={
            "outer_radius_m": 0.040,   # 80 mm outer diameter
            "height_m":       0.095,   # 95 mm tall
            "wall_m":         0.005,   # 5 mm ceramic wall
            "glaze_m":        0.0003,  # 0.3 mm glaze skin
            "base_m":         0.007,   # 7 mm base thickness
            "fill_fraction":  0.80,    # filled to 80% of interior height
        },
        layers=[
            {"name": "glaze",   "material": "glaze (glassy)", "density_kg_m3": 2400.0,
             "source": "typical ceramic glaze ≈ glass, ~2400 kg/m³"},
            {"name": "ceramic", "material": "stoneware",      "density_kg_m3": 2300.0,
             "source": "stoneware/earthenware body, ~2300 kg/m³"},
            {"name": "water",   "material": "liquid water",   "density_kg_m3": 998.0,
             "source": "liquid water at 20 °C, 998 kg/m³"},
        ],
        source="typical stoneware coffee mug (approx.)",
        identified=True,
    )


CATALOG = {
    "coffee cup": _coffee_cup,
    "coffee mug": _coffee_cup,
    "mug":        _coffee_cup,
    "cup":        _coffee_cup,
    "teacup":     _coffee_cup,
}


def research(name: str) -> ItemSpec:
    """Resolve a named object to a cited ItemSpec; flag if unidentified."""
    key = name.strip().lower()
    if key in CATALOG:
        return CATALOG[key]()
    # crude containment match ("a ceramic coffee cup" → coffee cup)
    for cat_key, builder in CATALOG.items():
        if cat_key in key:
            spec = builder()
            spec.source += f" (matched '{cat_key}' in request)"
            return spec
    # Unidentified → a flagged best-guess vessel, never a silent fake.
    spec = _coffee_cup()
    spec.name = name
    spec.identified = False
    spec.source = "no catalog/DB entry — defaulted to a generic small vessel"
    return spec
