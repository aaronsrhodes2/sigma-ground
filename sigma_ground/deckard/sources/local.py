"""Local data source — ground material facts in our own curated data (offline).

Queries, in order of trust:
  1. ``field.interface.surface.MATERIALS``   — 24 core engineering materials (kg/m³)
  2. ``inventory/data/materials.json``        — ~255 compounds (name, formula, density kg/m³)

Returns cited Facts so Deckard's researcher can attribute every density rather
than estimate it. Lookups are name-normalised with a small alias table for the
vocabulary the researcher tends to emit (e.g. "stoneware", "liquid water").
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

from ..schema import Fact

_MATERIALS_JSON = (pathlib.Path(__file__).resolve().parents[2]
                   / "inventory" / "data" / "materials.json")

# researcher/vocabulary name -> canonical data key
_ALIASES = {
    "liquid water": "water", "h2o": "water",
    "stoneware": "ceramic_alumina", "earthenware": "ceramic_alumina",
    "ceramic": "ceramic_alumina", "porcelain": "ceramic_alumina",
    "glaze": "glass", "glaze (glassy)": "glass",
    "steel": "steel_mild", "mild steel": "steel_mild",
    "aluminium": "aluminum",
    "oak": "wood_oak", "wood": "wood_oak",
}


@functools.lru_cache(maxsize=1)
def _materials_json() -> dict:
    """name(lower) -> (density_kg_m3, formula) from inventory materials.json."""
    try:
        data = json.loads(_MATERIALS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, tuple[float, str]] = {}
    for d in data:
        if not isinstance(d, dict):
            continue
        nm = (d.get("name") or "").strip().lower()
        rho = d.get("density")
        if nm and isinstance(rho, (int, float)) and rho > 0:
            out[nm] = (float(rho), d.get("formula", ""))
    return out


@functools.lru_cache(maxsize=1)
def _surface_materials() -> dict:
    """key/name(lower) -> density_kg_m3 from field.interface.surface.MATERIALS."""
    try:
        from sigma_ground.field.interface.surface import MATERIALS
    except Exception:
        return {}
    out: dict[str, float] = {}
    for key, m in MATERIALS.items():
        rho = m.get("density_kg_m3")
        if isinstance(rho, (int, float)) and rho > 0:
            out[key.strip().lower()] = float(rho)
            nm = (m.get("name") or "").strip().lower()
            if nm:
                out.setdefault(nm, float(rho))
    return out


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w}


def density_of(material: str) -> Fact | None:
    """A cited density Fact (kg/m³) for a material name, or None if unknown.

    Exact matches are confidence 0.9; loose (whole-word containment) matches 0.6.
    Containment is WORD-level, never a bare substring — so "ceramic mug body"
    still grounds in "ceramic", but "keratin" no longer mis-matches "tin" (and
    "hair" no longer matches "air", "marigold" no longer matches "gold", …).
    """
    key = material.strip().lower()
    key = _ALIASES.get(key, key)

    surface = _surface_materials()
    if key in surface:
        return Fact(surface[key], "field.interface.surface.MATERIALS", "", 0.9)

    mats = _materials_json()
    if key in mats:
        return Fact(mats[key][0], "inventory/data/materials.json", "", 0.9)

    # loose containment fallback ("ceramic mug body" → "ceramic"), word-level:
    # the data name's words must ALL appear in the query (not an arbitrary
    # substring, and not the reverse — so a bare "carbon" can't grab "carbon
    # dioxide"). Prefer the most specific (longest) such name.
    key_w = _words(key)
    if key_w:
        best = None
        for src, table in (("field.interface.surface.MATERIALS", surface),
                           ("inventory/data/materials.json", mats)):
            for nm, val in table.items():
                nm_w = _words(nm)
                if nm_w and nm_w <= key_w and (best is None or len(nm_w) > best[0]):
                    rho = val[0] if isinstance(val, tuple) else val
                    best = (len(nm_w), Fact(rho, src, "", 0.6))
        if best is not None:
            return best[1]
    return None


__all__ = ["density_of"]
