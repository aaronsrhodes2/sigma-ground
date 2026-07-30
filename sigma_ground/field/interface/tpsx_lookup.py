"""NASA TPSX materials lookup -- name/property matching onto sigma-ground's
own material vocabulary, with unit-normalized, provenance-tagged output.

Design precedent: explicit hand-curated alias tables, never fuzzy string
matching -- mirrors deckard/sources/aliases.py's discipline (module-level
cache, normalize-before-lookup, empty result beats a guessed one), adapted
to an inline dict (not a JSON file) because every entry here needs a
human-reviewed WHY comment that a JSON blob can't carry cleanly (see the
exclusions below -- "Silicone is not silicon" doesn't belong in data).

Input contract: `tpsx_property()`'s first argument is a resolve.py CANONICAL
token (e.g. "aluminum", "steel_mild") -- the same vocabulary
field/interface/resolve.py::material_profile() already produces via its own
_ALIAS/normalize step. This module deliberately does NOT re-implement name
normalization (no "gold" vs "Gold" vs "gold bar" handling here) -- that
would be a second, possibly-divergent answer to "what does the user mean by
X" alongside resolve.py's existing one. Callers other than resolve.py should
pass a canonical token too.

Coverage (verified against the live 700-entry file, 2026-07-29): TPSX has
NO biological/organic/civil-construction materials (bone, wood, concrete,
kevlar, rubber, plastic) and its "Silicone" entries are an organosilicon
POLYMER, not elemental silicon -- a false-friend explicitly excluded below,
not fuzzy-matched. Its mechanical (tensile/shear/Poisson) rows belong almost
entirely to aerospace TPS composites, not the plain-metal names in our
alias table -- for the materials aliased below, TPSX supplies density and
thermal properties, essentially never E/G/sigma_UTS today.
"""
from __future__ import annotations

import functools
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_JSON_PATH = (Path(__file__).resolve().parents[2] / "inventory" / "data"
             / "tpsx_materials.json")
_FOOTNOTE_DIGIT = re.compile(r"[0-9]+$")
_LICENSE = "US government work, freely retrievable (NASA TPSX, tpsx.arc.nasa.gov)"


# ── canon -> tpsx_id ─────────────────────────────────────────────────────
# Every entry hand-verified against a live density cross-check; every
# omission from sigma-ground's ~25-material vocabulary is a deliberate,
# documented exclusion below, not an oversight.
_MATERIAL_ALIAS: dict[str, int] = {
    "aluminum": 59,            # "Aluminum-Pure"
    "copper": 55,              # "Copper" (dup of id_257 -- 55 picked, has Melt Temp)
    "depleted_uranium": 639,   # "uranium" -- density 19100 exact match
    "gold": 575,               # "gold"
    "granite": 506,            # "granite (av prop)" -- not high-k/low-k variants
    "iron": 584,               # "iron (-51 to 1537 deg c)" -- widest range
    "lead": 589,               # "lead"
    "nickel": 600,             # "nickel"
    "platinum": 608,           # "platinum"
    "silver": 625,             # "silver"
    "titanium": 58,            # "Titanium" (pure) -- NOT id_261 6Al-4V alloy
    "tungsten": 62,            # "Tungsten" (dup of id_638 -- 62 has Melt Temp)
    "zinc": 644,               # "zinc"
    # Deliberately NOT aliased (no safe match -- do not "fix" with fuzzy
    # matching, each has a specific documented reason):
    #   bone, concrete, kevlar, plastic_abs, rubber, water_ice, wood_oak
    #     -- not present; TPSX is aerospace-materials only.
    #   carbon_fiber -- TPSX's carbon-carbon/carbon-phenolic entries are
    #     ablative composites, not structural CFRP.
    #   ceramic_alumina -- TPSX's alumina entries are porous refractory
    #     brick/porcelain (~3400 kg/m3), not dense technical alumina
    #     (surface.py models 3950 kg/m3) -- ~14% mismatch, different material.
    #   glass -- TPSX has flint/fused-quartz/fused-silica/lead glass, none
    #     literally soda-lime (surface.py's plain "glass").
    #   silicon -- TPSX's "Silicone A/B" entries are an organosilicon
    #     POLYMER, chemically unrelated to elemental crystalline silicon.
    #   steel_mild -- TPSX only has stainless/alloy grades (AISI 302/304,
    #     Type 321, 17-4 PH), not plain low-carbon mild steel.
}

# resolve.py key -> ordered TPSX (stripped, entity-decoded) property names,
# most-preferred orientation first, and the unit we verified it's reported
# in -- a live unit mismatch refuses the match rather than silently
# mis-scaling (see tpsx_property()).
_PROPERTY_MAP: dict[str, list[tuple[str, str]]] = {
    "density":                   [("Density", "kg/m^3")],
    "sigma_UTS":                 [("Tensile Strength (Isotropic)", "Pa"),
                                   ("Tensile Strength (In-Plane)", "Pa"),
                                   ("Tensile Strength (Thru-the-Thickness)", "Pa")],
    "E":                         [("Tensile Modulus (Isotropic)", "Pa"),
                                   ("Tensile Modulus (In-Plane)", "Pa"),
                                   ("Tensile Modulus (Thru-the-Thickness)", "Pa")],
    "G":                         [("Shear Modulus (Isotropic)", "Pa"),
                                   ("Shear Modulus (In-Plane)", "Pa"),
                                   ("Shear Modulus (Thru-the-Thickness)", "Pa")],
    "poisson_ratio":              [("Poisson's Ratio (Isotropic)", "-"),
                                   ("Poisson's Ratio (In-Plane)", "-"),
                                   ("Poisson's Ratio (Thru-the-Thickness)", "-")],
    "melt_temperature_k":        [("Melt Temperature", "K")],
    "thermal_conductivity_w_mk": [("Thermal Conductivity (Isotropic)", "W/m-K"),
                                   ("Thermal Conductivity (In-Plane)", "W/m-K"),
                                   ("Thermal Conductivity (Thru-the-Thickness)", "W/m-K")],
    "specific_heat_j_kgk":       [("Specific Heat", "J/kg-K")],
}

TPSX_RESOLVE_KEYS = tuple(_PROPERTY_MAP.keys())

# TPSX source_type -> confidence. "unknown" is the majority tag (2450 of
# 2685 rows) -- still a real per-material citation from a cited US-gov
# database, just without an explicit measured/predicted label; scored below
# the codebase's normal MEASURED tiers (0.8-0.9) but above the internal
# ESTIMATED default (0.5).
_CONFIDENCE_BY_SOURCE_TYPE = {
    "measured": 0.85, "predicted": 0.6, "interpolated": 0.55, "unknown": 0.65,
}
_DEFAULT_CONFIDENCE = 0.6


@dataclass(frozen=True)
class TPSXFact:
    value: float
    units: str
    source_type: str
    confidence: float
    tpsx_id: int
    tpsx_name: str
    raw_property: str
    license: str = _LICENSE

    @property
    def source(self) -> str:
        return (f"NASA TPSX id={self.tpsx_id} ({self.tpsx_name!r}) "
                f"{self.raw_property} [{self.source_type}]")


@functools.lru_cache(maxsize=1)
def _materials() -> dict:
    try:
        doc = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    doc.pop("_meta", None)
    return doc


@functools.lru_cache(maxsize=None)
def _properties_by_stripped_name(tpsx_id: int) -> dict:
    entry = _materials().get(f"id_{tpsx_id}")
    if entry is None:
        return {}
    out: dict[str, dict] = {}
    for row in entry.get("properties", []):
        stripped = html.unescape(_FOOTNOTE_DIGIT.sub("", row.get("property", "")))
        out.setdefault(stripped, row)          # first occurrence wins
    return out


def tpsx_property(canonical_key: str, resolve_key: str) -> Optional[TPSXFact]:
    """A single TPSX-cited property value for a resolve.py canonical
    material key, or None -- never a guessed/fuzzy-matched value. See the
    module docstring for the input contract (canonical_key, not free text).
    """
    tpsx_id = _MATERIAL_ALIAS.get(canonical_key)
    if tpsx_id is None:
        return None
    entry = _materials().get(f"id_{tpsx_id}")
    if entry is None:
        return None
    variants = _PROPERTY_MAP.get(resolve_key)
    if not variants:
        return None
    by_name = _properties_by_stripped_name(tpsx_id)
    for prop_name, expected_units in variants:
        row = by_name.get(prop_name)
        if row is None:
            continue
        if row.get("units") != expected_units:
            # unit drift from what this design verified -- refuse rather
            # than silently mis-scale a value in an unexpected unit.
            continue
        try:
            value = float(row["value"])
        except (TypeError, ValueError, KeyError):
            continue
        source_type = row.get("source_type", "unknown")
        confidence = _CONFIDENCE_BY_SOURCE_TYPE.get(source_type, _DEFAULT_CONFIDENCE)
        return TPSXFact(value=value, units=row["units"], source_type=source_type,
                        confidence=confidence, tpsx_id=tpsx_id,
                        tpsx_name=entry.get("name", ""), raw_property=row["property"])
    return None


__all__ = ["tpsx_property", "TPSXFact", "TPSX_RESOLVE_KEYS"]
