"""Free factual material data — Wikidata density (CC0), grounded + cited.

Used by :func:`sources.density_of` as the fallback when a material is not in our
local data. Wikidata's density (property ``P2054``) is a structured quantity
with an explicit unit, so the value is *attributable* — entity QID + CC0 license
— rather than an estimate. Excludes proprietary/ToS-restricted sources by design.

Two cached calls per lookup: name → entity (wbsearchentities), then the entity's
P2054 claim (wbgetclaims). Unknown units are refused (never unit-guessed).
"""
from __future__ import annotations

import urllib.parse

from ..schema import Fact
from . import web

_API = "https://www.wikidata.org/w/api.php"

# Wikidata unit QID -> multiplier to kg/m^3 (only recognised units are trusted)
_UNIT_TO_KG_M3 = {
    "Q13147228": 1000.0,   # gram per cubic centimetre
    "Q844211": 1.0,        # kilogram per cubic metre
    "Q3210011": 1.0,       # kilogram per cubic metre (alt)
}


def _search_qid(name: str) -> str | None:
    url = (f"{_API}?action=wbsearchentities&format=json&language=en&type=item"
           f"&limit=1&search={urllib.parse.quote(name)}")
    d = web.get_json(url)
    if not d:
        return None
    hits = d.get("search") or []
    return hits[0].get("id") if hits else None


def _density_kg_m3(qid: str) -> float | None:
    url = f"{_API}?action=wbgetclaims&format=json&entity={qid}&property=P2054"
    d = web.get_json(url)
    if not d:
        return None
    for claim in (d.get("claims") or {}).get("P2054", []):
        value = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
        amount = value.get("amount")
        unit = (value.get("unit") or "").rsplit("/", 1)[-1]
        if amount is None:
            continue
        try:
            val = float(amount)
        except (TypeError, ValueError):
            continue
        mult = _UNIT_TO_KG_M3.get(unit)
        if mult is None:
            continue                       # never guess an unrecognised unit
        if val > 0:
            return val * mult
    return None


def wikidata_density(material: str) -> Fact | None:
    """A cited density Fact (kg/m³) from Wikidata, or None."""
    qid = _search_qid(material)
    if not qid:
        return None
    rho = _density_kg_m3(qid)
    if rho is None or rho <= 0.0:
        return None
    return Fact(round(rho, 3), f"Wikidata {qid} (P2054)", "CC0", 0.75)


__all__ = ["wikidata_density"]
