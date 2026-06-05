"""Free factual dimensions — Wikidata length properties (CC0).

The dimension mirror of ``materials_api`` (which reads density P2054): resolve an
object's QID, read its dimension claims, trust ONLY whitelisted length units
(converted to metres), and map them onto the proposed primitive shape.

Conservative on purpose. Wikidata is sparse on object dimensions, and a *wrong*
cited dimension is worse than an honest estimate (cf. the keratin->tin lesson),
so we only emit UNAMBIGUOUS mappings:

  * sphere   : diameter (P2386) -> radius_m
  * cylinder : diameter (P2386) -> radius_m, height (P2048) -> height_m

Anything else (box width/length/height -> which axis?) is left to the local
standard table; here it returns None rather than guess an axis assignment.
"""
from __future__ import annotations

import urllib.parse

from . import web
from ..schema import Fact

_API = "https://www.wikidata.org/w/api.php"

# length unit QID -> metres (anything else is refused — never guessed)
_UNIT_TO_M = {
    "Q11573": 1.0,        # metre
    "Q174728": 0.01,      # centimetre
    "Q174789": 0.001,     # millimetre
    "Q218593": 0.0254,    # inch
    "Q3710": 0.3048,      # foot
}
_P_HEIGHT, _P_DIAMETER = "P2048", "P2386"


def _search_qid(name: str) -> str | None:
    url = (f"{_API}?action=wbsearchentities&format=json&language=en&type=item"
           f"&limit=1&search={urllib.parse.quote(name)}")
    d = web.get_json(url)
    hits = (d or {}).get("search") or []
    return hits[0].get("id") if hits else None


def _length_m(claims: dict, prop: str):
    """First whitelisted length value (metres) for a property, or None."""
    for claim in (claims or {}).get(prop, []):
        value = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
        amount = value.get("amount")
        unit = (value.get("unit") or "").rsplit("/", 1)[-1]
        try:
            val = float(amount)
        except (TypeError, ValueError):
            continue
        mult = _UNIT_TO_M.get(unit)
        if mult is not None and val > 0:
            return val * mult
    return None


def wikidata_dimensions(name: str, shape: str) -> dict | None:
    """Cited ``{dim_name: Fact}`` (CC0) from Wikidata for ``name`` mapped onto an
    unambiguous ``shape`` (sphere or cylinder), or None."""
    shape = (shape or "").strip().lower()
    if shape not in ("sphere", "cylinder", "cone"):
        return None
    qid = _search_qid(name)
    if not qid:
        return None
    d = web.get_json(f"{_API}?action=wbgetclaims&format=json&entity={qid}")
    claims = (d or {}).get("claims") or {}
    dia = _length_m(claims, _P_DIAMETER)
    src = f"Wikidata {qid} ({_P_DIAMETER}/{_P_HEIGHT})"
    out = {}
    if dia:
        out["radius_m"] = Fact(round(dia / 2.0, 6), src, "CC0", 0.6)
    if shape in ("cylinder", "cone"):
        h = _length_m(claims, _P_HEIGHT)
        if h:
            out["height_m"] = Fact(round(h, 6), src, "CC0", 0.6)
    return out or None


__all__ = ["wikidata_dimensions"]
