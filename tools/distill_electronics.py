"""Distill electrical properties for conductor materials from Wikidata (CC0).

For each element in our materials catalog: electrical resistivity (P5679, ohm
metre) and thermal conductivity (P2068, W/(m*K)) — the physics half of "working
lightbulb / ignition circuit" internals. Follows the materials_api doctrine:

  - entities are resolved by search AND VERIFIED (label matches + description
    says "chemical element") — never a blind first-hit;
  - units are whitelisted (Q1441459 ohm metre, Q1463969 W/(m*K)) — unknown
    units are refused, never unit-guessed;
  - when several temperature-qualified claims exist (P2076, degrees C), the one
    nearest 20 C is kept and its temperature recorded.

Output: sigma_ground/inventory/data/electrical_properties.json (cited, CC0).
Verification printed: copper ~1.7e-8 ohm*m, silver ~1.6e-8 (textbook).

Run:  python tools/distill_electronics.py
"""
import json
import os
import time
import urllib.parse
import urllib.request

_API = "https://www.wikidata.org/w/api.php"
_OUT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                    "inventory", "data", "electrical_properties.json")
_UNIT_OHM_M = "http://www.wikidata.org/entity/Q1441459"     # ohm metre
_UNIT_W_MK = "http://www.wikidata.org/entity/Q1463969"      # watt per metre-kelvin

# our material name -> the element label to resolve on Wikidata
ELEMENTS = ["copper", "aluminium", "silver", "gold", "iron", "nickel",
            "tungsten", "zinc", "lead", "titanium", "chromium", "platinum",
            "mercury", "tin", "silicon", "germanium", "carbon"]


def _get(url, _tries=5):
    """Polite fetch: paced, and on 429 honours Retry-After with backoff."""
    time.sleep(1.0)                                     # pace EVERY request
    for attempt in range(_tries):
        req = urllib.request.Request(url, headers={
            "User-Agent": "sigma-ground-data-lane/1.0 (research; polite; "
                          "contact: local project)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == _tries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 2 ** (attempt + 2))
            print(f"    (429 — backing off {wait:.0f}s)")
            time.sleep(wait)


def _element_qid(name: str) -> str | None:
    """Resolve + VERIFY: label match and 'chemical element' in the description."""
    q = urllib.parse.quote(name)
    d = _get(f"{_API}?action=wbsearchentities&format=json&language=en"
             f"&type=item&limit=8&search={q}")
    for hit in d.get("search", []):
        label = (hit.get("label") or "").lower()
        desc = (hit.get("description") or "").lower()
        if label == name.lower() and "chemical element" in desc:
            return hit["id"]
    return None


def _quantity_claims(qid: str, prop: str, unit_uri: str):
    """[(value, temp_C_or_None)] for claims of prop with the whitelisted unit."""
    d = _get(f"{_API}?action=wbgetclaims&format=json&entity={qid}&property={prop}")
    out = []
    for claim in (d.get("claims") or {}).get(prop, []):
        v = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if v.get("unit") != unit_uri:
            continue                                    # refuse unknown units
        try:
            val = float(v.get("amount"))
        except (TypeError, ValueError):
            continue
        temp = None
        for q in claim.get("qualifiers", {}).get("P2076", []):
            qv = q.get("datavalue", {}).get("value", {})
            try:
                temp = float(qv.get("amount"))
            except (TypeError, ValueError, AttributeError):
                pass
        out.append((val, temp))
    return out


def _nearest_room_temp(claims):
    """Prefer the claim whose temperature qualifier is nearest 20 C."""
    if not claims:
        return None
    return min(claims, key=lambda c: abs((c[1] if c[1] is not None else 20.0) - 20.0))


def main() -> None:
    out = {"_meta": {
        "source": "Wikidata (CC0): P5679 electrical resistivity [ohm m], "
                  "P2068 thermal conductivity [W/(m K)]; temperature-qualified "
                  "claims nearest 20 C preferred",
        "generated_by": "tools/distill_electronics.py"}}
    for name in ELEMENTS:
        qid = _element_qid(name)
        if qid is None:
            print(f"  ! {name}: no verified element entity — skipped")
            continue
        entry = {"qid": qid, "license": "CC0"}
        rho = _nearest_room_temp(_quantity_claims(qid, "P5679", _UNIT_OHM_M))
        if rho:
            entry["resistivity_ohm_m"] = rho[0]
            entry["resistivity_at_c"] = rho[1]
        kth = _nearest_room_temp(_quantity_claims(qid, "P2068", _UNIT_W_MK))
        if kth:
            entry["thermal_conductivity_w_mk"] = kth[0]
            entry["thermal_conductivity_at_c"] = kth[1]
        if len(entry) > 2:
            out[name] = entry
            r = entry.get("resistivity_ohm_m")
            k = entry.get("thermal_conductivity_w_mk")
            print(f"  + {name:10s} {qid:8s} rho={r if r is not None else '—'} ohm*m"
                  f"  k={k if k is not None else '—'} W/mK")
        else:
            print(f"  ! {name}: {qid} has neither property with known units")
    dest = os.path.abspath(_OUT)
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(f"\nwrote {dest} ({len(out) - 1} materials)")
    cu = out.get("copper", {}).get("resistivity_ohm_m")
    ag = out.get("silver", {}).get("resistivity_ohm_m")
    print(f"verify: copper rho = {cu} (expect ~1.7e-8), silver rho = {ag} (expect ~1.6e-8)")


if __name__ == "__main__":
    main()
