"""Smithsonian Open Access adapter (only objects whose media usage is CC0 are kept).

FORMAT STATUS: UNVERIFIED, written from memory of the api.si.edu open-access search response. Assumed shape:

    {"response": {"rows": [{"id": "edanmdm:...", "title": "...",
        "content": {
          "descriptiveNonRepeating": {
            "record_link": "<url>", "unit_code": "NASM",
            "online_media": {"media": [{"type": "3d_model_or_other", "usage": {"access": "CC0"}, "content": "<url>"}]}},
          "freetext": {"physicalDescription": [{"label": "Dimensions", "content": "H x W x D: 10 x 20 x 30 cm"}]}}}]}}

Live use needs an api.data.gov key, which is Aaron's to obtain; none is used or stored here. Dimension text is parsed with
common.parse_dims_text; a record without a parsable three-number dimension string gets no size (size_unknown).
"""
from __future__ import annotations

from .common import AdapterFormatError, filter_servable, found_card, label_order_to_whd, parse_dims_text

SOURCE = "Smithsonian Open Access"
FIXTURE_STATUS = "from memory, unverified"


def parse_listing_with_report(raw, *, include_disallowed: bool = False):
    try:
        rows = raw["response"]["rows"]
    except (KeyError, TypeError):
        raise AdapterFormatError("expected {'response': {'rows': [...]}}") from None
    cards, problems = [], []
    for r in rows:
        try:
            content = r["content"]
            dnr = content["descriptiveNonRepeating"]
            title = r.get("title") or dnr["title"]["content"]
        except (KeyError, TypeError):
            problems.append(f"malformed row {str(r)[:60]}")
            continue
        medias = (dnr.get("online_media") or {}).get("media") or []
        access = next((m.get("usage", {}).get("access") for m in medias if m.get("usage")), None)
        size, flags = None, ["source_format_unverified"]
        for fld in (content.get("freetext") or {}).get("physicalDescription", []):
            if str(fld.get("label", "")).lower().startswith("dimension"):
                dims = parse_dims_text(fld.get("content", ""))
                if dims:
                    w, d, h = label_order_to_whd(dims, fld["content"])
                    size = {"w": w, "d": d, "h": h, "basis": "sourced", "source": f"Smithsonian record dimensions: {fld['content']}"}
                    break
        cards.append(found_card(
            source=SOURCE, ext_id=r.get("id", title), name=title, tags=[dnr.get("unit_code", "")], size=size,
            license_raw=access or "UNKNOWN", author=f"Smithsonian Institution ({dnr.get('unit_code', '?')})",
            url=dnr.get("record_link"), style="scan", extra_flags=flags))
    kept, skipped = filter_servable(cards, include_disallowed)
    return kept, {"skipped": [c["id"] for c in skipped], "problems": problems}


def parse_listing(raw, **kw):
    return parse_listing_with_report(raw, **kw)[0]
