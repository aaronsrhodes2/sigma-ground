"""Poly Haven adapter (models are CC0).

FORMAT STATUS: VERIFIED 2026-10-10 against the live API (GET https://api.polyhaven.com/assets?t=models, no key): a dict keyed by asset id
with 521 models, every record type 2, name, categories, tags, authors {name: "All"}, polycount, and `dimensions` [x, y, z] in
MILLIMETRES with z the height (an armchair reads 848 x 766 x 1065, a barrel 563 x 563 x 880). The page URL is https://polyhaven.com/a/<id>.
Sizes are labelled `sourced` because the source publishes them.
"""
from __future__ import annotations

from .common import AdapterFormatError, bbox_to_size, filter_servable, found_card

SOURCE = "Poly Haven"
FIXTURE_STATUS = "real listing, verified 2026-10-10"


def parse_listing_with_report(raw, *, up: str = "z", include_disallowed: bool = False):
    if not isinstance(raw, dict):
        raise AdapterFormatError("Poly Haven listing must be a dict keyed by asset id")
    cards, problems = [], []
    for aid, info in raw.items():
        if not isinstance(info, dict) or "name" not in info:
            problems.append(f"{aid}: not an asset record")
            continue
        dims = info.get("dimensions")
        size = None
        if dims and len(dims) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in dims):
            size = bbox_to_size([v / 1000.0 for v in dims], up, "sourced", "Poly Haven published dimensions (mm, z up)")
        author = ", ".join((info.get("authors") or {}).keys()) or None
        cats = info.get("categories") or []
        cards.append(found_card(
            source=SOURCE, ext_id=aid, name=info["name"], tags=list(info.get("tags") or []) + list(cats), size=size,
            license_raw="CC0", author=author, url=f"https://polyhaven.com/a/{aid}", style="photoscan-or-pbr",
            extra_flags=[]))
    kept, skipped = filter_servable(cards, include_disallowed)
    return kept, {"skipped": [c["id"] for c in skipped], "problems": problems}


def parse_listing(raw, **kw):
    return parse_listing_with_report(raw, **kw)[0]
