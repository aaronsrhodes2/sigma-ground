"""Poly Haven adapter (models are CC0).

FORMAT STATUS: UNVERIFIED, written from memory. The listing is assumed to look like the JSON of the public asset-list call, a
dict keyed by asset id:

    {"<id>": {"name": "...", "type": 2, "categories": ["furniture", "seating"], "tags": ["sofa", "couch"],
              "authors": {"<Author Name>": "All"}, "dimensions": [x_mm, y_mm, z_mm]}, ...}

Whether the real `dimensions` are millimetres, which axis is up, and whether `type` 2 means models are all UNCHECKED here (the design
document says "real dimensions in mm for every model", also unchecked). The adapter therefore takes `up` as a parameter (default
'z', the Blender convention Poly Haven authors in; wrong if the real field is y-up) and labels the size `sourced` only because the
source publishes it. The page URL pattern https://polyhaven.com/a/<id> is also from memory.
"""
from __future__ import annotations

from .common import AdapterFormatError, bbox_to_size, filter_servable, found_card

SOURCE = "Poly Haven"
FIXTURE_STATUS = "from memory, unverified"


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
            size = bbox_to_size([v / 1000.0 for v in dims], up, "sourced", "Poly Haven published dimensions (mm; format unverified)")
        author = ", ".join((info.get("authors") or {}).keys()) or None
        cats = info.get("categories") or []
        cards.append(found_card(
            source=SOURCE, ext_id=aid, name=info["name"], tags=list(info.get("tags") or []) + list(cats), size=size,
            license_raw="CC0", author=author, url=f"https://polyhaven.com/a/{aid}", style="photoscan-or-pbr",
            extra_flags=["source_format_unverified"]))
    kept, skipped = filter_servable(cards, include_disallowed)
    return kept, {"skipped": [c["id"] for c in skipped], "problems": problems}


def parse_listing(raw, **kw):
    return parse_listing_with_report(raw, **kw)[0]
