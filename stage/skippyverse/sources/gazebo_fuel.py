"""Gazebo Fuel adapter. Only CC0 and CC-BY 4.0 models are kept (design section 6: the 47 NC/SA ones are filtered out).

FORMAT STATUS: UNVERIFIED, written from memory. The model list is assumed to be a JSON array of records:

    [{"name": "Fire Hydrant", "owner": "OpenRobotics", "description": "...", "tags": ["street"], "version": 4,
      "license_name": "Creative Commons - Public Domain", "bbox_m": [x, y, z]}, ...]

`license_name` strings and the model page URL pattern https://fuel.gazebosim.org/1.0/<owner>/models/<name> are from memory.
`bbox_m` is NOT a field I know Fuel lists (the real size would come from the model's SDF/mesh after download); the adapter accepts
it when the caller measured it. Without it the card has no size and is flagged size_unknown. The mapping of "Creative Commons -
Attribution" to CC-BY-4.0 is the design document's claim ("CC0 plus CC-BY 4.0"), not something checked here, so such cards carry the
flag `licence_version_assumed` and Aaron must confirm before anything is served.
"""
from __future__ import annotations

from ..cards import normalize_licence
from .common import AdapterFormatError, bbox_to_size, filter_servable, found_card

SOURCE = "Gazebo Fuel"
FIXTURE_STATUS = "from memory, unverified"

_LICENCE_NAMES = {
    "creative commons - public domain": ("CC0", False),
    "creative commons - attribution": ("CC-BY-4.0", True),                       # version assumed
    "creative commons - attribution - share alike": ("CC-SA", False),
    "creative commons - attribution - non commercial": ("CC-NC", False),
    "creative commons - attribution - non commercial - share alike": ("CC-NC", False),
    "creative commons - attribution - no derivatives": ("CC-ND", False),
}


def map_licence(name: str) -> tuple[str, bool]:
    """(canonical licence, version_was_assumed). Unknown names fall through to normalize_licence (and are not allowed)."""
    hit = _LICENCE_NAMES.get((name or "").strip().lower())
    return hit if hit else (normalize_licence(name), False)


def parse_listing_with_report(raw, *, up: str = "z", include_disallowed: bool = False):
    if not isinstance(raw, list):
        raise AdapterFormatError("Fuel listing must be a list of model records")
    cards, problems = [], []
    for r in raw:
        if not isinstance(r, dict) or "name" not in r or "owner" not in r:
            problems.append(f"not a model record: {str(r)[:60]}")
            continue
        lic, assumed = map_licence(r.get("license_name", ""))
        size = bbox_to_size(r["bbox_m"], up, "measured", "measured by the caller from the downloaded mesh") if r.get("bbox_m") else None
        flags = ["source_format_unverified"] + (["licence_version_assumed"] if assumed else [])
        cards.append(found_card(
            source=SOURCE, ext_id=f"{r['owner']}-{r['name']}", name=r["name"], tags=r.get("tags") or [], size=size,
            license_raw=lic, author=r["owner"], url=f"https://fuel.gazebosim.org/1.0/{r['owner']}/models/{r['name'].replace(' ', '%20')}",
            style="simulation", extra_flags=flags))
    kept, skipped = filter_servable(cards, include_disallowed)
    return kept, {"skipped": [c["id"] for c in skipped], "problems": problems}


def parse_listing(raw, **kw):
    return parse_listing_with_report(raw, **kw)[0]
