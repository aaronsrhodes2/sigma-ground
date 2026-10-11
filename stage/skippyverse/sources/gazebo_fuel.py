"""Gazebo Fuel adapter. Only CC0 and CC-BY 4.0 models are kept (design section 6: the NC/SA ones are filtered out).

FORMAT STATUS: VERIFIED 2026-10-10 against the live API (GET https://fuel.gazebosim.org/1.0/models?per_page=100&page=N, a JSON array,
3387 models, no key). Each record has name, owner, description, tags, categories, license_name, license_url, filesize, downloads and
thumbnail_url. The licence names are spelled out ("Creative Commons Attribution 4.0 International", "Creative Commons Zero v1.0
Universal"), and license_url carries the version, so no licence version is assumed. The listing has NO size: a card has none until
the model's mesh is fetched and measured (the caller may pass `bbox_m`). The model page is https://fuel.gazebosim.org/<owner>/models/<name>.
"""
from __future__ import annotations

from ..cards import normalize_licence
from .common import AdapterFormatError, bbox_to_size, filter_servable, found_card

SOURCE = "Gazebo Fuel"
FIXTURE_STATUS = "real listing, verified 2026-10-10 (fixtures in data/source_fixtures.json are older, from memory)"

def map_licence(name: str, url: str = "") -> tuple[str, bool]:
    """(canonical licence, version_was_assumed). The URL's version wins; unknown names fall through to normalize_licence."""
    n = (name or "").strip()
    u = (url or "").lower()
    if "publicdomain/zero" in u or n.lower().startswith("creative commons zero"):
        return "CC0", False
    if "/licenses/by/4.0" in u and "Attribution 4.0 International" == n.replace("Creative Commons ", ""):
        return "CC-BY-4.0", False
    if "/licenses/by/3.0" in u:
        return "CC-BY-3.0", False
    return normalize_licence(n), False


def _unique(seen, ext_id):
    """Two Fuel names can slug to the same id (case, underscores): the later one gets a number."""
    from ..cards import slug
    k = slug(ext_id)
    seen[k] = seen.get(k, 0) + 1
    return ext_id if seen[k] == 1 else f"{ext_id}-{seen[k]}"


def parse_listing_with_report(raw, *, up: str = "z", include_disallowed: bool = False):
    if not isinstance(raw, list):
        raise AdapterFormatError("Fuel listing must be a list of model records")
    cards, problems, seen = [], [], {}
    for r in raw:
        if not isinstance(r, dict) or "name" not in r or "owner" not in r:
            problems.append(f"not a model record: {str(r)[:60]}")
            continue
        lic, assumed = map_licence(r.get("license_name", ""), r.get("license_url", ""))
        size = bbox_to_size(r["bbox_m"], up, "measured", "measured by the caller from the downloaded mesh") if r.get("bbox_m") else None
        flags = (["licence_version_assumed"] if assumed else []) + (["size_unknown"] if size is None else [])
        cards.append(found_card(
            source=SOURCE, ext_id=_unique(seen, f"{r['owner']}-{r['name']}"), name=r["name"], tags=list(r.get("tags") or []) + list(r.get("categories") or []), size=size,
            license_raw=lic, author=r["owner"], url=f"https://fuel.gazebosim.org/{r['owner']}/models/{r['name'].replace(' ', '%20')}",
            style="simulation", extra_flags=flags))
    kept, skipped = filter_servable(cards, include_disallowed)
    return kept, {"skipped": [c["id"] for c in skipped], "problems": problems}


def parse_listing(raw, **kw):
    return parse_listing_with_report(raw, **kw)[0]
