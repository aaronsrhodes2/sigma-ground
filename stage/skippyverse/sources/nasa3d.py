"""NASA 3D Resources adapter (US government work: credit NASA, imply no endorsement, leave out insignia).

FORMAT STATUS: UNVERIFIED, written from memory. NASA's site is a browsable catalogue; this adapter assumes a hand-made or scraped
array the caller supplies:

    [{"title": "Saturn V", "category": "Launch Vehicles", "page": "<url of the resource page>", "file": "saturn_v.glb",
      "bbox": [x, y, z], "real_size_m": [w, d, h] or null}, ...]

NASA models' units are arbitrary (design: a Saturn V of 2 x 13 x 2 units). A `real_size_m` supplied by the caller from a published
figure makes the size `sourced`; otherwise `bbox` is taken as metres and labelled `guessed`.
"""
from __future__ import annotations

from .common import AdapterFormatError, bbox_to_size, filter_servable, found_card

SOURCE = "NASA 3D Resources"
FIXTURE_STATUS = "repo tree verified 2026-10-10 (from_github_tree); the hand-made record format is from memory"


def from_github_tree(tree: dict) -> list[dict]:
    """Records for the adapter from the official repo's file tree (GET https://api.github.com/repos/nasa/NASA-3D-Resources/git/trees/master?recursive=1,
    verified 2026-10-10: 1583 entries, 257 .glb). One record per folder under '3D Models/' that holds a .glb: title = the folder name,
    file = the glb path, no size (the files are in arbitrary units; a sourced real size has to be added by hand or by the Deckard sizes)."""
    out = []
    for e in tree.get("tree", []):
        path = e.get("path", "")
        if e.get("type") == "blob" and path.startswith("3D Models/") and path.lower().endswith(".glb"):
            parts = path.split("/")
            if len(parts) >= 3:
                out.append({"title": parts[1], "category": "NASA 3D Models", "file": path, "bytes": e.get("size"),
                            "page": "https://github.com/nasa/NASA-3D-Resources/tree/master/" + "/".join(p.replace(" ", "%20") for p in parts[:2]),
                            "bbox": None, "real_size_m": None})
    return out


def parse_listing_with_report(raw, *, up: str = "y", include_disallowed: bool = False):
    if not isinstance(raw, list):
        raise AdapterFormatError("NASA listing must be a list")
    cards, problems = [], []
    for r in raw:
        if not isinstance(r, dict) or "title" not in r:
            problems.append(f"not a resource record: {str(r)[:60]}")
            continue
        size, flags = None, ["source_format_unverified", "nasa_credit_no_endorsement"]
        if r.get("real_size_m"):
            w, d, h = r["real_size_m"]
            size = {"w": w, "d": d, "h": h, "basis": "sourced", "source": "published figure supplied by the caller"}
        elif r.get("bbox"):
            size = bbox_to_size(r["bbox"], up, "guessed", "NASA model units taken as metres (scale is arbitrary)")
            flags.append("scale_arbitrary")
        cards.append(found_card(
            source=SOURCE, ext_id=r["file"] if r.get("file") else r["title"], name=r["title"], tags=[r.get("category", "")], size=size,
            license_raw="NASA", author="NASA", url=r.get("page"), style="agency", extra_flags=flags, asset=r.get("file")))
    kept, skipped = filter_servable(cards, include_disallowed)
    return kept, {"skipped": [c["id"] for c in skipped], "problems": problems}


def parse_listing(raw, **kw):
    return parse_listing_with_report(raw, **kw)[0]
