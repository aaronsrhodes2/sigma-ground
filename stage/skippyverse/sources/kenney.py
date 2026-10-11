"""Kenney adapter (each pack is CC0 per its own licence file).

FORMAT STATUS: UNVERIFIED. Kenney publishes zip packs, not an API, so there is no real "listing" format. This adapter takes a
hand-assembled description of ONE pack, which the caller (or a future local scanner reading the unzipped pack) builds:

    {"pack": "Car Kit", "pack_url": "<page of the pack>", "license": "CC0",
     "models": [{"file": "sedan.glb", "name": "sedan", "bbox": [x, y, z]}, ...]}

Kenney models are stylised and their scale is arbitrary (the design notes a sedan of 1.5 x 1.45 x 2.55 units), so a size taken
from `bbox` is labelled `guessed` and flagged `scale_arbitrary` until a real size pins it. `up` defaults to 'y' (glTF).
"""
from __future__ import annotations

from .common import AdapterFormatError, bbox_to_size, filter_servable, found_card

SOURCE = "Kenney"
FIXTURE_STATUS = "from memory, unverified (no real listing format exists; pack description is hand-made)"


def parse_listing_with_report(raw, *, up: str = "y", include_disallowed: bool = False):
    if not isinstance(raw, dict) or "models" not in raw or "pack" not in raw:
        raise AdapterFormatError("Kenney pack description needs 'pack' and 'models'")
    cards, problems = [], []
    for m in raw["models"]:
        if "file" not in m:
            problems.append(f"model without file: {m!r}")
            continue
        stem = m.get("name") or m["file"].rsplit(".", 1)[0]
        size = None
        flags = ["source_format_unverified"]
        if m.get("bbox"):
            size = bbox_to_size(m["bbox"], up, "guessed", "Kenney model units taken as metres (scale is arbitrary in this pack)")
            flags.append("scale_arbitrary")
        cards.append(found_card(
            source=SOURCE, ext_id=f"{raw['pack']}-{stem}", name=stem.replace("_", " ").replace("-", " "),
            tags=[raw["pack"]], size=size, license_raw=raw.get("license", "UNKNOWN"), author="Kenney",
            url=raw.get("pack_url"), style="stylised low-poly", blocky=True, extra_flags=flags))
    kept, skipped = filter_servable(cards, include_disallowed)
    return kept, {"skipped": [c["id"] for c in skipped], "problems": problems}


def parse_listing(raw, **kw):
    return parse_listing_with_report(raw, **kw)[0]
