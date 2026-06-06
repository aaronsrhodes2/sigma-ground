"""Distill PartNet's part taxonomy into Deckard's composition table.

PartNet (Mo et al., CVPR 2019; MIT) publishes its per-category part-label
hierarchies in the repo's stats/after_merging_label_ids/ — PUBLIC, separate from
the account-gated meshes. A composition prior only needs the part *vocabulary*
(which parts a category has), not the geometry, so we can ground it from the
public taxonomy now.

For each of the 24 categories this tool, politely and cached:
  1. fetches <Category>-level-1.txt (the coarse parts),
  2. takes each line's leaf part label, strips the category prefix + grouping
     suffixes (_side/_set/_group/_unit), de-dupes and drops non-structural labels,
  3. attaches a rough primitive-shape hint by part name (our inference, flagged),
and MERGES the result into sigma_ground/inventory/data/compositions.json,
replacing any curated entry that PartNet covers and keeping the rest. PartNet
entries are cited "PartNet (Mo et al. 2019), MIT".

    python tools/distill_partnet.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sigma_ground.deckard.sources import web   # polite, cached fetch  # noqa: E402

_RAW = ("https://raw.githubusercontent.com/daerduoCarey/partnet_dataset/master/"
        "stats/after_merging_label_ids/{cat}-level-1.txt")
_OUT = (pathlib.Path(__file__).resolve().parents[1]
        / "sigma_ground" / "inventory" / "data" / "compositions.json")

# PartNet category -> (common object name, [aliases])
_CATEGORIES = {
    "Bag": ("bag", ["handbag"]), "Bed": ("bed", []), "Bottle": ("bottle", []),
    "Bowl": ("bowl", []), "Chair": ("chair", []), "Clock": ("clock", []),
    "Dishwasher": ("dishwasher", []), "Display": ("monitor", ["display", "screen"]),
    "Door": ("door", []), "Earphone": ("earphone", ["headphones", "earbuds"]),
    "Faucet": ("faucet", ["tap"]), "Hat": ("hat", ["cap"]), "Keyboard": ("keyboard", []),
    "Knife": ("knife", []), "Lamp": ("lamp", []), "Laptop": ("laptop", []),
    "Microwave": ("microwave", []), "Mug": ("mug", []),
    "Refrigerator": ("refrigerator", ["fridge"]), "Scissors": ("scissors", []),
    "StorageFurniture": ("cabinet", ["storage furniture", "dresser", "cupboard"]),
    "Table": ("table", ["desk"]), "TrashCan": ("trash can", ["bin", "garbage can"]),
    "Vase": ("vase", ["pot"]),
}

_NOISE = {"containing_things", "other", "others", "other_leaf"}
_SHAPE_HINT = {
    "body": "cylinder", "handle": "cylinder", "base": "cylinder", "leg": "cylinder",
    "stem": "cylinder", "shaft": "cylinder", "neck": "cylinder", "lid": "cylinder",
    "spout": "cylinder", "knob": "sphere", "button": "cylinder", "wheel": "cylinder",
    "foot": "cylinder", "screw": "cylinder", "cord": "cylinder", "chain": "cylinder",
    "seat": "box", "back": "box", "arm": "box", "top": "box", "tabletop": "box",
    "head": "box", "door": "box", "shelf": "box", "frame": "box", "drawer": "box",
    "surface": "box", "keyboard": "box", "screen": "box", "bench": "box",
    "blade": "outline", "bowl": "ellipsoid", "containing_things": "",
}


def _clean(path: str) -> str:
    root = path.split("/")[0]
    name = path.split("/")[-1]
    for suf in ("_side", "_group", "_set", "_unit"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    if name.startswith(root + "_"):
        name = name[len(root) + 1:]
    return name.strip("_")


def _parts_for(cat: str, obj: str):
    text = web.get_text(_RAW.format(cat=cat))
    if not text:
        return []
    skip = _NOISE | {obj.lower(), cat.lower()}    # drop self-references + noise labels
    parts, seen = [], set()
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 2:
            continue
        name = _clean(toks[1])
        if not name or name in skip or name in seen:
            continue
        seen.add(name)
        parts.append({"name": name, "shape": _SHAPE_HINT.get(name, ""), "count": 1})
    return parts[:8]                              # keep the coarse set readable


def main():
    try:
        existing = json.loads(_OUT.read_text(encoding="utf-8"))
    except Exception:
        existing = []
    pn_names = {nm.lower() for c in _CATEGORIES for nm in
                ([_CATEGORIES[c][0]] + _CATEGORIES[c][1])}
    # keep curated entries PartNet does NOT cover
    kept = [e for e in existing if (e.get("object") or "").lower() not in pn_names]

    pn_entries = []
    for cat, (name, aliases) in _CATEGORIES.items():
        parts = _parts_for(cat, name)
        if not parts:
            print("  (no taxonomy for", cat, "- skipped)")
            continue
        pn_entries.append({
            "object": name, "aliases": aliases, "parts": parts,
            "source": "PartNet (Mo et al. 2019)", "license": "MIT",
        })
        print("  %-16s -> %s" % (name, ", ".join(p["name"] for p in parts)))

    out = pn_entries + kept                       # PartNet first, then curated extras
    _OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {_OUT}  ({len(pn_entries)} PartNet + {len(kept)} curated)")


if __name__ == "__main__":
    main()
