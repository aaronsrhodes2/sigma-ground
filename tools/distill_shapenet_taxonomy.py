"""Distill ShapeNet's taxonomy into a name-alias table for Deckard's sources.

ShapeNetCore's ``taxonomy.json`` (WordNet synsets: ``name`` is a comma-separated
lemma list like "cellular telephone,cellphone,handphone") is read DIRECTLY out of
the local ``ShapeNetCore.v2.zip`` with stdlib zipfile — no extraction, no meshes.
We ship only the lemma GROUPS (WordNet vocabulary, no model ids or counts):
``inventory/data/shape_aliases.json``. The sources' miss paths use them so
"cellphone" finds an entry keyed "cellular telephone" and vice versa.

    python tools/distill_shapenet_taxonomy.py
"""
from __future__ import annotations

import json
import pathlib
import zipfile

_ZIP = pathlib.Path("D:/datasets/shapenet/ShapeNetCore-archive/ShapeNetCore.v2.zip")
_MEMBER = "ShapeNetCore.v2/taxonomy.json"
_OUT = (pathlib.Path(__file__).resolve().parents[1]
        / "sigma_ground" / "inventory" / "data" / "shape_aliases.json")


def main() -> None:
    with zipfile.ZipFile(_ZIP) as z:
        data = json.loads(z.read(_MEMBER))
    groups, seen = [], set()
    for entry in data:
        lemmas = [w.strip().lower() for w in (entry.get("name") or "").split(",")]
        lemmas = [w for w in lemmas if w]
        if len(lemmas) < 2:
            continue                                   # a group of one aliases nothing
        key = tuple(sorted(lemmas))
        if key in seen:
            continue
        seen.add(key)
        groups.append(lemmas)
    _OUT.write_text(json.dumps({
        "_source": "ShapeNet taxonomy (Chang et al. 2015) / WordNet 3.0 lemma groups",
        "_license": "WordNet 3.0 license (vocabulary only; no model data)",
        "groups": groups,
    }, indent=1), encoding="utf-8")
    print(f"wrote {_OUT} — {len(groups)} alias groups")


if __name__ == "__main__":
    main()
