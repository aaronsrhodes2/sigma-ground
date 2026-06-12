"""Name aliases from the ShapeNet/WordNet taxonomy — widen the sources' reach.

``expand(name)`` returns the OTHER lemmas of any WordNet group the (normalized)
name belongs to ("cellphone" -> {"cellular telephone", "mobile phone", ...}), so
a source keyed under one lemma still answers a query phrased with another. The
table (``inventory/data/shape_aliases.json``) ships vocabulary only — distilled
offline by ``tools/distill_shapenet_taxonomy.py``, no model data.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

_JSON = (pathlib.Path(__file__).resolve().parents[2]
         / "inventory" / "data" / "shape_aliases.json")
_ARTICLES = re.compile(r"^(a|an|the)\s+", re.I)


def _norm(s: str) -> str:
    return _ARTICLES.sub("", (s or "").strip().lower())


@functools.lru_cache(maxsize=1)
def _by_lemma() -> dict:
    try:
        groups = json.loads(_JSON.read_text(encoding="utf-8")).get("groups") or []
    except Exception:
        return {}
    out: dict[str, set] = {}
    for g in groups:
        gs = {str(w).lower() for w in g}
        for w in gs:
            out.setdefault(w, set()).update(gs - {w})
    return out

def expand(name: str) -> set:
    """Alternative names for ``name`` (possibly empty); never includes itself."""
    q = _norm(name)
    found = set(_by_lemma().get(q) or ())
    return found - {q}


__all__ = ["expand"]
