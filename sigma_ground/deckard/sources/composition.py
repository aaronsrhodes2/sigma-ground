"""Object part-decompositions — a COMPOSITION prior for the researcher.

``composition_of(name)`` returns the canonical parts of a multi-part object (each
with a primitive-shape hint, optionally a count), so the researcher can anchor its
decomposition in a known structure rather than pure recall — the structural
analogue of dimension grounding ("LLM proposes, our data grounds", applied to
*which parts* instead of *how big*).

Seeded today from common knowledge (every entry's ``source`` says so), under a
schema designed to be REPLACED, not merely extended, by PartNet's cited part
hierarchies once that data is available: a ``tools/distill_partnet.py`` will
overwrite ``inventory/data/compositions.json`` with PartNet-distilled entries
(object -> semantic leaf parts). Same loader, same researcher wiring — a clean
data swap, no code change.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

_JSON = (pathlib.Path(__file__).resolve().parents[2]
         / "inventory" / "data" / "compositions.json")


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w}


@functools.lru_cache(maxsize=1)
def _table() -> list:
    try:
        data = json.loads(_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for d in data:
        if not isinstance(d, dict):
            continue
        obj = (d.get("object") or "").strip().lower()
        raw = d.get("parts") or []
        if not (obj and isinstance(raw, list) and raw):
            continue
        names = {obj} | {str(a).strip().lower() for a in (d.get("aliases") or [])}
        parts = [{"name": str(p["name"]), "shape": str(p.get("shape", "")),
                  "count": int(p.get("count", 1))}
                 for p in raw if isinstance(p, dict) and p.get("name")]
        if parts:
            out.append((names, parts, d.get("source", ""), d.get("license", "")))
    return out


def composition_of(name: str):
    """(parts, source, license) for a known multi-part object, or None. Matched by
    whole-word containment (the object's words all appear in the query)."""
    qw = _words(name)
    best, best_len = None, 0
    for names, parts, source, lic in _table():
        m = max((len(w) for w in (_words(nm) for nm in names) if w and w <= qw), default=0)
        if m > best_len:
            best_len, best = m, (parts, source, lic)
    return best


def hint(name: str) -> str:
    """A short parts hint for the research prompt ('' if the object is unknown)."""
    got = composition_of(name)
    if not got:
        return ""
    items = []
    for p in got[0]:
        pre = f"{p['count']}x " if p.get("count", 1) > 1 else ""
        sh = f" ({p['shape']})" if p.get("shape") else ""
        items.append(f"{pre}{p['name']}{sh}")
    return "Typical parts of this object: " + ", ".join(items) + "."


__all__ = ["composition_of", "hint"]
