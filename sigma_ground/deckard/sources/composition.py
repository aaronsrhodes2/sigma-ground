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
        parts = []
        for p in raw:
            if not (isinstance(p, dict) and p.get("name")):
                continue
            part = {"name": str(p["name"]), "shape": str(p.get("shape", "")),
                    "count": int(p.get("count", 1))}
            # enriched PartNet-geometry keys (optional) pass straight through —
            # the researcher's placement/proportion priors consume them
            for k in ("freq", "size_frac", "z_frac", "r_frac"):
                if k in p:
                    part[k] = p[k]
            parts.append(part)
        if parts:
            out.append((names, parts, d.get("source", ""), d.get("license", "")))
    return out


def composition_of(name: str, _aliased: bool = False):
    """(parts, source, license) for a known multi-part object, or None. Matched by
    whole-word containment (the object's words all appear in the query); on a
    miss, the query's WordNet aliases (taxonomy table) are retried once."""
    qw = _words(name)
    best, best_len = None, 0
    for names, parts, source, lic in _table():
        m = max((len(w) for w in (_words(nm) for nm in names) if w and w <= qw), default=0)
        if m > best_len:
            best_len, best = m, (parts, source, lic)
    if best is None and not _aliased:
        from . import aliases
        for alt in sorted(aliases.expand(name)):
            best = composition_of(alt, _aliased=True)
            if best is not None:
                break
    return best


_ZONE = {None: "", -1: "lower half", 0: "mid", 1: "upper half"}


def hint(name: str) -> str:
    """A short parts hint for the research prompt ('' if the object is unknown).
    Geometry-enriched entries (PartNet census) say WHERE and HOW BIG each part
    typically is — compact enough for a local 7B model to digest."""
    got = composition_of(name)
    if not got:
        return ""
    items = []
    measured = False
    for p in got[0]:
        pre = f"{p['count']}x " if p.get("count", 1) > 1 else ""
        sh = f" ({p['shape']})" if p.get("shape") else ""
        place = ""
        if "z_frac" in p:
            measured = True
            zone = -1 if p["z_frac"] < -0.12 else (1 if p["z_frac"] > 0.12 else 0)
            bits = [_ZONE[zone]]
            sf = p.get("size_frac")
            if sf:
                bits.append(f"~{max(sf):.0%} of the object's largest extent")
            place = ", " + ", ".join(bits)
        items.append(f"{pre}{p['name']}{sh}{place}")
    head = ("Typical parts of this object (measured over many real examples): "
            if measured else "Typical parts of this object: ")
    return head + "; ".join(items) + "."


__all__ = ["composition_of", "hint"]
