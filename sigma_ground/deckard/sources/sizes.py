"""Typical OVERALL size of an object — its longest dimension — for plausibility
scaling. The local model is good at *proportions* but bad at *absolute size*; a
grounded overall size lets Deckard scale a whole construct to something sane
(the 90 kg toaster problem) while keeping the model's proportions.

A local curated reference (common everyday objects) first; then Wikidata's
largest length property (height/length/width/diameter) — agnostic but sparse.
Returns ``(size_m, source, license)`` or None.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

from . import dimensions_api

_JSON = (pathlib.Path(__file__).resolve().parents[2]
         / "inventory" / "data" / "object_sizes.json")


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
        size = d.get("size_m")
        if obj and isinstance(size, (int, float)) and size > 0:
            names = {obj} | {str(a).strip().lower() for a in (d.get("aliases") or [])}
            out.append((names, float(size), d.get("source", ""), d.get("license", "")))
    return out


def _local(name: str, _aliased: bool = False):
    qw = _words(name)
    best, best_len = None, 0
    for names, size, src, lic in _table():
        m = max((len(w) for w in (_words(nm) for nm in names) if w and w <= qw), default=0)
        if m > best_len:
            best_len, best = m, (size, src, lic)
    if best is None and not _aliased:
        # exact-name alias retries only (no containment stacking — see
        # shapenetsem._entry for the bowling-pin cautionary tale)
        from . import aliases
        for alt in sorted(aliases.expand(name)):
            aw = _words(alt)
            for names, size, src, lic in _table():
                if any(_words(nm) == aw for nm in names):
                    return (size, src, lic)
    return best


def _wikidata(name: str):
    qid = dimensions_api._search_qid(name)
    if not qid:
        return None
    d = dimensions_api.web.get_json(
        f"{dimensions_api._API}?action=wbgetclaims&format=json&entity={qid}")
    claims = (d or {}).get("claims") or {}
    sizes = [dimensions_api._length_m(claims, p)
             for p in ("P2048", "P2043", "P2049", "P2386")]      # height/length/width/diameter
    sizes = [s for s in sizes if s]
    if not sizes:
        return None
    return (round(max(sizes), 4), f"Wikidata {qid}", "CC0")


def typical_size_of(name: str, *, allow_web: bool = False):
    """Typical overall size (longest dimension, m) for ``name``: curated table
    first, then the ShapeNetSem-distilled medians (~1.2k names, measured-world),
    then Wikidata if ``allow_web``. None if unknown."""
    got = _local(name)
    if got:
        return got
    from . import shapenetsem
    got = shapenetsem.size_of(name)
    if got:
        return got
    if allow_web:
        return _wikidata(name)
    return None


__all__ = ["typical_size_of"]
