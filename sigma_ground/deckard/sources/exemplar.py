"""Representative real-object part layouts, distilled from PartNet.

For each category, ``tools/distill_partnet.py exemplar`` picked representative
real model(s) and recorded each one's leaf parts as primitives at their REAL
positions (normalized to the model's bbox). ``exemplar_of(name)`` returns one
such layout, so Deckard assembles "a chair" from how a real chair is actually
built — legs where they really are — rather than from a hand-written template.

When a category ships MORE THAN ONE exemplar (a variant POOL — distinct real
models with the same category name + distinct ``source_anno_id``), the selector
can pick among them:
  * ``exclude`` — a set of anno_ids already used, so "a *different* chair" grabs
    a fresh model rather than the one we already solved;
  * ``hint`` — prefers a model tagged with a subtype/material ("office", "wood").
If exclusion would empty the pool, the best remaining is returned (reuse) and
the caller can detect it by the returned anno_id. The SHAPE is always a real
model's layout — never fabricated to satisfy a request.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

_DIR = (pathlib.Path(__file__).resolve().parents[2]
        / "inventory" / "data" / "exemplars")


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w}


@functools.lru_cache(maxsize=1)
def _table() -> list:
    """Every shipped exemplar as (category_names, parts, source, license,
    anno_id, tags). Multiple files may share a category → a variant pool."""
    out = []
    if not _DIR.is_dir():
        return out
    for p in sorted(_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        parts = d.get("parts") or []
        if not parts:
            continue
        names = {(d.get("category") or p.stem).lower()} | {
            str(a).strip().lower() for a in (d.get("aliases") or [])}
        tags = {str(t).strip().lower() for t in (d.get("tags") or [])}
        out.append((names, parts, d.get("_source", ""), d.get("_license", ""),
                    str(d.get("source_anno_id", "")), tags))
    return out


def _tag_match(tags: set, hint: str) -> bool:
    """True if any word of the hint appears in any of the model's tags."""
    hw = _words(hint or "")
    return any(any(w in t for w in hw) for t in tags)


def _category_pool(name: str):
    """The variant pool: all exemplars of the best-matching category for name."""
    qw = _words(name)
    scored = []
    for entry in _table():
        names = entry[0]
        m = max((len(w) for w in (_words(nm) for nm in names) if w and w <= qw),
                default=0)
        if m > 0:
            scored.append((m, entry))
    if not scored:
        return []
    best = max(m for m, _ in scored)
    return [entry for m, entry in scored if m == best]


def exemplar_of(name: str, *, exclude=None, hint=None, _aliased: bool = False):
    """(parts, source, license, anno_id) for the object's category, or None.

    ``exclude`` (anno_ids) skips already-used models so a "different" request
    gets a fresh one; ``hint`` prefers a tagged variant. Matched by whole-word
    category containment (longest category wins); WordNet aliases retried once.
    """
    exclude = {str(x) for x in (exclude or ())}
    pool = _category_pool(name)
    if not pool:
        if not _aliased:
            from . import aliases
            for alt in sorted(aliases.expand(name)):
                aw = _words(alt)
                for names, parts, src, lic, anno, tags in _table():
                    if any(_words(nm) == aw for nm in names):
                        return (parts, src, lic, anno)
        return None

    def rank(entry):
        _names, _parts, _src, _lic, anno, tags = entry
        return (0 if (hint and _tag_match(tags, hint)) else 1,   # hinted variant first
                1 if anno in exclude else 0)                     # then a fresh model

    _names, parts, src, lic, anno, _tags = min(pool, key=rank)
    return (parts, src, lic, anno)


__all__ = ["exemplar_of"]
