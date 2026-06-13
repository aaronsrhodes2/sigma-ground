"""Representative real-object part layouts, distilled from PartNet.

For each category, ``tools/distill_partnet.py exemplar`` picked ONE typical real
model and recorded its leaf parts as primitives at their REAL positions
(normalized to the model's bbox). ``exemplar_of(name)`` returns that layout, so
Deckard assembles "a chair" from how a real chair is actually built — legs where
they really are, a back at its real place — rather than from a hand-written
template. Dimensions stay grounded: the caller scales the unit layout to the
object's real overall size (ShapeNetSem) and assigns its material composition.
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
        out.append((names, parts, d.get("_source", ""), d.get("_license", "")))
    return out


def exemplar_of(name: str, _aliased: bool = False):
    """(parts, source, license) for the object's category, or None. Matched by
    whole-word containment (longest category wins); WordNet aliases retried once."""
    qw = _words(name)
    best, best_len = None, 0
    for names, parts, src, lic in _table():
        m = max((len(w) for w in (_words(nm) for nm in names) if w and w <= qw),
                default=0)
        if m > best_len:
            best_len, best = m, (parts, src, lic)
    if best is None and not _aliased:
        from . import aliases
        for alt in sorted(aliases.expand(name)):
            aw = _words(alt)
            for names, parts, src, lic in _table():
                if any(_words(nm) == aw for nm in names):
                    return (parts, src, lic)
    return best


__all__ = ["exemplar_of"]
