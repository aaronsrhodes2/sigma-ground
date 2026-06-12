"""ShapeNetSem-distilled physical priors — REAL measured-world grounding.

``inventory/data/shapenetsem_sizes.json`` is the aggregate fact-table distilled
offline by ``tools/distill_shapenetsem.py`` from ShapeNetSem (Savva et al. 2015):
per name (category token / WordNet lemma / synset alias) the MEDIAN real-world
aligned dimensions over n>=3 models, the category's material-composition ratios,
and a material->density table. Derived facts only; the gated dataset itself is
never shipped (ShapeNet Terms of Use — see docs/DATA_SOURCES.md).

This grounds two of Deckard's weakest guesses:
  * ``size_of(name)``      — typical overall size (longest median extent, m);
  * ``materials_of(name)`` — what the object is typically MADE OF (ratios),
                             surfaced to the researcher as a prompt hint.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

_JSON = (pathlib.Path(__file__).resolve().parents[2]
         / "inventory" / "data" / "shapenetsem_sizes.json")
_SOURCE = "ShapeNetSem (Savva et al. 2015)"
_LICENSE = "ShapeNet Terms of Use (non-commercial research)"


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w}


@functools.lru_cache(maxsize=1)
def _doc() -> dict:
    try:
        return json.loads(_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _entry(name: str, _aliased: bool = False):
    """The best table entry for ``name``: exact key, else whole-word containment
    (the key's words all appear in the query; longest key wins), else the
    query's WordNet aliases (taxonomy table) retried once."""
    cats = _doc().get("categories") or {}
    key = name.strip().lower()
    if key in cats:
        return cats[key]
    qw = _words(name)
    best, best_len = None, 0
    for nm, entry in cats.items():
        nw = _words(nm)
        if nw and nw <= qw and len(nw) > best_len:
            best_len, best = len(nw), entry
    if best is None and not _aliased:
        # alias retries match EXACT keys only: aliases are precise lemma swaps,
        # and stacking word-containment on top cascades into false friends
        # ("pushpin" -> "drawing pin" -> containment -> bowling "pin").
        from . import aliases
        for alt in sorted(aliases.expand(name)):
            best = cats.get(alt.strip().lower())
            if best is not None:
                break
    return best


def size_of(name: str):
    """(size_m, source, license) — median real-world longest extent, or None."""
    e = _entry(name)
    if not e:
        return None
    return (e["size_m"], f"{_SOURCE}, median of {e['n']} models", _LICENSE)


def dims_of(name: str):
    """(dims_m, n, source, license) — the median aligned extents, or None."""
    e = _entry(name)
    if not e:
        return None
    return (list(e["dims_m"]), e["n"], _SOURCE, _LICENSE)


def materials_of(name: str):
    """({material: ratio}, source, license) for the object's category, or None."""
    e = _entry(name)
    if not e or not e.get("materials"):
        return None
    return (dict(e["materials"]), _SOURCE, _LICENSE)


def density_of(material: str):
    """kg/m3 from the ShapeNetSem material table, or None (our own materials.json
    is richer — this is a fallback vocabulary for Sem's category materials)."""
    d = (_doc().get("material_densities") or {}).get(material.strip().lower())
    return None if d is None else float(d)


def hint(name: str) -> str:
    """A one-line material-prior hint for the researcher's prompt, or ''."""
    got = materials_of(name)
    if not got:
        return ""
    mats, src, _ = got
    top = ", ".join(f"{m} ({v:.0%})" for m, v in list(mats.items())[:3])
    return f"Typical materials for this object ({src}): {top}."


__all__ = ["size_of", "dims_of", "materials_of", "density_of", "hint"]
