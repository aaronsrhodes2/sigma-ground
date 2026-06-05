"""Researched 2D outlines, distilled from Quick, Draw! (CC BY 4.0).

``outline_of(name)`` returns a canonical, NORMALIZED closed profile (unit extent
along its principal axis) for an organic noun, plus its cited source — the 2D
shape the kernel ``Outline`` primitive sweeps into 3D. The outlines are distilled
offline by ``tools/distill_quickdraw.py`` (medoid of many doodles) into
``inventory/data/outlines/<slug>.json``; here we just load and match the name.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

_DIR = (pathlib.Path(__file__).resolve().parents[2]
        / "inventory" / "data" / "outlines")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w}


@functools.lru_cache(maxsize=1)
def _table() -> dict:
    out = {}
    if not _DIR.is_dir():
        return out
    for p in _DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        prof = [(float(u), float(v)) for u, v in d.get("profile", [])]
        if len(prof) >= 3:
            out[p.stem] = (prof, d.get("source", ""), d.get("license", ""))
    return out


def outline_of(name: str):
    """(profile, source, license) for an organic noun, or None. Matched by slug,
    else whole-word containment (the outline's words all appear in the query)."""
    t = _table()
    key = _slug(name)
    if key in t:
        return t[key]
    qw = _words(name)
    best, best_len = None, 0
    for slug, val in t.items():
        sw = _words(slug)
        if sw and sw <= qw and len(sw) > best_len:
            best_len, best = len(sw), val
    return best


__all__ = ["outline_of"]
