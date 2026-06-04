"""Deckard's frozen catalog — slug → cited ConstructSpec (markdown on disk).

A catalog hit is the deterministic, offline path: resolve a name to a slug,
read ``catalog/<slug>.md``, and parse its canonical json payload into a
ConstructSpec. Researched specs are written here (frozen) and reused next time,
so a given object is researched once and identified deterministically after.
"""
from __future__ import annotations

import pathlib

from ..schema import ConstructSpec, emit_markdown, parse_markdown

_DIR = pathlib.Path(__file__).resolve().parent

# name (lowercased) -> slug.  Containment-matched too ("a ceramic mug" -> mug).
ALIASES = {
    "coffee cup": "coffee_cup",
    "coffee mug": "coffee_cup",
    "mug":        "coffee_cup",
    "cup":        "coffee_cup",
    "teacup":     "coffee_cup",
}


def slug_for(name: str) -> str | None:
    """Resolve a free-text name to a catalog slug, or None."""
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    for alias, slug in ALIASES.items():
        if alias in key:
            return slug
    return None


def path_for(slug: str) -> pathlib.Path:
    return _DIR / f"{slug}.md"


def has(slug: str) -> bool:
    return path_for(slug).is_file()


def load(slug: str) -> ConstructSpec:
    """Parse catalog/<slug>.md into a ConstructSpec (raises if absent)."""
    return parse_markdown(path_for(slug).read_text(encoding="utf-8"))


def lookup(name: str) -> ConstructSpec | None:
    """Catalog hit for a free-text name, or None on a miss."""
    slug = slug_for(name)
    if slug and has(slug):
        return load(slug)
    return None


def save(slug: str, spec: ConstructSpec) -> pathlib.Path:
    """Freeze a ConstructSpec to catalog/<slug>.md and return its path."""
    p = path_for(slug)
    p.write_text(emit_markdown(spec), encoding="utf-8")
    return p


__all__ = ["ALIASES", "slug_for", "path_for", "has", "load", "lookup", "save"]
