"""Deckard's frozen catalog — slug → cited ConstructSpec (markdown on disk).

A catalog hit is the deterministic, offline path: resolve a name to a slug,
read ``catalog/<slug>.md``, and parse its canonical json payload into a
ConstructSpec. Researched specs are frozen here (``save_for``) and reused next
time, so a given object is researched once and identified deterministically after.
"""
from __future__ import annotations

import pathlib
import re

from ..schema import ConstructSpec, emit_markdown, parse_markdown

_DIR = pathlib.Path(__file__).resolve().parent

# curated name (lowercased) -> slug.  Containment-matched too ("a ceramic mug").
ALIASES = {
    "coffee cup": "coffee_cup",
    "coffee mug": "coffee_cup",
    "mug":        "coffee_cup",
    "cup":        "coffee_cup",
    "teacup":     "coffee_cup",
}


def slugify(name: str) -> str:
    """Filesystem-safe slug for an arbitrary object name."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def slug_for(name: str) -> str | None:
    """Resolve a free-text name to a *curated* catalog slug, or None."""
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    for alias, slug in ALIASES.items():
        if alias in key:
            return slug
    return None


def _resolve(name: str) -> str:
    """The slug a name reads from / writes to (curated alias else slugified)."""
    return slug_for(name) or slugify(name)


def path_for(slug: str) -> pathlib.Path:
    return _DIR / f"{slug}.md"


def has(slug: str) -> bool:
    return path_for(slug).is_file()


def load(slug: str) -> ConstructSpec:
    """Parse catalog/<slug>.md into a ConstructSpec (raises if absent)."""
    return parse_markdown(path_for(slug).read_text(encoding="utf-8"))


def lookup(name: str) -> ConstructSpec | None:
    """Catalog hit for a free-text name (curated alias or a frozen slug)."""
    slug = _resolve(name)
    if slug and has(slug):
        return load(slug)
    return None


def save(slug: str, spec: ConstructSpec) -> pathlib.Path:
    """Freeze a ConstructSpec to catalog/<slug>.md and return its path."""
    p = path_for(slug)
    p.write_text(emit_markdown(spec), encoding="utf-8")
    return p


def save_for(name: str, spec: ConstructSpec) -> pathlib.Path:
    """Freeze a researched spec under the slug ``lookup(name)`` will find."""
    return save(_resolve(name), spec)


__all__ = ["ALIASES", "slugify", "slug_for", "path_for",
           "has", "load", "lookup", "save", "save_for"]
