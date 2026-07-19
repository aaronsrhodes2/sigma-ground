"""Blueprint's frozen catalog — slug -> cited MechanismSpec (markdown on disk).

Same discipline as ``deckard.catalog``: a mechanism is sourced once (real
document -> transcribed, quoted, validated MechanismSpec) and frozen here, so
a demo rebuild reads committed reviewed text, never a live extraction call.
"""
from __future__ import annotations

import pathlib
import re

from ..schema import MechanismSpec, emit_markdown, parse_markdown

_DIR = pathlib.Path(__file__).resolve().parent


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def path_for(slug: str) -> pathlib.Path:
    return _DIR / f"{slug}.md"


def has(slug: str) -> bool:
    return path_for(slug).is_file()


def load(slug: str) -> MechanismSpec:
    return parse_markdown(path_for(slug).read_text(encoding="utf-8"))


def lookup(name: str) -> MechanismSpec | None:
    slug = slugify(name)
    return load(slug) if has(slug) else None


def available() -> list[str]:
    return sorted(p.stem for p in _DIR.glob("*.md"))


def save(slug: str, spec: MechanismSpec) -> pathlib.Path:
    p = path_for(slug)
    p.write_text(emit_markdown(spec), encoding="utf-8")
    return p


__all__ = ["slugify", "path_for", "has", "load", "lookup", "available", "save"]
