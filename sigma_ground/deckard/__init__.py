"""Deckard — the shape researcher (the matter compiler).

Named for the Diablo NPC who identifies unidentified items: Deckard turns a
*name* ("coffee cup") into grounded *matter*. Two beats:

  1. RESEARCH  — resolve a named object to a concrete, cited ConstructSpec
                 (research.py): a frozen markdown catalog hit, else the
                 Researcher (LLM grounded by our data + free factual APIs),
                 else a flagged best-guess. Never a confident fake.
  2. COMPILE   — fit the primitive kit and build the layered CSG/SDF construct,
                 then integrate mass / centre-of-mass / inertia (construct.py),
                 cross-checking the SDF integrator against the closed-form
                 volumes so the matter is provably right.

Deckard's compiled `Construct` is consumed by TWO layers — Materia (to move it)
and Radiance (to render it) — so it lives in the matter core, below both, and
reuses the geometry kernel (`sigma_ground.kernel`).

It researches *form-facts* (dimensions, proportions, topology) and fits
primitives to them; it never ingests a foreign mesh.
"""
from __future__ import annotations

from . import catalog
from .schema import (ConstructSpec, Fact, SpecLayer,
                     emit_markdown, parse_markdown)
from .research import research, CATALOG
from .construct import compile, compile_vessel, Construct, Layer

# Back-compat: ItemSpec is now the generalised ConstructSpec.
ItemSpec = ConstructSpec


def identify(name: str, resolution: int = 64) -> Construct:
    """Full Deckard pipeline: a name → researched, compiled, validated matter."""
    return compile(research(name), resolution=resolution)


__all__ = [
    "identify", "research", "catalog", "CATALOG",
    "ConstructSpec", "ItemSpec", "Fact", "SpecLayer",
    "emit_markdown", "parse_markdown",
    "compile", "compile_vessel", "Construct", "Layer",
]
