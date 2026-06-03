"""Deckard — the shape researcher (the matter compiler).

Named for the Diablo NPC who identifies unidentified items: Deckard turns a
*name* ("coffee cup") into grounded *matter*. Two beats:

  1. RESEARCH  — resolve a named object to concrete, cited spatial dimensions
                 (research.py: a grounded catalog now; dimension DBs / web later).
  2. COMPILE   — fit the primitive kit and build the layered CSG/SDF construct,
                 then integrate its mass / centre-of-mass / inertia
                 (construct.py), cross-checking the SDF integrator against the
                 closed-form volumes so the matter is provably right.

Deckard's compiled `Construct` is consumed by TWO layers — Materia (to move it)
and Radiance (to render it) — so it lives in the matter core, below both, and
reuses the existing geometry kernel (`sigma_ground.shapes`, `sigma_ground.csg`).

It researches *form-facts* (dimensions, proportions, topology) and fits
primitives to them; it never ingests a foreign mesh.
"""
from __future__ import annotations

from .research import research, ItemSpec, CATALOG
from .construct import (compile_vessel, Construct, Layer)


def identify(name: str, resolution: int = 64) -> Construct:
    """Full Deckard pipeline: a name → researched, compiled, validated matter."""
    spec = research(name)
    return compile_vessel(spec, resolution=resolution)


__all__ = [
    "identify",
    "research",
    "ItemSpec",
    "CATALOG",
    "compile_vessel",
    "Construct",
    "Layer",
]
