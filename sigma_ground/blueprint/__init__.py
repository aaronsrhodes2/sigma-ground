"""Blueprint — the mechanism researcher (gear trains, springs, escapements).

Sibling to ``deckard`` (which researches SHAPE): where Deckard turns a name
into cited geometry, Blueprint turns a real source document (a patent claim,
a textbook worked example) into a cited ``MechanismSpec`` — tooth counts,
ratios, spring/escapement data, every value carrying a mandatory verbatim
quote back to the sentence it was read from.

The extraction discipline (Golden Rule, restated for mechanisms): the
pipeline's job is EXTRACTION, never proposal — locate and transcribe a
real source's stated numbers, compute derived ratios in plain deterministic
Python (validate.py), and never let an LLM fill a gap with a plausible
guess. A gap stays a gap (``# GEOMETRY_GAP:``), the same way the physics
layer flags ``# PHYSICS_GAP:`` rather than silently estimating.
"""
from __future__ import annotations

from . import catalog
from .schema import (BlueprintFact, GearSpec, MeshPair, SpringSpec,
                     EscapementSpec, MechanismSpec, emit_markdown, parse_markdown)
from .validate import ValidationReport, validate, cumulative_ratio

__all__ = [
    "catalog",
    "BlueprintFact", "GearSpec", "MeshPair", "SpringSpec", "EscapementSpec",
    "MechanismSpec", "emit_markdown", "parse_markdown",
    "ValidationReport", "validate", "cumulative_ratio",
]
