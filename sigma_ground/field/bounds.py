"""Domain bounds, as GOLDEN_RULES Rule 2 (Design by Contract) asks every model to state them.

A quantity is classified against the interval a model is valid on:

    SAFE    well inside the validated range
    EDGE    inside, but within `edge` (a fraction of the span) of a limit: the model still holds, extrapolation begins
    WALL    at or past a limit the model documents as its edge of validity (still computable, no longer trustworthy)
    BEYOND  impossible or undefined (negative absolute temperature, a speed above c, a probability of 1.2)

``classify`` is the shared implementation; field modules that already compute a regime by hand
(``field/dense``) use the same four words. Mentat's tool contract (``mcp/contract.py``) refuses a BEYOND input with a
structured error and discloses EDGE / WALL in the result; it never silently clamps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Regime(str, Enum):
    SAFE = "SAFE"
    EDGE = "EDGE"
    WALL = "WALL"
    BEYOND = "BEYOND"


@dataclass(frozen=True)
class Bounds:
    """The interval a quantity may take. ``hard_*``: the impossibility limit (definitional: T >= 0, v < c, 0 <= p <= 1);
    ``wall_*``: the documented edge of the model's validity (optional, inside the hard limits). ``None`` = unbounded that side.
    ``*_open``: the limit itself is excluded (a speed must be < c; a mass must be > 0)."""
    hard_lo: float | None = None
    hard_hi: float | None = None
    lo_open: bool = False
    hi_open: bool = False
    wall_lo: float | None = None
    wall_hi: float | None = None
    edge: float = 0.05                     # fraction of the validated span that counts as EDGE


def classify(x: float, b: Bounds) -> tuple[Regime, str]:
    """(regime, reason) of `x` against `b`. NaN is BEYOND."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return Regime.BEYOND, "not a number"
    if b.hard_lo is not None and (x < b.hard_lo or (b.lo_open and x == b.hard_lo)):
        return Regime.BEYOND, f"must be {'>' if b.lo_open else '>='} {b.hard_lo:g}"
    if b.hard_hi is not None and (x > b.hard_hi or (b.hi_open and x == b.hard_hi)):
        return Regime.BEYOND, f"must be {'<' if b.hi_open else '<='} {b.hard_hi:g}"
    if b.wall_lo is not None and x < b.wall_lo:
        return Regime.WALL, f"below the model's validated range (>= {b.wall_lo:g})"
    if b.wall_hi is not None and x > b.wall_hi:
        return Regime.WALL, f"above the model's validated range (<= {b.wall_hi:g})"
    lo = b.wall_lo if b.wall_lo is not None else None
    hi = b.wall_hi if b.wall_hi is not None else None
    if lo is not None and hi is not None and hi > lo:
        margin = b.edge * (hi - lo)
        if x - lo < margin or hi - x < margin:
            return Regime.EDGE, "within 5% of the edge of the validated range"
    return Regime.SAFE, ""


def describe(b: Bounds) -> str:
    """The bounds in words, for documentation: '> 0', '>= 0', 'in [0, 1]', '< c'."""
    lo = ("> " if b.lo_open else ">= ") + f"{b.hard_lo:g}" if b.hard_lo is not None else ""
    hi = ("< " if b.hi_open else "<= ") + f"{b.hard_hi:g}" if b.hard_hi is not None else ""
    if lo and hi and not b.lo_open and not b.hi_open:
        return f"in [{b.hard_lo:g}, {b.hard_hi:g}]"
    return ", ".join(x for x in (lo, hi) if x)
