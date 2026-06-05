"""Materia → Deckard: the shape seam.

Materia moves matter, but it does not carry a library of arbitrary shapes. When
a simulation needs a specific OBJECT (a steel anvil, a coffee cup, a piano) —
not just a generic sphere — Materia asks the **Deckard shape researcher** to
compile that object into a Construct (real mass, size, centre-of-mass, inertia),
and simulates THAT. This is the intended Deckard → Materia contract: Deckard
(tier 2) compiles a name into a Construct; Materia (tier 3) consumes it to move
it.

Infallibility: Deckard reports `identified` — True when it grounded the shape
(its catalogue or a researched spec) and False when it fell back to a default.
If Deckard can't ground the shape, this returns None and the verb REFUSES — we
never fake a sphere for a real object we couldn't research.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShapeProps:
    """The physical handful Materia needs to MOVE a named object."""
    name: str
    mass_kg: float
    cross_section_m2: float        # frontal/footprint area, for drag
    char_length_m: float           # largest dimension
    inertia: object                # inertia tensor (kg·m²) from Deckard
    source: str                    # provenance: catalogue / researched-by
    grounded: bool                 # Deckard.identified — True = real, not a default


# Cache researched shapes — the LLM research path is ~60 s the first time; a
# repeat ask for the same object should be instant.
_CACHE: dict[str, ShapeProps | None] = {}


def request_shape(name: str, research: bool = True) -> ShapeProps | None:
    """Ask Deckard to identify/compile the named object → its physical
    properties, or None if Deckard couldn't GROUND the shape (→ the verb
    refuses). Tries the fast catalogue first; if `research`, falls back to the
    LLM researcher for novel objects (slow, ~60 s, then cached).
    """
    key = (name or "").strip().lower()
    if not key:
        return None
    if key in _CACHE:
        return _CACHE[key]

    result: ShapeProps | None = None
    try:
        from sigma_ground import deckard
        for allow_llm in ((False, True) if research else (False,)):
            try:
                c = deckard.identify(name, allow_llm=allow_llm)
            except Exception:
                c = None
            if (c is not None and getattr(c, "identified", False)
                    and getattr(c, "mass_kg", None)):
                (x0, x1), (y0, y1), (z0, z1) = c.bbox
                dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)
                result = ShapeProps(
                    name=c.name,
                    mass_kg=float(c.mass_kg),
                    cross_section_m2=max(dx * dy, 1e-9),   # footprint
                    char_length_m=max(dx, dy, dz, 1e-9),
                    inertia=getattr(c, "inertia_kgm2", None),
                    source=str(getattr(c, "source", "deckard")),
                    grounded=True)
                break
    except Exception:
        result = None

    _CACHE[key] = result
    return result
