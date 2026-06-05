"""Deckard's grounding sources — facts the researcher cites.

  local         : the project's own curated data (offline, always available).
  web           : cached HTTP JSON fetch (stdlib only).
  materials_api : free factual APIs (Wikidata density, CC0).
  wikipedia     : a cached free-text summary (object proportions + composition).

``density_of`` is local-first; with ``allow_web=True`` it falls back to Wikidata,
so a material outside our own data is still *cited* (entity QID + license) rather
than silently estimated. A source returns cited Facts (value + source + license +
confidence) so the researcher attributes every density, never guesses.
"""
from __future__ import annotations

from . import local, web, materials_api, wikipedia


def density_of(material: str, *, allow_web: bool = False):
    """A cited density Fact (kg/m³): our data first, then Wikidata if allow_web."""
    f = local.density_of(material)
    if f is not None:
        return f
    if allow_web:
        return materials_api.wikidata_density(material)
    return None


def wikipedia_summary(name: str):
    """A free-text Wikipedia summary of an object (typical proportions/what it is
    made of), or None — grounds the researcher's shape proposal in web text, not
    model recall. Cached; degrades to None offline."""
    return wikipedia.summary(name)


__all__ = ["local", "web", "materials_api", "wikipedia", "density_of",
           "wikipedia_summary"]
