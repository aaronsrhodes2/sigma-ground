"""Deckard's grounding sources — facts the researcher cites.

  local : the project's own curated data (offline, always available).
  web   : free factual APIs (Wikidata · PubChem · Materials Project) — Phase 2.

A source returns cited Facts (value + source + license + confidence) so the
researcher grounds LLM-proposed materials in attributable numbers, never guesses.
"""
from __future__ import annotations

from . import local
from .local import density_of

__all__ = ["local", "density_of"]
