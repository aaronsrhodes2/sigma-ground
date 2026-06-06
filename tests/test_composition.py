"""Deckard grounds COMPOSITION — which parts a multi-part object has — as a prior
for the researcher's decomposition (the structural analogue of dimension
grounding). Curated today, PartNet-replaceable. Offline (local table + stubs).
"""
import json

from sigma_ground.deckard.sources import composition_of, composition
from sigma_ground.deckard.researcher import research_spec


def test_composition_of_returns_known_parts():
    got = composition_of("a claw hammer")          # word-match -> hammer
    assert got is not None
    parts, source, _lic = got
    names = {p["name"] for p in parts}
    assert {"handle", "head"} <= names
    assert "partnet" in source.lower() or "curated" in source.lower()   # honest provenance


def test_composition_of_counts_and_unknown():
    chair = composition_of("chair")
    legs = next(p for p in chair[0] if p["name"] == "leg")
    assert legs["count"] == 4
    assert composition_of("nonexistent contraption xyz") is None


def test_hint_is_readable():
    h = composition.hint("dumbbell")
    assert h.startswith("Typical parts") and "handle" in h and "weight" in h


def test_researcher_cites_the_composition_for_a_known_multipart_object():
    # the model proposes a hammer's two parts; Deckard cites the documented
    # decomposition alongside (stubbed ask + local materials -> offline).
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "handle", "shape": "cylinder", "dims": {"radius_m": 0.012, "height_m": 0.30},
         "material": "oak", "center_m": [0, 0, 0.15]},
        {"name": "head", "shape": "box", "dims": {"x_m": 0.10, "y_m": 0.03, "z_m": 0.03},
         "material": "steel", "center_m": [0, 0, 0.315]}]})
    spec = research_spec("hammer", ask=lambda n: payload, model="stub")
    assert spec is not None and len(spec.parts) == 2
    assert any("decomposition" in s.get("name", "").lower() or "partnet" in s.get("name", "").lower()
               for s in spec.sources)


def test_single_part_object_is_not_cited_as_composed():
    # one part only -> no composition citation, even if the name has a known one
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "ball", "shape": "sphere", "dims": {"radius_m": 0.03}, "material": "steel"}]})
    spec = research_spec("hammer", ask=lambda n: payload, model="stub")
    assert not any("decomposition" in s.get("name", "").lower() for s in spec.sources)
