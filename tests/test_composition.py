"""Deckard grounds COMPOSITION — which parts a multi-part object has — as a prior
for the researcher's decomposition (the structural analogue of dimension
grounding). PartNet's PUBLIC part taxonomy (MIT) grounds its 24 categories;
common objects PartNet lacks stay as flagged curated entries. Offline.
"""
import json

from sigma_ground.deckard.sources import composition_of, composition
from sigma_ground.deckard.researcher import research_spec


def test_partnet_grounds_a_category_with_cited_parts():
    got = composition_of("a pair of scissors")          # PartNet category, word-matched
    assert got is not None
    parts, source, _lic = got
    assert {"blade", "handle"} <= {p["name"] for p in parts}
    assert "PartNet" in source                          # cited to the public taxonomy


def test_curated_objects_partnet_lacks_are_still_covered():
    got = composition_of("a claw hammer")               # not a PartNet category
    assert got is not None
    assert {"handle", "head"} <= {p["name"] for p in got[0]}
    assert "curated" in got[1].lower()                  # honestly flagged provenance


def test_composition_counts_and_unknown():
    db = composition_of("dumbbell")
    weight = next(p for p in db[0] if p["name"] == "weight")
    assert weight["count"] == 2
    assert composition_of("nonexistent contraption xyz") is None


def test_hint_is_readable():
    h = composition.hint("dumbbell")
    assert h.startswith("Typical parts") and "handle" in h and "weight" in h


def test_researcher_cites_partnet_for_a_known_multipart_object():
    # the model proposes scissors' two parts; Deckard cites the PartNet taxonomy
    # it anchored on (stubbed ask + local material -> offline).
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "blade", "shape": "box", "dims": {"x_m": 0.08, "y_m": 0.01, "z_m": 0.002},
         "material": "steel"},
        {"name": "handle", "shape": "cylinder", "dims": {"radius_m": 0.006, "height_m": 0.05},
         "material": "steel"}]})
    spec = research_spec("scissors", ask=lambda n: payload, model="stub")
    assert spec is not None and len(spec.parts) == 2
    assert any("PartNet" in s.get("name", "") for s in spec.sources)


def test_single_part_object_is_not_cited_as_composed():
    # one part only -> no composition citation, even for a name with a known one
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "ball", "shape": "sphere", "dims": {"radius_m": 0.03}, "material": "steel"}]})
    spec = research_spec("scissors", ask=lambda n: payload, model="stub")
    assert not any("partnet" in s.get("name", "").lower() for s in spec.sources)
