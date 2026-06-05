"""Deckard grounds DIMENSIONS, not just densities: a standard object gets its
real, cited size; the model's estimate is replaced; a hallucinated giant is
rejected. Offline — local table + stubs, no real network.
"""
import json

from sigma_ground.deckard.sources import dimensions_of, dimensions_api
from sigma_ground.deckard.researcher import research_spec


def test_local_table_grounds_a_standard_object_exactly():
    d = dimensions_of("a tennis ball", "sphere")          # word-match + shape agree
    assert d is not None and "radius_m" in d
    assert abs(d["radius_m"].value - 0.0335) < 1e-9
    assert not d["radius_m"].estimated                     # cited, not a guess
    assert "ITF" in d["radius_m"].source


def test_a4_paper_box_dims_are_iso216():
    d = dimensions_of("A4", "box")
    assert d is not None
    assert abs(d["x_m"].value - 0.210) < 1e-9 and abs(d["y_m"].value - 0.297) < 1e-9
    assert "ISO 216" in d["x_m"].source


def test_shape_must_agree_and_unknown_returns_none():
    assert dimensions_of("tennis ball", "box") is None     # right object, wrong shape
    assert dimensions_of("nonexistent gizmo", "sphere") is None


def test_researcher_replaces_estimate_with_cited_dimension():
    # the model proposes a sphere with a WRONG radius; Deckard grounds it to the
    # real tennis-ball radius and cites the source (no network: stubbed `ask`).
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "ball", "shape": "sphere", "dims": {"radius_m": 0.05},
         "material": "steel"}]})
    spec = research_spec("tennis ball", ask=lambda n: payload, model="stub")
    r = spec.parts[0].dims["radius_m"]
    assert abs(r.value - 0.0335) < 1e-9 and not r.estimated   # grounded, not the 0.05 guess
    assert any("ITF" in s.get("name", "") for s in spec.sources)


def test_human_scale_clamp_rejects_a_giant():
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "blob", "shape": "sphere", "dims": {"radius_m": 500.0},
         "material": "steel"}]})
    assert research_spec("giant", ask=lambda n: payload, model="stub") is None


def test_wikidata_dimensions_map_diameter_to_radius(monkeypatch):
    # stub the two Wikidata calls; a sphere's diameter (mm) -> radius_m, CC0.
    def fake_get_json(url, timeout=10.0):
        if "wbsearchentities" in url:
            return {"search": [{"id": "Q42"}]}
        return {"claims": {"P2386": [{"mainsnak": {"datavalue": {"value": {
            "amount": "+200", "unit": "http://www.wikidata.org/entity/Q174789"}}}}]}}
    monkeypatch.setattr(dimensions_api.web, "get_json", fake_get_json)
    d = dimensions_api.wikidata_dimensions("mystery orb", "sphere")
    assert d is not None and abs(d["radius_m"].value - 0.1) < 1e-9   # 200 mm dia -> 0.1 m
    assert d["radius_m"].license == "CC0"
