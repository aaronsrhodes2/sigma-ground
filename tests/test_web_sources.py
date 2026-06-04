"""Deckard web grounding — Wikidata density (mocked, no network).

Verifies the fetch+parse, unit conversion to kg/m³, cited CC0 provenance,
local-first precedence (no web call when our own data has it), unknown-unit
refusal, and graceful None offline. ``web.get_json`` is monkeypatched, so
nothing here touches the network.
"""
from sigma_ground.deckard.sources import materials_api, web, local, density_of


def _fake(qid="Q897", amount="+8.73", unit="Q13147228"):
    search = {"search": [{"id": qid}]}
    claims = {"claims": {"P2054": [
        {"mainsnak": {"datavalue": {"value": {
            "amount": amount,
            "unit": f"http://www.wikidata.org/entity/{unit}"}}}}]}}

    def get_json(url, timeout=10):
        if "wbsearchentities" in url:
            return search
        if "wbgetclaims" in url:
            return claims
        return None
    return get_json


def test_wikidata_density_parses_and_cites(monkeypatch):
    monkeypatch.setattr(web, "get_json", _fake(qid="Q897", amount="+8.73"))
    f = materials_api.wikidata_density("inconel")
    assert f is not None
    assert abs(f.value - 8730.0) < 1e-6                 # 8.73 g/cm³ -> 8730 kg/m³
    assert not f.estimated and f.license == "CC0" and "Q897" in f.source


def test_wikidata_unit_kg_per_m3(monkeypatch):
    monkeypatch.setattr(web, "get_json", _fake(amount="+998", unit="Q844211"))
    assert abs(materials_api.wikidata_density("x").value - 998.0) < 1e-6


def test_wikidata_unknown_unit_is_refused(monkeypatch):
    monkeypatch.setattr(web, "get_json", _fake(unit="Q99999999"))
    assert materials_api.wikidata_density("x") is None   # never unit-guess


def test_wikidata_offline_returns_none(monkeypatch):
    monkeypatch.setattr(web, "get_json", lambda url, timeout=10: None)
    assert materials_api.wikidata_density("inconel") is None


def test_density_of_is_local_first_no_web(monkeypatch):
    calls = {"n": 0}

    def boom(url, timeout=10):
        calls["n"] += 1
        return None
    monkeypatch.setattr(web, "get_json", boom)
    f = density_of("iron", allow_web=True)
    assert f is not None and "surface.MATERIALS" in f.source
    assert calls["n"] == 0                               # local short-circuits the web


def test_density_of_web_fallback_for_unknown(monkeypatch):
    assert local.density_of("inconel") is None           # not in our data
    monkeypatch.setattr(web, "get_json", _fake(qid="Q897", amount="+8.73"))
    f = density_of("inconel", allow_web=True)
    assert f is not None and abs(f.value - 8730.0) < 1e-6 and f.license == "CC0"
    assert density_of("inconel", allow_web=False) is None  # web off -> None
