"""Deckard research-layer pipeline tests — deterministic, no network.

Covers markdown↔spec round-trip; catalog cache-hit (offline); the compiler
plausibility gate; the unidentified flag; the stubbed-LLM researcher (density
grounding + ``[estimated]`` flags + bad-output rejection); and the
freeze→cache-hit cycle against an isolated tmp catalog. The LLM ``ask`` hook is
always stubbed, so nothing here touches Gemini or ollama.
"""
import json

import pytest

from sigma_ground.deckard import (research, identify, compile,
                                  ConstructSpec, Fact, SpecLayer,
                                  emit_markdown, parse_markdown)
from sigma_ground.deckard.researcher import research_spec
from sigma_ground.deckard.sources import density_of
import sigma_ground.deckard.catalog as catalog
import sigma_ground.deckard.researcher as researcher


def _glass_json():
    return json.dumps({
        "kind": "layered_vessel",
        "geometry": {"outer_radius_m": 0.032, "height_m": 0.12, "wall_m": 0.003,
                     "glaze_m": 0.0, "base_m": 0.005, "fill_fraction": 0.85},
        "layers": [{"name": "glaze", "material": "glass"},
                   {"name": "ceramic", "material": "glass"},
                   {"name": "water", "material": "liquid water"}],
        "notes": "a drinking glass",
    })


def test_markdown_spec_roundtrip():
    spec = ConstructSpec(
        name="probe", kind="layered_vessel", identified=True,
        geometry={"outer_radius_m": Fact(0.04, "src", "CC0", 0.8)},
        layers=[SpecLayer("ceramic", "stoneware",
                          Fact(2300.0, "materials.json", "", 0.9),
                          Fact(0.005, "src", "", 0.6), ["air"])],
        sources=[{"name": "src", "license": "CC0"}], notes="n",
    )
    back = parse_markdown(emit_markdown(spec))
    assert back.to_payload() == spec.to_payload()
    assert not back.geometry["outer_radius_m"].estimated


def test_catalog_cache_hit_is_offline():
    spec = research("coffee cup", allow_llm=False)
    assert spec.identified and spec.name == "coffee cup"
    assert spec.geometry["outer_radius_m"].value == 0.040


def test_compile_gate_passes_for_cup():
    cup = identify("coffee cup", resolution=56, allow_llm=False)
    assert cup.validation["passed"], cup.validation["note"]
    assert 0.55 < cup.mass_kg < 0.65                       # ~599 g
    assert abs(cup.com_m[0]) < 1e-9 and abs(cup.com_m[1]) < 1e-9


def test_unidentified_is_flagged():
    spec = research("zorblax thunder widget", allow_llm=False)
    assert spec.identified is False
    assert identify("zorblax thunder widget", resolution=20,
                    allow_llm=False).identified is False


def test_researcher_stub_grounds_density_and_flags_estimates():
    spec = research_spec("drinking glass", ask=lambda n: _glass_json(), model="stub")
    assert spec is not None and spec.identified
    # LLM-proposed dimensions are flagged [estimated]
    assert spec.geometry["outer_radius_m"].estimated
    # the fill density is GROUNDED in our own data (cited, not estimated)
    water = next(L for L in spec.layers if L.name == "water")
    assert not water.density.estimated
    assert water.density.value == density_of("liquid water").value
    # and it compiles to validated matter
    cup = compile(spec, resolution=64)
    assert cup.validation["passed"]


@pytest.mark.parametrize("bad", [
    None,
    "no json here",
    json.dumps({"kind": "unknown"}),
    json.dumps({"kind": "layered_vessel", "geometry": {}}),
    json.dumps({"kind": "layered_vessel",
                "geometry": {"outer_radius_m": -1, "height_m": 0.1, "wall_m": 0.01,
                             "glaze_m": 0.0, "base_m": 0.01, "fill_fraction": 0.5}}),
])
def test_researcher_rejects_bad_output(bad):
    assert research_spec("x", ask=lambda n: bad) is None


def test_research_freezes_then_cache_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "_DIR", tmp_path)             # isolate the catalog
    monkeypatch.setattr(researcher, "_ask", lambda n: _glass_json())
    slug = catalog._resolve("drinking glass")
    assert not catalog.has(slug)                              # empty tmp catalog
    spec = research("drinking glass")                          # miss → researcher → freeze
    assert spec.identified and catalog.has(slug)               # frozen to tmp
    again = research("drinking glass")                         # now a cache hit
    assert again.to_payload() == spec.to_payload()


def test_density_of_is_cited():
    f = density_of("liquid water")
    assert f is not None and not f.estimated
    assert "materials.json" in f.source
    assert density_of("unobtanium-quux-9000") is None


def test_researcher_builds_solid_primitive():
    rod = json.dumps({"kind": "composite", "parts": [
        {"name": "rod", "shape": "cylinder",
         "dims": {"radius_m": 0.006, "height_m": 0.25}, "material": "steel"}]})
    spec = research_spec("steel rod", ask=lambda n: rod, model="stub")
    assert spec is not None and spec.kind == "composite" and len(spec.parts) == 1
    p = spec.parts[0]
    # dims are flagged guesses OR census-anchored ("scaled to ShapeNetSem …") —
    # never presented as a measurement of THIS rod
    assert all(f.estimated or "ShapeNetSem" in f.source
               for f in p.dims.values())
    assert not p.density.estimated                       # density grounded in our data
    assert p.density.value == density_of("steel").value
    assert compile(spec, resolution=48).validation["passed"]
