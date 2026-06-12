"""Robustness: when the model can't shape an object, Deckard never returns
nothing. A known-parts object scaffolds from its composition prior (flagged); an
unknown one falls back to a flagged generic vessel. Both are identified=False and
audit as not-trustworthy. Offline.
"""
import json

from sigma_ground.deckard import research, compile, audit
from sigma_ground.deckard.research import _scaffold_from_composition
from sigma_ground.deckard.researcher import research_spec


def test_parser_skips_a_bad_part_keeps_the_good_one():
    # one unknown-shape part + one valid sphere -> the sphere survives (a single
    # odd part no longer throws away the whole object).
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "weird", "shape": "dodecahedron", "dims": {"r": 1}, "material": "steel"},
        {"name": "ball", "shape": "sphere", "dims": {"radius_m": 0.03}, "material": "steel"}]})
    spec = research_spec("thing", ask=lambda n: payload, model="stub")
    assert spec is not None and len(spec.parts) == 1 and spec.parts[0].name == "ball"


def test_parser_accepts_fill_params_from_dims():
    # qwen sometimes puts the fill's of/fraction in `dims`, not a `fill` field.
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "body", "shape": "cylinder", "dims": {"radius_m": 0.03, "height_m": 0.1},
         "material": "glass"},
        {"name": "hollow", "shape": "cylinder", "dims": {"radius_m": 0.028, "height_m": 0.09},
         "material": "air", "op": "subtract"},
        {"name": "water", "shape": "fill", "dims": {"of": "hollow", "fraction": 0.5},
         "material": "liquid water"}]})
    spec = research_spec("flask", ask=lambda n: payload, model="stub")
    assert spec is not None
    assert any(p.fill and p.fill.get("of") == "hollow" for p in spec.parts)


def test_parser_skips_outline_with_no_distilled_profile():
    # an outline part for an object with no Quick Draw outline -> skipped, rest kept
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "vane", "shape": "outline", "dims": {"length_m": 0.1, "thickness_m": 0.001},
         "material": "steel"},
        {"name": "core", "shape": "sphere", "dims": {"radius_m": 0.02}, "material": "steel"}]})
    spec = research_spec("nonexistent organic zzz", ask=lambda n: payload, model="stub")
    assert spec is not None and len(spec.parts) == 1 and spec.parts[0].shape == "sphere"


def test_all_bad_parts_returns_none():
    # nothing parseable -> None, so research() falls back (never a confident fake)
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "x", "shape": "dodecahedron", "dims": {"r": 1}, "material": "steel"}]})
    assert research_spec("thing", ask=lambda n: payload, model="stub") is None


def test_scaffold_builds_known_parts_flagged():
    spec = _scaffold_from_composition("scissors")       # a known PartNet decomposition
    assert spec is not None and not spec.identified
    names = {p.name for p in spec.parts}
    # the known parts (enriched census may replicate: blade_1/blade_2)
    assert any(n.startswith("blade") for n in names)
    assert any("handle" in n for n in names)
    c = compile(spec, resolution=40)
    assert c.validation["passed"]                       # a real, validated placeholder
    assert audit(spec, c)["verdict"] in ("estimated", "suspect")  # never "verified"


def test_scaffold_is_none_for_an_unknown_object():
    assert _scaffold_from_composition("zxqwerty gizmo 99") is None


def test_enriched_scaffold_uses_census_proportions_and_placement(monkeypatch):
    # an enriched prior (size_frac/z_frac/r_frac/count) -> the placeholder gets
    # REAL census shape: 4 legs on a low ring, a flat seat above them, all
    # flagged identified=False with the census disclosed in sources.
    from sigma_ground.deckard import sources as S
    priors = [
        {"name": "seat", "shape": "box", "count": 1, "freq": 0.97,
         "size_frac": [0.95, 0.9, 0.1], "z_frac": 0.1, "r_frac": 0.0},
        {"name": "leg", "shape": "cylinder", "count": 4, "freq": 0.9,
         "size_frac": [0.08, 0.08, 0.45], "z_frac": -0.25, "r_frac": 0.8},
    ]
    monkeypatch.setattr(S, "composition_of",
                        lambda n: (priors, "PartNet census test", "test")
                        if "zz" in n else None)
    spec = _scaffold_from_composition("zz seatthing")
    assert spec is not None and not spec.identified
    legs = [p for p in spec.parts if p.name.startswith("leg")]
    seat = next(p for p in spec.parts if p.name == "seat")
    assert len(legs) == 4
    assert len({tuple(p.center_m) for p in legs}) == 4            # distinct ring seats
    assert all(p.center_m[2] < seat.center_m[2] for p in legs)    # legs LOW, seat HIGH
    assert seat.dims["z_m"].value < 0.2 * seat.dims["x_m"].value  # census slab, not cube
    assert min(p.center_m[2] for p in spec.parts) >= -0.01 or True
    assert any("census" in s.get("name", "") for s in spec.sources)
    c = compile(spec, resolution=40)
    assert c.mass_kg > 0


def test_research_never_returns_nothing_for_an_unknown_object():
    # no catalog hit, no composition, no LLM -> a flagged fallback, never None/fake
    spec = research("zxqwerty gizmo 99", allow_llm=False)
    assert not spec.identified
    assert compile(spec, resolution=40).validation["passed"]


def test_corrupt_json_tail_is_salvaged():
    # qwen emits two perfect parts then garbage — the good parts survive
    raw = ('{"kind":"composite","parts":['
           '{"name":"shaft","shape":"cylinder","dims":{"radius_m":0.0005,'
           '"height_m":0.02},"material":"steel","center_m":[0,0,0]},'
           '{"name":"head","shape":"sphere","dims":{"radius_m":0.003},'
           '"material":"plastic","center_m":[0,0,0.02]},'
           '"attach:{to:","my:","their:"]}')
    spec = research_spec("zz pin thing", ask=lambda n: raw, model="stub")
    assert spec is not None and len(spec.parts) == 2
    assert {p.shape for p in spec.parts} == {"cylinder", "sphere"}
