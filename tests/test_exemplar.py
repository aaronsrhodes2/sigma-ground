"""PartNet exemplar assembly: Deckard learns an object's SHAPE from a real model.

Manufactured shapes aren't derivable from physics — Deckard reads a representative
real PartNet model's part layout (legs/back/seat where they actually are) and
scales it to the ShapeNetSem median size. The assembly is data-driven, never a
hand-written template. Runs against the SHIPPED exemplars, offline.
"""
from sigma_ground.deckard import research, compile
from sigma_ground.deckard.research import _exemplar_spec
from sigma_ground.deckard.sources import exemplar_of


def test_exemplars_are_shipped_and_well_formed():
    # every shipped exemplar is a cited real model with sane normalized parts
    import json
    import pathlib
    from sigma_ground.deckard.sources.exemplar import _DIR
    files = sorted(_DIR.glob("*.json"))
    assert len(files) >= 15                                # the PartNet category set
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d.get("source_anno_id") and "PartNet" in d.get("_source", "")
        assert 2 <= len(d["parts"]) <= 14
        for part in d["parts"]:
            assert part["shape"] in ("box", "cylinder", "sphere")
            assert len(part["center_frac"]) == 3 and len(part["size_frac"]) == 3
            assert all(0 < s <= 1.05 for s in part["size_frac"])   # within the bbox
            assert all(-0.6 <= c <= 0.6 for c in part["center_frac"])


def test_chair_assembles_from_a_real_model_as_a_chair():
    spec = _exemplar_spec("a wooden chair")
    assert spec is not None and spec.identified
    names = " ".join(p.name for p in spec.parts)
    assert "seat" in names and "leg" in names           # real chair parts
    legs = [p for p in spec.parts if p.name.startswith("leg")]
    assert len(legs) >= 3
    # legs reach the floor and sit below the seat
    seat = next(p for p in spec.parts if "seat" in p.name)
    for L in legs:
        assert L.center_m[2] < seat.center_m[2]
    c = compile(spec, resolution=64)
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    assert z0 < 0.02 and (z1 - z0) > 0.5 * (y1 - y0)     # stands on the floor, upright


def test_exemplar_generalizes_beyond_furniture():
    # a mug is body + handle; scissors is blades + handles — real layouts, not legs
    mug = exemplar_of("a coffee mug")
    assert mug is not None
    mug_parts = {p["name"] for p in mug[0]}
    assert any("body" in n for n in mug_parts) and any("handle" in n for n in mug_parts)
    sc = _exemplar_spec("scissors")
    assert sc is not None and sum("blade" in p.name for p in sc.parts) >= 1


def test_dimensions_grounded_material_provided():
    spec = _exemplar_spec("a wooden chair")
    # every dim is an estimate (never a measurement of THIS chair); the overall
    # size cites ShapeNetSem and the material composition is handed on
    assert all(f.estimated for p in spec.parts for f in p.dims.values())
    src = " ".join(s.get("name", "") for s in spec.sources)
    assert "ShapeNetSem" in src and "material composition" in src
    assert "representative real model" in src           # honest: not a template


def test_research_uses_the_exemplar_path():
    spec = research("a wooden chair", allow_llm=False)
    assert spec.identified
    assert any("representative real model" in s.get("name", "") for s in spec.sources)
