"""Deckard per-part material: the SHAPE is the real exemplar's layout; each
part's MATERIAL is assigned by a CITED role-convention (PartNet has no per-part
material). A material adjective biases the frame material — it SELECTS, it does
not invent. Densities resolve through field.interface.resolve.material_profile.
"""
from sigma_ground.deckard.research import (
    _exemplar_spec, _role_material, _hint_to_material,
)


def test_chair_parts_carry_per_part_cited_materials():
    spec = _exemplar_spec("a chair")
    assert spec is not None and spec.parts
    legs = [p for p in spec.parts if p.name.startswith("leg")]
    assert legs, "a chair should have legs"
    for L in legs:
        assert L.material not in ("fabric", "leather", "carpet", "foam")  # legs are structural
        assert L.density.value > 0
    # provenance flags the convention honestly — not claimed as measured
    src = " ".join(s.get("name", "") for s in spec.sources)
    assert "role-convention" in src
    assert "PartNet carries no per-part material" in src


def test_seat_takes_upholstery_when_composition_has_one():
    spec = _exemplar_spec("a chair")
    soft = {"fabric", "leather", "carpet", "foam"}
    seats = [p for p in spec.parts if "seat" in p.name and "support" not in p.name]
    if seats and any(p.material in soft for p in spec.parts):
        assert any(s.material in soft for s in seats), \
            "the sitting surface should take the upholstery material"


def test_material_hint_biases_the_frame_it_selects_not_invents():
    wood = _exemplar_spec("a chair", material_hint="wooden")
    metal = _exemplar_spec("a chair", material_hint="metal")
    leg_w = next(p for p in wood.parts if p.name.startswith("leg"))
    leg_m = next(p for p in metal.parts if p.name.startswith("leg"))
    assert leg_w.material == "wood" and leg_m.material == "metal"
    assert leg_w.material != leg_m.material
    assert any("material_hint" in s.get("name", "") for s in wood.sources)


def test_part_density_resolves_cited_via_material_profile():
    spec = _exemplar_spec("a wooden chair", material_hint="wood")
    leg = next(p for p in spec.parts if p.name.startswith("leg"))
    # wood resolves to a real cited density (oak ~700), not the flagged-700 default
    assert 300 < leg.density.value < 1200
    assert not leg.density.estimated


def test_role_and_hint_helpers():
    assert _hint_to_material("a wooden chair") == "wood"
    assert _hint_to_material("metal") == "metal"
    assert _hint_to_material("stainless") == "stainless_steel"
    assert _hint_to_material(None) is None
    assert _hint_to_material("chartreuse") is None         # not a material word
    assert _role_material("seat_single_surface", "wood", "fabric") == "fabric"
    assert _role_material("seat_support", "wood", "fabric") == "wood"   # structural
    assert _role_material("leg_3", "wood", "fabric") == "wood"
    assert _role_material("back", "wood", None) == "wood"               # no upholstery
