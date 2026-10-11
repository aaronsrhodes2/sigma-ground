"""researcher._hollow_pass leaves board / sheet / rod stock solid.

The LLM-researched bookcase (the frozen catalog entry that was retired) was eight 2.9 cm
oak planks; the hollow pass treated each plank as a 0.8 mm-walled enclosure and set the frame and
back panel to 44 kg/m3 (6% of oak), so a bookcase weighed 4 kg. A plank is stock, not a box.
Offline, no compile.
"""
import json

from sigma_ground.deckard.researcher import _MIN_ENCLOSURE_M, _hollow_pass, research_spec
from sigma_ground.deckard.schema import Fact, Part

_OAK = Fact(700.0, "surface.MATERIALS", "", 0.9)
_STEEL = Fact(7850.0, "surface.MATERIALS", "", 0.9)


def _oak_box(name, x, y, z):
    return Part(name, "box", {"x_m": Fact(x), "y_m": Fact(y), "z_m": Fact(z)}, "oak", _OAK)


def test_the_retired_bookcase_planks_stay_solid():
    parts = [_oak_box("frame", 0.343, 0.686, 0.0286), _oak_box("back_panel", 0.294, 0.588, 0.0286),
             _oak_box("shelf_1", 0.294, 0.0735, 0.0286)]
    notes = []
    _hollow_pass("a bookcase", parts, notes)
    for p in parts:
        assert p.density.value == 700.0 and not p.density.estimated        # was 43.8 [estimated]
    assert notes == []                                                      # nothing was hollowed, so nothing to disclose


def test_a_thick_box_of_a_hollow_class_is_still_a_shell():
    body = Part("body", "box", {"x_m": Fact(0.45), "y_m": Fact(0.3), "z_m": Fact(0.3)}, "steel", _STEEL)
    notes = []
    _hollow_pass("microwave", [body], notes)
    assert body.density.estimated and body.density.value < 0.1 * 7850.0
    assert notes and "modeled hollow" in notes[0]["name"]


_GLASS = Fact(2500.0, "surface.MATERIALS", "", 0.9)


def _box(name, x, y, z, material, rho):
    return Part(name, "box", {"x_m": Fact(x), "y_m": Fact(y), "z_m": Fact(z)}, material, rho)


def test_the_box_guard_is_per_material_not_a_flat_4_cm():
    """Stock gauge is the material's: some edge <= 2 x research._stock_thickness(material)."""
    # a 3.8 cm steel door (steel stock is 1.5 mm) and a 3 cm glass panel are shells, not solid slabs
    steel_door = _box("door", 0.5, 0.4, 0.038, "steel", _STEEL)
    glass = _box("panel", 0.6, 0.4, 0.03, "glass", _GLASS)
    _hollow_pass("a microwave", [steel_door], [])
    _hollow_pass("a microwave", [glass], [])
    assert steel_door.density.estimated and steel_door.density.value < 0.1 * 7850.0
    assert glass.density.estimated and glass.density.value < 0.2 * 2500.0
    # ... while board stock stays solid: 2.9 cm oak ("oak" carries no 'wood' in its name) and 3.4 cm plywood
    oak = _box("shelf", 0.9, 0.3, 0.0286, "oak", _OAK)
    ply = _box("side", 0.9, 0.5, 0.034, "plywood", _OAK)
    _hollow_pass("a microwave", [oak], [])
    _hollow_pass("a microwave", [ply], [])
    assert not oak.density.estimated and not ply.density.estimated
    # sheet at its own gauge (3 mm steel) is solid stock
    sheet = _box("sheet", 0.5, 0.4, 0.003, "steel", _STEEL)
    _hollow_pass("a microwave", [sheet, _box("body", 0.5, 0.5, 0.5, "steel", _STEEL)], [])
    assert not sheet.density.estimated and sheet.density.value == 7850.0
    # a steel box thicker than 2 gauges, at any size, is a shell
    thick = _box("body", 0.5, 0.4, 0.05, "steel", _STEEL)
    _hollow_pass("a microwave", [thick], [])
    assert thick.density.estimated and thick.density.value < 7850.0
    # round parts keep the absolute 4 cm floor: a slim rod / knob is not an enclosure
    rod = Part("handle", "cylinder", {"radius_m": Fact(0.015), "height_m": Fact(0.6)}, "steel", _STEEL)
    ball = Part("knob", "sphere", {"radius_m": Fact(0.015)}, "steel", _STEEL)
    _hollow_pass("a microwave", [rod], [])
    _hollow_pass("a microwave", [ball], [])
    assert not rod.density.estimated and not ball.density.estimated
    assert _MIN_ENCLOSURE_M == 0.04


def test_through_research_spec_a_thin_steel_door_is_a_shell():
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "body", "shape": "box", "dims": {"x_m": 0.3, "y_m": 0.3, "z_m": 0.3}, "material": "steel"},
        {"name": "door", "shape": "box", "dims": {"x_m": 0.6, "y_m": 0.4, "z_m": 0.038}, "material": "steel",
         "center_m": [0.0, 0.2, 0.0]}]})
    spec = research_spec("microwave", ask=lambda n: payload, model="stub")
    assert spec is not None
    door = next(p for p in spec.parts if p.name == "door")
    assert door.density.estimated and door.density.value < 0.1 * 7850.0


def test_through_research_spec_a_slab_bookcase_keeps_cited_oak():
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "back_panel", "shape": "box", "dims": {"x_m": 0.9, "y_m": 1.2, "z_m": 0.02}, "material": "oak"},
        {"name": "side", "shape": "box", "dims": {"x_m": 0.02, "y_m": 0.3, "z_m": 1.2}, "material": "oak",
         "center_m": [0.46, 0.0, 0.0]}]})
    spec = research_spec("bookcase", ask=lambda n: payload, model="stub")
    assert spec is not None
    assert all(not p.density.estimated for p in spec.parts)
    assert not any("modeled hollow" in s.get("name", "") for s in spec.sources)
