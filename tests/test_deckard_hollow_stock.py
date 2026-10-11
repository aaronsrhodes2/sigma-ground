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


def test_the_guard_is_on_the_thinnest_edge():
    thin = Part("door", "box", {"x_m": Fact(0.5), "y_m": Fact(0.4), "z_m": Fact(_MIN_ENCLOSURE_M * 0.95)}, "steel", _STEEL)
    thick = Part("body", "box", {"x_m": Fact(0.5), "y_m": Fact(0.4), "z_m": Fact(_MIN_ENCLOSURE_M * 1.05)}, "steel", _STEEL)
    _hollow_pass("a microwave", [thin], [])
    _hollow_pass("a microwave", [thick], [])
    assert not thin.density.estimated and thin.density.value == 7850.0
    assert thick.density.estimated and thick.density.value < 7850.0
    # round parts: a slim rod is not an enclosure either
    rod = Part("handle", "cylinder", {"radius_m": Fact(0.015), "height_m": Fact(0.6)}, "steel", _STEEL)
    ball = Part("knob", "sphere", {"radius_m": Fact(0.015)}, "steel", _STEEL)
    _hollow_pass("a microwave", [rod], [])
    _hollow_pass("a microwave", [ball], [])
    assert not rod.density.estimated and not ball.density.estimated


def test_through_research_spec_a_slab_bookcase_keeps_cited_oak():
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "back_panel", "shape": "box", "dims": {"x_m": 0.9, "y_m": 1.2, "z_m": 0.02}, "material": "oak"},
        {"name": "side", "shape": "box", "dims": {"x_m": 0.02, "y_m": 0.3, "z_m": 1.2}, "material": "oak",
         "center_m": [0.46, 0.0, 0.0]}]})
    spec = research_spec("bookcase", ask=lambda n: payload, model="stub")
    assert spec is not None
    assert all(not p.density.estimated for p in spec.parts)
    assert not any("modeled hollow" in s.get("name", "") for s in spec.sources)
