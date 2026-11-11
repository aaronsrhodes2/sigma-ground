"""The matter record (sigma_ground.field.interface.matter_record): one block of matter as any renderer needs it, from sigma-ground's own numbers."""
import json
import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
from sigma_ground.field.interface import matter_record as matter  # noqa: E402

PITCH = 0.05


def test_a_static_record_has_no_clock():
    r = matter.matter_record("gold", 2.4, PITCH)
    assert r["t"] is None and r["v"] == [0.0, 0.0, 0.0]
    assert r["metal"] and r["rgba"][3] == 255 and r["light"] == 0
    assert r["f0"] and len(r["f0"]) == 3                                    # a metal's F0 is its colour


def test_a_moving_block_needs_a_time():
    with pytest.raises(ValueError, match="time"):
        matter.matter_record("gold", 2.4, PITCH, v=(1.0, 0.0, 0.0))
    r = matter.matter_record("gold", 2.4, PITCH, t=1.5, v=(1.0, 0.0, 0.0))
    assert r["t"] == 1.5 and r["v"][0] == 1.0


def test_a_hot_block_glows_and_lights_its_neighbours():
    cold, hot = matter.matter_record("tungsten", 2.4, PITCH, 293.15), matter.matter_record("tungsten", 2.4, PITCH, 2400.0)
    assert cold["light"] == 0 and cold["emission"] == 0.0
    assert 1 <= hot["light"] <= matter.LIGHT_LEVELS and hot["emission"] > 0 and hot["glow_rgb"][0] > hot["glow_rgb"][2]


def test_a_clear_block_carries_the_physics_a_renderer_needs_not_a_tuned_opacity():
    r = matter.matter_record("water", 0.125, PITCH)
    assert r["rgba"][3] < 25 and r["opacity_physical"] == pytest.approx(r["rgba"][3] / 255, abs=0.01)   # its own, physical
    assert 1.3 < r["n"] < 1.4 and len(r["absorption_per_m"]) == 3 and r["absorption_per_m"][0] > r["absorption_per_m"][2]
    assert not any("display exposure" in g for g in r["gaps"]) and "drawn_at_m" not in r
    gold = matter.matter_record("gold", 2.4, PITCH)
    assert gold["n"] is None and gold["absorption_per_m"] is None and gold["tables"] is None   # opaque: nothing passes


def test_the_tables_are_sigma_grounds_optics_sampled_not_the_renderers_physics():
    from sigma_ground.field.interface import optics
    r = matter.matter_record("glass", None, 0.0125)
    t = r["tables"]
    n = r["n"]
    k = 17                                                                           # a sample in the middle
    R, cos_t = optics.fresnel_interface(1.0, n, t["in"]["cos_i"][k])
    assert t["in"]["reflectance"][k] == pytest.approx(R, abs=1e-6) and t["in"]["cos_t"][k] == pytest.approx(cos_t, abs=1e-6)
    R2, cos_o = optics.fresnel_interface(n, 1.0, t["out"]["cos_i"][k])
    assert t["out"]["reflectance"][k] == pytest.approx(R2, abs=1e-6)
    assert t["transmittance"]["step_m"] == 0.0125 and len(t["transmittance"]["rgb"][0]) == matter.DEPTH_CELLS + 1
    assert t["transmittance"]["rgb"][1][10] == pytest.approx(optics.beer_lambert_transmittance(r["absorption_per_m"][1], 10 * 0.0125), abs=1e-6)


def test_the_wire_form_is_json_the_mod_reads():
    r = json.loads(matter.to_wire(matter.matter_record("wood_oak", 0.1, PITCH)))
    assert r["schema"] == matter.SCHEMA and r["material"] == "wood_oak" and len(r["rgba"]) == 4


def test_the_glow_brightness_is_the_one_tonemap_in_thermal_emission_not_a_renderers_constant():
    from sigma_ground.field.interface import block, thermal_emission as te
    r = matter.matter_record("tungsten", None, PITCH, 2400.0)
    g = block.glow("tungsten", 2400.0)
    assert r["emission"] == pytest.approx(te.glow_level(g["power_W_m2"]), abs=1e-4) and r["light"] == round(matter.LIGHT_LEVELS * g["level"])
    assert any("tonemap" in x for x in r["gaps"])                                      # said, not hidden
    warm = matter.matter_record("tungsten", None, PITCH, 900.0)
    assert 0 < warm["emission"] < r["emission"]                                        # dimmer when cooler
