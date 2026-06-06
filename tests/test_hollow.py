"""Hollowness prior: appliances/containers/furniture/electronics are mostly air,
not solid. A bulky part of a hollow-class object gets a thin interior carve so its
mass is a shell, not a block. Solid-class objects are left solid. Offline.
"""
import json

from sigma_ground.deckard import compile
from sigma_ground.deckard.researcher import research_spec, _is_hollow


def test_is_hollow_classifier():
    assert _is_hollow("microwave oven") and _is_hollow("a cardboard box")
    assert not _is_hollow("steel ball") and not _is_hollow("hammer")
    assert not _is_hollow("zxqwerty thing")           # unknown -> default solid (conservative)


def _has_hollow_source(spec):
    return any("hollow" in (s.get("name", "") if isinstance(s, dict) else str(s)).lower()
               for s in spec.sources)


def test_hollow_class_object_is_modeled_as_a_shell_not_solid():
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "body", "shape": "box", "dims": {"x_m": 0.45, "y_m": 0.3, "z_m": 0.3},
         "material": "steel"}]})
    spec = research_spec("microwave", ask=lambda n: payload, model="stub")
    body = spec.parts[0]
    assert body.density.estimated                            # bulk density is a heuristic -> flagged
    assert body.density.value < 0.2 * 7850                   # far below solid steel -> a shell
    assert body.material == "steel"                          # material name preserved
    assert _has_hollow_source(spec)                          # disclosed in the spec's sources
    c = compile(spec, resolution=64)
    solid = 0.45 * 0.3 * 0.3 * 7850                          # ~318 kg if solid
    assert c.mass_kg < 0.15 * solid                         # a shell's mass, far lighter
    assert c.validation["passed"]                           # solid body at bulk density integrates cleanly


def test_solid_class_object_is_not_hollowed():
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "head", "shape": "box", "dims": {"x_m": 0.1, "y_m": 0.04, "z_m": 0.04},
         "material": "steel"}]})
    spec = research_spec("hammer", ask=lambda n: payload, model="stub")
    assert not _has_hollow_source(spec)                      # solid class -> not hollowed
    assert spec.parts[0].density.value > 1000               # real steel density, not a bulk estimate
