"""Plausibility-scaling: a fully-estimated construct is uniformly scaled to a
grounded typical OVERALL size — the model gets proportions right but absolute
size wrong. Grounded or unknown objects are left alone. Offline (stub + local
size table). (Scaling fixes SIZE; mass also depends on solid-vs-hollow modelling,
which the audit flags separately.)
"""
import json

from sigma_ground.deckard.researcher import research_spec
from sigma_ground.deckard.sources import typical_size_of


def _spec(name, payload):
    return research_spec(name, ask=lambda n: payload, model="stub")


def test_typical_size_table_resolves_common_objects():
    got = typical_size_of("a toaster")
    assert got is not None and abs(got[0] - 0.30) < 1e-9


def test_grossly_oversized_estimate_is_scaled_to_typical_size():
    # the model makes a 2 m "toaster"; scale its longest extent to the ~0.30 m typical.
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "body", "shape": "box", "dims": {"x_m": 2.0, "y_m": 1.0, "z_m": 1.0},
         "material": "plastic"}]})
    spec = _spec("toaster", payload)
    ext = max(spec.parts[0].dims[k].value for k in ("x_m", "y_m", "z_m"))
    assert abs(ext - 0.30) < 1e-6                       # longest extent now the typical size
    assert not spec.parts[0].dims["x_m"].estimated      # re-provenanced: size-anchored, not a guess
    # proportions preserved (2:1:1)
    d = spec.parts[0].dims
    assert abs(d["x_m"].value / d["y_m"].value - 2.0) < 1e-6


def test_grounded_object_is_not_rescaled():
    # a tennis ball's radius is grounded -> trust it, don't rescale
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "ball", "shape": "sphere", "dims": {"radius_m": 0.05}, "material": "steel"}]})
    spec = _spec("tennis ball", payload)
    assert abs(spec.parts[0].dims["radius_m"].value - 0.0335) < 1e-9   # grounded value, untouched


def test_already_reasonable_size_is_left_alone():
    # a 0.28 m "toaster" is within range of the 0.30 m typical -> no rescale
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "body", "shape": "box", "dims": {"x_m": 0.28, "y_m": 0.18, "z_m": 0.18},
         "material": "plastic"}]})
    spec = _spec("toaster", payload)
    assert abs(spec.parts[0].dims["x_m"].value - 0.28) < 1e-9


def test_unknown_object_is_not_scaled():
    payload = json.dumps({"kind": "composite", "parts": [
        {"name": "blob", "shape": "box", "dims": {"x_m": 2.0, "y_m": 1.0, "z_m": 1.0},
         "material": "steel"}]})
    spec = _spec("zxqwerty contraption 88", payload)
    assert abs(spec.parts[0].dims["x_m"].value - 2.0) < 1e-9           # no typical size -> left alone
