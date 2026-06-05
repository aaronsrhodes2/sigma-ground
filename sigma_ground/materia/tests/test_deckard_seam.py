"""Materia↔Deckard shape seam: a drop of a NAMED object asks the Deckard
researcher for its real shape; if Deckard can't ground the shape the sim
REFUSES (never fakes a sphere). Spheres keep the fast analytic path."""
from sigma_ground.materia.translator import translate
from sigma_ground.materia import scenarios
from sigma_ground.materia import deckard_bridge


def _verbs(q):
    return [s.verb for s in translate(q, use_qwen=False).steps]


def test_named_object_routes_to_deckard_path():
    assert _verbs("how fast does a steel anvil hit the ground from 10 km") \
        == ["drop_object"]
    assert _verbs("drop a piano off a cliff") == ["drop_object"]


def test_sphere_keeps_fast_analytic_path():
    # a generic sphere is NOT diverted to Deckard
    assert "drop_object" not in _verbs(
        "how fast does a 5 cm steel ball hit from 10 km")


def test_object_name_is_extracted():
    s = translate("drop a heavy anvil from a tower", use_qwen=False).steps[0]
    assert s.verb == "drop_object" and s.params.get("object_name") == "anvil"


def test_refuses_when_shape_ungrounded(monkeypatch):
    # Deckard couldn't ground it → refuse, do NOT fake a sphere
    monkeypatch.setattr(deckard_bridge, "request_shape", lambda name, **k: None)
    r = scenarios.drop_object("flibbertigibbet")
    assert not r.validation["passed"]
    assert not r.outputs


def test_simulates_when_shape_grounded(monkeypatch):
    sh = deckard_bridge.ShapeProps("test-block", 100.0, 0.1, 0.5, None,
                                   "test-source", True)
    monkeypatch.setattr(deckard_bridge, "request_shape", lambda name, **k: sh)
    r = scenarios.drop_object("test-block")
    assert r.validation["passed"]
    assert r.outputs["mass_kg"] == 100.0
    assert r.outputs["terminal_velocity_m_s"] > 0


def test_render_handle_carries_grounded_facts(monkeypatch):
    """drop_object stays a pure-physics scalar, but it ALSO emits a render-handle
    — the grounded facts the tier-4 dispatcher needs to render the real fall via
    radiance.record_object_fall on a "yes", without re-researching the shape."""
    sh = deckard_bridge.ShapeProps("feather", 0.01, 0.003, 0.12, None,
                                   "catalogue", True)
    monkeypatch.setattr(deckard_bridge, "request_shape", lambda name, **k: sh)
    r = scenarios.drop_object("feather", drop_altitude_m=2.4384)
    assert r.outputs["can_render"] is True
    h = r.outputs["render_handle"]
    assert h["object_name"] == "feather"
    assert h["start_altitude_m"] == 2.4384      # the real 8 ft, not a default
    assert h["mass_kg"] == 0.01
    assert h["cross_section_m2"] == 0.003
    assert h["char_length_m"] == 0.12
    assert h["cd"] > 0
