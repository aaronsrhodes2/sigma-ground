"""Regression tests for the three 'honesty bug' fixes (the rock→anvil incident).

A user typed "a rock sitting in trough of running water … splash"; Mentat
force-fit it to drop_object, silently substituted a DEFAULT anvil, set a 0 m
drop, reported a terminal velocity, and the gate stamped it PASS. These lock in:

  A — no silent 'anvil' default: an objectless drop REFUSES.
  B — scenario coherence: a 0 m 'drop' refuses (it never reaches terminal velocity).
  C — out-of-vocabulary actions (static / fluid) are refused, not force-fit.

All deterministic or monkeypatched — no ollama needed.
"""
from sigma_ground.materia import translate
from sigma_ground.materia import scenarios
from sigma_ground.materia import translator as t


# ── A — no silent default ───────────────────────────────────────────────
def test_drop_object_refuses_without_object():
    for o in (None, "", "   "):
        r = scenarios.drop_object(o)
        assert r.validation["passed"] is False and not r.outputs


# ── B — scenario coherence (a 0 m 'drop' doesn't fall) ──────────────────
def test_drop_object_refuses_nonpositive_altitude():
    for alt in (0.0, -5.0):
        r = scenarios.drop_object("anvil", drop_altitude_m=alt)
        assert r.validation["passed"] is False and not r.outputs
    # a real drop from a real height still runs (Deckard grounds the anvil)
    assert scenarios.drop_object("anvil", drop_altitude_m=10.0).validation["passed"] is True


# ── C — refuse fluid/static actions; keep legit named drops ─────────────
def test_fluid_static_actions_decline():
    for q in ("a rock sitting in a trough of running water",
              "an anvil submerged in flowing water, what's the splash",
              "a rock sitting in trough of running water to see what the splash looks like"):
        assert not translate(q, use_qwen=False).is_runnable()


def test_legit_named_drops_still_route():
    for q in ("drop an anvil from 10 km", "drop a brick from 10 km",
              "drop a 4 inch long feather from 8 ft"):
        steps = translate(q, use_qwen=False).steps
        assert [s.verb for s in steps] == ["drop_object"]
        assert steps[0].params.get("object_name")     # a real object was extracted


def test_qwen_drop_force_fit_onto_fluid_declines(monkeypatch):
    # qwen tries to force a drop onto a static/fluid scene → must decline (clarify).
    monkeypatch.setattr(t, "_ollama_plan",
        lambda q, **k: '{"steps":[{"verb":"drop_object","params":{"drop_altitude_m":0}}]}')
    monkeypatch.setattr(t, "_ollama_chat",
        lambda q, **k: '{"verb":"drop_object","params":{"drop_altitude_m":0}}')
    spec = translate("a rock sitting in running water, show me the splash", use_qwen=True)
    assert spec.source == "clarify" and not spec.is_runnable()


# ── front door — name the user's word instead of dropping an anvil ──────
def test_front_door_objectless_drop_asks_for_object(monkeypatch):
    from sigma_ground.mcp import front_door as fd
    monkeypatch.setattr(t, "_ollama_plan",
        lambda q, **k: '{"steps":[{"verb":"drop_object","params":{"drop_altitude_m":10000}}]}')
    monkeypatch.setattr(t, "_ollama_chat",
        lambda q, **k: '{"verb":"drop_object","params":{"drop_altitude_m":10000}}')
    monkeypatch.setattr(t, "_route_consistent", lambda *a, **k: True)
    e = fd.dispatch("drop it from 10 km", use_llm=True, session=fd.Session())
    assert not e["can_render"]                        # no phantom render offered
    assert e["intent"] != "render"
    assert "object" in (e["text"] or "").lower()      # asks for an object to ground
