"""Translator regression tests — natural language routes to the right verb.

All deterministic (use_qwen=False), so they need no ollama. They pin the
babble→spec mapping: intent classification, slot extraction, multi-verb specs,
and the never-confidently-wrong clarification fallback.
"""
import os
import sys

_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

from sigma_ground.materia import translate, answer


def test_routes_speed_question():
    spec = translate("How fast does a 5 cm copper ball hit the ground if you "
                     "drop it from 10 km?", use_qwen=False)
    assert spec.source == "deterministic"
    assert [s.verb for s in spec.steps] == ["terminal_velocity_drop"]
    p = spec.steps[0].params
    assert p["material_key"] == "copper"
    assert abs(p["radius_m"] - 0.05) < 1e-9
    assert abs(p["drop_altitude_m"] - 10000.0) < 1e-6


def test_routes_heat_question():
    spec = translate("Does an iron sphere heat up falling from 10 km?",
                     use_qwen=False)
    assert [s.verb for s in spec.steps] == ["drag_heating_drop"]
    assert spec.steps[0].params["material_key"] == "iron"


def test_slot_extraction_units():
    spec = translate("how fast does a 2 cm lead ball hit the ground from 30 km?",
                     use_qwen=False)
    p = spec.steps[0].params
    assert p["material_key"] == "lead"
    assert abs(p["radius_m"] - 0.02) < 1e-9
    assert abs(p["drop_altitude_m"] - 30000.0) < 1e-6


def test_both_intents_make_a_chain():
    """A question asking BOTH speed and heat compiles to a two-verb spec."""
    spec = translate("How fast and how hot does a steel ball get, dropped "
                     "from 20 km?", use_qwen=False)
    assert {s.verb for s in spec.steps} == {"terminal_velocity_drop",
                                            "drag_heating_drop"}
    assert all(s.params["material_key"] == "steel_mild" for s in spec.steps)


def test_drop_prefixed_size_extracts_radius():
    """'Drop a 2 cm ...' must read 2 cm as the SIZE, not an altitude."""
    spec = translate("Drop a 2 cm lead sphere from 30 km — how fast does it "
                     "land?", use_qwen=False)
    p = spec.steps[0].params
    assert abs(p["radius_m"] - 0.02) < 1e-9, p
    assert abs(p["drop_altitude_m"] - 30000.0) < 1e-6, p


def test_unknown_question_asks_for_clarification():
    spec = translate("What is the capital of France?", use_qwen=False)
    assert not spec.is_runnable()
    assert spec.source == "clarify"


def test_answer_runs_end_to_end():
    out = answer("How fast does a 5 cm copper ball hit the ground from 10 km?",
                 use_qwen=False)
    assert "routed →" in out
    assert "Impact speed" in out and "m/s" in out


# ── Family D routing ────────────────────────────────────────────────────
def test_routes_parachute_to_descent():
    spec = translate("Does a skydiver who free-falls from the stratosphere go "
                     "supersonic and then slow down?", use_qwen=False)
    assert [s.verb for s in spec.steps] == ["high_altitude_descent"]
    # must NOT carry the drop verbs' material/radius slots
    assert "material_key" not in spec.steps[0].params


def test_descent_extracts_altitude():
    spec = translate("A skydiver free-falls from 30 km — do they break the "
                     "sound barrier?", use_qwen=False)
    # 'sound barrier' is more specific, but 'free-fall' wins (checked first)
    assert spec.steps[0].verb == "high_altitude_descent"
    assert abs(spec.steps[0].params["start_altitude_m"] - 30000.0) < 1e-6


def test_out_of_scope_declines_not_misroutes():
    """A stray 'speed'/'hot' must NOT grab a drop verb for an unbuilt domain."""
    # rocket sled — has 'top speed', but it's a variable-mass rocket (not built)
    assert not translate("A rocket sled burns fuel fast — what's its top speed "
                         "at burnout?", use_qwen=False).is_runnable()
    # transient pipe conduction — has 'hot'/'burn', but it's a thermal PDE
    assert not translate("A cold iron pipe is blasted with boiling liquid — how "
                         "long until the outside is hot enough to burn?",
                         use_qwen=False).is_runnable()
    # double pendulum — has 'how fast', but it's rigid-body rotation
    assert not translate("A swinging stick with another dangling off it — how "
                         "fast are both spinning?", use_qwen=False).is_runnable()


def test_routes_supersonic_and_extracts_mach():
    spec = translate("How does a bullet's drag change when it's fired at Mach 3 "
                     "and breaks the sound barrier?", use_qwen=False)
    assert spec.steps[0].verb == "supersonic_projectile"
    assert abs(spec.steps[0].params["launch_mach"] - 3.0) < 1e-9


# ── Corrosion slots: duration + environment ─────────────────────────────
def test_corrosion_extracts_duration_and_environment():
    """The Captain's live question: '5 years' and 'alkaline soil' must land
    in the params, not be silently ignored (the pre-fix behavior)."""
    spec = translate("simulate a zinc rod stuck in a layer of oxidizing, "
                     "alkaline soil and see it corrode over 5 years",
                     use_qwen=False)
    assert [s.verb for s in spec.steps] == ["corrosion_attack"]
    p = spec.steps[0].params
    assert p["material_key"] == "zinc"
    assert abs(p["duration_s"] - 5 * 3.15e7) < 1e-3
    assert p["environment"] == "alkaline soil, aerated"


def test_duration_phrases():
    from sigma_ground.materia.translator import _extract_duration_s
    assert abs(_extract_duration_s("for 10 days") - 864000.0) < 1e-6
    assert abs(_extract_duration_s("after 3 months") - 3 * 3.15e7 / 12) < 1e-3
    assert abs(_extract_duration_s("in 2 weeks") - 1209600.0) < 1e-6
    assert abs(_extract_duration_s("over 1.5 hours") - 5400.0) < 1e-6
    assert abs(_extract_duration_s("rusting for a year") - 3.15e7) < 1e-3
    assert _extract_duration_s("corrosion rate of iron") is None
    # a rate ("m/s") or a temperature ("300 K") must not read as a duration
    assert _extract_duration_s("moving at 5 m/s at 300 K") is None


def test_corrosion_duration_defaults_when_unstated():
    """No duration named → slot omitted → the verb's 1-year default applies."""
    spec = translate("corrosion rate of iron", use_qwen=False)
    assert [s.verb for s in spec.steps] == ["corrosion_attack"]
    assert "duration_s" not in spec.steps[0].params
    assert "environment" not in spec.steps[0].params


def test_environment_label_composition():
    from sigma_ground.materia.translator import _extract_environment
    assert (_extract_environment("buried in waterlogged acidic clay")
            == "acidic soil, deaerated")
    assert _extract_environment("submerged in seawater") == "seawater immersed"
    assert _extract_environment("what rusts faster") is None
