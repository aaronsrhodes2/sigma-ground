"""Materia routing gates for the two new hinge-arc demo verbs -- confirms
the EXACT, unmodified Captain prompts route deterministically (no LLM
residual needed) to the right verb with the right slots filled."""
from sigma_ground import materia
from sigma_ground.materia import scenarios as sc


def test_literal_pliers_prompt_routes_to_actuate_hand_tool():
    text = "Simulate a pair of pliers that are being open and closed over time."
    spec = materia.translate(text, use_qwen=False)
    assert spec.is_runnable()
    assert spec.source != "qwen"
    (step,) = spec.steps
    assert step.verb == "actuate_hand_tool"
    assert step.params.get("tool_name") == "pliers"


def test_literal_bell_prompt_routes_to_strike_bell():
    text = ("Simulate a hanging bell being struck by a stone, simulate the "
           "noise in an earth atmosphere.")
    spec = materia.translate(text, use_qwen=False)
    assert spec.is_runnable()
    assert spec.source != "qwen"
    (step,) = spec.steps
    assert step.verb == "strike_bell"


def test_actuate_hand_tool_verb_grounds_a_real_discovered_pivot():
    r = sc.actuate_hand_tool("pliers")
    assert r.outputs["can_render"]
    assert r.outputs["pivot_type"] == "revolute"
    assert r.outputs["render_handle"]["kind"] == "hand_tool_actuation"


def test_actuate_hand_tool_refuses_with_no_tool_named():
    r = sc.actuate_hand_tool(None)
    assert not r.outputs.get("can_render")
    assert r.validation["passed"] is False


def test_strike_bell_verb_grounds_real_ring_frequency():
    r = sc.strike_bell("iron", bell_diameter_m=0.15)
    assert r.outputs["can_render"]
    assert r.outputs["ring_frequency_hz"] > 0
    assert r.outputs["render_handle"]["kind"] == "bell_strike"


def test_drop_a_pliers_still_reads_as_a_drop_not_an_actuation():
    """A prompt with no actuation cue must NOT be hijacked by the new
    hand-tool routing -- 'pliers' alone isn't enough, an actuate cue is
    required (mirrors drop_object's own precedent of staying out of the
    way when its own cues aren't present)."""
    text = "How fast does a pair of pliers fall from 3 meters?"
    spec = materia.translate(text, use_qwen=False)
    verbs = {s.verb for s in spec.steps}
    assert "actuate_hand_tool" not in verbs
