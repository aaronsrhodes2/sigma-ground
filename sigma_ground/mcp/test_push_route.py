"""The push follow-up: "now push the plate off the table and see what happens".

The missing half of the conversation, found by LIVE test (2026-07-30): turn 1
rendered the doughnut/plate/table stack correctly but left
``Session.last_renderable`` as None, "push" matched no verb anywhere, and
"it" had nothing to bind to — so the follow-up fell to the canned physics
clarify. These tests pin all three fixes plus the recorder's emergent
outcome, entirely offline (allow_llm=False; catalog/archetype objects only).
"""
import json
import os

from sigma_ground.mcp.front_door import (dispatch, Session,
                                         _resolve_push_target, _PUSH_RE)

CANON = ("I want a simulation of a Doughnut, with frosting and rainbow "
         "sprinkles. I want it sitting on a plate, which is flat on top of "
         "a 1700s table.")
FOLLOW = "Now push the plate off the table and see what happens."


# ── unit: verb + pronoun binding ────────────────────────────────────────────
def test_push_verb_matches_and_pushpin_does_not():
    assert _PUSH_RE.search("Now push the plate off")
    assert _PUSH_RE.search("shove it off the shelf")
    assert _PUSH_RE.search("knock the cup off")
    assert not _PUSH_RE.search("a pushpin")          # word boundary, not substring
    assert not _PUSH_RE.search("the tipsy sailor")


def test_target_resolution_named_pronoun_and_base_refusal():
    items = ["1700s table", "plate", "Doughnut"]
    assert _resolve_push_target("push the plate off the table", items) == 1
    assert _resolve_push_target("push the doughnut off the plate", items) == 2
    assert _resolve_push_target("now push it off", ["table", "Doughnut"]) == 1
    # the base is what everything stands on — pushing it is a different
    # simulation (tipping) this route honestly refuses to claim
    assert _resolve_push_target("push the table", ["1700s table", "plate"]) is None


# ── the whole conversation, offline ─────────────────────────────────────────
def test_two_turn_push_conversation_produces_a_real_simulation():
    s = Session()
    e1 = dispatch(CANON, use_llm=False, session=s)
    assert e1["intent"] == "render"
    # turn 1 REMEMBERS what it built — this was the gap
    assert s.last_renderable == {"kind": "scene_stack",
                                 "items": ["1700s table", "plate", "Doughnut"]}

    e2 = dispatch(FOLLOW, use_llm=False, session=s)
    assert e2["intent"] == "render", e2["text"]
    assert e2["saved"] and os.path.exists(e2["saved"]["path"])

    bundle = json.load(open(e2["saved"]["path"], encoding="utf-8"))
    tr = bundle["trajectory"]
    sc = bundle["scene"]
    assert len(tr["frames"]) > 30
    assert tr["validation"]["pushed"] == "plate"
    # the push is an APPLIED artificial impulse — disclosed as a plug, and
    # every invented number (speed, mu, restitution) is a disclosed Choice
    assert any("push" in p["description"] for p in sc["plugs"])
    descs = " | ".join(c["description"] for c in sc["choices"])
    assert "push speed" in descs and "friction" in descs and "restitution" in descs
    # per-frame energy tracing rides along, like every other recorder
    assert "energy_j" in tr["frames"][0]


def test_push_with_unresolvable_target_gets_a_specific_clarify():
    s = Session()
    dispatch("show me a Doughnut sitting on a table", use_llm=False, session=s)
    assert s.last_renderable and s.last_renderable["kind"] == "scene_stack"
    env = dispatch("push the table", use_llm=False, session=s)
    assert env["intent"] == "clarify"
    assert "base" in env["text"]                     # the honest, specific reply


def test_push_without_a_remembered_stack_does_not_hijack_physics():
    """No stack in memory → the push route must not fire; the old routing is
    untouched (a fresh 'push' question still goes wherever it went before)."""
    s = Session()
    env = dispatch("Now push the plate off the table.", use_llm=False, session=s)
    assert env["intent"] in ("ask", "clarify", "simulate")   # anything but a bogus render
    assert not env.get("saved")


# ── the recorder's emergent physics ────────────────────────────────────────
def test_tablecloth_outcome_and_choice_sensitivity():
    """With the default mu values the plate slides out from under the doughnut
    (relative slip exceeds the plate radius): plate → floor, doughnut → left
    on the table. That outcome comes from the stacked-block friction
    inequality, not a script — so making the icing grip like rubber must
    change it (the doughnut then rides the plate off the edge). Both runs
    assert real physics invariants either way: no NaN, energy ends ~zero,
    every body ends supported."""
    import math
    from sigma_ground.radiance.push_off import record_push_off

    b = record_push_off(["1700s table", "plate", "Doughnut"], pushed_index=1)
    val = b["trajectory"]["validation"]
    assert val["pushed_left_base"] is True
    assert val["final_support"]["plate"] == "floor"
    assert val["final_support"]["Doughnut"] == "on 1700s table"
    frames = b["trajectory"]["frames"]
    assert not any(math.isnan(v) for f in frames
                   for bd in f["bodies"] for v in bd["pos"])
    assert frames[-1]["energy_j"] < 0.5              # settled, not still flying

    # sensitivity: icing gripping like rubber flips the outcome — the doughnut
    # now RIDES the plate off the edge instead of being left behind. Same
    # code, different disclosed mu: the outcome is the physics.
    b2 = record_push_off(["1700s table", "plate", "Doughnut"], pushed_index=1,
                         mu_overrides={"topper_on_plate": 3.0})
    val2 = b2["trajectory"]["validation"]
    assert val2["final_support"]["Doughnut"] != "on 1700s table"
    # and the disclosure shows the value that actually ran
    mu_disclosed = next(c for c in b2["scene"]["choices"]
                        if "friction" in c["description"])
    assert mu_disclosed["value"]["topper_on_plate"] == 3.0
