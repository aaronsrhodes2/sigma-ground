"""The closed artificial-force taxonomy (Golden Rule 7's vocabulary).

Pins the doctrine, not just the dataclass: the set is CLOSED (six kinds),
an unknown kind is a hard error naming the amendment process, each kind's
required parameters are enforced, and the spec IS its own plug disclosure
(round-trips through to_plug()/from_dict() without a second bookkeeping
object). Also pins the front door's verb table: every verb routes to a
member of the closed set, and kinds without a landed solver refuse
honestly instead of pretending.
"""
import pytest

from sigma_ground.dynamics.mechanisms.applied import AppliedForce, KINDS


def test_the_set_is_closed_and_has_exactly_six_kinds():
    assert set(KINDS) == {"impulse", "force", "drive", "hold", "source",
                          "tilt"}
    with pytest.raises(ValueError, match="CLOSED"):
        AppliedForce("yeet", "plate", vector=[1, 0, 0])
    # the error teaches the doctrine, not just the failure
    with pytest.raises(ValueError, match="doctrine amendment"):
        AppliedForce("teleport", "plate")


def test_impulse_is_instantaneous_by_definition():
    with pytest.raises(ValueError, match="instantaneous"):
        AppliedForce("impulse", "plate", vector=[1, 0, 0], duration=0.5)
    with pytest.raises(ValueError, match="vector"):
        AppliedForce("impulse", "plate")            # nothing delivered
    af = AppliedForce("impulse", "plate", vector=[0.5, 0, 0])
    assert af.duration == 0.0


def test_per_kind_required_parameters():
    with pytest.raises(ValueError, match="power_w"):
        AppliedForce("source", "Doughnut", duration=3.0)
    with pytest.raises(ValueError, match="angle_deg"):
        AppliedForce("tilt", "world", duration=2.0)
    # and the satisfied versions construct cleanly
    AppliedForce("source", "Doughnut", power_w=40.0, duration=3.0)
    AppliedForce("tilt", "world", duration=2.0,
                 variables={"angle_deg": 15.0, "axis": [0, 1, 0]})
    AppliedForce("force", "plate", vector=[2.0, 0, 0], duration=1.5)


def test_spec_and_disclosure_are_the_same_object():
    """to_plug() -> from_dict() round-trips: an artificial input cannot
    exist undisclosed, and the disclosure alone re-creates the input."""
    af = AppliedForce("impulse", "plate",
                      label="the push — an applied impulse on the plate",
                      vector=[0.3132, 0.0, 0.0],
                      variables={"push_speed_m_s": 1.2})
    plug = af.to_plug()
    assert plug["kind"] == "impulse" and plug["adjustable"] is True
    assert plug["variables"]["push_speed_m_s"] == 1.2
    assert plug["variables"]["vector"] == [0.3132, 0.0, 0.0]

    back = AppliedForce.from_dict(plug)
    assert back.kind == af.kind and back.target == af.target
    assert back.vector == [0.3132, 0.0, 0.0]
    assert back.variables["push_speed_m_s"] == 1.2


def test_front_door_verb_table_covers_only_the_closed_set():
    """Every verb row maps to a kind the taxonomy defines (impulse_twist is
    impulse's off-CoM/twist phrasing) — no verb can route to a kind the
    doctrine doesn't know."""
    from sigma_ground.mcp.front_door import _VERB_KINDS, _IMPLEMENTED_KINDS
    allowed = set(KINDS) | {"impulse_twist"}
    for _rx, kind in _VERB_KINDS:
        assert kind in allowed, kind
    assert _IMPLEMENTED_KINDS <= allowed


def test_front_door_refuses_unimplemented_kinds_honestly():
    from sigma_ground.mcp.front_door import dispatch, Session
    s = Session()
    dispatch("show me a Doughnut sitting on a plate, which is flat on top "
             "of a 1700s table", use_llm=False, session=s)
    assert s.last_renderable and s.last_renderable["kind"] == "scene_stack"
    env = dispatch("tilt the table until the plate slides", use_llm=False,
                   session=s)
    assert env["intent"] == "clarify"
    assert "Golden Rule 7" in env["text"]
    assert "IMPULSE" in env["text"]        # says what it CAN do today
    assert not env.get("saved")


def test_push_off_plug_is_an_applied_force_disclosure():
    """The recorder's push plug is generated THROUGH the taxonomy — parse
    it back into an AppliedForce and check J = m * v0."""
    from sigma_ground.radiance.push_off import record_push_off
    b = record_push_off(["1700s table", "plate", "Doughnut"],
                        pushed_index=1, push_speed_m_s=1.7)
    plug = next(p for p in b["scene"]["plugs"] if p["kind"] == "impulse")
    af = AppliedForce.from_dict(plug)
    assert af.target == "plate"
    assert af.variables["push_speed_m_s"] == 1.7
    # J = m*v0 cross-checked against the recorder's own energy trace:
    # right after the impulse, KE = J*v0/2 (only the pushed body moves)
    ke0 = b["trajectory"]["frames"][0]["energy_j"]
    assert ke0 == pytest.approx(af.vector[0] * 1.7 / 2, rel=0.05), (
        ke0, af.vector)
    # sim provenance rides in the bundle so /apply can re-run this matter
    assert b["sim"]["builder"] == "push_off"
    assert b["sim"]["pushed"] == "plate"
    assert b["sim"]["items"] == ["1700s table", "plate", "Doughnut"]


# ── the /apply endpoint (viewer drawer → real solver, no HTTP needed) ──────
SIM = {"builder": "push_off",
       "items": ["1700s table", "plate", "Doughnut"],
       "pushed": "plate", "pushed_index": 1}


def _apply(force, sim=SIM):
    from sigma_ground.radiance.web.serve import _apply_force
    return _apply_force({"sim": sim, "force": force})


def test_apply_endpoint_refuses_outside_the_closed_set():
    env = _apply({"kind": "yeet", "target": "plate"})
    assert env["intent"] == "clarify" and "CLOSED" in env["text"]


def test_apply_endpoint_refuses_unlanded_solvers_and_bad_targets():
    # payloads mirror the viewer drawer EXACTLY: a force REQUEST with no
    # vector — J = m*v0 needs the mass, which is matter and lives
    # server-side (found live 2026-07-30: requiring the vector here made
    # the real drawer's first run fail validation)
    env = _apply({"kind": "tilt", "target": "world",
                  "variables": {"angle_deg": 15.0}})
    assert env["intent"] == "clarify" and "impulse" in env["text"]
    env = _apply({"kind": "impulse", "target": "spoon",
                  "variables": {"push_speed_m_s": 2.0}})
    assert env["intent"] == "clarify" and "spoon" in env["text"]
    env = _apply({"kind": "impulse", "target": "1700s table",
                  "variables": {"push_speed_m_s": 2.0}})
    assert env["intent"] == "clarify" and "base" in env["text"]
    env = _apply({"kind": "impulse", "target": "plate",
                  "variables": {"push_speed_m_s": 2.0}}, sim={})
    assert env["intent"] == "clarify" and "provenance" in env["text"]


def test_apply_endpoint_runs_a_real_impulse():
    import json as _json
    import os
    env = _apply({"kind": "impulse", "target": "plate",
                  "variables": {"push_speed_m_s": 1.2}})
    assert env["intent"] == "render", env.get("text")
    assert env["saved"] and os.path.exists(env["saved"]["path"])
    b = _json.load(open(env["saved"]["path"], encoding="utf-8"))
    # 1.2 m/s is the gentle regime: everything stays on the table
    fs = b["trajectory"]["validation"]["final_support"]
    assert fs["plate"] == "on 1700s table"
    # and the disclosed plug reflects the speed that actually ran
    plug = next(p for p in b["scene"]["plugs"] if p["kind"] == "impulse")
    assert plug["variables"]["push_speed_m_s"] == 1.2
