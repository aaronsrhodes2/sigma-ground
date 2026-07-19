"""record_hand_tool_actuation gates -- the hinge arc's demo payoff: a
DISCOVERED (not declared) pivot, driven by a SOLVED RevoluteJoint through
OscillatingRevoluteActuator, not a scripted animation."""
import math

import pytest

from sigma_ground.radiance.trajectory import record_hand_tool_actuation


def test_pliers_jaw_actually_oscillates_between_the_set_limits():
    bundle = record_hand_tool_actuation("pliers", n_cycles=2.0)
    v = bundle["trajectory"]["validation"]
    assert v["reversals"] >= 4                    # >= 2 full cycles
    assert v["cycles_completed"] >= 2.0
    lo, hi = v["limits_rad"]
    seen_min, seen_max = v["angle_range_rad"]
    assert seen_min >= lo - 0.05 and seen_max <= hi + 0.05     # never blew the wall
    assert seen_min <= lo + 0.05 and seen_max >= hi - 0.05     # actually swept both ends


def test_energy_ledger_holds():
    bundle = record_hand_tool_actuation("pliers", n_cycles=1.0)
    v = bundle["trajectory"]["validation"]
    assert v["energy_ledger_ok"]
    assert v["motor_work_j"] >= 0.0


def test_two_body_frame_contract_and_scene_have_matching_body_count():
    bundle = record_hand_tool_actuation("scissors", n_cycles=1.0)
    scene = bundle["scene"]
    assert len(scene["bodies"]) == 2
    frame = bundle["trajectory"]["frames"][0]
    assert len(frame["bodies"]) == 2
    body_ids = {leaf["body"] for leaf in scene["csg_leaves"]}
    assert body_ids == {0, 1}


def test_held_handle_stays_essentially_fixed():
    """Body 0 (the world-pinned handle) should barely move across the whole
    recording -- it is welded to the world, not driven."""
    bundle = record_hand_tool_actuation("pliers", n_cycles=2.0)
    frames = bundle["trajectory"]["frames"]
    p0 = frames[0]["bodies"][0]["pos"]
    for f in frames[1:]:
        p = f["bodies"][0]["pos"]
        d = math.dist(p0, p)
        assert d < 1e-6, f"held handle moved {d} m"


def test_provenance_cites_a_discovered_not_declared_pivot():
    bundle = record_hand_tool_actuation("pliers", n_cycles=1.0)
    src = bundle["scene"]["source"]
    assert "DISCOVERED" in src
    assert bundle["scene"]["plugs"]
    assert bundle["scene"]["choices"]
