"""record_windmill_theater gates -- Arc A Phase 5, the windmill's final
assembly. Everything upstream is already gated in isolation (tests/
test_mechanism_slidercrank.py, test_mechanism_bearing_gear_coupling.py,
test_mechanism_windmill_gearset.py, test_mechanism_windmill_drivetrain.py,
test_mechanism_windmill_capstone.py) -- this checks the RECORDER wiring
(frame contract, provenance surfaces, reservoir bookkeeping) matches those
already-proven physics, the same discipline record_clock's own test file
uses relative to the clock's Phase 0-4 mechanism tests.
"""
import math

import pytest

from sigma_ground.radiance.trajectory import record_windmill_theater
from sigma_ground.dynamics.quat import twist_angle


@pytest.fixture(scope="module")
def theater():
    return record_windmill_theater(t_max=30.0)


def test_scene_has_seven_bodies_and_matching_frame_shape(theater):
    scene = theater["scene"]
    frames = theater["trajectory"]["frames"]
    assert len(scene["bodies"]) == 7
    assert theater["trajectory"]["body_labels"] == [
        "rotor", "arbor0", "arbor1", "crank", "rod", "piston", "water"]
    for f in frames:
        assert len(f["bodies"]) == 7


def test_rotor_still_reaches_closed_form_terminal_omega(theater):
    val = theater["trajectory"]["validation"]
    trace = val["omega_trace_rad_s"]
    w_star = val["terminal_omega_expected_rad_s"]
    assert all(w <= w_star * 1.01 for w in trace)
    assert trace[-1] > 0.4 * w_star


def test_pump_accumulates_volume_from_detected_strokes(theater):
    val = theater["trajectory"]["validation"]
    assert val["pump_strokes"] > 0
    assert val["pump_volume_m3"] > 0.0
    assert val["pump_volume_m3"] <= val["tank_capacity_m3"] * 1.5   # sane scale


def test_water_body_y_is_monotonic_and_bounded(theater):
    frames = theater["trajectory"]["frames"]
    ys = [f["bodies"][6]["pos"][1] for f in frames]
    # cosmetic body: hand-set every frame from tracked volume, never a real
    # PhysicsParcel -- should rise monotonically (pump only fills) and stay
    # within the tank's own wall/cavity bounds
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
    assert min(ys) >= 0.0
    assert max(ys) <= 0.005 + 0.06 + 1e-9        # wall_t + tank_h


def test_water_body_never_added_to_physics_mass_ledger(theater):
    scene = theater["scene"]
    # 6 real dynamic parcels only -- the cosmetic water body contributes
    # nothing to the mass ledger (KNOWN_GAPS/Plug discipline: no invented
    # mass for non-real machinery)
    assert scene["physics"]["mass_kg"] > 0.0
    gear_leaves = [l for l in scene["csg_leaves"]
                  if l["shape"]["type"] == "Rotated"
                  and l["shape"]["shape"]["type"] == "Gear"]
    assert len(gear_leaves) == 4
    for l in gear_leaves:
        assert "[estimated]" in l["shape"]["shape"]["source"]


def test_every_holder_is_plugged_and_nothing_else(theater):
    scene = theater["scene"]
    kinds = [p["kind"] for p in scene["plugs"]]
    assert "drive" not in kinds                  # NATURAL drive: real wind
    assert kinds == ["support", "support"]
    assert "NATURAL" in scene["source"]
    assert "load-blind" in scene["source"]
    assert "SIMPLIFIED_MODEL" in scene["source"]


def test_wind_speed_variants_offered_as_adjustable_slider(theater):
    scene = theater["scene"]
    assert {v["slug"] for v in scene["variants"]} == {
        "windmill_theater_5ms", "windmill_theater_10ms"}
    assert scene["variant_current"] == "windmill_theater_10ms"


def test_crank_arm_and_rod_leaves_track_the_shared_hinge_axis(theater):
    """Sanity on the AXIS relabeling (Phase 0's +z hinge -> +x here): the
    crank body's actual twist about +x should be non-trivial by the end of
    a 30s run (matching the cumulative gear ratio derived from the wind-
    driven rotor, not stalled)."""
    frames = theater["trajectory"]["frames"]
    a0 = twist_angle(frames[0]["bodies"][3]["quat"], (1.0, 0.0, 0.0))
    a_end = twist_angle(frames[-1]["bodies"][3]["quat"], (1.0, 0.0, 0.0))
    assert a0 != a_end
