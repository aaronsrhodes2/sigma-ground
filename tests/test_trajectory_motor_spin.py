"""Phase 0 gate — the render/dynamics bridge.

record_motor_spin wires dynamics/joints.py's validated RevoluteJoint motor
(test_joints.py::test_motor_ledger_bounds_energy_gain) into the trajectory
frame contract's `quat` channel for the first time: every prior "spinning"
demo (build_demo.py's chair tip) scripted the rotation, this one plays back
what the constraint solver actually solved. Gates here are closed-form/
ledger checks on the RECORDED FRAMES themselves, not on the live scene —
proving the bridge, not just the motor (which test_joints.py already covers).
"""
import math

import pytest

from sigma_ground.radiance.trajectory import record_motor_spin
from sigma_ground.dynamics.quat import twist_angle


def _z_angle(quat):
    """Twist angle about +z from the identity reference — the same
    convention RevoluteJoint.angle() uses against its construction pose."""
    return twist_angle(quat, (0.0, 0.0, 1.0))


def _unwrap(raw):
    """Standard phase unwrap: keep angles continuous across the +-pi seam
    twist_angle wraps at, so a steady spin reads as a straight ramp."""
    out = [raw[0]]
    offset = 0.0
    prev_raw = raw[0]
    for a in raw[1:]:
        d = a - prev_raw
        if d > math.pi:
            offset -= 2.0 * math.pi
        elif d < -math.pi:
            offset += 2.0 * math.pi
        prev_raw = a
        out.append(a + offset)
    return out


def test_motor_spin_frames_carry_the_solved_rotation_rate():
    motor_speed, motor_torque = -3.0, 2.0
    out = record_motor_spin(motor_speed_rad_s=motor_speed,
                            motor_max_torque=motor_torque, t_max=3.0,
                            frame_dt=0.05)
    frames = out["trajectory"]["frames"]
    assert len(frames) > 10

    raw = [_z_angle(f["bodies"][0]["quat"]) for f in frames]
    ts = [f["t_sim"] for f in frames]
    unwrapped = _unwrap(raw)

    # Sign convention (documented in dynamics/joints.py): with b=None the
    # WORLD plays the joint's "child" role, so RevoluteJoint.angle()/motor_speed
    # read "world relative to the disc" — a disc spinning +n̂ in the WORLD
    # frame (what these recorded quats directly encode) reads dθ/dt < 0 in
    # the joint's own s. motor_speed=-3.0 therefore commands +3.0 rad/s of
    # actual world-frame spin; expected_world_rate flips the sign to match.
    expected_world_rate = -motor_speed

    # Steady-state slope (skip the torque-capped ramp-up — for this I/torque
    # combo it settles in ~1 ms, negligible against a 3 s recording): the
    # RECORDED frames' rotation rate must match the commanded motor speed,
    # proving the bridge carries the physics faithfully, not just plausibly.
    n = len(unwrapped)
    i0 = n // 3
    rate = (unwrapped[-1] - unwrapped[i0]) / (ts[-1] - ts[i0])
    assert rate == pytest.approx(expected_world_rate, rel=0.05)

    # never reverses direction once steady — a continuous spin, not a wobble.
    # (>= 0, not > 0: like every other recorder here, an unconditional final
    # frame is appended after the loop and can duplicate the last in-loop
    # frame's t_sim, giving one benign zero-length diff — not a reversal.)
    tail = unwrapped[i0:]
    diffs = [b - a for a, b in zip(tail, tail[1:])]
    assert all(d * expected_world_rate >= 0 for d in diffs)
    assert sum(1 for d in diffs if d == 0.0) <= 1   # at most the trailing duplicate

    # the recorded final angle agrees with the joint's own solved angle
    # (negated per the sign convention above) — the bridge didn't drop or
    # misalign anything between solver and frames
    val = out["trajectory"]["validation"]
    diff = unwrapped[-1] - (-val["final_angle_rad"])
    diff -= 2 * math.pi * round(diff / (2 * math.pi))
    assert abs(diff) < 1e-4


def test_motor_spin_energy_ledger_holds():
    out = record_motor_spin(motor_speed_rad_s=-3.0, motor_max_torque=2.0,
                            t_max=2.0)
    val = out["trajectory"]["validation"]
    assert val["energy_gain_j"] > 0.0             # the motor really worked
    assert val["energy_ledger_ok"]                 # dE <= motor_work_j (ledger bound)


def test_motor_spin_quats_stay_unit_norm():
    out = record_motor_spin(t_max=1.5)
    for f in out["trajectory"]["frames"]:
        q = f["bodies"][0]["quat"]
        n = math.sqrt(sum(v * v for v in q))
        assert abs(n - 1.0) < 1e-6
