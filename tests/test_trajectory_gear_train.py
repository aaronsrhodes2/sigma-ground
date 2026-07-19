"""Phase 1 gate — multi-body coupled rotation recorded into the trajectory
frame contract.

record_gear_train_spin wires dynamics/joints.py's GearCouplingJoint (a
kinematic rate constraint between two RevoluteJoints) into a multi-body
trajectory bundle. Gates here check the RECORDED FRAMES carry each wheel's
own solved rate at the commanded ratio — dynamics/test_joints.py's
test_gear_coupling_holds_the_commanded_ratio already gates the constraint
itself; this proves the render bridge doesn't drop or misalign anything
across more than one body.
"""
import math

import pytest

from sigma_ground.radiance.trajectory import record_gear_train_spin
from sigma_ground.dynamics.quat import twist_angle


def _z_angle(quat):
    return twist_angle(quat, (0.0, 0.0, 1.0))


def _unwrap(raw):
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


def _steady_rate(frames, body_idx, ts):
    raw = [_z_angle(f["bodies"][body_idx]["quat"]) for f in frames]
    unwrapped = _unwrap(raw)
    n = len(unwrapped)
    i0 = n // 3
    return (unwrapped[-1] - unwrapped[i0]) / (ts[-1] - ts[i0])


def test_gear_train_frames_carry_each_wheels_solved_rate():
    ratios = (1.8, -1.5)
    out = record_gear_train_spin(ratios=ratios, motor_speed_rad_s=-3.0,
                                 motor_max_torque=2.0, t_max=3.0,
                                 frame_dt=0.02)
    frames = out["trajectory"]["frames"]
    n_bodies = len(out["scene"]["bodies"])
    assert n_bodies == 3
    assert all(len(f["bodies"]) == n_bodies for f in frames)

    ts = [f["t_sim"] for f in frames]
    rate0 = _steady_rate(frames, 0, ts)
    rate1 = _steady_rate(frames, 1, ts)
    rate2 = _steady_rate(frames, 2, ts)

    # GearCouplingJoint's constraint (proven directly in test_joints.py) is
    # s_a + ratio*s_b = 0 in the JOINTS' own convention; the recorder's frame
    # quats encode each wheel's raw WORLD rotation, which the constraint
    # ratio maps onto directly (both wheels share the same construction:
    # RevoluteJoint(wheel, None, ...) — b=None throughout, so the s<->world
    # sign relationship is identical for every joint in the chain, and the
    # ratio applies to the recorded world rates unchanged).
    assert rate1 == pytest.approx(-rate0 / ratios[0], rel=0.05)
    assert rate2 == pytest.approx(-rate1 / ratios[1], rel=0.05)

    # every wheel actually moves — this isn't three motionless discs
    assert abs(rate0) > 0.5
    assert abs(rate1) > 0.1
    assert abs(rate2) > 0.1


def test_gear_train_energy_ledger_holds():
    out = record_gear_train_spin(t_max=2.0)
    val = out["trajectory"]["validation"]
    assert val["energy_gain_j"] > 0.0
    assert val["energy_ledger_ok"]


def test_gear_train_quats_stay_unit_norm():
    out = record_gear_train_spin(t_max=1.5)
    for f in out["trajectory"]["frames"]:
        for b in f["bodies"]:
            q = b["quat"]
            norm = math.sqrt(sum(v * v for v in q))
            assert abs(norm - 1.0) < 1e-6
