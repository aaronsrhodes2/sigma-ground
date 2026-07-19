"""Phase 5 gates — THE CLOCK, the final assembly. Every prior phase
composed: Kelly (1944) cited tooth counts (Phase 2 catalog), GearCoupling
arbor chain (Phase 1), MainspringState + Escapement (Phase 3), InvoluteGear
leaves (Phase 4), all recorded through the frame contract (Phase 0).

The headline acceptance gate from the campaign plan: the minute hand
(center arbor) turns 2*pi per 3600 SIMULATED seconds — checked from the
RECORDED frame quaternions, not the solver's internal state. Timekeeping
here is DERIVED, never tuned: cited 600:1 escape:center ratio + 15 escape
teeth + anchor cadence -> 0.8 s pendulum -> 2*pi/3600 by construction.
Accuracy budget: pendulum period gate (~1%, test_joints.py) x exact
couplings (~0.1%, test_joints.py) — gate at 2%; the observed value in
development was 0.07%.

One shared recording (module fixture): record_clock simulates 7 bodies,
12 joints and 5 couplings at 960 steps/s — recording once and gating many
properties keeps the suite's wall time honest.
"""
import math

import pytest

from sigma_ground.radiance.trajectory import record_clock
from sigma_ground.dynamics.quat import twist_angle

_MINUTE_RATE = 2.0 * math.pi / 3600.0


@pytest.fixture(scope="module")
def clock():
    return record_clock(t_max=16.0)


def _unwrap(raw):
    out = [raw[0]]
    offset = 0.0
    prev = raw[0]
    for a in raw[1:]:
        d = a - prev
        if d > math.pi:
            offset -= 2.0 * math.pi
        elif d < -math.pi:
            offset += 2.0 * math.pi
        prev = a
        out.append(a + offset)
    return out


def _body_rate(frames, body):
    ts = [f["t_sim"] for f in frames]
    ang = _unwrap([twist_angle(f["bodies"][body]["quat"], (0, 0, 1))
                  for f in frames])
    return (ang[-1] - ang[0]) / (ts[-1] - ts[0]), (ang[-1] - ang[0])


def test_minute_hand_keeps_real_time(clock):
    """THE acceptance gate: 2*pi per 3600 simulated seconds, from frames."""
    rate, _ = _body_rate(clock["trajectory"]["frames"], body=1)  # center arbor
    assert abs(rate) == pytest.approx(_MINUTE_RATE, rel=0.02)


def test_minute_advance_is_exactly_geared_to_ticks(clock):
    """Independent of timing accuracy: per-tick advance == pitch/600, the
    cited train's exact gearing — catches coupling slippage even if the
    pendulum ran fast/slow."""
    val = clock["trajectory"]["validation"]
    _, advance = _body_rate(clock["trajectory"]["frames"], body=1)
    pitch = 2.0 * math.pi / val["escape_teeth_cited"]
    expected = val["ticks"] * pitch / val["esc_per_center_ratio_cited"]
    assert abs(advance) == pytest.approx(expected, rel=0.005)


def test_hour_hand_at_exactly_one_twelfth(clock):
    frames = clock["trajectory"]["frames"]
    _, d_minute = _body_rate(frames, body=1)
    _, d_hour = _body_rate(frames, body=6)
    assert d_hour / d_minute == pytest.approx(1.0 / 12.0, rel=0.005)
    assert d_hour * d_minute > 0.0          # co-rotating, like real hands


def test_ticks_at_derived_pendulum_half_period(clock):
    val = clock["trajectory"]["validation"]
    T = val["pendulum_period_expected_s"]
    assert T == pytest.approx(0.8)          # the derivation's own closed form
    gaps = [b - a for a, b in zip(val["tick_times_s"], val["tick_times_s"][1:])]
    assert len(gaps) >= 30
    for g in gaps:
        assert g == pytest.approx(T / 2.0, rel=0.02)


def test_energy_ledger_and_winding(clock):
    val = clock["trajectory"]["validation"]
    assert val["energy_gain_j"] <= val["motor_work_j"] + 1e-6
    w = val["winding_trace_rad"]
    assert all(b <= a for a, b in zip(w, w[1:]))     # monotone non-increasing
    assert w[-1] < w[0]                              # and actually unwinding


def test_every_gear_leaf_carries_citation_and_estimated_module_flag(clock):
    """The provenance gate from the plan: every mechanical parameter traces —
    each gear leaf names its cited source AND flags the estimated module."""
    gear_leaves = [l for l in clock["scene"]["csg_leaves"]
                  if l["shape"]["type"] == "Gear"]
    assert len(gear_leaves) == 9            # 72,12,80,10,75,10,80,8,15
    teeth = sorted(l["shape"]["teeth"] for l in gear_leaves)
    assert teeth == sorted([72, 12, 80, 10, 75, 10, 80, 8, 15])
    for l in gear_leaves:
        src = l["shape"]["source"]
        assert "Kelly" in src and "cited" in src
        assert "estimated" in src.lower()
