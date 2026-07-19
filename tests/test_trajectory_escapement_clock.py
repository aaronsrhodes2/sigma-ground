"""Phase 3 capstone gate — the spring + escapement wiring, recorded into the
trajectory frame contract. Both mechanisms are independently gated against
closed forms elsewhere (tests/test_mechanism_spring.py,
tests/test_mechanism_escapement.py); this proves the render bridge carries
BOTH bodies' solved motion faithfully once they're wired together.
"""
import math

from sigma_ground.radiance.trajectory import record_escapement_clock
from sigma_ground.dynamics.quat import twist_angle


def _z_angle(quat):
    return twist_angle(quat, (0.0, 0.0, 1.0))


def test_escapement_clock_ticks_and_winds_down():
    out = record_escapement_clock(t_max=14.0)
    val = out["trajectory"]["validation"]

    assert val["ticks"] >= 15
    gaps = [b - a for a, b in
           zip(val["tick_times_s"], val["tick_times_s"][1:])]
    mean_gap = sum(gaps) / len(gaps)
    # every gap close to the mean — a real, steady beat, not sporadic firing
    assert all(abs(g - mean_gap) < 0.02 * mean_gap for g in gaps)

    # energy ledger: same bound used throughout — total KE (both bodies) is
    # bounded by the escape wheel motor's own logged work
    assert val["energy_gain_j"] > 0.0
    assert val["energy_ledger_ok"]

    # the spring actually winds down over the observation window (not just
    # a static number) — one tooth pitch per tick, so the 5-rad default
    # exhausts at ~24 ticks (~10s) < the 14s window, and it must run out
    # fully (remaining floored at exactly 0, per MainspringState)
    assert val["theta_wound_remaining_rad"] == 0.0
    assert val["spring_wound_out"]


def test_escapement_clock_frames_carry_both_bodies_and_stay_unit_norm():
    out = record_escapement_clock(t_max=3.0)
    frames = out["trajectory"]["frames"]
    assert len(out["scene"]["bodies"]) == 2
    for f in frames:
        assert len(f["bodies"]) == 2
        for b in f["bodies"]:
            n = math.sqrt(sum(v * v for v in b["quat"]))
            assert abs(n - 1.0) < 1e-6


def test_escapement_clock_wheel_rate_matches_pendulum_half_period():
    """Cross-check the RECORDED escape-wheel frames against the pendulum's
    closed-form period — the same style of check test_trajectory_gear_train.py
    already uses for GearCouplingJoint, now for the escapement's tick cadence."""
    L, m_pend = 0.4, 0.05
    d = 0.5 * L
    I_pivot = m_pend * L * L / 3.0
    T_expect = 2.0 * math.pi * math.sqrt(I_pivot / (m_pend * 9.80665 * d))

    out = record_escapement_clock(t_max=8.0)
    val = out["trajectory"]["validation"]
    gaps = [b - a for a, b in
           zip(val["tick_times_s"], val["tick_times_s"][1:])]
    mean_gap = sum(gaps) / len(gaps)
    assert mean_gap == T_expect / 2.0 or abs(mean_gap - T_expect / 2.0) < 0.01 * (T_expect / 2.0)
