"""Per-frame energy/momentum/angular_momentum fields on trajectory frames.

Phase A of the render-layer energy/momentum tracing work: every recording
function's frame dict now carries top-level ``energy_j`` / ``momentum`` /
``angular_momentum`` (scene-level, PhysicsScene.total_kinetic_energy() etc.
for a multi-body scene; manual scalar arithmetic for record_object_fall,
which has no PhysicsParcel/PhysicsScene in scope at all). Covers a
representative sample of the 11 recorders: record_fall (single-parcel),
record_gear_train_spin (a multi-body PhysicsScene recorder), record_
object_fall (the manual-arithmetic outlier), and record_bell_strike (a
second multi-body scene, now a hanging chain + struck bell + stone).

A second class of tests is a REGRESSION check: trajectory.validation.
energy_gain_j (the existing start/end summary some recorders already
compute) must still be computed identically to before this change --
the new per-frame fields are ADDED detail, not a replacement.
"""
import math

import pytest

from sigma_ground.deckard.construct import compile as deckard_compile
from sigma_ground.deckard.schema import ConstructSpec, Fact, Part
from sigma_ground.radiance.trajectory import (record_fall, record_object_fall,
                                              record_gear_train_spin,
                                              record_bell_strike,
                                              record_motor_spin,
                                              record_hand_tool_actuation)


def _est(v):
    return Fact(v, "estimated", "", 0.5)


def _box_construct(name="probe", mat="steel_mild", rho=7850.0):
    part = Part(name, "box", {"x_m": _est(0.12), "y_m": _est(0.08),
                              "z_m": _est(0.05)}, mat, _est(rho), (0.0, 0.0, 0.025))
    return deckard_compile(ConstructSpec(name=name, kind="composite", identified=True,
                                         parts=[part], sources=[], notes=""),
                           resolution=28)


def _assert_every_frame_has_the_fields(frames):
    for f in frames:
        assert "energy_j" in f
        assert "momentum" in f and len(f["momentum"]) == 3
        assert "angular_momentum" in f and len(f["angular_momentum"]) == 3
        assert f["energy_j"] >= -1e-9                  # KE is never negative
        assert all(math.isfinite(v) for v in f["momentum"])
        assert all(math.isfinite(v) for v in f["angular_momentum"])


# ── (a) record_fall — single PhysicsParcel/PhysicsScene ────────────────────

def test_record_fall_frames_carry_sane_energy_momentum():
    out = record_fall("copper", radius_m=0.05, start_altitude_m=20.0,
                      dt_max=0.005, frame_dt=0.05)
    frames = out["trajectory"]["frames"]
    _assert_every_frame_has_the_fields(frames)

    m = out["scene"]["physics"]["mass_kg"]
    # a bare sphere never spins in this recorder (no torque, vertical bounce)
    assert all(f["angular_momentum"] == [0.0, 0.0, 0.0] for f in frames)
    # momentum is momentum = m*v, purely vertical (x=z=0)
    for f in frames:
        assert f["momentum"][0] == 0.0 and f["momentum"][2] == 0.0
        expected_ke = 0.5 * m * (f["momentum"][1] / m) ** 2
        assert f["energy_j"] == pytest.approx(expected_ke, rel=1e-6, abs=1e-9)

    # KE rises through the free-fall phase (energy is being gained from
    # gravity, no bounce yet) -- the same story validation.contacts already
    # tells via E_before at the first impact
    contacts = out["trajectory"]["validation"]["contacts"]
    assert contacts, "expected at least one bounce from a 20 m drop"
    e_before_first_bounce = contacts[0]["E_before"]
    pre_bounce = [f["energy_j"] for f in frames if f["t_sim"] < contacts[0]["t"]]
    assert pre_bounce, "expected frames before the first bounce"
    assert max(pre_bounce) <= e_before_first_bounce * 1.05   # never overshoots the impact KE

    # settles: the LAST frame is at rest
    assert frames[-1]["energy_j"] == 0.0
    assert frames[-1]["momentum"] == [0.0, 0.0, 0.0]


# ── (b) a multi-body PhysicsScene recorder ──────────────────────────────────

def test_gear_train_frames_carry_sane_energy_momentum():
    out = record_gear_train_spin(motor_speed_rad_s=-2.0, motor_max_torque=1.0,
                                 t_max=0.5, frame_dt=0.02)
    frames = out["trajectory"]["frames"]
    _assert_every_frame_has_the_fields(frames)

    # a motor-driven train GAINS kinetic energy over the recorded run: it
    # spins up from rest and settles at a nonzero steady-state KE (a torque-
    # capped motor reaching its commanded speed and holding it -- constant
    # KE thereafter is the CORRECT physics, not a frozen/stale field)
    assert frames[0]["energy_j"] == 0.0
    assert frames[-1]["energy_j"] > frames[0]["energy_j"]
    energies = [f["energy_j"] for f in frames]
    assert len(set(round(e, 9) for e in energies)) >= 2   # it did change at least once


# ── (c) record_object_fall — the manual-arithmetic outlier ─────────────────

def test_object_fall_frames_carry_sane_energy_momentum():
    c = _box_construct()
    out = record_object_fall(c, 1.0, frame_dt=0.03)
    frames = out["trajectory"]["frames"]
    _assert_every_frame_has_the_fields(frames)

    m = out["scene"]["physics"]["mass_kg"]
    # momentum stays purely vertical (record_object_fall only integrates a
    # vertical drop -- x/z components are always exactly zero)
    assert all(f["momentum"][0] == 0.0 and f["momentum"][2] == 0.0 for f in frames)
    # release: the hist-based pre-bounce frames start from (near) rest --
    # frame[0] is the state after Materia's FIRST integration sub-step (not
    # literally t=0), so it carries a small nonzero KE, not exactly zero;
    # it must still be tiny next to the peak reached just before impact
    peak = max(f["energy_j"] for f in frames)
    assert 0.0 <= frames[0]["energy_j"] < 0.05 * peak
    # KE rises during the fall (gravity does work) -- peak translational KE
    # should roughly match m*g*h for the drop, generously bounded (drag +
    # rotational share make an exact match wrong, but it must be the right
    # order of magnitude, not e.g. 100x off from a units bug)
    assert 0.0 < peak < 2.0 * m * 9.80665 * 1.0

    # comes to rest: final frames are settled (zero energy/momentum)
    assert frames[-1]["energy_j"] == 0.0
    assert frames[-1]["momentum"] == [0.0, 0.0, 0.0]


def test_object_fall_tumbling_shape_shows_nonzero_angular_momentum():
    # box + off-CoM knob: contact torque should show up as real angular
    # momentum at some point after the first bounce (not just quat drift)
    asym = deckard_compile(ConstructSpec(
        name="asym", kind="composite", identified=True,
        parts=[Part("block", "box", {"x_m": _est(0.12), "y_m": _est(0.08),
                                     "z_m": _est(0.05)}, "steel_mild",
                    _est(7850.0), (0.0, 0.0, 0.025)),
               Part("knob", "sphere", {"radius_m": _est(0.02)}, "steel_mild",
                    _est(7850.0), (0.05, 0.03, 0.06))],
        sources=[], notes=""), resolution=28)
    out = record_object_fall(asym, 1.0, frame_dt=0.03)
    frames = out["trajectory"]["frames"]
    _assert_every_frame_has_the_fields(frames)
    Lmags = [math.sqrt(sum(v * v for v in f["angular_momentum"])) for f in frames]
    assert max(Lmags) > 1e-8                        # it really did pick up spin


# ── (d) record_bell_strike — a second multi-body scene (chain + bell + stone) ─

def test_bell_strike_frames_carry_sane_energy_momentum():
    out = record_bell_strike(bell_material="iron", bell_diameter_m=0.15,
                             stone_mass_kg=0.2, observer_distance_m=50.0)
    frames = out["trajectory"]["frames"]
    _assert_every_frame_has_the_fields(frames)

    # released at rest
    assert frames[0]["energy_j"] == 0.0
    assert frames[0]["momentum"] == [0.0, 0.0, 0.0]
    # after the stone is dropped and strikes, SOMETHING in the scene
    # (chain + bell + stone) carries real kinetic energy and, since the
    # bell can swing off a hinge, real angular momentum too
    assert max(f["energy_j"] for f in frames) > 0.0
    assert any(any(abs(c) > 1e-9 for c in f["angular_momentum"]) for f in frames)
    val = out["trajectory"]["validation"]
    assert val["struck"] is True


# ── regression: trajectory.validation.energy_gain_j is UNCHANGED ───────────
# (per-frame detail is ADDED, the existing start/end summary must still be
# computed exactly as before -- and since energy_gain_j = KE(end) - KE(start)
# with KE(start)/KE(end) sampled at the SAME scene states as frames[0] and
# frames[-1], the two must agree exactly, not just approximately)

def test_motor_spin_energy_gain_matches_frame_delta_exactly():
    out = record_motor_spin(motor_speed_rad_s=-3.0, motor_max_torque=2.0,
                            t_max=0.5)
    frames = out["trajectory"]["frames"]
    val = out["trajectory"]["validation"]
    assert val["energy_gain_j"] == frames[-1]["energy_j"] - frames[0]["energy_j"]


def test_hand_tool_actuation_energy_gain_matches_frame_delta_exactly():
    out = record_hand_tool_actuation("pliers", n_cycles=1.0)
    frames = out["trajectory"]["frames"]
    val = out["trajectory"]["validation"]
    assert val["energy_gain_j"] == frames[-1]["energy_j"] - frames[0]["energy_j"]


def test_gear_train_energy_gain_matches_frame_delta_exactly():
    out = record_gear_train_spin(motor_speed_rad_s=-2.0, motor_max_torque=1.0,
                                 t_max=0.5)
    frames = out["trajectory"]["frames"]
    val = out["trajectory"]["validation"]
    assert val["energy_gain_j"] == frames[-1]["energy_j"] - frames[0]["energy_j"]
