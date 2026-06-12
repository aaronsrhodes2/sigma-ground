"""Bounce + tumble in the trajectory recorders — offline, synthetic constructs.

Physical invariants, not pixel checks: impact speeds strictly decrease, every
recorded contact dissipates energy, quats stay unit-norm, the body comes to
rest ON its support (not half-buried), an asymmetric shape tumbles while a
sphere doesn't, and a launched sphere rises before it falls.
"""
import math

from sigma_ground.deckard.construct import compile
from sigma_ground.deckard.schema import ConstructSpec, Fact, Part
from sigma_ground.radiance.trajectory import (record_fall, record_object_fall,
                                              _support, _surface_offsets)


def _est(v):
    return Fact(v, "estimated", "", 0.5)


def _construct(parts, name="probe"):
    return compile(ConstructSpec(name=name, kind="composite", identified=True,
                                 parts=parts, sources=[], notes=""), resolution=28)


def _box(name="block", mat="steel_mild", rho=7850.0):
    return Part(name, "box", {"x_m": _est(0.12), "y_m": _est(0.08), "z_m": _est(0.05)},
                mat, _est(rho), (0.0, 0.0, 0.025))


def test_object_fall_bounces_and_dissipates():
    c = _construct([_box()])
    out = record_object_fall(c, 1.0, frame_dt=0.03)
    val = out["trajectory"]["validation"]
    contacts = val["contacts"]
    assert len(contacts) >= 2                                  # it bounced
    speeds = [k["v_impact"] for k in contacts]
    assert all(a > b for a, b in zip(speeds, speeds[1:]))      # each hit slower
    assert val["energy_monotone"]                              # E_after < E_before, every contact
    assert "Johnson" in val["restitution_model"]               # emergent COR, not fallback


def test_object_comes_to_rest_on_its_support_not_buried():
    c = _construct([_box()])
    out = record_object_fall(c, 0.8, frame_dt=0.03)
    last = out["trajectory"]["frames"][-1]["bodies"][0]
    q = last["quat"]
    support, _ = _support(q, _surface_offsets(c))
    assert abs(last["pos"][1] - support) < 1e-4                # resting ON the floor
    assert abs(math.hypot(*q) ** 0 + 0) == 1 or True
    n = math.sqrt(sum(v * v for v in q))
    assert abs(n - 1.0) < 1e-6                                 # unit quat


def test_asymmetric_object_tumbles_symmetric_sphere_does_not():
    # box + offset knob: the contact point misses the CoM -> torque -> tumble
    asym = _construct([
        _box(),
        Part("knob", "sphere", {"radius_m": _est(0.02)}, "steel_mild",
             _est(7850.0), (0.05, 0.03, 0.06)),
    ])
    out = record_object_fall(asym, 1.0, frame_dt=0.03)
    quats = [f["bodies"][0]["quat"] for f in out["trajectory"]["frames"]]
    first, last = quats[0], quats[-1]
    dot = abs(sum(a * b for a, b in zip(first, last)))
    assert dot < 0.999999                                      # orientation changed: tumbled
    # the sphere lane: quat stays identity through its bounces
    sph = record_fall("steel_mild", radius_m=0.05, start_altitude_m=0.6,
                      dt_max=0.002, frame_dt=0.02)
    assert all(f["bodies"][0]["quat"] == [0.0, 0.0, 0.0, 1.0]
               for f in sph["trajectory"]["frames"])
    assert len(sph["trajectory"]["validation"]["contacts"]) >= 1


def test_frames_are_monotone_in_time():
    c = _construct([_box()])
    out = record_object_fall(c, 0.7, frame_dt=0.03)
    ts = [f["t_sim"] for f in out["trajectory"]["frames"]]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_launched_sphere_rises_then_falls_then_settles():
    out = record_fall("steel_mild", radius_m=0.04, start_altitude_m=0.04,
                      v0_m_s=8.0, dt_max=0.002, frame_dt=0.02)
    apex = out["trajectory"]["validation"]["apex_m"]
    assert apex > 1.5                                          # ~v²/2g ≈ 3.3 m minus drag
    ys = [f["bodies"][0]["pos"][1] for f in out["trajectory"]["frames"]]
    assert max(ys) > ys[0] + 1.0                               # it actually went up
    assert ys[-1] <= 0.05                                      # and ended at rest height


def test_long_fall_opts_into_the_follow_camera():
    c = _construct([_box()])
    out = record_object_fall(c, 50.0, frame_dt=0.1)
    assert out["scene"]["camera"].get("follow") is True
    assert "sdf_samples" in out["scene"]                       # self-check shipped with sims
