"""The in-game grader (stage/gmod/verify_autotest.py) must pass a faithful run and FAIL a wrong one.

A grader that cannot fail is worthless, so this builds synthetic autotest recordings from the
fall_skillet bundle: an ideal engine (right frame mapping, Havok falling in vacuum then resting
on its base) and several deliberately broken ones (wrong axis mapping, wrong floor, upside down,
never settles, wrong gravity). No game and no lupa needed.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest

from sigma_ground.radiance import export_gmod
from sigma_ground.radiance.trajectory import poses_at

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("verify_autotest", REPO / "stage" / "gmod" / "verify_autotest.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

TICK = 1 / 66.6667


def _bundle():
    b, name, sha, _ = export_gmod.load_bundle("fall_skillet")
    return b


def _ideal_run(release_at=0.31):
    """What a correct engine would record for fall_skillet."""
    b = _bundle()
    traj = b["trajectory"]
    anchor = [300.0, 40.0, 12.0]
    pivot = b["scene"]["bodies"][0]["pivot"][2]
    g = 9.80665 * V.HU_PER_M
    samples, t, ct = [], traj["frames"][0]["t_sim"], 1000.0
    rest_z = anchor[2] + pivot * V.HU_PER_M
    released = False
    pos = vel = None
    while True:
        if not released and t >= release_at:
            released = True
            rel_ct, rel_t = ct, t
            p0, p1 = poses_at(traj, t - 1e-4)[0]["pos"], poses_at(traj, t + 1e-4)[0]["pos"]
            vz = (p1[1] - p0[1]) / 2e-4 * V.HU_PER_M                # sigma +y is Source +z
            pos = list(V.expected_pos("y", poses_at(traj, t)[0]["pos"], anchor, 0.0))
            vel = [0.0, 0.0, vz]
        if released:
            vel[2] -= g * TICK
            pos[2] += vel[2] * TICK
            if pos[2] <= rest_z:                                    # lands flat on its base
                pos[2], vel = rest_z, [0.0, 0.0, 0.0]
            f, r, u = (1, 0, 0), (0, -1, 0), (0, 0, 1)
            s = {"ct": ct, "sim_t": t, "pose_t": None, "released": True, "pos": list(pos),
                 "fwd": list(f), "right": list(r), "up": list(u), "vel": list(vel)}
        else:
            pose = poses_at(traj, t)[0]
            f, r, u = V.expected_axes("y", pose["quat"])
            s = {"ct": ct, "sim_t": t, "pose_t": t, "released": False,
                 "pos": list(V.expected_pos("y", pose["pos"], anchor, 0.0)),
                 "fwd": list(f), "right": list(r), "up": list(u), "vel": [0.0, 0.0, 0.0]}
        samples.append(s)
        t += TICK
        ct += TICK
        if released and ct - rel_ct > 3.0:
            break
    return {"scenario": "fall_skillet", "ok": True, "anchor": anchor, "samples": samples, "settled": True,
            "released_ct": rel_ct, "released_sim_t": rel_t, "release_at": release_at,
            "gravity": [0.0, 0.0, -g], "tick_interval": TICK, "map": "gm_construct", "version": "test",
            "air_density": 0, "drag_enabled": False}, b, pivot


def test_grader_passes_a_faithful_run():
    run, b, pivot = _ideal_run()
    rows, fails = V.grade(run, b, pivot)
    assert not fails, fails
    names = {r[0] for r in rows}
    assert {"REPLAY max position error", "REPLAY max axis error", "resting pivot height"} <= names


def test_ideal_run_exercises_the_axis_check():
    """The skillet stands up in Source: the recorded up vector is world +Z."""
    run, b, pivot = _ideal_run()
    first = next(s for s in run["samples"] if not s["released"])
    assert first["up"] == pytest.approx([0, 0, 1], abs=1e-9)


@pytest.mark.parametrize("breakage", ["swap_yz_mapping", "floor_off", "upside_down", "never_settles",
                                      "low_gravity", "no_release", "sloppy_orientation"])
def test_grader_fails_a_broken_run(breakage):
    run, b, pivot = _ideal_run()
    run = copy.deepcopy(run)
    if breakage == "swap_yz_mapping":              # engine mapped sigma z -> Source y with the wrong sign
        for s in run["samples"]:
            if not s["released"]:
                s["pos"][1] = -s["pos"][1] + 2 * run["anchor"][1]
        # the skillet is on the x/y=0 axis, so also shift to make the error visible
        for s in run["samples"]:
            if not s["released"]:
                s["pos"][1] += 3.0
    elif breakage == "floor_off":
        for s in run["samples"]:
            s["pos"][2] += 10.0
    elif breakage == "upside_down":
        for s in run["samples"]:
            s["up"] = [0, 0, -1]
    elif breakage == "never_settles":
        run["settled"] = False
    elif breakage == "low_gravity":
        run["gravity"][2] *= 0.5
    elif breakage == "no_release":
        run["samples"] = [s for s in run["samples"] if not s["released"]]
    elif breakage == "sloppy_orientation":         # 3 degree error in forward axis
        c, s_ = math.cos(math.radians(3)), math.sin(math.radians(3))
        for smp in run["samples"]:
            if not smp["released"]:
                f = smp["fwd"]
                smp["fwd"] = [f[0] * c - f[1] * s_, f[0] * s_ + f[1] * c, f[2]]
    rows, fails = V.grade(run, b, pivot)
    assert fails, f"{breakage}: the grader let a broken run pass"


def test_independent_axis_helpers_agree_on_the_skillet():
    ex, right, ez = V.expected_axes("y", [-math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert ez == pytest.approx((0, 0, 1), abs=1e-12)
    assert ex == pytest.approx((1, 0, 0), abs=1e-12)
    assert right == pytest.approx((0, -1, 0), abs=1e-12)


def test_engine_drag_on_is_informational_not_a_failure():
    """Source's default air resistance slows a falling body; with drag declared ON the free-fall
    comparison must not fail the run (it is the engine, not the handoff)."""
    run, b, pivot = _ideal_run()
    run = copy.deepcopy(run)
    run.pop("air_density"); run.pop("drag_enabled")
    for s in run["samples"]:                       # a draggy fall: stretch the descent by 8 %
        if s["released"]:
            s["pos"][2] = run["anchor"][2] + (s["pos"][2] - run["anchor"][2]) * 1.0
    rows, fails = V.grade(run, b, pivot)
    assert not fails
    assert any("engine drag ON" in r[0] for r in rows)


# ----------------------------------------------------------------------- spin handoff check
def _spin_samples(rate, axis, n=8, dt=TICK, z0=500.0):
    """Entity-axis samples of a body spinning at `rate` rad/s about the world `axis` (Rodrigues)."""
    ax = [c / math.sqrt(sum(x * x for x in axis)) for c in axis]

    def rot(theta, v):
        c, s_ = math.cos(theta), math.sin(theta)
        d = sum(a * b for a, b in zip(ax, v))
        cr = [ax[1] * v[2] - ax[2] * v[1], ax[2] * v[0] - ax[0] * v[2], ax[0] * v[1] - ax[1] * v[0]]
        return [v[i] * c + cr[i] * s_ + ax[i] * d * (1 - c) for i in range(3)]

    out = []
    for k in range(n):
        th = rate * dt * k
        f, left, u = rot(th, (1, 0, 0)), rot(th, (0, 1, 0)), rot(th, (0, 0, 1))
        out.append({"ct": 10.0 + dt * k, "released": True, "pos": [0, 0, z0 - k], "fwd": f,
                    "right": [-c for c in left], "up": u, "vel": [0, 0, 0]})
    return out


def _spin_bundle(w_rad_s=4.0, axis_sigma_z=True):
    """A one-body trajectory spinning about sigma +Z at w rad/s (Z-up bundle => Source +Z)."""
    frames = []
    for i in range(20):
        t = i * 0.05
        th = w_rad_s * t
        frames.append({"t_sim": t, "bodies": [{"pos": [0, 0, 1], "quat": [0, 0, math.sin(th / 2), math.cos(th / 2)]}]})
    return {"frames": frames}


@pytest.mark.parametrize("engine_rate,engine_axis,should_pass", [
    (4.0, (0, 0, 1), True),                    # faithful
    (4.0 * 57.29578, (0, 0, 1), False),        # degrees treated as radians
    (4.0, (1, 0, 0), False),                   # wrong axis
    (4.0, (0, 0, -1), False),                  # reversed sense
    (2.0, (0, 0, 1), False),                   # half rate
])
def test_spin_check_accepts_only_the_right_rotation(engine_rate, engine_axis, should_pass):
    traj = _spin_bundle(4.0)
    run = {"samples": [{"released": True, "ct": 0.0, "pos": [0, 0, 600], "fwd": [1, 0, 0], "right": [0, -1, 0], "up": [0, 0, 1]}]
           + _spin_samples(engine_rate, engine_axis), "released_sim_t": 0.42}
    rows, fails = [], []
    V.check_angular(run, "z", traj, rows, fails, rest_z=0.0)
    assert bool(fails) != should_pass, (rows, fails)


def test_expected_omega_matches_the_construction():
    w = V.expected_omega_sigma(_spin_bundle(4.0), 0.42)
    assert w == pytest.approx((0, 0, 4.0), abs=1e-9)
    assert V.expected_omega_sigma(_spin_bundle(4.0), 99.0) == pytest.approx((0, 0, 4.0), abs=1e-9)  # last segment continues
