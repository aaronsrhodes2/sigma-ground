"""Windmill spin-up recorder gates — the first fully NATURAL drive bundle
(plugs: [] and it means it). The wind model itself is gated against closed
forms in tests/test_mechanism_wind.py; this checks the recorder wiring and
the provenance surfaces.
"""
import math

import pytest

from sigma_ground.radiance.trajectory import record_windmill_spinup
from sigma_ground.dynamics.quat import twist_angle


def test_windmill_spins_up_monotonically_toward_terminal():
    out = record_windmill_spinup(t_max=6.0)
    val = out["trajectory"]["validation"]
    trace = val["omega_trace_rad_s"]
    w_star = val["terminal_omega_expected_rad_s"]
    assert len(trace) > 50
    assert all(b >= a - 1e-6 for a, b in zip(trace, trace[1:]))   # monotone
    assert all(w <= w_star * 1.01 for w in trace)                  # never overshoots
    assert trace[-1] > 0.4 * w_star                                # really moving


def test_windmill_frames_carry_the_rotation():
    out = record_windmill_spinup(t_max=4.0)
    frames = out["trajectory"]["frames"]
    # twist about +x (the rotor axis) must advance across the recording
    a0 = twist_angle(frames[0]["bodies"][0]["quat"], (1, 0, 0))
    a_mid = twist_angle(frames[len(frames) // 2]["bodies"][0]["quat"], (1, 0, 0))
    assert a0 != a_mid


def test_windmill_declares_natural_drive_and_bearing_honesty():
    out = record_windmill_spinup(t_max=4.0)
    scene = out["scene"]
    val = out["trajectory"]["validation"]
    # NATURAL drive: no drive-kind plug — but the HOLDER is honestly cited
    # (ideal bearing atop a rendered-but-not-simulated mast = support plug)
    kinds = [p["kind"] for p in scene["plugs"]]
    assert "drive" not in kinds
    assert kinds == ["support"]
    assert "NATURAL" in scene["source"]
    # adjustable-by-default for cited artificial/explicit variables: the
    # wind sweep is exposed as bundle variants (frozen-run parameter sweep)
    assert {v["slug"] for v in scene["variants"]} == {"windmill_5ms",
                                                      "windmill_10ms"}
    # the ideal bearing's absorbed energy stays negligible vs the spin energy
    assert val["bearing_absorbed_energy_j"] < 1e-6 * max(val["spin_energy_j"], 1e-9)
    # blades render as Rotated boxes riding the rotor body
    rotated = [l for l in scene["csg_leaves"]
              if l["shape"]["type"] == "Rotated" and l.get("body") == 0]
    assert len(rotated) >= 4 + 1                       # blades + hub
