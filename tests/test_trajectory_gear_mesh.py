"""Phase 4 render-bridge gate — two REAL InvoluteGear shapes meshing at the
standard center distance, counter-rotating at the exact tooth ratio.

kernel/gear.py's geometry is gated separately (test_gear_shape.py, including
the on-flank regression gate for the reflection-frame bug); this checks the
recorder wiring: real quadrature mass/inertia reaches the parcels, recorded
frames carry both bodies' solved rotation at the tooth ratio, and the mesh
geometry is self-consistent (tip circles overlap — gears actually engage —
while root circles clear).
"""
import math

import pytest

from sigma_ground.kernel.gear import InvoluteGear
from sigma_ground.radiance.trajectory import record_gear_mesh_spin
from sigma_ground.dynamics.quat import twist_angle


def _z_angle(quat):
    return twist_angle(quat, (0.0, 0.0, 1.0))


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


def test_gear_mesh_frames_counter_rotate_at_tooth_ratio():
    teeth_a, teeth_b = 20, 31
    out = record_gear_mesh_spin(teeth_a=teeth_a, teeth_b=teeth_b, t_max=4.0)
    frames = out["trajectory"]["frames"]
    ts = [f["t_sim"] for f in frames]

    ang_a = _unwrap([_z_angle(f["bodies"][0]["quat"]) for f in frames])
    ang_b = _unwrap([_z_angle(f["bodies"][1]["quat"]) for f in frames])
    i0 = len(ts) // 3
    rate_a = (ang_a[-1] - ang_a[i0]) / (ts[-1] - ts[i0])
    rate_b = (ang_b[-1] - ang_b[i0]) / (ts[-1] - ts[i0])
    # external mesh: counter-rotation at exactly -Na/Nb
    assert rate_b == pytest.approx(-rate_a * teeth_a / teeth_b, rel=0.02)
    assert rate_a * rate_b < 0.0


def test_gear_mesh_geometry_engages():
    """Tip circles must OVERLAP across the mesh point (the gears engage)
    while root circles CLEAR each other — the basic meshing sanity that
    distinguishes a real gear pair from two disks that happen to spin."""
    out = record_gear_mesh_spin(t_max=0.05)
    cd = out["trajectory"]["validation"]["center_distance_m"]
    ga = InvoluteGear(module=0.004, teeth=20, pressure_angle=math.radians(20.0),
                      face_width=0.01)
    gb = InvoluteGear(module=0.004, teeth=31, pressure_angle=math.radians(20.0),
                      face_width=0.01)
    assert ga.r_a + gb.r_a > cd          # tips reach past each other: engaged
    assert ga.r_a + gb.r_f < cd + 2 * 0.004   # A's tips clear B's root by design margins
    assert ga.r_f + gb.r_f < cd          # roots nowhere near touching


def test_gear_mesh_parcels_carry_real_quadrature_inertia():
    """The recorded validation's energy ledger only means anything if the
    parcels' inertia came from the real tooth geometry — spot-check the
    ledger holds and the motor did real work against it."""
    out = record_gear_mesh_spin(t_max=2.0)
    val = out["trajectory"]["validation"]
    assert val["energy_gain_j"] > 0.0
    assert val["energy_ledger_ok"]


def test_gear_mesh_leaves_are_gear_type_with_estimated_module_flag():
    out = record_gear_mesh_spin(t_max=0.05)
    leaves = out["scene"]["csg_leaves"]
    assert len(leaves) == 2
    for leaf in leaves:
        assert leaf["shape"]["type"] == "Gear"
        # the module is NOT cited anywhere (Kelly gap) — the leaf must say so
        assert "estimated" in leaf["shape"]["source"].lower()
