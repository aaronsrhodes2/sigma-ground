"""Quaternion kernel — orientation math for rigid bodies, tier 1.

LIFTED VERBATIM from ``radiance/trajectory.py`` (the proven tumble-recorder
math; that module now aliases these) and extended with the constraint
solver's needs. Layout is ``[x, y, z, w]`` — the viewer's convention — and
angular velocity is a world-frame 3-sequence. Pure functions on plain
sequences; no numpy (dynamics stays import-clean).

``quat_step`` integrates q̇ = ½·(ω⊗q): one explicit step + renormalize. Used
at the DRIFT stage with the half-kicked ω, it is the angular member of the
KDK leapfrog — O(dt²) like the position drift. Between impulses bodies are
torque-free; the gyroscopic ω×(Iω) term is handled (or flag-omitted) by the
stepper, not here.
"""
from __future__ import annotations

import math


def qrot(q, v):
    """Rotate v by quaternion q=(x,y,z,w) — the viewer's forward convention."""
    ux, uy, uz, w = q[0], q[1], q[2], q[3]
    tx = 2.0 * (uy * v[2] - uz * v[1])
    ty = 2.0 * (uz * v[0] - ux * v[2])
    tz = 2.0 * (ux * v[1] - uy * v[0])
    return (v[0] + w * tx + uy * tz - uz * ty,
            v[1] + w * ty + uz * tx - ux * tz,
            v[2] + w * tz + ux * ty - uy * tx)


def qrot_inv(q, v):
    """Rotate v by q⁻¹ (the conjugate, for unit q)."""
    return qrot((-q[0], -q[1], -q[2], q[3]), v)


def quat_step(q, w, dt):
    """q ← normalize(q + ½·(ω⊗q)·dt), ω in WORLD frame, layout (x,y,z,w)."""
    wx, wy, wz = w[0], w[1], w[2]
    dx = 0.5 * (wx * q[3] + wy * q[2] - wz * q[1])
    dy = 0.5 * (-wx * q[2] + wy * q[3] + wz * q[0])
    dz = 0.5 * (wx * q[1] - wy * q[0] + wz * q[3])
    dw = 0.5 * (-wx * q[0] - wy * q[1] - wz * q[2])
    out = [q[0] + dx * dt, q[1] + dy * dt, q[2] + dz * dt, q[3] + dw * dt]
    n = (out[0] ** 2 + out[1] ** 2 + out[2] ** 2 + out[3] ** 2) ** 0.5 or 1.0
    return [v / n for v in out]


def quat_mul(a, b):
    """Hamilton product a⊗b (apply b's rotation, then a's), layout (x,y,z,w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_normalize(q):
    n = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) or 1.0
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_from_axis_angle(axis, angle_rad):
    """Unit quaternion rotating by ``angle_rad`` about ``axis`` (normalized here)."""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    s = math.sin(0.5 * angle_rad) / n
    return (ax * s, ay * s, az * s, math.cos(0.5 * angle_rad))


def twist_angle(q_rel, axis):
    """The rotation of ``q_rel`` ABOUT ``axis`` (the swing-twist decomposition's
    twist), in (−π, π]. The joint-limit measure for a revolute: how far the
    relative orientation has turned around the hinge line."""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ax, ay, az = ax / n, ay / n, az / n
    proj = q_rel[0] * ax + q_rel[1] * ay + q_rel[2] * az   # q.xyz · axis
    ang = 2.0 * math.atan2(proj, q_rel[3])
    if ang > math.pi:
        ang -= 2.0 * math.pi
    elif ang <= -math.pi:
        ang += 2.0 * math.pi
    return ang


def solve3(K, b):
    """Solve the 3×3 system K·x = b by the cofactor (adjugate) inverse — the
    block solve for ball/angular-lock constraint rows. K is a row-major 3×3
    (sequence of 3 rows). Near-singular K (a degenerate constraint block)
    falls back to the least-norm diagonal solve, flagged by returning it
    anyway: NOT_PHYSICS — a numerical guard, the solver iterates it out."""
    (a11, a12, a13), (a21, a22, a23), (a31, a32, a33) = K
    c11 = a22 * a33 - a23 * a32
    c12 = a13 * a32 - a12 * a33
    c13 = a12 * a23 - a13 * a22
    det = a11 * c11 + a21 * c12 + a31 * c13
    if abs(det) < 1e-18:
        return (b[0] / a11 if abs(a11) > 1e-18 else 0.0,
                b[1] / a22 if abs(a22) > 1e-18 else 0.0,
                b[2] / a33 if abs(a33) > 1e-18 else 0.0)
    c21 = a23 * a31 - a21 * a33
    c22 = a11 * a33 - a13 * a31
    c23 = a13 * a21 - a11 * a23
    c31 = a21 * a32 - a22 * a31
    c32 = a12 * a31 - a11 * a32
    c33 = a11 * a22 - a12 * a21
    inv = 1.0 / det
    return ((c11 * b[0] + c12 * b[1] + c13 * b[2]) * inv,
            (c21 * b[0] + c22 * b[1] + c23 * b[2]) * inv,
            (c31 * b[0] + c32 * b[1] + c33 * b[2]) * inv)


__all__ = ["qrot", "qrot_inv", "quat_step", "quat_mul", "quat_conj",
           "quat_normalize", "quat_from_axis_angle", "twist_angle", "solve3"]
