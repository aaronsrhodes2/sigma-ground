"""Analyse the angular-velocity engine probe (stage/gmod/addon/lua/mentat/probe.lua).

For each setter the probe applied one vector (0, 0, 360) to a body tilted to a known
orientation and recorded the entity's forward/right/up each tick. From consecutive
orientations this script recovers the rotation the ENGINE actually performed:

    world-frame rate (rad/s) and axis      -> units and frame of the setter

Interpretation of the printed table
    rate ~ 6.283 rad/s, axis ~ (0,0,1)        : degrees/s, WORLD frame        (what server.lua assumes)
    rate ~ 6.283 rad/s, axis ~ entity up      : degrees/s, LOCAL frame
    rate ~ 360   rad/s                        : radians/s
    rate ~ 0 after the first tick             : the setter is a one-shot impulse the solver then damped

    python stage/gmod/verify_probe.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DATA = Path("D:/Steam/steamapps/common/GarrysMod/garrysmod/data/mentat/probe_angvel.json")


def mat(f, r, u):
    """Columns forward, left, up (GMod GetRight is the entity's right = minus left)."""
    left = [-c for c in r]
    return [[f[0], left[0], u[0]], [f[1], left[1], u[1]], [f[2], left[2], u[2]]]


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def tr(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def axis_angle(R):
    c = max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-9:
        return (0.0, 0.0, 1.0), 0.0
    s = 2 * math.sin(ang)
    ax = ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s)
    n = math.sqrt(sum(a * a for a in ax))
    return tuple(a / n for a in ax), ang


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main() -> int:
    if not DATA.exists():
        print(f"no probe file: {DATA}")
        return 2
    d = json.loads(DATA.read_text(encoding="utf-8"))
    if d.get("error"):
        print("probe error:", d["error"])
        return 1
    print(f"tick {d['tick']:.5f}s  spin vector {d['spin']}  tilt (pitch,yaw,roll) {d['tilt']}")
    print("rotation the engine performed, recovered from the recorded entity axes (world-frame increment per tick):")
    for c in d["cases"]:
        S = c["samples"]
        rates, axes = [], []
        for a, b in zip(S, S[1:]):
            dt = b["ct"] - a["ct"]
            if dt <= 0:
                continue
            Ra, Rb = mat(a["fwd"], a["right"], a["up"]), mat(b["fwd"], b["right"], b["up"])
            ax, ang = axis_angle(mul(Rb, tr(Ra)))               # world-frame increment
            rates.append(ang / dt)
            axes.append(ax)
        if not rates:
            continue
        # skip the first increment: the setter may act before/after the first sample
        use = slice(1, None) if len(rates) > 2 else slice(0, None)
        rate = sum(rates[use]) / len(rates[use])
        ax = tuple(sum(a[i] for a in axes[use]) / len(axes[use]) for i in range(3))
        n = math.sqrt(sum(x * x for x in ax)) or 1.0
        ax = tuple(x / n for x in ax)
        f0, r0, u0 = S[0]["fwd"], S[0]["right"], S[0]["up"]
        left0 = [-c for c in r0]
        rb = c["readback"]
        print(f"{c['name']:30s} rate {math.degrees(rate):8.2f} deg/s  axis(world) {tuple(round(x, 3) for x in ax)}"
              f"   signed dot with ent fwd {dot(ax, f0):+.3f}  left {dot(ax, left0):+.3f}  up {dot(ax, u0):+.3f}"
              f"   | vs world Z {dot(ax, (0, 0, 1)):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
