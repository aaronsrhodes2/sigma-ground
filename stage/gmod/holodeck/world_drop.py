"""Live check of the construct's physics: real drops in the engine, graded against the sigma-ground library's real-world physics.

    python stage/gmod/holodeck/world_drop.py [--out FILE] [--height 3] [--layouts "a pile of crates" "a table with six chairs"]
    python stage/gmod/holodeck/world_drop.py --grade FILE      # grade a saved result again, no game

Runs the headless job mentat/tools/world_drop.lua on the dedicated server (no window, no mouse), then grades:
  * fall time and impact speed of a crate, a ball and a light plate at the Earth and Moon presets, in three modes:
      real    the construct: gravity + the real F = 1/2 rho Cd A v^2 (world.lua), Source's own drag off
      vacuum  gravity only (the engine's integration by itself)
      engine  Source's own air drag on (what the game does by default)
  * expectations come from the library: g from the preset, rho from atmosphere.air_density, the force from projectile.drag_force,
    integrated with the library's own PhysicsScene stepper (the same one materia.engine.simulate_drag_run uses); with no drag the
    library's mcp free_fall_time / free_fall_velocity.
  * whether a pile of crates and a table with six chairs still settle and sleep without drifting under the new constants.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

HU_M = 0.01905


def library_fall(mass, cd, area, g, h, rho, dt=0.0005):
    """Fall time and impact speed from rest through still air, by the library's stepper and its drag_force."""
    from sigma_ground.dynamics.vec import Vec3
    from sigma_ground.dynamics.parcel import PhysicsParcel
    from sigma_ground.dynamics.scene import PhysicsScene
    from sigma_ground.dynamics.stepper import step
    from sigma_ground.materia.engine import _DensityMaterial
    from sigma_ground.shapes import Sphere
    from sigma_ground.field.interface.projectile import drag_force
    if cd is None or area is None or rho is None or rho == 0:
        from sigma_ground.mcp.tools.kinematics import free_fall_time, free_fall_velocity
        return free_fall_time(h, g).value, free_fall_velocity(h, g).value
    parcel = PhysicsParcel(Sphere(math.sqrt(area / math.pi)), _DensityMaterial(1.0), mass=mass,
                           position=Vec3(0.0, h, 0.0), velocity=Vec3(0.0, 0.0, 0.0), label="body")
    scene = PhysicsScene([parcel], gravity=Vec3(0.0, -g, 0.0), ground=False)

    def cb(p):
        v = p.velocity
        sp = v.length()
        if sp < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return v * (-drag_force(sp, cd, area, rho) / sp)

    prev_y, prev_t, prev_v = h, 0.0, 0.0
    while scene.time < 60.0:
        step(scene, dt=dt, sub_steps=1, external_forces=cb)
        y = parcel.position.y
        if y <= 0.0:
            frac = prev_y / (prev_y - y) if prev_y != y else 1.0           # interpolate the crossing
            return prev_t + frac * (scene.time - prev_t), prev_v + frac * (parcel.velocity.length() - prev_v)
        prev_y, prev_t, prev_v = y, scene.time, parcel.velocity.length()
    return float("nan"), float("nan")


def area_down(tr, obb_hu):
    """Projected area (m^2) of the object falling flat, from the world's own record of it: its sphere circle or its footprint."""
    d = tr["dims_m"]
    d = [d[str(i)] if isinstance(d, dict) else d[i - 1] for i in (1, 2, 3)] if isinstance(d, dict) else d
    if tr["class"] == "sphere":
        dm = sum(d) / 3
        return math.pi / 4 * dm * dm
    return d[0] * d[1]


def measure(row):
    """Fall time (s), impact speed (m/s) and fall distance (m) the engine produced, from the per-tick samples.

    The impact is the first tick whose velocity jumps UP (a bounce) or whose height reaches the resting height. A height test alone
    misses a bouncing ball: it touches the floor between two ticks, is already kicked back up on the next sample, and never
    comes within a fixed tolerance of the final resting height before it rises again (it then looked like a 1.16 s fall).
    The last tick before the impact gives speed and acceleration; the rest of the distance to the resting height is crossed at those.
    """
    s = row["samples"]
    dt = row["dt"]
    z0, zr = row["z_start"], row["z_rest"]
    tol = 0.5                                              # HU above the resting height: the collision skin is about 0.4
    for i in range(1, len(s)):
        n, z, vz = s[i]
        bounced = vz > s[i - 1][2] + 1.0
        if bounced or z <= zr + tol:
            j = i - 1 if bounced else i - 1
            n_j, z_j, v_j = s[j]
            if bounced:
                a = (v_j - s[j - 1][2]) / dt if j > 0 else (v_j / dt)     # HU/s^2, negative
                rem = max(0.0, z_j - zr)
                sp = -v_j
                acc = -a
                # solve rem = sp*t + acc*t^2/2
                t_rem = rem / sp if acc <= 0 or sp <= 0 else (-sp + math.sqrt(sp * sp + 2 * acc * rem)) / acc
                t = (n_j - 1) * dt + t_rem
                v = (sp + acc * t_rem) * HU_M
                return t, v, (z0 - zr) * HU_M
            n0, z_a, v_a = s[i - 1]
            frac = (z_a - (zr + tol)) / (z_a - z) if z_a != z else 1.0
            t = (n0 - 1 + frac) * dt                       # sample 1 is the first tick after the release
            return t, -s[i - 1][2] * HU_M, (z0 - zr) * HU_M
    return None


def grade(res):
    from sigma_ground.field.interface import atmosphere as atm
    rho = atm.air_density(atm._T_STANDARD, atm._P_STANDARD)
    out = []
    for r in res["drops"]:
        r = dict(r, dt=res["dt"])
        if r.get("error"):
            out.append({"error": r["error"], **{k: r.get(k) for k in ("preset", "mode", "name")}})
            continue
        m = measure(r)
        d = json.loads((HERE.parent / "world_constants.json").read_text(encoding="utf-8"))
        g = d["presets"][r["preset"]]["gravity_m_s2"]
        if m is None:
            out.append({"preset": r["preset"], "mode": r["mode"], "name": r["name"], "error": "no impact in samples"})
            continue
        t, v, h = m
        tr = r.get("tracked")
        cd = tr["cd"] if tr else None
        area = area_down(tr, r["obb_hu"]) if tr else None
        if r["mode"] == "real":
            et, ev = library_fall(r["mass_kg"], cd, area, g, h, rho)
        else:
            et, ev = library_fall(r["mass_kg"], None, None, g, h, None)
        row = {"preset": r["preset"], "mode": r["mode"], "name": r["name"], "mass_kg": round(r["mass_kg"], 4), "class": tr and tr["class"],
               "cd": cd, "area_m2": area and round(area, 5), "drop_m": round(h, 4), "g_in_force_m_s2": r["gravity_hu"] * HU_M,
               "g_library_m_s2": g, "measured_t_s": round(t, 4), "expected_t_s": round(et, 4), "t_err_pct": round(100 * (t - et) / et, 2),
               "measured_v_m_s": round(v, 3), "expected_v_m_s": round(ev, 3), "v_err_pct": round(100 * (v - ev) / ev, 2),
               "engine_drag_on": r.get("engine_drag_enabled")}
        if area and cd:
            from sigma_ground.field.interface.projectile import terminal_velocity
            row["terminal_m_s"] = round(terminal_velocity(r["mass_kg"], cd, area, rho, g), 2)
        out.append(row)
    return out


def grade_layouts(res):
    out = []
    for L in res["layouts"]:
        snaps = L["snaps"]
        last, prev = snaps[-1], snaps[-2] if len(snaps) > 1 else None
        drift = None
        if prev:
            a = {x["id"]: x["pos"] for x in prev["rows"]}
            drift = max((math.dist(a[x["id"]], x["pos"]) for x in last["rows"] if x["id"] in a), default=0) * HU_M
        out.append({"text": L["text"], "preset": "engine default (construct off)" if L.get("control") else L["preset"],
                    "count": last["count"], "asleep_at_end": last["asleep"], "moving_at_end": last["moving"],
                    "asleep_by_t": [(s["t"], s["asleep"]) for s in snaps],
                    "drift_last_interval_m": None if drift is None else round(drift, 4)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "world_drop_result.json"))
    ap.add_argument("--grade")
    ap.add_argument("--height", type=float, default=3.0)
    ap.add_argument("--layouts", nargs="*", default=["a pile of crates", "a table with six chairs"])
    ap.add_argument("--timeout", type=int, default=420)
    a = ap.parse_args(argv)
    if a.grade:
        res = json.loads(Path(a.grade).read_text(encoding="utf-8"))
    else:
        import run_tool
        res = run_tool.run_tool("world_drop", {"height_m": a.height, "layouts": a.layouts}, map_name="gm_flatgrass", timeout=a.timeout)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res), encoding="utf-8")
        print("wrote", a.out)
    print("engine:", json.dumps(res["engine"]))
    rows = grade(res)
    cols = ["preset", "mode", "name", "class", "cd", "mass_kg", "drop_m", "g_in_force_m_s2", "measured_t_s", "expected_t_s", "t_err_pct",
            "measured_v_m_s", "expected_v_m_s", "v_err_pct", "terminal_m_s", "error"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))
    print(json.dumps(grade_layouts(res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
