"""Live check of the construct's water: real bodies in real water in the engine, graded against the sigma-ground library.

    python stage/gmod/holodeck/water_live.py [--out FILE] [--scenarios frame float sink splash moon mercury swim cost]
    python stage/gmod/holodeck/water_live.py --grade FILE      # grade a saved result again, no game

Runs the headless job mentat/tools/water_live.lua on the dedicated server (no window, no mouse), then grades:
  * a block of density 600 and one of 300 kg/m3 floating in water: the draft against the library's submerged_fraction (density ratio)
    and that they settle and sleep;
  * a steel block sinking: its speed plateau against the terminal speed from weight less buoyancy = 1/2 rho Cd A v^2 + Stokes
    (rho and viscosity from the library, Cd the construct's box value);
  * the splash energy the water sends against m g h of the drop;
  * the Moon (the draft is the same, the sinking speed goes with sqrt(g));
  * a steel block on mercury (floats at the density ratio);
  * a bot swimming (where it floats, how fast it swims against the thrust/drag balance);
  * what the water costs per physics tick with 40 bodies in it.
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
CD_BOX = 1.05
CONSTANTS = HERE.parent / "water_constants.json"


def lib():
    from sigma_ground.field.interface import fluid
    from sigma_ground.field.interface.fluid_surface import submerged_fraction
    from sigma_ground.field.interface.projectile import G_EARTH
    from sigma_ground.materia.bodies import resolve_body
    return fluid, submerged_fraction, G_EARTH, resolve_body("moon").surface_gravity_m_s2


def terminal(rho_b, V, rho_l, g, area, eta, cd=CD_BOX):
    """Speed at which weight less buoyancy equals 1/2 rho Cd A v^2 + Stokes (the library's drag_force for the quadratic part)."""
    from sigma_ground.field.interface.projectile import drag_force
    net = (rho_b - rho_l) * V * g
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        f = drag_force(mid, cd, area, rho_l) + 12 * eta * area * mid / math.sqrt(4 * area / math.pi)
        lo, hi = (mid, hi) if f < net else (lo, mid)
    return (lo + hi) / 2


def plateau(rows, floor_m, skip=0.35):
    """Mean downward speed over the steadiest stretch before the body reaches the bottom."""
    pts = [(r[0], -r[2]) for r in rows if r[1] > floor_m + 0.4]
    if len(pts) < 20:
        return None
    pts = pts[int(len(pts) * skip):]
    best, best_v = None, None
    w = 12
    for i in range(0, max(1, len(pts) - w)):
        seg = [v for _, v in pts[i:i + w]]
        spread = max(seg) - min(seg)
        if best is None or spread < best:
            best, best_v = spread, sum(seg) / len(seg)
    return best_v


def grade(res: dict) -> dict:
    fluid, subfrac, g_earth, g_moon = lib()
    water = fluid.KNOWN_LIQUIDS["water"]
    merc = fluid.KNOWN_LIQUIDS["mercury"]
    human = json.loads(CONSTANTS.read_text(encoding="utf-8"))["human"]
    rep = {"rows": []}
    g_pool = res["pool"]["g"] if res.get("pool") else g_earth

    def row(name, measured, expected, tol, unit="", note=""):
        err = None if expected in (None, 0) or measured is None else 100 * (measured - expected) / expected
        ok = measured is not None and expected is not None and (abs(measured - expected) <= tol * abs(expected))
        rep["rows"].append({"name": name, "measured": measured, "expected": expected, "err_pct": err and round(err, 2), "tol_pct": tol * 100,
                            "unit": unit, "ok": bool(ok), "note": note})

    if res.get("frame"):
        rep["angular_velocity_frame"] = res["frame"].get("kind")
    for d in res.get("drafts", []):
        exp = subfrac(d["rho"], water["density_kg_m3"]) * d["side"]
        row(f"draft of a {d['rho']} kg/m3 block", d["draft_m"], exp, 0.04, "m", f"asleep={d['asleep']} after {d['t_end']:.1f} s")
        rep["rows"].append({"name": f"{d['rho']} kg/m3 block sleeps", "measured": d["asleep"], "expected": True, "ok": bool(d["asleep"])})
    for s in res.get("sinks", []):
        V = s["side"] ** 3
        exp = terminal(7850.0, V, water["density_kg_m3"], g_pool, s["side"] ** 2, water["viscosity_pa_s"])
        row("steel block terminal speed (Earth)", plateau(s["rows"], s["floor_m"]), exp, 0.08, "m/s")
    ms = res.get("moon", {})
    if ms.get("draft"):
        d = ms["draft"]
        row("draft on the Moon (same as Earth)", d["draft_m"], subfrac(d["rho"], water["density_kg_m3"]) * d["side"], 0.05, "m",
            f"g={d['g']:.4f} asleep={d['asleep']} after {d['t_end']:.1f} s")
    if ms.get("sink"):
        s = ms["sink"]
        V = s["side"] ** 3
        exp = terminal(7850.0, V, water["density_kg_m3"], s["g"], s["side"] ** 2, water["viscosity_pa_s"])
        row("steel block terminal speed (Moon)", plateau(s["rows"], s["floor_m"]), exp, 0.10, "m/s", f"g={s['g']:.4f}")
    mc = res.get("mercury")
    if mc and mc.get("draft_m") is not None:
        row("steel on mercury: draft", mc["draft_m"], subfrac(mc["rho"], merc["density_kg_m3"]) * mc["side"], 0.05, "m", f"asleep={mc['asleep']}")
    for sp in res.get("splashes", []):
        ev = [e for e in sp["events"] if e["kind"] == 1]
        if not ev:
            rep["rows"].append({"name": f"splash from {sp['h_m']} m", "measured": None, "expected": "an entry event", "ok": False})
            continue
        e = ev[0]
        row(f"entry speed, drop of {sp['h_m']} m", e["speed"], math.sqrt(2 * g_pool * sp["h_m"]), 0.12, "m/s")
        row(f"entry energy, drop of {sp['h_m']} m", 0.5 * e["mass"] * e["speed"] ** 2, e["mass"] * g_pool * sp["h_m"], 0.2, "J")
    sw = res.get("swim", {})
    if sw.get("rest"):
        r = sw["rest"]
        hull = 72 * HU_M
        f = human["density_kg_m3"]["value"] / water["density_kg_m3"]
        row("swimmer at rest: feet below the surface", r["surf_m"] - r["feet_z_end"], f * hull, 0.15, "m", f"vz_end={r['vz_end']:.3f}")
    if sw.get("forward"):
        thrust = human["swim_thrust_n"]["value"]
        area = 32 * HU_M * 72 * HU_M
        lo, hi = 0.0, 20.0
        for _ in range(100):
            mid = (lo + hi) / 2
            fdrag = 0.5 * water["density_kg_m3"] * CD_BOX * area * mid ** 2 + 12 * water["viscosity_pa_s"] * area * mid / math.sqrt(4 * area / math.pi)
            lo, hi = (mid, hi) if fdrag < thrust else (lo, mid)
        row("swimmer's forward speed", sw["forward"]["vx_mean"], (lo + hi) / 2, 0.15, "m/s")
    if res.get("cost"):
        rep["cost"] = res["cost"]
    rep["notes"] = res.get("notes", [])
    rep["pass"] = all(r["ok"] for r in rep["rows"])
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "water_live.json"))
    ap.add_argument("--grade")
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--depth", type=float, default=12.0)
    a = ap.parse_args(argv)
    if a.grade:
        res = json.loads(Path(a.grade).read_text(encoding="utf-8"))
    else:
        import run_tool
        targs = {"depth_m": a.depth}
        if a.scenarios:
            targs["scenarios"] = a.scenarios
        res = run_tool.run_tool("water_live", targs, timeout=420)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    rep = grade(res)

    def fm(v):
        return f"{v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    for r in rep["rows"]:
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['name']:<45} measured {fm(r.get('measured')):>10}  expected {fm(r.get('expected')):>10}  "
              f"{r.get('err_pct') or ''}% {r.get('note', '')}")
    if rep.get("cost"):
        print("cost:", json.dumps(rep["cost"]))
    if rep.get("angular_velocity_frame"):
        print("AddAngleVelocity frame:", rep["angular_velocity_frame"])
    for n in rep["notes"]:
        print("note:", n)
    print("PASS" if rep["pass"] else "SOME FAILED")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
