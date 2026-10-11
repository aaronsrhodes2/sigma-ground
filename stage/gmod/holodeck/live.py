"""Run the holodeck for real, headless, on the dedicated server, and grade what the PHYSICS ENGINE left standing.

    python stage/gmod/holodeck/live.py "add three couches" "a living room" [--map gm_flatgrass] [--pos 0,0[,z]] [--yaw 0] [--out r.json]
    (a negative position needs the equals sign: --pos=-100,0)

The grader is independent of the planner: footprints come from the engine-measured model bounds (data/catalog.json), turned by
the yaw the engine reports for each prop after it settled. It checks that every prop is still where it was put (nothing was
shoved), nothing fell through the floor, nothing overlaps, everything has gone to sleep, and that each kind of arrangement
really looks right (seats face what they are around, TVs are faced, tops carry their things).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_tool  # noqa: E402
from lab import poly_of, sat_penetration, angle_diff  # noqa: E402

CATALOG = json.loads((HERE / "data" / "catalog.json").read_text(encoding="utf-8"))


def variant(kind: str, vi: int) -> dict:
    return CATALOG["kinds"][kind]["variants"][vi - 1]


SEATS = ("chair", "office chair", "stool")
TUCK_HU = 14.0          # how far a seat may slide under a table (the planner tucks at most 9 HU; the grader allows a little more)


def has_table_top(e: dict) -> bool:
    """A body with a flat top high enough to slide a seat under: the planner's own condition for tucking (top at 28 HU or more)."""
    top = variant(e["kind"], e["vi"]).get("top")
    return bool(top) and top["z"] >= 28


def tucked(a: dict, b: dict, pen: float) -> bool:
    """A seat pushed under a table or desk overlaps it in plan view by design, and only that: not a seat against a seat, a couch or a wall."""
    return pen <= TUCK_HU and ((a["kind"] in SEATS and has_table_top(b)) or (b["kind"] in SEATS and has_table_top(a)))


def parse_pos(text: str) -> dict:
    """'x,y' or 'x,y,z' (Hammer units) as the position the run stands at; z is left to the floor unless given."""
    try:
        c = [float(v) for v in text.split(",")]
    except ValueError:
        c = []
    if len(c) not in (2, 3):
        raise argparse.ArgumentTypeError(f"{text!r}: expected x,y or x,y,z")
    return {"x": c[0], "y": c[1], **({"z": c[2]} if len(c) == 3 else {})}


def actual_poly(e: dict) -> list:
    v = variant(e["kind"], e["vi"])
    pos = e["actual"]["pos"]
    return poly_of(v, {"x": pos[0], "y": pos[1]}, e["actual"]["ang"][1])


def grade(report: dict, floor_z: float = 0.0, max_shove: float = 6.0) -> list[str]:
    """Problems found in the settled scene (empty = clean)."""
    problems: list[str] = []
    ents = [e for e in report["entities"]]
    for e in ents:
        if e.get("missing"):
            problems.append(f"{e['kind']}#{e['id']} never appeared")
    live = [e for e in ents if not e.get("missing")]
    for e in live:
        v = variant(e["kind"], e["vi"])
        pos, ang = e["actual"]["pos"], e["actual"]["ang"]
        pl = e["planned"]["origin"]
        shove = math.dist((pos[0], pos[1]), (pl["x"], pl["y"]))
        drop = abs(pos[2] - pl["z"])
        if shove > max_shove:
            problems.append(f"{e['kind']}#{e['id']} was shoved {shove:.1f} HU from where it was put")
        if drop > 8:
            problems.append(f"{e['kind']}#{e['id']} ended {drop:.1f} HU off its planned height")
        if abs(ang[0]) > 12 or abs(ang[2]) > 12:
            problems.append(f"{e['kind']}#{e['id']} toppled (pitch {ang[0]:.0f}, roll {ang[2]:.0f})")
        if pos[2] + v["mins"][2] < floor_z - 3:
            problems.append(f"{e['kind']}#{e['id']} sank {floor_z - (pos[2] + v['mins'][2]):.1f} HU into the floor")
        if e.get("in_wall", 0) > 0:
            problems.append(f"{e['kind']}#{e['id']} has {e['in_wall']} bounding-box corners inside the world (through a wall)")
        if e.get("speed", 0) > 15:
            problems.append(f"{e['kind']}#{e['id']} is still moving ({e['speed']:.0f} HU/s)")
    polys = [actual_poly(e) for e in live]
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a, b = live[i], live[j]
            pen = sat_penetration(polys[i], polys[j])
            if pen <= 2.0:
                continue
            # a prop standing on another one (planned z above the floor) overlaps it in plan view by design
            za = a["planned"]["z"] - floor_z
            zb = b["planned"]["z"] - floor_z
            if za > 1 or zb > 1:
                continue
            if tucked(a, b, pen):
                continue                                           # a chair tucked under a table
            problems.append(f"{a['kind']}#{a['id']} and {b['kind']}#{b['id']} interpenetrate by {pen:.1f} HU")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("requests", nargs="+")
    ap.add_argument("--map", default="gm_flatgrass")
    ap.add_argument("--pos", type=parse_pos, default="0,0", help="x,y[,z] to stand at")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle", type=float, default=7.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--start-bridge", action="store_true", help="run the Skippy bridge for this run (agent, persona, voice)")
    ap.add_argument("--bridge", default="", help="URL of a running Skippy bridge")
    ap.add_argument("--persona", action="store_true")
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--agent", type=int, default=1)
    ap.add_argument("--content", default="", help="all | hl2 (only the models every player has)")
    ap.add_argument("--engine-mass", action="store_true", help="keep the engine's own gameplay masses instead of the catalog's real-world masses")
    a = ap.parse_args(argv)
    targs = {"requests": a.requests, "pos": a.pos, "yaw": a.yaw, "seed": a.seed, "settle": a.settle}
    srv = None
    if a.start_bridge:
        import secrets
        import skippy_bridge
        skippy_bridge.TOKEN = secrets.token_hex(8)
        srv = skippy_bridge.serve("127.0.0.1", 8777)
        a.bridge = "http://127.0.0.1:8777"
        targs["token"] = skippy_bridge.TOKEN
    if a.content:
        targs["content"] = a.content
    if a.engine_mass:
        targs["real_mass"] = False
    if a.bridge:
        targs.update({"bridge": a.bridge, "persona": a.persona, "voice": a.voice, "agent": a.agent})
        targs["timeout_note"] = "bridge on"
    try:
        res = run_tool.run_tool("holodeck_run", targs, map_name=a.map, timeout=300)
    finally:
        if srv:
            srv.shutdown()
            srv.server_close()
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    floor_z = res["pos"]["z"]
    print(f"floor at z={floor_z:.1f}; {res['count']} props standing")
    if res.get("library"):
        lib = res["library"]
        print(f"  library: {lib['kept']} of {lib['total']} models, {lib['kinds']} kinds" + (f", no model for {lib['lost_kinds']}" if lib["lost_kinds"] else ""))
    for s in res["steps"]:
        print(f"  {s['text']!r}: {s.get('action')} x{s.get('placed')}{' [agent]' if s.get('by_agent') else ''}: {s.get('say')}"
              + (f"  | said: {s['said']}" if s.get('said') else ""))
    if res.get("last_audio"):
        print(f"  last voice: {res['last_audio']['bytes']} bytes in {res['last_audio']['chunks']} chunk(s), made by {res.get('last_voice')}")
    for k in ("agent_error", "speak_error"):
        if res.get(k):
            print(f"  {k}: {res[k]}")
    problems = grade(res, floor_z=floor_z)
    asleep = sum(1 for e in res["entities"] if e.get("asleep"))
    print(f"{asleep}/{res['count']} asleep after settling")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("clean: nothing shoved, sunk, toppled or overlapping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
