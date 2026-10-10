"""Offline laboratory for the holodeck: run the real Lua parser and planner (LuaJIT through lupa) in synthetic worlds, and
grade what comes out with an INDEPENDENT checker that never trusts the planner's own footprints.

    from lab import Lab
    lab = Lab()
    res = lab.run("add three couches", world="open")
    problems = lab.grade(res)

The grader rebuilds every footprint from the catalog's engine-measured bounds, the entity origin and yaw the planner chose,
so a wrong front, a wrong footprint centre or a wrong floor offset shows up as an overlap, a floor gap or a thing facing the
wrong way.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from eval_parse import runtime, to_py  # noqa: E402

HERE = Path(__file__).resolve().parent
LUA_ROOT = HERE.parent / "addon" / "lua"


def rot(x: float, y: float, deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)


def poly_of(variant: dict, origin: dict, yaw: float) -> list[tuple[float, float]]:
    """The model's measured bounding box, turned by the entity yaw and moved to its origin."""
    mn, mx = variant["mins"], variant["maxs"]
    out = []
    for x, y in ((mn[0], mn[1]), (mx[0], mn[1]), (mx[0], mx[1]), (mn[0], mx[1])):
        rx, ry = rot(x, y, yaw)
        out.append((origin["x"] + rx, origin["y"] + ry))
    return out


def sat_penetration(a: list, b: list) -> float:
    """Depth of overlap of two convex polygons along the best separating axis; 0 when apart."""
    best = math.inf
    for poly in (a, b):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            ax, ay = -(y2 - y1), x2 - x1
            n = math.hypot(ax, ay)
            if n == 0:
                continue
            ax, ay = ax / n, ay / n
            pa = [x * ax + y * ay for x, y in a]
            pb = [x * ax + y * ay for x, y in b]
            overlap = min(max(pa), max(pb)) - max(min(pa), min(pb))
            if overlap <= 0:
                return 0.0
            best = min(best, overlap)
    return best


def angle_diff(a: float, b: float) -> float:
    d = (a - b) % 360
    return d - 360 if d > 180 else d


class Lab:
    def __init__(self):
        self.rt, self.mod = runtime()
        root = LUA_ROOT.as_posix()
        self.rt.execute(f'''
            local ROOT = "{root}"
            function include(path) return dofile(ROOT .. "/" .. path) end
        ''')
        self.load = self.rt.eval("function(p) return dofile(p) end")
        hd = (LUA_ROOT / "mentat" / "holodeck").as_posix()
        self.catalog_lua = self.load(f"{hd}/catalog.lua")
        self.Parse = self.load(f"{hd}/parse.lua")
        self.Plan = self.load(f"{hd}/plan.lua")
        self.Layout = self.load(f"{hd}/layout.lua")
        self.testenv = self.load((HERE / "testenv.lua").as_posix())
        self.parser = self.Parse.new(self.catalog_lua)
        self.planner = self.Plan.new(self.catalog_lua)
        self.catalog = json.loads((HERE / "data" / "catalog.json").read_text(encoding="utf-8"))
        self.registry: list[dict] = []
        self.next_id = 1
        self.seq = 0

    # ---- worlds
    def world(self, name: str):
        if name == "open":
            return self.testenv.open()
        if name.startswith("room"):
            # room, room:WxD  (HU), a 560 x 480 room by default (about 10.7 x 9.1 m)
            w, d = (560, 480)
            if ":" in name:
                w, d = (int(v) for v in name.split(":")[1].split("x"))
            return self.testenv.room(-w / 2, -d / 2, w / 2, d / 2)
        raise ValueError(name)

    def lua_table(self, obj):
        """Python -> Lua table (lists and dicts, recursively)."""
        if isinstance(obj, dict):
            t = self.rt.table()
            for k, v in obj.items():
                t[k] = self.lua_table(v)
            return t
        if isinstance(obj, (list, tuple)):
            t = self.rt.table()
            for i, v in enumerate(obj, 1):
                t[i] = self.lua_table(v)
            return t
        return obj

    # ---- run
    def run(self, text: str, world="open", pos=(0.0, 0.0, 0.0), yaw=0.0, aim=None, seed=None, wake=False):
        env = self.world(world) if isinstance(world, str) else world
        if wake:
            ok, rest = self.Parse.strip_wake(text)
            if not ok:
                return {"action": "none", "placements": [], "notes": [], "say": "", "request": None}
            text = rest
        req = self.parser.parse(self.parser, text)
        ctx = self.rt.table()
        ctx["pos"] = self.lua_table({"x": pos[0], "y": pos[1], "z": pos[2]})
        ctx["yaw"] = yaw
        ctx["env"] = env
        if aim is not None:
            ctx["aim"] = self.lua_table({"x": aim[0], "y": aim[1], "z": 0})
        ctx["seed"] = seed if seed is not None else 12345
        if getattr(self, "debug", None):
            ctx["debug"] = self.debug
        ctx["registry"] = self.lua_table(self.registry)
        res = self.planner.run(self.planner, req, ctx)
        out = to_py(self.mod, res)
        out["request"] = to_py(self.mod, req)
        return out

    def commit(self, res: dict) -> None:
        """Apply a plan to the registry the way the game does, so a following utterance sees it."""
        for p in res.get("placements", []):
            self.seq += 1
            self.registry.append({
                "id": self.next_id, "kind": p["kind"], "vi": p["vi"], "x": p["x"], "y": p["y"], "z": p["z"], "theta": p["theta"],
                "w": p["w"], "d": p["d"], "h": p["h"], "color_name": p.get("color_name"), "group": p.get("group", 0) + self.seq,
                "seq": self.seq,
            })
            self.next_id += 1
        removed = set(res.get("removals", []))
        if removed:
            self.registry = [e for e in self.registry if e["id"] not in removed]
        for m in res.get("moves", []):
            for e in self.registry:
                if e["id"] == m["id"]:
                    e["x"], e["y"], e["theta"] = m["x"], m["y"], m["theta"]
                    e["z"] = m.get("z", e["z"])
        if res.get("action") == "clear":
            self.registry = []

    # ---- the independent grader
    def variant(self, p: dict) -> dict:
        return self.catalog["kinds"][p["kind"]]["variants"][p["vi"] - 1]

    def grade(self, res: dict, world=None, tuck=14.0) -> list[str]:
        problems: list[str] = []
        pls = res.get("placements", [])
        polys = []
        for p in pls:
            v = self.variant(p)
            polys.append(poly_of(v, p["origin"], p["yaw"]))
        # floor gap: a thing on the floor has its bottom at 0, a thing on another stands on that thing's top
        for p, poly in zip(pls, polys):
            v = self.variant(p)
            bottom = p["origin"]["z"] + v["mins"][2]
            if not p.get("on"):
                if abs(bottom) > 0.6:
                    problems.append(f"{p['kind']} bottom is {bottom:.1f} HU off the floor")
            else:
                ok = False
                for q, qpoly in zip(pls, polys):
                    qv = self.variant(q)
                    if q is p:
                        continue
                    tops = [q["origin"]["z"] + qv["maxs"][2]]               # stacked on the whole thing
                    if qv.get("top"):
                        tops.append(q["origin"]["z"] + qv["mins"][2] + qv["top"]["z"])   # or on its flat top
                    if any(abs(bottom - t) < 1.0 for t in tops) and sat_penetration(poly, qpoly) > 0:
                        ok = True
                if not ok:
                    problems.append(f"{p['kind']} is said to stand on something but nothing's top matches (bottom {bottom:.1f})")
        # overlaps (floor things only; things on a top are inside their support by construction and checked above)
        for i in range(len(pls)):
            for j in range(i + 1, len(pls)):
                a, b = pls[i], pls[j]
                if a.get("on") or b.get("on"):
                    continue
                pen = sat_penetration(polys[i], polys[j])
                if pen <= 1.5:
                    continue
                tucked = ({a.get("role"), b.get("role")} & {"around", "at_desk"}) and pen <= tuck
                if not tucked:
                    problems.append(f"{a['kind']}#{i} and {b['kind']}#{j} interpenetrate by {pen:.1f} HU")
        # the room
        if world is not None and hasattr(world, "bounds"):
            x0, y0, x1, y1 = (world.bounds[k] for k in (1, 2, 3, 4))
            for p, poly in zip(pls, polys):
                for x, y in poly:
                    if x < x0 - 0.7 or x > x1 + 0.7 or y < y0 - 0.7 or y > y1 + 0.7:
                        problems.append(f"{p['kind']} pokes through a wall at ({x:.0f},{y:.0f})")
                        break
        # things the world already holds
        for p, poly in zip(pls, polys):
            if p.get("on"):
                continue
            for e in self.registry:
                ev = self.catalog["kinds"][e["kind"]]["variants"][e["vi"] - 1]
                er = self._reg_poly(e)
                pen = sat_penetration(poly, er)
                if pen > 1.5:
                    problems.append(f"{p['kind']} overlaps standing {e['kind']} by {pen:.1f}")
        return problems

    def _reg_poly(self, e):
        # registry rows carry the footprint (w, d, theta); rebuild its polygon
        w, d, th = e["w"], e["d"], e["theta"]
        pts = []
        for u, v in ((d / 2, w / 2), (d / 2, -w / 2), (-d / 2, -w / 2), (-d / 2, w / 2)):
            fx, fy = math.cos(math.radians(th)), math.sin(math.radians(th))
            rx, ry = fy, -fx
            pts.append((e["x"] + u * fx + v * rx, e["y"] + u * fy + v * ry))
        return pts

    def front_vec(self, p: dict) -> tuple[float, float]:
        v = self.variant(p)
        ang = p["yaw"] + v["front"]
        return math.cos(math.radians(ang)), math.sin(math.radians(ang))

    def facing_error(self, p: dict, tx: float, ty: float) -> float:
        """Angle (deg) between where the thing faces and the direction from it to (tx, ty)."""
        fx, fy = self.front_vec(p)
        want = math.degrees(math.atan2(ty - p["y"], tx - p["x"]))
        have = math.degrees(math.atan2(fy, fx))
        return abs(angle_diff(have, want))
