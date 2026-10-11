"""Offline laboratory for the holodeck: run the real Lua parser and planner (LuaJIT through lupa) in synthetic worlds, and
grade what comes out with an INDEPENDENT checker that never trusts the planner's own footprints.

    from lab import Lab
    lab = Lab()
    res = lab.run("add three couches", world="open")
    problems = lab.grade(res)

    lab.commit(res, owner="alice")                        # what the game does with a plan: one group, one registry row a thing
    lab.run("remove it", owner="alice")                   # ctx.last, ctx.mine and the rest are filled the way holodeck.lua fills them
    lab.run_request({...}, world="room")                  # a hand-built request (the parser's output contract)
    lab.run_all("a couch and a table in front of it")    # a compound sentence: parse_all, each part planned after the last
    lab.audit(world)                                      # grade EVERYTHING standing (after moves and turns), not just a plan

The planner is handed the engine's own seed (a hash of the sentence) unless a test gives one: a small fixed seed always draws the
smallest model of every kind and hid most of what the review found. `budget` defaults to a generous 10 s (the engine's are 0.05 and
0.25); tests of the deadline pass their own, and `now=` a Lua clock. HOLODECK_LUA_ROOT, HOLODECK_PARSE and HOLODECK_PLAN load another
copy of the addon, the parser or the planner (to compare versions).

The grader rebuilds every footprint from the catalog's engine-measured bounds, the entity origin and yaw the planner chose,
so a wrong front, a wrong footprint centre or a wrong floor offset shows up as an overlap, a floor gap or a thing facing the
wrong way.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from eval_parse import runtime, to_py  # noqa: E402

HERE = Path(__file__).resolve().parent
SEATS = {"chair", "stool", "office chair", "bench", "armchair"}          # what may be tucked under a table...
TABLES = {"table", "desk", "meeting table", "patio table", "coffee table"}   # ...up to a few HU
LUA_ROOT = Path(os.environ.get("HOLODECK_LUA_ROOT") or HERE.parent / "addon" / "lua")      # a copy of the addon, to compare versions


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
    def __init__(self, parse_path=None, plan_path=None):
        """parse_path / plan_path: load the parser or planner from somewhere else (default: the addon's own, or $HOLODECK_PARSE /
        $HOLODECK_PLAN), to compare two versions of either."""
        self.rt, self.mod = runtime()
        root = LUA_ROOT.as_posix()
        self.rt.execute(f'''
            local ROOT = "{root}"
            function include(path) return dofile(ROOT .. "/" .. path) end
        ''')
        self.load = self.rt.eval("function(p) return dofile(p) end")
        hd = (LUA_ROOT / "mentat" / "holodeck").as_posix()
        self.catalog_lua = self.load(f"{hd}/catalog.lua")
        self.Parse = self.load(Path(parse_path or os.environ.get("HOLODECK_PARSE") or f"{hd}/parse.lua").as_posix())
        self.Plan = self.load(Path(plan_path or os.environ.get("HOLODECK_PLAN") or f"{hd}/plan.lua").as_posix())
        self.Layout = self.load(f"{hd}/layout.lua")
        self.testenv = self.load((HERE / "testenv.lua").as_posix())
        self.parser = self.Parse.new(self.catalog_lua)
        self.planner = self.Plan.new(self.catalog_lua)
        self.catalog = json.loads((HERE / "data" / "catalog.json").read_text(encoding="utf-8"))
        self.registry: list[dict] = []
        self.next_id = 1
        self.seq = 0
        self.gseq = 0                             # one group number per committed plan, as the game does
        # Tests are not about speed: the planner gets a generous allowance unless a test asks for the engine's own (0.05 / 0.25 s)
        self.default_budget = 10.0

    # ---- worlds
    def world(self, name: str):
        if name == "open":
            return self.testenv.open()
        if name.startswith("room"):
            # room, room:WxD, room:WxDxH  (HU), a 560 x 480 room by default (about 10.7 x 9.1 m); H is the ceiling height
            w, d, h = (560, 480, None)
            if ":" in name:
                parts = [int(v) for v in name.split(":")[1].split("x")]
                w, d = parts[0], parts[1]
                h = parts[2] if len(parts) > 2 else None
            return self.testenv.room(-w / 2, -d / 2, w / 2, d / 2, None, h)
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
    def last_ids(self, owner):
        """The ids of the requester's most recent group, as the game keeps them (H.last_group)."""
        mine = [e for e in self.registry if e.get("owner") == owner]
        if not mine:
            return []
        group = max(mine, key=lambda e: e["seq"])["group"]
        return [e["id"] for e in mine if e["group"] == group]

    def context(self, env, pos=(0.0, 0.0, 0.0), yaw=0.0, aim=None, seed=None, owner=None, budget=None, max_total=None,
                max_per_kind=None, players=None, last=None, now=None, restrict=None):
        """A planner context the way holodeck.lua builds it. seed=None means the engine's own (a hash of the sentence)."""
        ctx = self.rt.table()
        ctx["pos"] = self.lua_table({"x": pos[0], "y": pos[1], "z": pos[2]})
        ctx["yaw"] = yaw
        ctx["env"] = env
        if aim is not None:
            ctx["aim"] = self.lua_table({"x": aim[0], "y": aim[1], "z": 0})
        if seed is not None:
            ctx["seed"] = seed
        if getattr(self, "debug", None):
            ctx["debug"] = self.debug
        ctx["registry"] = self.lua_table(self.registry)
        restrict = (owner is not None) if restrict is None else restrict
        if restrict:                             # a player who may change only their own things
            ctx["mine"] = self.lua_table([e for e in self.registry if e.get("owner") == owner])
        ctx["last"] = self.lua_table(self.last_ids(owner) if last is None else list(last))
        ctx["budget"] = self.default_budget if budget is None else budget
        if max_total is not None:
            ctx["max_total"] = max_total
        if max_per_kind is not None:
            ctx["max_per_kind"] = max_per_kind
        if players is not None:
            ctx["players"] = self.lua_table([{"x": p[0], "y": p[1], "z": p[2] if len(p) > 2 else 0.0} for p in players])
        if now is not None:
            ctx["now"] = now
        return ctx

    @staticmethod
    def _lists(out: dict) -> dict:
        """An empty Lua table has no way to say it is a list: the reply fields that are lists always are."""
        for k in ("placements", "moves", "removals", "notes", "unknown_words", "groups"):
            if k in out and not isinstance(out[k], list):
                out[k] = []
        return out

    def run(self, text: str, world="open", pos=(0.0, 0.0, 0.0), yaw=0.0, aim=None, seed=None, wake=False, owner=None, planner=None, **kw):
        env = self.world(world) if isinstance(world, str) else world
        if wake:
            ok, rest = self.Parse.strip_wake(text)
            if not ok:
                return {"action": "none", "placements": [], "notes": [], "say": "", "request": None}
            text = rest
        req = self.parser.parse(self.parser, text)
        ctx = self.context(env, pos, yaw, aim, seed, owner, **kw)
        planner = planner or self.planner
        res = planner.run(planner, req, ctx)
        out = self._lists(to_py(self.mod, res))
        out["request"] = to_py(self.mod, req)
        return out

    def run_all(self, text: str, world="open", pos=(0.0, 0.0, 0.0), yaw=0.0, owner=None, **kw) -> list[dict]:
        """A compound sentence the way the game runs it: parse_all, then each request in order, each committed before the next
        is planned (so a later "it" is the group the earlier one made)."""
        env = self.world(world) if isinstance(world, str) else world
        reqs = self.parser.parse_all(self.parser, text)
        outs = []
        for i in range(1, len(reqs) + 1):
            req = reqs[i]
            ctx = self.context(env, pos, yaw, None, None, owner, **kw)
            out = self._lists(to_py(self.mod, self.planner.run(self.planner, req, ctx)))
            out["request"] = to_py(self.mod, req)
            self.commit(out, owner=owner)
            outs.append(out)
        return outs

    def run_request(self, req: dict, world="open", pos=(0.0, 0.0, 0.0), yaw=0.0, aim=None, seed=None, owner=None, planner=None, **kw):
        """Plan a hand-built request (the parser's output contract), so planner behaviour does not depend on the parser."""
        env = self.world(world) if isinstance(world, str) else world
        full = {"action": "add", "items": [], "arrangement": "unspecified", "target": "", "scene": "", "raw": "request",
                "normalised": "", "notes": [], "unknown": []}
        full.update(req)
        ctx = self.context(env, pos, yaw, aim, seed, owner, **kw)
        planner = planner or self.planner
        res = planner.run(planner, self.lua_table(full), ctx)
        out = self._lists(to_py(self.mod, res))
        out["request"] = full
        return out

    def commit(self, res: dict, owner=None) -> None:
        """Apply a plan to the registry the way the game does, so a following utterance sees it."""
        if res.get("placements"):
            self.gseq += 1                        # the whole plan is one group, like H.group in the game
        for p in res.get("placements", []):
            self.seq += 1
            self.registry.append({
                "id": self.next_id, "kind": p["kind"], "vi": p["vi"], "x": p["x"], "y": p["y"], "z": p["z"], "theta": p["theta"],
                "w": p["w"], "d": p["d"], "h": p["h"], "color_name": p.get("color_name"), "group": self.gseq,
                "seq": self.seq, "owner": owner,
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

    def _geom(self, e: dict):
        """A registry row's variant, entity origin and yaw (the planner's own inverse of its footprint pose)."""
        v = self.variant(e)
        lv = self.catalog_lua.kinds[e["kind"]].variants[e["vi"]]
        origin, yaw = self.Plan.entity_pose(lv, e["x"], e["y"], e["z"], e["theta"])
        return v, {"x": origin.x, "y": origin.y, "z": origin.z}, yaw

    @staticmethod
    def inside_top(sv: dict, sorigin: dict, syaw: float, poly: list, margin: float = 3.0) -> bool:
        """Do all corners of `poly` lie over the flat top of the support (give or take `margin` HU)?"""
        t = sv["top"]
        r = math.radians(-syaw)
        c, s_ = math.cos(r), math.sin(r)
        for x, y in poly:
            dx, dy = x - sorigin["x"], y - sorigin["y"]
            lx, ly = dx * c - dy * s_, dx * s_ + dy * c
            if not (t["x0"] - margin <= lx <= t["x1"] + margin and t["y0"] - margin <= ly <= t["y1"] + margin):
                return False
        return True

    def grade(self, res: dict, world=None, tuck=14.0) -> list[str]:
        problems: list[str] = []
        pls = res.get("placements", [])
        polys = []
        for p in pls:
            v = self.variant(p)
            polys.append(poly_of(v, p["origin"], p["yaw"]))
        # what a thing may stand on: the other placements, and what already stands
        supports = [(q, self.variant(q), q["origin"], q["yaw"], qpoly) for q, qpoly in zip(pls, polys)]
        for e in self.registry:
            v, origin, yaw = self._geom(e)
            supports.append((e, v, origin, yaw, poly_of(v, origin, yaw)))
        # floor gap: a thing on the floor has its bottom at 0, a thing on another stands on that thing's top, over it
        for p, poly in zip(pls, polys):
            v = self.variant(p)
            bottom = p["origin"]["z"] + v["mins"][2]
            if not p.get("on"):
                if abs(bottom) > 0.6:
                    problems.append(f"{p['kind']} bottom is {bottom:.1f} HU off the floor")
            else:
                ok = False
                for q, qv, qorigin, qyaw, qpoly in supports:
                    if q is p:
                        continue
                    if qv.get("top") and abs(bottom - (qorigin["z"] + qv["mins"][2] + qv["top"]["z"])) < 1.0:
                        if self.inside_top(qv, qorigin, qyaw, poly):                     # on its flat top, all of it
                            ok = True
                    elif abs(bottom - (qorigin["z"] + qv["maxs"][2])) < 1.0 and sat_penetration(poly, qpoly) > 0:
                        ok = True                                                           # stacked on the whole thing
                if not ok:
                    problems.append(f"{p['kind']} is said to stand on something but is not over a top that matches (bottom {bottom:.1f})")
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
        if world is not None and getattr(world, "bounds", None) is not None:
            x0, y0, x1, y1 = (world.bounds[k] for k in (1, 2, 3, 4))
            for p, poly in zip(pls, polys):
                for x, y in poly:
                    if x < x0 - 0.7 or x > x1 + 0.7 or y < y0 - 0.7 or y > y1 + 0.7:
                        problems.append(f"{p['kind']} pokes through a wall at ({x:.0f},{y:.0f})")
                        break
        # things the world already holds (a chair may be tucked into a table, up to `tuck` HU)
        for p, poly in zip(pls, polys):
            if p.get("on"):
                continue
            for e in self.registry:
                er = self._reg_poly(e)
                pen = sat_penetration(poly, er)
                if pen > 1.5:
                    seat = {p["kind"], e["kind"]} & SEATS
                    table = {p["kind"], e["kind"]} & TABLES
                    if seat and table and pen <= tuck and p.get("role") in ("around", "at_desk"):
                        continue
                    problems.append(f"{p['kind']} overlaps standing {e['kind']} by {pen:.1f}")
        return problems

    def audit(self, world=None, tuck=14.0) -> list[str]:
        """Grade everything that stands (after moves and turns): overlaps, walls, floating and unsupported things."""
        problems: list[str] = []
        things = []
        for e in self.registry:
            v, origin, yaw = self._geom(e)
            things.append((e, v, origin, yaw, poly_of(v, origin, yaw)))
        for i, (e, v, origin, yaw, poly) in enumerate(things):
            bottom = origin["z"] + v["mins"][2]
            if bottom > 1.0:           # standing on something: over a flat top, or stacked
                ok = False
                for j, (q, qv, qorigin, qyaw, qpoly) in enumerate(things):
                    if i == j:
                        continue
                    if qv.get("top") and abs(bottom - (qorigin["z"] + qv["mins"][2] + qv["top"]["z"])) < 1.5:
                        if self.inside_top(qv, qorigin, qyaw, poly, margin=4.0):
                            ok = True
                    elif abs(bottom - (qorigin["z"] + qv["maxs"][2])) < 1.5 and sat_penetration(poly, qpoly) > 0:
                        ok = True
                if not ok:
                    problems.append(f"{e['kind']}#{e['id']} floats ({bottom:.1f} HU up) over nothing that holds it")
            else:
                for j, (q, qv, qorigin, qyaw, qpoly) in enumerate(things):
                    if j <= i or (qorigin["z"] + qv["mins"][2]) > 1.0:
                        continue
                    pen = sat_penetration(poly, qpoly)
                    if pen > 1.5:
                        seat = {e["kind"], q["kind"]} & SEATS
                        table = {e["kind"], q["kind"]} & TABLES
                        if seat and (table or (e["kind"] in SEATS and q["kind"] in SEATS)) and pen <= tuck:
                            continue                    # a chair tucked under a table, or touching the next chair at its corner
                        problems.append(f"{e['kind']}#{e['id']} and {q['kind']}#{q['id']} interpenetrate by {pen:.1f} HU")
            if world is not None and getattr(world, "bounds", None) is not None:
                x0, y0, x1, y1 = (world.bounds[k] for k in (1, 2, 3, 4))
                for x, y in poly:
                    if x < x0 - 0.7 or x > x1 + 0.7 or y < y0 - 0.7 or y > y1 + 0.7:
                        problems.append(f"{e['kind']}#{e['id']} pokes through a wall at ({x:.0f},{y:.0f})")
                        break
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
