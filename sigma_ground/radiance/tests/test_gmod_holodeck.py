"""Tests for the Garry's Mod holodeck: ask for things in plain words, get a sensible arrangement of real models.

The parser, layout and planner are pure Lua (no GMod globals), so they run here under LuaJIT via `lupa` (an optional, test-only
dependency: the module is skipped when it is not installed). The planner is exercised in synthetic worlds (open ground, rooms
with walls) and graded by an INDEPENDENT checker (stage/gmod/holodeck/lab.py) that rebuilds every footprint from the
engine-measured model bounds, never from the planner's own bookkeeping.

The live checks (a dedicated server, real physics) are `stage/gmod/holodeck/live.py`; they need the game files and are not run here.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("lupa", reason="lupa (LuaJIT binding) not installed; holodeck tests skipped")

ROOT = Path(__file__).resolve().parents[3]
HOLODECK = ROOT / "stage" / "gmod" / "holodeck"
sys.path.insert(0, str(HOLODECK))

import build_catalog  # noqa: E402
import eval_parse  # noqa: E402
from lab import Lab, angle_diff, poly_of, sat_penetration  # noqa: E402

CATALOG = json.loads((HOLODECK / "data" / "catalog.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lab():
    return Lab()


def fresh(lab):
    lab.registry, lab.next_id, lab.seq = [], 1, 0
    return lab


def ray_hits(lab, p, poly, reach=80.0):
    """Does the line out of a thing's front, within `reach` HU, cross the polygon?"""
    fx, fy = lab.front_vec(p)
    steps = 40
    for i in range(1, steps + 1):
        x, y = p["x"] + fx * reach * i / steps, p["y"] + fy * reach * i / steps
        inside = False
        j = len(poly) - 1
        for k in range(len(poly)):
            xi, yi = poly[k]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = k
        if inside:
            return True
    return False


def centroid(pls):
    return sum(p["x"] for p in pls) / len(pls), sum(p["y"] for p in pls) / len(pls)


# --------------------------------------------------------------------------------------------- the library
def test_catalog_is_consistent():
    kinds = CATALOG["kinds"]
    assert len(kinds) >= 40
    models = set()
    for kid, k in kinds.items():
        assert k["names"], kid
        assert k["variants"], kid
        for v in k["variants"]:
            assert v["w"] > 0 and v["d"] > 0 and v["h"] > 0, (kid, v["model"])
            assert v["front"] % 90 == 0, (kid, v["model"], v["front"])
            assert v["mass"] > 0
            assert v["model"].startswith("models/") and v["model"].endswith(".mdl")
            assert v["model"] not in models, f"{v['model']} is used by two kinds"
            models.add(v["model"])
            if v.get("top"):
                assert k["surface"], f"{kid}: a top on a kind that is not a surface"
                assert 0 < v["top"]["z"] <= v["h"] + 0.5, (kid, v["model"], v["top"])
    # the roles a recipe may use
    allowed = {"main_seat", "side_seat", "center", "focus", "corner", "wall", "around", "wall_head", "beside_bed", "at_desk",
               "on_desk", "row_wall", "on_wall_unit"}
    for name, lines in CATALOG["recipes"].items():
        for n, kind, role in lines:
            assert kind in kinds, f"recipe {name} uses unknown {kind}"
            assert role in allowed, f"recipe {name} uses unknown role {role}"
            assert n >= 1


def test_generated_catalog_is_current():
    """catalog.json / catalog.lua must be the output of build_catalog from the curated list and the measurements."""
    built = build_catalog.build()
    assert not built["problems"], built["problems"]
    body = {k: v for k, v in built.items() if k != "problems"}
    assert json.loads(json.dumps(body)) == CATALOG, "run: python stage/gmod/holodeck/build_catalog.py"


# --------------------------------------------------------------------------------------------- the words
def test_parser_matches_the_audited_corpus():
    cases = json.loads((HOLODECK / "data" / "utterance_corpus.json").read_text(encoding="utf-8"))["cases"]
    parse = eval_parse.make_parser()
    failures = []
    for c in cases:
        if c.get("known_miss"):
            continue
        bad = eval_parse.compare(c, parse(c["utterance"]))
        if bad:
            failures.append((c["utterance"], bad))
    assert not failures, failures[:5]
    assert len(cases) >= 200


@pytest.mark.parametrize("text,woke,rest", [
    ("Skippy, add three couches", True, "add three couches"),
    ("hey skippy put a lamp here", True, "put a lamp here"),
    ("OK Skippy: clear the room", True, "clear the room"),
    ("  SKIPPY add a desk", True, "add a desk"),
    ("skippy", True, ""),
    ("we should get a couch", False, "we should get a couch"),
    ("skipping stones is fun", False, "skipping stones is fun"),
    ("did you hear what skippy said", False, "did you hear what skippy said"),
])
def test_wake_word(lab, text, woke, rest):
    ok, got = lab.Parse.strip_wake(text)
    assert bool(ok) == woke
    assert got == rest


def test_the_parser_never_asks_a_question(lab):
    for text in ["add three flurbles", "make it nice", "put the thing over there", "a dragon and two couches", "???", "skippy"]:
        res = lab.run(text)
        assert "?" not in res.get("say", ""), (text, res.get("say"))
    res = lab.run("a dragon and two couches")
    assert len(res["placements"]) == 2, "the couches still appear when one word is unknown"
    # the unknown word is kept apart for the requester's eyes only: nothing a player typed is echoed in a sentence others may see
    assert res["unknown_words"] == ["dragon"], res.get("unknown_words")
    assert any("my library" in n for n in res["notes"]) and not any("dragon" in n for n in res["notes"]), res["notes"]


# --------------------------------------------------------------------------------------------- geometry
def test_footprint_overlap_is_exact(lab):
    L = lab.Layout
    a = L.rect(0, 0, 40, 40, 0)
    assert L.overlap(a, L.rect(30, 0, 40, 40, 0), 0)
    assert not L.overlap(a, L.rect(41, 0, 40, 40, 0), 0)
    assert L.overlap(a, L.rect(30, 30, 40, 40, 45), 0)           # a corner poking in
    assert not L.overlap(a, L.rect(60, 60, 40, 40, 45), 0)
    assert L.penetration(a, L.rect(30, 0, 40, 40, 0)) == pytest.approx(10, abs=1e-6)


def test_perimeter_seats_face_in_without_overlap(lab):
    L = lab.Layout
    items = lab.lua_table([{"w": 20, "d": 21} for _ in range(6)])
    sides = lab.lua_table({"left": [items[1], items[2], items[3]], "right": [items[4], items[5], items[6]]})
    poses, inner = L.perimeter(lab.lua_table({"x": 0, "y": 0, "theta": 0}), sides, lab.lua_table({"gap": 6, "inner_w": 50}))
    rects = []
    for i in range(1, 7):
        p = poses[items[i]]
        rects.append(L.rect(p.x, p.y, 20, 21, p.theta))
        # facing the middle line x-axis: left row faces +v (right of frame), i.e. toward y = 0 from the -v side
    for i in range(6):
        for j in range(i + 1, 6):
            assert not L.overlap(rects[i], rects[j], 0)


# --------------------------------------------------------------------------------------------- the planner, graded independently
SCENES = [
    "a living room", "a dining room", "a kitchen", "a bedroom", "an office", "a waiting room", "a bathroom",
    "add three couches", "a couple of armchairs", "six chairs around a table", "ten stools in a circle", "four desks", "two beds",
    "a row of bookshelves", "put a lamp next to the couch", "a tv and a couch", "three crates", "a pile of boxes", "five barrels",
    "a coffee table", "a desk with a computer", "a bed with two nightstands", "two couches facing each other", "a bench against the wall",
    "a plant in the corner", "three plants", "a mug on the desk", "a couch and a coffee table and two armchairs",
    "a fridge, a stove and a sink", "a toilet", "eight chairs around a table", "a tv", "a computer", "three mugs", "two lamps",
    "a vending machine", "a row of 4 lockers", "a hospital bed", "a wardrobe", "a bunk bed", "a patio table and four chairs",
]


@pytest.mark.parametrize("world", ["open", "room", "room:420x360"])
@pytest.mark.parametrize("text", SCENES)
def test_scene_is_clean(lab, text, world):
    """Every placement is on the floor or on a real top, inside the room, clear of the rest, in every world a request fits."""
    fresh(lab)
    res = lab.run(text, world=world, pos=(0, 0, 0), yaw=0)
    if not res["placements"]:
        # only a genuinely tight room may refuse, and it must say so rather than leave things in walls
        assert world == "room:420x360", f"{text!r} found no place even in open ground or a large room: {res['say']}"
        assert "could not find" in res["say"]
        return
    problems = lab.grade(res, world=lab.world(world) if world != "open" else None)
    assert not problems, (text, world, problems[:4])


def test_explicit_counts_are_honoured(lab):
    for text, n in [("add three couches", 3), ("seven chairs", 7), ("a dozen crates", 12), ("two beds", 2), ("add 5 lamps", 5)]:
        fresh(lab)
        res = lab.run(text)
        assert len(res["placements"]) == n, (text, len(res["placements"]))


def test_three_couches_form_a_u_that_faces_in(lab):
    fresh(lab)
    res = lab.run("add three couches")
    pls = res["placements"]
    assert len(pls) == 3
    cx, cy = centroid(pls)
    for p in pls:
        assert lab.facing_error(p, cx, cy) < 75, "each couch faces the middle of the group"
    # a U: two couches face each other (180 degrees apart), the third at right angles to both
    fronts = sorted(round(math.degrees(math.atan2(*lab.front_vec(p)[::-1])) % 360) for p in pls)
    diffs = [abs(angle_diff(fronts[i], fronts[j])) for i in range(3) for j in range(i + 1, 3)]
    assert sum(1 for d in diffs if abs(d - 180) < 6) == 1 and sum(1 for d in diffs if abs(d - 90) < 6) == 2, fronts


def test_chairs_around_a_table_face_it(lab):
    fresh(lab)
    res = lab.run("six chairs around a table")
    chairs = [p for p in res["placements"] if p["kind"] == "chair"]
    tables = [p for p in res["placements"] if p["kind"] != "chair"]
    assert len(chairs) == 6 and len(tables) == 1
    t = tables[0]
    table_poly = poly_of(lab.variant(t), t["origin"], t["yaw"])
    for c in chairs:
        assert ray_hits(lab, c, table_poly), "a chair looks straight at the table it sits at"
        assert math.dist((c["x"], c["y"]), (t["x"], t["y"])) < max(t["w"], t["d"]) / 2 + 40
    assert "no table" in " ".join(res["notes"]), "it says it added the table"


def test_a_table_big_enough_is_chosen_for_many_chairs(lab):
    fresh(lab)
    res = lab.run("eight chairs around a table")
    kinds = {p["kind"] for p in res["placements"]}
    assert len([p for p in res["placements"] if p["kind"] == "chair"]) == 8
    assert kinds & {"table", "meeting table"}


def test_couches_facing_each_other(lab):
    fresh(lab)
    res = lab.run("two couches facing each other")
    a, b = res["placements"]
    assert lab.facing_error(a, b["x"], b["y"]) < 12 and lab.facing_error(b, a["x"], a["y"]) < 12


def test_the_couch_faces_the_tv(lab):
    fresh(lab)
    res = lab.run("a tv and a couch")
    tv = next(p for p in res["placements"] if p["kind"] == "tv")
    couch = next(p for p in res["placements"] if p["kind"] == "couch")
    assert lab.facing_error(couch, tv["x"], tv["y"]) < 20, "the couch looks at the TV"
    assert lab.facing_error(tv, couch["x"], couch["y"]) < 20, "and the TV looks at the couch"
    assert 70 < math.dist((couch["x"], couch["y"]), (tv["x"], tv["y"])) < 330


def test_a_desk_gets_a_chair_in_front_and_a_computer_on_top(lab):
    fresh(lab)
    res = lab.run("a desk with a computer and an office chair")
    desk = next(p for p in res["placements"] if p["kind"] == "desk")
    comp = next(p for p in res["placements"] if p["kind"] == "computer")
    chair = next(p for p in res["placements"] if p["kind"] == "office chair")
    assert comp.get("on"), "the computer stands on the desk"
    assert abs(angle_diff(math.degrees(math.atan2(*lab.front_vec(comp)[::-1])), math.degrees(math.atan2(*lab.front_vec(desk)[::-1])))) < 10
    assert lab.facing_error(chair, desk["x"], desk["y"]) < 20, "the chair faces the desk"
    # the monitor's bottom is on the desk top
    assert not lab.grade(res)


def test_a_bookshelf_stands_flush_against_a_real_wall(lab):
    fresh(lab)
    env = lab.world("room")
    res = lab.run("a bookshelf", world=env, pos=(0, 0, 0), yaw=90)           # looking toward the +y wall at y = 240
    p = res["placements"][0]
    poly = poly_of(lab.variant(p), p["origin"], p["yaw"])
    gap = 240 - max(y for _, y in poly)
    assert -0.7 <= gap <= 3.5, f"back of the shelf is {gap:.1f} HU from the wall"
    fx, fy = lab.front_vec(p)
    assert fy < -0.98, "and it faces into the room"


def test_a_lamp_goes_beside_an_existing_couch(lab):
    fresh(lab)
    first = lab.run("a couch")
    lab.commit(first)
    res = lab.run("put a lamp next to the couch")
    assert len(res["placements"]) == 1, "the existing couch is reused, not duplicated"
    lamp, couch = res["placements"][0], first["placements"][0]
    assert not lab.grade(res)
    gap = math.dist((lamp["x"], lamp["y"]), (couch["x"], couch["y"])) - couch["w"] / 2 - lamp["w"] / 2
    assert 0 <= gap <= 14, f"the lamp stands {gap:.1f} HU from the couch"


def test_things_never_land_on_what_already_stands(lab):
    fresh(lab)
    for text in ["add three couches", "a living room", "a table with four chairs", "a bedroom"]:
        res = lab.run(text)
        assert res["placements"], text
        assert not lab.grade(res), (text, lab.grade(res))
        lab.commit(res)
    assert len(lab.registry) == 20


def test_planning_is_deterministic(lab):
    fresh(lab)
    a = lab.run("a living room", seed=11)
    b = lab.run("a living room", seed=11)
    assert [(p["kind"], p["vi"], round(p["x"], 3), round(p["y"], 3), round(p["theta"], 3)) for p in a["placements"]] == \
           [(p["kind"], p["vi"], round(p["x"], 3), round(p["y"], 3), round(p["theta"], 3)) for p in b["placements"]]


def test_adjectives_steer_the_choice(lab):
    fresh(lab)
    small = lab.run("a small couch", seed=3)["placements"][0]
    big = lab.run("a large couch", seed=3)["placements"][0]
    assert small["w"] * small["d"] < big["w"] * big["d"]
    red = lab.run("a red couch")["placements"][0]
    assert red["color_name"] == "red" and red["color"] == [200, 45, 45]


def test_a_television_is_a_proper_screen_unless_a_small_one_is_asked_for(lab):
    for seed in range(1, 8):
        fresh(lab)
        tv = lab.run("a tv", seed=seed)["placements"][0]
        assert tv["h"] >= 30
    small = lab.run("a small tv")["placements"][0]
    assert small["h"] < 30


# --------------------------------------------------------------------------------------------- managing what stands
def stand(lab, *texts, world="open"):
    fresh(lab)
    for t in texts:
        res = lab.run(t, world=world)
        lab.commit(res)
    return lab


def test_remove_one_all_or_the_last(lab):
    stand(lab, "add three couches")
    res = lab.run("remove the couch")
    assert res["action"] == "remove" and len(res["removals"]) == 1
    res = lab.run("get rid of all the couches")
    assert len(res["removals"]) == 3
    res = lab.run("remove the lamp")
    assert not res["removals"] and "no floor lamp" in res["say"]
    stand(lab, "a couch", "a lamp")
    res = lab.run("remove it")
    assert len(res["removals"]) >= 1


def test_clear_and_undo_are_recognised(lab):
    stand(lab, "a living room")
    assert lab.run("clear the room")["action"] == "clear"
    assert lab.run("undo")["action"] == "undo"


def test_turn_a_couch_around(lab):
    stand(lab, "a couch")
    before = lab.registry[0]["theta"]
    res = lab.run("turn the couch around")
    assert res["action"] == "rotate"
    assert abs(angle_diff(res["moves"][0]["theta"], before + 180)) < 1e-6


def test_make_a_chair_face_me(lab):
    stand(lab, "a chair")
    res = lab.run("turn the chair to face me")
    c = lab.registry[0]
    want = math.degrees(math.atan2(0 - c["y"], 0 - c["x"])) % 360
    assert abs(angle_diff(res["moves"][0]["theta"], want)) < 1e-6


def test_move_a_couch_closer(lab):
    stand(lab, "a couch")
    c = lab.registry[0]
    d0 = math.hypot(c["x"], c["y"])
    res = lab.run("move the couch closer")
    assert res["action"] == "move" and res["moves"]
    m = res["moves"][0]
    assert math.hypot(m["x"], m["y"]) < d0


def test_rearrange_what_is_there_into_a_row(lab):
    stand(lab, "five chairs in a circle")
    res = lab.run("put the chairs in a row")
    assert res["action"] in ("rearrange", "move")
    assert len(res["moves"]) == 5
    ys = sorted(round(m["y"]) for m in res["moves"])
    xs = sorted(round(m["x"]) for m in res["moves"])
    spread = lambda v: max(v) - min(v)
    assert min(spread(xs), spread(ys)) < 12, "all five now stand along one line"


def test_a_registry_row_is_never_overlapped_by_a_move(lab):
    stand(lab, "a couch", "a bookshelf")
    res = lab.run("put the couch next to the bookshelf")
    assert res["action"] == "move"
    m = res["moves"][0]
    shelf = next(e for e in lab.registry if e["kind"] == "bookshelf")
    couch = next(e for e in lab.registry if e["kind"] == "couch")
    r1 = lab.Layout.rect(m["x"], m["y"], couch["w"], couch["d"], m["theta"])
    r2 = lab.Layout.rect(shelf["x"], shelf["y"], shelf["w"], shelf["d"], shelf["theta"])
    assert not lab.Layout.overlap(r1, r2, 0)


# --------------------------------------------------------------------------------------------- proximity voice (the numbers)
@pytest.fixture(scope="module")
def voice(lab):
    return lab.load((ROOT / "stage" / "gmod" / "addon" / "lua" / "mentat" / "voice" / "core.lua").as_posix())


def test_voice_is_full_then_fades_smoothly_to_silence(voice):
    rng = 1100
    assert voice.gain(0, rng) == 1 and voice.gain(0.6 * rng, rng) == 1
    assert voice.gain(rng, rng) == 0 and voice.gain(5 * rng, rng) == 0
    samples = [voice.gain(d, rng) for d in range(int(0.6 * rng), rng + 1, 20)]
    assert all(a >= b for a, b in zip(samples, samples[1:])), "never gets louder with distance"
    assert 0.45 < voice.gain(0.8 * rng, rng) < 0.55, "half loudness half way through the fade"


def test_a_whisper_a_shout_and_a_wall_change_the_reach(voice, lab):
    o = lab.lua_table({"range": 1000})
    normal = voice.effective_range(o, None)
    whisper = voice.effective_range(o, lab.lua_table({"crouching": True}))
    shout = voice.effective_range(o, lab.lua_table({"shouting": True}))
    assert whisper < 0.5 * normal < normal < shout
    assert voice.effective_range(o, lab.lua_table({"shouting": True, "crouching": True})) == shout
    heard_open, _ = voice.decide(500, normal, False, o)
    heard_wall, _ = voice.decide(700, normal, True, o)
    assert heard_open and not heard_wall, "a wall shortens how far a voice carries"
    heard_near_wall, g = voice.decide(200, normal, True, o)
    assert heard_near_wall and g == 1


# --------------------------------------------------------------------------------------------- what this server can show
def test_library_keeps_only_models_the_server_has(lab):
    lib = lab.load((ROOT / "stage" / "gmod" / "addon" / "lua" / "mentat" / "holodeck" / "library.lua").as_posix())
    cat = lab.catalog_lua
    gone = {"models/props_c17/furniturecouch001a.mdl", "models/props_interiors/furniture_couch01a.mdl"}
    is_valid = lab.rt.eval("function(set) return function(m) return not set[m] end end")(lab.lua_table({m: True for m in gone}))
    filtered, report = lib.filter(cat, is_valid, lab.lua_table({}))
    assert report.kept == report.total - 2
    couch = filtered.kinds["couch"]
    assert len(couch.variants) == len(CATALOG["kinds"]["couch"]["variants"]) - 2
    # nothing is left of a kind with no model, and rooms that lost their furniture go with it
    nothing = lab.rt.eval("function(m) return false end")
    filtered, report = lib.filter(cat, nothing, lab.lua_table({}))
    assert len(filtered.order) == 0 and report.kept == 0


def test_hl2_only_mode_drops_the_css_models(lab):
    lib = lab.load((ROOT / "stage" / "gmod" / "addon" / "lua" / "mentat" / "holodeck" / "library.lua").as_posix())
    filtered, report = lib.filter(lab.catalog_lua, None, lab.lua_table({"hl2_only": True}))
    for kid in filtered.order.values():
        for i in range(1, len(filtered.kinds[kid].variants) + 1):
            assert not filtered.kinds[kid].variants[i].model.startswith("models/props/"), kid
    assert 0 < len(filtered.order) < len(CATALOG["order"]) or len(filtered.order) == len(CATALOG["order"])
    # a parser built on the thinner library says so instead of inventing the missing thing
    thin_parser = lab.Parse.new(filtered)
    from eval_parse import to_py
    req = to_py(lab.mod, thin_parser.parse(thin_parser, "add a plant"))
    assert any(str(i["kind"]).startswith("UNKNOWN") for i in req["items"]) or any(i["kind"] == "plant" for i in req["items"])


def test_relations_to_standing_things_never_overlap_them(lab):
    fresh(lab)
    for text in ["add a coffee table", "put three red couches around the coffee table"]:
        res = lab.run(text, pos=(64, 0, 0))
        assert res["placements"], text
        assert not lab.grade(res), (text, lab.grade(res))
        lab.commit(res)
    for text in ["eight chairs in front of the coffee table", "an armchair and a floor lamp beside the coffee table",
                 "two armchairs around the coffee table", "a bookshelf behind the couch"]:
        res = lab.run(text, pos=(64, 0, 0))
        assert res["placements"], (text, res["say"])
        assert not lab.grade(res), (text, lab.grade(res))


def test_a_player_changes_only_their_own_things(lab):
    fresh(lab)
    a = lab.run("two couches", owner="alice")
    lab.commit(a, owner="alice")
    b = lab.run("a couch", owner="bob")
    lab.commit(b, owner="bob")
    assert len(lab.registry) == 3
    res = lab.run("get rid of all the couches", owner="alice")
    mine = {e["id"] for e in lab.registry if e["owner"] == "alice"}
    assert set(res["removals"]) == mine and len(mine) == 2, "bob's couch is not alice's to remove"
    res = lab.run("turn the couch around", owner="bob")
    bobs = [e["id"] for e in lab.registry if e["owner"] == "bob"]
    assert [m["id"] for m in res["moves"]] == bobs
    # a request that names something that is only someone else's finds nothing of theirs
    fresh(lab)
    lab.commit(lab.run("a lamp"), owner="alice")
    res = lab.run("remove the lamp", owner="bob")
    assert not res["removals"] and "no floor lamp" in res["say"]
    # an admin (no restriction) can
    assert len(lab.run("remove the lamp")["removals"]) == 1


# --------------------------------------------------------------------------------------------- Skippy: agent, voice, ears
import skippy_bridge  # noqa: E402
import skippy_ears  # noqa: E402


@pytest.mark.parametrize("text,woke,rest", [
    ("Skippy, add three couches", True, "add three couches"),
    ("Skipy add a lamp", True, "add a lamp"),
    ("skip e put a table here", True, "put a table here"),
    ("Hey Skippy. Clear the room", True, "Clear the room"),
    ("skipping stones", False, "skipping stones"),
    ("Skippers", False, "Skippers"),
    ("can you pass the salt", False, "can you pass the salt"),
])
def test_the_ears_only_wake_for_skippy(text, woke, rest):
    ok, got = skippy_ears.strip_wake(text)
    assert ok == woke and got == rest


def test_the_ears_keep_only_requests_for_skippy(tmp_path, monkeypatch):
    monkeypatch.setattr(skippy_ears, "transcribe", lambda wav, api, timeout=60: "Skippy, add two lamps")
    assert skippy_ears.handle_speech(b"x", "http://none", tmp_path, False, log=lambda *_: None) == "add two lamps"
    data = json.loads((tmp_path / "voice_in.json").read_text(encoding="utf-8"))
    assert data["text"] == "add two lamps" and data["heard"] == "Skippy, add two lamps" and data["direct"] is False
    (tmp_path / "voice_in.json").unlink()
    monkeypatch.setattr(skippy_ears, "transcribe", lambda wav, api, timeout=60: "what did you have for lunch")
    assert skippy_ears.handle_speech(b"x", "http://none", tmp_path, False, log=lambda *_: None) is None
    assert not (tmp_path / "voice_in.json").exists(), "talk between players is dropped, never handed on"
    # the +skippy key needs no wake word
    assert skippy_ears.handle_speech(b"x", "http://none", tmp_path, True, log=lambda *_: None) == "what did you have for lunch"


def test_the_agent_can_only_ask_for_things_in_the_library():
    req, notes = skippy_bridge.validate_request({
        "action": "add", "scene": "dining room", "arrangement": "bogus", "target": "Coffee Table",
        "items": [{"kind": "table", "count": 1}, {"kind": "chair", "count": 4}, {"kind": "side table", "count": 9999},
                  {"kind": "dragon", "count": 2}, {"kind": "sofa", "count": 2, "adjectives": ["Red"]}],
    })
    assert req["scene"] == "dining room" and req["arrangement"] == "unspecified" and req["target"] == "coffee table"
    kinds = {i["kind"]: i for i in req["items"]}
    assert "dragon" not in kinds and "table" not in kinds and "chair" not in kinds, "the room already brings its own table and chairs"
    assert kinds["side table"]["count"] == 40, "counts are clamped"
    assert kinds["couch"]["count"] == 2 and kinds["couch"]["adjectives"] == ["red"], "a sofa is a couch"
    assert any("dragon" in n for n in notes)
    empty, _ = skippy_bridge.validate_request({"action": "add", "items": [{"kind": "dragon", "count": 1}]})
    assert empty["action"] == "none", "nothing buildable means nothing to do"


def test_the_agent_plans_through_the_model_it_is_given(monkeypatch):
    seen = {}

    def fake_chat(messages, schema, model=None, timeout=0, temperature=0):
        seen["system"] = messages[0]["content"]
        seen["user"] = messages[-1]["content"]
        return json.dumps({"action": "add", "items": [{"kind": "armchair", "count": 2}, {"kind": "floor lamp", "count": 1}],
                           "arrangement": "unspecified", "target": "", "scene": "", "say": "Snug."})

    monkeypatch.setattr(skippy_bridge, "chat", fake_chat)
    out = skippy_bridge.plan("make it cozy", ["couch"])
    assert [(i["kind"], i["count"]) for i in out["request"]["items"]] == [("armchair", 2), ("floor lamp", 1)]
    assert "armchair" in seen["system"] and "bookshelf" in seen["system"], "the agent is shown the whole library"
    assert "already standing nearby: couch" in seen["user"]
    monkeypatch.setattr(skippy_bridge, "chat", lambda *a, **k: "I think you should add some chairs {oops")
    assert skippy_bridge.plan("anything")["request"]["action"] == "none", "a model that rambles builds nothing"


def test_the_bridge_is_authenticated_and_answers_in_fixed_ways(monkeypatch):
    import urllib.error
    import urllib.request
    monkeypatch.setattr(skippy_bridge, "chat", lambda *a, **k: json.dumps({"action": "none", "say": "Nothing to do."}))
    monkeypatch.setattr(skippy_bridge, "TOKEN", "sekrit")
    srv = skippy_bridge.Server(("127.0.0.1", 0), skippy_bridge.Handler)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        def post(path, body, token="sekrit"):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json", "X-Skippy-Token": token})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        assert post("/plan", {"text": "hello"}, token="wrong")[0] == 401
        code, body = post("/plan", {"text": "do something"})
        assert code == 200 and body["request"]["action"] == "none"
        assert post("/plan", {"text": ""})[0] == 400
        assert post("/rm-rf", {})[0] == 404, "there is no way to ask it to do anything else"
        assert post("/plan", {"text": "x" * 50000})[0] in (400, 413)
    finally:
        srv.shutdown()


@pytest.mark.skipif(not (__import__("shutil").which("powershell.exe") and __import__("shutil").which("ffmpeg")),
                    reason="needs Windows speech (PowerShell) and ffmpeg")
def test_skippy_can_speak_with_the_local_voice():
    data, voice = skippy_bridge.speak("Three couches, in a U shape.", mode="sapi")
    assert voice == "sapi" and len(data) > 4000
    assert data[:3] == b"ID3" or data[0] == 0xFF, "an mp3"
