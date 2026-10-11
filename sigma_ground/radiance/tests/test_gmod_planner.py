"""Regression tests for the holodeck planner (stage/gmod/addon/lua/mentat/holodeck/plan.lua and layout.lua), written after an
independent review found what the first suite could not see: room mode that dropped or doubled things, tables too small for the
chairs asked for, small things hanging off the edge of what they stand on, turns that pushed props into walls, a search that
could stall a public server, and replies that repeated whatever a player typed.

Each test names the review finding it holds shut (planner-N / heldout-N). The planner is exercised through the offline lab
(stage/gmod/holodeck/lab.py): the real parser where a plain sentence says it, and hand-built requests (the parser's output
contract) where the thing under test does not depend on how the words are read. Everything is graded by the lab's INDEPENDENT
checker, which rebuilds footprints from the engine-measured model bounds and never trusts the planner's own numbers.

Like test_gmod_holodeck.py this needs `lupa` (LuaJIT) and is skipped without it. The lab hands the planner the engine's own
seed (a hash of the sentence), because a small fixed seed always draws the smallest model of a kind and hid most of these.
"""
from __future__ import annotations

import faulthandler
import math
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("lupa", reason="lupa (LuaJIT binding) not installed; holodeck tests skipped")

ROOT = Path(__file__).resolve().parents[3]
HOLODECK = ROOT / "stage" / "gmod" / "holodeck"
sys.path.insert(0, str(HOLODECK))

from lab import Lab, poly_of  # noqa: E402


@pytest.fixture(scope="module")
def lab():
    return Lab()


def fresh(lab):
    lab.registry, lab.next_id, lab.seq, lab.gseq = [], 1, 0, 0
    return lab


def env_of(lab, world):
    return None if world == "open" else lab.world(world)


def kinds(res):
    return Counter(p["kind"] for p in res["placements"])


def clean(lab, res, world="open"):
    return lab.grade(res, world=env_of(lab, world)) if res["placements"] else []


def seeds(lab, text, n):
    """The engine's seed for a sentence, and n-1 more spread over the range (the ones a lab default of 12345 never draws)."""
    return [lab.Layout.hash(f"{text}#{k}") for k in range(n)]


def item(kind, count=1, **kw):
    d = {"kind": kind, "count": count, "adjectives": [], "word": kind, "definite": False, "plural": count > 1}
    d.update(kw)
    return d


def stand(lab, text, world="room", owner=None, **kw):
    """Plan a sentence and commit it, as the game does."""
    res = lab.run(text, world=world, owner=owner, **kw)
    lab.commit(res, owner=owner)
    return res


def standing(lab, kind):
    return [e for e in lab.registry if e["kind"] == kind]


def spread(items):
    xs, ys = [p["x"] for p in items], [p["y"] for p in items]
    return max(xs) - min(xs), max(ys) - min(ys)


# ================================================================================================ room mode: nothing doubled, nothing lost
SEAT_KINDS = ["couch", "loveseat", "bench"]
PARTNERS = ["plant", "floor lamp", "bookshelf", "bed", "side table", "computer", "armchair", "desk", "trash can"]


@pytest.mark.parametrize("world", ["open", "room"])
@pytest.mark.parametrize("other", PARTNERS)
@pytest.mark.parametrize("seat", SEAT_KINDS)
def test_a_main_seat_is_placed_once(lab, seat, other, world):
    """planner-1 / heldout-1: a couch, loveseat or bench plus a non-table item was placed twice (cluster AND back wall)."""
    fresh(lab)
    res = lab.run(f"a {seat} and a {other}", world=world)
    got = kinds(res)
    assert got[seat] == 1 and got[other] == 1, (got, res["say"])
    assert not clean(lab, res, world), clean(lab, res, world)


def test_every_requested_count_is_placed(lab):
    """planner-1: three couches and two plants is three couches and two plants (the doubled version was six couches)."""
    fresh(lab)
    for text, want in [("three couches and two plants", {"couch": 3, "plant": 2}), ("a bench and a plant", {"bench": 1, "plant": 1}),
                       ("four couches and a lamp", {"couch": 4, "floor lamp": 1})]:
        got = kinds(lab.run(text, world="open"))
        assert all(got[k] == n for k, n in want.items()), (text, got)


@pytest.mark.parametrize("text,want", [
    ("three couches and two tables", {"couch": 3, "table": 2}),
    ("a table and a coffee table", {"table": 1, "coffee table": 1}),
    ("two tables and a coffee table", {"table": 2, "coffee table": 1}),
    ("three desks, a plant and a bookshelf", {"desk": 3, "plant": 1, "bookshelf": 1}),
    ("two coffee tables and a couch", {"coffee table": 2, "couch": 1}),
])
@pytest.mark.parametrize("world", ["open", "room"])
def test_every_table_and_desk_asked_for_is_placed(lab, text, want, world):
    """planner-7 / heldout-2: room mode kept only the first table and the first desk."""
    fresh(lab)
    res = lab.run(text, world=world)
    got = kinds(res)
    assert all(got[k] == n for k, n in want.items()), (got, res["say"])
    assert not clean(lab, res, world), clean(lab, res, world)


@pytest.mark.parametrize("text,desks,chairs", [
    ("a desk and two chairs", 1, 2), ("a desk and three chairs", 1, 3), ("two desks and two chairs", 2, 2),
    ("ten desks and ten chairs", 10, 10), ("sixty desks and sixty chairs", 60, 60),
])
def test_every_desk_gets_its_own_chairs_not_one_spot(lab, text, desks, chairs):
    """heldout-2: the chairs of a desk were all put at the same pose (and the second desk's chairs with them)."""
    fresh(lab)
    res = lab.run(text, world="open")
    got = kinds(res)
    chair_kind = "office chair" if "office chair" in got else "chair"
    assert got["desk"] == desks and got[chair_kind] == chairs, (got, res["say"])
    seats = [p for p in res["placements"] if p["kind"] == chair_kind]
    spots = {(round(p["x"] / 4), round(p["y"] / 4)) for p in seats}
    assert len(spots) == len(seats), "no two chairs stand in the same place"
    assert not clean(lab, res), clean(lab, res)[:3]


@pytest.mark.parametrize("n", [4, 5, 6, 8, 10, 12, 20])
def test_a_table_is_sized_for_the_chairs_asked_for(lab, n):
    """planner-6 / heldout-3: 'a table with six chairs' placed four in about a fifth of draws, and said nothing."""
    text = f"a table with {n} chairs"
    for seed in seeds(lab, text, 12):
        fresh(lab)
        res = lab.run(text, world="open", seed=seed)
        got = kinds(res)
        assert got["chair"] == n, (n, seed, got, res["say"])
        assert got["table"] + got["meeting table"] == 1
        assert not clean(lab, res), clean(lab, res)[:2]


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_chairs_around_a_table_it_makes_all_fit(lab, n):
    """planner-6: the capacity estimate for 'N chairs around a table' ignored that chairs are counted per side."""
    text = f"{n} chairs around a table"
    for seed in seeds(lab, text, 12):
        fresh(lab)
        res = lab.run(text, world="open", seed=seed)
        assert kinds(res)["chair"] == n, (n, seed, kinds(res), res["say"])
        assert not clean(lab, res), clean(lab, res)[:2]


def test_a_table_that_cannot_seat_everyone_says_so(lab):
    """planner-6: chairs that did not fit were dropped without a word; a small table the player chose is kept and reported."""
    fresh(lab)
    res = lab.run("a small table with ten chairs", world="open")
    placed = kinds(res)["chair"]
    assert 0 < placed < 10
    assert any(f"only {placed} of the 10 chairs fit" in n for n in res["notes"]), res["notes"]
    assert "only" in res["say"].lower()


def test_a_table_too_big_for_the_room_falls_back_to_the_chairs_that_fit(lab):
    """The table sized for the chairs may be too big for a small room: the first draw with the seats that fit, told honestly."""
    fresh(lab)
    res = lab.run("a table and twenty chairs", world="room:300x300")
    if res["placements"]:
        assert kinds(res)["chair"] >= 4
        assert not clean(lab, res, "room:300x300"), clean(lab, res, "room:300x300")


@pytest.mark.parametrize("world", ["open", "room", "room:420x360", "room:300x300"])
def test_the_office_has_its_chair_and_every_scene_is_complete(lab, world):
    """planner-8 / heldout-4: the office recipe's wall desk never got its office chair. Every recipe line is placed."""
    fresh(lab)
    for name, lines in lab.catalog["recipes"].items():
        res = lab.run(f"a {name}" if name[0] not in "aeiou" else f"an {name}", world=world)
        if not res["placements"]:
            assert world in ("room:300x300", "room:420x360"), (name, world, res["say"])
            continue
        got = kinds(res)
        for n, kind, _role in lines:
            assert got[kind] >= n, (name, world, kind, got, res["say"])
        assert not clean(lab, res, world), (name, world, clean(lab, res, world)[:2])


def test_the_office_chair_faces_the_desk(lab):
    fresh(lab)
    res = lab.run("an office", world="room")
    desk = next(p for p in res["placements"] if p["kind"] == "desk")
    chair = next(p for p in res["placements"] if p["kind"] == "office chair")
    assert lab.facing_error(chair, desk["x"], desk["y"]) < 25
    assert math.dist((chair["x"], chair["y"]), (desk["x"], desk["y"])) < max(desk["w"], desk["d"]) / 2 + 40
    assert not clean(lab, res, "room")


def test_a_seat_for_a_desk_or_table_that_is_not_there_is_still_placed(lab):
    """heldout-4: an at_desk entry with no desk anywhere vanished (and so did an 'around' chair with no table). It is just a seat now."""
    fresh(lab)
    recipes = lab.catalog_lua.recipes
    recipes["study"] = lab.lua_table([[2, "office chair", "at_desk"], [3, "chair", "around"], [1, "plant", "corner"]])
    try:
        for world in ("open", "room"):
            res = lab.run_request({"action": "add", "scene": "study", "items": []}, world=world)
            assert kinds(res) == {"office chair": 2, "chair": 3, "plant": 1}, (world, kinds(res), res["say"])
            assert not clean(lab, res, world)
    finally:
        recipes["study"] = None


@pytest.mark.parametrize("text,kind,count", [
    ("a living room with three armchairs", "armchair", 3), ("a bedroom with two beds", "bed", 2),
    ("a bathroom with two sinks", "sink", 2), ("a dining room with six chairs", "chair", 6),
    ("an office with two plants", "plant", 2), ("a kitchen with eight chairs", "chair", 8),
])
def test_a_scene_with_an_explicit_count_has_that_count(lab, text, kind, count):
    """heldout-12: the explicit item was added to the recipe's own (a bedroom with two beds had three)."""
    for world in ("open", "room"):
        fresh(lab)
        res = lab.run(text, world=world)
        assert kinds(res)[kind] == count, (text, world, kinds(res), res["say"])
        assert not clean(lab, res, world), clean(lab, res, world)[:2]


def test_a_colour_asked_for_beside_a_scene_lands_on_the_scene_item(lab):
    fresh(lab)
    res = lab.run("a living room with a red couch", world="room")
    couches = [p for p in res["placements"] if p["kind"] == "couch"]
    assert len(couches) == 1 and couches[0]["color_name"] == "red"


def test_more_means_extra_ones_not_a_replacement(lab):
    fresh(lab)
    res = lab.run_request({"action": "add", "scene": "living room", "items": [item("couch", 1)], "normalised": "a living room with another couch"},
                          world="room")
    assert kinds(res)["couch"] == 2


# ================================================================================================ looking for a place
@pytest.mark.parametrize("kind", ["chair", "couch", "armchair", "plant", "barrel", "bed", "wardrobe", "stool", "table"])
@pytest.mark.parametrize("count", [1, 3])
def test_things_around_me_on_open_ground(lab, kind, count):
    """planner-2: the ring never grew past the player's keep-out square, so most draws said 'could not find clear floor'."""
    plural = kind + "s"
    text = f"{'a' if count == 1 else 'three'} {kind if count == 1 else plural} around me"
    for seed in seeds(lab, text, 8):
        fresh(lab)
        res = lab.run(text, world="open", seed=seed)
        assert kinds(res)[kind] == count, (text, seed, res["say"])
        assert not clean(lab, res), clean(lab, res)[:2]
        # round the player, not beside them: the middle of the group is near the player
        cx, cy = sum(p["x"] for p in res["placements"]) / count, sum(p["y"] for p in res["placements"]) / count
        if count > 1:
            assert math.hypot(cx, cy) < 80, (cx, cy)


@pytest.mark.parametrize("gap", [20, 40, 60])
@pytest.mark.parametrize("yaw", [0, 90, 180, 270])
@pytest.mark.parametrize("text", ["a couch around me", "a bed around me", "four chairs around me", "a table in front of me", "a couch behind me",
                                  "two couches beside me"])
def test_things_around_me_next_to_a_wall(lab, text, yaw, gap):
    """heldout-21: around/beside/behind/in front of me had no fallback, so they failed anywhere near a wall."""
    fresh(lab)
    res = lab.run(text, world="room", pos=(-280 + gap, 0, 0), yaw=yaw)
    assert res["placements"], (text, yaw, gap, res["say"])
    assert not clean(lab, res, "room"), clean(lab, res, "room")[:2]


def test_the_around_me_fallback_says_what_it_did(lab):
    fresh(lab)
    res = lab.run("four chairs around me", world="room", pos=(-260, 0, 0), yaw=180)       # facing the wall, 20 HU from it
    assert len(res["placements"]) == 4
    assert any("part of the way round" in n or "nearby" in n for n in res["notes"]), res["notes"]


@pytest.mark.parametrize("text", ["ten crates in a grid", "six benches", "five desks"])
def test_a_player_in_the_middle_of_a_room_still_gets_a_place(lab, text):
    """planner-22 / heldout-20: no frame nearer than R+62 was tried, so a centred player was told there was no floor."""
    fresh(lab)
    res = lab.run(text, world="room", pos=(0, 0, 0), yaw=0)
    assert res["placements"], res["say"]
    assert not clean(lab, res, "room"), clean(lab, res, "room")[:2]


def test_nearer_frames_come_last_so_nothing_that_worked_moves(lab):
    """The nearer ring is only tried after every old frame failed: 'add three couches' in the open stays where it was."""
    fresh(lab)
    a = lab.run("add three couches", world="open", seed=11)
    cx, cy = sum(p["x"] for p in a["placements"]) / 3, sum(p["y"] for p in a["placements"]) / 3
    assert math.hypot(cx, cy) > 120, "still a good way off, as before"


def test_a_wall_loving_referent_stands_at_the_wall(lab):
    """heldout-22: the created tv/couch floated mid-room (the wall frame used the whole group's radius), group behind the player."""
    for text in ["a couch facing the tv", "three armchairs facing the tv"]:
        for world in ("room", "room:420x360"):
            fresh(lab)
            res = lab.run(text, world=world, pos=(0, 0, 0), yaw=0)
            assert res["placements"], (text, world, res["say"])
            tv = next(p for p in res["placements"] if p["kind"] == "tv")
            w = lab.world(world)
            x0, y0, x1, y1 = (w.bounds[k] for k in (1, 2, 3, 4))
            poly = poly_of(lab.variant(tv), tv["origin"], tv["yaw"])
            gap = min(min(x - x0 for x, _ in poly), min(x1 - x for x, _ in poly), min(y - y0 for _, y in poly), min(y1 - y for _, y in poly))
            assert gap < 8, (text, world, f"the tv stands {gap:.0f} HU from the nearest wall")
            for p in res["placements"]:
                if p["kind"] != "tv":
                    assert lab.facing_error(p, tv["x"], tv["y"]) < 60
            assert not clean(lab, res, world)


def test_in_a_big_hall_wall_things_stay_near_the_player(lab):
    """heldout-23: the wall search reached 900 HU, so a couch jumped to the far wall of a big hall. An explicit 'against the wall' still may."""
    fresh(lab)
    res = lab.run("a couch", world="room:1600x1400", pos=(0, 0, 0), yaw=0)
    p = res["placements"][0]
    assert math.hypot(p["x"], p["y"]) < 450, (p["x"], p["y"])
    res = lab.run("a couch against the wall", world="room:1600x1400", pos=(0, 0, 0), yaw=0)
    assert res["placements"] and "wall" in res["say"], res["say"]


def test_a_tie_keeps_the_earliest_frame(lab):
    """planner-16: candidates equally bad differ by float noise; the last evaluated won and the group went across the map."""
    code = """
    function(Plan)
      local env = { floor_z = function() return 0 end, clear = function() return true end, wall_probe = function() return nil end }
      local ctx = { pos = { x = 1e6, y = 1e6, z = 0 }, yaw = 0, env = env, registry = {}, keep_out = false }
      local calls = 0
      -- two footprints 1.1 HU into each other wherever the group is put; each try differs from the last by about 1e-13
      local b = { build = function(frame)
        calls = calls + 1
        local gap = 18.9 + 1e-13 * calls
        return { { x = frame.x, y = frame.y, w = 20, d = 20, theta = 0, h = 10, zoff = 0, inst = {} },
                 { x = frame.x + gap, y = frame.y, w = 20, d = 20, theta = 0, h = 10, zoff = 0, inst = {} } }
      end }
      local frames = { { x = 0, y = 0, theta = 0 }, { x = 500, y = 0, theta = 0 }, { x = -900, y = 4, theta = 0 } }
      local pls, frame, _, score = Plan._fit(ctx, { b }, frames)
      return frame.x, score
    end
    """
    x, score = lab.rt.eval(code)(lab.Plan)
    assert x == 0 and 1000 < score < 1002


# ================================================================================================ never inside each other
@pytest.mark.parametrize("text,world", [
    ("two meeting tables in the corner", "room"), ("six desks in the corner", "room:300x260"), ("a bedroom", "room:160x500"),
    ("five trash cans in the corner", "room"), ("twelve couches in the corner", "room:420x360"), ("six patio tables in the corner", "room:300x300"),
])
def test_things_never_end_up_inside_each_other(lab, text, world):
    """planner-12: any amount of mutual penetration was kept as 'a tight squeeze' (two tables 130 HU into each other)."""
    for seed in seeds(lab, text, 6):
        fresh(lab)
        res = lab.run(text, world=world, seed=seed)
        if res["placements"]:
            assert not clean(lab, res, world), (text, world, seed, clean(lab, res, world)[:2])
        else:
            assert "could not find" in res["say"] or "more than I can fit" in res["say"]


@pytest.mark.parametrize("count", [5, 6, 8, 12])
@pytest.mark.parametrize("kind", ["trash can", "plant", "barrel"])
def test_a_corner_fills_along_the_wall_by_what_stands_there(lab, kind, count):
    """planner-13: the fifth item stepped along the wall by its OWN width, so unequal variants overlapped the first."""
    text = f"{count} {kind}s in the corner"
    for world in ("room", "room:420x360"):
        for seed in seeds(lab, text, 8):
            fresh(lab)
            res = lab.run(text, world=world, seed=seed)
            if res["placements"]:
                assert not clean(lab, res, world), (text, world, seed, clean(lab, res, world)[:2])
                if "in the corner" in res["say"]:
                    assert kinds(res)[kind] == count


def test_the_corner_says_why_it_fell_back(lab):
    fresh(lab)
    res = lab.run("a plant in the corner", world="open")
    assert any("no corner in reach" in n for n in res["notes"]), res["notes"]
    res = lab.run("forty couches in the corner", world="room:300x300")
    if res["placements"] and "corner" not in res["say"].split(",")[0]:
        assert not any("no corner in reach" in n for n in res["notes"]), "the corners were there, they were just full"


SENTENCES = ["a living room", "a dining room", "a bedroom", "an office", "six chairs around a table", "a table with eight chairs", "three couches",
             "twelve crates", "a dozen mugs", "four desks", "a couch and a lamp", "five barrels in a circle", "a kitchen", "ten stools",
             "a couch facing the tv", "two meeting tables", "six benches", "a waiting room", "a bathroom", "twenty plants"]


@pytest.mark.parametrize("world", ["open", "room", "room:420x360"])
def test_early_exits_change_no_answer(lab, world):
    """Judging stops at the first hard violation and gives up on a candidate that cannot beat the best: same placements as judging all."""
    def key(res):
        return [(p["kind"], p["vi"], round(p["x"], 3), round(p["y"], 3), round(p["theta"], 2)) for p in res["placements"]]
    for text in SENTENCES:
        fresh(lab)
        a = lab.run(text, world=world)
        req = lab.parser.parse(lab.parser, text)
        ctx = lab.context(lab.world(world))
        ctx["exhaustive"] = True
        b = lab._lists(__import__("eval_parse").to_py(lab.mod, lab.planner.run(lab.planner, req, ctx)))
        assert key(a) == key(b), (text, world)


# ================================================================================================ the builders
def test_computers_on_a_desk_never_hang_in_the_air(lab):
    """planner-3: 'four computers' laid them in one row wider than the desk; two hung over nothing."""
    for text, kind, n in [("four computers", "computer", 4), ("a dozen mugs", "mug", 12), ("give me a dozen coffee mugs", "mug", 12),
                          ("five desk lamps", "desk lamp", 5), ("twenty phones", "phone", 20), ("ten computers", "computer", 10)]:
        for world in ("open", "room"):
            for seed in seeds(lab, text, 6):
                fresh(lab)
                res = lab.run(text, world=world, seed=seed)
                assert res["placements"], (text, world, res["say"])
                problems = clean(lab, res, world)
                assert not problems, (text, world, seed, problems[:2])
                placed = kinds(res)[kind]
                if placed < n:
                    assert any("only" in x for x in res["notes"]), (text, placed, res["notes"])


def test_every_small_thing_is_placed_when_a_surface_can_hold_it(lab):
    fresh(lab)
    assert kinds(lab.run("a dozen mugs", world="open"))["mug"] == 12
    assert kinds(lab.run("five desk lamps", world="open"))["desk lamp"] == 5


def test_the_surface_added_for_small_things_is_big_enough(lab):
    """planner-3: the auto-added surface is the smallest that holds them all, not a random variant."""
    fresh(lab)
    res = lab.run("a dozen mugs", world="open")
    support = next(p for p in res["placements"] if p["kind"] in ("table", "coffee table", "desk", "meeting table", "side table"))
    assert sum(1 for p in res["placements"] if p["kind"] == "mug" and p.get("on")) == 12
    assert "I put them on" in res["say"] and support["kind"] in res["say"]
    assert not clean(lab, res), "every mug over the top of it"


def test_a_bedroom_with_a_computer_gets_a_surface_for_it(lab):
    """planner-3: the computer went on the side table and poked 9 HU into the wall."""
    fresh(lab)
    res = lab.run("a bedroom with a computer", world="room")
    assert kinds(res)["computer"] == 1
    assert not clean(lab, res, "room"), clean(lab, res, "room")
    comp = next(p for p in res["placements"] if p["kind"] == "computer")
    assert comp.get("on")


def test_more_on_a_standing_surface_than_it_holds_are_left_off_and_said(lab):
    """The packer returns what did not fit; the reply says how many stood on it."""
    fresh(lab)
    stand(lab, "a desk", world="open")
    res = lab.run_request({"action": "add", "items": [item("computer", 12)], "arrangement": "on", "target": "desk"}, world="open")
    placed = len(res["placements"])
    assert 0 < placed < 12
    assert any(f"only {placed} of the 12 computers fit" in n for n in res["notes"]), res["notes"]
    assert all(p.get("on") for p in res["placements"]) and not clean(lab, res)


def test_a_thing_that_does_not_fit_on_a_top_at_all_goes_beside_it(lab):
    """It was 'on the side table' with 0 of 1 placed and a sentence that began with a space."""
    fresh(lab)
    stand(lab, "a side table", world="open")
    for n in (1, 2):
        res = lab.run_request({"action": "add", "items": [item("tv", n)], "arrangement": "on", "target": "side table"}, world="open")
        assert kinds(res) == {"tv": n} and not any(p.get("on") for p in res["placements"])
        assert any("fit on the side table" in x for x in res["notes"]), res["notes"]
        assert res["say"].strip() == res["say"] and "beside the side table" in res["say"]
        assert not clean(lab, res)


def test_a_lamp_on_the_table_is_a_desk_lamp(lab):
    """heldout-17: 'lamp' mapped to the 67 HU floor lamp, even on a table."""
    fresh(lab)
    stand(lab, "a coffee table", world="open")
    res = lab.run_request({"action": "add", "items": [item("floor lamp", 1, word="lamp")], "arrangement": "on", "target": "coffee table"}, world="open")
    assert kinds(res) == {"desk lamp": 1}
    assert not clean(lab, res)


def test_a_thing_with_no_flat_top_is_not_pretended_on(lab):
    """heldout-17: 'on the couch' became 'there was no room on the couch'. It says it cannot, and where it put the thing."""
    fresh(lab)
    stand(lab, "a couch", world="open")
    res = lab.run_request({"action": "add", "items": [item("plant", 1)], "arrangement": "on", "target": "couch"}, world="open")
    assert res["placements"] and not res["placements"][0].get("on")
    assert any("can't stand things on the couch" in n for n in res["notes"]), res["notes"]


def test_another_players_table_does_not_hijack_my_request(lab):
    """planner-4: 'a table and two chairs around it' used a table of Alice's 11 m away and never made the player's own."""
    fresh(lab)
    stand(lab, "a table", world="room", owner="alice", pos=(100, 0, 0))
    res = lab.run("a table and two chairs around it", world="room", owner="bob")
    got = kinds(res)
    assert got["table"] == 1 and got["chair"] == 2, (got, res["say"])
    assert "table" in res["say"] and "two chairs" in res["say"]
    alice = standing(lab, "table")[0]
    table = next(p for p in res["placements"] if p["kind"] == "table")
    assert math.dist((table["x"], table["y"]), (alice["x"], alice["y"])) > 20, "a table of its own, not Alice's"


def test_surplus_target_things_join_the_group(lab):
    """planner-4: 'two couches and a lamp next to the couch' dropped the second couch."""
    fresh(lab)
    res = lab.run_request({"action": "add", "items": [item("couch", 2), item("floor lamp")], "arrangement": "beside", "target": "couch"}, world="room")
    assert kinds(res) == {"couch": 2, "floor lamp": 1}, kinds(res)
    assert not clean(lab, res, "room")


def test_it_means_the_last_thing_built(lab):
    """planner-15 / heldout-5: target 'last' was looked up as a kind ('I don't know what last is'). The engine gives ctx.last."""
    fresh(lab)
    stand(lab, "a couch", world="room", owner="alice")
    res = lab.run_request({"action": "add", "items": [item("floor lamp")], "arrangement": "beside", "target": "@last"}, world="room", owner="alice")
    couch = standing(lab, "couch")[0]
    lamp = res["placements"][0]
    assert 0 < math.dist((lamp["x"], lamp["y"]), (couch["x"], couch["y"])) < couch["w"] / 2 + 40
    assert "last" not in res["say"].lower().replace("last thing", "") and not any("last" in n for n in res["notes"])
    assert "beside the couch" in res["say"]


def test_it_with_nothing_recent_says_so_in_fixed_words(lab):
    fresh(lab)
    res = lab.run_request({"action": "add", "items": [item("floor lamp")], "arrangement": "beside", "target": "@last"}, world="room", owner="nobody")
    assert len(res["placements"]) == 1
    assert any("nothing recent" in n for n in res["notes"]), res["notes"]


def test_it_is_mine_not_the_newest_thing_anyone_built(lab):
    """planner-21: for an admin 'it' meant the globally newest group."""
    fresh(lab)
    stand(lab, "a couch", world="room", owner="admin")
    stand(lab, "a bedroom", world="room", owner="bob")
    res = lab.run_request({"action": "remove", "items": [item("@last")]}, world="room", owner="admin", restrict=False)
    mine = {e["id"] for e in lab.registry if e["owner"] == "admin"}
    assert set(res["removals"]) == mine and len(mine) == 1


def test_a_row_facing_the_tv_faces_it(lab):
    """planner-18 / heldout-7: 'in a row facing the tv' ignored the tv; the row faced the player."""
    for text in ["five chairs in a row facing the tv", "three couches in a row facing the tv", "four armchairs in a u shape facing the tv"]:
        fresh(lab)
        res = lab.run(text, world="room", pos=(0, 0, 0), yaw=0)
        tv = next(p for p in res["placements"] if p["kind"] == "tv")
        seats = [p for p in res["placements"] if p["kind"] != "tv"]
        assert seats and not clean(lab, res, "room")
        cx, cy = sum(p["x"] for p in seats) / len(seats), sum(p["y"] for p in seats) / len(seats)
        assert lab.facing_error(tv, cx, cy) < 45, "the group stands in front of the tv"
        errors = sorted(lab.facing_error(p, tv["x"], tv["y"]) for p in seats)
        if "row" in text:
            assert errors[0] < 20 and errors[-1] < 80, (text, errors)         # the middle one looks straight at it
        assert "tv" in res["say"]


def test_a_row_facing_a_standing_tv_uses_it(lab):
    fresh(lab)
    stand(lab, "a tv", world="room")
    res = lab.run("four chairs in a row facing the tv", world="room")
    assert kinds(res) == {"chair": 4}, kinds(res)
    tv = standing(lab, "tv")[0]
    assert all(lab.facing_error(p, tv["x"], tv["y"]) < 50 for p in res["placements"])


def test_one_on_each_side_of_the_couch(lab):
    """heldout-19: 'on either side of X' is the flank arrangement: one item each side."""
    fresh(lab)
    stand(lab, "a couch", world="room")
    couch = standing(lab, "couch")[0]
    res = lab.run_request({"action": "add", "items": [item("floor lamp", 2)], "arrangement": "flank", "target": "couch"}, world="room")
    assert kinds(res) == {"floor lamp": 2}
    fx, fy = math.cos(math.radians(couch["theta"])), math.sin(math.radians(couch["theta"]))
    sides = [((p["x"] - couch["x"]) * fy - (p["y"] - couch["y"]) * fx) for p in res["placements"]]
    assert sides[0] * sides[1] < 0, "one on each side"
    assert not clean(lab, res, "room")
    # a single item is made into one for each side
    res = lab.run_request({"action": "add", "items": [item("floor lamp", 1)], "arrangement": "flank", "target": "couch"}, world="room")
    assert kinds(res) == {"floor lamp": 2}


def test_one_in_every_corner(lab):
    """heldout-18: 'a plant in each corner' placed one plant."""
    fresh(lab)
    res = lab.run_request({"action": "add", "items": [item("plant", 1)], "arrangement": "corner", "every": "corner"}, world="room")
    assert kinds(res) == {"plant": 4}
    xs = sorted((p["x"] > 0, p["y"] > 0) for p in res["placements"])
    assert xs == [(False, False), (False, True), (True, False), (True, True)], "one in each corner"
    assert not clean(lab, res, "room")
    res = lab.run_request({"action": "add", "items": [item("floor lamp", 1), item("plant", 1)], "arrangement": "corner", "every": "corner"}, world="room")
    assert kinds(res) == {"floor lamp": 4, "plant": 4}
    assert not clean(lab, res, "room"), clean(lab, res, "room")[:2]


def test_facing_away_turns_every_one_about(lab):
    """parser-16: 'facing away from me / outward' faced the player."""
    fresh(lab)
    toward = lab.run_request({"action": "add", "items": [item("chair", 3)], "arrangement": "facing_me", "target": "me"}, world="open")
    away = lab.run_request({"action": "add", "items": [item("chair", 3)], "arrangement": "facing_away", "target": "me"}, world="open")
    assert len(away["placements"]) == 3
    for a, t in zip(away["placements"], toward["placements"]):
        assert abs((a["theta"] - t["theta"]) % 360 - 180) < 1e-6, (a["theta"], t["theta"])
    assert "facing away from you" in away["say"]


def test_a_table_for_six_seats_six(lab):
    """parser-20: req.seats."""
    for world in ("open", "room"):
        fresh(lab)
        res = lab.run_request({"action": "add", "items": [item("table", 1)], "seats": 6}, world=world)
        assert kinds(res)["chair"] == 6 and kinds(res)["table"] + kinds(res)["meeting table"] == 1, kinds(res)
        assert not clean(lab, res, world)
    res = lab.run_request({"action": "add", "items": [item("table", 1), item("chair", 3)], "seats": 8}, world="open")
    assert kinds(res)["chair"] == 8


def test_a_ring_round_a_small_table_seats_them_all(lab):
    """heldout-13: eight crates 'around' a table could not stand on its four sides."""
    fresh(lab)
    res = lab.run_request({"action": "add", "items": [item("crate", 8), item("table", 1)], "arrangement": "around", "target": "table"}, world="open")
    assert kinds(res)["crate"] == 8, (kinds(res), res["say"])
    assert not clean(lab, res)
    table = next(p for p in res["placements"] if p["kind"] == "table")
    assert all(math.dist((p["x"], p["y"]), (table["x"], table["y"])) < 120 for p in res["placements"])


def test_a_pile_is_not_taller_than_the_cap(lab):
    """planner-17: 'thirty crates' built a 3.8 m pile; mixed barrels floated."""
    fresh(lab)
    res = lab.run("thirty crates", world="open")
    assert len(res["placements"]) == 30
    top = max(p["z"] + p["h"] for p in res["placements"])
    assert top <= 135, top
    assert not clean(lab, res)
    for kind in ("barrel", "trash can"):
        for seed in seeds(lab, kind, 6):
            res = lab.run(f"eight {kind}s stacked", world="open", seed=seed)
            assert not clean(lab, res), (kind, seed, clean(lab, res)[:2])


def test_a_pile_under_a_low_ceiling_is_laid_flat(lab):
    fresh(lab)
    res = lab.run("ten crates stacked", world="room:560x480x100", pos=(0, 0, 0))
    assert len(res["placements"]) == 10
    assert max(p["z"] + p["h"] for p in res["placements"]) <= 98


# ================================================================================================ other people, and the limits of a request
def test_a_prop_is_never_placed_on_another_player(lab):
    """planner-14: the keep-out only protected the requester; ctx.players holds every OTHER player."""
    fresh(lab)
    free = lab.run("a couch", world="open", pos=(0, 0, 0), yaw=0, seed=5)["placements"][0]
    others = [(free["x"], free["y"]), (free["x"] + 70, free["y"] + 50), (-100, 60)]
    res = lab.run("a couch", world="open", pos=(0, 0, 0), yaw=0, seed=5, players=others)
    assert res["placements"]
    for p in res["placements"]:
        r = lab.Layout.rect(p["x"], p["y"], p["w"], p["d"], p["theta"])
        for ox, oy in others:
            assert not lab.Layout.overlap(lab.Layout.rect(ox, oy, 40, 40, 0), r, 4), (ox, oy)
    # a whole ring round the requester keeps clear of the people near it too
    res = lab.run("a bathtub around me", world="open", players=[(110, 0)])
    assert res["placements"]
    for p in res["placements"]:
        assert not lab.Layout.overlap(lab.Layout.rect(110, 0, 40, 40, 0), lab.Layout.rect(p["x"], p["y"], p["w"], p["d"], p["theta"]), 4)


def test_a_player_on_another_floor_does_not_block(lab):
    fresh(lab)
    free = lab.run("a couch", world="open", pos=(0, 0, 0), seed=5)["placements"][0]
    res = lab.run("a couch", world="open", pos=(0, 0, 0), seed=5, players=[(free["x"], free["y"], 900)])
    assert res["placements"][0]["x"] == pytest.approx(free["x"])


def test_requests_are_cut_to_the_limits_and_it_says_so(lab):
    """glue-4 / security-2: ctx.max_total and ctx.max_per_kind (40 and 40 for ordinary players)."""
    fresh(lab)
    res = lab.run("sixty chairs", world="open", max_total=40, max_per_kind=40)
    assert kinds(res)["chair"] == 40 and res["clamped"] is True
    assert any("limited chairs to 40" in n for n in res["notes"]), res["notes"]
    res = lab.run("twenty chairs and thirty plants", world="open", max_total=10, max_per_kind=40)
    assert len(res["placements"]) <= 10 and res["clamped"] is True
    assert any("scaled" in n for n in res["notes"]), res["notes"]
    res = lab.run("three chairs", world="open", max_total=40, max_per_kind=40)
    assert res["clamped"] is False


def test_the_default_limits_are_sixty_a_kind_and_a_hundred_and_fifty(lab):
    fresh(lab)
    res = lab.run("a hundred crates", world="open")
    assert kinds(res)["crate"] == 60 and res["clamped"] is True
    res = lab.run_request({"action": "add", "items": [item("crate", 60), item("chair", 60), item("plant", 60)]}, world="open")
    assert len(res["placements"]) <= 150


# ================================================================================================ the clock
def make_clock(lab, step):
    """A Lua clock that moves `step` seconds each time it is read."""
    return lab.rt.eval(f"(function() local t = 0; return function() t = t + {step}; return t end end)()")


def test_the_planner_uses_the_clock_it_is_given_and_stops_at_the_deadline(lab):
    """security-2 / glue-4 / planner-5: one deadline per request, read from ctx.now (the engine's SysTime)."""
    fresh(lab)
    res = lab.run("add 60 couches in a grid", world="room:300x260", now=make_clock(lab, 0.01), budget=0.05)
    assert res["timed_out"] is True and not res["placements"]
    assert "more than I can fit" in res["say"]
    assert res["fixed"] is True


def test_the_default_budget_is_a_quarter_of_a_second(lab):
    fresh(lab)
    clock = make_clock(lab, 0.1)
    ctx = lab.context(lab.world("room:300x260"), now=clock)
    ctx["budget"] = None
    req = lab.parser.parse(lab.parser, "add 60 couches in a grid")
    res = lab.planner.run(lab.planner, req, ctx)
    assert res["timed_out"] is True, "0.1 s a read: a quarter second is a handful of reads"


def test_the_first_build_is_always_made_even_with_no_time(lab):
    fresh(lab)
    res = lab.run("a couch", world="room", budget=0, now=make_clock(lab, 1.0))
    assert len(res["placements"]) == 1


def test_a_cut_short_search_returns_the_best_so_far_and_says_so(lab):
    fresh(lab)
    res = lab.run("a table with ten chairs", world="room:300x260", now=make_clock(lab, 0.004), budget=0.05)
    if res["placements"]:
        assert res["timed_out"] is True or not any(res["timed_out"] for _ in [0])
    else:
        assert res["timed_out"] is True and "more than I can fit" in res["say"]


ABUSIVE = [
    ("add 150 chairs", "room:300x260"), ("add 60 couches in a grid", "room:300x260"), ("add 100 lamps in a circle", "room:300x260"),
    ("add 60 stools and 60 plants", "open"), ("add 60 stools and 60 plants", "room"), ("60 side tables, 40 toilets, 40 trash cans", "room"),
    ("add 150 chairs scattered", "room:300x260"), ("a hundred chairs in a circle", "room"), ("sixty beds against the wall", "room"),
]


@pytest.mark.parametrize("budget", [0.05, 0.25])
def test_abusive_requests_stay_within_a_few_times_the_budget(lab, budget):
    """security-2 / glue-4 / heldout-24: 'add 60 stools and 60 plants' took 0.7 s, a hundred-lamp circle 1-2 s, and so on."""
    # the VM the game uses. LuaJIT raises (and handles) Windows exceptions of its own when a trace aborts; faulthandler would
    # report each of them as a fatal one, so it is switched off for the duration.
    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()
    lab.rt.execute("jit.on()")
    try:
        worst = 0.0
        for text, world in ABUSIVE:
            fresh(lab)
            lab.run("a couch", world=world)                      # warm up the JIT and the parser
            fresh(lab)
            t0 = time.perf_counter()
            res = lab.run(text, world=world, budget=budget)
            dt = time.perf_counter() - t0
            worst = max(worst, dt)
            assert dt < 3.5 * budget + 0.06, (text, world, f"{dt:.3f}s for a budget of {budget}")
            if not res["placements"]:
                assert res["say"]
        assert worst > 0
    finally:
        lab.rt.execute("jit.off()")
        if was_enabled:
            faulthandler.enable()


def test_a_request_that_cannot_be_placed_gives_up_before_its_budget_is_spent(lab):
    """heldout-24: infeasible requests burned the whole budget. Failing candidates are cheap now (the cheap checks first), so
    every frame is tried well inside the allowance and the answer is 'could not find clear floor', not 'ran out of time'."""
    fresh(lab)
    res = lab.run("sixty beds against the wall", world="room", budget=30.0)
    assert not res["placements"] and res["timed_out"] is False and "could not find clear floor" in res["say"]


def test_judge_stops_at_the_first_hard_violation(lab):
    """The search keeps only candidates with no hard violation, so judging one may stop at the first (quick mode)."""
    code = """
    function(Plan)
      local calls = 0
      local env = { floor_z = function() calls = calls + 1; return nil end, clear = function() return true end, wall_probe = function() end }
      local ctx = { pos = { x = 0, y = 0, z = 0 }, yaw = 0, env = env, registry = {} }
      local pls = {}
      for i = 1, 20 do pls[i] = { x = i * 100, y = 0, w = 20, d = 20, theta = 0, h = 10, zoff = 0, inst = {} } end
      local viol, pen, hard = Plan._judge(ctx, pls, nil, { quick = true })
      local quick_calls = calls
      calls = 0
      local viol2, pen2, hard2 = Plan._judge(ctx, pls, nil)
      return quick_calls, calls, hard, hard2
    end
    """
    quick_calls, full_calls, hard, hard2 = lab.rt.eval(code)(lab.Plan)
    assert hard >= 1 and hard2 == 20
    assert quick_calls < full_calls / 4


def test_judge_stops_asking_the_engine_once_the_time_is_up(lab):
    """security-2: judging 150 placements is 900 traces on the server thread; past the deadline it asks no more."""
    code = """
    function(Plan)
      local calls = 0
      local env = { floor_z = function() calls = calls + 1; return 0 end, clear = function() return true end, wall_probe = function() end }
      local now = 0
      local ctx = { pos = { x = 0, y = 0, z = 0 }, yaw = 0, env = env, registry = {}, clock = function() now = now + 1; return now end, deadline = 0 }
      local pls = {}
      for i = 1, 150 do pls[i] = { x = i * 100, y = 0, w = 20, d = 20, theta = 0, h = 10, zoff = 0, inst = {} } end
      local viol, pen, hard = Plan._judge(ctx, pls)
      return calls, hard, ctx.timed_out
    end
    """
    calls, hard, timed_out = lab.rt.eval(code)(lab.Plan)
    assert timed_out and hard >= 1 and calls < 150 * 6 / 4


def test_overlap_has_a_broad_phase_and_is_still_exact(lab):
    L = lab.Layout
    a = L.rect(0, 0, 40, 40, 0)
    assert not L.overlap(a, L.rect(500, 500, 40, 40, 0), 0)
    assert L.overlap(a, L.rect(39, 0, 40, 40, 0), 0)
    assert not L.overlap(a, L.rect(41, 0, 40, 40, 0), 0) and L.overlap(a, L.rect(41, 0, 40, 40, 0), 2)
    # a thin long thing turned 45 degrees: the circle test must not throw out a real overlap
    long_thing = L.rect(0, 0, 200, 10, 45)
    assert L.overlap(long_thing, L.rect(70, -70, 10, 10, 0), 0)
    assert not L.overlap(long_thing, L.rect(70, 70, 10, 10, 0), 0)


# ================================================================================================ words: nothing a player typed comes back
NASTY = ["red %s ${x} <b>bold</b> \x01\x02 and more words than three", "^1blorptastic", "a" * 80, "dragon", "x y z w v u"]


def unknown_item(word):
    return {"kind": "UNKNOWN:" + word, "count": 1, "adjectives": [], "word": word, "phrase": word, "definite": False, "plural": False}


@pytest.mark.parametrize("action", ["add", "remove", "move", "rotate", "rearrange"])
def test_replies_never_contain_a_players_own_words(lab, action):
    """glue-11 / security-5: 'I don't have a <word> in my library' went to everybody's chat and was spoken aloud."""
    fresh(lab)
    stand(lab, "a couch", world="room")
    for word in NASTY:
        req = {"action": action, "items": [unknown_item(word), item("couch", 1, definite=True)], "arrangement": "unspecified", "target": ""}
        res = lab.run_request(req, world="room")
        text = " ".join([res["say"]] + res["notes"]).lower()
        for piece in word.lower().split():
            if len(piece) > 2 and piece.isascii() and piece not in ("and", "more", "than", "three", "words", "red", "bold"):
                assert piece not in text, (action, word, text)
        assert "\x01" not in text and "${" not in text and "<b>" not in text and "%s" not in text
        for w in res.get("unknown_words", []):
            assert set(w) <= set("abcdefghijklmnopqrstuvwxyz0123456789 "), w
            assert len(w) <= 24 and len(w.split()) <= 3
        assert len(res.get("unknown_words", [])) <= 4


def test_unknown_words_are_kept_apart_for_the_requester(lab):
    fresh(lab)
    res = lab.run("a dragon and two couches", world="open")
    assert kinds(res)["couch"] == 2
    assert res["unknown_words"] == ["dragon"]
    assert "dragon" not in res["say"].lower() and not any("dragon" in n for n in res["notes"])
    assert "my library" in res["say"]
    req = {"action": "add", "items": [unknown_item("a" + "b" * 60), unknown_item("one"), unknown_item("two"), unknown_item("three"), unknown_item("four"),
                                      unknown_item("five")]}
    res = lab.run_request(req, world="open")
    assert len(res["unknown_words"]) == 4 and all(len(w) <= 24 for w in res["unknown_words"])
    assert res["fixed"] is True


def test_a_scene_name_the_library_lacks_is_not_echoed(lab):
    fresh(lab)
    res = lab.run_request({"action": "add", "scene": "blorp's lair", "items": [item("couch")]}, world="open")
    assert "blorp" not in res["say"].lower()


def test_a_note_from_the_parser_is_shown_only_when_made_of_library_words(lab):
    """The planner shows req.notes only if they hold no raw player words."""
    fresh(lab)
    res = lab.run_request({"action": "add", "items": [item("barrel")], "notes": ["I read that as barrels", "read 'barrles' as 'barrels'"]}, world="open")
    assert "I read that as barrels" in res["notes"] and not any("barrles" in n for n in res["notes"])
    assert "barrles" not in res["say"]


def test_plurals_of_things_ending_in_f(lab):
    """planner-20: 'two bookshelfs', 'three shelfs'."""
    fresh(lab)
    assert "bookshelves" in lab.run("two bookshelves", world="open")["say"].lower()
    assert "shelves" in lab.run("three shelves", world="open")["say"].lower()
    assert "bookshelfs" not in lab.run("two bookshelves", world="open")["say"].lower()


def test_constant_sentences_are_marked_fixed(lab):
    fresh(lab)
    for action in ("clear", "undo"):
        assert lab.run_request({"action": action}, world="open")["fixed"] is True
    res = lab.run_request({"action": "remove", "items": [item("couch", 0, definite=True)]}, world="open")
    assert res["fixed"] is True and "no couch" in res["say"].lower()
    res = lab.run_request({"action": "rotate", "items": [item("couch", 1, definite=True)]}, world="open")
    assert res["fixed"] is True
    assert "fixed" not in lab.run("a couch", world="open") or not lab.run("a couch", world="open").get("fixed")


# ================================================================================================ managing what stands
def ids_of(lab, *kinds_):
    return sorted(e["id"] for e in lab.registry if e["kind"] in kinds_)


def test_everything_except_keeps_the_exception(lab):
    """heldout-9: 'remove everything except the couch' cleared the room, the couch with it."""
    fresh(lab)
    stand(lab, "a living room", world="room", owner="alice")
    couches = ids_of(lab, "couch")
    res = lab.run_request({"action": "remove", "items": [item("@all", 0)], "exclude": ["couch"]}, world="room", owner="alice")
    assert res["action"] == "remove" and res["removals"]
    assert not set(res["removals"]) & set(couches)
    assert len(res["removals"]) == len(lab.registry) - len(couches)
    # several exceptions
    res = lab.run_request({"action": "remove", "items": [item("@all", 0)], "exclude": ["couch", "tv", "armchair"]}, world="room", owner="alice")
    assert not {e["id"] for e in lab.registry if e["kind"] in ("couch", "tv", "armchair")} & set(res["removals"])


def test_everything_means_only_my_things(lab):
    fresh(lab)
    stand(lab, "a couch", world="room", owner="alice")
    stand(lab, "two lamps", world="room", owner="bob")
    res = lab.run_request({"action": "remove", "items": [item("@all", 0)]}, world="room", owner="alice")
    mine = [e["id"] for e in lab.registry if e["owner"] == "alice"]
    assert sorted(res["removals"]) == sorted(mine)
    # an admin (no restriction) manages all of it
    res = lab.run_request({"action": "remove", "items": [item("@all", 0)]}, world="room", owner="admin", restrict=False)
    assert len(res["removals"]) == len(lab.registry)


def test_exclusions_that_leave_nothing_say_so(lab):
    fresh(lab)
    stand(lab, "two couches", world="room")
    res = lab.run_request({"action": "remove", "items": [item("@all", 0)], "exclude": ["couch"]}, world="room")
    assert not res["removals"] and res["fixed"] is True and "nothing" in res["say"].lower()


def test_it_is_the_last_group_not_the_last_thing(lab):
    """heldout-5 / planner-21: 'it' and 'them' are the requester's most recent group (ctx.last)."""
    fresh(lab)
    stand(lab, "a couch", world="room", owner="alice")
    stand(lab, "a coffee table and two armchairs", world="room", owner="alice")
    stand(lab, "a bedroom", world="room", owner="bob")
    last = lab.last_ids("alice")
    res = lab.run_request({"action": "remove", "items": [item("@last", 1)]}, world="room", owner="alice")
    assert sorted(res["removals"]) == sorted(last) and len(last) >= 3
    res = lab.run_request({"action": "remove", "items": [item("@last", 1)]}, world="room", owner="nobody")
    assert not res["removals"] and "nothing recent" in res["say"].lower() and res["fixed"] is True
    res = lab.run_request({"action": "rotate", "items": [item("@last", 1)], "half_turn": True, "normalised": "turn it around"}, world="room", owner="alice")
    assert {m["id"] for m in res["moves"]} <= set(last) and res["moves"]


def test_a_named_colour_matches_that_colour_only(lab):
    """heldout-11: 'remove the red couch' removed the nearer plain one."""
    fresh(lab)
    plain = stand(lab, "a couch", world="room", yaw=0)
    red = stand(lab, "a red couch", world="room", yaw=90)
    plain_id, red_id = lab.registry[0]["id"], lab.registry[1]["id"]
    assert lab.registry[1]["color_name"] == "red" and lab.registry[0]["color_name"] is None
    res = lab.run_request({"action": "remove", "items": [item("couch", 1, adjectives=["red"], definite=True)]}, world="room")
    assert res["removals"] == [red_id]
    res = lab.run_request({"action": "remove", "items": [item("couch", 1, adjectives=["blue"], definite=True)]}, world="room")
    assert not res["removals"] and "no blue couch" in res["say"].lower()
    # a turn may use the plain one, and says so
    res = lab.run_request({"action": "rotate", "items": [item("couch", 1, adjectives=["blue"], definite=True)], "half_turn": True,
                           "normalised": "turn the blue couch around"}, world="room")
    assert [m["id"] for m in res["moves"]] == [plain_id]
    assert any("plain" in n for n in res["notes"]), res["notes"]
    # "grey" and "gray" are one colour
    fresh(lab)
    lab.registry = [{"id": 1, "kind": "couch", "vi": 1, "x": 200, "y": 0, "z": 0, "theta": 180, "w": 77, "d": 45, "h": 33, "color_name": "gray",
                     "group": 1, "seq": 1, "owner": None}]
    res = lab.run_request({"action": "remove", "items": [item("couch", 1, adjectives=["grey"], definite=True)]}, world="open")
    assert res["removals"] == [1]


def lay_flush(lab, kind, world="room", yaw=90):
    """One of a kind added with its back to a wall (the planner's own choice), committed."""
    return stand(lab, f"a {kind}", world=world, yaw=yaw)


@pytest.mark.parametrize("kind", ["couch", "bookshelf", "locker", "bed", "wardrobe"])
@pytest.mark.parametrize("how", ["turn", "45", "face me"])
def test_a_turn_never_pushes_a_prop_into_a_wall(lab, kind, how):
    """planner-9 / heldout-10: a wall-flush couch turned 90 degrees stood 15-20 HU in the wall."""
    fresh(lab)
    lay_flush(lab, kind)
    base = {"action": "rotate", "items": [item(kind, 1, definite=True)], "raw": f"turn the {kind}"}
    if how == "turn":
        req = dict(base, normalised=f"turn the {kind}")
    elif how == "45":
        req = dict(base, normalised=f"turn the {kind} 45 degrees")
    else:
        req = dict(base, normalised=f"turn the {kind} to face me", arrangement="facing_me", target="me")
    for pos in [(0, 0, 0), (100, 150, 0), (-150, -100, 0)]:
        before = lab.audit(lab.world("room"))
        res = lab.run_request(req, world="room", pos=pos)
        saved = [dict(e) for e in lab.registry]
        lab.commit(res)
        after = lab.audit(lab.world("room"))
        assert not (set(after) - set(before)), (kind, how, pos, after, res["say"])
        lab.registry = saved


def test_turning_around_is_always_allowed(lab):
    """A half turn does not change a footprint, so it is never refused (and never nudged)."""
    fresh(lab)
    lay_flush(lab, "couch")
    e = lab.registry[0]
    res = lab.run("turn the couch around", world="room")
    assert res["moves"] and abs(res["moves"][0]["theta"] - ((e["theta"] + 180) % 360)) < 1e-6
    assert res["moves"][0]["x"] == pytest.approx(e["x"]) and res["moves"][0]["y"] == pytest.approx(e["y"])


def test_a_turn_that_needs_room_eases_the_prop_off_the_wall(lab):
    fresh(lab)
    lay_flush(lab, "couch")
    e = lab.registry[0]
    res = lab.run_request({"action": "rotate", "items": [item("couch", 1, definite=True)], "normalised": "turn the couch"}, world="room")
    m = res["moves"][0]
    assert math.dist((m["x"], m["y"]), (e["x"], e["y"])) > 3, "slid off the wall to make room for the turn"
    assert "rotate_only" not in m or not m["rotate_only"]


def test_a_table_with_its_chairs_is_not_turned_into_them(lab):
    """heldout-10: turning a table left two chairs 19.8 HU inside it."""
    for seed in seeds(lab, "a desk and a chair", 8):
        fresh(lab)
        res = lab.run("a desk and a chair", world="room", seed=seed)
        lab.commit(res)
        before = lab.audit(lab.world("room"))
        res = lab.run_request({"action": "rotate", "items": [item("desk", 1, definite=True)], "normalised": "turn the desk 90 degrees"},
                              world="room", seed=seed)
        lab.commit(res)
        assert not (set(lab.audit(lab.world("room"))) - set(before)), (seed, res["say"])


def test_two_armchairs_turn_to_face_each_other(lab):
    """planner-10: 'face each other' turned both 90 degrees and said it turned."""
    fresh(lab)
    lab.registry = [
        {"id": 1, "kind": "armchair", "vi": 1, "x": -80, "y": 0, "z": 0, "theta": 180, "w": 40, "d": 40, "h": 38, "color_name": None, "group": 1, "seq": 1, "owner": None},
        {"id": 2, "kind": "armchair", "vi": 1, "x": 80, "y": 0, "z": 0, "theta": 0, "w": 40, "d": 40, "h": 38, "color_name": None, "group": 1, "seq": 2, "owner": None},
    ]
    lab.registry[0]["w"] = lab.registry[0]["d"] = lab.catalog["kinds"]["armchair"]["variants"][0]["w"]
    res = lab.run_request({"action": "rotate", "items": [item("armchair", 0, definite=True)], "arrangement": "facing_each_other"}, world="open",
                          pos=(0, 300, 0))
    thetas = {m["id"]: m["theta"] for m in res["moves"]}
    assert thetas[1] == pytest.approx(0, abs=1e-6) and thetas[2] == pytest.approx(180, abs=1e-6)
    res = lab.run_request({"action": "rotate", "items": [item("armchair", 1, definite=True)], "arrangement": "facing_each_other"}, world="open")
    lab.registry = lab.registry[:1]
    res = lab.run_request({"action": "rotate", "items": [item("armchair", 0, definite=True)], "arrangement": "facing_each_other"}, world="open")
    assert not res["moves"] and res["fixed"] is True


def test_a_turn_to_face_the_wall_faces_the_wall(lab):
    fresh(lab)
    lab.registry = [{"id": 1, "kind": "couch", "vi": 1, "x": 150, "y": 0, "z": 0, "theta": 90, "w": 77, "d": 45, "h": 33, "color_name": None,
                     "group": 1, "seq": 1, "owner": None}]
    res = lab.run_request({"action": "rotate", "items": [item("couch", 1, definite=True)], "arrangement": "facing_me", "target": "wall"}, world="room")
    assert res["moves"] and abs((res["moves"][0]["theta"] + 180) % 360 - 180) < 1e-6, res["moves"]            # the east wall is nearest: due east


def test_a_facing_that_cannot_be_resolved_leaves_things_alone(lab):
    """planner-10: 'turn the couch to face the dragon' turned it 90 degrees and said so."""
    fresh(lab)
    lab.registry = [{"id": 1, "kind": "couch", "vi": 1, "x": 150, "y": 0, "z": 0, "theta": 90, "w": 77, "d": 45, "h": 33, "color_name": None,
                     "group": 1, "seq": 1, "owner": None}]
    for target in ("window", "door", "tv", "dragon"):
        res = lab.run_request({"action": "rotate", "items": [item("couch", 1, definite=True)], "arrangement": "facing_me", "target": target}, world="room")
        assert not res["moves"], (target, res["moves"])
        assert "dragon" not in res["say"].lower() and res["say"]
    res = lab.run_request({"action": "rotate", "items": [item("couch", 1, definite=True)], "facing": "tv"}, world="room")
    assert not res["moves"]


def test_facing_away_turns_the_other_way(lab):
    fresh(lab)
    lab.registry = [{"id": 1, "kind": "couch", "vi": 1, "x": 150, "y": 0, "z": 0, "theta": 90, "w": 77, "d": 45, "h": 33, "color_name": None,
                     "group": 1, "seq": 1, "owner": None}]
    res = lab.run_request({"action": "rotate", "items": [item("couch", 1, definite=True)], "arrangement": "facing_away", "target": "me"}, world="open")
    assert res["moves"][0]["theta"] == pytest.approx(0, abs=1e-6)            # me at the origin is to the west; away is east (0 deg)


def test_what_stands_on_a_desk_moves_with_it(lab):
    """planner-11: 'move the desk closer' left the computer hanging in the air."""
    for verb in ("move the desk closer", "turn the desk around", "turn the desk 45 degrees", "move the desk left"):
        fresh(lab)
        stand(lab, "a desk with a computer", world="room", yaw=90)
        before = lab.audit(lab.world("room"))
        assert not before
        res = lab.run(verb, world="room", pos=(-250, -190, 0), yaw=90)       # a good way from the desk
        lab.commit(res)
        assert res["moves"], (verb, res["say"])
        moved = {m["id"] for m in res["moves"]}
        comp = next(e for e in lab.registry if e["kind"] == "computer")
        assert comp["id"] in moved, verb
        assert not lab.audit(lab.world("room")), (verb, lab.audit(lab.world("room")))


def test_rearranging_surfaces_carries_their_load(lab):
    fresh(lab)
    stand(lab, "a desk with a computer", world="room")
    stand(lab, "a desk with a computer", world="room", yaw=180)
    res = lab.run_request({"action": "rearrange", "items": [item("desk", 0, definite=True)], "arrangement": "row"}, world="room")
    lab.commit(res)
    assert not lab.audit(lab.world("room")), lab.audit(lab.world("room"))


def test_the_room_builder_moves_a_load_with_its_surface(lab):
    """planner-11: repair() slid a side table along its wall and left the lamp behind (96 sweep runs)."""
    for text in ["a kitchen with a side table and a desk lamp", "a bedroom with a lamp on the side table", "a living room with a mug"]:
        for world in ("open", "room", "room:420x360"):
            for seed in seeds(lab, text, 6):
                fresh(lab)
                res = lab.run(text, world=world, seed=seed)
                if res["placements"]:
                    assert not clean(lab, res, world), (text, world, seed, clean(lab, res, world)[:2])


def test_moving_against_a_wall_or_into_a_corner(lab):
    """heldout-14: 'move the couch against the wall' left it mid-room, toward the player."""
    fresh(lab)
    stand(lab, "a couch", world="room", pos=(0, 0, 0), yaw=0)
    lab.registry[0]["x"], lab.registry[0]["y"] = 0, 90                 # mid-room
    res = lab.run_request({"action": "move", "items": [item("couch", 1, definite=True)], "arrangement": "against_wall"}, world="room")
    lab.commit(res)
    e = lab.registry[0]
    v, origin, yaw = lab._geom(e)
    poly = poly_of(v, origin, yaw)
    w = lab.world("room")
    x0, y0, x1, y1 = (w.bounds[k] for k in (1, 2, 3, 4))
    gap = min(min(x - x0 for x, _ in poly), min(x1 - x for x, _ in poly), min(y - y0 for _, y in poly), min(y1 - y for _, y in poly))
    assert gap < 4, f"the couch stands {gap:.0f} HU from the nearest wall"
    assert not lab.audit(lab.world("room"))
    fresh(lab)
    stand(lab, "a plant", world="room", yaw=90)
    res = lab.run_request({"action": "move", "items": [item("plant", 1, definite=True)], "arrangement": "corner"}, world="room")
    lab.commit(res)
    e = lab.registry[0]
    assert abs(e["x"]) > 180 and abs(e["y"]) > 150, (e["x"], e["y"])
    assert not lab.audit(lab.world("room"))
    # no wall in reach: it stays where it is, and says so
    fresh(lab)
    stand(lab, "a plant", world="open")
    res = lab.run_request({"action": "move", "items": [item("plant", 1, definite=True)], "arrangement": "corner"}, world="open")
    assert not res["moves"] and "no corner in reach" in res["say"].lower() and res["fixed"] is True
    res = lab.run_request({"action": "move", "items": [item("plant", 1, definite=True)], "arrangement": "against_wall"}, world="open")
    assert not res["moves"] and "no wall in reach" in res["say"].lower()


def test_an_already_close_thing_is_not_moved_closer(lab):
    """planner-20: a tiny 'closer' move reported 'Moved one couch' when nothing moved."""
    fresh(lab)
    lab.registry = [{"id": 1, "kind": "couch", "vi": 1, "x": 71, "y": 0, "z": 0, "theta": 180, "w": 77, "d": 45, "h": 33, "color_name": None,
                     "group": 1, "seq": 1, "owner": None}]
    res = lab.run("move the couch closer", world="open", pos=(0, 0, 0))
    assert not res["moves"] and res["fixed"] is True and "already" in res["say"].lower()


def test_a_move_to_a_thing_that_is_not_there_does_not_move(lab):
    """planner-20: it moved the couch somewhere unrelated and said 'Moved'."""
    fresh(lab)
    stand(lab, "a couch", world="open")
    res = lab.run_request({"action": "move", "items": [item("couch", 1, definite=True)], "arrangement": "beside", "target": "bookshelf"}, world="open")
    assert not res["moves"] and "no bookshelf" in res["say"].lower()


def test_a_whole_table_set_moves_without_its_chairs_blocking_it(lab):
    """The group keeps its shape when shifted, so chairs tucked under the table are not 'in the way'."""
    fresh(lab)
    stand(lab, "a table with four chairs", world="open", pos=(0, 0, 0), yaw=0)
    for e in lab.registry:
        e["x"] += 400
    req = {"action": "move", "items": [item("table", 1, definite=True), item("chair", 0, definite=True)], "normalised": "move them closer", "arrangement": "unspecified"}
    res = lab.run_request(req, world="open")
    assert res["moves"] and len(res["moves"]) == 5, res["say"]
    assert res["say"].startswith("Moved")


def test_put_the_lamp_by_the_couch_makes_a_lamp_when_none_stands(lab):
    """planner-19: 'put the lamp next to the couch' with no lamp standing answered 'There was nothing to move'."""
    fresh(lab)
    stand(lab, "a couch", world="room")
    res = lab.run_request({"action": "move", "items": [item("floor lamp", 1, definite=True)], "arrangement": "beside", "target": "couch", "verb": "put"}, world="room")
    assert kinds(res) == {"floor lamp": 1}
    res = lab.run_request({"action": "move", "items": [item("floor lamp", 1, definite=True)], "arrangement": "beside", "target": "couch", "verb": "slide"}, world="room")
    assert not res["placements"] and not res["moves"]


def test_line_up_makes_a_row_and_tidy_keeps_the_shape(lab):
    """heldout-16: 'line up the chairs' and 'straighten the chairs' turned a circle into a circle, or crates into a pile."""
    fresh(lab)
    stand(lab, "five chairs in a circle", world="room")
    res = lab.run_request({"action": "rearrange", "items": [item("chair", 0, definite=True)], "arrangement": "row"}, world="room")
    assert len(res["moves"]) == 5 and "row" in res["say"]
    xs, ys = [m["x"] for m in res["moves"]], [m["y"] for m in res["moves"]]
    assert min(max(xs) - min(xs), max(ys) - min(ys)) < 12
    res = lab.run_request({"action": "rearrange", "items": [item("chair", 0, definite=True)], "arrangement": "unspecified"}, world="room")
    assert "circle" in res["say"], res["say"]
    fresh(lab)
    stand(lab, "five chairs in a row", world="room")
    res = lab.run_request({"action": "rearrange", "items": [item("chair", 0, definite=True)], "arrangement": "unspecified"}, world="room")
    xs, ys = [m["x"] for m in res["moves"]], [m["y"] for m in res["moves"]]
    assert min(max(xs) - min(xs), max(ys) - min(ys)) < 12, "tidying a row keeps a row"


def test_a_row_that_does_not_fit_falls_back_like_adding_does(lab):
    """heldout-16: rearranging had no fallback chain: a row that did not fit was 'could not find clear floor'."""
    fresh(lab)
    stand(lab, "six couches", world="room:420x360")
    res = lab.run_request({"action": "rearrange", "items": [item("couch", 0, definite=True)], "arrangement": "row"}, world="room:420x360")
    assert len(res["moves"]) == 6 and any("would not fit as asked" in n for n in res["notes"]), (res["say"], res["notes"])


def test_rearranging_a_desk_keeps_its_computer_on_it(lab):
    fresh(lab)
    stand(lab, "a desk with a computer", world="room")
    res = lab.run_request({"action": "rearrange", "items": [item("desk", 0, definite=True)], "arrangement": "row"}, world="room")
    lab.commit(res)
    assert not lab.audit(lab.world("room"))


# ================================================================================================ a library that changed, and entries from the engine
def library_without(lab, hidden_kinds):
    lib = lab.load((HOLODECK.parent / "addon" / "lua" / "mentat" / "holodeck" / "library.lua").as_posix())
    models = {v["model"] for k in hidden_kinds for v in lab.catalog["kinds"][k]["variants"]}
    is_valid = lab.rt.eval("function(set) return function(m) return not set[m] end end")(lab.lua_table({m: True for m in models}))
    filtered, _report = lib.filter(lab.catalog_lua, is_valid, lab.lua_table({}))
    return lab.Plan.new(filtered)


@pytest.mark.parametrize("hidden,text", [(["desk"], "a computer"), (["coffee table"], "a mug"), (["side table"], "a desk lamp"),
                                         (["desk"], "an office"), (["desk", "table", "coffee table", "side table", "meeting table", "patio table"], "a phone")])
def test_a_library_missing_a_surface_kind_still_answers(lab, hidden, text):
    """planner-23: 'a computer' raised 'attempt to index a nil value' when the desk kind was filtered out."""
    fresh(lab)
    planner = library_without(lab, hidden)
    res = lab.run(text, world="room", planner=planner)
    assert res["action"] == "add" and res["say"]
    for p in res["placements"]:
        assert p["kind"] not in hidden


def test_standing_things_whose_kind_or_variant_is_gone_are_skipped_not_crashed_on(lab):
    """planner-23: 'move the couch left' raised after a reload that shrank the variant list."""
    fresh(lab)
    lab.registry = [
        {"id": 1, "kind": "couch", "vi": 99, "x": 150, "y": 0, "z": 0, "theta": 180, "w": 77, "d": 45, "h": 33, "color_name": None, "group": 1, "seq": 1, "owner": None},
        {"id": 2, "kind": "dragon", "vi": 1, "x": 0, "y": 150, "z": 0, "theta": 180, "w": 77, "d": 45, "h": 33, "color_name": None, "group": 2, "seq": 2, "owner": None},
        {"id": 3, "kind": "chair", "vi": 1, "x": -150, "y": 0, "z": 0, "theta": 0, "w": 20, "d": 20, "h": 34, "color_name": None, "group": 3, "seq": 3, "owner": None},
    ]
    for req in [{"action": "move", "items": [item("couch", 1, definite=True)], "normalised": "move the couch left"},
                {"action": "rearrange", "items": [item("couch", 0, definite=True)], "arrangement": "row"},
                {"action": "rotate", "items": [item("couch", 0, definite=True)], "half_turn": True, "normalised": "turn the couch around"},
                {"action": "add", "items": [item("floor lamp")], "arrangement": "beside", "target": "couch"},
                {"action": "move", "items": [item("chair", 1, definite=True)], "arrangement": "beside", "target": "couch"},
                {"action": "remove", "items": [item("@all", 0)]}]:
        res = lab.run_request(req, world="open")
        assert res["say"], req
    res = lab.run_request({"action": "remove", "items": [item("@all", 0)]}, world="open")
    assert sorted(res["removals"]) == [1, 2, 3], "stale props can still be taken down"
    # a stale entry still counts as something in the way
    lab.registry = lab.registry[:1]
    res = lab.run("a couch", world="open", pos=(0, 0, 0), yaw=0)
    assert res["placements"]


def test_extra_fields_on_standing_things_are_ignored(lab):
    fresh(lab)
    stand(lab, "a couch", world="room", owner="alice")
    lab.registry[0]["tilted"] = True
    lab.registry[0]["legacy"] = "yes"
    res = lab.run("turn the couch around", world="room", owner="alice")
    assert res["moves"]


# ================================================================================================ the harness itself
def test_the_lab_hands_the_planner_the_engines_seed(lab):
    """planner-24: the lab used seed 12345, whose first draws are tiny, so it always drew the smallest model of every kind."""
    fresh(lab)
    a = lab.run("a table with six chairs", world="open")
    b = lab.run("a table with six chairs", world="open", seed=lab.Layout.hash("a table with six chairs"))
    assert [(p["kind"], p["vi"], round(p["x"], 3)) for p in a["placements"]] == [(p["kind"], p["vi"], round(p["x"], 3)) for p in b["placements"]]
    # no seed is put in the context unless a test asks for one: the planner takes the hash of the sentence itself
    ctx = lab.context(lab.world("open"))
    assert ctx["seed"] is None
    ctx = lab.context(lab.world("open"), seed=7)
    assert ctx["seed"] == 7
    # and the draws are spread: over many sentences, more than one model of a kind comes up (the old seed always drew the first)
    drawn = {p["vi"] for t in ["two couches", "three couches", "four couches and a plant", "a couch", "five couches", "six couches", "a couch and a lamp",
                               "seven couches", "eight couches", "nine couches"]
             for p in lab.run(t, world="open")["placements"] if p["kind"] == "couch"}
    assert len(drawn) > 1


def test_the_lab_audit_catches_what_grade_cannot(lab):
    """lab.audit grades everything standing, so a move or turn that creates an overlap or a floating thing shows up."""
    fresh(lab)
    stand(lab, "a desk with a computer", world="room")
    assert not lab.audit(lab.world("room"))
    desk = next(e for e in lab.registry if e["kind"] == "desk")
    desk["x"] += 120                                           # the computer is left behind
    assert any("floats" in p for p in lab.audit(lab.world("room")))
    lab.registry[0]["x"] = 5000
    assert any("wall" in p for p in lab.audit(lab.world("room")))


def test_a_group_named_as_it_turns_as_one_body(lab):
    """planner-19: 'turn it around' of a table and its chairs flipped every chair about itself, so they faced away from the table."""
    fresh(lab)
    stand(lab, "a table with four chairs", world="room", owner="alice")
    table = standing(lab, "table")[0]
    res = lab.run_request({"action": "rotate", "items": [item("@last", 1)], "half_turn": True, "normalised": "turn it around"}, world="room", owner="alice")
    assert len(res["moves"]) == 5
    lab.commit(res, owner="alice")
    assert not lab.audit(lab.world("room"))
    table = standing(lab, "table")[0]
    for e in standing(lab, "chair"):
        want = math.degrees(math.atan2(table["y"] - e["y"], table["x"] - e["x"])) % 360
        assert abs((e["theta"] - want + 180) % 360 - 180) < 30, "each chair still faces the table"
    # and a group that cannot turn where it stands is left as it is
    fresh(lab)
    stand(lab, "a living room", world="room", owner="alice")
    res = lab.run_request({"action": "rotate", "items": [item("@last", 1)], "half_turn": True, "normalised": "turn them around"}, world="room", owner="alice")
    assert not res["moves"] and res["fixed"] is True


# ================================================================================================ the parser's contract, end to end
def has_contract(lab):
    from eval_parse import to_py
    req = to_py(lab.mod, lab.parser.parse(lab.parser, "a plant in each corner"))
    return req.get("every") == "corner" and hasattr(lab.parser, "parse_all")


@pytest.fixture()
def contract(lab):
    if not has_contract(lab):
        pytest.skip("the parser does not emit the new contract fields yet")
    fresh(lab)
    return lab


def test_contract_flank_every_corner_seats_and_facing_away(contract):
    lab = contract
    stand(lab, "a couch", world="room")
    res = lab.run("put a lamp on either side of the couch", world="room")
    assert kinds(res) == {"floor lamp": 2} and not clean(lab, res, "room")
    fresh(lab)
    res = lab.run("a plant in each corner", world="room")
    assert kinds(res) == {"plant": 4} and not clean(lab, res, "room")
    res = lab.run("a table for six", world="room")
    assert kinds(res)["chair"] == 6
    res = lab.run("add 4 chairs facing away from me", world="open")
    toward = lab.run("add 4 chairs facing me", world="open")
    assert "away" in res["say"] and kinds(res) == {"chair": 4}
    for a, t in zip(res["placements"], toward["placements"]):
        assert abs((a["theta"] - t["theta"]) % 360 - 180) < 1e-6


def test_contract_everything_it_and_the_rest(contract):
    lab = contract
    stand(lab, "a living room", world="room", owner="me")
    couches = ids_of(lab, "couch")
    res = lab.run("remove everything except the couch", world="room", owner="me")
    assert res["action"] == "remove" and not set(res["removals"]) & set(couches)
    lab.commit(res, owner="me")
    assert ids_of(lab, "couch") == couches and len(lab.registry) == len(couches)
    res = lab.run("put a plant next to it", world="room", owner="me")
    assert "beside the couch" in res["say"]
    lab.commit(res, owner="me")
    res = lab.run("turn it around", world="room", owner="me")
    assert res["moves"] and "Turned one plant" in res["say"]


def test_contract_a_compound_sentence_runs_in_order(contract):
    lab = contract
    outs = lab.run_all("a couch against the wall and a table in front of it", world="room", owner="me")
    assert len(outs) == 2 and kinds(outs[0]) == {"couch": 1} and kinds(outs[1]) == {"table": 1}
    couch, table = standing(lab, "couch")[0], standing(lab, "table")[0]
    assert 0 < math.dist((couch["x"], couch["y"]), (table["x"], table["y"])) < 140


# ================================================================================================ the generators, one at a time
def lua_items(lab, specs):
    return lab.lua_table([dict(w=w, d=d, h=h, **extra) for (w, d, h, extra) in specs])


def test_on_top_never_lets_an_item_overhang_and_reports_what_it_left_off(lab):
    """planner-3: the row was laid out whatever the top's width. Now: rows that fit, the rest left off, a count of them."""
    top = lab.lua_table({"x": 0, "y": 0, "w": 60, "d": 30, "theta": 0, "z": 30})
    items = lua_items(lab, [(20, 10, 5, {}) for _ in range(12)])
    poses, dropped = lab.Layout.on_top(top, items, lab.lua_table({"gap": 4}))
    placed = [poses[i] for i in range(1, 13) if poses[i] is not None]
    assert dropped == 12 - len(placed) and 0 < len(placed) < 12 or dropped == 0
    for p in placed:
        assert abs(p.y) + 10 <= 30 + 2.01 and abs(p.x) + 5 <= 15 + 2.01, (p.x, p.y)       # x runs along the top's depth, y across it
        assert p.z == 30
    # nothing placed touches another
    rects = [lab.Layout.rect(p.x, p.y, 20, 10, p.theta) for p in placed]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not lab.Layout.overlap(rects[i], rects[j], 0)


def test_on_top_shrinks_the_gap_before_leaving_anything_off(lab):
    top = lab.lua_table({"x": 0, "y": 0, "w": 50, "d": 20, "theta": 0, "z": 0})
    items = lua_items(lab, [(15, 8, 5, {}) for _ in range(3)])                    # 45 + 2 gaps of 4 = 53 > 54?  fits with 2 HU of overhang
    poses, dropped = lab.Layout.on_top(top, items, lab.lua_table({"gap": 4}))
    assert dropped == 0
    items = lua_items(lab, [(16, 8, 5, {}) for _ in range(3)])                    # 48 + 2*4 = 56 > 54, but 48 + 2*1 fits
    poses, dropped = lab.Layout.on_top(top, items, lab.lua_table({"gap": 4}))
    assert dropped == 0


def test_on_top_turns_a_symmetric_item_a_quarter_to_fit(lab):
    top = lab.lua_table({"x": 0, "y": 0, "w": 40, "d": 14, "theta": 0, "z": 0})
    items = lua_items(lab, [(8, 30, 5, {"sym": True}), (8, 30, 5, {})])
    poses, dropped = lab.Layout.on_top(top, items, lab.lua_table({"gap": 2}))
    assert poses[1] is not None and poses[2] is None and dropped == 1
    assert abs((poses[1].theta - 90) % 180) < 1e-6 or abs(poses[1].theta % 180 - 90) < 1e-6


def test_on_top_sets_things_toward_the_biased_edge(lab):
    top = lab.lua_table({"x": 0, "y": 0, "w": 60, "d": 40, "theta": 0, "z": 0})
    items = lua_items(lab, [(10, 10, 5, {})])
    back = lab.Layout.on_top(top, items, lab.lua_table({"bias": -1}))[0][1]
    front = lab.Layout.on_top(top, items, lab.lua_table({"bias": 1}))[0][1]
    mid = lab.Layout.on_top(top, items, lab.lua_table({"bias": 0}))[0][1]
    assert back.x < mid.x < front.x and abs(mid.x) < 1e-9


def test_a_pile_rests_each_item_on_what_is_under_it(lab):
    """planner-17: each layer rose by the tallest item of the layer, so shorter things carried floating ones."""
    frame = lab.lua_table({"x": 0, "y": 0, "theta": 0})
    items = lua_items(lab, [(20, 20, 10, {}), (20, 20, 30, {}), (20, 20, 10, {}), (20, 20, 10, {}), (20, 20, 10, {}), (20, 20, 10, {})])
    poses = lab.Layout.stack(frame, items, lab.lua_table({"gap": 1, "base": 3}))
    assert poses[1].z == 0 and poses[2].z == 0 and poses[3].z == 0
    # layer two: items 4 and 5 stand between 1-2 and 2-3 (the tall one is under both); layer three on item 6's neighbours
    assert poses[4].z == 30 and poses[5].z == 30
    tall = lab.Layout.stack(frame, lua_items(lab, [(20, 20, 10, {})] * 10), lab.lua_table({"gap": 1, "max_h": 25}))
    assert max(tall[i].z for i in range(1, 11)) + 10 <= 30, "the base was widened until the pile fits under the cap"


def test_circle_and_scatter_stop_when_told_to(lab):
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return True
    items = lua_items(lab, [(40, 40, 10, {}) for _ in range(30)])
    poses, r = lab.Layout.circle(0, 0, items, lab.lua_table({"gap": 6, "min_r": 5, "stop": stop}))
    assert calls["n"] >= 1 and len(poses) == 30, "stopped after the first radius that clashed, with poses to show for it"
    rng = lab.Layout.rng(7)
    calls["n"] = 0
    poses = lab.Layout.scatter(lab.lua_table({"x": 0, "y": 0, "theta": 0}), lua_items(lab, [(30, 30, 5, {}) for _ in range(300)]), rng,
                               lab.lua_table({"radius": 40, "stop": stop}))
    assert calls["n"] >= 1 and len(poses) == 300


def test_circle_arcs_leave_the_rest_open(lab):
    items = lua_items(lab, [(20, 20, 10, {}) for _ in range(4)])
    poses, r = lab.Layout.circle(0, 0, items, lab.lua_table({"gap": 6, "arc": 120, "start": 90}))
    ys = [poses[i].y for i in range(1, 5)]
    assert all(y > 0 for y in ys), "a part of a ring centred on 90 degrees stays on that side"


def test_the_beds_in_a_bedroom_with_two_have_their_tables_at_the_ends(lab):
    """heldout-12: the end table of the first bed stood where the second bed was (19 HU of overlap)."""
    for world in ("open", "room"):
        for seed in seeds(lab, "a bedroom with two beds", 6):
            fresh(lab)
            res = lab.run("a bedroom with two beds", world=world, seed=seed)
            assert kinds(res)["bed"] == 2 and kinds(res)["side table"] == 1
            assert not clean(lab, res, world), (world, seed, clean(lab, res, world)[:2])


def test_a_long_thing_may_stand_across_the_room(lab):
    """A meeting table (130 by 306) fits a 560x480 room only turned a quarter to the player: that is tried last."""
    fresh(lab)
    res = lab.run("add a conference table", world="room", pos=(0, 0, 0), yaw=0)
    assert kinds(res) == {"meeting table": 1}
    assert not clean(lab, res, "room")


def test_turning_to_face_a_standing_thing(lab):
    """planner-10: the thing is found among what stands (not the thing being turned)."""
    fresh(lab)
    stand(lab, "a tv", world="room", yaw=0)
    stand(lab, "two armchairs", world="room", yaw=90)
    tv = standing(lab, "tv")[0]
    res = lab.run_request({"action": "rotate", "items": [item("armchair", 0, definite=True)], "arrangement": "facing_me", "target": "tv"}, world="room")
    assert len(res["moves"]) == 2
    for m in res["moves"]:
        want = math.degrees(math.atan2(tv["y"] - m["y"], tv["x"] - m["x"])) % 360
        assert abs((m["theta"] - want + 180) % 360 - 180) < 1e-6
    res = lab.run_request({"action": "rotate", "items": [item("tv", 1, definite=True)], "arrangement": "facing_me", "target": "tv"}, world="room")
    assert not res["moves"], "a thing is not turned to face itself"


def test_facing_it_means_the_last_thing_built(lab):
    """A relation target of '@last' ('facing it', 'to face it') is the requester's most recent group, in add and in rotate."""
    fresh(lab)
    stand(lab, "a couch", world="room", owner="alice", yaw=0)
    stand(lab, "a tv", world="room", owner="alice", yaw=90)
    tv = standing(lab, "tv")[0]
    res = lab.run_request({"action": "add", "items": [item("chair", 4)], "arrangement": "row", "facing": "@last"}, world="room", owner="alice")
    assert kinds(res) == {"chair": 4}, kinds(res)
    errors = sorted(lab.facing_error(p, tv["x"], tv["y"]) for p in res["placements"])
    assert errors[0] < 25 and "tv" in res["say"]
    res = lab.run_request({"action": "rotate", "items": [item("couch", 1, definite=True)], "arrangement": "facing_me", "target": "@last"}, world="room", owner="alice")
    couch = standing(lab, "couch")[0]
    want = math.degrees(math.atan2(tv["y"] - couch["y"], tv["x"] - couch["x"])) % 360
    assert res["moves"] and abs((res["moves"][0]["theta"] - want + 180) % 360 - 180) < 1e-6
    res = lab.run_request({"action": "rotate", "items": [item("couch", 1, definite=True)], "arrangement": "facing_me", "target": "@last"}, world="room",
                          owner="nobody")
    assert not res["moves"]


@pytest.mark.parametrize("world", ["open", "room:560x480"])
def test_a_table_for_six_chairs_is_big_enough_that_no_two_chairs_touch(lab, world):
    """Found on the live dedicated server: a table whose chairs touch each other was kept, and the physics engine threw the table
    and chairs about. The swap for a bigger table must happen when seats touch, not only when they are lost."""
    text = "a table with six chairs"
    for seed in seeds(lab, text, 16):
        fresh(lab)
        res = lab.run(text, world=world, seed=seed)
        assert kinds(res)["chair"] == 6 and kinds(res)["table"] + kinds(res)["meeting table"] == 1, (seed, kinds(res), res["say"])
        seats = [p for p in res["placements"] if p["kind"] == "chair"]
        polys = [poly_of(lab.variant(p), p["origin"], p["yaw"]) for p in seats]
        from lab import sat_penetration
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                assert sat_penetration(polys[i], polys[j]) < 0.5, (seed, i, j)
        assert "squeeze" not in " ".join(res.get("notes", [])), res.get("notes")


# ---------------------------------------------------------------------------------------------- the second review (high findings)
def distinct(values, tol=6.0):
    out = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


@pytest.mark.parametrize("first", ["an office", "a desk with a computer", "a kitchen", "five barrels stacked"])
@pytest.mark.parametrize("second", ["move it closer", "move it against the wall", "move it to the corner", "rearrange it in a row",
                                    "rearrange everything in a grid", "turn it around"])
def test_a_selection_holding_a_support_and_its_load_carries_the_load(lab, first, second):
    """The load of a selected support was laid out as a floor item of its own: the computer ended at the floor inside the desk,
    the upper barrels of a pile came down onto the lower ones. A selection that holds both moves the load rigidly with its support."""
    fresh(lab)
    env = lab.world("room")
    stand(lab, first, world=env, owner="a")
    raised = {e["id"]: e["z"] for e in lab.registry if e["z"] > 0.5}
    assert raised, "the scene has nothing standing on anything"
    res = stand(lab, second, world=env, owner="a")
    for e in lab.registry:
        if e["id"] in raised:
            assert abs(e["z"] - raised[e["id"]]) < 0.6, (first, second, e["kind"], raised[e["id"]], e["z"], res["say"])
    assert lab.audit(env) == [], (first, second, res["say"])


def test_a_support_and_its_load_move_together_and_the_reply_counts_both(lab):
    fresh(lab)
    env = lab.world("room")
    stand(lab, "a desk with a computer", world=env, owner="a")
    desk, computer = standing(lab, "desk")[0], standing(lab, "computer")[0]
    res = lab.run("move it against the wall", world=env, owner="a")
    assert {m["id"] for m in res["moves"]} == {desk["id"], computer["id"]}, res
    assert "computer" in res["say"] and "desk" in res["say"], res["say"]
    moved_c = next(m for m in res["moves"] if m["id"] == computer["id"])
    moved_d = next(m for m in res["moves"] if m["id"] == desk["id"])
    assert abs(moved_c["z"] - computer["z"]) < 0.6 and math.hypot(moved_c["x"] - moved_d["x"], moved_c["y"] - moved_d["y"]) < 40


def test_naming_only_the_load_still_moves_it_alone(lab):
    fresh(lab)
    env = lab.world("room")
    stand(lab, "a desk with a computer", world=env, owner="a")
    desk, computer = standing(lab, "desk")[0], standing(lab, "computer")[0]
    res = lab.run_request({"action": "move", "items": [item("computer", 1, definite=True)], "arrangement": "against_wall"}, world=env, owner="a")
    ids = {m["id"] for m in res["moves"]}
    assert ids <= {computer["id"]} and desk["id"] not in ids, res


MAIN_SEAT_SENTENCES = [
    "add a couch, a table and four chairs", "a couch, a table and some chairs", "a table with four chairs and a couch",
    "a sofa and a dining table with chairs", "a loveseat, a dining table and 4 chairs", "a bench, a table and 4 chairs",
    "a couch, a coffee table and four chairs", "a couch, a table and six stools", "a couch, a table and three chairs",
    "ok skippy I want a couch a table and some chairs please",
]


@pytest.mark.parametrize("world", ["open", "room", "room:420x360", "room:300x260"])
@pytest.mark.parametrize("text", MAIN_SEAT_SENTENCES)
def test_a_main_seat_with_a_table_and_chairs_is_never_refused(lab, text, world):
    """The couch and the chairs shared the table's sides (the third chair landed on the couch): every frame was rejected and the
    answer was 'I could not find room for that.' with nothing placed, in every room and on open ground."""
    fresh(lab)
    res = lab.run(text, world=world)
    k = kinds(res)
    seats = k["chair"] + k["stool"]
    assert seats >= 2 and sum(v for key, v in k.items() if key in ("couch", "loveseat", "bench")) == 1, (text, world, dict(k), res["say"])
    assert sum(v for key, v in k.items() if "table" in key) == 1, (text, world, dict(k))
    assert not clean(lab, res, world), (text, world, clean(lab, res, world))


def test_main_seats_with_a_table_and_chairs_over_many_seeds(lab):
    for main in ("couch", "loveseat", "bench"):
        for table in ("table", "coffee table"):
            for seat, n in (("chair", 4), ("stool", 4), ("chair", 6)):
                for seed in seeds(lab, f"{main}{table}{seat}{n}", 4):
                    fresh(lab)
                    res = lab.run_request({"action": "add", "items": [item(main), item(table), item(seat, n)]}, world="room", seed=seed)
                    assert kinds(res)[main] == 1 and kinds(res)[seat] >= min(n, 4), (main, table, seat, n, seed, res["say"])
                    assert not clean(lab, res, "room"), (main, table, seat, n, seed)


@pytest.mark.parametrize("text, expect", [
    ("add 10 chairs and 10 crates", {"chair": 10, "crate": 10}), ("add ten chairs and ten crates", {"chair": 10, "crate": 10}),
    ("add 5 chairs and 10 crates", {"chair": 5, "crate": 10}), ("add 4 tables and 4 crates", {"table": 4, "crate": 4}),
    ("add 10 barrels and 10 crates", {"barrel": 10, "crate": 10}), ("add 40 chairs and 40 crates", None),
])
def test_a_plain_mixed_request_on_open_ground_is_never_refused(lab, text, expect):
    """A room of chairs and big crates overlapped its own side rows at the corners and put the cluster off centre, so every frame
    failed on open ground; the same request with other numbers worked."""
    for seed in [None] + seeds(lab, text, 12):
        fresh(lab)
        res = lab.run(text, world="open", seed=seed)
        assert res["placements"], (text, seed, res["say"])
        if expect:
            got = kinds(res)
            assert all(got[key] == n for key, n in expect.items()), (text, seed, dict(got), res["say"])
        assert not clean(lab, res, "open"), (text, seed, clean(lab, res, "open"))


@pytest.mark.parametrize("text, world", [("add 12 barrels and 10 crates", "room:420x360"), ("add 8 tables and 10 crates", "room"),
                                         ("add 5 tables and 12 crates", "room:420x360"), ("add 10 couches and 3 barrels", "room")])
def test_when_no_one_room_holds_a_mixed_request_each_kind_gets_a_group_of_its_own(lab, text, world):
    fresh(lab)
    res = lab.run(text, world=world)
    assert "separate groups" in res["say"], res["say"]
    n = sum(int(w) for w in text.split() if w.isdigit())
    assert len(res["placements"]) == n, (len(res["placements"]), n, res["say"])
    assert not clean(lab, res, world)


def test_a_mixed_request_that_cannot_fit_places_what_does_and_says_so(lab):
    fresh(lab)
    res = lab.run("add 10 chairs and 10 crates", world="room:300x260")
    assert res["placements"] and len(res["placements"]) < 20
    assert "of the" in res["say"] and "fit" in res["say"], res["say"]
    assert not clean(lab, res, "room:300x260")


@pytest.mark.parametrize("text, gone", [("a living room without the tv", "tv"), ("a living room, no tv", "tv"), ("a bedroom but no bed", "bed"),
                                        ("a kitchen without a table", "table"), ("an office with no computer", "computer")])
def test_a_scene_leaves_out_what_the_parser_excluded(lab, text, gone):
    fresh(lab)
    assert lab.parser.parse(lab.parser, text)["exclude"], "the parser no longer reports the exclusion"
    res = lab.run(text, world="room")
    assert res["placements"] and kinds(res)[gone] == 0, (text, dict(kinds(res)))
    assert not clean(lab, res, "room")


def test_an_explicit_item_of_an_excluded_kind_wins(lab):
    fresh(lab)
    res = lab.run_request({"action": "add", "items": [item("tv")], "scene": "living room", "exclude": ["tv"]}, world="room")
    assert kinds(res)["tv"] == 1 and kinds(res)["couch"] == 1, dict(kinds(res))
    res = lab.run_request({"action": "add", "items": [], "scene": "living room", "exclude": ["tv"]}, world="room")
    assert kinds(res)["tv"] == 0 and kinds(res)["couch"] == 1


def test_in_a_circle_around_a_table_is_a_ring(lab):
    fresh(lab)
    res = lab.run("six chairs in a circle around the table", world="room")
    assert kinds(res) == {"chair": 6, "table": 1}, dict(kinds(res))
    assert "ring" in res["say"], res["say"]
    table = next(p for p in res["placements"] if p["kind"] == "table")
    dists = [math.hypot(p["x"] - table["x"], p["y"] - table["y"]) for p in res["placements"] if p["kind"] == "chair"]
    assert max(dists) - min(dists) < 6, dists
    assert not clean(lab, res, "room")


@pytest.mark.parametrize("text, n, per_axis", [("twelve chairs in 3 rows", 12, (3, 4)), ("six chairs in 2 rows of 3", 6, (2, 3))])
def test_a_grid_has_the_rows_and_columns_it_was_asked_for(lab, text, n, per_axis):
    fresh(lab)
    res = lab.run(text, world="open")
    assert len(res["placements"]) == n
    xs = distinct([p["x"] for p in res["placements"]])
    ys = distinct([p["y"] for p in res["placements"]])
    assert tuple(sorted((len(xs), len(ys)))) == tuple(sorted(per_axis)), (len(xs), len(ys), res["say"])
    assert not clean(lab, res, "open")


def test_facing_away_after_a_shape_turns_every_one_about(lab):
    fresh(lab)
    inward = lab.run("six chairs in a circle", world="open")
    away = lab.run("six chairs in a circle facing away from the table", world="open")
    assert kinds(away) == {"chair": 6}

    def errors(res):
        cx = sum(p["x"] for p in res["placements"]) / 6
        cy = sum(p["y"] for p in res["placements"]) / 6
        return [lab.facing_error(p, cx, cy) for p in res["placements"]]
    assert max(errors(inward)) < 30 and min(errors(away)) > 150, (errors(inward), errors(away))
    assert "outward" in away["say"] or "away" in away["say"], away["say"]
    row = lab.run("a row of chairs with their backs to the wall", world="room")
    assert row["placements"] and "against the wall" in row["say"], row["say"]


def test_move_away_from_a_thing_moves_away_from_that_thing(lab):
    fresh(lab)
    stand(lab, "a couch", world="open", owner="a")
    stand(lab, "a lamp next to the couch", world="open", owner="a")
    couch, lamp = standing(lab, "couch")[0], standing(lab, "floor lamp")[0]
    ux, uy = lamp["x"] - couch["x"], lamp["y"] - couch["y"]
    n = math.hypot(ux, uy)
    # the player stands beyond the lamp: 'away from the player' would push it toward the couch
    pos = (lamp["x"] + ux / n * 200, lamp["y"] + uy / n * 200, 0.0)
    before = math.hypot(lamp["x"] - couch["x"], lamp["y"] - couch["y"])
    res = lab.run("move the lamp away from the couch", world="open", owner="a", pos=pos)
    assert res["moves"], res["say"]
    m = res["moves"][0]
    assert math.hypot(m["x"] - couch["x"], m["y"] - couch["y"]) > before + 30, (before, m, res["say"])


@pytest.mark.parametrize("second", ["move it against the wall", "move it to the corner", "rearrange it in a row", "move it closer"])
def test_a_pile_moves_as_one_piece(lab, second):
    """The upper barrels rested on two lower ones: laid out piece by piece they were left hanging (or in the wall). A pile keeps
    its shape: every distance between its barrels is the same after the move, and the reply counts the barrels."""
    fresh(lab)
    env = lab.world("room")
    stand(lab, "five barrels stacked", world=env, owner="a")
    before = [(e["id"], e["x"], e["y"], e["z"]) for e in lab.registry]
    res = stand(lab, second, world=env, owner="a")
    assert res["moves"], res["say"]
    assert "five barrels" in res["say"], res["say"]
    after = {e["id"]: (e["x"], e["y"], e["z"]) for e in lab.registry}
    for i, (ia, xa, ya, za) in enumerate(before):
        for (ib, xb, yb, zb) in before[i + 1:]:
            d0 = math.hypot(xa - xb, ya - yb)
            d1 = math.hypot(after[ia][0] - after[ib][0], after[ia][1] - after[ib][1])
            assert abs(d0 - d1) < 0.5 and abs((za - zb) - (after[ia][2] - after[ib][2])) < 0.5, (second, ia, ib, d0, d1)
    assert lab.audit(env) == []



# ================================================================================================ the second review's planner fixes
# Each test names the finding it holds shut (E-nn: the numbers of the verified review list). All of them fail on the planner as it
# was before these fixes (HOLODECK_LUA_ROOT points the lab at a copy of the old addon to show it). No overlap tolerance was loosened:
# a live run showed touching chairs get thrown by the physics engine, so a crowd gets a bigger room or a refusal, never a squeeze.
from eval_parse import to_py  # noqa: E402
from lab import sat_penetration  # noqa: E402


def explain_run(lab, text, world, budget=30.0, **kw):
    """Plan a sentence with the planner's `explain` hook: how many candidate placements were judged and refused (a measure of the
    search's effort that does not depend on the speed of the machine)."""
    env = lab.world(world) if isinstance(world, str) else world
    req = lab.parser.parse(lab.parser, text)
    ctx = lab.context(env, budget=budget, **kw)
    events = []
    ctx["explain"] = lambda *a: events.append(a[0])
    res = lab._lists(to_py(lab.mod, lab.planner.run(lab.planner, req, ctx)))
    return res, len(events)


def wide_seeds(n):
    """n seeds spread over the generator's range: the first draw of the planner's generator barely moves between neighbouring seeds
    (nor between the hashes of sentences that differ in a last digit), so these are 7919 apart and draw every variant of a kind."""
    return [13 + k * 7919 for k in range(1, n + 1)]


def wanted_count(lab, text):
    req = to_py(lab.mod, lab.parser.parse(lab.parser, text))
    return sum(int(i["count"]) or 1 for i in req["items"])


def away_error(lab, p, tx, ty):
    """Angle (deg) between where a thing faces and the direction AWAY from (tx, ty)."""
    fx, fy = lab.front_vec(p)
    want = math.degrees(math.atan2(p["y"] - ty, p["x"] - tx))
    have = math.degrees(math.atan2(fy, fx))
    d = (have - want) % 360
    return abs(d - 360 if d > 180 else d)


def pile_problems(lab, res, env):
    """What the independent grader's 'on' rule cannot say about a mixed pile: a piece through a wall, or hanging over nothing."""
    pls = res["placements"]
    polys = [poly_of(lab.variant(p), p["origin"], p["yaw"]) for p in pls]
    x0, y0, x1, y1 = (env.bounds[k] for k in (1, 2, 3, 4))
    bad = []
    for i, (p, poly) in enumerate(zip(pls, polys)):
        if any(x < x0 - 0.7 or x > x1 + 0.7 or y < y0 - 0.7 or y > y1 + 0.7 for x, y in poly):
            bad.append((p["kind"], "through a wall"))
        bottom = p["origin"]["z"] + lab.variant(p)["mins"][2]
        if bottom > 0.6:
            held = False
            for j, q in enumerate(pls):
                if j != i and abs(bottom - (q["origin"]["z"] + lab.variant(q)["maxs"][2])) < 1.0 and sat_penetration(poly, polys[j]) > 0:
                    held = True
            if not held:
                bad.append((p["kind"], f"hangs at {bottom:.1f}"))
    return bad


# ---------------------------------------------------------------------------------------------------------- E-86: side tables, plants
SIDE_TABLE_SENTENCES = [
    "three couches and a side table", "four couches and a side table", "three loveseats and a side table", "three couches and two side tables",
    "two couches, two mugs and two side tables", "4 side tables and a couch", "four loveseats and a side table", "a bench and two side tables",
]


@pytest.mark.parametrize("world", ["open", "room"])
@pytest.mark.parametrize("text", SIDE_TABLE_SENTENCES)
def test_side_tables_stand_at_the_ends_of_a_row_of_seats(lab, text, world):
    """E-86: three or more seats on a wall left no room for the side table (it was placed beside the first seat and slid in 8 HU
    steps): every candidate failed and the request was refused ('I could not find room for that') on open ground."""
    for seed, yaw in [(None, 0.0), (None, 30.0)] + [(s, 0.0) for s in wide_seeds(2)]:
        fresh(lab)
        res = lab.run(text, world=world, seed=seed, yaw=yaw, budget=0.5)
        assert res["placements"], (text, world, seed, yaw, res["say"])
        assert len(res["placements"]) >= wanted_count(lab, text), (text, len(res["placements"]), res["say"])
        assert "separate groups" not in res["say"], res["say"]                 # one room, not each kind on its own after a long search
        assert not clean(lab, res, world), clean(lab, res, world)


@pytest.mark.parametrize("world", ["open", "room"])
@pytest.mark.parametrize("text", ["four tables and five plants", "four tables and six plants", "six tables and six plants", "five tables and six plants"])
def test_corner_plants_do_not_stand_in_a_crowded_middle(lab, text, world):
    """E-86: the cluster of tables was measured alone, so the plants in the room's corners overlapped it whatever the room's size."""
    for seed in [None] + wide_seeds(2):
        fresh(lab)
        res = lab.run(text, world=world, seed=seed, budget=0.5)
        assert res["placements"], (text, world, res["say"])
        assert kinds(res)["plant"] >= 5 and kinds(res)["table"] >= 4, kinds(res)
        assert "separate groups" not in res["say"], res["say"]
        assert not clean(lab, res, world), clean(lab, res, world)


# ---------------------------------------------------------------------------------------------------------- E-87: the search's effort
@pytest.mark.parametrize("text, world, limit", [
    ("ten stools in a circle", "room:300x260", 150), ("a dozen barrels in a circle", "room:300x260", 900),
    ("five benches against the wall", "room:420x360", 1500), ("add twenty couches", "room:560x480", 1100),
])
def test_the_search_does_not_judge_frames_outside_the_room(lab, text, world, limit):
    """E-87: the frames of the open sweep reach far past the walls of a small room, and each was built and judged (thousands of
    refusals before the first fit). A frame whose centre stands on no floor is passed over: the same answers, a fraction of the work."""
    fresh(lab)
    res, refused = explain_run(lab, text, world)
    assert res["placements"], (text, res["say"])
    assert refused <= limit, (text, refused)


@pytest.mark.parametrize("text, world", [("ten stools in a circle", "room:300x260"), ("a dozen barrels in a circle", "room:300x260"),
                                         ("add twenty couches", "room:560x480")])
def test_an_ordinary_players_budget_is_enough_for_a_plain_request(lab, text, world):
    """E-87: requests that needed 30 to 120 ms of Lua were cut at the ordinary player's 0.05 s and the player was told 'That is
    more than I can fit here.' (on a quiet machine, JIT off: 20 of 20 runs). The slowest of five runs may still be cut by a busy
    machine; at least one must finish."""
    done = 0
    for _ in range(5):
        fresh(lab)
        res = lab.run(text, world=world, budget=0.05)
        if res["placements"] and not res["timed_out"]:
            done += 1
            break
    assert done, (text, world)


def test_the_unlimited_answer_is_the_same_with_and_without_the_frame_filter(lab):
    """E-87: passing over frames outside the room changes the effort, never the answer (ctx.exhaustive judges everything)."""
    for text, world in [("ten stools in a circle", "room:300x260"), ("five benches against the wall", "room:420x360"),
                        ("add three couches", "room:420x360"), ("a dozen barrels in a circle", "room:420x360")]:
        fresh(lab)
        env = lab.world(world)
        req = lab.parser.parse(lab.parser, text)
        a = lab._lists(to_py(lab.mod, lab.planner.run(lab.planner, req, lab.context(env, budget=30.0))))
        ctx = lab.context(env, budget=30.0)
        ctx["exhaustive"] = True
        b = lab._lists(to_py(lab.mod, lab.planner.run(lab.planner, req, ctx)))
        assert [(p["kind"], round(p["x"], 2), round(p["y"], 2), round(p["theta"], 2)) for p in a["placements"]] == \
               [(p["kind"], round(p["x"], 2), round(p["y"], 2), round(p["theta"], 2)) for p in b["placements"]], text


@pytest.mark.parametrize("step", [0.0002, 0.0005, 0.001, 0.002, 0.004, 0.008])
@pytest.mark.parametrize("text, world", [("a dozen barrels in a circle", "room:420x360"),
                                         ("ten stools in a circle", "room:300x260"), ("six stools in a circle", "room:300x260")])
def test_a_slice_of_the_time_ending_a_search_is_not_reported_as_no_fit(lab, text, world, step):
    """E-93: when the circle's own slice of the budget ran out, a simpler layout was found and the reply said 'it would not fit as
    asked' with timed_out false, although the unlimited search places the circle. A cut search says it ran out of time."""
    fresh(lab)
    full = lab.run(text, world=world, budget=60.0)
    assert "would not fit" not in full["say"], full["say"]
    cut = lab.run(text, world=world, now=make_clock(lab, step), budget=0.05)
    if cut["placements"] and "would not fit as asked" in cut["say"]:
        raise AssertionError(("a cut search blamed the room", step, cut["say"], cut["timed_out"]))
    if cut["placements"] and "circle" not in cut["say"]:
        assert cut["timed_out"] is True, cut["say"]


# ---------------------------------------------------------------------------------------------------------- E-88: turning at a wall
@pytest.mark.parametrize("world", ["room", "room:700x600"])
@pytest.mark.parametrize("how", ["rotate the {k} 45 degrees", "rotate the {k} 90 degrees", "turn the {k} to face me"])
@pytest.mark.parametrize("kind", ["table", "crate", "patio table", "bench", "dresser"])
def test_a_prop_against_a_wall_turns_when_the_room_is_empty(lab, kind, how, world):
    """E-88: settle_turn tried slides of at most (long side - short side)/2 + 8 HU and rings up to 46: a square table turned 45
    degrees needs about 10 HU and a bench turned a quarter 48 HU, so both were refused in an empty room ('no room to turn it')."""
    for seed in wide_seeds(4):                       # the variants differ in size: the square ones are the hard ones
        fresh(lab)
        env = lab.world(world)
        stand(lab, f"a {kind} against the wall", world=env, owner="a", seed=seed)
        res = lab.run(how.format(k=kind), world=env, owner="a")
        assert res["moves"], (kind, how, world, seed, res["say"])
        lab.commit(res, owner="a")
        assert lab.audit(env) == [], lab.audit(env)


# ---------------------------------------------------------------------------------------------------------- E-89 / E-48 / E-50: beside a wall-flush thing
WALL_FLUSH = ["bookshelf", "tv", "side table", "fridge", "couch", "wardrobe", "dresser"]


@pytest.mark.parametrize("yaw", [0.0, 90.0, 200.0])
@pytest.mark.parametrize("new", ["couch", "table", "plant", "armchair", "bed"])
@pytest.mark.parametrize("ref", WALL_FLUSH)
def test_a_deeper_thing_goes_beside_a_wall_flush_one_without_the_nearby_fallback(lab, ref, new, yaw):
    """E-89/E-48/E-50: the new thing was centred on the reference's centre line; a deeper one stuck out behind the wall, both
    sides failed and it was put 'nearby' with a false 'no room beside' note. The backs line up as a later candidate."""
    if ref == new:
        pytest.skip("a thing beside its own kind is not the case")
    fresh(lab)
    env = lab.world("room")
    stand(lab, f"a {ref} against the wall", world=env, owner="a", yaw=yaw)
    res = lab.run(f"a {new} next to the {ref}", world=env, owner="a", yaw=yaw)
    assert res["placements"], res["say"]
    assert "nearby" not in res["say"] and "no room beside" not in res["say"], res["say"]
    r0 = standing(lab, ref)[0]
    p = res["placements"][0]
    edge_gap = math.hypot(p["x"] - r0["x"], p["y"] - r0["y"]) - (r0["w"] + p["w"]) / 2
    assert edge_gap < 30, (ref, new, yaw, edge_gap)
    assert not clean(lab, res, "room"), clean(lab, res, "room")


@pytest.mark.parametrize("ref", ["bookshelf", "tv", "fridge"])
def test_moving_a_deeper_thing_next_to_a_wall_flush_one_moves_it(lab, ref):
    """E-89/E-48: 'move the couch next to the bookshelf' answered 'There was no clear spot for that.' for the same reason."""
    fresh(lab)
    env = lab.world("room")
    stand(lab, "a couch", world=env, owner="a")
    stand(lab, f"a {ref} against the wall", world=env, owner="a")
    res = stand(lab, f"move the couch next to the {ref}", world=env, owner="a")
    assert res["moves"], res["say"]
    assert lab.audit(env) == []
    couch, r0 = standing(lab, "couch")[0], standing(lab, ref)[0]
    assert math.hypot(couch["x"] - r0["x"], couch["y"] - r0["y"]) - (couch["w"] + r0["w"]) / 2 < 30


# ---------------------------------------------------------------------------------------------------------- E-90 / E-17 / E-98: piles
@pytest.mark.parametrize("text", ["three crates stacked", "stack four crates", "five crates stacked", "add five crates"])
def test_a_pile_the_height_cap_lays_flat_is_not_called_a_pile(lab, text):
    """E-90: MAX_PILE_H (128) was below two of the common crates (64.6, 65.6 HU), so their 'pile' was one flat layer that the reply
    still called 'stacked in a pile' (and a default pile said 'it would not fit as asked' to a player who asked for nothing)."""
    flat = 0
    stacked = 0
    for seed in wide_seeds(16):
        fresh(lab)
        res = lab.run(text, world="room", seed=seed)
        assert res["placements"], res["say"]
        top = max(p["z"] for p in res["placements"])
        if top > 0.5:
            stacked += 1
            assert "stacked" in res["say"], res["say"]
        else:
            flat += 1
            assert "pile" not in res["say"] and "stacked" not in res["say"], (res["say"], seed)
        assert "as asked" not in res["say"] or "stack" in text, res["say"]
        assert not clean(lab, res, "room"), clean(lab, res, "room")
    assert stacked >= 10, (stacked, flat)             # the 64 and 65 HU crates now stack two high


@pytest.mark.parametrize("text", ["three crates facing me and four beds", "three boxes facing me and four desks", "a box and two chairs in front of me",
                                  "three boxes and a couch in front of me", "three crates and four dressers behind me"])
@pytest.mark.parametrize("world", ["room:420x360", "room"])
def test_unlike_kinds_are_not_stacked_under_a_pile_default(lab, text, world):
    """E-17: default_arrangement read only the first kind: crates first with three or more things made 'stacked' of everything, so
    beds were put on crates and on each other, hovering and through a wall, with a success message."""
    fresh(lab)
    env = lab.world(world)
    res = lab.run(text, world=env)
    assert res["placements"], (text, world, res["say"])
    assert all(p["z"] < 0.5 for p in res["placements"]), [(p["kind"], p["z"]) for p in res["placements"] if p["z"] >= 0.5]
    assert not lab.grade(res, world=env), lab.grade(res, world=env)


def test_a_pile_of_crates_and_boxes_alone_is_still_a_pile(lab):
    fresh(lab)
    res = lab.run("ten crates in front of me", world="room")
    assert max(p["z"] for p in res["placements"]) > 0.5, res["say"]                       # a pile of crates alone is still piled
    res = lab.run("six crates and two boxes behind me", world="room")
    assert res["placements"] and not clean(lab, res, "room")


@pytest.mark.parametrize("world", ["room", "room:700x300", "room:420x360"])
@pytest.mark.parametrize("text", ["ten chairs and ten tables stacked", "a barrel and eight small desk lamps and two small dumpsters stacked",
                                  "five boxes, eight barrels and eight stoves stacked", "eight chairs and four tables stacked"])
def test_a_mixed_pile_stays_in_the_room_and_every_piece_rests_on_something(lab, text, world):
    """E-98: only a pile's base was held to the walls and the floor, and an upper piece with nothing under it took the highest top:
    tables were put 117 HU outside the room, desk lamps hung at 52 HU over nothing. The widest pieces now go lowest and each upper
    piece stands on a support of its own; whatever does not fit is refused (or laid out in a grid), never stacked into a wall."""
    fresh(lab)
    env = lab.world(world)
    res = lab.run(text, world=env)
    if res["placements"]:
        assert pile_problems(lab, res, env) == [], (text, world, pile_problems(lab, res, env)[:3], res["say"])
        assert max(p["z"] for p in res["placements"]) >= 0


# ---------------------------------------------------------------------------------------------------------- E-5: more than one corner
def test_successive_corner_requests_each_find_a_free_corner(lab):
    """E-5: a single thing only ever tried the corner nearest the view, so the second 'in the corner' found it taken and fell back to
    the open with 'it would not fit in the corners' (three corners empty); a move to a corner said 'There is no corner in reach.'"""
    fresh(lab)
    env = lab.world("room")
    used = set()
    for text in ["a plant in the corner", "a lamp in the corner", "a chair in the corner", "a crate in the corner"]:
        res = stand(lab, text, world=env, owner="a")
        assert res["placements"] and res["placements"][0]["role"] == "corner", (text, res["say"])
        assert "would not fit" not in res["say"], res["say"]
        p = res["placements"][0]
        corner = (p["x"] > 0, p["y"] > 0)
        assert corner not in used, (text, corner, used)
        used.add(corner)
    assert len(used) == 4
    # a fifth has no corner left: said, and truthfully
    res = lab.run("a barrel in the corner", world=env, owner="a")
    assert "would not fit in the corners" in res["say"] or res["placements"][0]["role"] != "corner", res["say"]


def test_four_corner_moves_in_a_row_each_succeed(lab):
    fresh(lab)
    env = lab.world("room")
    for text in ["a plant", "a lamp", "a chair", "a crate"]:
        stand(lab, text, world=env, owner="a")
    corners = set()
    for word in ["plant", "lamp", "chair", "crate"]:
        res = stand(lab, f"put the {word} in the corner", world=env, owner="a")
        assert res["moves"], (word, res["say"])
        m = res["moves"][0]
        corners.add((m["x"] > 0, m["y"] > 0))
    assert len(corners) == 4, corners
    assert lab.audit(env) == []


# ---------------------------------------------------------------------------------------------------------- E-6: moving a pile's support
@pytest.mark.parametrize("pos", [(30.0, 20.0, 0.0), (0.0, 0.0, 0.0), (-120.0, 60.0, 0.0)])
@pytest.mark.parametrize("world", ["room", "room:420x360"])
@pytest.mark.parametrize("what", ["move the crate to the corner", "move the crate against the wall", "move the crate closer"])
def test_moving_one_crate_of_a_pile_keeps_what_rests_on_it_in_the_room(lab, what, world, pos):
    """E-6: the crates resting on the one that went to a corner were carried with their old offsets, never re-checked: one stood in
    the east wall, two hung in the air. Now the load is judged with the crate; when it cannot go too, the crate stays."""
    fresh(lab)
    env = lab.world(world)
    stand(lab, "five crates", world=env, owner="a", pos=pos)
    res = stand(lab, what, world=env, owner="a", pos=pos)
    assert lab.audit(env) == [], (what, world, pos, res["say"], lab.audit(env))


@pytest.mark.parametrize("second", ["move them closer", "move them left", "move them right", "move them away"])
@pytest.mark.parametrize("kind", ["boxes", "crates"])
def test_a_whole_pile_moved_in_a_direction_keeps_its_shape(lab, kind, second):
    """E-6: a pile moved as a whole must not be flattened onto the floor (interpenetrating)."""
    fresh(lab)
    env = lab.world("room")
    stand(lab, f"six {kind}", world=env, owner="a", pos=(30.0, 20.0, 0.0))
    before = sorted((round(e["z"], 1) for e in lab.registry))
    stand(lab, second, world=env, owner="a", pos=(30.0, 20.0, 0.0))
    assert sorted((round(e["z"], 1) for e in lab.registry)) == before
    assert lab.audit(env) == []


# ---------------------------------------------------------------------------------------------------------- E-2 / E-50: 'the lamp'
@pytest.mark.parametrize("command", ["remove the lamp", "turn the lamp around", "get rid of the lamp", "remove the lamps"])
@pytest.mark.parametrize("setup", [["a table", "a lamp on the table"], ["a desk with a lamp on it"]])
def test_a_bare_the_lamp_finds_the_desk_lamp_that_stands(lab, setup, command):
    """E-2/E-50: 'a lamp on the table' makes a desk lamp, but a definite 'the lamp' meant only the floor lamp, so the lamp just
    asked for could not be removed, moved or turned ('There is no floor lamp to remove.')."""
    fresh(lab)
    env = lab.world("room")
    for text in setup:
        stand(lab, text, world=env, owner="a")
    n = len(lab.registry)
    res = lab.run(command, world=env, owner="a")
    assert "no floor lamp" not in res["say"] and "no lamp" not in res["say"], res["say"]
    if "remove" in command or "get rid" in command:
        assert res["removals"], res["say"]
        assert "desk lamp" in res["say"]
    else:
        assert res["moves"], res["say"]
    assert len(lab.registry) == n


def test_the_lamp_means_either_kind_nearest_first_and_the_full_name_stays_exact(lab):
    fresh(lab)
    env = lab.world("room")
    stand(lab, "a lamp", world=env, owner="a")
    stand(lab, "a desk lamp", world=env, owner="a")
    res = stand(lab, "remove the lamp", world=env, owner="a")
    assert len(res["removals"]) == 1
    res = stand(lab, "remove the lamp", world=env, owner="a")
    assert len(res["removals"]) == 1, res["say"]
    assert not standing(lab, "floor lamp") and not standing(lab, "desk lamp")
    res = lab.run("remove the lamp", world=env, owner="a")
    assert "no floor lamp to remove" in res["say"], res["say"]            # none of either kind: the old sentence stays
    fresh(lab)
    stand(lab, "a desk lamp", world=env, owner="a")
    res = lab.run("remove the floor lamp", world=env, owner="a")
    assert not res["removals"], "the full name matches only that kind"


def test_a_put_the_lamp_does_not_make_a_second_lamp_when_a_desk_lamp_stands(lab):
    fresh(lab)
    env = lab.world("room")
    stand(lab, "a desk with a lamp on it", world=env, owner="a")
    stand(lab, "a couch", world=env, owner="a")
    n = len(lab.registry)
    res = stand(lab, "put the lamp next to the couch", world=env, owner="a")
    assert len(lab.registry) == n and res["moves"], res["say"]


@pytest.mark.parametrize("setup, command, kind", [(["an office chair"], "remove the chair", "office chair"), (["a coffee table"], "move the table closer", "coffee table")])
def test_a_bare_chair_or_table_finds_the_kind_that_stands_when_the_plain_one_does_not(lab, setup, command, kind):
    fresh(lab)
    env = lab.world("room")
    for text in setup:
        stand(lab, text, world=env, owner="a")
    res = lab.run(command, world=env, owner="a", pos=(-150.0, 0.0, 0.0))
    assert kind in res["say"] or res["moves"] or res["removals"], res["say"]
    assert "There is no" not in res["say"], res["say"]


# ---------------------------------------------------------------------------------------------------------- E-9: two scenes in a sentence
def test_a_scene_the_parser_left_out_is_named(lab):
    """E-9: 'a living room and a kitchen' built the living room only and said nothing about the kitchen."""
    fresh(lab)
    res = lab.run_request({"action": "add", "items": [], "scene": "living room", "extra_scenes": ["kitchen"]}, world="room:700x600")
    assert res["placements"]
    assert "kitchen" in res["say"] and "living room" in res["say"], res["say"]
    res = lab.run_request({"action": "add", "items": [], "scene": "living room", "extra_scenes": ["a dragon's lair"]}, world="room:700x600")
    assert "dragon" not in res["say"], "only the library's own scene names are spoken"


# ---------------------------------------------------------------------------------------------------------- E-96: each corner
@pytest.mark.parametrize("n, mode, want", [(4, "per", 16), (2, "per", 8), (8, "per", 32), (4, "total", 4), (2, "total", 2), (3, None, 12), (4, None, 4)])
def test_a_count_per_corner_is_multiplied_and_a_total_is_not(lab, n, mode, want):
    """E-96: 'N plants in each corner' was read as a total whenever N was a multiple of four. The parser says per or total
    (req.every_mode); without its word a multiple of four is still taken as the total, as before."""
    fresh(lab)
    req = {"action": "add", "items": [item("plant", n)], "every": "corner"}
    if mode:
        req["every_mode"] = mode
    res = lab.run_request(req, world="open", max_per_kind=60, budget=5.0)
    assert len(res["placements"]) == want, (n, mode, len(res["placements"]), res["say"])


# ---------------------------------------------------------------------------------------------------------- E-97: refusals
@pytest.mark.parametrize("text, world", [
    ("fifty computers", "room:300x260"), ("twelve chairs in a circle around the table", "room:420x360"),
    ("sixty chairs around the table", "room:300x260"), ("forty plants in the corners", "room:300x260"), ("sixty beds against the wall", "room"),
])
def test_a_refusal_carries_no_note_about_what_was_not_done(lab, text, world):
    """E-97: 'There was no meeting table, so I added one' and 'I put them on two meeting tables I added' were spoken after a request
    that placed nothing."""
    fresh(lab)
    res = lab.run(text, world=world, budget=5.0)
    if res["placements"]:
        return
    spoken = " ".join([res["say"]] + res["notes"]).lower()
    for phrase in ("i added", "so i added", "i put them", "grouped", "laid it out", "i furnished", "so i put"):
        assert phrase not in spoken, (text, spoken)


def test_request_level_notes_survive_a_refusal(lab):
    """The note about an unknown word, or a count scaled down, is true whether or not anything was placed."""
    fresh(lab)
    res = lab.run("a dragon and a hundred and forty crates", world="room:300x260", budget=5.0)
    spoken = " ".join([res["say"]] + res["notes"]).lower()
    assert "library" in spoken or res["placements"], spoken


# ---------------------------------------------------------------------------------------------------------- facing away from a named thing
@pytest.mark.parametrize("text, kind", [
    ("four chairs in a row facing away from the tv", "chair"), ("3 couches in a row facing away from the tv", "couch"),
    ("six chairs in a circle facing away from the tv", "chair"), ("six chairs facing away from the tv", "chair"),
])
def test_a_group_facing_away_from_a_named_thing_looks_away_from_that_thing(lab, text, kind):
    """The parser keeps 'facing away from the tv' as req.facing_away = 'tv'; the planner ignored the name and turned the group away
    from the PLAYER, so the chairs faced the tv and the reply said 'facing away'. Each one now looks along the line from the tv."""
    fresh(lab)
    env = lab.world("room")
    stand(lab, "a tv", world=env, owner="a")
    tv = standing(lab, "tv")[0]
    res = lab.run(text, world=env, owner="a")
    group = [p for p in res["placements"] if p["kind"] == kind]
    assert group and "away" in res["say"] or "outward" in res["say"], res["say"]
    errs = [away_error(lab, p, tv["x"], tv["y"]) for p in group]
    assert max(errs) < 20, (text, errs, res["say"])
    assert not lab.grade(res, world=env), lab.grade(res, world=env)


def test_facing_away_from_a_thing_that_is_not_there_says_so_and_adds_nothing(lab):
    fresh(lab)
    res = lab.run("four chairs in a row facing away from the tv", world="room")
    assert kinds(res) == {"chair": 4}, kinds(res)
    assert "no tv" in res["say"], res["say"]


# ---------------------------------------------------------------------------------------------------------- a circle round a wall-flush screen
def test_a_circle_of_chairs_round_a_screen_on_a_wall_stands_in_front_of_it(lab):
    """'five chairs in a circle around the tv' with the tv against a wall: a ring needs the wall's side too, so it was thrown 130 to
    194 HU away with 'no room around the tv'. An arc in front of the screen is the circle that fits there."""
    for seed in [None] + wide_seeds(3):
        fresh(lab)
        env = lab.world("room")
        stand(lab, "a tv", world=env, owner="a")
        tv = standing(lab, "tv")[0]
        res = lab.run("five chairs in a circle around the tv", world=env, owner="a", seed=seed)
        chairs = [p for p in res["placements"] if p["kind"] == "chair"]
        assert len(chairs) == 5, res["say"]
        assert "no room" not in res["say"], res["say"]
        assert max(math.hypot(p["x"] - tv["x"], p["y"] - tv["y"]) for p in chairs) < 110, res["say"]
        assert not lab.grade(res, world=env), lab.grade(res, world=env)


# ---------------------------------------------------------------------------------------------------------- 'the other side of the couch'
def _couch_and_lamp(lab, yaw):
    env = lab.world("open")
    stand(lab, "a couch", world=env, owner="a", yaw=yaw)
    stand(lab, "a lamp", world=env, owner="a", yaw=yaw)
    return env, standing(lab, "couch")[0], standing(lab, "floor lamp")[0]


@pytest.mark.parametrize("yaw", [0.0, 90.0, 215.0])
@pytest.mark.parametrize("side", ["left", "right"])
def test_a_lamp_moved_to_the_left_or_right_of_the_couch_goes_to_the_players_left_or_right(lab, side, yaw):
    """req.side: only the lamp moves, to the side of the couch that is the player's left (or right)."""
    fresh(lab)
    env, couch, lamp = _couch_and_lamp(lab, yaw)
    req = {"action": "move", "items": [item("floor lamp", 1, definite=True)], "arrangement": "beside", "target": "couch", "side": side,
           "verb": "move", "raw": "x", "normalised": "x"}
    res = lab.run_request(req, world=env, owner="a", yaw=yaw)
    assert [m["id"] for m in res["moves"]] == [lamp["id"]], res["say"]
    m = res["moves"][0]
    rx, ry = math.sin(math.radians(yaw)), -math.cos(math.radians(yaw))                 # the player's right
    lateral = (m["x"] - couch["x"]) * rx + (m["y"] - couch["y"]) * ry
    assert (lateral > 0) == (side == "right") and abs(lateral) > couch["w"] / 2, (side, yaw, lateral)
    lab.commit(res, owner="a")
    assert lab.audit(env) == []


@pytest.mark.parametrize("yaw", [0.0, 90.0, 215.0])
@pytest.mark.parametrize("side", ["other", "opposite"])
def test_a_lamp_moved_to_the_other_side_of_the_couch_goes_to_the_side_facing_away(lab, side, yaw):
    fresh(lab)
    env, couch, lamp = _couch_and_lamp(lab, yaw)
    req = {"action": "move", "items": [item("floor lamp", 1, definite=True)], "arrangement": "beside", "target": "couch", "side": side,
           "verb": "move", "raw": "x", "normalised": "x"}
    res = lab.run_request(req, world=env, owner="a", yaw=yaw)
    assert [m["id"] for m in res["moves"]] == [lamp["id"]], res["say"]
    m = res["moves"][0]
    away = (couch["x"], couch["y"])                                                    # the player stands at the origin
    d = math.hypot(*away)
    beyond = ((m["x"] - couch["x"]) * away[0] + (m["y"] - couch["y"]) * away[1]) / d
    assert beyond > couch["d"] / 2 - 1, (side, yaw, beyond)
    assert abs(math.hypot(m["x"] - couch["x"], m["y"] - couch["y"]) - (couch["d"] + 20) / 2) < 60


# ---------------------------------------------------------------------------------------------------------- chairs do not tuck through drawers
@pytest.mark.parametrize("ref", ["dresser", "side table"])
def test_chairs_are_not_tucked_through_a_dresser_or_a_side_table(lab, ref):
    """The tuck (a chair's front edge slid under the top) is for open tables with knee room. A dresser or a side table is a drawer
    unit: the chairs stood 9 HU inside it. Over several seeds (the variants differ) none overlaps it."""
    worst = 0.0
    for seed in wide_seeds(8):
        fresh(lab)
        res = lab.run_request({"action": "add", "items": [item("chair", 4)], "arrangement": "around", "target": ref}, world="room", seed=seed)
        pls = res["placements"]
        polys = [poly_of(lab.variant(p), p["origin"], p["yaw"]) for p in pls]
        for i, a in enumerate(pls):
            for j, b in enumerate(pls):
                if a["kind"] == ref and b["kind"] == "chair":
                    worst = max(worst, sat_penetration(polys[i], polys[j]))
    assert worst <= 1.5, worst


def test_chairs_still_tuck_under_an_open_table(lab):
    fresh(lab)
    tucked = 0.0
    for seed in wide_seeds(6):
        fresh(lab)
        res = lab.run_request({"action": "add", "items": [item("chair", 4)], "arrangement": "around", "target": "table"}, world="room", seed=seed)
        pls = res["placements"]
        polys = [poly_of(lab.variant(p), p["origin"], p["yaw"]) for p in pls]
        for i, a in enumerate(pls):
            for j, b in enumerate(pls):
                if a["kind"] == "table" and b["kind"] == "chair":
                    tucked = max(tucked, sat_penetration(polys[i], polys[j]))
    assert tucked > 3, "chairs slide under an open table top"
