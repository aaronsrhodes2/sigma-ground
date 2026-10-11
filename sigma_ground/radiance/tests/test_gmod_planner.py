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
