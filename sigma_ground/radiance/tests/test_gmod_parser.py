"""Tests for the holodeck's parser (stage/gmod/addon/lua/mentat/holodeck/parse.lua), run offline under LuaJIT through `lupa`.

Two kinds of test:
  * a held-out corpus (stage/gmod/holodeck/data/utterance_corpus_heldout.json): the hand-judged requests of an independent review,
    stressed, messy and compound sentences the audited corpus does not cover, with the meaning of each written down by a person.
    The test requires a high pass rate and lists every failure;
  * one regression test per defect the review found: each fails on the parser that had the defect.

The parser's contract (what the planner consumes) is documented at the top of parse.lua.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("lupa", reason="lupa (LuaJIT binding) not installed; holodeck tests skipped")

ROOT = Path(__file__).resolve().parents[3]
HOLODECK = ROOT / "stage" / "gmod" / "holodeck"
sys.path.insert(0, str(HOLODECK))

import eval_parse  # noqa: E402
from lab import Lab  # noqa: E402


@pytest.fixture(scope="module")
def parse():
    return eval_parse.make_parser()


@pytest.fixture(scope="module")
def lab():
    return Lab()


def kinds(req):
    return [(i["kind"], i["count"]) for i in req["items"]]


def one(parse, text):
    reqs = parse.parse_all(text)
    assert len(reqs) == 1, (text, [(r["action"], kinds(r)) for r in reqs])
    return reqs[0]


# --------------------------------------------------------------------------------------------- the held-out corpus
def test_heldout_corpus_pass_rate(parse):
    cases = eval_parse.load_cases("utterance_corpus_heldout.json")
    assert len(cases) >= 450
    failures = []
    for c in cases:
        if c.get("known_miss"):
            continue
        try:
            bad = eval_parse.compare_case(c, parse)
        except Exception as e:                                # a crash is a failure with its reason
            bad = [f"CRASH {e}"]
        if bad:
            failures.append((c["slice"], c["utterance"][:70], bad[:2]))
    rate = 1 - len(failures) / len(cases)
    listing = "\n".join(f"  [{s}] {u!r}: {b}" for s, u, b in failures[:40])
    assert rate >= 0.97, f"{len(failures)} of {len(cases)} held-out requests are misread ({rate:.1%}):\n{listing}"


def test_audited_corpus_keeps_its_meaning(parse):
    cases = eval_parse.load_cases("utterance_corpus.json")
    bad = [c["utterance"] for c in cases if not c.get("known_miss") and eval_parse.compare(c, parse(c["utterance"]))]
    assert not bad, bad[:5]


# --------------------------------------------------------------------------------------------- the pseudo items: it, them, everything
@pytest.mark.parametrize("text,action", [
    ("turn it to face me", "rotate"), ("move it next to the tv", "move"), ("rotate it", "rotate"), ("move them against the wall", "move"),
    ("put them in a circle", "rearrange"), ("line them up", "rearrange"), ("make it face the tv", "rotate"),
    ("put it next to the tv", "move"), ("get rid of the last one", "remove"), ("delete the thing i just made", "remove"),
    ("move what i just made closer", "move"), ("the new stuff should face me", "rotate"),
])
def test_it_and_them_are_the_last_group(parse, text, action):
    req = one(parse, text)
    assert req["action"] == action and kinds(req) == [("@last", 1)], req


@pytest.mark.parametrize("text,action", [
    ("rearrange everything in a circle", "rearrange"), ("move everything against the wall", "move"), ("turn everything to face me", "rotate"),
    ("tidy up", "rearrange"), ("rearrange the room", "rearrange"), ("move the furniture", "move"), ("shuffle all the furniture", "rearrange"),
])
def test_everything_is_the_all_item(parse, text, action):
    req = one(parse, text)
    assert req["action"] == action and kinds(req) == [("@all", 0)], req


def test_a_relation_to_it_targets_the_last_group(parse):
    for text, arr in [("put a lamp next to it", "beside"), ("put four chairs around it", "around"), ("put a plant behind them", "behind"),
                      ("a chair next to the last one", "beside"), ("put a table next to the thing I just made", "beside")]:
        req = one(parse, text)
        assert req["action"] == "add" and req["arrangement"] == arr and req["target"] == "@last", (text, req)
    assert [i["kind"] for i in one(parse, "put a lamp next to it")["items"]] == ["floor lamp"], "it is never an item to add"


def test_it_inside_one_sentence_is_the_thing_just_named(parse):
    req = one(parse, "a couch and a lamp next to it")
    assert req["target"] == "couch" and req["arrangement"] == "beside"
    req = one(parse, "turn the couch around so it faces the tv")
    assert req["action"] == "rotate" and kinds(req) == [("couch", 1)] and req["target"] == "tv"
    req = one(parse, "a table with 4 chairs and a lamp on it")
    assert req["arrangement"] == "on" and req["target"] == "table", "the lamp goes on the table, not on a chair"
    assert "desk lamp" in [i["kind"] for i in req["items"]]


# --------------------------------------------------------------------------------------------- exceptions
@pytest.mark.parametrize("text,exclude", [
    ("remove everything except the couch", ["couch"]), ("delete everything but the table", ["table"]),
    ("clear everything but the bed", ["bed"]), ("delete all but the tv", ["tv"]), ("take away everything but the table", ["table"]),
    ("clear the room except the couch", ["couch"]), ("wipe the room except for the bed", ["bed"]),
    ("keep the couch and remove the rest", ["couch"]), ("remove everything but keep the couch", ["couch"]),
    ("remove everything except the bed and the lamp", ["bed", "floor lamp"]), ("get rid of everything other than the chairs", ["chair"]),
])
def test_exceptions_are_kept_never_cleared(parse, text, exclude):
    req = one(parse, text)
    assert req["action"] == "remove", (text, req["action"])
    assert kinds(req) == [("@all", 0)]
    assert sorted(req["exclude"]) == sorted(exclude)


@pytest.mark.parametrize("text", [
    "remove all the chairs except one", "delete all the crates but two", "remove all the chairs except the red one",
    "delete all chairs except the red ones", "remove half of the crates", "remove the rest", "remove all the others",
])
def test_a_share_of_a_kind_is_refused_not_wiped(parse, text):
    req = one(parse, text)
    assert req["action"] == "none" and not req["items"], (text, req)
    assert req["notes"], "the player is told why nothing happened"


def test_a_clear_is_the_whole_sentence_only(parse):
    for text in ["clear the room", "clear everything", "wipe it all", "start over", "end program", "arch", "get rid of everything",
                 "computer, wipe everything and start over", "ok that's enough, clear it all out", "clean up the room"]:
        assert one(parse, text)["action"] == "clear", text
    for text, what in [("clear the floor lamp", "floor lamp"), ("clear the floor of crates", "crate"), ("clear the room of barrels", "barrel"),
                       ("get rid of the floor lamp", "floor lamp")]:
        req = one(parse, text)
        assert req["action"] == "remove" and req["items"][0]["kind"] == what, (text, req)
    for text in ["add an arch", "make an arch", "put an arch over the door", "build a stone arch"]:
        req = one(parse, text)
        assert req["action"] == "add" and [i["kind"] for i in req["items"]] == ["UNKNOWN:arch"], (text, req)
    for text in ["clear everything off the table", "get rid of everything on the table", "take everything off the desk", "remove everything on the left"]:
        assert one(parse, text)["action"] == "none", text
    assert one(parse, "start over with a kitchen")["action"] == "add"


# --------------------------------------------------------------------------------------------- relations
@pytest.mark.parametrize("text,arr,target", [
    ("put a lamp by the couch", "beside", "couch"), ("put a lamp under the table", "beside", "table"),
    ("put a chair at the foot of the bed", "in_front_of", "bed"), ("add 2 chairs for the table", "around", "table"),
    ("put a chair opposite the couch", "facing_me", "couch"), ("a couch across from the tv", "facing_me", "tv"),
    ("two chairs at the window", "beside", "window"), ("a lamp over by the door", "beside", "door"),
    ("put a plant between the couch and the tv", "beside", "couch"), ("add a couch that faces the tv", "facing_me", "tv"),
    ("put a tv above the fireplace", "on", "wood stove"), ("put a lamp next to the nice big red couch", "beside", "couch"),
    ("put a lamp next to the very large old brown couch", "beside", "couch"),
])
def test_reference_nouns_are_places_never_new_things(parse, text, arr, target):
    req = one(parse, text)
    assert req["arrangement"] == arr and req["target"] == target, (text, req["arrangement"], req["target"])
    assert target not in [i["kind"] for i in req["items"]], f"{target} must not be added: {kinds(req)}"
    assert len(req["items"]) == 1


@pytest.mark.parametrize("text,kind", [
    ("move the couch to the tv", "couch"), ("move the chairs to the table", "chair"), ("move the lamp by the couch", "floor lamp"),
    ("put the chair at the table", "chair"), ("take the chair to the table", "chair"), ("send the chair to the table", "chair"),
    ("pull the chair up to the table", "chair"), ("bring the chair over to the table", "chair"), ("slide the chair under the table", "chair"),
    ("push the chairs under the table", "chair"), ("put the lamp by the couch", "floor lamp"), ("move the lamp to the couch", "floor lamp"),
])
def test_moving_to_a_thing_moves_only_the_mover(parse, text, kind):
    req = one(parse, text)
    assert req["action"] == "move" and [i["kind"] for i in req["items"]] == [kind] and req["arrangement"] == "beside", (text, req)


def test_destination_prepositions_in_the_planner(lab):
    """The real planner, on a table and a chair already standing: the words move the chair, they do not build a second table."""
    for text in ["put the chair at the table", "put the chair under the table", "take the chair to the table", "move the chair to the table"]:
        lab.registry, lab.next_id, lab.seq = [], 1, 0
        for t in ["a table", "a chair"]:
            lab.commit(lab.run(t, world="room"))
        res = lab.run(text, world="room")
        assert res["action"] == "move" and not res["placements"] and len(res["moves"]) == 1, (text, res["say"])


def test_two_relations_with_their_own_nouns_are_two_requests(parse):
    reqs = parse.parse_all("put a mug on the desk and a plant on the shelf")
    assert [(r["action"], kinds(r), r["arrangement"], r["target"]) for r in reqs] == \
        [("add", [("mug", 1)], "on", "desk"), ("add", [("plant", 1)], "on", "shelf")]
    reqs = parse.parse_all("put a couch against the wall and a table in front of it")
    assert [(r["arrangement"], r["target"]) for r in reqs] == [("against_wall", ""), ("in_front_of", "@last")]


@pytest.mark.parametrize("text,count", [
    ("a lamp on either side of the couch", 1), ("put a lamp on each side of the couch", 1), ("put a lamp on both sides of the couch", 1),
    ("flank the couch with two plants", 2), ("put chairs on both sides of the table", 2), ("a pair of lamps on either side of the bed", 2),
])
def test_flank(parse, text, count):
    req = one(parse, text)
    assert req["arrangement"] == "flank" and req["items"][0]["count"] == count and req["target"] in ("couch", "table", "bed"), (text, req)


def test_flank_with_the_flanked_thing_named(parse):
    req = one(parse, "a couch flanked by two lamps")
    assert req["arrangement"] == "flank" and req["target"] == "couch" and kinds(req) == [("couch", 1), ("floor lamp", 2)]
    req = one(parse, "a sofa with a lamp on each side")
    assert req["arrangement"] == "flank" and req["target"] == "couch" and kinds(req) == [("couch", 1), ("floor lamp", 1)]


@pytest.mark.parametrize("text,target", [
    ("add 4 chairs facing away from me", "me"), ("add 4 chairs facing away from the table", "table"), ("add a couch facing away from the tv", "tv"),
    ("add 4 chairs facing outward", ""), ("add 4 chairs facing outwards", ""), ("make the chairs face outward", ""),
])
def test_facing_away(parse, text, target):
    req = one(parse, text)
    assert req["arrangement"] == "facing_away" and req["target"] == target, (text, req)
    assert "table" not in [i["kind"] for i in req["items"]] and "tv" not in [i["kind"] for i in req["items"]]


def test_facing_away_beside_a_shape(parse):
    req = one(parse, "six chairs in a circle facing outward")
    assert req["arrangement"] == "circle" and req["facing_away"] == ""
    req = one(parse, "add a couch with its back to the wall")
    assert req["arrangement"] == "against_wall"


def test_a_shape_and_a_place_keep_both(parse):
    req = one(parse, "five chairs in a circle around me")
    assert (req["arrangement"], req["target"], req["shape"]) == ("around", "me", "circle")
    req = one(parse, "twelve chairs in a circle around the table")
    assert (req["arrangement"], req["target"], req["shape"]) == ("around", "table", "circle")
    req = one(parse, "six chairs in a row facing the tv")
    assert (req["arrangement"], req["facing"]) == ("row", "tv")
    req = one(parse, "eight lockers in a row against the wall")
    assert (req["arrangement"], req["shape"]) == ("against_wall", "row")


@pytest.mark.parametrize("text,every", [
    ("a plant in each corner", "corner"), ("a lamp in every corner", "corner"), ("put a plant in every corner of the room", "corner"),
    ("a plant in all four corners", "corner"), ("a chair at each corner", "corner"),
    ("four plants in the corners", None), ("two armchairs in the corners", None), ("add 4 chairs one in each corner", None),
])
def test_every_corner(parse, text, every):
    req = one(parse, text)
    assert req["arrangement"] == "corner" and req.get("every") == every, (text, req)


@pytest.mark.parametrize("text,seats", [
    ("a table for six", 6), ("add a dining table for 8", 8), ("add a table that seats 6", 6), ("a table big enough for ten", 10),
    ("a meeting table for 12 people", 12), ("a table which sleeps 4", 4),
])
def test_seats(parse, text, seats):
    req = one(parse, text)
    assert req.get("seats") == seats, (text, req)
    assert [i["kind"] for i in req["items"]] in (["table"], ["meeting table"]), "the seats are not a thing to add"


def test_chairs_for_a_number(parse):
    req = one(parse, "a table and chairs for four")
    assert kinds(req) == [("table", 1), ("chair", 4)]
    assert kinds(one(parse, "add seating for ten")) == [("chair", 10)]


@pytest.mark.parametrize("text", ["on the left side of the room", "put a couch on the left side of the room", "put a plant on the floor"])
def test_places_that_are_not_things(parse, text):
    assert one(parse, text)["arrangement"] == "unspecified"


@pytest.mark.parametrize("text,count,arr", [
    ("two rows of five chairs", 10, "grid"), ("a 3 by 3 grid of crates", 9, "grid"), ("make a 4x4 grid of boxes", 16, "grid"),
    ("add 2 groups of 3 chairs", 6, "unspecified"), ("three rows of four chairs", 12, "grid"), ("a grid of 4 by 6 crates", 24, "grid"),
])
def test_products_of_counts(parse, text, count, arr):
    req = one(parse, text)
    assert req["items"][0]["count"] == count and req["arrangement"] == arr and len(req["items"]) == 1, (text, req)
    assert not req["unknown"], "the x of 4x4 is not a thing"


def test_grid_dimensions(parse):
    req = one(parse, "two rows of five chairs")
    assert (req["rows"], req["cols"]) == (2, 5)


# --------------------------------------------------------------------------------------------- verbs
@pytest.mark.parametrize("text,verb", [
    ("make the couch blue", "recolour"), ("paint the couch red", "paint"), ("repaint the couch", "repaint"), ("fix the lamp", "fix"),
    ("make the table bigger", "resize"), ("replace the chair with a stool", "replace"), ("change the couch to a bed", "change"),
    ("swap the couch and the chair", "swap"), ("turn off the tv", "turn off"), ("group the chairs together", "group"),
    ("frobnicate the couch", "frobnicate"), ("make it red", "recolour"), ("open the fridge", "open"),
])
def test_an_unknown_verb_with_a_definite_object_adds_nothing(parse, text, verb):
    req = one(parse, text)
    assert req["action"] == "none" and req["unknown_verb"] == verb and not req["items"], (text, req)


@pytest.mark.parametrize("text,what", [
    ("if you could remove the lamp", "floor lamp"), ("I think you should remove the couch", "couch"), ("do you mind removing the lamp", "floor lamp"),
    ("would you be able to remove the lamp", "floor lamp"), ("nice, now remove the couch", "couch"), ("take the chairs away", "chair"),
    ("throw the couch away", "couch"), ("get that couch out of here", "couch"), ("wipe the chairs", "chair"), ("i dont want the couch anymore", "couch"),
    ("the lamp has to go", "floor lamp"), ("take the couch away", "couch"), ("scratch the lamp", "floor lamp"), ("clean up the crates", "crate"),
    ("i want the lamp gone", "floor lamp"), ("have the lamp removed", "floor lamp"), ("make the lamp disappear", "floor lamp"),
    ("put the lamp away", "floor lamp"), ("the chairs can go", "chair"), ("removing the lamp", "floor lamp"), ("put the crates in the trash", "crate"),
    ("i no longer want the lamp", "floor lamp"),
])
def test_removal_said_many_ways_is_a_removal(parse, text, what):
    req = one(parse, text)
    assert req["action"] == "remove" and req["items"][0]["kind"] == what, (text, req["action"], kinds(req))


def test_a_removal_verb_inside_a_noun_phrase_is_not_a_verb(parse):
    for text, kind in [("add a clear table", "table"), ("add a clear glass table", "table"), ("make a clear crate", "crate"),
                       ("add a flip phone", "phone"), ("add a bar stool", "stool")]:
        req = one(parse, text)
        assert req["action"] == "add" and req["items"][0]["kind"] == kind, (text, req)


def test_other_verb_forms(parse):
    assert one(parse, "line up the chairs")["arrangement"] == "row"
    assert one(parse, "line up the chairs")["action"] == "rearrange"
    assert one(parse, "line up 4 chairs")["action"] == "add"
    assert one(parse, "make the chairs straight")["arrangement"] == "row"
    for text in ["have the chairs face the table", "let the chairs face the table", "i want the chairs to face the tv", "the chairs should face the table",
                 "get the chairs to face the tv"]:
        req = one(parse, text)
        assert req["action"] == "rotate" and req["arrangement"] == "facing_me", text
    for text, scene in [("turn this into a waiting room", "waiting room"), ("turn the room into a kitchen", "kitchen"), ("turn it into a bedroom", "bedroom")]:
        req = one(parse, text)
        assert req["action"] == "add" and req["scene"] == scene, text
    assert one(parse, "spread the chairs out")["arrangement"] == "scattered"


# --------------------------------------------------------------------------------------------- undo
def test_undo_is_the_whole_sentence_only(parse):
    for text in ["undo", "undo that", "oops, put them back", "never mind", "scratch that", "cancel", "go back", "revert", "put it back"]:
        assert one(parse, text)["action"] == "undo", text
    reqs = parse.parse_all("never mind, add a lamp")
    assert [(r["action"], kinds(r)) for r in reqs] == [("undo", []), ("add", [("floor lamp", 1)])]
    reqs = parse.parse_all("scratch that, add a lamp")
    assert [r["action"] for r in reqs] == ["undo", "add"]
    for text, action in [("put them back in a row", "rearrange"), ("put them back against the wall", "move"), ("take it back to the wall", "move"),
                         ("cancel the red couch", "remove")]:
        assert one(parse, text)["action"] == action, text
    assert one(parse, "go back to the living room")["action"] == "add"


# --------------------------------------------------------------------------------------------- several requests at once
def test_sentences_are_separate_requests(parse):
    reqs = parse.parse_all("add a couch. remove the lamp")
    assert [(r["action"], kinds(r)) for r in reqs] == [("add", [("couch", 1)]), ("remove", [("floor lamp", 1)])]
    assert parse("add a couch. remove the lamp")["action"] == "add", "parse() is the first request, with no second sentence fused into it"
    assert kinds(parse("add a couch. remove the lamp")) == [("couch", 1)]
    reqs = parse.parse_all("Skippy, add a couch. Turn it to face me. Put a lamp next to it.")
    assert [(r["action"], r["target"]) for r in reqs] == [("add", ""), ("rotate", "me"), ("add", "@last")]
    assert reqs[1]["items"][0]["kind"] == "@last" and reqs[2]["arrangement"] == "beside"
    assert [r["action"] for r in parse.parse_all("add a couch! remove the lamp? undo; clear the room")] == ["add", "remove", "undo", "clear"]


@pytest.mark.parametrize("text,actions", [
    ("add a couch and then remove the lamp", ["add", "remove"]), ("remove the lamp and add a plant", ["remove", "add"]),
    ("add 2 chairs and remove the table", ["add", "remove"]), ("add a table, then clear the chairs", ["add", "remove"]),
    ("put a lamp next to the couch and turn the chair to face me", ["add", "rotate"]), ("add a bed and move the couch against the wall", ["add", "move"]),
    ("add a couch after that add a table", ["add", "add"]), ("add a couch then undo then add a table", ["add", "undo", "add"]),
    ("clear the room, then add a kitchen", ["clear", "add"]), ("add a couch and undo", ["add", "undo"]),
])
def test_verbs_of_their_own_make_requests_of_their_own(parse, text, actions):
    assert [r["action"] for r in parse.parse_all(text)] == actions


def test_lists_are_not_cut(parse):
    for text, n in [("add a couch, a table, and a lamp", 3), ("a couch and a lamp next to it", 2), ("add a couch and also a lamp", 2),
                    ("ditch the old bookcase and take out three stools", 2), ("get rid of the tv and the lamp", 2)]:
        reqs = parse.parse_all(text)
        assert len(reqs) == 1 and len(reqs[0]["items"]) == n, (text, [(r["action"], kinds(r)) for r in reqs])
    reqs = parse.parse_all("add a table and put four chairs around it")
    assert len(reqs) == 1 and kinds(reqs[0]) == [("table", 1), ("chair", 4)] and reqs[0]["target"] == "table"
    reqs = parse.parse_all("add four chairs and arrange them in a circle")
    assert len(reqs) == 1 and reqs[0]["arrangement"] == "circle"


def test_a_noise_part_is_dropped_when_something_is_asked(parse):
    reqs = parse.parse_all("Skippy, I'm feeling hungry, can you add a kitchen")
    assert [(r["action"], r["scene"]) for r in reqs] == [("add", "kitchen")]


def test_the_number_of_requests_is_bounded(parse):
    reqs = parse.parse_all("add a couch. " * 40)
    assert 1 <= len(reqs) <= 8


# --------------------------------------------------------------------------------------------- words
@pytest.mark.parametrize("text,count", [
    ("add twenty five chairs", 25), ("add forty-two crates", 42), ("add thirty two barrels", 32), ("add ninety nine barrels", 99),
    ("add three hundred crates", 300), ("add two dozen mugs", 24), ("add three dozen crates", 36), ("add a couple dozen crates", 24),
    ("add a thousand crates", 1000), ("add 1,000 crates", 1000), ("add 100,000 crates", 1000), ("add 2.5 chairs", 2),
    ("add one hundred and twenty barrels", 120), ("add a whole bunch of barrels", 6), ("add a good few chairs", 4),
    ("add a large number of crates", 6), ("add ten thousand crates", 1000), ("add a trio of lamps", 3),
])
def test_numbers(parse, text, count):
    req = one(parse, text)
    assert req["action"] == "add" and len(req["items"]) == 1 and req["items"][0]["count"] == count, (text, kinds(req))


def test_zero_is_nothing_never_everything(parse):
    for text in ["add 0 chairs", "add zero chairs", "remove zero chairs", "remove 0 couches"]:
        req = one(parse, text)
        assert req["action"] == "none" and not req["items"], text


def test_around_a_number_means_about(parse):
    for text, count in [("I need around 10 crates", 10), ("add around 5 chairs", 5), ("put around 6 chairs in a circle", 6)]:
        req = one(parse, text)
        assert kinds(req) == [(req["items"][0]["kind"], count)] and req["arrangement"] in ("unspecified", "circle"), (text, req)
    assert one(parse, "add chairs around 3 tables")["arrangement"] == "around"
    assert one(parse, "add 3 chairs around the table")["target"] == "table"


@pytest.mark.parametrize("text,count", [
    ("add to chairs", 2), ("add too chairs", 2), ("add for chairs", 4), ("add ate chairs", 8), ("add tree couches", 3), ("add free chairs", 3),
])
def test_speech_recognition_numbers(parse, text, count):
    req = one(parse, text)
    assert req["items"][0]["count"] == count and any("read" in n for n in req["notes"]), (text, req)


def test_words_that_look_like_numbers_stay_words(parse):
    assert kinds(one(parse, "make room for chairs")) == [("chair", 3)]
    assert one(parse, "put a chair next to the table")["target"] == "table"
    assert one(parse, "add a tree")["items"][0]["kind"] == "plant", "a tree is the nearest thing to a plant"


def test_anaphora_needs_the_last_request(parse):
    for text in ["add another one", "add one more of those", "add the same again", "do that again", "add a red one"]:
        req = one(parse, text)
        assert req["action"] == "add" and req.get("again") and not [i for i in req["items"]], text
    prev = parse("add three red chairs")
    req = parse.parse_all("add another one", prev)[0]
    assert kinds(req) == [("chair", 1)], req
    assert kinds(parse.parse_all("do that again", prev)[0]) == [("chair", 3)]
    req = parse.parse_all("add three red chairs and two blue ones")[0]
    assert kinds(req) == [("chair", 3), ("chair", 2)]


def test_postposed_colours(parse):
    for text in ["add a couch in red", "add 3 chairs colored red", "add 3 chairs, red", "add a couch that is red"]:
        assert "red" in one(parse, text)["items"][0]["adjectives"], text


# --------------------------------------------------------------------------------------------- scenes
def test_a_room_word_that_only_says_where_builds_nothing(parse):
    for text, kind in [("add a lamp in the kitchen", "floor lamp"), ("a kitchen stool", "stool"), ("an office lamp", "floor lamp"), ("a bedroom lamp", "floor lamp"),
                       ("a bathroom sink", "sink"), ("add a reception desk", "desk"), ("add a study lamp", "floor lamp"),
                       ("I need a couple of stools for the kitchen", "stool"), ("put a toilet in the bathroom", "toilet"),
                       ("add chairs to the kitchen", "chair"), ("a living room couch", "couch"), ("add a plant to the living room", "plant")]:
        req = one(parse, text)
        assert req["scene"] == "" and [i["kind"] for i in req["items"]] == [kind], (text, req["scene"], kinds(req))
    req = one(parse, "a dining room table with 6 chairs")
    assert req["scene"] == "" and kinds(req) == [("table", 1), ("chair", 6)]


def test_a_room_asked_for_is_built(parse):
    for text, scene in [("a living room", "living room"), ("I want a kitchen", "kitchen"), ("build me an office", "office"), ("make it a bedroom", "bedroom"),
                        ("a lounge", "living room"), ("a study", "office"), ("a kitchen with eight chairs", "kitchen"), ("start again with a bedroom", "bedroom"),
                        ("make this a dining room", "dining room"), ("can you make the room look like an office", "office")]:
        assert one(parse, text)["scene"] == scene, text
    req = one(parse, "a bedroom with two beds")
    assert req["scene"] == "bedroom" and kinds(req) == [("bed", 2)]


def test_negation(parse):
    for text in ["dont add a couch", "no chairs", "no more chairs", "add no chairs", "stop adding chairs", "do not remove the lamp", "don't move the table"]:
        assert one(parse, text)["action"] == "none", text
    assert kinds(one(parse, "add a table without chairs")) == [("table", 1)]
    assert kinds(one(parse, "add a couch but no table")) == [("couch", 1)]
    assert kinds(one(parse, "add a couch not a chair")) == [("couch", 1)]
    req = one(parse, "a living room without the tv")
    assert req["scene"] == "living room" and req["exclude"] == ["tv"] and not req["items"]
    req = one(parse, "a bedroom but no bed")
    assert req["exclude"] == ["bed"]


# --------------------------------------------------------------------------------------------- typos
@pytest.mark.parametrize("text,kind", [
    ("add couhc", "couch"), ("spawn chiars", "chair"), ("tabel", "table"), ("add coffe table", "coffee table"), ("add an ofice chair", "office chair"),
    ("add a refridgerator", "fridge"), ("add three barrles", "barrel"), ("add a lmap", "floor lamp"), ("add a tiolet", "toilet"), ("add flor lamp", "floor lamp"),
    ("add a microwav", "microwave"), ("add a telivision", "tv"),
])
def test_a_typo_is_repaired_and_the_player_is_told(parse, text, kind):
    req = one(parse, text)
    assert req["items"][0]["kind"] == kind and any(n.startswith("read '") for n in req["notes"]), (text, req)


@pytest.mark.parametrize("word", ["bridge", "truck", "crane", "coach", "beach", "choir", "tablet", "cable", "stable", "stone", "store", "shell",
                                  "planet", "plane", "chain", "plank", "garage", "piano", "dragon", "clock", "rug"])
def test_a_real_word_the_library_lacks_is_never_turned_into_furniture(parse, word):
    art = "an" if word[0] in "aeiou" else "a"
    req = one(parse, f"add {art} {word}")
    assert [i["kind"] for i in req["items"]] == [f"UNKNOWN:{word}"], (word, kinds(req))
    assert not req["notes"]


def test_a_possessive_or_plural_s_is_not_a_typo(parse):
    req = one(parse, "add a couch's lamp")
    assert not any("couch" in n for n in req["notes"])
    assert one(parse, "scatter a handful of boxs everywhere")["items"][0]["kind"] == "box"


# --------------------------------------------------------------------------------------------- hygiene
@pytest.mark.parametrize("text", [
    "add \"a couch\"", "add a `couch`", "add\u00a0a\u00a0couch", "add a couch\u200b", "add\u3000a\u3000couch",
    "\uff41\uff44\uff44 \uff41 \uff43\uff4f\uff55\uff43\uff48", "add a couch \U0001F600", "\x00add a couch\x00", "add a 'couch'", "ADD A COUCH!!!",
])
def test_odd_characters_are_folded_away(parse, text):
    req = one(parse, text)
    assert req["action"] == "add" and [i["kind"] for i in req["items"]] == ["couch"], repr(text)


@pytest.mark.parametrize("text", ["", " ", "\t\n", ",,,,,", "!!!???...", "-", "\u2019", "skippy\u2026", "please", "- - -", "\x00", "\U0001F600"])
def test_nothing_to_say_is_none(parse, text):
    assert one(parse, text)["action"] == "none", repr(text)


def test_long_input_is_cut_and_fast(parse):
    long = "add a couch and " + "a flibbert " * 100
    req = parse.parse_all(long)[0]
    assert req["truncated"] is True and len(req["raw"]) <= 400
    assert "truncated" not in one(parse, "add a couch")
    for pattern in ["please ", ", ", "a flibbert ", "and add a lamp ", "the table around ", "x "]:
        t0 = time.perf_counter()
        r = parse.parse_all(pattern * 4000)
        assert r and time.perf_counter() - t0 < 0.5, pattern


def test_normalise_is_linear(parse):
    normalise = parse.Parse.normalise
    for pattern in ["please ", ", ", "a ", "and ", "chair ", "1,000,"]:
        times = []
        for n in (20_000, 200_000):
            s = (pattern * (n // len(pattern) + 1))[:n]
            t0 = time.perf_counter()
            normalise(s)
            times.append(time.perf_counter() - t0)
        assert times[1] < 2.0 and times[1] < 40 * max(times[0], 0.002), (pattern, times)


@pytest.mark.parametrize("text,woke,rest", [
    ("hey there skippy add a couch", True, "add a couch"), ("okay so skippy add a couch", True, "add a couch"), ("um skippy add a couch", True, "add a couch"),
    ("yo skippy add a couch", True, "add a couch"), ("alright skippy add a couch", True, "add a couch"), ("hey, skippy: add a couch", True, "add a couch"),
    ("excuse me skippy add a couch", True, "add a couch"), ("skeepy add a couch", True, "add a couch"), ("skippey add a couch", True, "add a couch"),
    ("skippee add a couch", True, "add a couch"), ("Skippy\u2026add a couch", True, "add a couch"), ("\"skippy\" add a couch", True, "add a couch"),
    ("   " * 400 + "skippy add a couch", True, "add a couch"),
    ("skipping stones", False, "skipping stones"), ("skimpy dress", False, "skimpy dress"), ("skippers", False, "skippers"),
    ("skippy_add a couch", False, "skippy_add a couch"), ("tell skippy to add a couch", False, "tell skippy to add a couch"),
    ("did you hear what skippy said", False, "did you hear what skippy said"), ("we should get a couch", False, "we should get a couch"),
    ("skipper add a couch", False, "skipper add a couch"),
])
def test_wake_word(parse, text, woke, rest):
    ok, got = parse.Parse.strip_wake(text)
    assert bool(ok) == woke and got == rest, (text, ok, got)


def test_strip_wake_never_scans_a_flood(parse):
    t0 = time.perf_counter()
    for s in [" " * 1_000_000 + "x", "hey " * 250_000, "a" * 1_000_000]:
        parse.Parse.strip_wake(s)
    assert time.perf_counter() - t0 < 1.0


# --------------------------------------------------------------------------------------------- robustness
def test_random_sentences_never_crash_and_never_empty(parse):
    import random
    rnd = random.Random(7)
    words = ("add remove delete clear undo move rotate turn put make a an the it them and then also but except not no dont without of in on at by to from "
             "for next in front behind around beside against facing away couch table chair lamp tv plant crate kitchen bedroom three 2 ten dozen half all "
             "every everything one last row circle grid corner wall me here . , ! ? ; skippy please oops never mind both sides either flank seats six").split()
    odd = ["\u00a0", "\u2019", "\u200b", "\U0001F600", "\x00", "\u3000", "\u2026"]
    for _ in range(1500):
        toks = [rnd.choice(words) if rnd.random() > 0.05 else rnd.choice(odd) for _ in range(rnd.randint(1, 28))]
        reqs = parse.parse_all(" ".join(toks))
        assert reqs and all(r["action"] in ("add", "remove", "clear", "undo", "move", "rotate", "rearrange", "none") for r in reqs)
        assert len(reqs) <= 8


def test_a_destructive_action_is_never_a_side_effect_of_an_add(parse):
    for text in ["add an arch", "add a clear table", "put an arch over the door", "build a stone arch", "add a clean slate", "make a wall of crates",
                 "add a bin", "add a trash can", "put a barrel in the corner", "get a couch", "put a lamp away from the wall",
                 "add some chairs, a table and a lamp", "a table with four chairs", "put a plant on the floor", "add a reset button"]:
        for req in parse.parse_all(text):
            assert req["action"] in ("add", "none"), (text, req["action"])


def test_brackets_around_a_number_are_dropped(parse):
    assert kinds(one(parse, "add (2) chairs")) == [("chair", 2)]
    assert kinds(one(parse, "add [3] chairs")) == [("chair", 3)]
    assert kinds(one(parse, "add *3* chairs")) == [("chair", 3)]


def test_a_split_count_tells_the_two_apart(parse):
    req = one(parse, "add two couches one red and one blue")
    assert kinds(req) == [("couch", 1), ("couch", 1)]
    assert [i["adjectives"] for i in req["items"]] == [["red"], ["blue"]]
    assert kinds(one(parse, "three chairs, two red and one blue")) == [("chair", 2), ("chair", 1)]


def test_a_noun_that_is_also_a_verb_is_a_verb_only_before_a_thing(parse):
    assert kinds(one(parse, "bin")) == [("trash can", 1)]
    assert one(parse, "trash the barrels")["action"] == "remove"
    assert one(parse, "swivel chair")["action"] == "add"
    assert one(parse, "computer")["items"][0]["kind"] == "computer"
    assert one(parse, "computer, add three couches")["items"][0]["kind"] == "couch"
    assert one(parse, "computer desk")["items"][0]["kind"] == "desk"


def test_statements_and_questions_are_not_requests(parse):
    for text in ["the lamp is ugly", "how many chairs are there", "the table is too big", "what can you build", "this room is too crowded"]:
        assert one(parse, text)["action"] == "none", text


@pytest.mark.parametrize("text,scene", [("i want a dinng room", "dining room"), ("a bedrom", "bedroom"), ("build me an ofice", "office"),
                                        ("a kitchn", "kitchen"), ("a bathrom please", "bathroom")])
def test_a_room_word_with_a_typo_is_repaired(parse, text, scene):
    req = one(parse, text)
    assert req["scene"] == scene and any(n.startswith("read '") for n in req["notes"]), (text, req["scene"], req["notes"])


@pytest.mark.parametrize("text,target", [("put a lamp next to the coch", "couch"), ("put two stools beside the cuch", "couch"), ("arrange all the chairs around the meeting tble", "meeting table")])
def test_a_dropped_letter_in_a_short_word_is_a_slip(parse, text, target):
    assert one(parse, text)["target"] == target, text


def test_a_mistyped_greeting_is_not_a_computer(parse):
    for text in ["comupter, add a television", "computr, i need a reading lamp", "computeer, couches please"]:
        req = one(parse, text)
        assert "computer" not in [i["kind"] for i in req["items"]], (text, kinds(req))
    assert [i["kind"] for i in one(parse, "computer desk and a chair")["items"]] == ["desk", "chair"]
    assert not one(parse, "plan")["items"], "a real word one letter short of a thing is not a typo of it"


@pytest.mark.parametrize("text,arr", [
    ("put a lamp in the far corner", "corner"), ("a bed in the back corner", "corner"), ("put a chair in the far left corner", "corner"),
    ("a desk against the back wall", "against_wall"), ("put a bookshelf on the left wall", "against_wall"), ("add 6 chairs along the back wall", "against_wall"),
    ("a couch next to the wall", "against_wall"),
])
def test_walls_and_corners_with_their_adjectives(parse, text, arr):
    req = one(parse, text)
    assert req["arrangement"] == arr and req["target"] == "" and len(req["items"]) == 1, (text, req["arrangement"], req["target"], kinds(req))


@pytest.mark.parametrize("text,kind", [("add a sofa bed", "couch"), ("add a computer desk", "desk"), ("add an ice box", "fridge"),
                                       ("add a bed side table", "side table"), ("add a desk phone", "phone"), ("add a gaming pc", "computer")])
def test_a_compound_name_is_one_thing(parse, text, kind):
    assert [i["kind"] for i in one(parse, text)["items"]] == [kind], text


def test_a_lamp_on_a_table_is_the_desk_lamp(parse):
    for text in ["put a lamp on the table", "a lamp on the desk", "a mug and a lamp on the desk", "put a lamp on the nightstand"]:
        assert "desk lamp" in [i["kind"] for i in one(parse, text)["items"]], text
    assert [i["kind"] for i in one(parse, "put a floor lamp on the table")["items"]] == ["floor lamp"]
    assert [i["kind"] for i in one(parse, "put a lamp next to the table")["items"]] == ["floor lamp"]
    assert [i["kind"] for i in one(parse, "put the lamp on the table")["items"]] == ["floor lamp"], "an existing lamp keeps its kind"


def test_straighten_keeps_the_shape_and_line_up_makes_a_row(parse):
    assert one(parse, "straighten the chairs")["arrangement"] == "unspecified"
    assert one(parse, "line up all the chairs")["arrangement"] == "row"
    assert one(parse, "line the chairs up in front of me")["shape"] == "row"
