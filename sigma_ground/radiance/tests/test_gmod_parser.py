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
    reqs = parse.parse_all("a table with 4 chairs and a lamp on it")
    assert len(reqs) == 2, [(r["action"], kinds(r)) for r in reqs]
    assert kinds(reqs[0]) == [("table", 1), ("chair", 4)] and reqs[0]["arrangement"] == "unspecified", "the chairs stand around the table"
    assert kinds(reqs[1]) == [("desk lamp", 1)] and reqs[1]["arrangement"] == "on" and reqs[1]["target"] == "table", "only the lamp goes on it"


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


# --------------------------------------------------------------------------------------------- the high-severity review (2026-10-10)
# One class of defect each. Every sentence here is one whose meaning is certain; the held-out corpus holds many more of the same
# classes (slices keep_clauses, exceptions_hard, lead_ins, my_things, dont_want, scene_exclusions, attached_relations).
NEW_SLICES = ("keep_clauses", "exceptions_hard", "lead_ins", "my_things", "dont_want", "scene_exclusions", "attached_relations")


def test_the_review_slices_of_the_heldout_corpus_are_exact(parse):
    """The general pass-rate test allows a few misses; these slices are the defects of the review and allow none."""
    cases = [c for c in eval_parse.load_cases("utterance_corpus_heldout.json") if c["slice"] in NEW_SLICES]
    assert len(cases) >= 250
    bad = [(c["utterance"], eval_parse.compare_case(c, parse)) for c in cases]
    bad = [b for b in bad if b[1]]
    assert not bad, bad[:8]


def seeded(lab, *requests):
    lab.registry, lab.next_id, lab.seq, lab.gseq = [], 1, 0, 0
    for t in requests:
        lab.commit(lab.run(t, world="room"))


def standing(lab):
    out = {}
    for e in lab.registry:
        out[e["kind"]] = out.get(e["kind"], 0) + 1
    return out


def play(lab, text):
    """The utterance the way the game runs it (each request committed before the next), on what is already standing."""
    return lab.run_all(text, world="room")


# ---- a keep / leave / spare / don't-touch clause after a removal says what STAYS
@pytest.mark.parametrize("text,gone,kept", [
    ("remove the couch and keep the chairs", [("couch", 1)], ["chair"]),
    ("remove the couch while keeping the chairs", [("couch", 1)], ["chair"]),
    ("remove the couch, keep the chairs", [("couch", 1)], ["chair"]),
    ("delete the lamp, keep the couch", [("floor lamp", 1)], ["couch"]),
    ("delete the lamp and leave the table alone", [("floor lamp", 1)], ["table"]),
    ("remove the lamp and don't touch the chairs", [("floor lamp", 1)], ["chair"]),
    ("remove the lamp and do not touch the chairs", [("floor lamp", 1)], ["chair"]),
    ("get rid of the tv, leave the couch alone", [("tv", 1)], ["couch"]),
    ("delete the chairs and spare the table", [("chair", 0)], ["table"]),
    ("remove the chairs while leaving the table", [("chair", 0)], ["table"]),
    ("remove the chairs and preserve the table", [("chair", 0)], ["table"]),
    ("remove the tv and the lamp but keep the couch", [("tv", 1), ("floor lamp", 1)], ["couch"]),
    ("clear the room, keep the couch", [("@all", 0)], ["couch"]),
    ("clear the room but don't touch the couch", [("@all", 0)], ["couch"]),
    ("remove everything while keeping the couch", [("@all", 0)], ["couch"]),
    ("remove the lamp, leave everything else", [("floor lamp", 1)], []),
    ("get rid of the lamp, keep the rest", [("floor lamp", 1)], []),
    ("remove the lamp and leave everything else alone", [("floor lamp", 1)], []),
    ("delete the table and don't touch anything else", [("table", 1)], []),
])
def test_a_keep_clause_after_a_removal_is_never_removed(parse, text, gone, kept):
    req = one(parse, text)
    assert req["action"] == "remove" and kinds(req) == gone, (text, req["action"], kinds(req))
    assert sorted(req.get("exclude") or []) == sorted(kept), (text, req.get("exclude"))


def test_a_keep_clause_in_the_planner_leaves_the_kept_things_standing(lab):
    for text, left in [
        ("remove the couch and keep the chairs", {"chair": 2, "floor lamp": 1}),
        ("remove the couch while keeping the chairs", {"chair": 2, "floor lamp": 1}),
        ("remove the lamp, leave everything else", {"couch": 1, "chair": 2}),
        ("clear the room, keep the couch", {"couch": 1}),
        ("delete the lamp, keep the couch", {"couch": 1, "chair": 2}),
        ("delete the lamp and leave the table alone", {"couch": 1, "chair": 2}),
        ("remove the lamp and don't touch the chairs", {"couch": 1, "chair": 2}),
        ("get rid of the lamp, keep the rest", {"couch": 1, "chair": 2}),
        ("remove everything and don't touch the couch", {"couch": 1}),
    ]:
        seeded(lab, "a couch", "two chairs", "a lamp")
        play(lab, text)
        assert standing(lab) == left, (text, standing(lab))


@pytest.mark.parametrize("text,kinds_added", [
    ("add a couch and keep the chairs", [("couch", 1)]), ("add a lamp, keep the couch", [("floor lamp", 1)]),
    ("add three chairs and keep the table", [("chair", 3)]), ("add a table and spare the chairs", [("table", 1)]),
    ("spawn a lamp while keeping the couch", [("floor lamp", 1)]), ("add a lamp and leave the tv", [("floor lamp", 1)]),
    ("put a plant next to the couch and keep the tv", [("plant", 1)]),
])
def test_a_keep_clause_after_an_add_adds_nothing(parse, text, kinds_added):
    req = one(parse, text)
    assert req["action"] == "add" and kinds(req) == kinds_added and not req.get("exclude"), (text, kinds(req))


@pytest.mark.parametrize("text,action", [
    ("move the couch and leave the chairs", "move"), ("rotate the couch and spare the lamp", "rotate"),
    ("turn the chairs to face the tv and keep the couch as is", "rotate"), ("tidy up the chairs and keep the table", "rearrange"),
])
def test_a_keep_clause_after_a_move_or_turn_is_not_moved(parse, text, action):
    req = one(parse, text)
    assert req["action"] == action and [k for k, _ in kinds(req)] in (["couch"], ["chair"]), (text, kinds(req))


@pytest.mark.parametrize("text", ["keep the couch", "leave the couch alone", "spare the lamp", "keep them", "leave everything as it is",
                                  "remove the red couch and keep the blue couch", "remove the lamp and keep one chair"])
def test_keeping_or_a_keep_that_cannot_be_told_apart_changes_nothing(parse, text):
    req = one(parse, text)
    assert req["action"] == "none" and not req["items"], (text, req["action"], kinds(req))


def test_keep_then_add_is_two_requests_and_a_keep_alone_is_not_an_add(parse):
    reqs = parse.parse_all("keep the couch and add a lamp")
    assert [r["action"] for r in reqs] == ["none", "add"] and kinds(reqs[1]) == [("floor lamp", 1)]
    assert [r["action"] for r in parse.parse_all("keep the couch and remove the rest")] == ["remove"]


# ---- everything that isn't X / except the last one / get rid of everything else
@pytest.mark.parametrize("text,kept", [
    ("remove everything that isn't the couch", ["couch"]), ("remove everything that is not the couch", ["couch"]),
    ("delete anything thats not the bed", ["bed"]), ("remove everything which is not the table", ["table"]),
    ("delete everything that isnt a chair", ["chair"]), ("get rid of everything thats not the couch", ["couch"]),
    ("delete everything that isn't the tv or the lamp", ["tv", "floor lamp"]), ("clear all but the couch", ["couch"]),
])
def test_everything_that_is_not_x_keeps_x(parse, text, kept):
    req = one(parse, text)
    assert req["action"] == "remove" and kinds(req) == [("@all", 0)] and sorted(req["exclude"]) == sorted(kept), (text, req)


@pytest.mark.parametrize("text", [
    "remove everything except the last one", "remove everything but what I just built", "remove the furniture except what i just made",
    "remove all chairs except the first", "remove all chairs except the newest", "get rid of everything except the first one",
    "remove everything except", "remove everything but", "remove the chairs that are not red", "remove everything except red things",
    "remove everything but the dragon", "clear the room except what i just made", "remove everything and keep nothing",
    "delete everything that isnt the one i just made",
])
def test_an_exception_that_cannot_be_named_changes_nothing_never_clears(parse, text):
    reqs = parse.parse_all(text)
    assert [r["action"] for r in reqs] == ["none"] and not reqs[0]["items"], (text, [(r["action"], kinds(r)) for r in reqs])


def test_an_exception_that_cannot_be_named_leaves_the_room_alone(lab):
    for text in ["remove everything except the last one", "remove everything that isn't the one i just made", "remove everything but",
                 "remove all chairs except the first"]:
        seeded(lab, "a couch", "two chairs", "a lamp")
        before = standing(lab)
        play(lab, text)
        assert standing(lab) == before, (text, standing(lab))


def test_everything_that_is_not_the_couch_in_the_planner(lab):
    for text in ["remove everything that isn't the couch", "remove everything that is not the couch"]:
        seeded(lab, "a couch", "two chairs", "a lamp")
        play(lab, text)
        assert standing(lab) == {"couch": 1}, (text, standing(lab))


def test_everything_else_does_not_wipe_what_the_same_sentence_makes(parse, lab):
    reqs = parse.parse_all("i just want the couch, get rid of everything else")
    assert [r["action"] for r in reqs] == ["clear", "add"] and kinds(reqs[1]) == [("couch", 1)]
    reqs = parse.parse_all("add a table and remove everything else")
    assert [r["action"] for r in reqs] == ["clear", "add"] and kinds(reqs[1]) == [("table", 1)]
    reqs = parse.parse_all("move the couch next to the tv and delete everything else")
    assert [r["action"] for r in reqs] == ["move", "remove"] and reqs[1]["exclude"] == ["couch"]
    assert [r["action"] for r in parse.parse_all("add a lamp then clear the room")] == ["add", "clear"]
    seeded(lab, "a couch", "two chairs", "a lamp")
    play(lab, "add a table and remove everything else")
    assert standing(lab) == {"table": 1}
    seeded(lab, "a couch", "two chairs", "a lamp")
    play(lab, "i just want a couch, get rid of everything else")
    assert standing(lab) == {"couch": 1}


# ---- words before the verb that no list can hold
@pytest.mark.parametrize("text", [
    "you need to remove the lamp", "basically remove the lamp", "honestly delete the lamp", "literally remove the lamp",
    "definitely remove the lamp", "totally remove the lamp", "probably remove the lamp", "hey buddy remove the lamp",
    "hey bro remove the lamp", "hey captain remove the lamp", "you may remove the lamp", "you must remove the lamp",
    "you gotta remove the lamp", "last thing remove the lamp", "one second remove the lamp", "make sure to remove the lamp",
    "boss delete the lamp", "frankly get rid of the lamp", "you know what remove the lamp", "um yeah so basically get rid of the lamp",
    "you probably want to remove the lamp", "you have to get rid of the lamp",
])
def test_a_removal_behind_a_lead_in_is_still_a_removal(parse, text):
    req = one(parse, text)
    assert req["action"] == "remove" and kinds(req) == [("floor lamp", 1)] and req["items"][0]["definite"], (text, req["action"], kinds(req))


@pytest.mark.parametrize("text,action", [
    ("basically clear the room", "clear"), ("hey buddy clear everything", "clear"), ("you must undo that", "undo"),
    ("you need to undo that", "undo"), ("basically undo that", "undo"), ("hey buddy undo", "undo"),
    ("honestly dont add a couch", "none"), ("totally dont remove the couch", "none"), ("do you want to remove the lamp", "none"),
    ("you need to move the couch next to the tv", "move"), ("okay captain rotate the couch", "rotate"),
    ("honestly straighten the chairs", "rearrange"), ("basically tidy up", "rearrange"),
])
def test_other_managing_verbs_behind_a_lead_in(parse, text, action):
    assert one(parse, text)["action"] == action, text


def test_a_lead_in_does_not_remove_a_real_request(parse):
    assert one(parse, "i need a couch")["action"] == "add" and kinds(one(parse, "i need three chairs")) == [("chair", 3)]
    assert one(parse, "basically add a couch next to the tv")["arrangement"] == "beside"
    assert one(parse, "dont move the table")["action"] == "none"
    assert kinds(one(parse, "hey so like i was thinking maybe we could use a bookshelf")) == [("bookshelf", 1)]


def test_a_lead_in_removal_in_the_planner(lab):
    for text in ["you need to remove the lamp", "basically remove the lamp", "hey buddy remove the lamp", "honestly delete the lamp"]:
        seeded(lab, "a couch", "a lamp")
        play(lab, text)
        assert standing(lab) == {"couch": 1}, (text, standing(lab))
    seeded(lab, "a couch", "a lamp")
    play(lab, "basically clear the room")
    assert standing(lab) == {}


# ---- "my" names the things that stand, like "the" and "our"
@pytest.mark.parametrize("text,action,item", [
    ("remove my chairs", "remove", ("chair", 0)), ("remove all of my chairs", "remove", ("chair", 0)),
    ("get rid of my chairs", "remove", ("chair", 0)), ("remove my couch", "remove", ("couch", 1)),
    ("put my couch next to the tv", "move", ("couch", 1)), ("put my chairs around the table", "move", ("chair", 0)),
    ("make my chairs face the tv", "rotate", ("chair", 0)), ("bring my chairs over here", "move", ("chair", 0)),
    ("stack my crates", "rearrange", ("crate", 0)), ("line up my chairs", "rearrange", ("chair", 0)),
    ("spread my chairs out", "rearrange", ("chair", 0)), ("move my chairs next to the table", "move", ("chair", 0)),
    ("turn my chairs to face the tv", "rotate", ("chair", 0)), ("push my bed into the corner", "move", ("bed", 1)),
    ("put my lamp on my desk", "move", ("floor lamp", 1)),
])
def test_my_names_what_already_stands(parse, text, action, item):
    req = one(parse, text)
    assert req["action"] == action and kinds(req) == [item] and req["items"][0]["definite"], (text, req["action"], kinds(req))


def test_my_is_not_a_new_thing(parse):
    assert one(parse, "my couch is ugly")["action"] == "none"
    assert one(parse, "clear my room")["action"] == "clear" and one(parse, "clear our room")["action"] == "clear"
    assert one(parse, "undo my last action")["action"] == "undo"
    req = one(parse, "put a lamp on my desk")
    assert req["action"] == "add" and kinds(req) == [("desk lamp", 1)] and req["target"] == "desk"


def test_my_chairs_in_the_planner(lab):
    seeded(lab, "four chairs")
    play(lab, "remove my chairs")
    assert standing(lab) == {}, "all four, not the default three"
    seeded(lab, "a couch", "a lamp")
    res = play(lab, "put my couch next to the lamp")
    assert standing(lab) == {"couch": 1, "floor lamp": 1} and res[0]["action"] == "move", "the couch is moved, not duplicated"
    seeded(lab, "four chairs", "a table")
    play(lab, "put my chairs around the table")
    assert standing(lab) == {"chair": 4, "table": 1}


# ---- "i dont want X <where or how>" is not "remove X"
@pytest.mark.parametrize("text", [
    "i dont want the couch by the window", "i don't want the couch by the window", "i dont want the couch to face the tv",
    "i dont want the couch facing the wall", "i dont want the couch there", "i dont like the couch there", "i dont like the couch",
    "i dont like it", "i dont want it by the window", "i dont want it facing the tv", "i dont want the couch next to the table",
    "i dont want the couch in the corner", "i dont want the tv that close", "i dont want the couch so close to the tv",
    "i dont want the lamp on the table", "i dont want the chairs around the table", "i dont want the couch here",
    "i dont want the couch because it is ugly", "i no longer like the couch", "we dont like the lamp",
])
def test_i_dont_want_it_somewhere_is_not_a_removal(parse, text):
    req = one(parse, text)
    assert req["action"] == "none" and not req["items"], (text, req["action"], kinds(req))


@pytest.mark.parametrize("text,item", [
    ("i dont want the couch", ("couch", 1)), ("i dont want the couch anymore", ("couch", 1)), ("i dont need the lamp anymore", ("floor lamp", 1)),
    ("i dont want the chairs anymore", ("chair", 0)), ("i dont want the ugly couch", ("couch", 1)), ("we dont want the couch", ("couch", 1)),
    ("i no longer want the couch", ("couch", 1)), ("i dont want the old wooden table anymore", ("table", 1)),
])
def test_i_dont_want_the_thing_still_removes_it(parse, text, item):
    req = one(parse, text)
    assert req["action"] == "remove" and kinds(req) == [item], text


def test_i_dont_want_it_there_keeps_the_room(lab):
    for text in ["i dont want the couch by the window", "i dont want the couch to face the tv", "i dont like the couch"]:
        seeded(lab, "a couch", "a tv")
        play(lab, text)
        assert standing(lab) == {"couch": 1, "tv": 1}, (text, standing(lab))


# ---- what the parser says about a room without something (the planner reads req.exclude)
@pytest.mark.parametrize("text,scene,exclude", [
    ("a living room without the tv", "living room", ["tv"]), ("a living room with no tv", "living room", ["tv"]),
    ("a bedroom but no bed", "bedroom", ["bed"]), ("a kitchen without a fridge", "kitchen", ["fridge"]),
    ("a living room except the tv", "living room", ["tv"]), ("a living room sans tv", "living room", ["tv"]),
    ("a kitchen without the fridge and the stove", "kitchen", ["fridge", "stove"]),
    ("a living room without the tv or the lamp", "living room", ["tv", "floor lamp"]),
    ("a living room and leave out the tv", "living room", ["tv"]), ("a living room and omit the plant", "living room", ["plant"]),
    ("a living room that doesn't have a tv", "living room", ["tv"]), ("a dining room but no table", "dining room", ["table"]),
    ("a living room with no tv and no lamp", "living room", ["tv", "floor lamp"]),
])
def test_a_room_without_a_thing_names_it_in_exclude_and_does_not_add_it(parse, text, scene, exclude):
    req = one(parse, text)
    assert req["scene"] == scene and sorted(req["exclude"]) == sorted(exclude) and not req["items"], (text, req["exclude"], kinds(req))


def test_a_room_without_a_thing_keeps_the_rest_it_asks_for(parse):
    req = one(parse, "a bedroom with a bed but no wardrobe")
    assert kinds(req) == [("bed", 1)] and req["exclude"] == ["wardrobe"]
    req = one(parse, "a living room without the tv and a plant in the corner")
    assert kinds(req) == [("plant", 1)] and req["exclude"] == ["tv"] and req["arrangement"] == "corner"


# ---- "a table with four chairs and a lamp on it": the relation is about the lamp alone
@pytest.mark.parametrize("text,second,arr,target", [
    ("add a table with four chairs and a lamp on it", [("desk lamp", 1)], "on", "table"),
    ("add a table with four chairs and a plant next to it", [("plant", 1)], "beside", "table"),
    ("add a desk with a chair and a computer on it", [("computer", 1)], "on", "desk"),
    ("add a table with four chairs and put a lamp on it", [("desk lamp", 1)], "on", "table"),
    ("add a table with four chairs and put a plant next to it", [("plant", 1)], "beside", "table"),
    ("a table, four chairs and a lamp on it", [("desk lamp", 1)], "on", "table"),
])
def test_a_relation_after_a_list_belongs_to_the_last_thing_only(parse, text, second, arr, target):
    reqs = parse.parse_all(text)
    assert len(reqs) == 2, (text, [(r["action"], kinds(r)) for r in reqs])
    assert reqs[0]["arrangement"] == "unspecified" and len(reqs[0]["items"]) == 2, "the chairs are not placed by the lamp's words"
    assert kinds(reqs[1]) == second and reqs[1]["arrangement"] == arr and reqs[1]["target"] == target, (text, reqs[1]["target"])


def test_the_chairs_of_a_table_are_not_put_on_it(lab):
    for text, top in [("add a table with four chairs and a lamp on it", "desk lamp"), ("add a desk with a chair and a computer on it", "computer")]:
        seeded(lab)
        play(lab, text)
        high = [e for e in lab.registry if e["kind"] == top]
        chairs = [e for e in lab.registry if e["kind"] == "chair"]
        assert len(high) == 1 and high[0]["z"] > 20, (text, high)
        assert chairs and all(c["z"] < 1 for c in chairs), (text, [c["z"] for c in chairs])


def test_listing_without_a_pointer_back_is_still_one_request(parse):
    assert len(parse.parse_all("add a table and put four chairs around it")) == 1
    assert len(parse.parse_all("a couch and a lamp next to it")) == 1
    assert len(parse.parse_all("a couch, a table and a lamp in front of the tv")) == 1


def test_the_new_clause_readers_stay_linear(parse):
    """Keep clauses, hedges, 'that isnt', 'else', 'i dont want', 'without' lists and 'with' lists on the longest input the parser reads."""
    for pattern in ["remove the couch and keep the chairs, ", "keep ", "basically honestly literally ", "you need to ", "remove everything that isnt the couch and ",
                    "add a table and remove everything else. ", "i dont want the couch by the window and ", "a living room without the tv and the lamp and ",
                    "a table with four chairs and a lamp on it and ", "remove everything that is red and that is blue ", "remove my chairs and my couch "]:
        t0 = time.perf_counter()
        for _ in range(20):
            parse.parse_all(pattern * 400)
        assert (time.perf_counter() - t0) / 20 < 0.1, pattern


@pytest.mark.parametrize("text", ["you cannot remove the couch", "you shouldnt remove the lamp", "you wont remove the lamp",
                                  "you mustnt remove the lamp", "you dont need to remove the lamp"])
def test_a_negated_modal_before_a_removal_is_a_refusal(parse, text):
    assert one(parse, text)["action"] == "none", text


@pytest.mark.parametrize("text", ["remove everything thats red", "remove everything that is red", "remove everything which is big",
                                  "delete anything that is blue", "remove everything that is next to the tv", "remove the chairs that are broken"])
def test_a_description_that_picks_some_out_never_clears_everything(parse, text):
    reqs = parse.parse_all(text)
    assert [r["action"] for r in reqs] == ["none"], (text, [(r["action"], kinds(r)) for r in reqs])


def test_a_colour_after_that_are_is_a_colour(parse):
    req = one(parse, "remove the chairs that are red")
    assert req["action"] == "remove" and kinds(req) == [("chair", 0)] and req["items"][0]["adjectives"] == ["red"]


@pytest.mark.parametrize("text,second", [("add a couch and delete the others", ("add", [("couch", 1)])),
                                         ("add two chairs, clear the rest", ("add", [("chair", 2)]))])
def test_the_others_after_an_add_are_what_stood_before(parse, text, second):
    reqs = parse.parse_all(text)
    assert [r["action"] for r in reqs] == ["clear", second[0]] and kinds(reqs[1]) == second[1], text


@pytest.mark.parametrize("text", ["it would be great if you could remove the lamp", "what i want is for you to get rid of the couch",
                                  "so basically what i need is for you to remove the lamp"])
def test_a_polite_frame_before_a_removal_is_still_a_removal(parse, text):
    req = one(parse, text)
    assert req["action"] == "remove" and len(req["items"]) == 1 and req["items"][0]["definite"], text


# --------------------------------------------------------------------------------------------- P0 classes (2026-10-10)
# Sentences that are NOT a request, or are a different request, must never be run as a destructive action or a wrong add.
# Each class below has 26-61 sentences in the held-out corpus (written from the meaning of each sentence BEFORE the parser's answer was
# read); a class is required at 100%, because one destroyed room is worse than a hundred misses.
P0_SLICES = ["selfcorrect", "undo_forms", "unplaced_object", "chat_retraction", "colour_one", "hypothetical", "modal_requests", "past_purpose",
             "toss_chuck", "casual_clear", "leading_no", "apostrophes", "fewer_less", "neg_scope", "chat_words", "bare_plural_remove",
             "reference_prep", "side_of", "opinions"]


@pytest.mark.parametrize("name", P0_SLICES)
def test_p0_class_is_perfect(parse, name):
    cases = [c for c in eval_parse.load_cases("utterance_corpus_heldout.json") if c["slice"] == name]
    assert len(cases) >= 26, (name, len(cases))
    failures = []
    for c in cases:
        try:
            bad = eval_parse.compare_case(c, parse)
        except Exception as e:
            bad = [f"CRASH {e}"]
        if bad:
            failures.append((c["utterance"], bad[:2]))
    assert not failures, f"{name}: {len(failures)} of {len(cases)} misread:\n" + "\n".join(f"  {u!r}: {b}" for u, b in failures[:15])


def test_spoken_correction_replaces_never_adds_both(parse):
    for text, kind in [("remove the couch, no wait, the lamp", "floor lamp"), ("remove the lamp actually the couch", "couch"),
                       ("remove the couch no the lamp", "floor lamp"), ("remove the couch, no wait remove the lamp", "floor lamp"),
                       ("Remove the couch. No wait, the lamp.", "floor lamp")]:
        reqs = parse.parse_all(text)
        assert [r["action"] for r in reqs] == ["remove"] and [i["kind"] for i in reqs[0]["items"]] == [kind], (text, reqs)
    # the dropped part carried a destination: nothing is moved rather than the wrong thing
    assert all(r["action"] == "none" for r in parse.parse_all("move the couch next to the tv, no wait, the lamp"))
    # a correction that belongs to the second of two requests leaves the first alone
    reqs = parse.parse_all("add a lamp and remove the couch, no wait, the table")
    assert [(r["action"], [i["kind"] for i in r["items"]]) for r in reqs] == [("add", ["floor lamp"]), ("remove", ["table"])]
    # words that merely contain a marker are not corrections
    assert [i["kind"] for i in one(parse, "add a table with no chairs")["items"]] == ["table"]
    assert [i["kind"] for i in one(parse, "remove the couch and the lamp")["items"]] == ["couch", "floor lamp"]


def test_undo_with_a_count_carries_times_and_never_clears(parse):
    for text, times in [("undo twice", 2), ("undo 3 times", 3), ("undo the last two", 2), ("undo once", 1), ("undo my last two requests", 2)]:
        req = one(parse, text)
        assert req["action"] == "undo" and req.get("times") == times, (text, req)
    for text in ["undo everything", "cancel everything", "revert everything", "undo it all"]:
        req = one(parse, text)
        assert req["action"] == "undo" and (req.get("times") or 30) >= 10, (text, req)
    for text in ["undo what you just did", "undo that", "undo the whole thing"]:
        req = one(parse, text)
        assert req["action"] == "undo" and not req.get("times"), (text, req)


def test_the_side_of_a_thing_is_a_field_and_the_reference_stays_put(parse):
    for text, side in [("move the lamp to the other side of the couch", "other"), ("move the lamp to the opposite side of the couch", "opposite"),
                       ("move the lamp to the right side of the couch", "right"), ("move the lamp to the left side of the couch", "left")]:
        req = one(parse, text)
        assert req["action"] == "move" and [i["kind"] for i in req["items"]] == ["floor lamp"], (text, req)
        assert req["target"] == "couch" and req["side"] == side, (text, req)
    assert "side" not in one(parse, "move the lamp next to the couch") or not one(parse, "move the lamp next to the couch").get("side")


@pytest.mark.parametrize("text,woke,rest", [
    ("Skippy's couch is ugly", False, "Skippy's couch is ugly"), ("Skippy’s couch is ugly", False, None), ("Skippy‘s chairs are cool", False, None),
    ("skippy's a good boy", False, None), ("skippy`s lamp broke", False, None), ("Skippyʼs bed is big", False, None),
    ("Skippy´s desk is messy", False, None), ("Skippy′s tv is on", False, None), ("hey Skippy's table is nice", False, None),
    ("hey skippy's couch", False, None), ("ok skippy's the best", False, None), ("skippys couch", False, None), ("Skippy's chairs are cool", False, None),
    ("Skippy's lamp broke", False, None), ("skippy's a bed", False, None), ("so Skippy's plant died", False, None), ("Skippy’s", False, None),
    ("Skippy's room is great", False, None), ("Skippy’s couch", False, None), ("Skippy's house", False, None),
    ("Skippy add a couch", True, "add a couch"), ("skippy, add a couch", True, "add a couch"), ("Skippy: remove the lamp", True, "remove the lamp"),
    ("hey skippy undo", True, "undo"), ("Skippy's add a couch", True, "add a couch"), ("Skippy’s remove the lamp", True, "remove the lamp"),
    ("Skippy's clear the room", True, "clear the room"), ("okay skippy put a lamp here", True, "put a lamp here"), ("Skippy", True, ""),
    ("Skippy, the couch is ugly", True, "the couch is ugly"), ("skipping stones", False, None), ("skippers", False, None),
])
def test_the_possessive_skippys_is_not_the_wake_word(parse, text, woke, rest):
    got_woke, got_rest = parse.Parse.strip_wake(text)
    assert got_woke == woke, (text, got_woke, got_rest)
    if rest is not None:
        assert got_rest == rest, (text, got_rest)
    if not woke:
        assert got_rest == text


def test_new_readers_stay_linear(parse):
    """The mood gate, the correction fold, the refusal scan and the undo reader on the longest input the parser reads."""
    for pattern in ["remove the couch, no wait, ", "should i clear the room and ", "undo twice ", "add a couch and dont add a lamp and ",
                    "i like the couch and ", "no wait ", "sorry ", "remove the red one and ", "one less chair and ", "move the lamp to the other side of the couch and ",
                    "toss a crate and ", "a lamp past the couch and ", "that is enough, "]:
        t0 = time.perf_counter()
        for _ in range(20):
            parse.parse_all(pattern * 400)
        assert (time.perf_counter() - t0) / 20 < 0.1, pattern
