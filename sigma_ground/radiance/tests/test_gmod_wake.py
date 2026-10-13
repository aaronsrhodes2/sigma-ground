"""Per-player wake words (Aaron, 2026-10-12: "if someone decides to name you asshole, they have that right"), offline.

  * wakewords.lua (pure) and its Python twin wakewords.py are held to the same table of sentences (wake_vectors.json) and the same
    accent folding;
  * wakenames.lua + holodeck.lua + avatar.lua + skippy_client.lua run for real under LuaJIT (lupa) against gmod_mock.lua: storage,
    persistence, the commands in typed / spoken / console form, privacy, the colliding-name refusals, the last-name guard, the cap,
    corrupt-file fallback, the owner-only net message and the client's file;
  * skippy_ears.py's use of the names file is tested through its pure functions (no microphone, no Whisper).
What cannot be proved here: the net message to a real client, speech recognition of unusual names (see stage/gmod/holodeck/README.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "stage" / "gmod" / "holodeck"))

from test_gmod_glue import ADDON, World  # noqa: E402
import wakewords as PW  # noqa: E402  (the Python twin)

A, B, C = "76561190000000001", "76561190000000002", "76561190000000003"
VECTORS = json.loads((HERE / "wake_vectors.json").read_text(encoding="utf-8"))
FILE = "mentat/wake_words.json"


# ------------------------------------------------------------------------------------------------ the pure module, in Lua
@pytest.fixture(scope="module")
def lw():
    try:
        from lupa import luajit21 as mod
    except ImportError:  # pragma: no cover
        from lupa import lua51 as mod
    rt = mod.LuaRuntime(unpack_returned_tuples=True)
    rt.execute("jit.off()")
    rt.execute(f'function include(p) return dofile("{ADDON.as_posix()}/" .. p) end')
    mod_ = rt.eval('include("mentat/holodeck/wakewords.lua")')
    return rt, mod_


def lua_list(rt, items):
    t = rt.table()
    for i, v in enumerate(items, 1):
        t[i] = v
    return t


@pytest.mark.parametrize("v", VECTORS, ids=[f"{'+'.join(v['names'])}|{v['text']}" for v in VECTORS])
def test_lua_and_python_agree_on_every_sentence(lw, v):
    """The server (Lua) and the microphone bridge (Python) wake on exactly the same sentences."""
    rt, W = lw
    got = W.strip(v["text"], lua_list(rt, v["names"]))
    woke, rest = got[0], got[1]
    assert (woke, rest) == (v["woke"], v["rest"]), "Lua"
    pw, prest, _ = PW.strip(v["text"], v["names"])
    assert (pw, prest) == (v["woke"], v["rest"]), "Python"


def test_accent_folding_is_the_same_in_lua_and_python(lw):
    rt, W = lw
    for cp in list(range(0xA0, 0x180)) + [0x2018, 0x2019, 0x201C, 0x2013, 0x2014, 0x2026, 0x4E2D, 0x1F600]:
        ch = chr(cp)
        assert W.fold(ch) == PW.fold(ch), hex(cp)


@pytest.mark.parametrize("raw,expect", [
    ("Rex", "rex"), ("  “Ass-Hole”  ", "ass-hole"), ("BIG   tony", "big tony"), ("'Rex'", "rex"), ("Rex!", "rex"),
    ("Éléonore", "eleonore"), ("O’Brien", "o'brien"), ("R2D2", "r2d2"), ("Captain Obvious Jr", "captain obvious jr"),
])
def test_names_are_normalised(lw, raw, expect):
    _rt, W = lw
    assert W.normalise_name(raw) == expect
    assert PW.normalise_name(raw) == expect


@pytest.mark.parametrize("raw,code", [
    ("", "empty"), ("x", "short"), ("a" * 25, "long"), ("one two three four", "words"), ("rex<script>", "chars"),
    ("中文", "chars"), ("\U0001F600 rex", "chars"), ("   ", "empty"), ("--", "empty"),
])
def test_unusable_names_are_refused_with_a_code(lw, raw, code):
    _rt, W = lw
    name, got, why = W.normalise_name(raw)
    assert name is None and got == code and why
    assert PW.normalise_name(raw) is None


@pytest.mark.parametrize("name", ["add", "remove", "couch", "come", "here", "go away", "come here", "two", "the", "undo", "skipping",
                                  "name", "answer", "reset", "please", "undo it"])
def test_names_that_would_wake_him_on_ordinary_talk_are_refused(lw, name):
    _rt, W = lw
    ok, code, why = W.check_name(name, None)
    assert ok is False and code in ("reserved", "pronoun") and why


@pytest.mark.parametrize("name", ["asshole", "ass hole", "jerk", "dumbass", "butthead", "poop", "big tony", "rex", "couch potato", "o'brien",
                                  "jarvis", "computer", "bob", "fuckface", "idiot"])
def test_rude_and_silly_names_are_never_refused(lw, name):
    """Aaron's rule: the word itself is never a reason to refuse."""
    _rt, W = lw
    assert W.check_name(name, None) is True


def test_a_noun_of_the_library_is_refused_through_the_parser_lookup(lw):
    _rt, W = lw
    is_noun = lambda w: w in ("ottoman", "futon")   # noqa: E731
    assert W.check_name("ottoman", is_noun)[0] is False
    assert W.check_name("rex", is_noun) is True


@pytest.mark.parametrize("text,kind,name", [
    ("Answer to Rex", "add", "rex"), ("answer to the name Rex", "add", "rex"), ("Also answer to Big Tony, too", "add", "big tony"),
    ("call yourself Asshole", "add", "asshole"), ("from now on call yourself Rex", "add", "rex"),
    ("your name is also Rex", "add", "rex"), ("could you please answer to Rex", "add", "rex"), ("anwser to Rex", "add", "rex"),
    ("go by Rex", "add", "rex"), ("you can also be called Rex", "add", "rex"), ("add the name Rex", "add", "rex"),
    ("answer to “Rex”", "add", "rex"), ("responding to Rex", None, None),
    ("stop answering to Rex", "remove", "rex"), ("forget the name Rex", "remove", "rex"), ("don't answer to Rex anymore", "remove", "rex"),
    ("remove Rex from your names", "remove", "rex"), ("stop answerin to Rex", "remove", "rex"),
    ("only answer to Rex", "only", "rex"), ("answer only to Rex", "only", "rex"), ("change your wake word to Rex", "only", "rex"),
    ("what do you answer to", "list", None), ("What are your names?", "list", None), ("what names do you answer to", "list", None),
    ("reset your names", "reset", None), ("Reset your name.", "reset", None),
    ("answer to", "add", ""), ("stop answering to", "remove", ""),
    ("add three couches", None, None), ("what is the weather", None, None), ("the name of the lamp is nice", None, None),
    ("remove the couch", None, None), ("forget it", None, None), ("clear everything", None, None), ("what do you think", None, None),
])
def test_the_name_commands_are_recognised_whole_and_nothing_else_is(lw, text, kind, name):
    _rt, W = lw
    got = W.intent(text)
    if kind is None:
        assert got is None, text
    else:
        assert got is not None, text
        assert got["kind"] == kind and (name is None or got["name"] == name), text


def test_intent_never_folds_a_flood(lw):
    _rt, W = lw
    assert W.intent("answer to " + "x" * 100000) is None


def test_the_possessive_wakes_him_only_before_an_order(lw):
    rt, W = lw
    assert W.strip("Rex's lamp is nice", lua_list(rt, ["rex"]))[0] is False
    assert W.strip("Rex's remove the lamp", lua_list(rt, ["rex"]))[0] is True


# ------------------------------------------------------------------------------------------------ helpers over a booted world
def make_world(files=None, load_avatar=False, **cv):
    cvars = {"holodeck_bridge": "", "holodeck_agent": "0", "holodeck_persona": "0", "holodeck_voice_out": "0", **cv}
    w = World(cvars=cvars, load=())
    for k, v in (files or {}).items():
        w.rt.eval("function(k, v) MOCK.files[k] = v end")(k, v)
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/holodeck.lua")')
    if load_avatar:
        w.rt.execute(
            "MOCK.bots = {}\n"
            "function SetGlobalEntity(k, v) end\nfunction GetGlobalEntity(k) end\n")
        from test_gmod_avatar import PRELUDE
        w.rt.execute(PRELUDE)
        w.rt.execute('dofile(MOCK.ROOT .. "/mentat/avatar.lua")')
    return w


def names_of(w, p):
    t = w.rt.eval("function(p) return Holodeck.Wake.names(p) end")(p)
    return [t[i] for i in range(1, len(list(t.values())) + 1)]


def say(w, p, text):
    """Typed chat: the PlayerSay hook (its return value is what the engine uses to hide the line), then let the clock pass."""
    r = w.rt.eval("function(p, t) return hook.Run('PlayerSay', p, t) end")(p, text)
    w.advance(1.3)
    return r[0] if isinstance(r, tuple) else r


def ask(w, p, text):
    return w.ask(p, text)


def says(w):
    return list(w.rt.eval("function() return T.nets('holodeck_say') end")().values())


def writes(m):
    return list(m.writes.values())


def recipients(m):
    return [m.to[i].sid for i in range(1, len(list(m.to.values())) + 1)]


def last_to(w, sid):
    """The last line sent to this player alone (what Skippy told them)."""
    out = [m for m in says(w) if recipients(m) == [sid]]
    return writes(out[-1])[0] if out else None


def lamps(w, sid=None):
    return w.count(sid)


def stored(w):
    return w.rt.eval("function() return MOCK.files['mentat/wake_words.json'] end")()


# ------------------------------------------------------------------------------------------------ storage and the commands
def test_the_default_is_skippy_and_skippy_still_works():
    w = make_world()
    p = w.player(A)
    assert names_of(w, p) == ["skippy"]
    say(w, p, "Skippy, add two lamps")
    assert lamps(w, A) == 2
    say(w, p, "Skipy add a lamp")
    assert lamps(w, A) == 3
    say(w, p, "tell Skippy to add a lamp")           # a name later in the sentence does not wake him
    assert lamps(w, A) == 3


def test_a_player_adds_a_name_by_typing_it_and_it_then_wakes_him():
    w = make_world()
    p = w.player(A)
    say(w, p, "Skippy, answer to Rex")
    assert names_of(w, p) == ["skippy", "rex"]
    assert "Rex" in last_to(w, A)
    say(w, p, "Rex, add two lamps")
    assert lamps(w, A) == 2
    say(w, p, "Skippy add a lamp")
    assert lamps(w, A) == 3


def test_a_rude_name_is_added_without_a_word_of_complaint():
    w = make_world()
    p = w.player(A)
    say(w, p, "Skippy, call yourself Asshole")
    assert names_of(w, p) == ["skippy", "asshole"]
    reply = last_to(w, A)
    assert "Asshole" in reply and "cannot" not in reply.lower() and "refuse" not in reply.lower()
    say(w, p, "Asshole, add a lamp")
    assert lamps(w, A) == 1
    say(w, p, "ass hole add a lamp")                  # as speech recognition writes it
    assert lamps(w, A) == 2
    say(w, p, "hey asshole, add a lamp")
    assert lamps(w, A) == 3


def test_every_current_name_works_as_the_wake_word_for_the_commands_and_the_console_needs_none():
    w = make_world()
    p = w.player(A)
    say(w, p, "Skippy, answer to Rex")
    say(w, p, "Rex, answer to Zed")                   # said with the new name
    assert names_of(w, p) == ["skippy", "rex", "zed"]
    ask(w, p, "answer to Bob")                         # the console / the voice bridge after the wake word: no name at all
    assert names_of(w, p) == ["skippy", "rex", "zed", "bob"]
    ask(w, p, "Zed, stop answering to Rex")
    assert names_of(w, p) == ["skippy", "zed", "bob"]
    ask(w, p, "forget the name Zed")
    assert names_of(w, p) == ["skippy", "bob"]


def test_the_commands_do_not_need_the_wake_word_in_the_console_form_either():
    w = make_world()
    p = w.player(A)
    w.rt.eval("function(p) MOCK.cmds['holodeck'](p, 'holodeck', {'answer', 'to', 'Rex'}, 'answer to Rex') end")(p)
    assert names_of(w, p) == ["skippy", "rex"]


def test_removing_a_name_including_skippy_itself_as_long_as_one_remains():
    w = make_world()
    p = w.player(A)
    ask(w, p, "answer to Rex")
    ask(w, p, "stop answering to Skippy")
    assert names_of(w, p) == ["rex"]
    say(w, p, "Skippy, add a lamp")                   # no longer his name for this player
    assert lamps(w, A) == 0
    say(w, p, "Rex, add a lamp")
    assert lamps(w, A) == 1


def test_the_last_name_cannot_be_removed_and_the_reason_is_told():
    w = make_world()
    p = w.player(A)
    ask(w, p, "stop answering to Skippy")
    assert names_of(w, p) == ["skippy"]
    assert "only name" in last_to(w, A)
    ask(w, p, "only answer to Rex")
    ask(w, p, "stop answering to Rex")
    assert names_of(w, p) == ["rex"]


def test_removing_a_name_that_is_not_there_changes_nothing():
    w = make_world()
    p = w.player(A)
    ask(w, p, "stop answering to Nobody")
    assert names_of(w, p) == ["skippy"] and "do not answer" in last_to(w, A)


def test_only_answer_to_replaces_all_and_reset_brings_skippy_back():
    w = make_world()
    p = w.player(A)
    ask(w, p, "answer to Rex")
    ask(w, p, "answer to Zed")
    ask(w, p, "only answer to Asshole")
    assert names_of(w, p) == ["asshole"]
    ask(w, p, "reset your names")
    assert names_of(w, p) == ["skippy"]
    assert stored(w) is not None and "76561190000000001" not in stored(w)      # the default is not stored


def test_listing_the_names_is_said_back_privately():
    w = make_world()
    p = w.player(A)
    ask(w, p, "what do you answer to")
    assert "Skippy" in last_to(w, A)
    ask(w, p, "answer to Rex")
    ask(w, p, "answer to Big Tony")
    ask(w, p, "what are your names?")
    reply = last_to(w, A)
    assert "Skippy" in reply and "Rex" in reply and "Big Tony" in reply


def test_at_most_eight_names_a_player():
    w = make_world()
    p = w.player(A)
    for i in range(12):
        ask(w, p, f"answer to Name{i}")
    assert len(names_of(w, p)) == 8
    assert "8" in last_to(w, A)
    ask(w, p, "stop answering to Name0")
    ask(w, p, "answer to Extra")
    assert len(names_of(w, p)) == 8 and "extra" in names_of(w, p)


@pytest.mark.parametrize("bad", ["add", "remove", "couch", "come", "here", "undo", "two", "go away", "come here", "the lamp"])
def test_names_that_collide_with_the_parser_are_refused_politely_and_nothing_changes(bad):
    w = make_world()
    p = w.player(A)
    ask(w, p, f"answer to {bad}")
    assert names_of(w, p) == ["skippy"]
    reply = last_to(w, A)
    assert reply and ("cannot" in reply.lower())
    say(w, p, "add a lamp")                            # ordinary talk is still not a request
    assert lamps(w, A) == 0


def test_a_player_may_still_add_a_name_equal_to_the_word_of_a_command_inside_a_longer_phrase():
    w = make_world()
    p = w.player(A)
    ask(w, p, "answer to couch potato")
    assert names_of(w, p) == ["skippy", "couch potato"]


def test_a_real_library_noun_is_refused():
    w = make_world()
    p = w.player(A)
    ask(w, p, "answer to chair")
    assert names_of(w, p) == ["skippy"]
    ask(w, p, "answer to chairs")
    assert names_of(w, p) == ["skippy"]


def test_typos_and_junk_never_corrupt_the_state():
    w = make_world()
    p = w.player(A)
    ask(w, p, "answer to")
    ask(w, p, "answer to    ")
    ask(w, p, "answer to " + "x" * 40)
    ask(w, p, "answer to a b c d")
    ask(w, p, "answer to 中文")
    ask(w, p, "answer to my question")
    ask(w, p, "answer to rex<script>")
    ask(w, p, "stop answering to")
    assert names_of(w, p) == ["skippy"]
    ask(w, p, "anwser to Rex")                         # one slip in the command word still works
    assert names_of(w, p) == ["skippy", "rex"]
    ask(w, p, "answer to Rex")                         # a repeat is not a second copy
    ask(w, p, "answer to REX!")
    ask(w, p, "answer to  rex ")
    assert names_of(w, p) == ["skippy", "rex"]


def test_a_flood_of_commands_is_rate_limited_and_never_floods_the_disk():
    w = make_world()
    p = w.player(A)
    before = w.rt.eval("function() return (MOCK.write_count or {})['mentat/wake_words.json'] or 0 end")()
    w.rt.eval("function(p) for i = 1, 200 do Holodeck.Handle(p, 'answer to name' .. i) end end")(p)
    after = w.rt.eval("function() return (MOCK.write_count or {})['mentat/wake_words.json'] or 0 end")()
    assert after - before <= 1 and len(names_of(w, p)) <= 8


def test_an_ordinary_request_is_not_taken_for_a_name_command():
    w = make_world()
    p = w.player(A)
    ask(w, p, "add three couches")
    ask(w, p, "remove the couch")
    ask(w, p, "answer to the left of the table")
    assert names_of(w, p) == ["skippy"]


# ------------------------------------------------------------------------------------------------ the console commands
def con(w, p, cmd, *args):
    w.rt.eval("function(p, cmd, args) MOCK.cmds[cmd](p, cmd, args, table.concat(args, ' ')) end")(p, cmd, w.rt.table_from(list(args)))
    w.advance(1)


def test_the_console_commands():
    w = make_world()
    p = w.player(A)
    con(w, p, "skippy_wake_add", "Rex")
    con(w, p, "skippy_wake_add", "Big", "Tony")
    assert names_of(w, p) == ["skippy", "rex", "big tony"]
    con(w, p, "skippy_wake_remove", "Rex")
    assert names_of(w, p) == ["skippy", "big tony"]
    con(w, p, "skippy_wake_list")
    assert "Big Tony" in last_to(w, A)
    con(w, p, "skippy_wake_only", "Zed")
    assert names_of(w, p) == ["zed"]
    con(w, p, "skippy_wake_reset")
    assert names_of(w, p) == ["skippy"]


def test_the_server_console_has_no_names_and_says_so():
    w = make_world()
    w.rt.eval("function() MOCK.cmds['skippy_wake_add'](nil, 'skippy_wake_add', {'Rex'}, 'Rex') end")()
    assert w.rt.eval("function() return T.printed('belong to a player') end")() is True


# ------------------------------------------------------------------------------------------------ privacy
def test_one_players_names_are_never_matched_against_another_players_lines():
    w = make_world()
    a, b = w.player(A), w.player(B)
    ask(w, a, "answer to Asshole")
    say(w, b, "Asshole, add a lamp")
    assert lamps(w, B) == 0
    say(w, a, "Asshole, add a lamp")
    assert lamps(w, A) == 1
    assert names_of(w, b) == ["skippy"]
    say(w, b, "Skippy, add a lamp")
    assert lamps(w, B) == 1


def test_what_skippy_says_about_a_private_name_goes_to_its_owner_alone():
    w = make_world()
    a, b = w.player(A, x=0), w.player(B, x=10)          # B stands right beside A
    c = w.player(C, x=5000)
    say(w, a, "Skippy, answer to Asshole")
    ask(w, a, "what do you answer to")
    ask(w, a, "stop answering to Asshole")
    ask(w, a, "answer to Asshole")
    ask(w, a, "Asshole, add a lamp")
    for m in says(w):
        text = " ".join(x for x in writes(m) if isinstance(x, str))
        if "asshole" in text.lower():
            assert recipients(m) == [A], text
            assert writes(m)[2] == 0, "must be a private Tell (no asker id), not a Say heard near by"


def test_the_chat_line_of_a_name_command_is_hidden_from_the_other_players():
    """Typed in public chat, 'answer to Asshole' would show the name to everybody: the hook hides the line."""
    w = make_world()
    p = w.player(A)
    assert say(w, p, "Skippy, answer to Asshole") == ""
    assert say(w, p, "Skippy, add a lamp") is None      # ordinary requests keep their line
    assert say(w, p, "hello everyone") is None


def test_a_name_command_is_not_spoken_or_logged_for_others():
    w = make_world(holodeck_voice_out="1", holodeck_bridge="http://127.0.0.1:9")
    p = w.player(A)
    say(w, p, "Skippy, answer to Asshole")
    posts = [h for h in w.rt.eval("function() return MOCK.https end")().values()]
    for h in posts:
        assert "asshole" not in str(h.body).lower() if hasattr(h, "body") and h.body else True


# ------------------------------------------------------------------------------------------------ persistence
def test_the_lists_survive_a_restart():
    w = make_world()
    a, b = w.player(A), w.player(B)
    ask(w, a, "answer to Asshole")
    ask(w, a, "answer to Big Tony")
    ask(w, b, "only answer to Rex")
    raw = stored(w)
    assert raw and json.loads(raw)["players"][A] == ["skippy", "asshole", "big tony"]
    assert w.rt.eval("function() return MOCK.files['mentat/wake_words.tmp'] end")() is None      # the temp file is gone
    w2 = make_world(files={FILE: raw})
    a2, b2 = w2.player(A), w2.player(B)
    assert names_of(w2, a2) == ["skippy", "asshole", "big tony"] and names_of(w2, b2) == ["rex"]
    say(w2, a2, "Big Tony, add a lamp")
    assert lamps(w2, A) == 1


@pytest.mark.parametrize("garbage", [
    "{{{ not json", "[]", "42", '"text"', '{"players": 5}', '{"players": {"123": "nope"}}', "\x00\x01\x02", "null",
    '{"players": {"abc": ["rex"]}}',                                       # not a SteamID64
    '{"players": {"%s": []}}' % A, '{"players": {"%s": [1, 2, {}]}}' % A, '{"players": {"%s": ["<bad>", "x"]}}' % A,
])
def test_a_corrupt_file_means_the_default_never_a_crash(garbage):
    w = make_world(files={FILE: garbage})
    p = w.player(A)
    assert names_of(w, p) == ["skippy"]
    say(w, p, "Skippy, add a lamp")
    assert lamps(w, A) == 1


def test_an_oversized_file_is_ignored():
    w = make_world(files={FILE: json.dumps({"players": {A: ["rex"]}}) + " " * (3 * 1024 * 1024)})
    assert names_of(w, w.player(A)) == ["skippy"]


def test_a_half_good_file_keeps_the_good_part_within_the_caps():
    good = {"players": {A: ["rex", "REX", "Big  Tony", "a", "<x>"] + [f"name{i}" for i in range(20)], B: ["zed"], "oops": ["x"]}}
    w = make_world(files={FILE: json.dumps(good)})
    a, b = w.player(A), w.player(B)
    assert names_of(w, a)[:2] == ["rex", "big tony"] and len(names_of(w, a)) <= 8
    assert names_of(w, b) == ["zed"]


def test_the_stored_file_is_small_and_valid_json():
    w = make_world()
    p = w.player(A)
    for n in ("Rex", "Zed", "Bob"):
        ask(w, p, f"answer to {n}")
    raw = stored(w)
    assert len(raw) < 500 and json.loads(raw)["version"] == 1


# ------------------------------------------------------------------------------------------------ the owner's client
def wake_msgs(w):
    return list(w.rt.eval("function() return T.nets('holodeck_wake_names') end")().values())


def test_the_list_goes_to_its_owner_alone_on_every_change_and_on_request():
    w = make_world()
    a, b = w.player(A), w.player(B)
    ask(w, a, "answer to Asshole")
    msgs = wake_msgs(w)
    assert msgs and all(recipients(m) == [A] for m in msgs)
    assert writes(msgs[-1]) == [2, "skippy", "asshole"]
    n = len(msgs)
    ask(w, a, "stop answering to Skippy")
    assert len(wake_msgs(w)) == n + 1 and writes(wake_msgs(w)[-1]) == [1, "asshole"]
    w.rt.eval("function(p) MOCK.net_from_client(p, 'holodeck_wake_sync', {}) end")(b)
    mine = [m for m in wake_msgs(w) if recipients(m) == [B]]
    assert mine and writes(mine[-1]) == [1, "skippy"]
    k = len(wake_msgs(w))
    w.rt.eval("function(p) MOCK.net_from_client(p, 'holodeck_wake_sync', {}) end")(b)    # asked again at once: not answered again
    assert len(wake_msgs(w)) == k
    w.advance(3)
    w.rt.eval("function(p) MOCK.net_from_client(p, 'holodeck_wake_sync', {}) end")(b)
    assert len(wake_msgs(w)) == k + 1


def test_a_player_who_joins_is_sent_their_stored_list():
    w = make_world(files={FILE: json.dumps({"players": {A: ["rex"]}})})
    p = w.player(A)
    w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
    w.advance(10)
    msgs = [m for m in wake_msgs(w) if recipients(m) == [A]]
    assert msgs and writes(msgs[-1]) == [1, "rex"]


def test_the_client_writes_the_list_where_the_microphone_bridge_reads_it():
    w = World(server=False, load=())
    w.lua("function() MOCK.me = { IsValid = function() return true end } end")()
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/skippy_client.lua")')
    w.rt.eval("""function() MOCK.deliver_seq('holodeck_wake_names', { {'u', 2, 4}, {'s', 'skippy'}, {'s', 'asshole'} }) end""")()
    raw = w.rt.eval("function() return MOCK.files['mentat/wake_names.json'] end")()
    assert json.loads(raw)["names"] == ["skippy", "asshole"]
    assert PW.parse_names(json.loads(raw)) == ["skippy", "asshole"]


def test_the_client_asks_the_server_for_its_list_a_few_times_then_stops():
    w = World(server=False, load=())
    w.lua("function() MOCK.me = { IsValid = function() return true end } end")()
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/skippy_client.lua")')
    w.advance(30)
    assert not list(w.rt.eval("function() return T.nets('holodeck_wake_sync') end")().values()), "nothing is asked before the game has loaded"
    w.rt.execute("hook.Run('InitPostEntity')")
    w.advance(30)
    asked = list(w.rt.eval("function() return T.nets('holodeck_wake_sync') end")().values())
    assert 1 <= len(asked) <= 5
    w.rt.eval("""function() MOCK.deliver_seq('holodeck_wake_names', { {'u', 1, 4}, {'s', 'skippy'} }) end""")()
    k = len(asked)
    w.advance(30)
    assert len(list(w.rt.eval("function() return T.nets('holodeck_wake_sync') end")().values())) == k


# ------------------------------------------------------------------------------------------------ the avatar's phrases
def test_the_avatar_comes_and_goes_on_the_players_own_names_and_answers_privately():
    w = make_world(load_avatar=True)
    p = w.player(A, x=0, y=0)
    other = w.player(B, x=20, y=0)
    ask(w, p, "answer to Asshole")
    say(w, p, "Asshole, come here")                    # typed chat
    assert w.rt.eval("function() return Mentat.Avatar.IsActive() end")() is True and w.count() == 0
    mine = [m for m in says(w) if "Here I am, Asshole" in " ".join(x for x in writes(m) if isinstance(x, str))]
    assert mine and all(recipients(m) == [A] and writes(m)[2] == 0 for m in mine)
    w.advance(2)
    say(w, p, "Asshole, go away")
    assert w.rt.eval("function() return Mentat.Avatar.IsActive() end")() is False
    w.advance(2)
    w.rt.eval("function(p) Holodeck.Handle(p, 'asshole come here'); MOCK.advance(2) end")(p)       # console: the name is stripped by the wrapper
    assert w.rt.eval("function() return Mentat.Avatar.IsActive() end")() is True
    for m in says(w):                                   # B never saw the name
        if B in recipients(m):
            assert "asshole" not in " ".join(x for x in writes(m) if isinstance(x, str)).lower()


def test_another_players_name_does_not_call_my_avatar():
    w = make_world(load_avatar=True)
    a, b = w.player(A), w.player(B)
    ask(w, a, "answer to Asshole")
    say(w, b, "Asshole, come here")
    assert w.rt.eval("function() return Mentat.Avatar.IsActive() end")() is False


# ------------------------------------------------------------------------------------------------ the microphone bridge
import skippy_ears as E  # noqa: E402


def write_names(d, obj):
    (d / "wake_names.json").write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")


def test_the_bridge_defaults_to_skippy_without_a_names_file(tmp_path):
    assert E.current_names(tmp_path) == ["skippy"]
    assert E.strip_wake("Skippy, add two lamps", E.current_names(tmp_path)) == (True, "add two lamps")
    assert E.strip_wake("Asshole add two lamps", E.current_names(tmp_path))[0] is False


def test_the_bridge_reads_the_players_names_and_rereads_when_the_file_changes(tmp_path):
    write_names(tmp_path, {"version": 1, "names": ["skippy", "asshole"]})
    assert E.current_names(tmp_path) == ["skippy", "asshole"]
    assert E.strip_wake("Asshole, add two lamps", E.current_names(tmp_path)) == (True, "add two lamps")
    write_names(tmp_path, {"version": 1, "names": ["rex"], "ts": 1234567})
    assert E.current_names(tmp_path) == ["rex"]
    assert E.strip_wake("Skippy, add two lamps", E.current_names(tmp_path))[0] is False
    assert E.strip_wake("Rex add two lamps", E.current_names(tmp_path)) == (True, "add two lamps")
    (tmp_path / "wake_names.json").unlink()
    assert E.current_names(tmp_path) == ["skippy"]


@pytest.mark.parametrize("junk", ["{{{", "[]", "null", '{"names": "rex"}', '{"names": []}', '{"names": [1, null]}', '{"names": ["<x>", ""]}',
                                  "\x00\x01", "x" * 10000])
def test_a_missing_or_corrupt_names_file_falls_back_to_skippy(tmp_path, junk):
    write_names(tmp_path, junk)
    assert E.current_names(tmp_path) == ["skippy"]


def test_the_bridge_caps_what_it_trusts_from_the_file(tmp_path):
    write_names(tmp_path, {"names": [f"name{i}" for i in range(50)] + ["rex"]})
    names = E.current_names(tmp_path)
    assert len(names) == 8 and "rex" not in names


def test_a_sentence_with_a_custom_name_is_passed_on_exactly_like_skippy(tmp_path, monkeypatch):
    write_names(tmp_path, {"names": ["skippy", "asshole"]})
    for spoken in ("Asshole, add two lamps", "Skippy, add two lamps", "ass hole add two lamps"):
        monkeypatch.setattr(E, "transcribe", lambda wav, api, timeout=60, s=spoken: s)
        assert E.handle_speech(b"x", "http://none", tmp_path, False, log=lambda *_: None) == "add two lamps"
        got = json.loads((tmp_path / "voice_in.json").read_text(encoding="utf-8"))
        assert got["text"] == "add two lamps" and got["direct"] is False
        (tmp_path / "voice_in.json").unlink()


def test_talk_to_other_players_with_the_name_later_in_the_sentence_is_dropped(tmp_path, monkeypatch):
    write_names(tmp_path, {"names": ["skippy", "asshole"]})
    for spoken in ("did you hear what the asshole said", "tell Skippy to add a lamp", "what did you have for lunch"):
        monkeypatch.setattr(E, "transcribe", lambda wav, api, timeout=60, s=spoken: s)
        assert E.handle_speech(b"x", "http://none", tmp_path, False, log=lambda *_: None) is None
    assert not (tmp_path / "voice_in.json").exists()


def test_the_push_to_talk_key_is_unchanged_by_the_names(tmp_path, monkeypatch):
    write_names(tmp_path, {"names": ["rex"]})
    monkeypatch.setattr(E, "transcribe", lambda wav, api, timeout=60: "add two lamps")
    assert E.handle_speech(b"x", "http://none", tmp_path, True, log=lambda *_: None) == "add two lamps"
    monkeypatch.setattr(E, "transcribe", lambda wav, api, timeout=60: "Rex, add two lamps")
    assert E.handle_speech(b"x", "http://none", tmp_path, True, log=lambda *_: None) == "add two lamps"


def test_the_bridges_skippy_matching_is_unchanged():
    for text, woke, rest in [("Skippy, add a couch", True, "add a couch"), ("hey skippy add a couch", True, "add a couch"),
                             ("skipping rope", False, "skipping rope"), ("add a couch", False, "add a couch")]:
        assert E.strip_wake(text) == (woke, rest)
