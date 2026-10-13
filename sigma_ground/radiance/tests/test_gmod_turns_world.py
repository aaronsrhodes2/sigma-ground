"""Skippy's solid turns in the mock world: the game half (avatar/turns.lua + holodeck/turnlogic.lua + the seams in holodeck.lua and avatar.lua), offline.

The REAL addon files run under LuaJIT (lupa) against stage/gmod/holodeck/gmod_mock.lua, with the real parser, the real wake matcher, the real avatar and
the real director. The bridge is never contacted: the tests read the /turn and /quip requests the game sends and answer them by hand. What this proves:
who a line is for (wake names, his conversation window, gaze and distance, chatter between humans, a line addressed to another player, push-to-talk, the
console), what is sent to the model route (and that it carries no SteamID in the text), what the game does with a reply (shows it; runs a validated
action through the parser; ignores a malformed one; never goes silent), the captains' file, the sign-off line, and the scores of the labelled corpus.
What it cannot prove (the live model, the engine's real gaze numbers, how it sounds) is in stage/gmod/holodeck/README.md "Solid turns".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HOLO = HERE.parents[2] / "stage" / "gmod" / "holodeck"
sys.path.insert(0, str(HOLO))

from test_gmod_avatar import L, PRELUDE, bot, run  # noqa: E402
from test_gmod_glue import World  # noqa: E402

A, B, C = "76561190000000001", "76561190000000002", "76561190000000003"

BASE = {"mentat_skippy_director": "0", "holodeck_bridge": "http://127.0.0.1:9", "holodeck_agent": "1", "holodeck_persona": "0", "holodeck_voice_out": "0",
        "holodeck_bridge_per_min": "600", "holodeck_voice_chars_per_min": "100000", "mentat_avatar_ack_voice": "0", "holodeck_cooldown": "0"}
DIRECTOR = {"mentat_skippy_director": "1", "mentat_skippy_chat": "chatty", "mentat_skippy_quiet_hours": "", "mentat_skippy_new_lines_per_day": "20"}


def tworld(cv=None, director=False, summon=True, names=("Alice", "Bob")):
    cvars = {**BASE, **(DIRECTOR if director else {}), **(cv or {})}
    w = World(cvars=cvars, load=())
    w.rt.execute(PRELUDE)
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/holodeck.lua")')
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/avatar.lua")')
    w.rt.execute("Mentat.Director.Rand = function() return 0.0 end")
    a = w.player(A, x=0, y=0, yaw=0, name=names[0])
    b = w.player(B, x=300, y=0, yaw=180, name=names[1])
    if summon:
        w.ask(a, "come here")
        run(w, 3)
        w.advance(25)                          # (his "Here I am." opened the conversation window: let it close, the tests open it on purpose)
    return w, a, b


def chat(w, p, text, wait=0.1):
    r = w.rt.eval("function(p, t) return hook.Run('PlayerSay', p, t) end")(p, text)
    w.advance(wait)
    return r[0] if isinstance(r, tuple) else r


def voice(w, p, text, wait=0.1):
    w.rt.eval("function(p, t) MOCK.rd = { string = t }; MOCK.receivers.holodeck_voice(#t, p) end")(p, text)
    w.advance(wait)


def posts(w, path="/turn"):
    return list(w.rt.eval("function(p) return T.posts(p) end")(path).values())


def body(w, i):
    return json.loads(w.rt.eval("function(i) return MOCK.https[i].body end")(i))


def turn_bodies(w):
    return [body(w, i) for i in posts(w)]


def answer(w, i, **obj):
    w.post_json(i, {"ok": True, **obj})


def says(w):
    out = []
    for m in w.rt.eval("function() return T.nets('holodeck_say') end")().values():
        out.append(m.writes[1])
    return out


def count(w):
    return w.count()


def face_bot(w, p):
    w.rt.eval("function(p) local b = Mentat.Avatar.Get(); local pp = p:GetPos(); p.yaw = math.deg(math.atan2(b:GetPos().y - pp.y, b:GetPos().x - pp.x)) end")(p)


def look(w, p, yaw):
    w.rt.eval("function(p, y) p.yaw = y end")(p, yaw)


def move(w, p, x, y):
    w.rt.eval("function(p, x, y) p.pos = Vector(x, y, 0) end")(p, x, y)


def skippy_says_to(w, p, text="Done, monkey."):
    w.rt.eval("function(p, t) Holodeck.Say(p, t, nil, { speak = false }) end")(p, text)


def log(w):
    t = w.rt.eval("function() local o = {}; for i, e in ipairs(Mentat.Turns.ring) do o[i] = { e.addressed, e.final, e.src, e.why } end; return o end")()
    return [[e[k] for k in (1, 2, 3, 4)] for e in t.values()]


def decide(w, p, text, src="chat"):
    d = w.rt.eval("function(p, t, s) return Mentat.Turns.Decide(p, t, s) end")(p, text, src)
    return {"addressed": d.addressed, "final": d.final, "route": d.route, "score": d.score, "woke": d.woke}


# ============================================================================================================ the old flow is untouched
def test_a_named_command_still_goes_to_the_parser_and_asks_no_model():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, add a couch")
    assert count(w) == 1 and posts(w) == []


def test_everything_is_as_before_with_the_turns_switched_off():
    w, a, b = tworld({"mentat_skippy_turns": "0"})
    chat(w, a, "Skippy, what can you do")
    assert posts(w) == []
    skippy_says_to(w, a)
    w.advance(3)
    chat(w, a, "and another one")
    assert count(w) == 0 and posts(w) == []


def test_without_the_agent_there_is_no_model_route():
    w, a, b = tworld({"holodeck_agent": "0"}, summon=False)
    chat(w, a, "Skippy, what can you do")
    assert posts(w) == []                      # the owner said no model: the old flow answers (a private note)


# ============================================================================================================ the model route from the game side
def test_a_question_goes_to_the_turn_route_with_nothing_more_than_the_words_and_a_few_facts():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, what can you do")
    bodies = turn_bodies(w)
    assert len(bodies) == 1
    t = bodies[0]
    assert t["kind"] == "talk" and t["text"] == "what can you do" and t["roast"] == "gentle"
    assert all(isinstance(f, str) and len(f) < 100 for f in t["facts"])
    assert "Alice" in t["strip"] and "Bob" in t["strip"], "the local bridge is told which names to remove before anything leaves the machine"
    assert A not in json.dumps({k: v for k, v in t.items() if k not in ("who",)}), "the SteamID64 is only the rate-limit key"
    assert t["frontier_cap"] == 300 and t["timeout"] == 8


def test_a_valid_reply_is_shown_and_opens_the_conversation_window():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, what can you do")
    (i,) = posts(w)
    answer(w, i, say="I build things, monkey. Say add a couch.", mood="helpful", source="frontier", tries=1, action=None)
    assert "I build things, monkey. Say add a couch." in says(w)
    assert L(w, "return CurTime() - Mentat.Turns.spoke['" + A + "']") < 1


def test_a_valid_action_runs_through_the_parser_and_the_holodeck_reports_what_happened():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, could we use a few couches")
    (i,) = posts(w)
    answer(w, i, say="Three couches, monkey.", mood="boast", source="frontier", tries=1, action={"name": "add", "kind": "couch", "count": 3})
    w.advance(1.5)
    assert count(w) == 3
    assert "Three couches, monkey." not in says(w), "the holodeck's own answer reports what was really built; the model's sentence is not added on top"


@pytest.mark.parametrize("action", [
    {"name": "clear"}, {"name": "add", "kind": "couch", "count": 99}, {"name": "add", "kind": "couch; clear", "count": 1}, {"name": "add", "kind": "<script>", "count": 1},
    {"name": "add", "kind": "couch", "count": 0}, {"name": "add", "kind": "couch", "count": 2.5}, {"name": "add", "kind": 7, "count": 1}, "add couch", 5, [], {"name": "remove", "kind": "couch", "count": 1},
    {"name": "rcon", "cmd": "quit"}, {"name": "add", "kind": "couch", "count": "3"}, {"name": "add", "kind": "a" * 80, "count": 1}])
def test_a_malformed_or_forbidden_action_does_nothing_in_the_game(action):
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, could we use a few couches")
    (i,) = posts(w)
    answer(w, i, say="Sure, monkey, anything for you.", mood="helpful", source="frontier", tries=1, action=action)
    w.advance(1.5)
    assert count(w) == 0, action
    assert "Sure, monkey, anything for you." in says(w), "the sentence is still said: never silence"


def test_a_failed_call_still_gets_a_line():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, what can you do")
    (i,) = posts(w)
    w.rt.eval("function(i) MOCK.respond(i, 'fail', 'timeout') end")(i)
    assert any(s.startswith("I heard you, monkey") for s in says(w))


def test_a_reply_without_a_sentence_still_gets_a_line():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, what can you do")
    (i,) = posts(w)
    answer(w, i, say="", mood="helpful", action=None)
    assert any(s.startswith("I heard you, monkey") for s in says(w))


def test_one_turn_at_a_time_per_player():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, what can you do")
    chat(w, a, "Skippy, what else can you do")
    assert len(posts(w)) == 1


def test_a_bare_name_is_a_call_and_gets_an_answer():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy")
    bodies = turn_bodies(w)
    assert len(bodies) == 1 and bodies[0]["text"] == ""


def test_the_frontier_cap_and_timeout_are_the_convars():
    w, a, b = tworld({"mentat_skippy_frontier_per_day": "7", "mentat_skippy_turn_timeout": "5"}, summon=False)
    chat(w, a, "Skippy, what can you do")
    t = turn_bodies(w)[0]
    assert t["frontier_cap"] == 7 and t["timeout"] == 5


def test_the_gravity_fact_is_the_worlds_own_readback_as_a_percent_of_earths_and_never_a_guess():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, what can you do")
    t = turn_bodies(w)[0]
    assert t["sit"] == "g=unknown" and not any("gravity" in f for f in t["facts"]), "no world module running: no gravity number is given"
    w2, a2, b2 = tworld(summon=False)
    w2.rt.execute("SetGlobalFloat('mentat_world_g_m_s2', 1.62)")           # the moon, as world.lua publishes it
    chat(w2, a2, "Skippy, what can you do")
    t2 = turn_bodies(w2)[0]
    assert t2["sit"] == "g=low" and "gravity here is 17 percent of Earth's" in t2["facts"]
    w3, a3, b3 = tworld(summon=False)
    w3.rt.execute("SetGlobalFloat('mentat_world_g_m_s2', 9.80665)")
    chat(w3, a3, "Skippy, what can you do")
    t3 = turn_bodies(w3)[0]
    assert t3["sit"] == "g=norm" and "gravity here is 100 percent of Earth's" in t3["facts"]


# ============================================================================================================ who a line is for
def test_chatter_between_two_humans_gets_no_answer():
    w, a, b = tworld()
    chat(w, b, "did you see that jump")
    w.advance(3)
    chat(w, a, "lol nice one")
    assert posts(w) == [] and count(w) == 0 and not any("nice one" in s for s in says(w))
    last = log(w)[-1]
    assert last[1] == "ignore" and last[0] in ("no", "maybe")


def test_an_order_with_no_name_and_no_conversation_is_not_his():
    w, a, b = tworld()
    chat(w, a, "add a couch")
    assert count(w) == 0 and posts(w) == []


def test_a_line_addressed_to_another_player_by_name_is_not_his_even_if_it_is_an_order():
    w, a, b = tworld()
    chat(w, a, "Bob, add a couch")
    chat(w, a, "hey Bob come here")
    chat(w, a, "guys come look at this")
    assert count(w) == 0 and posts(w) == []


def test_his_name_as_a_topic_is_not_an_address():
    w, a, b = tworld()
    chat(w, a, "Skippy is silly")
    chat(w, a, "Skippy keeps floating behind me")
    assert posts(w) == [] and count(w) == 0
    assert any(e[1] == "ignore" and e[0] == "no" for e in log(w))


def test_a_follow_up_needs_no_name_inside_the_window_and_none_after_it():
    w, a, b = tworld()
    face_bot(w, a)
    skippy_says_to(w, a, "Done, monkey.")
    w.advance(4)
    chat(w, a, "and why is that")
    assert len(posts(w)) == 1, "inside the window, looking at him: answered"
    (i,) = posts(w)
    answer(w, i, say="Because I said so, monkey.", mood="helpful", action=None, source="local", tries=1)
    w.advance(25)
    chat(w, a, "and why is that")
    chat(w, a, "add a couch")
    assert len(posts(w)) == 1 and count(w) == 0, "twenty-five seconds on, without his name: nothing"


def test_a_follow_up_order_inside_the_window_runs_through_the_parser():
    w, a, b = tworld()
    face_bot(w, a)
    w.ask(a, "add a lamp")
    skippy_says_to(w, a, "A lamp.")
    w.advance(3)
    chat(w, a, "and another one")
    w.advance(1.5)
    assert count(w) == 2 and posts(w) == []


def test_a_window_alone_is_not_enough_when_the_speaker_has_walked_off_and_looks_away():
    w, a, b = tworld()
    skippy_says_to(w, b, "Hello over there, monkey.")
    move(w, b, 2000, 0)
    look(w, b, 0)
    w.advance(4)
    chat(w, b, "okay and then")
    assert posts(w) == [], "maybe means silence"
    assert log(w)[-1][0] in ("maybe", "no") and log(w)[-1][1] == "ignore"


def test_the_window_plus_standing_by_him_or_looking_at_him_is_enough():
    w, a, b = tworld()
    skippy_says_to(w, b, "Hello over there, monkey.")
    bp = L(w, "local p = Mentat.Avatar.Get():GetPos(); return p.x, p.y")
    move(w, b, bp[0] + 120, bp[1] + 80)
    look(w, b, 0)
    w.advance(4)
    chat(w, b, "okay and then")
    assert len(posts(w)) == 1, "the window rule: near him, so yes"
    assert log(w)[-1][0] == "yes"


def test_two_humans_talking_keep_him_out_even_inside_the_window_until_they_stop():
    w, a, b = tworld()
    chat(w, b, "what do you think we should build")
    w.advance(1)
    chat(w, a, "I think we need something bigger")
    assert posts(w) == [] and log(w)[-1][1] == "ignore", "no window: chatter between humans"
    w2, a2, b2 = tworld()
    face_bot(w2, a2)
    skippy_says_to(w2, a2, "Done, monkey.")
    w2.advance(2)
    chat(w2, b2, "what do you think we should build")
    w2.advance(1)
    chat(w2, a2, "I think we need something bigger")
    assert posts(w2) == [] and log(w2)[-1][0] == "maybe", "he just spoke to a and a looks at him, but b is talking right there: a maybe is silence"
    w2.advance(12)                                     # b has been quiet for ten seconds; the window (20 s) is still open
    skippy_says_to(w2, a2, "Anything else, monkey?")
    w2.advance(1)
    chat(w2, a2, "I think we need something bigger")
    assert len(posts(w2)) == 1


def test_looking_straight_at_another_player_keeps_him_out_of_it():
    w, a, b = tworld()
    move(w, b, 300, 0)
    look(w, a, 0)                                # a looks along +x straight at b; the bot hovers behind a's right shoulder
    skippy_says_to(w, a, "Done, monkey.")
    w.advance(4)
    assert w.rt.eval("function(p) return Mentat.Turns.Context(p, 'chat').looking_at_human end")(a) is True
    chat(w, a, "add a lamp")
    w.advance(1.5)
    assert count(w) == 0 and log(w)[-1][1] == "ignore", "the window is open and he is close, but a is talking to b"
    look(w, a, 90)                               # now a looks at nobody
    chat(w, a, "add a lamp")
    w.advance(1.5)
    assert count(w) == 1


def test_push_to_talk_and_the_console_are_always_for_him():
    w, a, b = tworld(summon=False)
    voice(w, a, "add a couch")                                    # no wake word: the microphone bridge only sends such a sentence for the talk key
    w.advance(1.5)
    assert count(w) == 1
    w.ask(a, "add a lamp")
    assert count(w) == 2
    assert log(w)[-1][2] == "console"


def test_a_spoken_line_with_his_name_goes_the_same_way_as_a_typed_one():
    w, a, b = tworld(summon=False)
    voice(w, a, "Skippy what can you do")
    bodies = turn_bodies(w)
    assert len(bodies) == 1 and bodies[0]["text"] == "what can you do"
    assert log(w)[-1][2] == "voice"


def test_a_custom_wake_name_wakes_him_and_a_rude_one_too():
    w, a, b = tworld(summon=False)
    w.ask(a, "answer to Asshole")
    chat(w, a, "Asshole, what can you do")
    bodies = turn_bodies(w)
    assert len(bodies) == 1 and bodies[0]["text"] == "what can you do"
    assert "asshole" not in json.dumps(bodies[0]["facts"]).lower()
    assert any(n.lower() == "asshole" for n in bodies[0]["strip"]), "the name is handed to the LOCAL bridge so it is removed before anything leaves the machine"


def test_noise_through_the_talk_key_is_ignored():
    w, a, b = tworld(summon=False)
    voice(w, a, "uh")
    voice(w, a, "...")
    assert posts(w) == []


def test_every_decision_is_logged_without_the_words():
    w, a, b = tworld()
    chat(w, a, "a very private sentence about nothing")
    e = w.rt.eval("function() return Mentat.Turns.ring[#Mentat.Turns.ring] end")()
    assert set(e.keys()) >= {"t", "src", "n", "addressed", "score", "final", "why"}
    assert "private" not in json.dumps({k: str(v) for k, v in e.items()})


# ============================================================================================================ the tiebreak
def test_the_tiebreak_is_off_by_default_and_budgeted_by_the_bridge():
    w, a, b = tworld()
    d = {"score": 50, "route": "parser", "ctx": {"since_spoke": 5, "others_near": 1}}
    w.rt.eval("function(p, d) Mentat.Turns.Tiebreak(p, 'add a couch', d, 'chat') end")(a, w.rt.eval("function() return { score = 50, route = 'parser', ctx = { since_spoke = 5, others_near = 1 } } end")())
    assert posts(w) == []


def test_a_tiebreak_yes_lets_the_line_through_and_a_no_does_not():
    w, a, b = tworld({"mentat_skippy_tiebreak": "1"})
    mk = "function() return { score = 50, route = 'parser', final = 'ignore', addressed = 'maybe', ctx = { since_spoke = 5, others_near = 1, avatar = { dist = 500, gaze = 50 } } } end"
    w.rt.eval("function(p, d) Mentat.Turns.Tiebreak(p, 'add a couch', d, 'chat') end")(a, w.rt.eval(mk)())
    (i,) = posts(w)
    tb = body(w, i)
    assert tb["kind"] == "tiebreak" and tb["signals"]["window"] is True and tb["signals"]["others"] is True and tb["tiebreak_cap"] == 60
    answer(w, i, addressed="no", source="frontier")
    w.advance(1.5)
    assert count(w) == 0
    w.advance(7)
    w.rt.eval("function(p, d) Mentat.Turns.Tiebreak(p, 'add a couch', d, 'chat') end")(a, w.rt.eval(mk)())
    i2 = posts(w)[-1]
    answer(w, i2, addressed="yes", source="frontier")
    w.advance(1.5)
    assert count(w) == 1


# ============================================================================================================ seams
def test_the_chat_hook_tells_turns_about_lines_without_his_name_and_marks_the_others():
    w, a, b = tworld(summon=False)
    seen = []
    w.rt.execute("OVERHEARD = {}; hook.Add('HolodeckOverheard', 'T', function(ply, text) OVERHEARD[#OVERHEARD + 1] = text end)")
    chat(w, a, "hello everybody")
    chat(w, a, "Skippy, add a lamp")
    got = [v for v in w.rt.eval("function() return OVERHEARD end")().values()]
    assert got == ["hello everybody"]


def test_the_closing_remarks_reach_the_old_flow_marked_as_decided():
    w, a, b = tworld(summon=False)
    chat(w, a, "Skippy, add a lamp")
    w.advance(1.5)
    assert count(w) == 1
    n_before = L(w, "return Mentat.Turns.counts.decided")
    L(w, "Holodeck.Handle(..., 'add a lamp', { turn_ok = true })", a)
    w.advance(1.5)
    assert L(w, "return Mentat.Turns.counts.decided") == n_before, "a line already judged is not judged twice"


# ============================================================================================================ the captains' file
def lua_str(text):
    bs = chr(92)
    return '"' + "".join(c if (c.isprintable() and c not in '"' + bs) else (bs + "%03d" % ord(c) if ord(c) < 256 else "?") for c in text) + '"'


def captain_world(raw=None, **cv):
    w, a, b = tworld({**DIRECTOR, **cv}, summon=False)
    if raw is not None:
        w.rt.execute("MOCK.files['mentat/captain_sids.txt'] = " + lua_str(raw))
    return w, a, b


def roast(w, p):
    return L(w, "return Mentat.Director.RoastLevel(...)", p)


def test_the_captain_file_makes_those_ids_savage_and_nobody_else():
    w, a, b = captain_world(f"{B}\n")
    assert roast(w, b) == "savage" and roast(w, a) == "gentle"


def test_the_captain_file_tolerates_comments_crlf_blank_lines_and_junk():
    raw = f"# the captains\r\n\r\n{A}   # me\r\nnot an id\r\n12ab\r\n{'9' * 40}\r\n  {B}\r\n"
    w, a, b = captain_world(raw)
    assert roast(w, a) == "savage" and roast(w, b) == "savage"
    assert L(w, "local n = 0; for _ in pairs(Mentat.Director.CaptainSids()) do n = n + 1 end; return n") == 2


def test_a_missing_or_empty_captain_file_is_fine():
    w, a, b = captain_world(None)
    assert roast(w, a) == "gentle"
    w2, a2, b2 = captain_world("")
    assert roast(w2, a2) == "gentle"
    w3, a3, b3 = captain_world(chr(0) + chr(1) + chr(2) + " garbage")
    assert roast(w3, a3) == "gentle"


def test_the_captain_file_falls_back_to_its_backup_copy():
    w, a, b = captain_world(None)
    w.rt.execute("MOCK.files['mentat/captain_sids.bak.txt'] = " + lua_str(f"{A}" + chr(10)))
    assert roast(w, a) == "savage"


def test_the_captain_file_is_reread_but_not_every_line():
    w, a, b = captain_world(f"{B}\n")
    assert roast(w, a) == "gentle"
    w.rt.execute("MOCK.files['mentat/captain_sids.txt'] = " + lua_str(f"{A}" + chr(10)))
    assert roast(w, a) == "gentle", "cached for 30 s"
    w.advance(31)
    assert roast(w, a) == "savage"


def test_the_host_and_the_convar_still_make_a_captain():
    w, a, b = captain_world(None, mentat_skippy_captain_sids=B)
    assert roast(w, b) == "savage"
    w2, a2, b2 = captain_world(None)
    w2.rt.eval("function(p) p.host = true end")(a2)
    assert roast(w2, a2) == "savage"


def test_a_players_own_choice_beats_the_captain_file():
    w, a, b = captain_world(f"{A}\n")
    w.rt.execute("Mentat.Director.TryCommand(MOCK.players[1], 'go easy on me')")
    w.advance(1)
    L(w, "Mentat.Director.TryCommand(..., 'roast level gentle')", a)
    assert roast(w, a) == "gentle"


def test_the_captain_file_is_never_committed():
    ignore = (HERE.parents[2] / ".gitignore").read_text(encoding="utf-8")
    assert "captain_sids.txt" in ignore


# ============================================================================================================ the sign-off
def quips(w):
    return list(w.rt.eval("function() return T.posts('/quip') end")().values())


def dir_world(**cv):
    w, a, b = tworld({**DIRECTOR, **cv}, summon=False)
    w.ask(a, "come here")
    for i in quips(w):
        w.rt.eval("function(i) MOCK.respond(i, 'fail', 'no') end")(i)           # (the arrival line is not asked for here: the canned "Here I am." comes)
    run(w, 6)
    w.rt.eval("MOCK.reset_traffic()")
    return w, a, b


def active(w):
    return L(w, "return Mentat.Avatar.IsActive()")


def test_going_away_speaks_a_signoff_from_the_bank_and_leaves_after_it():
    w, a, b = dir_world()
    w.ask(a, "go away")
    (i,) = quips(w)
    q = body(w, i)
    assert q["situation"] == "dismiss" and q["roast"] == "gentle"
    assert active(w) is True, "he stays until the line is said"
    w.post_json(i, {"ok": True, "text": "Farewell, monkey, try not to miss me.", "mood": "musing", "source": "bank", "id": "so01", "cat": "signoff", "tags": [], "cached": False,
                    "new": False, "dur": 2.0, "voice": "elevenlabs", "budget": {"day": "2026-10-13", "used": 20, "limit": 20, "left": 0}})
    assert "Farewell, monkey, try not to miss me." in says(w), "text only when the budget is spent"
    assert active(w) is True
    w.advance(3.0)
    assert active(w) is False
    assert "Fine. Call me if you need me." not in says(w)


def test_a_signoff_that_cannot_be_made_falls_back_to_the_canned_line_and_he_still_leaves():
    w, a, b = dir_world()
    w.ask(a, "go away")
    (i,) = quips(w)
    w.rt.eval("function(i) MOCK.respond(i, 'fail', 'down') end")(i)
    assert "Fine. Call me if you need me." in says(w)
    assert active(w) is False


def test_a_signoff_that_never_comes_does_not_keep_him_for_ever():
    w, a, b = dir_world()
    w.ask(a, "go away")
    w.advance(10)
    assert active(w) is False and "Fine. Call me if you need me." in says(w)


def test_calling_him_again_during_the_signoff_cancels_the_leaving():
    w, a, b = dir_world()
    w.ask(a, "go away")
    assert active(w) is True
    w.ask(a, "come here")
    w.advance(11)
    assert active(w) is True


def test_with_the_director_off_the_signoff_is_the_canned_line_at_once():
    w, a, b = tworld(summon=True)
    w.ask(a, "go away")
    assert active(w) is False and "Fine. Call me if you need me." in says(w)


def test_a_private_wake_name_keeps_the_private_goodbye():
    w, a, b = dir_world()
    w.ask(a, "answer to Rex")
    w.rt.execute("MOCK.reset_traffic()")
    chat(w, a, "Rex, go away")
    w.advance(1)
    assert active(w) is False and quips(w) == [], "a name is private: no public line is asked for"


# ============================================================================================================ the scores
def test_the_water_words_do_not_read_like_as_lake():
    w, a, b = tworld(summon=False)
    parse = w.rt.eval("function(t) local W = dofile(MOCK.ROOT .. '/mentat/water/intent.lua'); local r = W.parse(t); return r and r.kind or false end")
    for t in ["add a couch like this one", "make it look like a house", "a lamp like that", "add a lace curtain", "add a pole", "i want to pour tea"]:
        assert parse(t) is False, t
    assert parse("add a lake") == "add" and parse("make a pond") == "add"


def test_the_corpus_scores_hold():
    import turn_eval as E
    T = E.Turns()
    turns = E.load_corpus()
    assert len(turns) >= 250
    res = E.score(turns, T.decide)
    m = res["metrics"]
    assert m["false_wake_rate"] <= 0.02, [r["turn"]["text"] for r in res["false_wake"]]
    assert m["missed_wake_rate"] <= 0.03, [r["turn"]["text"] for r in res["missed"]]
    assert m["route_accuracy"] >= 0.95
    assert m["addressee_precision"] >= 0.97 and m["addressee_recall"] >= 0.97
    assert m["maybe_answered"] == 0, "a maybe is never answered by the deterministic layers"


def test_the_corpus_is_original_labelled_and_wide():
    import turn_eval as E
    turns = E.load_corpus()
    cats = {t["cat"] for t in turns}
    assert len(cats) >= 20
    for need in ("wake_command", "wake_talk", "custom_wake", "mishear_wake", "push_to_talk", "console", "followup_gaze", "chatter_humans", "named_human", "topic_name",
                 "noise", "ambiguous", "ramble", "typo_command", "followup_expired"):
        assert need in cats, need
    ids = [t["id"] for t in turns]
    assert len(ids) == len(set(ids))
    assert {t["addressee"] for t in turns} == {"yes", "no", "maybe"} and {t["route"] for t in turns} == {"parser", "plan", "talk", "ignore"}
    assert sum(1 for t in turns if t["addressee"] == "yes") >= 100 and sum(1 for t in turns if t["addressee"] != "yes") >= 100
    assert any(not t["text"].isascii() for t in turns), "other scripts"
    assert any(t["text"].strip() == "" for t in turns), "empty lines"
    assert sum(1 for t in turns if len(t["text"].split()) > 25) >= 3, "rambles"
    assert all(set(t["ctx"]) >= {"names", "human_names", "avatar", "looking_at_human", "since_spoke", "chatter_s", "others_near"} for t in turns)


def test_the_stress_set_is_scored_apart_and_held_out_from_tuning():
    import turn_eval as E
    T = E.Turns()
    stress = [t for t in E.load_corpus() if t["cat"].startswith("stress_")]
    assert len(stress) >= 40
    res = E.score(stress, T.decide)
    assert res["metrics"]["false_wake_rate"] <= 0.03 and res["metrics"]["missed_wake_rate"] <= 0.03


# ============================================================================================================ the rules, one by one
@pytest.fixture(scope="module")
def rules():
    import turn_eval as E
    return E.Turns()


def run_turn(rules, text, src="chat", names=("skippy",), humans=("Bob",), avatar=None, since=None, chatter=None, near=0, look=False):
    return rules.decide({"text": text, "src": src, "ctx": {"names": list(names), "human_names": list(humans), "avatar": avatar, "looking_at_human": look,
                                                              "since_spoke": since, "chatter_s": chatter, "others_near": near}})


def test_scoring_weights_are_the_documented_ones(rules):
    W = rules.rt.eval("function() return Logic.WEIGHTS, Logic.THRESH end")()
    w, th = W
    assert (w.wake_start, w.topic, w.vocative, w.gaze_12, w.gaze_30, w.gaze_60, w.gaze_away, w.looking_human) == (100, -60, -70, 30, 20, 8, -15, -30)
    assert (w.near_300, w.near_700, w.far, w.window_8, w.window_20, w.chatter, w.short, w.long) == (15, 6, -10, 50, 40, -30, -15, -10)
    assert (th.yes, th.maybe) == (60, 35)
    assert rules.rt.eval("function() return Logic.WINDOW_S end")() == 20


def test_a_wake_name_at_the_start_is_yes_whatever_else_is_going_on(rules):
    r = run_turn(rules, "Skippy add a couch", avatar={"dist": 2000, "gaze": 150}, chatter=1, near=3, look=True)
    assert r["addressed"] == "yes" and r["final"] == "parser"


def test_console_and_push_to_talk_are_yes_with_no_other_signal(rules):
    assert run_turn(rules, "what is that", src="console")["addressed"] == "yes"
    assert run_turn(rules, "what is that", src="ptt", chatter=1, near=3)["addressed"] == "yes"


@pytest.mark.parametrize("gaze,dist,expect", [(5, 100, "maybe"), (25, 100, "maybe"), (40, 100, "no"), (90, 100, "no"), (150, 100, "no"), (5, 2000, "no"), (150, 2000, "no")])
def test_gaze_and_distance_alone_never_make_a_line_his_without_a_name_or_a_window(rules, gaze, dist, expect):
    r = run_turn(rules, "what is that hat made of", avatar={"gaze": gaze, "dist": dist})
    assert r["addressed"] == expect, r
    assert r["final"] == "ignore"


def test_the_best_possible_score_without_his_name_the_window_or_the_talk_key_is_below_yes(rules):
    r = run_turn(rules, "can you tell me what you think of your hat please?", avatar={"gaze": 0, "dist": 10})
    assert r["score"] < 60 and r["final"] == "ignore"


@pytest.mark.parametrize("since,expect_answered", [(1, True), (8, True), (12, True), (20, True), (21, False), (60, False)])
def test_the_window_is_twenty_seconds_and_needs_gaze_or_closeness_to_agree(rules, since, expect_answered):
    r = run_turn(rules, "and then what", avatar={"gaze": 10, "dist": 150}, since=since)
    assert (r["final"] != "ignore") == expect_answered, r


def test_a_maybe_inside_the_window_becomes_yes_only_when_gaze_or_distance_agree(rules):
    far_side = run_turn(rules, "okay and then", avatar={"gaze": 75, "dist": 900}, since=14)
    assert far_side["addressed"] == "maybe" and far_side["final"] == "ignore"
    near_side = run_turn(rules, "okay and then", avatar={"gaze": 75, "dist": 300}, since=14)
    assert near_side["addressed"] == "yes" and near_side["promoted"] is True
    glance = run_turn(rules, "okay and then", avatar={"gaze": 40, "dist": 900}, since=14)
    assert glance["addressed"] == "yes" and glance["promoted"] is True


def test_the_promotion_never_overrides_the_blockers(rules):
    for kw in ({"look": True}, {"chatter": 2, "near": 1}):
        r = run_turn(rules, "okay and then", avatar={"gaze": 40, "dist": 300}, since=14, **kw)
        assert r["promoted"] is False
    r = run_turn(rules, "Bob okay and then", avatar={"gaze": 10, "dist": 100}, since=3)
    assert r["final"] == "ignore", "addressed to Bob by name"


def test_no_window_no_name_looking_at_him_is_still_only_maybe(rules):
    r = run_turn(rules, "how tall are you", avatar={"gaze": 5, "dist": 100})
    assert r["addressed"] == "maybe" and r["final"] == "ignore"


def test_another_players_name_or_a_group_call_is_not_for_him(rules):
    for t in ("Bob add a couch", "hey bob, come here", "guys come here", "everyone look at this"):
        r = run_turn(rules, t, avatar={"gaze": 5, "dist": 100})
        assert r["final"] == "ignore", t


def test_a_nick_with_symbols_or_two_words_is_still_a_name(rules):
    assert run_turn(rules, "Big Mike add a lamp", humans=["Big Mike"])["final"] == "ignore"
    assert run_turn(rules, "mike add a lamp", humans=["Big Mike"], names=["skippy"])["final"] == "ignore"
    assert run_turn(rules, "xx_bob_xx add a lamp", humans=["xX_Bob_Xx"])["final"] == "ignore"


def test_a_name_that_is_a_command_word_is_not_a_vocative(rules):
    r = run_turn(rules, "add a lamp", humans=["Add"], src="console")
    assert r["final"] == "parser"


def test_the_topic_rule_and_its_exception_for_spoken_questions(rules):
    assert run_turn(rules, "Skippy is silly")["final"] == "ignore"
    assert run_turn(rules, "Skippy has a really big hat")["final"] == "ignore"
    assert run_turn(rules, "Skippy, is it silly")["addressed"] == "yes", "a comma is an address"
    assert run_turn(rules, "Skippy is the moon heavier or lighter than earth")["addressed"] == "yes"
    assert run_turn(rules, "Skippy are you there")["addressed"] == "yes"
    assert run_turn(rules, "Skippy can you add a lamp")["final"] == "parser"


def test_routes(rules):
    for text, route in [("add a couch", "parser"), ("undo", "parser"), ("make it cozy", "plan"), ("what can you do", "talk"), ("hello", "talk"), ("come here", "parser"),
                        ("shush", "parser"), ("roast me harder", "parser"), ("add a pond", "parser"), ("set gravity to the moon", "parser"), ("what is gravity", "parser"),
                        ("answer to Rex", "parser"), ("how do I clear the room", "talk"), ("what is a couch", "talk"), ("um", "ignore"), ("...", "ignore"),
                        ("can you add a lamp", "parser"), ("would you like to hear a joke", "talk")]:
        assert run_turn(rules, text, src="console")["final"] == route, text
    assert run_turn(rules, "thank you", src="ptt")["final"] == "ignore", "a silence hallucination"
    assert run_turn(rules, "thank you", src="chat", names=["skippy"], avatar={"gaze": 5, "dist": 100}, since=3)["final"] == "talk"


def test_a_wake_name_alone_is_a_call(rules):
    assert run_turn(rules, "Skippy")["final"] == "talk"
    assert run_turn(rules, "Skippy?", src="voice")["final"] == "talk"


def test_every_command_the_fallback_lines_suggest_is_a_real_parser_command(rules):
    import re
    import skippy_turns as ST
    cmds = set()
    for _name, _rx, _cheap, gentle, extra in ST._FB_RE:
        for line in list(gentle) + list(extra):
            for m in re.findall(r"'([^']+)'", line):
                cmds.add(m)
    assert {"add a couch", "undo", "help"} <= cmds
    for c in sorted(cmds):
        if c in ("help",):
            continue
        r = run_turn(rules, c, src="console")
        assert r["final"] in ("parser", "plan"), (c, r["final"])
    assert "pour water" not in ST.CAPABILITIES.lower()
