"""Skippy alive: the behaviour director, the animation, the notebook (avatar/director.lua, director_logic.lua, anim.lua, notes.lua, journal.lua), offline.

The REAL addon files run under LuaJIT (lupa) against stage/gmod/holodeck/gmod_mock.lua. The bridge is never contacted: the tests read the /quip
requests the game sends and answer them by hand. What this proves: when he may speak (chattiness, cooldowns, never over a human or himself, yielding
to a command, quiet hours, interruption), the exact words of the commands, the animation as a pure bounded state machine and how the client draws it,
the notebook (what is recorded, rotation, forget, opt-out), and above all what a model provider would be told (no SteamIDs, no names). What it cannot
prove (how the animation LOOKS, what the voice sounds like) is in stage/gmod/AVATAR.md "Skippy alive".
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_gmod_avatar import L, avatar_world, bot, events, run, where  # noqa: E402
from test_gmod_avatar_body import MANIFEST, client, draws, full_pipeline, synthetic_set  # noqa: E402
from test_gmod_glue import World, lua_table  # noqa: E402

A, B, C = "76561190000000001", "76561190000000002", "76561190000000003"
HU_PER_M = 1 / 0.01905


def pure():
    """A bare runtime with the pure modules loaded."""
    return World(server=True, load=())


def mod(w, name):
    return w.rt.eval(f"dofile(MOCK.ROOT .. '/mentat/avatar/{name}.lua')")


def rng(v):
    return lambda: v


# ============================================================================================================ the pure logic
def test_quiet_hours_wrap_midnight():
    w = pure()
    D = mod(w, "director_logic")
    q = D.parse_quiet_hours("23-7")
    assert (q.__getitem__("from"), q.to) == (23, 7)
    assert [D.in_quiet_hours(q, h) for h in (22, 23, 0, 6, 7, 12)] == [False, True, True, True, False, False]
    q2 = D.parse_quiet_hours("9-17")
    assert [D.in_quiet_hours(q2, h) for h in (8, 9, 16, 17)] == [False, True, True, False]
    assert D.parse_quiet_hours("") is None and D.parse_quiet_hours("nonsense") is None and D.parse_quiet_hours("5-5") is None


def decide(w, D, st, name, now, ctx=None, r=0.0):
    c = w.rt.table()
    for k, v in (ctx or {}).items():
        c[k] = v
    c["chat"] = c["chat"] or "chatty"
    out = D.decide(st, w.rt.table_from({"name": name}) if hasattr(w.rt, "table_from") else lua_table(w.rt, {"name": name}), now, c, w.rt.eval(f"function() return {r} end"))
    return out


def test_chattiness_sets_which_events_may_speak_at_all():
    w = pure()
    D = mod(w, "director_logic")
    st = D.new()
    assert decide(w, D, st, "splash", 1000, {"chat": "quiet"})["say"] is False, "quiet: only reactions that matter"
    assert decide(w, D, st, "splash", 1000, {"chat": "quiet"})["why"].startswith("chattiness")
    assert decide(w, D, st, "arrive", 1000, {"chat": "quiet"})["say"] is True, "an answer to a call is always spoken"
    assert decide(w, D, D.new(), "musing", 1000, {"chat": "normal"})["say"] is False, "normal: a pure whim stays silent"
    assert decide(w, D, D.new(), "musing", 1000, {"chat": "chatty"})["say"] is True
    assert decide(w, D, D.new(), "beam_ok", 1000, {"chat": "quiet"}, r=0.5)["say"] is False, "quiet: even a reaction is rare"


def test_cooldowns_pace_a_behaviour_and_the_gap_paces_all_of_them():
    w = pure()
    D = mod(w, "director_logic")
    st = D.new()
    d = decide(w, D, st, "splash", 1000)
    assert d["say"] is True
    D.commit(st, "splash", 1000)
    assert decide(w, D, st, "splash", 1050)["why"] in ("cooldown", "too soon after his last remark")
    assert decide(w, D, st, "water_jump", 1010)["why"] == "too soon after his last remark", "the gap holds for every kind of remark"
    assert decide(w, D, st, "water_jump", 1100)["say"] is True, "after the gap another kind may speak"
    D.commit(st, "water_jump", 1100)
    assert decide(w, D, st, "splash", 1110)["why"] in ("cooldown", "too soon after his last remark")
    assert decide(w, D, st, "splash", 1100 + 101)["say"] is True, "the cooldown (90 s) and the gap have passed"


def test_the_hourly_cap_and_the_chance():
    w = pure()
    D = mod(w, "director_logic")
    st = D.new()
    t = 1000
    spoke = 0
    for i in range(60):
        t += 40
        if decide(w, D, st, "musing", t, {"chat": "chatty"})["say"]:
            D.commit(st, "musing", t)
            spoke += 1
    assert spoke == D.CHAT.chatty.hour_cap == 50, "never more than the hourly cap in an hour"
    st2 = D.new()
    assert decide(w, D, st2, "beam_ok", 1000, {"chat": "normal"}, r=0.99)["why"].startswith("not this time"), "the dice can say no"
    assert decide(w, D, st2, "beam_ok", 1000, {"chat": "normal"}, r=0.0)["say"] is True
    st3 = D.new()
    assert decide(w, D, st3, "beam_ok", 1000, {"chat": "normal", "budget_left": 0}, r=0.30)["say"] is False, "with the day's new lines spent he is much quieter"


def test_he_never_talks_over_a_human_or_over_himself_and_a_reaction_waits_instead_of_being_lost():
    w = pure()
    D = mod(w, "director_logic")
    st = D.new()
    d = decide(w, D, st, "beam_fail", 1000, {"human_speaking": True})
    assert d["say"] is False and d["why"] == "a human is speaking" and d["defer"] is True, "a failure remark waits for a gap"
    d = decide(w, D, st, "splash", 1000, {"human_speaking": True})
    assert d["say"] is False and d["defer"] is None, "a mere comment is simply dropped"
    d = decide(w, D, st, "arrive", 1000, {"human_speaking": True})
    assert d["say"] is False and d["defer"] is True, "even the arrival line waits for the human to finish"
    D.line_started(st, 1000, 5)
    d = decide(w, D, st, "beam_fail", 1002)
    assert d["say"] is False and d["why"] == "he is still speaking" and d["defer"] is True
    assert decide(w, D, st, "beam_fail", 1005.6)["say"] is True, "after his line and the tail"


def test_he_is_cut_off_only_while_he_is_speaking():
    w = pure()
    D = mod(w, "director_logic")
    st = D.new()
    assert D.should_interrupt(st, 1000, True) is False, "nothing to interrupt"
    D.line_started(st, 1000, 6)
    assert D.should_interrupt(st, 1003, True) is True
    assert D.should_interrupt(st, 1003, False) is False
    D.line_cut(st, 1003)
    assert D.should_interrupt(st, 1004, True) is False


def test_he_yields_while_the_owner_is_mid_command_but_still_answers_the_command():
    w = pure()
    D = mod(w, "director_logic")
    d = decide(w, D, D.new(), "beam_ok", 1000, {"owner_command_age": 1.0})
    assert d["say"] is False and d["why"] == "the owner is mid-command" and d["defer"] is True
    assert decide(w, D, D.new(), "beam_ok", 1000, {"owner_command_age": 6.0})["say"] is True
    assert decide(w, D, D.new(), "recall", 1000, {"owner_command_age": 0.1})["say"] is True, "the answer to a command is not held back by the command"


def test_quiet_hours_silence_everything_but_answers():
    w = pure()
    D = mod(w, "director_logic")
    assert decide(w, D, D.new(), "musing", 1000, {"quiet_hours": True})["why"] == "quiet hours"
    assert decide(w, D, D.new(), "arrive", 1000, {"quiet_hours": True})["say"] is True


def test_switching_the_director_off_stops_every_decision():
    w = pure()
    D = mod(w, "director_logic")
    assert decide(w, D, D.new(), "arrive", 1000, {"disabled": True})["say"] is False


@pytest.mark.parametrize("text,want", [
    ("shush", {"cmd": "chat", "level": "quiet"}), ("Skippy be quiet please", {"cmd": "chat", "level": "quiet"}), ("hush!", {"cmd": "chat", "level": "quiet"}),
    ("be chatty", {"cmd": "chat", "level": "chatty"}), ("talk more", {"cmd": "chat", "level": "chatty"}), ("be less chatty", {"cmd": "chat", "level": "normal"}),
    ("roast me harder", {"cmd": "roast", "delta": 1}), ("go easy on me", {"cmd": "roast", "delta": -1}), ("be nice", {"cmd": "roast", "delta": -1}),
    ("roast me", {"cmd": "roast", "now": True}), ("savage mode", {"cmd": "roast", "delta": 1}), ("roast level gentle", {"cmd": "roast", "level": "gentle"}),
    ("what do you remember", {"cmd": "recall"}), ("What did I do today?", {"cmd": "recall"}), ("can you tell me what have I done", None),
    ("forget that", {"cmd": "forget", "scope": "that"}), ("forget today", {"cmd": "forget", "scope": "today"}), ("forget everything", {"cmd": "forget", "scope": "all"}),
    ("stop remembering", {"cmd": "journal", "on": False}), ("start remembering", {"cmd": "journal", "on": True}),
    # never commands: object requests and the wake-name commands that share words with them
    ("add a quiet room", None), ("add a couch", None), ("come here", None), ("forget the name Rex", None), ("make a harder floor", None),
    ("remove the couch", None), ("what do you answer to", None), ("roast beef sandwich", None), ("", None),
])
def test_his_own_commands_are_matched_exactly_and_object_requests_never_are(text, want):
    w = pure()
    D = mod(w, "director_logic")
    got = D.match_command(text)
    if want is None:
        assert got is None, text
        return
    assert got is not None, text
    for k, v in want.items():
        assert got[k] == v, (text, k)


def test_the_roast_ladder_has_three_rungs_and_stops_at_both_ends():
    w = pure()
    D = mod(w, "director_logic")
    assert [D.roast_step("gentle", 1), D.roast_step("standard", 1), D.roast_step("savage", 1), D.roast_step("gentle", -1), D.roast_step("savage", -1)] == \
        ["standard", "savage", "savage", "gentle", "standard"]


def test_an_idle_owner_is_noticed_once_per_idle_spell_and_walking_resets_it():
    w = pure()
    D = mod(w, "director_logic")
    st = D.new()
    pos = lua_table(w.rt, {"x": 0, "y": 0})
    assert D.owner_idle(st, 0, pos, None, "chatty") is False
    assert D.owner_idle(st, 59, pos, None, "chatty") is False
    assert D.owner_idle(st, 61, pos, None, "chatty") is True
    assert D.owner_idle(st, 200, pos, None, "chatty") is False, "once per spell"
    assert D.owner_idle(st, 210, lua_table(w.rt, {"x": 500, "y": 0}), None, "chatty") is False, "they walked"
    assert D.owner_idle(st, 300, lua_table(w.rt, {"x": 500, "y": 0}), None, "chatty") is True
    st2 = D.new()
    D.owner_idle(st2, 0, pos, None, "chatty")
    assert D.owner_idle(st2, 100, pos, 90, "chatty") is False, "a command 10 s ago restarts the idle clock"


def test_whims_are_mostly_musings_and_opera_and_bananas_are_rare():
    w = pure()
    D = mod(w, "director_logic")
    counts = {}
    for i in range(1000):
        k = D.pick_whim(i / 1000)
        counts[k] = counts.get(k, 0) + 1
    assert counts["musing"] >= 450 and counts["opera"] <= 80 and counts["banana"] <= 80 and sum(counts.values()) == 1000
    assert D.next_whim(100, "quiet", 0.5) > D.next_whim(100, "chatty", 0.5) > 100


def test_he_looks_at_what_you_just_placed_else_at_what_you_aim_at():
    w = pure()
    D = mod(w, "director_logic")
    placed = lua_table(w.rt, {"placed": {"pos": {"x": 1, "y": 2, "z": 3}, "t": 100}, "aim": {"x": 9, "y": 9, "z": 9}})
    pos, why = D.look_target(placed, 105)
    assert pos.x == 1 and why == "what you just placed"
    pos, why = D.look_target(placed, 130)
    assert pos.x == 9 and why == "what you are aiming at"
    assert D.look_target(lua_table(w.rt, {}), 0) is None


# ============================================================================================================ the animation (pure)
def test_every_cue_ends_at_zero_and_stays_inside_the_limits():
    w = pure()
    An = mod(w, "anim")
    lim = {k: v for k, v in An.LIMITS.items()}
    for name in ("hat_tip", "flourish", "conduct", "shrug", "emphasis", "sway", "bounce", "quizzical", "nod"):
        dur = An.DUR[name]
        before = An.cue_pose(name, -0.1)
        after = An.cue_pose(name, dur + 0.001)
        peak = 0
        for i in range(0, 201):
            p = An.cue_pose(name, dur * i / 200)
            for ch, v in p.items():
                assert abs(v) <= lim[ch] + 1e-9, (name, ch, v)
                peak = max(peak, abs(v) / lim[ch])
        for ch, v in list(before.items()) + list(after.items()):
            assert v == 0, (name, ch)
        edge = An.cue_pose(name, dur * 0.999)
        assert all(abs(v) / lim[ch] < 0.05 for ch, v in edge.items()), (name, "it comes back to rest smoothly")
        assert peak > 0.15, (name, "it really moves")


def test_the_layered_pose_never_leaves_the_limits_whatever_the_inputs():
    w, An = anim_env()
    lim = {k: v for k, v in An.LIMITS.items()}
    import random as _r
    rnd = _r.Random(5)
    st = An.new()
    for i in range(3000):
        t = i * 0.016
        p = step(An, st, now=t, dt=rnd.choice([0.0, 0.016, 0.05, 0.5]), cue=rnd.randint(0, 12), cue_seq=i // 40, mood=rnd.choice(["opera", "boast", "roast", "helpful", "musing", "exclaim", "bogus"]),
                 talking=rnd.random() < 0.7, level=rnd.uniform(-1, 3))
        for ch, v in p.items():
            assert abs(v) <= lim[ch] + 1e-9, (ch, v)
        assert 0.55 - 1e-9 <= st.tongue <= 1.0 + 1e-9
    # the worst case on purpose: every cue at full voice in the biggest mood (the layers add up, and the sum is clamped)
    for name, cid in An.CUES.items():
        st = An.new()
        for i in range(int((An.DUR[name] if name in An.DUR else 1) / 0.016) + 5):
            p = step(An, st, now=i * 0.016, cue=cid, cue_seq=1 if i else 0, mood="opera", talking=True, level=1.0)
            for ch, v in p.items():
                assert abs(v) <= lim[ch] + 1e-9, (name, ch, v)


def test_the_hat_tip_lifts_the_hat_and_tips_it_forward_and_comes_back():
    w = pure()
    An = mod(w, "anim")
    mid = An.cue_pose("hat_tip", 0.5)
    assert mid.hat_tip > 0.9 and mid.hat_p > 15
    assert An.cue_pose("hat_tip", 0.0).hat_tip == 0 and An.cue_pose("hat_tip", An.DUR.hat_tip - 0.01).hat_tip < 0.1


def test_a_conductors_hand_sweeps_both_ways_through_the_aria():
    w = pure()
    An = mod(w, "anim")
    sides = [An.cue_pose("conduct", An.DUR.conduct * i / 100).hand_side for i in range(101)]
    assert max(sides) > 0.6 and min(sides) < -0.6


def step(An, st, **kw):
    inp = {"now": 0.0, "dt": 0.016, "cue": 0, "cue_seq": 0, "mood": "musing", "talking": False, "level": 0}
    inp.update(kw)
    return An.step(st, lambda_table(An, inp))


def lambda_table(An, d):
    # (a lupa table made from the runtime the module lives in)
    return RT[0].table_from(d) if hasattr(RT[0], "table_from") else lua_table(RT[0], d)


RT = [None]


def anim_env():
    w = pure()
    RT[0] = w.rt
    return w, mod(w, "anim")


def test_a_new_cue_sequence_number_starts_the_gesture_and_a_new_one_replaces_it():
    w, An = anim_env()
    st = An.new()
    for i in range(10):
        step(An, st, now=i * 0.016)
    assert st.cue is None
    step(An, st, now=1.0, cue=An.CUES.hat_tip, cue_seq=1)
    assert st.cue == "hat_tip"
    p = step(An, st, now=1.5, cue=An.CUES.hat_tip, cue_seq=1)
    assert p.hat_tip > 0.9
    step(An, st, now=1.6, cue=An.CUES.flourish, cue_seq=2)
    assert st.cue == "flourish", "a flourish is never late: it replaces the gesture in progress"
    step(An, st, now=1.7, cue=An.CUES.flourish, cue_seq=2)
    step(An, st, now=1.6 + An.DUR.flourish + 0.1, cue=An.CUES.flourish, cue_seq=2)
    assert st.cue is None, "and it ends"


def test_the_voice_drives_the_head_chatter_the_jaw_and_the_tongue_glow():
    w, An = anim_env()
    quiet, loud = An.new(), An.new()
    pitches_q, pitches_l = [], []
    for i in range(200):
        t = i * 0.016
        pitches_q.append(step(An, quiet, now=t, talking=False, level=0).head_pitch)
        pitches_l.append(step(An, loud, now=t, talking=True, level=1.0, mood="exclaim").head_pitch)
    assert max(abs(x) for x in pitches_q) < 0.1, "no voice, no chatter"
    assert max(pitches_l) > 2.5, "a loud voice nods the head with the syllables"
    assert max(pitches_l) <= An.LIMITS.head_pitch + 1e-9
    assert quiet.tongue == pytest.approx(0.55, abs=1e-6) and loud.tongue > 0.95, "the red tongue brightens with the voice"
    assert max(pitches_l) - min(pitches_l) > 2, "and it chatters, not just tilts"


def test_the_mood_shapes_the_style_while_he_talks():
    w, An = anim_env()
    sway = {}
    for mood in ("opera", "boast", "roast", "helpful"):
        st = An.new()
        peak = 0
        for i in range(400):
            p = step(An, st, now=i * 0.016, talking=True, level=0.5, mood=mood)
            peak = max(peak, abs(p.tilt_r))
        sway[mood] = peak
    assert sway["opera"] > 2 * sway["helpful"], "an aria is a big sway, a helpful remark a small one"
    st = An.new()
    for i in range(200):
        p = step(An, st, now=i * 0.016, talking=True, level=0.0, mood="roast")
    assert p.tilt_p > 3, "a roast leans in"
    st = An.new()
    for i in range(200):
        p = step(An, st, now=i * 0.016, talking=True, level=0.0, mood="boast")
    assert p.tilt_p < -2, "a boast leans back, chest out"


def test_idle_breathing_keeps_him_alive_and_small():
    w, An = anim_env()
    st = An.new()
    seen = [step(An, st, now=i * 0.05).tilt_r for i in range(400)]
    assert 0.5 < max(seen) < 3 and min(seen) < -0.5


def test_the_animation_step_is_cheap():
    w, An = anim_env()
    st = An.new()
    inp = lambda_table(An, {"now": 0.0, "dt": 0.016, "cue": 3, "cue_seq": 1, "mood": "opera", "talking": True, "level": 0.7})
    n = 20000
    t0 = time.perf_counter()
    f = w.rt.eval("function(An, st, inp, n) for i = 1, n do inp.now = i * 0.016; An.step(st, inp) end end")
    f(An, st, inp, n)
    us = (time.perf_counter() - t0) / n * 1e6
    # the interpreter here is the slow one (jit.off, plus the python/lupa boundary only once): the real engine's LuaJIT is several times faster
    assert us < 60, f"{us:.1f} us per frame"
    print(f"anim.step: {us:.1f} us per call (interpreted)")


# ============================================================================================================ the notes (pure)
NOW = 1_760_400_000          # 2025-10-14 ~00:00 UTC; the tests use a fixed UTC clock
def utc(fmt, t):
    return time.strftime(fmt, time.gmtime(t))


def summarise(events, now=NOW + 20 * 3600, banned=()):
    w = pure()
    N = mod(w, "notes")
    fn = w.rt.eval("""function(N, events, now, banned, date_fn) return N.summarise(events, now, { banned = banned, date_fn = date_fn }) end""")
    df = w.rt.eval("function(f) return function(fmt, t) return f(fmt, t) end end")(utc)
    res = fn(N, lua_table(w.rt, events), now, lua_table(w.rt, list(banned)), df)
    return list(res.notes.values()), {k: v for k, v in res.facts.items()}


def ev(k, t_h, **kw):
    return {"t": NOW + int(t_h * 3600), "k": k, **kw}


def test_notes_are_a_handful_of_plain_facts():
    events = [ev("session_start", 14.0), ev("add", 14.5, name="couch", n=1), ev("add", 14.6, name="couch", n=1), ev("add", 15.0, name="couch", n=1),
              ev("add", 15.1, name="pool", n=1), ev("water", 15.1, cmd="pool", liquid="mercury"), ev("beam_fail", 15.5, name="kettle"), ev("beam_fail", 15.6, name="kettle"),
              ev("gravity", 16.0, preset="moon"), ev("undo", 16.1), ev("joke", 16.2, tag="banana")]
    notes, facts = summarise(events)
    text = " | ".join(notes)
    assert len(notes) <= 8 and all(len(n) <= 100 for n in notes)
    assert "built three couches and a pool today" in text
    assert "third couch of the evening" in text
    assert "kettle failed twice" in text
    assert "set gravity to moon" in text and "filled water with mercury" in text
    assert "undid one thing" in text and "already joked about banana" in text
    assert facts["top_item"] == "couch" and facts["top_n"] == 3 and facts["fail_item"] == "kettle" and facts["fail_n"] == 2
    assert facts["pool_built"] == "yes" and facts["gravity"] == "moon"


def test_the_notes_are_deterministic():
    events = [ev("add", 14.5, name="couch", n=1), ev("add", 14.6, name="lamp", n=2), ev("beam_fail", 15.5, name="kettle")]
    assert summarise(events) == summarise(list(reversed(events))[::-1]) == summarise(events)


def test_only_todays_events_count():
    yesterday = [ev("add", -5, name="sofa", n=4)]
    notes, facts = summarise(yesterday + [ev("add", 14.5, name="lamp", n=1)])
    assert "lamp" in " ".join(notes) and "sofa" not in " ".join(notes)


def test_no_player_name_or_wake_name_or_id_gets_into_the_notes():
    events = [ev("beam_fail", 15.0, name="rexxy the dragon"), ev("beam_fail", 15.1, name="Aaron"), ev("add", 15.2, name="76561190000000001"),
              ev("add", 15.3, name="couch", n=1), ev("beam_fail", 15.4, name="a b c d e f"), ev("beam_fail", 15.5, name="rex")]
    notes, facts = summarise(events, banned=["Aaron", "Rexxy"])
    text = " ".join(notes).lower()
    for bad in ("aaron", "rexxy", "7656119", "dragon", "rex "):
        assert bad not in text, bad
    assert "couch" in text
    assert all(v == v for v in facts.values())
    blob = json.dumps([notes, facts])
    assert "76561190000000001" not in blob and "Aaron" not in blob


def test_the_safe_noun_rule():
    w = pure()
    N = mod(w, "notes")
    assert N.safe_noun("Fire Hydrant", lua_table(w.rt, [])) == "fire hydrant"
    for bad in ("", "a" * 30, "co<uch", "couch 2", "one two three four", "http://x.y", "76561190000000001"):
        assert N.safe_noun(bad, lua_table(w.rt, [])) == "something", bad
    assert N.safe_noun("rexalot", lua_table(w.rt, ["Rex"])) == "something", "part of a wake name"


def test_ordinals_and_streaks_and_time_since():
    events = [ev("session_start", 13.0)] + [ev("add", 14 + i * 0.01, name="chair", n=1) for i in range(5)] + [ev("beam_ok", 14.2, name="boombox")]
    notes, facts = summarise(events, now=NOW + int(16.5 * 3600))
    text = " | ".join(notes)
    assert "fifth chair of the afternoon" in text and "asked for a chair five times" in text
    assert "six builds in a row without an undo" in text
    assert "last action 138 minutes ago" in text and facts["minutes_since"] == 138
    assert "playing for 210 minutes" in text


# ============================================================================================================ the journal (server files)
def jworld():
    w = pure()
    w.rt.execute("SERVER = true; " + ENGINE)
    J = mod(w, "journal")
    return w, J


def test_the_journal_round_trips_and_keeps_only_known_small_events():
    w, J = jworld()
    assert J.log(A, lua_table(w.rt, {"k": "add", "name": "couch", "n": 2, "src": "curated"}), NOW) is True
    assert J.log(A, lua_table(w.rt, {"k": "evil", "name": "x"}), NOW) is False, "an unknown kind is refused"
    assert J.log("not-a-sid", lua_table(w.rt, {"k": "add"}), NOW) is False
    J.log(A, lua_table(w.rt, {"k": "add", "name": 'x"y\\z\n' + "q" * 100, "n": 99999999, "extra": "ignored"}), NOW + 1)
    evs = list(J.read(A, NOW + 2).values())
    assert [e.k for e in evs] == ["add", "add"] and evs[0].name == "couch" and evs[0].n == 2
    assert evs[1].n == 9999 and len(evs[1].name) <= 28 and '"' not in evs[1].name and "\n" not in evs[1].name and evs[1].extra is None
    files = w.rt.eval("function() local o = {} for p in pairs(MOCK.files) do o[#o + 1] = p end table.sort(o) return table.concat(o, ',') end")()
    assert files == f"mentat/journal/{A}.txt"


def test_the_journal_rotates_at_its_size_cap_and_holds_at_most_two_files():
    w, J = jworld()
    J.MAX_BYTES = 600
    for i in range(80):
        J.log(A, lua_table(w.rt, {"k": "add", "name": "couch", "n": 1}), NOW + i)
    files = w.rt.eval("function() local o = {} for p in pairs(MOCK.files) do o[#o + 1] = p end table.sort(o) return table.concat(o, ',') end")()
    assert files == f"mentat/journal/{A}.1.txt,mentat/journal/{A}.txt", "a current file and one rotated file, never more"
    sizes = w.rt.eval(f"function() return #MOCK.files['mentat/journal/{A}.txt'], #MOCK.files['mentat/journal/{A}.1.txt'] end")()
    assert max(sizes) < 600 + 120
    assert len(list(J.read(A, NOW + 100).values())) < 80, "the oldest events fell off"


def test_events_older_than_the_keep_window_are_dropped_on_reading():
    w, J = jworld()
    J.log(A, lua_table(w.rt, {"k": "add", "name": "couch", "n": 1}), NOW - 60 * 86400)
    J.log(A, lua_table(w.rt, {"k": "add", "name": "lamp", "n": 1}), NOW)
    assert [e.name for e in J.read(A, NOW + 5).values()] == ["lamp"]


def test_forget_that_today_and_everything():
    w, J = jworld()
    df = w.rt.eval("function(f) return function(fmt, t) return f(fmt, t) end end")(utc)
    for h, name in ((-30, "old"), (14, "a"), (15, "b"), (16, "c")):
        J.log(A, lua_table(w.rt, {"k": "add", "name": name, "n": 1}), NOW + h * 3600)
    J.log(A, lua_table(w.rt, {"k": "joke", "tag": "banana"}), NOW + 16 * 3600 + 5)
    now = NOW + 17 * 3600
    assert J.forget(A, "that", now, df) == 1
    names = [e.name or e.k for e in J.read(A, now).values()]
    assert names == ["old", "a", "b", "joke"], "'that' drops the last ACTION, not a joke"
    assert J.forget(A, "today", now, df) == 3
    assert [e.name for e in J.read(A, now).values()] == ["old"], "'today' keeps earlier days"
    assert J.forget(A, "all", now, df) == 1
    assert list(J.read(A, now).values()) == [] and w.rt.eval("function() return next(MOCK.files) end")() is None
    other = J.log(B, lua_table(w.rt, {"k": "add", "name": "x", "n": 1}), now)
    J.forget(A, "all", now, df)
    assert len(list(J.read(B, now).values())) == 1, "one player's forget never touches another's journal"


# ============================================================================================================ the director in the world
BASE_CV = {"mentat_skippy_director": "1", "mentat_skippy_chat": "chatty", "holodeck_bridge_per_min": "600", "holodeck_voice_chars_per_min": "100000",
           "mentat_skippy_quiet_hours": "", "mentat_skippy_new_lines_per_day": "20"}


ENGINE = "MOCK.strict_files = true; MOCK.json_number_keys = true"       # two things the real engine does (found in the live pass): see gmod_mock.lua


def dworld(cv=None, summon=True, sid=A, **pkw):
    w = avatar_world({**BASE_CV, **(cv or {})}, bridge=True)
    w.rt.execute(ENGINE)
    p = w.player(sid, x=0, y=0, yaw=0, **pkw)
    w.rt.execute("Mentat.Director.Rand = function() return 0.0 end")             # the dice always say yes: the tests choose what happens
    if summon:
        w.ask(p, "come here")
        resolve_quips(w, "fail")                      # (the arrival line is asked for: not answered here, so the canned "Here I am." comes)
        run(w, 6)
    return w, p


def fresh(w):
    """Forget the traffic so far (the request numbers start again from 1)."""
    w.rt.execute("for i, t in ipairs(MOCK.https) do if not t.done then MOCK.respond(i, 'fail', 'x') end end")        # (nothing stays in flight)
    w.rt.execute("MOCK.reset_traffic()")
    w._answered = set()


def quips(w):
    return list(w.rt.eval("function() return T.posts('/quip') end")().values())


def quip_body(w, i):
    return json.loads(w.rt.eval("function(i) return MOCK.https[i].body end")(i))


def resolve_quips(w, how="fail"):
    done = getattr(w, "_answered", set())
    for i in quips(w):
        if i in done:
            continue
        done.add(i)
        if how == "fail":
            w.rt.eval("function(i) MOCK.respond(i, 'fail', 'no') end")(i)
    w._answered = done


def answer(w, i, meta=None, audio=True, **kw):
    m = {"ok": True, "text": "Behold, the magnificent can has arrived, monkey.", "mood": "boast", "source": "bank", "id": "arr01", "cat": "arrival", "tags": [],
         "cached": False, "new": True, "dur": 3.0, "voice": "elevenlabs", "budget": {"day": "2026-10-13", "used": 1, "limit": 20, "left": 19}}
    m.update(meta or {})
    m.update(kw)
    if audio:
        w.rt.eval("function(i, h) MOCK.respond(i, 200, 'ID3' .. string.rep('x', 3000), h) end")(i, lua_table(w.rt, {"X-Skippy-Quip": json.dumps(m)}))
    else:
        w.rt.eval("function(i, b) MOCK.respond(i, 200, b, {}) end")(i, json.dumps(m))
    w._answered = getattr(w, "_answered", set()) | {i}
    return m


def new_quips(w):
    done = getattr(w, "_answered", set())
    return [i for i in quips(w) if i not in done]


def dr(w, code, *a):
    return L(w, "local Dr = Mentat.Director; " + code, *a)


def nw(w, key, default=None):
    return w.rt.eval("function(k, d) local b = Mentat.Avatar.Get(); return b:GetNWInt(k, d) end")(key, default)


def test_called_he_asks_for_a_fresh_arrival_line_and_plays_a_hat_tip_at_once():
    w = avatar_world({**BASE_CV}, bridge=True)
    p = w.player(A, x=0, y=0)
    w.ask(p, "come here")
    qs = quips(w)
    assert len(qs) == 1
    body = quip_body(w, qs[0])
    assert body["situation"] == "arrive" and body["roast"] == "gentle" and body["daily_limit"] == 20 and body["audio"] is True
    assert nw(w, "mentat_av_cue_seq", 0) >= 1 and nw(w, "mentat_av_cue") == 1, "the hat tip (cue 1) starts at once, before the line comes"
    meta = answer(w, qs[0])
    w.advance(1)
    chats = [m for m in w.rt.eval("function() return T.nets('holodeck_say') end")().values()]
    assert any(meta["text"] in str(w.rt.eval("function(m) return m.writes[1] end")(c)) for c in chats), "the line is shown in chat"
    audio = list(w.rt.eval("function() return T.nets('holodeck_audio') end")().values())
    assert audio, "and the clip goes out"
    v = w.rt.eval("function(m) return m.vector.x, m.vector.y, m.vector.z end")(audio[0])
    bx, by, bz = where(w, bot(w), ) if False else tuple(w.rt.eval("function() local p = Mentat.Avatar.Get():GetPos(); return p.x, p.y, p.z end")())
    assert (v[0], v[1], v[2]) == pytest.approx((bx, by, bz), abs=1e-6), "from where he floats"
    assert dr(w, "return Dr.budget.left") == 19, "the budget the bridge reports is kept"
    assert dr(w, "return Dr.st.speaking_until > CurTime()") is True
    assert dr(w, "return Dr.last_line.mood") == "boast" and nw(w, "mentat_av_mood") is not None


def test_when_the_arrival_line_cannot_be_made_the_canned_answer_is_spoken_instead():
    w = avatar_world({**BASE_CV}, bridge=True)
    p = w.player(A, x=0, y=0)
    w.ask(p, "come here")
    (i,) = quips(w)
    assert not w.rt.eval("function() return Holodeck.last_say end")(), "the canned answer waits for the model/bank line"
    w.rt.eval("function(i) MOCK.respond(i, 503, 'busy') end")(i)
    assert w.rt.eval("function() return Holodeck.last_say end")() == "Here I am."


def test_with_the_director_off_he_behaves_exactly_as_before():
    w = avatar_world({**BASE_CV, "mentat_skippy_director": "0"}, bridge=True)
    p = w.player(A, x=0, y=0)
    w.ask(p, "come here")
    assert quips(w) == []
    assert w.rt.eval("function() return Holodeck.last_say end")() == "Here I am."
    run(w, 600)
    assert quips(w) == [], "no whims either"
    assert dr(w, "return Dr.Event('beam_ok', {}) ") is False


def test_a_text_only_answer_is_shown_without_audio_and_does_not_hold_him_silent_for_long():
    w, p = dworld()
    fresh(w)
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    (i,) = new_quips(w)
    answer(w, i, audio=False, text="That one failed, monkey.", mood="musing", new=False)
    w.advance(0.5)
    assert list(w.rt.eval("function() return T.nets('holodeck_audio') end")().values()) == []
    assert dr(w, "return Dr.st.speaking_until - CurTime() <= 4.01") is True


def test_he_never_talks_over_a_human_and_the_reaction_waits_for_a_gap():
    w, p = dworld()
    n0 = len(quips(w))
    w.rt.execute(f"Mentat.Director.talk[MOCK.players[1]] = {{ on = true, t = CurTime(), last = CurTime() }}")
    run(w, 5)                                      # (the first player is the owner)
    dr(w, "Dr.talk[...] = { on = true, t = CurTime(), last = CurTime() }", p)
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    assert len(quips(w)) == n0, "no request while they speak"
    assert dr(w, "return #Dr.deferred") == 1
    dr(w, "Dr.talk[...] = nil", p)
    run(w, 3)
    assert len(quips(w)) == n0 + 1, "the deferred remark was asked for once they stopped"


def test_a_human_who_is_still_talking_after_the_deferral_window_loses_him_the_remark():
    w, p = dworld()
    n0 = len(quips(w))
    dr(w, "Dr.talk[...] = { on = true, t = CurTime() + 100, last = CurTime() }", p)
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    run(w, 14)
    assert len(quips(w)) == n0 and dr(w, "return #Dr.deferred") == 0, "gave up after DEFER_MAX_S"


def test_he_does_not_start_a_second_line_while_the_first_is_playing():
    w, p = dworld()
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    (i,) = new_quips(w)
    answer(w, i, dur=6.0, text="Oh dear. A missing object, monkey.", mood="musing")
    n = len(quips(w))
    run(w, 1)
    dr(w, "Dr.Event('beam_ok', { ply = ... })", p)
    assert len(quips(w)) == n, "inside his own line: nothing asked"
    run(w, 7)
    dr(w, "Dr.Event('beam_ok', { ply = ... })", p)
    assert len(quips(w)) in (n, n + 1)          # (the dice may still say no; the rule is that it may ask now)


def test_a_human_speaking_over_him_cuts_the_line_and_tells_the_clients():
    w, p = dworld()
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    (i,) = new_quips(w)
    answer(w, i, dur=8.0, text="A very long boast follows, monkey.", mood="boast")
    run(w, 1)
    assert dr(w, "return Dr.st.speaking_until > CurTime()") is True
    fresh(w)
    w.rt.eval("function(p) MOCK.net_from_client(p, 'mentat_av_talk', { { 'b', true } }) end")(p)
    hush = list(w.rt.eval("function() return T.nets('mentat_av_hush') end")().values())
    assert len(hush) == 1, "every client is told to stop the clip"
    assert dr(w, "return Dr.st.speaking_until <= CurTime()") is True and dr(w, "return Dr.counts.interrupted") == 1
    w.rt.eval("function(p) MOCK.net_from_client(p, 'mentat_av_talk', { { 'b', true } }) end")(p)
    assert dr(w, "return Dr.counts.interrupted") == 1, "talking on does not cut him again"


def test_a_client_that_goes_silent_cannot_gag_him_for_ever():
    w, p = dworld()
    w.rt.eval("function(p) MOCK.net_from_client(p, 'mentat_av_talk', { { 'b', true } }) end")(p)
    assert dr(w, "return Dr.HumanSpeaking()") is True
    run(w, 6)
    assert dr(w, "return Dr.HumanSpeaking()") is False, "an unrefreshed report lapses"


def test_he_yields_while_the_owner_is_mid_command():
    w, p = dworld()
    n0 = len(quips(w))
    w.ask(p, "add a couch")                          # a request: the owner is mid-command for a few seconds
    run(w, 0.2)
    n1 = len(quips(w))
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    assert len(quips(w)) == n1, "not now"
    run(w, 6)
    assert len(quips(w)) >= n1, "later"


def test_his_remarks_are_about_his_owner_only():
    w, p = dworld()
    q = w.player(B, x=300, y=0)
    n0 = len(quips(w))
    assert dr(w, "return Dr.Event('beam_ok', { ply = ... })", q) is False
    assert len(quips(w)) == n0


def test_quiet_hours_keep_whims_and_comments_silent():
    h = int(time.strftime("%H"))
    w, p = dworld({"mentat_skippy_quiet_hours": f"{h}-{(h + 2) % 24}"})
    fresh(w)
    assert dr(w, "return Dr.Event('musing', { ply = ... })", p) is False
    assert dr(w, "return Dr.Event('beam_fail', { ply = ... })", p) is False
    assert quips(w) == []


# ---- chattiness and roast level, per player, private, persisted
def prefs_file(w):
    """The preference file as {steamid64: {...}} (on disk it is a list of records: the engine mangles numeric-looking JSON KEYS)."""
    raw = w.rt.eval("function() return MOCK.files['mentat/skippy_prefs.json'] end")()
    if not raw:
        return None
    obj = json.loads(raw)
    assert isinstance(obj["players"], list), "a list of records, never a table keyed by SteamID64"
    return {"players": {r.pop("sid"): {k: v for k, v in r.items() if v is not None} for r in obj["players"]}}


def say_line(w):
    return w.rt.eval("function() return Holodeck.last_say end")()


def test_shush_and_be_chatty_are_remembered_per_player_by_steamid():
    w, p = dworld()
    w.ask(p, "shush")
    assert dr(w, "return Dr.ChatLevel(...)", p) == "quiet" and prefs_file(w)["players"][A]["chat"] == "quiet"
    assert "Hush mode" in say_line(w)
    q = w.player(B, x=300, y=0)
    assert dr(w, "return Dr.ChatLevel(...)", q) == "chatty", "someone else's chattiness is unaffected (the server default is chatty here)"
    w.ask(p, "Skippy, be chatty")
    assert dr(w, "return Dr.ChatLevel(...)", p) == "chatty" and prefs_file(w)["players"][A]["chat"] == "chatty"


def test_a_shushed_owner_gets_no_comments_only_answers():
    w, p = dworld()
    w.ask(p, "shush")
    fresh(w)
    assert dr(w, "return Dr.Event('beam_ok', { ply = ... })", p) is False
    assert dr(w, "return Dr.Event('splash', { ply = ... })", p) is False
    assert quips(w) == []


def test_everyone_starts_gentle_the_host_and_the_listed_captains_start_savage():
    w, p = dworld()
    assert dr(w, "return Dr.RoastLevel(...)", p) == "gentle"
    host = w.player(B, x=100, y=0, host=True)
    assert dr(w, "return Dr.RoastLevel(...)", host) == "savage"
    w2 = avatar_world({**BASE_CV, "mentat_skippy_captain_sids": f"123,{C},456"}, bridge=True)
    cap = w2.player(C)
    assert L(w2, "return Mentat.Director.RoastLevel(...)", cap) == "savage"
    assert L(w2, "return Mentat.Director.RoastLevel(...)", w2.player(A)) == "gentle"


def test_roast_me_harder_and_go_easy_on_me_step_the_level_and_it_is_stored_privately():
    w, p = dworld()
    w.ask(p, "roast me harder")
    assert dr(w, "return Dr.RoastLevel(...)", p) == "standard"
    w.ask(p, "roast me harder")
    assert dr(w, "return Dr.RoastLevel(...)", p) == "savage" and "Savage mode" in say_line(w)
    w.ask(p, "roast me harder")
    assert "already as savage" in say_line(w)
    w.ask(p, "go easy on me")
    assert dr(w, "return Dr.RoastLevel(...)", p) == "standard"
    w.ask(p, "be nice")
    w.ask(p, "be nice")
    assert dr(w, "return Dr.RoastLevel(...)", p) == "gentle" and "already as gentle" in say_line(w)
    assert prefs_file(w)["players"][A]["roast"] == "gentle"
    # private: the other players' chat and the server console never see the command or the level
    assert not any("roast" in l.lower() and "harder" in l.lower() for l in w.rt.eval("function() return MOCK.printed end")().values())


def test_the_engines_rules_that_the_first_live_pass_found_are_honoured():
    """.jsonl and .tmp files are silently never written by the real engine; a SteamID64 used as a JSON KEY comes back as a rounded number."""
    w, p = dworld({"holodeck_found": "0"})
    w.ask(p, "add a couch")
    files = w.rt.eval("function() local o = {} for k in pairs(MOCK.files) do o[#o + 1] = k end table.sort(o) return table.concat(o, ',') end")().split(",")
    assert f"mentat/journal/{A}.txt" in files and not any(f.endswith((".jsonl", ".tmp")) for f in files), files
    assert journal_events(w), "so the journal really was written under the strict rule"
    w.ask(p, "roast me harder")
    assert prefs_file(w)["players"][A]["roast"] == "standard"
    raw = w.rt.eval("function() return MOCK.files['mentat/skippy_prefs.json'] end")()
    w3 = avatar_world({**BASE_CV}, bridge=True)
    w3.rt.execute(ENGINE)
    w3.rt.eval("function(r) MOCK.files['mentat/skippy_prefs.json'] = r end")(raw)
    L(w3, "Mentat.Director.LoadPrefs()")
    assert L(w3, "return Mentat.Director.RoastLevel(...)", w3.player(A)) == "standard", "the level comes back for the right player after the engine's own JSON conversions"
    assert w3.rt.eval("function() return MOCK.files['mentat/skippy_prefs_tmp.txt'] end")() is None, "the temp file was renamed away"


def test_the_roast_level_survives_a_restart_and_a_corrupt_file_means_the_defaults():
    w, p = dworld()
    w.ask(p, "roast me harder")
    raw = w.rt.eval("function() return MOCK.files['mentat/skippy_prefs.json'] end")()
    w2 = avatar_world({**BASE_CV}, bridge=True)
    w2.rt.execute(ENGINE)
    w2.rt.eval("function(r) MOCK.files['mentat/skippy_prefs.json'] = r end")(raw)
    L(w2, "Mentat.Director.LoadPrefs()")
    assert L(w2, "return Mentat.Director.RoastLevel(...)", w2.player(A)) == "standard"
    w2.rt.eval("function() MOCK.files['mentat/skippy_prefs.json'] = '{not json' end")()
    L(w2, "Mentat.Director.LoadPrefs()")
    assert L(w2, "return Mentat.Director.RoastLevel(...)", w2.player(A)) == "gentle"
    w2.rt.eval("function() MOCK.files['mentat/skippy_prefs.json'] = '{\"players\": [{\"sid\": \"abc\", \"roast\": \"savage\"}, {\"sid\": \"%s\", \"roast\": \"mean\", \"chat\": \"loud\"}]}' end" % A)()
    L(w2, "Mentat.Director.LoadPrefs()")
    assert L(w2, "return Mentat.Director.RoastLevel(...)", w2.player(A)) == "gentle", "an unknown level or a bad id is ignored"


def test_the_roast_level_goes_with_the_request_so_a_gentle_player_is_never_sent_a_savage_line():
    w, p = dworld()
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    (i,) = new_quips(w)
    assert quip_body(w, i)["roast"] == "gentle"
    answer(w, i, audio=False)
    run(w, 10)
    w.ask(p, "roast me harder")
    run(w, 10)
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    qs = new_quips(w)
    assert quips(w) and quip_body(w, quips(w)[-1])["roast"] == "standard"


def test_roast_me_asks_for_a_roast_line_now():
    w, p = dworld()
    fresh(w)
    w.ask(p, "roast me")
    qs = quips(w)
    assert len(qs) == 1 and quip_body(w, qs[0])["situation"] == "roast"


# ---- the journal, fed by what really happens
def journal_events(w, sid=A):
    raw = w.rt.eval(f"function() return MOCK.files['mentat/journal/{sid}.txt'] end")()
    return [json.loads(x) for x in (raw or "").splitlines() if x.strip()]


def test_what_the_player_does_is_written_down_without_names():
    w, p = dworld({"holodeck_found": "0"})
    w.ask(p, "add three couches")
    w.ask(p, "add a lamp")
    w.ask(p, "undo")
    evs = journal_events(w)
    kinds = [(e["k"], e.get("name")) for e in evs]
    assert ("session_start", None) in kinds
    assert ("add", "couch") in kinds and ("add", "lamp") in kinds or any(e["k"] == "add" for e in evs)
    assert any(e["k"] == "undo" for e in evs)
    blob = json.dumps(evs)
    assert A not in blob and "Skippy" not in blob and "name" not in {k for e in evs for k in e if k == "nick"}
    counts = {e.get("name"): e.get("n") for e in evs if e["k"] == "add"}
    assert any(v == 3 for v in counts.values()), "three couches are one event of three, not three events"


def test_a_beam_arriving_or_failing_and_a_resolve_failure_are_recorded():
    w, p = dworld()
    w.rt.eval("""function(p) hook.Run('HolodeckBeam', 'ready', { card = 'found.poly-haven.boombox', asker = p }) end""")(p)
    w.rt.eval("""function(p) hook.Run('HolodeckBeam', 'failed', { card = 'found.poly-haven.kettle', asker = p }, 'timeout') end""")(p)
    w.rt.eval("""function(p) hook.Run('HolodeckResolve', p, 'flux capacitor', { say = 'I do not have that.', class = 'failure', strict = true }) end""")(p)
    evs = journal_events(w)
    got = {(e["k"], e.get("name")) for e in evs}
    assert ("beam_ok", "boombox") in got and ("beam_fail", "kettle") in got and ("beam_fail", "flux capacitor") in got


def test_a_flourish_plays_when_what_he_asked_for_lands_even_with_nothing_to_say():
    w, p = dworld({"mentat_skippy_chat": "quiet"})
    seq = nw(w, "mentat_av_cue_seq", 0)
    run(w, 6)
    w.rt.eval("""function(p) hook.Run('HolodeckBeam', 'ready', { card = 'found.poly-haven.boombox', asker = p }) end""")(p)
    assert nw(w, "mentat_av_cue_seq", 0) == seq + 1 and nw(w, "mentat_av_cue") == 2, "a flourish (cue 2)"


def test_a_big_pile_is_noticed_once_per_step():
    w, p = dworld()
    fresh(w)
    w.rt.eval("""function(p)
      for i = 1, 31 do Holodeck.registry[1000 + i] = { id = 1000 + i, kind = 'crate', x = i, y = 0, z = 0, owner = p:SteamID64() }; Holodeck.order[#Holodeck.order + 1] = 1000 + i end
      hook.Run('HolodeckResult', p, {}, { action = 'add', ids = { 1031 } })
    end""")(p)
    sits = [quip_body(w, i)["situation"] for i in quips(w)]
    assert "bigpile" not in sits and ("big_pile" in sits or "build" in sits or sits == []), sits
    assert dr(w, "return Dr.pile['%s']" % A) == 30


def test_a_splash_near_the_owner_is_counted_and_one_far_away_is_not():
    w, p = dworld()
    fresh(w)
    dr(w, "Dr.OnSplash(100, 0, 80)")
    dr(w, "Dr.OnSplash(9000, 0, 80)")
    splashes = [e for e in journal_events(w) if e["k"] == "splash"]
    assert len(splashes) == 1 and splashes[0]["n"] == 1, "one near splash recorded, the far one ignored"


def test_a_wake_name_change_is_noted_and_remarked_on_without_the_name():
    w, p = dworld()
    fresh(w)
    w.ask(p, "answer to Rexalot")
    evs = journal_events(w)
    assert any(e["k"] == "name_change" for e in evs)
    assert "rexalot" not in json.dumps(evs).lower()
    for i in quips(w):
        assert "rexalot" not in json.dumps(quip_body(w, i)).lower(), "the name never goes to the bridge"


def test_a_wake_word_miss_can_raise_a_remark():
    w, p = dworld()
    fresh(w)
    w.rt.eval("function(p) hook.Run('HolodeckWakeMiss', p) end")(p)
    sits = [quip_body(w, i)["situation"] for i in quips(w)]
    assert sits in (["wake_miss"], []), sits


def test_the_idle_owner_is_remarked_on_and_the_whims_come_by_themselves():
    w, p = dworld({"mentat_skippy_chat": "chatty"})
    fresh(w)
    sits = set()
    for _ in range(40):
        run(w, 20)
        for i in new_quips(w):
            sits.add(quip_body(w, i)["situation"])
            w.rt.eval("function(i) MOCK.respond(i, 'fail', 'x') end")(i)
            w._answered = getattr(w, "_answered", set()) | {i}
    assert "owner_idle" in sits or "musing" in sits, sits
    assert len(sits) >= 2, sits


def test_a_quiet_owner_hears_far_fewer_whims_than_a_chatty_one():
    counts = {}
    for level in ("quiet", "chatty"):
        w, p = dworld({"mentat_skippy_chat": level})
        fresh(w)
        n = 0
        for _ in range(60):
            run(w, 20)
            for i in new_quips(w):
                n += 1
                w.rt.eval("function(i) MOCK.respond(i, 'fail', 'x') end")(i)
                w._answered = getattr(w, "_answered", set()) | {i}
        counts[level] = n
    assert counts["chatty"] >= 3 * max(1, counts["quiet"]) or counts["quiet"] == 0 and counts["chatty"] > 0, counts


# ---- what leaves the machine
def test_the_request_carries_notes_but_never_an_id_a_name_or_a_wake_name():
    w, p = dworld({"holodeck_found": "0"}, name="Aaron Rhodes")
    w.ask(p, "answer to Rexalot")
    for t in ("add a couch", "add a couch", "add a couch"):
        w.ask(p, t)
    w.rt.eval("""function(p) hook.Run('HolodeckResolve', p, 'Aaron Rhodes', { say = 'nope', class = 'failure', strict = true }) end""")(p)
    fresh(w)
    run(w, 12)
    w.ask(p, "what do you remember")
    qs = quips(w)
    assert qs, "the recall question is sent to the bridge"
    body = json.loads(w.rt.eval("function(i) return MOCK.https[i].body end")(qs[-1]))
    wire = json.dumps({k: body[k] for k in ("situation", "roast", "notes", "facts")})
    assert body["notes"], "there are notes to send"
    for secret in (A, "Rexalot", "rexalot", "Aaron", "Rhodes", "76561190"):
        assert secret not in wire, secret
    assert any("couch" in n for n in body["notes"])
    assert body["who"] == A, "the id is only the rate-limit key between the game and the LOCAL bridge; the bridge never forwards it"


def test_what_do_you_remember_is_answered_from_the_notes_in_character_or_plainly_if_the_bridge_fails():
    w, p = dworld({"holodeck_found": "0"})
    w.ask(p, "add a couch")
    fresh(w)
    w.ask(p, "what do you remember")
    (i,) = quips(w)
    assert quip_body(w, i)["situation"] == "recall"
    w.rt.eval("function(i) MOCK.respond(i, 'fail', 'down') end")(i)
    assert "couch" in say_line(w).lower(), "plain fallback built from the same notes"
    w2, p2 = dworld({"holodeck_found": "0"})
    fresh(w2)
    w2.ask(p2, "what do you remember")
    assert quips(w2) == [] and "nothing in my notes" in say_line(w2).lower()


def test_forget_commands_delete_the_notes_and_nothing_else():
    w, p = dworld({"holodeck_found": "0"})
    w.ask(p, "add a couch")
    w.ask(p, "add a lamp")
    assert len(journal_events(w)) >= 3
    w.ask(p, "forget that")
    names = [e.get("name") for e in journal_events(w) if e["k"] == "add"]
    assert names == ["couch"], names
    w.ask(p, "forget today")
    assert [e for e in journal_events(w) if e["k"] == "add"] == []
    w.ask(p, "add a rug")
    w.ask(p, "forget everything")
    assert journal_events(w) == [] and "deleted" in say_line(w).lower()
    assert prefs_file(w) is None or "roast" not in prefs_file(w)["players"].get(A, {}), "preferences are not memories"


def test_the_player_can_opt_out_of_the_notebook_and_back_in():
    w, p = dworld({"holodeck_found": "0"})
    w.ask(p, "add a couch")
    w.ask(p, "stop remembering")
    assert journal_events(w) == [], "the old notes are deleted"
    assert prefs_file(w)["players"][A]["journal"] is False
    w.ask(p, "add a lamp")
    assert journal_events(w) == [], "and nothing new is written"
    assert dr(w, "return #Dr.NotesFor(...).notes", p) == 0
    w.ask(p, "what do you remember")
    assert "not keeping notes" in say_line(w)
    w.ask(p, "start remembering")
    w.ask(p, "add a lamp")
    assert any(e["k"] == "add" for e in journal_events(w))
    q = w.player(B, x=300)
    assert dr(w, "return Dr.JournalOn(...)", q) is True, "another player's choice is their own"


def test_the_journal_convar_switches_the_whole_notebook_off():
    w, p = dworld({"holodeck_found": "0", "mentat_skippy_journal": "0"})
    w.ask(p, "add a couch")
    run(w, 8)
    fresh(w)
    assert journal_events(w) == [] and w.rt.eval("function() return next(MOCK.files, nil) end")() in (None, "mentat/skippy_prefs.json")
    assert dr(w, "return #Dr.NotesFor(...).notes", p) == 0
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    (i,) = new_quips(w)
    assert quip_body(w, i)["notes"] == [], "no notes means nothing about the player reaches the model"


def test_a_line_that_names_a_joke_is_remembered_so_a_callback_is_fresh():
    w, p = dworld()
    dr(w, "Dr.Event('beam_fail', { ply = ... })", p)
    (i,) = new_quips(w)
    answer(w, i, text="A banana would have worked, monkey.", tags=["banana", "callback"], dur=2.0, mood="musing")
    assert any(e["k"] == "joke" and e["tag"] == "banana" for e in journal_events(w))
    assert not any(e.get("tag") == "callback" for e in journal_events(w))
    notes = dr(w, "return Dr.NotesFor(...).notes", p)
    assert any("already joked about banana" in n for n in notes.values())


# ---- replication to the clients
def test_the_cue_and_mood_are_replicated_on_the_bot_and_each_cue_has_a_new_sequence_number():
    w, p = dworld()
    seq0 = nw(w, "mentat_av_cue_seq", 0)
    assert dr(w, "return Dr.Cue('flourish', 'boast')") is True
    assert (nw(w, "mentat_av_cue"), nw(w, "mentat_av_cue_seq")) == (2, seq0 + 1)
    assert w.rt.eval("function() return Mentat.Avatar.Get():GetNWString('mentat_av_mood', '') end")() == "boast"
    dr(w, "Dr.Cue('conduct', 'opera')")
    assert (nw(w, "mentat_av_cue"), nw(w, "mentat_av_cue_seq")) == (3, seq0 + 2)
    assert dr(w, "return Dr.Cue('no_such_cue')") is False
    run(w, 1)
    assert w.rt.eval("function() return Mentat.Avatar.Get():GetNWBool('mentat_av_dir', false) end")() is True
    dr(w, "Dr.LookAt(Vector(10, 20, 30), 'test')")
    run(w, 1)
    assert w.rt.eval("function() return Mentat.Avatar.Get():GetNWBool('mentat_av_look_on', false), Mentat.Avatar.Get():GetNWVector('mentat_av_look').x end")() == (True, 10)


def test_an_opera_is_always_conducted_and_a_roast_leans_in():
    w, p = dworld()
    dr(w, "Dr.Event('opera', { ply = ... })", p)
    qs = new_quips(w)
    if qs:
        assert quip_body(w, qs[0])["situation"] == "opera"
        answer(w, qs[0], text="LA LA LAAAA, I am the nightingale of the nebula!", mood="opera", tags=["opera"], dur=4.0)
        assert nw(w, "mentat_av_cue") == 3, "the conductor's sweep"
        assert w.rt.eval("function() return Mentat.Avatar.Get():GetNWString('mentat_av_mood', '') end")() == "opera"


def test_he_looks_at_what_you_just_placed_and_wanders_a_little_instead_of_hanging_rigid():
    w, p = dworld({"holodeck_found": "0"})
    w.rt.eval("""function(p) Holodeck.registry[777] = { id = 777, kind = 'lamp', x = 400, y = 0, z = 0, owner = p:SteamID64() }; Holodeck.order[#Holodeck.order + 1] = 777
       hook.Run('HolodeckResult', p, {}, { action = 'add', ids = { 777 } }) end""")(p)
    run(w, 3)
    yaw = w.rt.eval("function() return Mentat.Avatar.Get().yaw end")()
    bx, by = w.rt.eval("function() local p = Mentat.Avatar.Get():GetPos(); return p.x, p.y end")()
    want = math.degrees(math.atan2(0 - by, 400 - bx))
    assert abs((yaw - want + 180) % 360 - 180) < 2, "he turned to face the new lamp"
    zs = []
    for _ in range(60):
        run(w, 1)
        zs.append(w.rt.eval("function() return Mentat.Avatar.Get():GetPos().y end")())
    assert max(zs) - min(zs) > 3, "he drifts"


def test_the_director_survives_a_lua_refresh_without_stacking_hooks():
    w, p = dworld()
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/avatar.lua")')
    n = len(quips(w))
    fresh(w)
    w.rt.eval("""function(p) hook.Run('HolodeckBeam', 'failed', { card = 'found.x.kettle', asker = p }, 'x') end""")(p)
    assert len([e for e in journal_events(w) if e["k"] == "beam_fail"]) == 1, "one hook, one entry"


# ============================================================================================================ the client draws the animation
def posed_client(pose_set=None, dir_on=True):
    s, c, msgs = full_pipeline(synthetic_set())
    c.rt.execute(f"MOCK.bot:SetNWBool('mentat_av_dir', {'true' if dir_on else 'false'})")
    # remember each matrix's parent so a test can see what a part was drawn under (the fake does not multiply matrices)
    c.rt.execute("cam.PushModelMatrix = function(m, mul) m.parent = RB.stack[#RB.stack]; RB.stack[#RB.stack + 1] = m end")
    return c


def top_ops(c, part_index=1):
    d = draws(c)[part_index]
    return d


def test_without_the_directors_flag_the_client_draws_exactly_as_before():
    c = posed_client(dir_on=False)
    c.rt.eval("DRAW_FRAME")(5, 0.05)
    assert c.rt.eval("Mentat.Avatar.anim_seen") is False
    base = draws(c)[1].top
    assert base is not None


def test_a_hat_tip_cue_lifts_the_hat_and_tips_it_and_it_returns():
    c = posed_client()
    c.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")("can,jacket,head,hand_a,hand_b")
    c.rt.eval("DRAW_FRAME")(3, 0.05)
    rest = draws(c)[1].top
    rest_pitch = rest.rot.p
    c.rt.execute("MOCK.bot:SetNWInt('mentat_av_cue', 1); MOCK.bot:SetNWInt('mentat_av_cue_seq', 1)")
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    best, lifted = 0, 0
    for _ in range(40):
        c.rt.eval("DRAW_FRAME")(1, 0.05)
        top = draws(c)[1].top
        best = max(best, top.rot.p - rest_pitch)
        t0 = list(top.ops.values())[0]
        lifted = max(lifted, t0[2].z)
    assert best > 10, "the hat tips forward (degrees)"
    assert lifted > 0.04 * HU_PER_M * 0.5, "and lifts"
    c.rt.eval("DRAW_FRAME")(60, 0.05)
    assert abs(draws(c)[1].top.rot.p - rest_pitch) < 3, "and goes back on his head"


def test_the_free_hand_swings_for_a_conductor_and_never_beyond_its_limit():
    c = posed_client()
    c.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")("can,jacket,head,hat,hand_b")
    c.rt.eval("DRAW_FRAME")(2, 0.05)                      # (the first sight of a sequence number is only recorded)
    c.rt.execute("MOCK.bot:SetNWInt('mentat_av_cue', 3); MOCK.bot:SetNWInt('mentat_av_cue_seq', 1)")
    worst_roll, worst_yaw = 0, 0
    for _ in range(80):
        c.rt.eval("DRAW_FRAME")(1, 0.05)
        wm = draws(c)[1].top.parent                    # the matrix the hand's pointing turn sits under
        r = wm.rot
        worst_roll, worst_yaw = max(worst_roll, abs(r.r)), max(worst_yaw, abs(r.y))
        assert abs(r.r) <= 60 + 1e-6 and abs(r.y) <= 45 + 1e-6
    assert worst_roll > 30 and worst_yaw > 20


def test_the_voice_level_nods_the_head_and_brightens_the_tongue():
    c = posed_client()
    c.rt.execute("""
      local chan = { IsValid = function() return true end, GetState = function() return GMOD_CHANNEL_PLAYING end, GetLevel = function() return 1, 1 end,
                     SetPos = function() end, SetVolume = function() end }
      Mentat.Avatar.tracked[chan] = { base = 1 }
    """)
    c.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")("can,jacket,hat,hand_a,hand_b")
    pitches, mods = [], []
    for _ in range(60):
        c.rt.eval("DRAW_FRAME")(1, 0.02)
        pitches.append(draws(c)[1].top.rot.p)
    assert max(pitches) > 2.0 and max(pitches) <= 9.0 + 1e-6, "the head chatters with the voice, within its limit"
    assert max(pitches) - min(pitches) > 2.0


def test_the_body_tilts_about_his_middle_while_he_talks_in_an_opera_mood():
    c = posed_client()
    c.rt.execute("""
      local chan = { IsValid = function() return true end, GetState = function() return GMOD_CHANNEL_PLAYING end, GetLevel = function() return 0.8, 0.8 end,
                     SetPos = function() end, SetVolume = function() end }
      Mentat.Avatar.tracked[chan] = { base = 1 }
      MOCK.bot:SetNWString('mentat_av_mood', 'opera')
    """)
    worst = 0
    for _ in range(120):
        c.rt.eval("DRAW_FRAME")(1, 0.05)
        base = draws(c)[1].top
        worst = max(worst, abs(base.rot.r) if base.rot else 0)
    assert worst > 0.5, "the figure sways"


def test_a_late_joiner_does_not_replay_an_old_cue():
    c = posed_client()
    c.rt.execute("MOCK.bot:SetNWInt('mentat_av_cue', 3); MOCK.bot:SetNWInt('mentat_av_cue_seq', 41)")
    c.rt.eval("DRAW_FRAME")(2, 0.05)
    assert c.rt.eval("Mentat.Avatar.anim.cue") is None, "the first sight of a sequence number only records it"


def test_the_hush_message_stops_the_playing_clip_and_drops_one_in_flight():
    c = posed_client()
    c.rt.execute("""
      local chan = { IsValid = function() return true end, GetState = function() return GMOD_CHANNEL_PLAYING end, GetLevel = function() return 0.5, 0.5 end,
                     SetPos = function() end, SetVolume = function(self, v) self.vol = v end, Stop = function(self) self.stopped = true end }
      Mentat.Avatar.tracked[chan] = { base = 1 }
      HUSHED = chan
      MOCK.receivers['mentat_av_hush']()
    """)
    assert c.rt.eval("HUSHED.stopped") is True and c.rt.eval("next(Mentat.Avatar.tracked)") is None
    c.rt.execute("""
      local late = { IsValid = function() return true end, Stop = function(self) self.stopped = true end }
      LATE = late
      hook.Run('SkippySpoke', late)
    """)
    assert c.rt.eval("LATE.stopped") is True, "a clip that arrives just after the interruption is dropped"


def test_the_client_tells_the_server_when_its_player_starts_and_stops_talking():
    c = posed_client()
    c.rt.execute("MOCK.me.speaking = true")
    c.rt.eval("function() MOCK.advance(0.3) end")()
    c.rt.execute("MOCK.me.speaking = false")
    c.rt.eval("function() MOCK.advance(0.3) end")()
    sent = [m for m in c.rt.eval("MOCK.nets").values() if m.name == "mentat_av_talk"]
    states = [m.writes[1] for m in sent]
    assert states[0] is True and states[-1] is False and len(states) <= 3, states
