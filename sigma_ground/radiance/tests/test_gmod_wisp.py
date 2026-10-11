"""The failure wisp: offline.

When a found-object transport fails, Skippy's honest reason is left hanging as a small floating light at about eye height; the owner who
looks at it up close reads it, and after a dwell it dissolves; unread it fades by itself. What is pinned here:

  * the PURE module (mentat/fx/wisp.lua): colour by reason class, dwell scaling with the text, the state machine (reading time counts only
    while it is looked at up close), who can read (distance, view angle, a wall), replacement and the per-owner cap, text capping;
  * the SERVER glue: every failure path (not found, closest, a refusal from the game's own checks, a failed fetch, a timeout) leaves a wisp
    carrying the SAME sentence Skippy says, of the right class, at the right place and after the collapse has played; only the owner's
    looking reads it; it is not a placed thing (registry, undo, clear, limits never see it); "clear the wisps" dismisses them;
  * the CLIENT drawing against fakes of the render library: colour, owner-only text, the chat filter, far / behind draws nothing, a bounded
    number of sprites.
What these tests CANNOT show is how it looks: that was read off screenshots in the real game (stage/gmod/holodeck/README.md, "Failure wisp").
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

import skippy_bridge  # noqa: E402
from test_gmod_found import ALICE, BOB, Game, ix, no_network  # noqa: E402,F401
from skippyverse import index as IX  # noqa: E402
from test_gmod_found_mesh import MeshGame, cards as mesh_cards, fetch, summon  # noqa: E402,F401
from test_gmod_glue import World, lua_table  # noqa: E402
from test_gmod_transporter import FxGame  # noqa: E402


# ============================================================================================================ the pure module
def pure():
    w = World(cvars={}, load=(), server=False)
    return w, w.rt.eval("function() return dofile(MOCK.ROOT .. '/mentat/fx/wisp.lua') end")()


def run_state(w, W, st, steps, dt=0.1, looking=lambda i: False):
    states = []
    for i in range(steps):
        states.append(W.step(st, dt, looking(i)))
    return states


def test_the_colour_follows_the_reason_class_in_the_projects_palette():
    w, W = pure()
    assert tuple(W.colour("failure").values()) == (255, 170, 0), "a real failure is amber"
    assert tuple(W.colour("suggest").values()) == (0, 204, 255), "a neutral suggestion is cyan"
    assert tuple(W.colour("refusal").values()) == (255, 68, 68), "red is for refusals only"
    assert tuple(W.colour("ok").values()) == (0, 255, 136)
    assert tuple(W.colour("nonsense").values()) == (255, 170, 0), "an unknown class is a failure, never red"


@pytest.mark.parametrize("tag,expect", [
    ("not_found", "failure"), ("would_fetch", "failure"), ("fetch_failed", "failure"), ("needs_size", "failure"), ("", "failure"), (None, "failure"),
    ("closest", "suggest"), ("ambiguous", "suggest"), ("category", "suggest"),
    ("too_big", "refusal"), ("untrusted_size", "refusal"), ("not_shareable", "refusal"),
])
def test_the_resolvers_status_gives_the_class(tag, expect):
    w, W = pure()
    assert W.class_for(tag) == expect


def test_the_dwell_scales_with_the_text_and_has_a_floor():
    w, W = pure()
    assert W.dwell("short") == 2.5, "never less than 2.5 s"
    assert W.dwell("x" * 45) == 2.5
    assert W.dwell("x" * 90) == pytest.approx(5.0)
    assert W.dwell("x" * 160) == pytest.approx(160 / 18)
    assert W.dwell("x" * 160) > W.dwell("x" * 90) > W.dwell("x" * 20), "more words, more time"


def test_it_appears_then_idles_and_never_reads_itself():
    w, W = pure()
    st = W.new("A jumbo jet is bigger than this room (70 m). Try it outdoors.")
    states = run_state(w, W, st, 14)
    assert states[0] == "appear" and states[10] == "appear" and states[12] == "idle", states
    assert st["read"] == 0


def test_looking_during_the_appear_fade_does_not_count():
    w, W = pure()
    st = W.new("hi")
    run_state(w, W, st, 10, looking=lambda i: True)
    assert st["read"] == 0 and W.step(st, 0.0, True) == "appear"


def test_reading_time_accrues_only_while_it_is_looked_at_up_close():
    w, W = pure()
    st = W.new("x" * 90)                                   # dwell 5 s
    run_state(w, W, st, 13)                                # appear done
    for _ in range(20):                                    # 2 s looking
        W.step(st, 0.1, True)
    assert st["state"] == "reading" and st["read"] == pytest.approx(2.0, abs=1e-6)
    for _ in range(100):                                   # 10 s looking away: the clock does not move
        W.step(st, 0.1, False)
    assert st["state"] == "idle" and st["read"] == pytest.approx(2.0, abs=1e-6)
    for _ in range(31):                                    # 3.1 more seconds of looking
        W.step(st, 0.1, True)
    assert st["state"] == "dissolving" and st["why"] == "read"


def test_a_glance_away_does_not_fold_the_text_up_but_a_long_look_away_does():
    w, W = pure()
    st = W.new("x" * 90)
    run_state(w, W, st, 13)
    for _ in range(10):
        W.step(st, 0.1, True)
    for _ in range(2):                                     # 0.2 s away
        W.step(st, 0.1, False)
    assert st["state"] == "reading", "within the grace"
    for _ in range(4):
        W.step(st, 0.1, False)
    assert st["state"] == "idle"


def test_a_read_wisp_dissolves_for_a_second_and_is_gone():
    w, W = pure()
    st = W.new("hi")                                       # dwell 2.5
    run_state(w, W, st, 13)
    n = 0
    while st["state"] != "dissolving" and n < 100:
        W.step(st, 0.1, True); n += 1
    assert 24 <= n <= 26, "2.5 s of looking"
    for _ in range(9):
        W.step(st, 0.1, True)
    assert st["state"] == "dissolving"
    for _ in range(2):
        W.step(st, 0.1, True)
    assert st["state"] == "gone"
    assert W.step(st, 0.1, True) == "gone", "gone is final"


def test_a_wisp_nobody_read_fades_by_itself_after_its_life():
    w, W = pure()
    st = W.new("hi", {"life": 30})
    run_state(w, W, st, 290)
    assert st["state"] == "idle"
    run_state(w, W, st, 20)
    assert st["state"] == "dissolving" and st["why"] == "expired"
    run_state(w, W, st, 10)                                # 2.0 s into the fade
    assert st["state"] == "dissolving", "the slow fade takes 2.5 s"
    run_state(w, W, st, 10)
    assert st["state"] == "gone"


def test_the_default_life_is_about_two_minutes():
    w, W = pure()
    assert 100 <= W.LIFE_S <= 140


def test_dismiss_dissolves_at_once_and_does_nothing_twice():
    w, W = pure()
    st = W.new("hi")
    run_state(w, W, st, 20)
    W.dismiss(st)
    assert st["state"] == "dissolving" and st["why"] == "dismissed"
    run_state(w, W, st, 5)
    W.dismiss(st)
    assert st["since"] == pytest.approx(0.5), "a second dismissal does not restart the dissolve"


def v(x, y, z):
    return {"x": x, "y": y, "z": z}


def test_only_a_close_view_toward_it_without_a_wall_reads():
    w, W = pure()
    eye, fwd, pos = v(0, 0, 64), v(1, 0, 0), v(150, 0, 64)
    assert W.sees(eye, fwd, pos, True)[0] is True
    assert W.sees(eye, fwd, v(250, 0, 64), True)[0] is False, "farther than 4 m (210 HU)"
    assert W.sees(eye, v(0, 1, 0), pos, True)[0] is False, "looking away"
    assert W.sees(eye, v(-1, 0, 0), pos, True)[0] is False, "looking the other way"
    assert W.sees(eye, fwd, pos, False)[0] is False, "a wall between"
    assert W.sees(eye, v(0.9205, 0.3907, 0), pos, True)[0] is True, "23 degrees off: still toward it"
    assert W.sees(eye, v(0.7071, 0.7071, 0), pos, True)[0] is False, "45 degrees off: not reading it"


def entry(owner, x, born, y=0, z=64):
    return {"owner": owner, "pos": v(x, y, z), "born": born}


def test_a_new_wisp_close_to_an_old_one_of_the_same_owner_replaces_it():
    w, W = pure()
    lst = lua_table(w.rt, [entry("a", 0, 1), entry("a", 500, 2), entry("b", 10, 3)])
    gone = sorted(W.victims(lst, "a", lua_table(w.rt, v(50, 0, 64))).values())
    assert gone == [1], "only the near one of the same owner (1.5 m = 79 HU)"
    assert sorted(W.victims(lst, "a", lua_table(w.rt, v(200, 0, 64))).values()) == [], "farther than 1.5 m: both stay"


def test_the_cap_removes_the_oldest_of_that_owner_first_and_leaves_others_alone():
    w, W = pure()
    rows = [entry("a", 0, 5), entry("a", 300, 1), entry("b", 600, 0), entry("a", 900, 3)]
    far = lua_table(w.rt, v(5000, 0, 64))
    gone = sorted(W.victims(lua_table(w.rt, rows), "a", far, 3).values())
    assert gone == [2], "three of a's would be four with the new one: the oldest (born 1, index 2) goes"
    rows.append(entry("a", 1200, 4))
    assert sorted(W.victims(lua_table(w.rt, rows), "a", far, 3).values()) == [2, 4], "two over the cap: the two oldest"
    assert len(W.victims(lua_table(w.rt, rows), "c", far, 3)) == 0


def test_the_text_is_one_clean_line_capped_at_160_and_never_cut_inside_a_character():
    w, W = pure()
    assert W.cap_text("a\nb\t c\x07  d") == "a b c d"
    long = W.cap_text("word " * 100)
    assert len(long) <= 160 and long.endswith("word..."), "cut at a word, and it says it was cut"
    sentence = "I read 'cannnon' as 'cannon'. If that's wrong, say so. A photoscan-or-pbr model by Yann Kervran, James Ray Cock, CC0; size from a published source; it shows as a box for now."
    cut = W.cap_text(sentence)
    assert len(cut) <= 160 and cut.endswith("...") and sentence.startswith(cut[:-3]), "an honest prefix of what was said"
    assert W.cap_text("x" * 160) == "x" * 160 and W.cap_text("short") == "short", "within the cap nothing changes"
    ok = w.rt.eval("""function(W)
      local e = string.char(195, 169)                                    -- 2 bytes each
      local cut = W.cap_text(string.rep(e, 100))
      local body = cut:sub(1, -4)
      if #cut > 160 or cut:sub(-3) ~= "..." or body ~= string.rep(e, #body / 2) then return "whole: " .. #cut end
      cut = W.cap_text(string.rep("a", 157) .. e .. "bcd")               -- the e-acute would straddle the cut
      if #cut > 160 or cut:sub(-3) ~= "..." or cut:sub(1, -4):find("[^a]") then return "straddle: " .. #cut end
      return "ok"
    end""")(W)
    assert ok == "ok", ok
    assert W.cap_text(None) == "" and W.cap_text("   ") == ""


def test_the_text_wraps_to_three_or_four_short_lines():
    w, W = pure()
    for text in ("I don't have a dragon on file and couldn't make one easily.",
                 "A jumbo jet is bigger than this room (70 m). Try it outdoors.",
                 "The model was too big to fetch in time.", "The closest I have is a gargoyle.",
                 "I could not bring in the garden gnome: the bridge could not fetch it. I took the placeholder down."):
        lines, widest = W.wrap(text, 4, 40)
        lines = list(lines.values())
        assert 1 <= len(lines) <= 4 and widest <= 40, (text, lines)
        assert " ".join(lines) == text, "nothing lost, nothing added"
        assert all(len(l) <= 40 for l in lines)
    lines, _ = W.wrap("I could not bring in the garden gnome: the bridge could not fetch it. I took the placeholder down.", 4, 40)
    assert len(lines) <= 4
    lines, _ = W.wrap("word " * 80, 4, 40)
    lines = list(lines.values())
    assert len(lines) == 4 and lines[-1].endswith("..."), "text that still does not fit ends in an ellipsis"


def test_other_players_see_a_dim_light_and_no_text_and_the_owner_the_full_picture():
    w, W = pure()
    mine = W.look("reading", 1.0, 1.0, True, None)
    other = W.look("reading", 1.0, 1.0, False, None)
    idle = W.look("idle", 5, 0, True, None)
    assert mine.text > 0.9 and other.text == 0
    assert other.orb < mine.orb * 0.5, "dim"
    assert mine.orb > idle.orb, "it brightens when read"


def test_the_dissolve_turns_the_text_to_sparks_then_nothing():
    w, W = pure()
    a, b, c = (W.look("dissolving", t, 1.0, True, "read") for t in (0.05, 0.5, 0.99))
    assert a.text > b.text > c.text and a.sparks < b.sparks < c.sparks
    assert c.orb < 0.2 and a.orb > 0.8
    exp = W.look("dissolving", 1.2, 0, True, "expired")
    assert exp.sparks == 0 and exp.text == 0 and 0 < exp.orb < 0.55, "an unread wisp just dims"


# ============================================================================================================ the server glue
def load_wisp(game):
    game.w.rt.execute('MOCK.load_entity("holodeck_wisp", "server")')
    game.w.rt.execute(WISP_HELPERS)
    return game


WISP_HELPERS = r"""
function T.wisps()
  local out = {}
  for _, e in ipairs(MOCK.ents) do
    if e.class == "holodeck_wisp" and e.valid then
      local o = e:GetOwnerPly()
      out[#out + 1] = { text = e:GetReason(), class = e:WispClass(), state = e:WispState(), why = e:WispWhy(), read = e:GetRead(), dwell = e:GetDwell(),
                        x = e.pos.x, y = e.pos.y, z = e.pos.z, owner = o and o.sid or nil, solid = e.solid, movetype = e.movetype, idx = e:EntIndex() }
    end
  end
  return MOCK.json_encode(out)
end
function T.say_log() return MOCK.json_encode(Holodeck.log) end
function T.face(p, x, y, z, yaw) p.pos = Vector(x, y, z); p.yaw = yaw end
function T.registry_n() return #Holodeck.order end
"""


def wisps(game):
    return json.loads(game.w.call("T.wisps"))


def stand_and_look(game, p, wx, wy, wz, dist=100.0):
    """Put the player `dist` HU from the wisp, eyes level with it (64 HU up), looking straight at it."""
    game.w.call("T.face", p, wx - dist, wy, wz - 64, 0)


@pytest.fixture()
def wgame(ix):
    return load_wisp(Game())


@pytest.fixture()
def wfx(tmp_path, monkeypatch, fetch):
    """The mesh tests' world (a garden gnome with a real shape): its transport can fail."""
    IX.build_index(mesh_cards(), tmp_path / "ixm", synonym_groups=[])
    monkeypatch.setattr(skippy_bridge, "FOUND_INDEX", IX.load_index(tmp_path / "ixm", shared=True))
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", None)
    monkeypatch.setattr(skippy_bridge, "FOUND_FETCH", False)
    monkeypatch.setattr(skippy_bridge, "GATE", skippy_bridge.Gate())
    monkeypatch.setattr(skippy_bridge, "MESH_PROGRESS", {})
    return load_wisp(FxGame())


def test_a_noun_nobody_has_leaves_a_wisp_with_skippys_own_sentence_in_front_of_the_player(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.w.call("T.face", p, 0, 0, 0, 90)                                  # at the origin, facing +y
    wgame.ask(p, "add a dragon")
    assert wisps(wgame) == [], "nothing until the resolver has answered"
    wgame.serve()
    (wisp,) = wisps(wgame)
    said = wgame.say()
    assert "dragon" in wisp["text"] and wisp["text"] == said, "the wisp carries the sentence Skippy said, not a made-up one"
    assert wisp["class"] == "failure" and wisp["owner"] == ALICE and wisp["state"] == "appear"
    assert abs(wisp["x"]) < 1 and 80 < wisp["y"] < 130, "about an arm's length ahead, where it was looking"
    assert 45 < wisp["z"] < 70, "about eye height"
    assert wisp["solid"] in (0, None) and wisp["movetype"] in (0, None), "no collision, no physics"


def test_the_closest_i_have_is_a_neutral_cyan_suggestion(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a wheelbarow")
    wgame.serve()
    (wisp,) = wisps(wgame)
    assert "wheelbarrow" in wisp["text"] and wisp["class"] == "suggest"


@pytest.mark.parametrize("patch,expect", [
    ({"w": 4000.0, "d": 4000.0, "h": 40.0}, "bigger"),
    ({"w": 0.2, "d": 0.2, "h": 0.2}, "too small"),
])
def test_the_games_own_refusals_are_red(wgame, patch, expect):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a wheelbarrow")
    bad = json.loads(json.dumps(skippy_bridge.resolve_found("wheelbarrow")))
    bad["kind"]["variants"][0].update(patch)
    wgame.serve(lambda noun: bad)
    (wisp,) = wisps(wgame)
    assert expect in wisp["text"] and wisp["class"] == "refusal"


def test_the_bridges_too_big_status_is_a_refusal_in_the_game_too(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a jumbo jet")
    wgame.serve(lambda noun: {"ok": True, "noun": noun, "place": False, "status": "too_big", "kind": None, "card": None,
                              "say": "A jumbo jet is bigger than this room (70 m). Try it outdoors."})
    (wisp,) = wisps(wgame)
    assert wisp["text"] == "A jumbo jet is bigger than this room (70 m). Try it outdoors." and wisp["class"] == "refusal"


def test_a_model_i_may_not_share_is_a_red_refusal_even_though_the_resolver_matched_it(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a valve widget")
    wgame.serve()
    (wisp,) = wisps(wgame)
    assert "may not share" in wisp["text"] and wisp["class"] == "refusal", wisp


def test_a_remembered_no_still_leaves_its_class(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a wheelbarow")
    wgame.serve()
    wgame.w.call("T.face", p, 5000, 0, 0, 0)                                # (moved on: the next wisp is not "near" the first)
    wgame.ask(p, "add a wheelbarow")                                        # answered from the cache, no second bridge call
    ws = wisps(wgame)
    assert len(ws) == 2 and ws[1]["class"] == "suggest"


def test_nothing_is_left_when_the_parser_merely_guessed_or_the_bridge_is_down(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a fire dragon")                                       # "fire" is approximate: a non-strict ask, nothing said
    wgame.serve()
    assert wisps(wgame) == []
    wgame.ask(p, "add a wheelbarrow")
    (i,) = [x for x in wgame.posts() if x not in wgame.answered]
    wgame.w.rt.eval("function(i) MOCK.respond(i, 'fail', 'timeout') end")(i)
    assert wisps(wgame) == [], "no sentence from the resolver, no wisp"


def test_a_failed_fetch_leaves_the_wisp_after_the_collapse_with_the_same_sentence(wfx):
    summon(wfx)
    wfx.serve_mesh(lambda b: (502, {"ok": False, "error": "boom"}))
    assert wfx.fx()[0]["stage"] == "failed"
    assert wisps(wfx) == [], "not while the collapse is playing"
    wfx.w.advance(1.0)
    assert wisps(wfx) == []
    wfx.w.advance(0.8)
    (wisp,) = wisps(wfx)
    assert wisp["text"] == wfx.say_fx() and wisp["text"].startswith("I could not bring in the garden gnome")
    assert wisp["class"] == "failure" and wisp["owner"] == ALICE
    assert wfx.fx() == [], "the placeholder is gone by then"
    assert 50 < wisp["z"] < 70, "eye height above where the object stood"
    assert wfx.count() == 0, "the registry knows nothing of it"


def test_a_timeout_leaves_an_amber_wisp(wfx):
    summon(wfx)
    wfx.w.advance(67)
    wfx.w.advance(4)
    (wisp,) = wisps(wfx)
    assert "too long" in wisp["text"] and wisp["class"] == "failure"


def test_three_objects_that_fail_together_leave_one_wisp_with_the_one_sentence(wfx):
    summon(wfx, "add three garden gnomes")
    wfx.serve_mesh(lambda b: (502, {"ok": False}))
    wfx.w.advance(2.5)
    (wisp,) = wisps(wfx)
    assert wisp["text"].startswith("I could not bring in 3 of the garden gnome")


def test_the_owner_reads_it_and_it_notices(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a dragon")
    wgame.serve()
    (wisp,) = wisps(wgame)
    stand_and_look(wgame, p, wisp["x"], wisp["y"], wisp["z"])
    wgame.w.advance(1.5)                                                    # the appear fade
    assert wisps(wgame)[0]["state"] in ("appear", "idle", "reading")
    wgame.w.advance(0.5)
    assert wisps(wgame)[0]["state"] == "reading"
    dwell = wisps(wgame)[0]["dwell"]
    assert dwell == pytest.approx(max(2.5, len(wisp["text"]) / 18))
    wgame.w.advance(dwell - wisps(wgame)[0]["read"] - 0.6)
    assert wisps(wgame)[0]["state"] == "reading", "not yet"
    wgame.w.advance(1.0)
    w2 = wisps(wgame)
    assert w2 and w2[0]["state"] == "dissolving" and w2[0]["why"] == "read"
    wgame.w.advance(1.3)
    assert wisps(wgame) == [], "gone after the dissolve"


def test_standing_too_far_or_looking_away_never_reads_it(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a dragon")
    wgame.serve()
    (wisp,) = wisps(wgame)
    stand_and_look(wgame, p, wisp["x"], wisp["y"], wisp["z"], dist=400)
    wgame.w.advance(20)
    far = wisps(wgame)[0]
    assert far["state"] == "idle" and far["read"] == 0, "5 m away is not up close"
    wgame.w.call("T.face", p, wisp["x"] - 100, wisp["y"], wisp["z"] - 64, 180)
    wgame.w.advance(20)
    away = wisps(wgame)[0]
    assert away["state"] == "idle" and away["read"] == 0, "close, but looking the other way"


def test_a_wall_between_blocks_the_reading(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a dragon")
    wgame.serve()
    (wisp,) = wisps(wgame)
    stand_and_look(wgame, p, wisp["x"], wisp["y"], wisp["z"])
    wgame.w.rt.execute("""
      real_trace = util.TraceLine
      util.TraceLine = function(t) if t.mask == MASK_SOLID_BRUSHONLY and t.start.z > 0 then return { Hit = true, Fraction = 0.5, HitPos = t.start } end return real_trace(t) end
    """)
    wgame.w.advance(20)
    assert wisps(wgame)[0]["read"] == 0
    wgame.w.rt.execute("util.TraceLine = real_trace")
    wgame.w.advance(1.0)
    assert wisps(wgame)[0]["read"] > 0, "and when the wall is gone it counts"


def test_only_the_owner_can_dismiss_it_by_reading(wgame):
    alice = wgame.player(ALICE, admin=True)
    bob = wgame.player(BOB, admin=True)
    wgame.ask(alice, "add a dragon")
    wgame.serve()
    (wisp,) = wisps(wgame)
    wgame.w.call("T.face", alice, wisp["x"] + 3000, wisp["y"], 0, 0)       # Alice walks away
    stand_and_look(wgame, bob, wisp["x"], wisp["y"], wisp["z"])             # Bob stands right there looking straight at it
    wgame.w.advance(30)
    w2 = wisps(wgame)[0]
    assert w2["state"] == "idle" and w2["read"] == 0, "Bob's gaze is not Alice's reading"
    stand_and_look(wgame, alice, wisp["x"], wisp["y"], wisp["z"])
    wgame.w.advance(wisp_dwell := 9)
    assert wisps(wgame) == [] or wisps(wgame)[0]["state"] == "dissolving"


def test_a_wisp_nobody_reads_fades_after_its_life_which_is_a_convar(wgame):
    wgame.w.rt.eval("function() GetConVar('holodeck_wisp_life'):SetValue(20) end")()
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a dragon")
    wgame.serve()
    wgame.w.call("T.face", p, 5000, 5000, 0, 0)
    wgame.w.advance(15)
    assert wisps(wgame)[0]["state"] == "idle"
    wgame.w.advance(7)
    assert wisps(wgame)[0]["state"] == "dissolving" and wisps(wgame)[0]["why"] == "expired"
    wgame.w.advance(3)
    assert wisps(wgame) == []


def drop(game, p, text, x):
    return game.w.rt.eval("function(p, t, x) return (Holodeck.Wisps.drop(p, t, 'failure', Vector(x, 0, 64))) end")(p, text, x)


def test_a_new_failure_close_to_an_old_wisp_replaces_it(wgame):
    p = wgame.player(ALICE, admin=True)
    drop(wgame, p, "first", 0)
    drop(wgame, p, "second", 60)                                            # 60 HU = 1.1 m: replaces
    wgame.w.advance(1.5)
    ws = wisps(wgame)
    assert [w["text"] for w in ws] == ["second"], ws


def test_the_per_player_cap_removes_the_oldest_first_and_other_players_keep_theirs(wgame):
    a, b = wgame.player(ALICE, admin=True), wgame.player(BOB, admin=True)
    drop(wgame, b, "bob's", 5000)
    for i, x in enumerate((0, 500, 1000, 1500)):
        drop(wgame, a, f"a{i}", x)
        wgame.w.advance(0.5)
    wgame.w.advance(1.5)
    texts = sorted(w["text"] for w in wisps(wgame))
    assert texts == ["a1", "a2", "a3", "bob's"], texts


def test_the_cap_is_a_convar(wgame):
    wgame.w.rt.eval("function() GetConVar('holodeck_wisp_max'):SetValue(1) end")()
    a = wgame.player(ALICE, admin=True)
    drop(wgame, a, "x", 0)
    wgame.w.advance(0.5)
    drop(wgame, a, "y", 900)
    wgame.w.advance(1.5)
    assert [w["text"] for w in wisps(wgame)] == ["y"]


def test_the_server_wide_cap_is_bounded_too(wgame):
    ps = [wgame.player(str(76561190000000100 + i), admin=True) for i in range(30)]
    for i, p in enumerate(ps):
        drop(wgame, p, f"w{i}", i * 400)
        wgame.w.advance(0.2)
    wgame.w.advance(1.5)
    assert len(wisps(wgame)) <= 24


def test_wisps_are_not_placed_things_and_undo_and_clear_do_not_touch_them(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a dragon")
    wgame.serve()
    wgame.w.call("T.face", p, 0, 0, 0, 180)                                 # (looking away: reading it would dismiss it)
    assert wgame.count() == 0 and wgame.w.call("T.registry_n") == 0, "not in the registry: not in any limit"
    wgame.ask(p, "add two couches")
    wgame.settle()
    assert wgame.count() == 2
    wgame.ask(p, "clear everything")
    wgame.settle()
    assert wgame.count() == 0
    assert len(wisps(wgame)) == 1, "clear everything is about things, not wisps"
    wgame.ask(p, "undo")
    wgame.settle()
    assert wgame.count() == 2 and len(wisps(wgame)) == 1
    wgame.ask(p, "remove the couches")
    wgame.settle()
    assert wgame.count() == 0 and len(wisps(wgame)) == 1


def test_a_limit_of_one_thing_does_not_count_the_wisps(ix):
    g = load_wisp(Game(holodeck_max_per_player="1")) if False else load_wisp(Game())
    p = g.player(ALICE)
    for i in range(3):
        drop(g, p, f"w{i}", i * 500)
    g.w.rt.eval("function() GetConVar('holodeck_max_per_player'):SetValue(2) end")()
    g.ask(p, "add two couches")
    g.settle()
    assert g.count() == 2, "three wisps stand and the player's limit of two is still all theirs"


@pytest.mark.parametrize("phrase", [
    "clear the wisps", "clean up the notes", "Skippy, dismiss the wisps", "clear all wisps please", "get rid of the whisps", "remove the floating notes",
    "dismiss all the notes.",
])
def test_the_clear_command_dismisses_the_wisps(wgame, phrase):
    p = wgame.player(ALICE, admin=True)
    drop(wgame, p, "one", 0)
    drop(wgame, p, "two", 900)
    wgame.w.advance(0.5)
    wgame.ask(p, phrase)
    assert all(w["state"] == "dissolving" for w in wisps(wgame)), phrase
    wgame.w.advance(1.5)
    assert wisps(wgame) == []
    assert "2 wisps" in wgame.w.rt.eval("Holodeck.last_tell")


@pytest.mark.parametrize("phrase", ["clear the couches", "add a note", "remove the lamp", "clear everything", "make a wisp", "clean up the wisps and add a couch"])
def test_other_sentences_are_not_the_wisp_command(wgame, phrase):
    w, W = pure()
    assert wgame.w.rt.eval("function(t) return Holodeck.Wisps.is_clear_command(t) end")(phrase) is False


def test_one_players_clear_leaves_anothers_alone_and_the_console_command_works(wgame):
    a, b = wgame.player(ALICE, admin=True), wgame.player(BOB, admin=True)
    drop(wgame, a, "alice", 0)
    drop(wgame, b, "bob", 900)
    wgame.w.advance(0.5)
    wgame.ask(a, "clear the wisps")
    wgame.w.advance(1.5)
    assert [w["text"] for w in wisps(wgame)] == ["bob"]
    wgame.w.rt.execute("MOCK.cmds['mentat_wisps_clear'](MOCK.players[2], 'mentat_wisps_clear', {}, '')")
    wgame.w.advance(1.5)
    assert wisps(wgame) == []


def test_the_stored_text_is_capped_and_cleaned(wgame):
    p = wgame.player(ALICE, admin=True)
    drop(wgame, p, "line one\nline two\t" + "word " * 100, 0)
    (wisp,) = wisps(wgame)
    assert len(wisp["text"]) <= 160 and "\n" not in wisp["text"] and "\t" not in wisp["text"]
    assert wisp["text"].startswith("line one line two word")


def test_nothing_is_made_without_text_and_wisps_can_be_switched_off(wgame):
    p = wgame.player(ALICE, admin=True)
    assert drop(wgame, p, "   ", 0) is None
    wgame.w.rt.eval("function() GetConVar('holodeck_wisps'):SetValue(0) end")()
    assert drop(wgame, p, "hello", 0) is None
    assert wisps(wgame) == []


def test_when_the_owner_leaves_their_wisps_go(wgame):
    a, b = wgame.player(ALICE, admin=True), wgame.player(BOB, admin=True)
    drop(wgame, a, "alice", 0)
    drop(wgame, b, "bob", 900)
    wgame.w.rt.eval("function(p) MOCK.disconnect(p) end")(a)
    assert [w["text"] for w in wisps(wgame)] == ["bob"]


def test_the_state_machine_timer_stops_when_there_is_nothing_left(wgame):
    p = wgame.player(ALICE, admin=True)
    drop(wgame, p, "x", 0)
    assert wgame.w.rt.eval("function() return timer.Exists('holodeck_wisps') end")() is True
    wgame.ask(p, "clear the wisps")
    wgame.w.advance(3)
    assert wgame.w.rt.eval("function() return timer.Exists('holodeck_wisps') end")() is False


def test_the_wisp_carries_its_state_in_networked_values_so_a_late_joiner_sees_the_same(wgame):
    p = wgame.player(ALICE, admin=True)
    wgame.ask(p, "add a dragon")
    wgame.serve()
    (wisp,) = wisps(wgame)
    got = wgame.w.rt.eval("""function()
      for _, e in ipairs(MOCK.ents) do
        if e.class == "holodeck_wisp" then
          local names = {}
          for k in pairs(e.nvars) do names[#names + 1] = k end
          table.sort(names)
          return table.concat(names, ",")
        end
      end
    end""")()
    assert got == "Dwell,OwnerPly,Phase,Read,Reason,ReasonClass,StateT,Why,WispStateIndex"


# ============================================================================================================ the client drawing
CLIENT_WISP_FAKES = r"""
DRAW = { sprites = {}, texts = {}, rects = {}, cam3d = 0, fonts = {} }
render = render or {}
render.SetMaterial = function() end
render.DrawSprite = function(pos, w, h, col) DRAW.sprites[#DRAW.sprites + 1] = { x = pos.x, y = pos.y, z = pos.z, w = w, r = col.r, g = col.g, b = col.b, a = col.a } end
cam = { Start3D2D = function(pos, ang, s) DRAW.cam3d = DRAW.cam3d + 1; DRAW.last_cam = { pos = pos, ang = ang, scale = s } end, End3D2D = function() end }
surface = { CreateFont = function(n, t) DRAW.fonts[n] = t end, SetFont = function() end, GetTextSize = function(t) return #t * 22, 44 end,
            SetDrawColor = function() end, DrawRect = function(...) DRAW.rects[#DRAW.rects + 1] = { ... } end }
draw = { SimpleText = function(t, f, x, y, col) DRAW.texts[#DRAW.texts + 1] = { t = t, a = col.a } end }
TEXT_ALIGN_LEFT, TEXT_ALIGN_TOP = 0, 0
function CreateMaterial() return {} end
function GetRenderTarget() return { GetName = function() return "rt" end } end
FrameNumber = function() DRAW.frame = (DRAW.frame or 0) + 1; return DRAW.frame end
function LocalPlayer() return MOCK.me end
function T.draw_reset() DRAW.sprites, DRAW.texts, DRAW.rects, DRAW.cam3d = {}, {}, {}, 0 end
function T.draw(e, dt)
  MOCK.advance(dt or 0)
  T.draw_reset()
  e:DrawTranslucent()
  local r, g, b, a = 0, 0, 0, 0
  for _, s in ipairs(DRAW.sprites) do r, g, b, a = r + s.r * s.a, g + s.g * s.a, b + s.b * s.a, a + s.a end
  local texts = {}
  for _, t in ipairs(DRAW.texts) do texts[#texts + 1] = t.t end
  return MOCK.json_encode({ n = #DRAW.sprites, texts = texts, cam3d = DRAW.cam3d, r = a > 0 and r / a or 0, g = a > 0 and g / a or 0, b = a > 0 and b / a or 0,
                            amax = (function() local m = 0 for _, s in ipairs(DRAW.sprites) do if s.a > m then m = s.a end end return m end)() })
end
"""


def wisp_client(text="I don't have a dragon on file and couldn't make one easily.", cls="failure", state="reading", why=None, owner=True, eye=(0, 0, 64), yaw=0):
    w = World(cvars={}, load=(), server=False)
    w.rt.execute(CLIENT_WISP_FAKES)
    w.rt.execute("Mentat = Mentat or {}")
    w.advance(50.0)
    w.rt.execute('MOCK.load_entity("holodeck_wisp", "client")')
    me, other = w.rt.eval("""function(eye, yaw)
      MOCK.me = MOCK.newplayer('1', { pos = Vector(eye[1], eye[2], eye[3] - 64), yaw = yaw })
      local other = MOCK.newplayer('2', { pos = Vector(0, 0, 0) })
      return MOCK.me, other
    end""")(w.rt.table_from(list(eye)), yaw)
    e = w.rt.eval("""function(text, cls, state, why, owner, me, other)
      local e = ents.Create("holodeck_wisp")
      e:SetReason(text)
      e:SetReasonClass(({ failure = 1, suggest = 2, refusal = 3, ok = 4 })[cls])
      e:SetWispStateIndex(({ appear = 1, idle = 2, reading = 3, dissolving = 4, gone = 5 })[state])
      e:SetWhy(({ read = 1, expired = 2, dismissed = 3 })[why] or 0)
      e:SetStateT(CurTime() - 0.0)
      e:SetOwnerPly(owner and me or other)
      e:SetPos(Vector(100, 0, 64))
      e.SetRenderBounds = function() end
      e:Initialize()
      return e
    end""")(text, cls, state, why, owner, me, other)
    return w, e


def draw_frames(w, e, n=12, dt=0.1):
    res = None
    for _ in range(n):
        res = json.loads(w.call("T.draw", e, dt))
    return res


def test_the_owner_reading_it_sees_the_whole_sentence_typed_out_on_a_plate_facing_them():
    w, e = wisp_client()
    r = draw_frames(w, e, 14)
    assert r["cam3d"] == 1, "one 3D2D block"
    assert "".join(r["texts"]) == "I don't have a dragon on file and couldn't make one easily.".replace("  ", " ") or " ".join(r["texts"]) == \
        "I don't have a dragon on file and couldn't make one easily."
    cam = w.rt.eval("DRAW.last_cam")
    assert abs(cam.ang.y - (-90)) < 1.0, "yaw toward the reader: they look along +x, so the text faces -x"
    assert cam.ang.r == 90, "upright"


def test_the_text_unfurls_over_time_it_is_not_there_at_once():
    w, e = wisp_client()
    first = json.loads(w.call("T.draw", e, 0.05))
    assert "".join(first["texts"]) == "" or len("".join(first["texts"])) < 20
    last = draw_frames(w, e, 15)
    assert len("".join(last["texts"])) > len("".join(first["texts"]))


def test_someone_else_sees_the_light_dimly_and_no_text():
    w, mine = wisp_client(owner=True)
    w2, other = wisp_client(owner=False)
    a, b = draw_frames(w, mine), draw_frames(w2, other)
    assert b["texts"] == [] and b["cam3d"] == 0
    assert b["amax"] < a["amax"] * 0.6, "dimmer"


def test_the_idle_light_is_drawn_without_text_and_the_unread_ones_class_sets_the_colour():
    cols = {}
    for cls in ("failure", "suggest", "refusal"):
        w, e = wisp_client(cls=cls, state="idle")
        r = draw_frames(w, e, 6)
        assert r["texts"] == [] and r["cam3d"] == 0
        cols[cls] = (r["r"], r["g"], r["b"])
    assert cols["failure"][0] > cols["failure"][2] and cols["failure"][1] > 100, "amber: red and green high, blue low"
    assert cols["suggest"][2] > cols["suggest"][0], "cyan: blue and green over red"
    assert cols["refusal"][0] > cols["refusal"][1] * 1.5 and cols["refusal"][0] > cols["refusal"][2] * 1.5, "red: green and blue low"
    assert cols["failure"][1] > cols["refusal"][1] * 1.2, "amber has more green than red does"


def test_the_chat_filter_applies_to_the_words_on_this_viewers_screen():
    w, e = wisp_client(text="This is a dragon thing")
    w.rt.execute("Mentat.FilterText = function(s) return (s:gsub('dragon', '******')) end")
    r = draw_frames(w, e, 15)
    shown = " ".join(r["texts"])
    assert "******" in shown and "dragon" not in shown, "the server's words are never shown unfiltered"


def test_holodeck_client_exports_the_same_filter_it_uses_for_chat():
    w = World(cvars={}, load=(), server=False)
    w.rt.execute('util = { FilterText = function(s, mode) return (s:gsub("bad", "***")) end }; net = { Receive = function() end }; chat = { AddText = function() end }')
    w.rt.execute('Mentat = {}; dofile(MOCK.ROOT .. "/mentat/holodeck_client.lua")')
    assert w.rt.eval("Mentat.FilterText('a bad word')") == "a *** word"


def test_far_away_and_behind_the_viewer_nothing_is_drawn():
    w, e = wisp_client()
    assert draw_frames(w, e, 3)["n"] > 0
    far = w.rt.eval("function(e) e:SetPos(Vector(2000, 0, 64)) end")(e)
    assert json.loads(w.call("T.draw", e, 0.1))["n"] == 0, "past 25 m (1312 HU)"
    w.rt.eval("function(e) e:SetPos(Vector(-300, 0, 64)) end")(e)
    assert json.loads(w.call("T.draw", e, 0.1))["n"] == 0, "behind the viewer"


def test_the_number_of_sprites_is_small_and_bounded():
    w, e = wisp_client()
    counts = [json.loads(w.call("T.draw", e, 0.1))["n"] for _ in range(30)]
    assert max(counts) <= 12, counts
    w, e = wisp_client(state="idle")
    assert max(json.loads(w.call("T.draw", e, 0.1))["n"] for _ in range(30)) <= 12
    w, e = wisp_client(state="dissolving", why="read")
    w.rt.eval("function(e) e.unfurl = 1 end")(e)
    assert max(json.loads(w.call("T.draw", e, 0.04))["n"] for _ in range(25)) <= 12 + 26


def test_the_dissolve_after_a_read_throws_sparks_and_a_gone_wisp_draws_nothing():
    w, e = wisp_client(state="dissolving", why="read")
    w.rt.eval("function(e) e.unfurl = 1; e:SetStateT(CurTime() - 0.3) end")(e)
    r = json.loads(w.call("T.draw", e, 0.0))
    assert r["n"] > 12 + 10, "the sparks of the dissolving text"
    w2, e2 = wisp_client(state="gone")
    assert json.loads(w2.call("T.draw", e2, 0.1))["n"] == 0


def test_it_draws_nothing_that_needs_game_content():
    src = (HERE.parents[2] / "stage" / "gmod" / "addon" / "lua" / "entities" / "holodeck_wisp" / "cl_init.lua").read_text(encoding="utf-8")
    for bad in (".mdl", "sound.", "EmitSound", "PrecacheModel", "models/", "sprites/", "effects/", "particles/", "materials/"):
        assert bad not in src, bad
    assert 'color/white' in src and 'mentat_wisp_glow' in src, "only the engine's own white texture and a render target painted here"
