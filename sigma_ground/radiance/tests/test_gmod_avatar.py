"""Skippy's in-world avatar (avatar.lua, avatar_client.lua, avatar/logic.lua, avatar/geometry.lua), offline.

The REAL addon files run under LuaJIT (lupa) against stage/gmod/holodeck/gmod_mock.lua, extended here with the few things a bot
player needs (player.CreateNextBot and the Player methods the avatar uses). What this proves is the logic: where he appears, that he
stays out of walls, follows and snaps, which words call him, that his voice carries exactly as far as proximity voice says, that the
bot's own AI is switched off, and that the geometry is sound. What it cannot prove (bots, 3D sound and the look of the mesh in the
real engine) is listed in stage/gmod/AVATAR.md.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from test_gmod_glue import ADDON, World  # noqa: E402

A, B, C = "76561190000000001", "76561190000000002", "76561190000000003"

PRELUDE = r"""
MOCK.bots = {}
do
  local probe = MOCK.newplayer("probe")
  local Ply = getmetatable(probe)
  probe.valid = false
  function Ply:SetPos(v) self.pos = v end
  function Ply:SetEyeAngles(a) self.yaw = a.y; self.eyeset = a end
  function Ply:SetAngles(a) self.angset = a end
  function Ply:StripWeapons() self.stripped = true end
  function Ply:GodEnable() self.god = true end
  function Ply:SetMoveType(m) self.movetype = m end
  function Ply:SetNotSolid(b) self.notsolid = b end
  function Ply:SetCollisionGroup(g) self.group = g end
  function Ply:SetNoTarget(b) self.notarget = b end
  function Ply:SetNoDraw(b) self.nodraw = b end
  function Ply:IsBot() return self.bot == true end
  function Ply:Kick(r) self.kicked = r; MOCK.disconnect(self) end
  function Ply:SetNWEntity(k, v) self.nwe = self.nwe or {}; self.nwe[k] = v end
  function Ply:GetNWEntity(k) return self.nwe and self.nwe[k] end
end
function player.CreateNextBot(name)
  if MOCK.no_slot then return MOCK.NULL end
  local b = MOCK.newplayer("bot" .. (#MOCK.players + 1), { name = name })
  b.bot = true
  MOCK.bots[#MOCK.bots + 1] = b
  return b
end
game.MaxPlayers = function() return MOCK.maxplayers or 16 end
do
  local G = {}
  function SetGlobalEntity(k, v) G[k] = v end
  function GetGlobalEntity(k) return G[k] end
end
AT = {}
function AT.run(sec, step)
  step = step or 0.05
  for _ = 1, math.floor(sec / step + 0.5) do MOCK.advance(step); hook.Run("Tick") end
end
function AT.pos(p) return p:GetPos().x, p:GetPos().y, p:GetPos().z end
function AT.events()
  local out = {}
  for _, e in ipairs(AT.log or {}) do out[#out + 1] = e end
  return out
end
AT.log = {}
hook.Add("SkippyAvatarArrive", "T", function(bot, from, to) AT.log[#AT.log + 1] = { "arrive", from, to } end)
hook.Add("SkippyAvatarLeave", "T", function(bot, from) AT.log[#AT.log + 1] = { "leave", from } end)
"""


def avatar_world(cv=None, *, holodeck=True, bridge=False):
    cvars = dict(cv or {})
    if bridge:
        cvars.update({"holodeck_bridge": "http://127.0.0.1:9", "holodeck_agent": "0", "holodeck_persona": "0", "holodeck_voice_out": "1",
                      "mentat_avatar_ack_voice": "0"})
    else:
        cvars.setdefault("holodeck_voice_out", "0")
        cvars.setdefault("holodeck_persona", "0")
        cvars.setdefault("holodeck_agent", "0")
    w = World(cvars=cvars, load=())
    w.rt.execute(PRELUDE)
    if holodeck:
        w.rt.execute('dofile(MOCK.ROOT .. "/mentat/holodeck.lua")')
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/avatar.lua")')
    return w


def L(w, code, *args):
    """Evaluate a Lua function expression with arguments."""
    return w.rt.eval(f"function(...) {code} end")(*args)


def bot(w):
    return w.rt.eval("function() return MOCK.bots[#MOCK.bots] end")()


def where(w, p):
    return tuple(w.rt.eval("AT.pos")(p))


def run(w, seconds, step=0.05):
    w.rt.eval("AT.run")(seconds, step)


def events(w):
    out = []
    for e in w.rt.eval("AT.events")().values():
        out.append((e[1], e[2], e[3]))
    return out


def wrap180(a):
    return (a + 180) % 360 - 180


def gain_oracle(dist, rng, edge=0.6):
    """The proximity-voice fade, written independently of voice/core.lua."""
    if dist >= rng:
        return 0.0
    start = rng * edge
    if dist <= start:
        return 1.0
    t = (dist - start) / (rng - start)
    return 0.5 * (1 + math.cos(math.pi * t))


# ================================================================================================ calling him
def test_summon_puts_him_beside_the_caller_facing_them():
    w = avatar_world()
    p = w.player(A, x=100, y=100, yaw=90)                 # facing +y: their right is +x
    assert w.rt.eval("function(p) return Mentat.Avatar.Summon(p) end")(p) is True
    b = bot(w)
    x, y, z = where(w, b)
    assert 70 <= x - 100 <= 88 and abs(y - 100) < 20, (x, y)        # about 1.5 m (79 HU) to the side, a hand ahead
    assert 64 < z < 100                                              # a little above their eyes
    bx, by, _ = x - 100, y - 100, z
    facing = w.rt.eval("function(b) return b:EyeAngles().y end")(b)
    to_caller = math.degrees(math.atan2(100 - y, 100 - x))
    assert abs(wrap180(facing - to_caller)) < 1.0
    assert w.rt.eval("function() return MOCK.cvars.mentat_avatar_enabled ~= nil end")()
    assert L(w, "return Mentat.Avatar.IsActive()") is True
    assert L(w, "return GetGlobalBool('mentat_avatar_on')") is True


def test_the_bot_is_made_once_with_the_right_name_and_reused():
    w = avatar_world()
    p = w.player(A)
    L(w, "Mentat.Avatar.Summon(...)", p)
    L(w, "Mentat.Avatar.Summon(...)", p)
    assert L(w, "return #MOCK.bots") == 1
    assert L(w, "return MOCK.bots[1]:Nick()") == "Skippy"


def test_no_free_slot_is_reported_and_nothing_appears():
    w = avatar_world()
    w.rt.execute("MOCK.no_slot = true")
    p = w.player(A)
    ok, why = w.rt.eval("function(p) return Mentat.Avatar.Summon(p) end")(p)
    assert ok is False and why
    assert L(w, "return Mentat.Avatar.IsActive()") is False
    w.rt.execute("MOCK.no_slot = false; MOCK.maxplayers = 1")                # the server is full
    ok, why = w.rt.eval("function(p) return Mentat.Avatar.Summon(p) end")(p)
    assert ok is False and "slot" in why


def test_he_never_appears_inside_a_wall():
    w = avatar_world()
    w.rt.execute("MOCK.room = { x0 = 0, y0 = -500, x1 = 1000, y1 = 500 }")
    p = w.player(A, x=500, y=-480, yaw=0)                 # facing +x: their right is -y, straight into the wall 20 HU away
    L(w, "Mentat.Avatar.Summon(...)", p)
    x, y, z = where(w, bot(w))
    assert -500 < y < 500 and 0 < x < 1000
    assert y > -480, "the right side is the wall: he must take the left"
    # boxed in on every side: he still ends up inside the room
    w2 = avatar_world()
    w2.rt.execute("MOCK.room = { x0 = 460, y0 = -20, x1 = 540, y1 = 20 }")
    p2 = w2.player(A, x=500, y=0, yaw=30)
    L(w2, "Mentat.Avatar.Summon(...)", p2)
    x, y, z = where(w2, bot(w2))
    assert 460 < x < 540 and -20 < y < 20, (x, y)


def test_dismiss_hides_him_and_keeps_the_body_by_default():
    w = avatar_world()
    p = w.player(A)
    L(w, "Mentat.Avatar.Summon(...)", p)
    assert L(w, "return Mentat.Avatar.Dismiss()") is True
    b = bot(w)
    assert L(w, "return Mentat.Avatar.Get()") is None and L(w, "return GetGlobalBool('mentat_avatar_on')") is False
    assert w.rt.eval("function(b) return b.valid, b.nodraw end")(b) == (True, True)
    L(w, "Mentat.Avatar.Summon(...)", p)
    assert L(w, "return #MOCK.bots") == 1
    assert w.rt.eval("function(b) return b.nodraw end")(b) is False


def test_dismiss_can_kick_the_bot_to_free_the_slot():
    w = avatar_world({"mentat_avatar_kick": "1"})
    p = w.player(A)
    L(w, "Mentat.Avatar.Summon(...)", p)
    b = bot(w)
    L(w, "Mentat.Avatar.Dismiss()")
    assert w.rt.eval("function(b) return b.valid end")(b) is False
    L(w, "Mentat.Avatar.Summon(...)", p)
    assert L(w, "return #MOCK.bots") == 2


def test_arrival_and_leave_hooks_fire_with_positions():
    w = avatar_world()
    p = w.player(A, x=0, y=0, yaw=0)
    L(w, "Mentat.Avatar.Summon(...)", p)
    ev = events(w)
    assert [e[0] for e in ev] == ["arrive"] and ev[0][1] is None                      # first time: nowhere before
    to = ev[0][2]
    assert abs(to.x - where(w, bot(w))[0]) < 1e-6
    # called again by someone else elsewhere: he leaves where he was, arrives where they are
    q = w.player(B, x=2000, y=0, yaw=0)
    L(w, "Mentat.Avatar.Summon(...)", q)
    ev = events(w)
    assert [e[0] for e in ev] == ["arrive", "leave", "arrive"]
    assert abs(ev[1][1].x - to.x) < 1e-6                                              # he left from the first spot
    assert ev[2][1] is not None and abs(ev[2][1].x - to.x) < 1e-6
    assert abs(ev[2][2].x - 2000) < 120
    L(w, "Mentat.Avatar.Dismiss()")
    assert events(w)[-1][0] == "leave"


# ================================================================================================ following
def test_he_drifts_smoothly_to_the_shoulder_and_bobs():
    w = avatar_world()
    p = w.player(A, x=0, y=0, yaw=0)
    L(w, "Mentat.Avatar.Summon(...)", p)
    w.rt.eval("function(p) p.pos = Vector(250, 0, 0) end")(p)
    run(w, 0.05)
    x1, y1, z1 = where(w, bot(w))
    assert 12 <= x1 < 224 - 5, "one tick moves him toward the shoulder, not all the way (that would be a snap)"
    zs = []
    for _ in range(40):
        run(w, 0.1)
        zs.append(where(w, bot(w))[2])
    x, y, z = where(w, bot(w))
    # yaw 0: right is -y, so the shoulder is 46 to the right and 26 behind, 16 above the eyes
    assert abs(x - (250 - 26)) < 5 and abs(y - (-46)) < 5, (x, y)
    assert 80 - 4 <= min(zs) and max(zs) <= 80 + 4 and max(zs) - min(zs) > 1.0, "gentle bob around the hover height"
    assert L(w, "return #AT.events()") == 1, "smooth following fires no arrival effects"


def test_he_snaps_when_the_owner_teleports_or_is_left_far_behind():
    w = avatar_world()
    p = w.player(A, x=0, y=0, yaw=0)
    L(w, "Mentat.Avatar.Summon(...)", p)
    w.rt.eval("function(p) p.pos = Vector(5000, 0, 0) end")(p)
    run(w, 0.05)                                                   # ONE tick
    x, y, z = where(w, bot(w))
    assert abs(x - (5000 - 26)) < 2 and abs(y + 46) < 2, (x, y)
    assert [e[0] for e in events(w)] == ["arrive", "leave", "arrive"]
    # left far behind without any single jump (a slow owner and a stuck follower): farther than 8 m
    L(w, "Mentat.Avatar.st.pos = { x = 5000 + 600, y = 0, z = 80 }")
    run(w, 0.05)
    x, y, z = where(w, bot(w))
    assert abs(x - (5000 - 26)) < 2
    assert [e[0] for e in events(w)][-2:] == ["leave", "arrive"]


def test_he_stays_on_the_owners_side_of_a_wall_while_following():
    w = avatar_world()
    w.rt.execute("MOCK.room = { x0 = 0, y0 = -500, x1 = 1000, y1 = 500 }")
    p = w.player(A, x=500, y=-470, yaw=0)
    L(w, "Mentat.Avatar.Summon(...)", p)
    run(w, 3)
    x, y, z = where(w, bot(w))
    assert y > -500, "the shoulder point is through the wall: he hovers short of it"


def test_he_waits_when_the_owner_is_dead_and_leaves_when_the_owner_leaves():
    w = avatar_world()
    p = w.player(A)
    L(w, "Mentat.Avatar.Summon(...)", p)
    here = where(w, bot(w))
    w.rt.eval("function(p) p.alive = false; p.pos = Vector(100, 0, 0) end")(p)
    run(w, 1)
    assert where(w, bot(w)) == here
    w.rt.eval("function(p) MOCK.disconnect(p) end")(p)
    assert L(w, "return Mentat.Avatar.IsActive()") is False


def test_the_bots_own_ai_is_switched_off_and_it_is_harmless():
    w = avatar_world()
    p = w.player(A)
    L(w, "Mentat.Avatar.Summon(...)", p)
    b = bot(w)
    r = w.rt.eval("""function(b, p)
      local function cmd() local c = { log = {} }
        function c:ClearButtons() self.log.buttons = true end
        function c:ClearMovement() self.log.move = true end
        function c:SetImpulse(i) self.log.impulse = i end
        function c:SetViewAngles(a) self.log.view = a end
        return c end
      local cb, cp = cmd(), cmd()
      hook.Run('StartCommand', b, cb); hook.Run('StartCommand', p, cp)
      return cb.log.buttons == true, cb.log.move == true, cb.log.impulse, cb.log.view ~= nil, cp.log.buttons == nil,
             hook.Run('PlayerLoadout', b), hook.Run('PlayerLoadout', p), hook.Run('PlayerShouldTakeDamage', b),
             b.stripped, b.god, b.movetype, b.notsolid, b.notarget
    end""")(b, p)
    assert tuple(r) == (True, True, 0, True, True, True, None, False, True, True, 8, True, True)


def test_a_killed_bot_is_put_back_on_his_feet_without_a_ragdoll():
    """Live pass: god mode does not stop Player:Kill(); the bot then stayed dead for ever (bots never respawn by themselves)."""
    w = avatar_world()
    p = w.player(A)
    L(w, "Mentat.Avatar.Summon(...)", p)
    r = w.rt.eval("""function(b, p)
      b.spawned = 0
      function b:Spawn() self.spawned = self.spawned + 1; self.alive = true; hook.Run('PlayerSpawn', self) end
      local ragdoll_bot, ragdoll_player = hook.Run('DoPlayerDeath', b), hook.Run('DoPlayerDeath', p)
      b.alive = false
      hook.Run('PlayerDeath', b)
      p.alive = false
      hook.Run('PlayerDeath', p)
      MOCK.advance(0.5)
      return ragdoll_bot, ragdoll_player, b.spawned, b.alive, p.spawned
    end""")(bot(w), p)
    assert tuple(r) == (True, None, 1, True, None), "the bot is revived (once), the death makes no ragdoll of him, other players are untouched"
    run(w, 0.5)
    assert abs(where(w, bot(w))[0] - where(w, p)[0]) < 200, "and he is back with his owner"


def test_a_walking_body_is_put_back_to_noclip_on_the_next_tick():
    """Live pass: after Player:Spawn() the engine had set the bot's movetype to WALK, so gravity worked on him."""
    w = avatar_world()
    p = w.player(A)
    L(w, "Mentat.Avatar.Summon(...)", p)
    r = w.rt.eval("""function(b)
      function b:GetMoveType() return self.movetype end
      b.movetype = MOVETYPE_WALK
      hook.Run('Tick')
      return b.movetype == MOVETYPE_NOCLIP
    end""")(bot(w))
    assert r is True


def test_the_tool_runner_runs_once_for_the_first_human_and_never_for_a_joining_bot():
    """Live pass: Skippy's bot joining a listen server fired the runner's PlayerInitialSpawn hook again, and the job ran a second
    time with the bot as 'the player'."""
    w = World(load=())
    w.rt.execute(PRELUDE)
    w.rt.execute("""
      game.IsDedicated = function() return false end
      local real_include = include
      include = function(path)
        if path == "mentat/tools/probe_runs.lua" then return function(args, done, ply) RUNS = (RUNS or 0) + 1; RAN_FOR = ply; done({}) end end
        return real_include(path)
      end
      MOCK.files["mentat/tool.txt"] = '{"script":"mentat/tools/probe_runs.lua","id":"t"}'
      dofile(MOCK.ROOT .. "/mentat/tool.lua")
    """)
    human = w.player(A)
    bot_ply = w.rt.eval("function() local b = player.CreateNextBot('Skippy'); return b end")()
    L(w, "hook.Run('PlayerInitialSpawn', ...)", bot_ply)
    L(w, "hook.Run('PlayerInitialSpawn', ...)", human)
    L(w, "hook.Run('PlayerInitialSpawn', ...)", bot_ply)
    w.rt.eval("function() MOCK.advance(6) end")()
    assert L(w, "return RUNS") == 1 and L(w, "return RAN_FOR == ...", human) is True


# ================================================================================================ the words
CALLS = ["come here", "Skippy, come here", "skippy come here please", "skippy come over", "Skippy, come over here!", "Skippy come",
         "come to me", "cmoe here", "come hear", "skippy comee here", "get over here", "skippy could you come here",
         "I want you to come here", "follow me", "come closer", "over here", "show yourself", "appear", "hover here", "come on over",
         "Skippy, get here", "skippy come back"]
SENDS = ["go away", "Skippy, go away", "skippy dismiss", "dismiss", "please go away", "go home", "skippy leave me alone", "disapear",
         "bye", "vanish", "get lost", "skippy leave", "Skippy, dismiss yourself"]
NOT_COMMANDS = ["add a couch", "skippy add a couch", "add a couch here", "come up with a name for the couch", "remove the couch",
                "leave the couch alone", "hide the couch", "get me a couch", "show me a couch", "clear everything", "undo", "go",
                "turn the chair around", "move the couch here", "build a house", "comb the hair", "add a welcome mat",
                "dismiss the couch", "go away from the couch", "skippy", "", "make the couch come here", "follow the lamp",
                "come to the kitchen", "put three lamps over here", "home"]


def test_phrases_that_call_him_or_send_him_away():
    w = avatar_world()
    m = w.rt.eval("function(t) return Mentat.Avatar.Logic.match_command(t) end")
    for t in CALLS:
        assert m(t) == "summon", t
    for t in SENDS:
        assert m(t) == "dismiss", t
    for t in NOT_COMMANDS:
        assert m(t) is None, t


def test_saying_come_here_in_any_of_the_three_ways_calls_him_and_builds_nothing():
    w = avatar_world()
    p = w.player(A, x=0, y=0)
    w.ask(p, "come here")                                          # the console, and the voice bridge after the wake word
    assert L(w, "return Mentat.Avatar.IsActive()") is True and w.count() == 0
    L(w, "Mentat.Avatar.Dismiss()")
    w.advance(2)
    w.rt.eval("function(p) hook.Run('PlayerSay', p, 'Skippy, come here') end")(p)                      # typed chat
    w.advance(2)
    assert L(w, "return Mentat.Avatar.IsActive()") is True and w.count() == 0
    w.rt.eval("function(p) Mentat.Avatar.Dismiss(); MOCK.advance(2); Holodeck.Handle(p, 'go away'); MOCK.advance(2) end")(p)
    assert L(w, "return Mentat.Avatar.IsActive()") is False
    w.rt.eval("function(p) Holodeck.Handle(p, 'skippy come here please'); MOCK.advance(2) end")(p)
    assert L(w, "return Mentat.Avatar.IsActive()") is True


def test_ordinary_holodeck_requests_still_build_things_and_do_not_call_him():
    w = avatar_world()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.ask(p, "skippy add a couch")
    assert w.count(A) >= 2
    assert L(w, "return Mentat.Avatar.IsActive()") is False and L(w, "return #MOCK.bots") == 0


def test_switched_off_the_words_go_to_the_holodeck_as_before():
    w = avatar_world({"mentat_avatar_enabled": "0"})
    p = w.player(A)
    w.ask(p, "come here")
    assert L(w, "return #MOCK.bots") == 0
    ok, why = w.rt.eval("function(p) return Mentat.Avatar.Summon(p) end")(p)
    assert ok is False


def test_wrapping_twice_does_not_stack_and_a_refresh_of_the_holodeck_is_picked_up():
    w = avatar_world()
    p = w.player(A)
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/avatar.lua")')                          # a Lua refresh of the avatar
    w.ask(p, "come here")
    assert L(w, "return #MOCK.bots") == 1
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/holodeck.lua"); Mentat.Avatar.Install()')   # a refresh of the holodeck brings plain functions back
    L(w, "Mentat.Avatar.Dismiss()")
    w.advance(2)
    w.ask(p, "come here")
    assert L(w, "return Mentat.Avatar.IsActive()") is True


# ================================================================================================ his voice (server)
def speak_world(**cv):
    return avatar_world({"holodeck_bridge_per_min": "600", "holodeck_voice_chars_per_min": "100000", **cv}, bridge=True)


def speak_pos(w, p, text="hello there"):
    """Say a line and return the position the audio message carries."""
    n0 = len(list(w.rt.eval("function() return T.nets('holodeck_audio') end")().values()))
    before = len(list(w.rt.eval("function() return T.posts('/speak') end")().values()))
    w.rt.eval("function(p, t) Holodeck.Speak(t, nil, p); MOCK.advance(1) end")(p, text)
    posts = list(w.rt.eval("function() return T.posts('/speak') end")().values())
    assert len(posts) == before + 1, "the line was sent to the bridge"
    w.rt.eval("function(i) MOCK.respond(i, 200, 'ID3' .. string.rep('x', 3000), {}) end")(posts[-1])
    w.advance(2)
    msgs = list(w.rt.eval("function() return T.nets('holodeck_audio') end")().values())
    assert len(msgs) > n0, "the clip was sent"
    v = w.rt.eval("function(m) return m.vector.x, m.vector.y, m.vector.z end")(msgs[-1])
    return tuple(v)


def test_with_the_avatar_out_his_speech_comes_from_him():
    w = speak_world()
    p = w.player(A, x=0, y=0, yaw=0)
    assert speak_pos(w, p) == (0, 0, 0), "no avatar: from the player, exactly as before"
    L(w, "Mentat.Avatar.Summon(...)", p)
    assert speak_pos(w, p) == where(w, bot(w))
    # and it follows him: after he has moved, the next line comes from the new place
    w.rt.eval("function(p) p.pos = Vector(250, 0, 0) end")(p)
    run(w, 3)
    assert speak_pos(w, p) == where(w, bot(w))
    L(w, "Mentat.Avatar.Dismiss()")
    assert speak_pos(w, p) == (250, 0, 0)


def test_his_chat_answers_are_placed_at_him_too():
    w = speak_world()
    p = w.player(A, x=0, y=0, yaw=0)
    stayed = w.player(B, x=0, y=300)                    # still where the caller was: far from where the avatar is now
    L(w, "Mentat.Avatar.Summon(...)", p)
    w.rt.eval("function(p) p.pos = Vector(4000, 0, 0) end")(p)
    run(w, 0.05)
    n = len(list(w.rt.eval("function() return T.nets('holodeck_say') end")().values()))
    w.rt.eval("function(p) Holodeck.Say(p, 'Right here.', nil, { speak = false }) end")(p)
    msgs = list(w.rt.eval("function() return T.nets('holodeck_say') end")().values())
    assert len(msgs) == n + 1
    got = {w.rt.eval("function(q) return q:SteamID64() end")(q) for q in msgs[-1].to.values()}
    assert A in got and B not in got and stayed, "audience is around the avatar (with the asker), not around where they were"


def hearing(w, listener=None):
    out = []
    for h in w.rt.eval("function(p) return Mentat.Avatar.WhoHears(p) end")(listener).values():
        out.append({k: h[k] for k in ("nick", "dist", "range", "blocked", "gain", "heard")})
    return out


def test_who_hears_him_uses_exactly_the_proximity_voice_numbers():
    w = avatar_world()
    owner = w.player(A, x=0, y=0, yaw=0)
    L(w, "Mentat.Avatar.Summon(...)", owner)
    bx, by, bz = where(w, bot(w))
    # listeners at chosen distances along +x from him, at standing height, on open ground (no walls)
    spots = {"close": 100, "edge": 800, "far": 1300}
    listeners = {k: w.player(str(500 + i), x=bx + d, y=by) for i, (k, d) in enumerate(spots.items())}
    run(w, 0.3)
    for p in listeners.values():                 # the owner walked nowhere, but the listeners must not have moved him
        pass
    bx, by, bz = where(w, bot(w))
    rows = {h["nick"]: h for h in hearing(w)}
    for k, p in listeners.items():
        h = rows[w.rt.eval("function(p) return p:Nick() end")(p)]
        px, py, pz = where(w, p)
        d = math.sqrt((px - bx) ** 2 + (py - by) ** 2 + (pz + 64 - bz) ** 2)
        assert abs(h["dist"] - d) < 1e-6
        assert h["range"] == 1100 and h["blocked"] is False
        assert abs(h["gain"] - gain_oracle(d, 1100)) < 1e-9
        assert h["heard"] is (h["gain"] > 0)
    assert rows[w.rt.eval("function(p) return p:Nick() end")(listeners["far"])]["gain"] == 0
    assert 0 < rows[w.rt.eval("function(p) return p:Nick() end")(listeners["edge"])]["gain"] < 1
    # the same through the engine's own table of numbers
    d_edge = hearing(w, listeners["edge"])[0]["dist"]
    assert w.rt.eval("function(d) return Mentat.Voice and 0 or 1 end")(d_edge) == 1            # voice.lua is not needed


def test_walls_shrink_his_range_like_they_shrink_a_players_voice():
    w = avatar_world()
    w.rt.execute("MOCK.room = { x0 = 0, y0 = -2000, x1 = 1000, y1 = 2000 }")
    owner = w.player(A, x=-300, y=0, yaw=0)                # outside the room: open ground around him
    L(w, "Mentat.Avatar.Summon(...)", owner)
    inside = w.player(B, x=300, y=0)                       # inside: the room's wall x0 = 0 is between them
    h = hearing(w, inside)[0]
    assert h["blocked"] is True
    r = 1100 * 0.55
    assert abs(h["gain"] - gain_oracle(h["dist"], r)) < 1e-9
    # 55% of 1100 is 605: someone behind the wall at 700 HU hears nothing, someone in the open at 700 hears him
    w.rt.eval("function(p) p.pos = Vector(400, 0, 0) end")(inside)
    far = hearing(w, inside)[0]
    assert far["dist"] > 605 and far["gain"] == 0 and far["heard"] is False


def test_a_whisper_to_a_crouching_caller_carries_less_and_global_mode_is_flat():
    w = avatar_world()
    owner = w.player(A, x=0, y=0, yaw=0)
    L(w, "Mentat.Avatar.Summon(...)", owner)
    listener = w.player(B, x=400, y=-79)
    assert hearing(w, listener)[0]["range"] == 1100
    w.rt.eval("function(p) p.crouch = true end")(owner)
    run(w, 0.4)                                            # the range is published a few times a second
    h = hearing(w, listener)[0]
    assert abs(h["range"] - 1100 * 0.35) < 1e-9 and abs(h["gain"] - gain_oracle(h["dist"], 385)) < 1e-9
    w.rt.execute("SetGlobalString('mentat_voice_mode', 'global')")
    assert hearing(w, listener)[0]["gain"] == 1


def test_the_voice_range_follows_the_servers_range_convar():
    w = avatar_world({"mentat_voice_range": "500"})
    owner = w.player(A, x=0, y=0, yaw=0)
    L(w, "Mentat.Avatar.Summon(...)", owner)
    assert hearing(w, owner)[0]["range"] == 500


def test_debug_log_prints_one_line_per_listener_per_line_spoken():
    w = speak_world(mentat_avatar_debug="1")
    owner = w.player(A, x=0, y=0, yaw=0)
    w.player(B, x=900, y=-79)
    L(w, "Mentat.Avatar.Summon(...)", owner)
    w.rt.execute("MOCK.printed = {}")
    w.rt.eval("function(p) Holodeck.Speak('hello there', nil, p) end")(owner)
    lines = [l for l in w.rt.eval("function() return MOCK.printed end")().values() if l.startswith("[avatar]")]
    assert len(lines) == 2
    assert all(re.search(r"dist=\d+ range=1100 blocked=(true|false) gain=\d\.\d\d", l) for l in lines), lines
    # off by default
    w2 = speak_world()
    o2 = w2.player(A, x=0, y=0, yaw=0)
    L(w2, "Mentat.Avatar.Summon(...)", o2)
    w2.rt.execute("MOCK.printed = {}")
    w2.rt.eval("function(p) Holodeck.Speak('hello there', nil, p) end")(o2)
    assert not [l for l in w2.rt.eval("function() return MOCK.printed end")().values() if l.startswith("[avatar]")]
    assert len(L(w2, "return Mentat.Avatar.WhoHears()")) == 1 and L(w2, "return Mentat.Avatar.last_heard ~= nil")


def test_who_hears_is_empty_when_he_is_not_out():
    w = avatar_world()
    w.player(A)
    assert hearing(w) == []


# ================================================================================================ his voice (client)
CLIENT_PRELUDE = r"""
do
  local G = {}
  function SetGlobalEntity(k, v) G[k] = v end
  function GetGlobalEntity(k) return G[k] end
end
MOCK.bot = MOCK.newplayer("bot", { pos = Vector(800, 0, 0) })
MOCK.me = MOCK.newplayer("me", { pos = Vector(0, 0, 0) })
function sound.PlayFile(path, flags, cb)
  local chan = { valid = true, state = GMOD_CHANNEL_STOPPED, path = path, len = 4, fade = {} }
  function chan:IsValid() return self.valid end
  function chan:SetPos(v) self.pos = v end
  function chan:Set3DFadeDistance(a, b) self.fade = { a, b } end
  function chan:SetVolume(v) self.vol = v end
  function chan:Play() self.state = GMOD_CHANNEL_PLAYING end
  function chan:Stop() self.state = GMOD_CHANNEL_STOPPED; self.valid = false end
  function chan:GetState() return self.state end
  function chan:GetLength() return self.len end
  MOCK.sounds[#MOCK.sounds + 1] = chan
  cb(chan, 0, "")
end
function AT_out(on)
  SetGlobalBool("mentat_avatar_on", on)
  SetGlobalEntity("mentat_avatar", on and MOCK.bot or nil)
end
function AT_think(n) for _ = 1, n do MOCK.advance(0.05); hook.Run("Think") end end
"""


def client_avatar(on=True, bot_at=(800, 0, 0), range_=1100.0):
    w = World(server=False, load=())
    w.rt.execute(CLIENT_PRELUDE)
    w.rt.execute(f"MOCK.bot.pos = Vector({bot_at[0]}, {bot_at[1]}, {bot_at[2]}); MOCK.bot:SetNWFloat('mentat_avatar_range', {range_})")
    for f in ("skippy_client.lua", "avatar_client.lua"):
        w.rt.execute(f'dofile(MOCK.ROOT .. "/mentat/{f}")')
    w.rt.eval("AT_out")(on)
    return w


def clip(w, clip_id=1):
    w.rt.eval("""function(id) MOCK.deliver({ name = 'holodeck_audio', writes = { id, 1, 1 }, vector = Vector(1, 2, 3), data = 'ID3' .. string.rep('a', 300) }) end""")(clip_id)
    return w.rt.eval("function() return MOCK.sounds[#MOCK.sounds] end")()


def chan_state(w, ch):
    return w.rt.eval("function(c) return c.vol, c.pos and c.pos.x, c.pos and c.pos.y, c.pos and c.pos.z, c.fade[1], c.fade[2] end")(ch)


def test_client_without_an_avatar_plays_skippy_as_before():
    w = client_avatar(on=False)
    ch = clip(w)
    vol, x, y, z, f0, f1 = chan_state(w, ch)
    assert (x, y, z) == (1, 2, 3) and (f0, f1) == (250, 2400)
    w.rt.eval("AT_think")(10)
    assert chan_state(w, ch) == (vol, 1, 2, 3, 250, 2400), "nothing moves it or changes its volume"


@pytest.mark.parametrize("dist,expect", [(100, 1.0), (800, gain_oracle(800, 1100)), (1000, gain_oracle(1000, 1100)), (1100, 0.0), (1500, 0.0)])
def test_client_plays_his_voice_from_him_with_the_proximity_gain(dist, expect):
    w = client_avatar(bot_at=(dist, 0, 64))              # my eyes are at z = 64: the distance is exactly `dist`
    ch = clip(w)
    vol, x, y, z, f0, f1 = chan_state(w, ch)
    assert (x, y, z) == (dist, 0, 64), "the clip sits at the avatar, not where the server said"
    assert abs(vol - expect) < 1e-9
    assert (f0, f1) == (100000, 200000), "the engine's own fade is flattened: the gain is Core.gain's"


def test_client_voice_follows_him_and_obeys_walls_whisper_and_modes():
    w = client_avatar(bot_at=(400, 0, 64))
    ch = clip(w)
    assert abs(chan_state(w, ch)[0] - 1.0) < 1e-9
    # he flies away: position tracks per frame, volume eases to the new gain
    w.rt.execute("MOCK.bot.pos = Vector(900, 0, 64)")
    w.rt.eval("AT_think")(80)
    vol, x, y, z, *_ = chan_state(w, ch)
    assert (x, y, z) == (900, 0, 64) and abs(vol - gain_oracle(900, 1100)) < 1e-3
    # a wall between us (the room's x0 = 500 wall; I stand inside, he is outside it): range * 0.55
    w.rt.execute("MOCK.room = { x0 = -5000, y0 = -500, x1 = 5000, y1 = 500 }; MOCK.me.pos = Vector(0, 0, 0); MOCK.bot.pos = Vector(600, 900, 64)")
    w.rt.eval("AT_think")(80)
    d = math.sqrt(600 ** 2 + 900 ** 2)
    assert abs(chan_state(w, ch)[0] - gain_oracle(d, 1100 * 0.55)) < 1e-3 and d > 605
    # a whisper
    w.rt.execute("MOCK.room = nil; MOCK.bot.pos = Vector(300, 0, 64); MOCK.bot:SetNWFloat('mentat_avatar_range', 385)")
    w.rt.eval("AT_think")(80)
    assert abs(chan_state(w, ch)[0] - gain_oracle(300, 385)) < 1e-3
    # global mode: distance does not matter
    w.rt.execute("SetGlobalString('mentat_voice_mode', 'global'); MOCK.bot.pos = Vector(5000, 0, 64)")
    w.rt.eval("AT_think")(80)
    assert abs(chan_state(w, ch)[0] - 1.0) < 1e-3
    # the player's own volume setting scales it
    w2 = client_avatar(bot_at=(100, 0, 64))
    w2.rt.execute("GetConVar('skippy_volume'):SetValue('0.5')")
    assert abs(chan_state(w2, clip(w2))[0] - 0.5) < 1e-9


def test_client_stops_tracking_a_finished_clip():
    w = client_avatar()
    ch = clip(w)
    assert w.rt.eval("function() local n = 0; for _ in pairs(Mentat.Avatar.tracked) do n = n + 1 end; return n end")() == 1
    w.rt.eval("function(c) c.state = GMOD_CHANNEL_STOPPED end")(ch)
    w.rt.eval("AT_think")(2)
    assert w.rt.eval("function() return next(Mentat.Avatar.tracked) end")() is None


def test_client_hides_the_bots_body_and_reports_his_hearing():
    w = client_avatar(bot_at=(300, 0, 64))
    assert w.rt.eval("function() return (hook.Run('PrePlayerDraw', MOCK.bot)) end")() is True
    assert w.rt.eval("function() return (hook.Run('PrePlayerDraw', MOCK.me)) end")() is None
    heard, gain, dist, rng, blocked = w.rt.eval("function() return Mentat.Avatar.Hearing() end")()
    assert heard is True and gain == 1 and abs(dist - 300) < 1e-9 and rng == 1100 and blocked is False


# ================================================================================================ the pure parts
def lua_pure(code):
    w = World(load=())
    w.rt.execute('LG = dofile(MOCK.ROOT .. "/mentat/avatar/logic.lua")')
    return w.rt.eval(f"function() {code} end")()


def test_logic_hear_is_voice_cores_decide():
    w = World(load=())
    w.rt.execute('LG = dofile(MOCK.ROOT .. "/mentat/avatar/logic.lua"); VC = dofile(MOCK.ROOT .. "/mentat/voice/core.lua")')
    f = w.rt.eval("""function(d, r, blocked)
      local h1, g1 = LG.hear(d, r, blocked, 0.55, nil, 'proximity')
      local h2, g2 = VC.decide(d, r, blocked, { wall = 0.55 })
      return h1 == h2, g1 == g2, g1
    end""")
    for d in (0, 300, 700, 1000, 1099, 1100, 2000):
        for blocked in (False, True):
            a, b, g = f(d, 1100, blocked)
            assert a and b
    assert tuple(w.rt.eval("function() return LG.hear(5000, 10, true, 0.55, nil, 'global') end")()) == (True, 1)


def test_logic_step_is_exponential_and_speed_limited():
    w = World(load=())
    w.rt.execute('LG = dofile(MOCK.ROOT .. "/mentat/avatar/logic.lua")')
    r = w.rt.eval("""function()
      local st = { pos = { x = 0, y = 0, z = 0 } }
      local owner = { x = 0, y = 0, z = 0 }
      local snapped = LG.step(st, { x = 100, y = 0, z = 0 }, owner, 0.1)
      local one = st.pos.x
      local st2 = { pos = { x = 0, y = 0, z = 0 } }
      LG.step(st2, { x = 400, y = 0, z = 0 }, { x = 400, y = 0, z = 0 }, 0.1, { max_speed = 50, snap_dist = 1e9 })
      return snapped, one, st2.pos.x
    end""")()
    snapped, one, capped = r
    assert snapped is False and abs(one - 100 * (1 - math.exp(-4.5 * 0.1))) < 1e-9
    assert abs(capped - 5.0) < 1e-9


def test_logic_turn_hat_tilt_and_envelope():
    w = World(load=())
    w.rt.execute('LG = dofile(MOCK.ROOT .. "/mentat/avatar/logic.lua")')
    f = w.rt.eval("function(c, t, s) return LG.turn(c, t, s) end")
    assert f(350, 10, 5) == 355 and f(10, 350, 5) == 5 and f(0, 3, 5) == 3
    p, r = w.rt.eval("function() return LG.hat_tilt(10, 0, 1) end")()
    assert abs(p - 14) < 1e-9 and abs(r) < 1e-9
    p, r = w.rt.eval("function() return LG.hat_tilt(0, -5, 0.5) end")()
    assert abs(p) < 1e-9 and abs(r + 7) < 1e-9
    assert tuple(w.rt.eval("function() return LG.hat_tilt(0, 0, 1) end")()) == (0, 0)
    env = [w.rt.eval("function(t) return LG.speech_envelope(t) end")(i * 0.01) for i in range(600)]
    assert min(env) == 0 and max(env) > 0.9 and all(0 <= v <= 1 for v in env)


# ================================================================================================ the geometry
@pytest.fixture(scope="module")
def geo():
    w = World(load=())
    w.rt.execute('GEO = dofile(MOCK.ROOT .. "/mentat/avatar/geometry.lua")')
    raw = w.rt.eval("""function()
      local b = GEO.build()
      local out = {}
      for name, part in pairs(b.parts) do
        local ts = {}
        for i, t in ipairs(part.tris) do
          ts[i] = { a = { t.a.x, t.a.y, t.a.z }, b = { t.b.x, t.b.y, t.b.z }, c = { t.c.x, t.c.y, t.c.z }, n = { t.n.x, t.n.y, t.n.z },
                    color = t.color, kind = t.kind }
        end
        out[name] = ts
      end
      return MOCK.json_encode({ parts = out, count = GEO.triangle_count() })
    end""")()
    data = json.loads(raw)
    data["w"] = w
    return data


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def bbox(tris):
    pts = [p for t in tris for p in (t["a"], t["b"], t["c"])]
    return [min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]


def test_geometry_is_small_and_complete(geo):
    n = sum(len(v) for v in geo["parts"].values())
    assert n == geo["count"]
    assert 300 < n < 2500
    assert set(geo["parts"]) == {"can", "label", "face", "mouth", "hat"}


def test_every_triangle_is_real_finite_and_its_normal_matches_its_winding(geo):
    for name, tris in geo["parts"].items():
        for t in tris:
            for p in (t["a"], t["b"], t["c"], t["n"]):
                assert all(math.isfinite(v) for v in p)
            cr = cross(sub(t["b"], t["a"]), sub(t["c"], t["a"]))
            area = norm(cr) / 2
            assert area > 1e-4, (name, t)                                     # non-degenerate
            assert abs(norm(t["n"]) - 1) < 1e-9
            assert dot(t["n"], cr) / (2 * area) > 1 - 1e-9, "stored normal = the winding's normal"


def test_the_can_is_half_a_metre_tall_and_slim(geo):
    lo, hi = bbox(geo["parts"]["can"] + geo["parts"]["label"])
    height_m = (hi[2] - lo[2]) / 52.49
    assert 0.47 <= height_m <= 0.53, height_m
    diam = max(hi[0] - lo[0], hi[1] - lo[1])
    assert 12 <= diam <= 17, diam                                             # about 14.4 HU across
    assert abs(lo[2] + hi[2]) < 0.5, "the origin is the middle of the can"


def test_the_can_normals_face_outward(geo):
    checked = {"side": 0, "up": 0, "down": 0, "inner": 0}
    for name in ("can", "label"):
        for t in geo["parts"][name]:
            if t["kind"] == "decor":
                continue
            cx = [(t["a"][i] + t["b"][i] + t["c"][i]) / 3 for i in range(3)]
            radial = (cx[0] * t["n"][0] + cx[1] * t["n"][1]) / max(math.hypot(cx[0], cx[1]), 1e-9)
            if t["kind"] == "side":
                assert radial > 0.2, (name, cx, t["n"])                           # away from the axis
            elif t["kind"] == "up":
                assert t["n"][2] > 0.99
            elif t["kind"] == "down":
                assert t["n"][2] < -0.99
            elif t["kind"] == "inner":
                assert radial < -0.99                                              # the lid's wall faces the axis
            checked[t["kind"]] += 1
    assert all(v > 0 for v in checked.values()), checked


def test_decor_boxes_face_outward_from_their_centre(geo):
    for name in ("can", "face", "mouth"):
        decor = [t for t in geo["parts"][name] if t["kind"] == "decor"]
        assert decor
    mouth = geo["parts"]["mouth"]
    lo, hi = bbox(mouth)
    c = [(lo[i] + hi[i]) / 2 for i in range(3)]
    for t in mouth:
        cx = [(t["a"][i] + t["b"][i] + t["c"][i]) / 3 for i in range(3)]
        assert dot(t["n"], sub(cx, c)) > 0


def test_the_hat_is_much_wider_than_the_can_and_sits_on_top(geo):
    clo, chi = bbox(geo["parts"]["can"])
    hlo, hhi = bbox(geo["parts"]["hat"])
    can_diameter = max(chi[0] - clo[0], chi[1] - clo[1])
    hat_width = hhi[1] - hlo[1]                                                  # point to point, across the way he faces
    assert hat_width >= 2.5 * can_diameter and 40 <= hat_width <= 55, (hat_width, can_diameter)
    assert hhi[2] > chi[2] + 4, "the crown and the points rise above the can"
    assert hlo[2] < chi[2] and hlo[2] > -2, "the hat sits on the top end of the can, not around its middle"
    assert (hhi[0] - hlo[0]) < hat_width * 0.6, "a bicorn is a thin crescent, not a disc"
    # symmetric points
    assert abs(hlo[1] + hhi[1]) < 1e-6


def test_the_face_stands_proud_of_the_label_so_it_can_be_seen(geo):
    """Live pass: the eyes sat at 0.97 R, inside the label band (1.012 R): the label hid them and only a sliver of mouth showed."""
    label_r = 7.2 * 1.012
    eyes = [t for t in geo["parts"]["face"] if t["kind"] == "decor"]
    assert eyes
    assert min(p[0] for t in eyes for p in (t["a"], t["b"], t["c"])) >= label_r - 0.001
    mlo, mhi = bbox(geo["parts"]["mouth"])
    assert mhi[0] > label_r + 0.2 and mlo[0] < label_r, "the mouth pokes through the label's surface"


def test_the_hat_is_a_chunky_crown_that_covers_the_top_of_the_can(geo):
    """Live pass: a thin banana floating over the can read as a smile, not a hat."""
    clo, chi = bbox(geo["parts"]["can"])
    mid = [t for t in geo["parts"]["hat"] if all(abs(p[1]) < 6 for p in (t["a"], t["b"], t["c"]))]
    zs = [p[2] for t in mid for p in (t["a"], t["b"], t["c"])]
    xs = [p[0] for t in mid for p in (t["a"], t["b"], t["c"])]
    assert max(zs) - min(zs) >= 14, "the crown is tall in the middle"
    assert max(xs) - min(xs) >= 20, "and deep front to back"
    assert min(zs) <= chi[2] - 4, "it comes down over the top of the can"


def test_the_cockade_stands_clear_of_the_hat_it_is_pinned_to(geo):
    """Live pass: with depth writing right, the left third of the cockade was seen to be sunk into the hat."""
    cyan = [t for t in geo["parts"]["hat"] if tuple(t["color"]) == (0, 204, 255)]
    pts = [p for t in cyan for p in (t["a"], t["b"], t["c"])]
    x0 = min(p[0] for p in pts)
    ylo, yhi = min(p[1] for p in pts), max(p[1] for p in pts)
    zlo, zhi = min(p[2] for p in pts), max(p[2] for p in pts)
    felt = [p for t in geo["parts"]["hat"] if tuple(t["color"]) != (0, 204, 255) and t["kind"] == "side" for p in (t["a"], t["b"], t["c"])
            if ylo <= p[1] <= yhi and zlo <= p[2] <= zhi]
    assert felt and max(p[0] for p in felt) <= x0 + 1e-6, (max(p[0] for p in felt), x0)


def hat_centre_z(geo, y):
    return geo["w"].rt.eval("function(y) local _, _, _, zc = GEO.hat_station(y / GEO.HAT_HALF_WIDTH); return zc end")(y)


def test_hat_triangles_face_away_from_the_hats_centre_line(geo):
    bad = 0
    for t in geo["parts"]["hat"]:
        if t["kind"] != "side":
            continue
        cx = [(t["a"][i] + t["b"][i] + t["c"][i]) / 3 for i in range(3)]
        y = cx[1]
        zc = hat_centre_z(geo, y)
        if dot(t["n"], [cx[0], 0, cx[2] - zc]) < -1e-9:
            bad += 1
    assert bad == 0


def test_the_palette_is_the_projects(geo):
    from_lua = geo["w"].rt.eval("function() return GEO.COLORS.green[1], GEO.COLORS.green[2], GEO.COLORS.green[3], GEO.COLORS.amber[1], GEO.COLORS.amber[2], GEO.COLORS.amber[3] end")()
    assert tuple(from_lua) == (0, 255, 136, 255, 170, 0)                          # #00FF88, #FFAA00
    label_colors = {tuple(t["color"].values()) if isinstance(t["color"], dict) else tuple(t["color"]) for t in geo["parts"]["label"]}
    assert label_colors == {(0, 255, 136)}
    hat_colors = {tuple(t["color"]) for t in geo["parts"]["hat"]}
    assert (255, 170, 0) in hat_colors, "amber trim"
    assert (34, 38, 58) in hat_colors, "dark felt"
    assert {(0, 204, 255), (255, 255, 255), (255, 68, 68)} <= hat_colors, "the cockade"
    amber = sum(1 for t in geo["parts"]["hat"] if tuple(t["color"]) == (255, 170, 0))
    dark = sum(1 for t in geo["parts"]["hat"] if tuple(t["color"]) == (34, 38, 58))
    assert dark > amber > 10, "mostly dark, with a trim"
    can_colors = {tuple(t["color"]) for t in geo["parts"]["can"]}
    assert (176, 184, 192) in can_colors


def test_baked_shading_stays_in_range(geo):
    f = geo["w"].rt.eval("""function(nx, ny, nz) local r, g, b = GEO.vertex_color({ n = { x = nx, y = ny, z = nz }, color = { 0, 255, 136 } }); return r, g, b, GEO.shade({ x = nx, y = ny, z = nz }) end""")
    for n in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (-0.6, 0.8, 0)):
        r, g, b, s = f(*n)
        assert 0.4 <= s <= 1.0 and 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
        assert r == 0


# ================================================================================================ packaging
def test_the_loader_ships_and_loads_every_avatar_file():
    text = (ADDON / "autorun" / "mentat_stage.lua").read_text(encoding="utf-8")
    shipped = set(re.findall(r'AddCSLuaFile\("([^"]+)"\)', text))
    for rel in ("mentat/avatar/logic.lua", "mentat/avatar/geometry.lua", "mentat/avatar_client.lua", "mentat/voice/core.lua"):
        assert rel in shipped, rel
        assert (ADDON / rel).is_file()
    assert 'include("mentat/avatar.lua")' in text and 'include("mentat/avatar_client.lua")' in text
    # the client file includes only files that are shipped
    client = (ADDON / "mentat" / "avatar_client.lua").read_text(encoding="utf-8")
    for inc in re.findall(r'include\("([^"]+)"\)', client):
        assert inc in shipped, inc
    shared = (ADDON / "mentat" / "avatar" / "logic.lua").read_text(encoding="utf-8")
    for inc in re.findall(r'include\("([^"]+)"\)', shared):
        assert inc in shipped, inc


def test_the_client_draws_with_depth_writing_on_and_restores_it():
    """Live pass: the colour material does not write depth, so the label (drawn after the face) painted over the eyes."""
    src = (ADDON / "mentat" / "avatar_client.lua").read_text(encoding="utf-8")
    first_draw = src.index("meshes.can:Draw()")
    on = src.rindex("render.OverrideDepthEnable(true, true)", 0, first_draw)          # the procedural body's own switch-on ...
    last_draw = src.index("meshes.hat:Draw()")
    off = src.index("render.OverrideDepthEnable(false, false)", last_draw)
    assert on < first_draw < last_draw < off
    # ... and the real body's: on before BodyC.draw, off after it
    real_on = src.index("render.OverrideDepthEnable(true, true)")
    real_draw = src.index("BodyC.draw(")
    real_off = src.index("render.OverrideDepthEnable(false, false)")
    assert real_on < real_draw < real_off


def test_a_module_wrapping_handle_on_top_does_not_make_the_avatars_wrapper_call_itself():
    """FAIL FIRST (2026-10-12): the water module wraps Holodeck.Handle on top of the avatar's wrapper (and registers H.wrappers.water); the avatar's
    install then ran again, took the water's function for 'the original' and stored it as H.AvatarOrig.Handle, which its first wrapper calls: the
    chain looped (avatar -> water -> avatar -> ...) and the real game froze on the first request."""
    w = avatar_world()
    p = w.player(A)
    w.rt.execute("""
      local H = Holodeck
      H.wrappers = H.wrappers or {}
      H.wrappers.water = true
      local inner = H.Handle
      H.Handle = function(ply, text, opts) return inner(ply, text, opts) end
      Mentat.Avatar.Install()
      Mentat.Avatar.Install()
    """)
    w.ask(p, "come here")
    assert L(w, "return #MOCK.bots") == 1, "the avatar's own command still works through the other module's wrapper"
    w.ask(p, "add a chair")                                                           # an ordinary request reaches the holodeck (no endless recursion)
