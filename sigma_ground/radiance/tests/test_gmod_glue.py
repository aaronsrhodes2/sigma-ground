"""Tests for the holodeck's engine glue (holodeck.lua), the proximity-voice rule (voice.lua) and the client files, run offline.

The REAL addon files are loaded under LuaJIT (lupa) against stage/gmod/holodeck/gmod_mock.lua, a small stand-in for the Garry's
Mod API with a virtual clock, recorded network messages and HTTP requests, and a flat world. The parser and planner are the
real ones; what these tests pin down is everything around them: ids and owners across undo, the prop caps, who may hear what
Skippy says, what the bridge may be asked, the audio queue, disconnects, the server's own prop rules, and the voice rule.

Each case came from a confirmed finding of the independent review (glue-N, security-N); the id is in the test docstring.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HOLO = Path(__file__).resolve().parents[3] / "stage" / "gmod" / "holodeck"
ADDON = Path(os.environ.get("HOLODECK_TEST_ADDON") or Path(__file__).resolve().parents[3] / "stage" / "gmod" / "addon" / "lua")
sys.path.insert(0, str(HOLO))

HELPERS = r"""
T = {}
function T.player(sid, o)
  o = o or {}
  return MOCK.newplayer(sid, { pos = Vector(o.x or 0, o.y or 0, 0), yaw = o.yaw or 0, admin = o.admin, name = o.name, host = o.host })
end
-- one request, then let the clock pass the per-player rest between requests
function T.ask(p, text) local r = Holodeck.Handle(p, text); MOCK.advance(1.3); return r end
function T.settle() MOCK.advance(12) end
function T.count(sid)
  local n = 0
  for _, id in ipairs(Holodeck.order) do
    if not sid or Holodeck.registry[id].owner == sid then n = n + 1 end
  end
  return n
end
function T.ids()
  local o = {}
  for i, id in ipairs(Holodeck.order) do o[i] = id end
  return table.concat(o, ",")
end
function T.owners()
  local o = {}
  for i, id in ipairs(Holodeck.order) do o[i] = Holodeck.registry[id].owner end
  return table.concat(o, ",")
end
function T.nets(name)
  local out = {}
  for _, m in ipairs(MOCK.nets) do if m.name == name then out[#out + 1] = m end end
  return out
end
function T.posts(path)
  local out = {}
  for i, h in ipairs(MOCK.https) do if h.url:sub(-#path) == path then out[#out + 1] = i end end
  return out
end
function T.printed(pattern)
  for _, l in ipairs(MOCK.printed) do if l:find(pattern, 1, true) then return true end end
  return false
end
"""


def lua_table(rt, obj):
    if isinstance(obj, dict):
        t = rt.table()
        for k, v in obj.items():
            t[k] = lua_table(rt, v)
        return t
    if isinstance(obj, (list, tuple)):
        t = rt.table()
        for i, v in enumerate(obj, 1):
            t[i] = lua_table(rt, v)
        return t
    return obj


class World:
    """A booted mock engine with the real holodeck loaded."""

    def __init__(self, cvars=None, load=("mentat/holodeck.lua",), server=True, gamemode=False):
        try:
            from lupa import luajit21 as mod
        except ImportError:  # pragma: no cover
            from lupa import lua51 as mod
        self.rt = mod.LuaRuntime(unpack_returned_tuples=True)
        self.rt.execute("jit.off()")
        self.rt.execute(f"SERVER = {'true' if server else 'false'}")
        self.rt.execute((HOLO / "gmod_mock.lua").read_text(encoding="utf-8"))
        self.rt.execute(f'MOCK.ROOT = "{ADDON.as_posix()}"')
        self.rt.globals().MOCK.json_decode = self._decode
        self.rt.execute(HELPERS)
        if gamemode:
            self.rt.execute("GAMEMODE = {}")
        for k, v in (cvars or {}).items():
            self.rt.eval("function(k, v) CreateConVar(k, v) end")(k, v)
        for path in load:
            self.rt.execute(f'dofile(MOCK.ROOT .. "/{path}")')

    def _decode(self, s):
        try:
            return lua_table(self.rt, json.loads(s))
        except Exception:
            return None

    def lua(self, code):
        return self.rt.eval(code) if code.strip().startswith("function") else self.rt.execute(code)

    def call(self, name, *args):
        return self.rt.eval(name)(*args)

    def player(self, sid, **kw):
        t = ", ".join(f"{k} = {json.dumps(v)}" for k, v in kw.items())
        return self.rt.eval(f'T.player("{sid}", {{{t}}})')

    def ask(self, p, text):
        return self.call("T.ask", p, text)

    def count(self, sid=None):
        return self.call("T.count", sid) if sid else self.call("T.count")

    def post_json(self, i, obj):
        self.rt.eval("function(i, code, body) MOCK.respond(i, code, body) end")(i, 200, json.dumps(obj))

    def advance(self, dt):
        self.rt.eval("function(dt) MOCK.advance(dt) end")(dt)


A, B, C = "76561190000000001", "76561190000000002", "76561190000000003"


@pytest.fixture
def world():
    return World()


# ------------------------------------------------------------------------------------------------ undo, ids and owners
def test_undo_twice_after_a_clear_undoes_the_original_add(world):
    """glue-1: restoring a cleared group keeps the ids, so the older 'add' on the undo stack still finds its props."""
    p = world.player(A)
    world.ask(p, "add three couches")
    assert world.count(A) == 3
    ids = world.call("T.ids")
    world.ask(p, "clear everything")
    assert world.count() == 0
    world.ask(p, "undo")
    assert world.count(A) == 3 and world.call("T.ids") == ids
    world.ask(p, "undo")
    assert world.count() == 0


def test_undo_of_a_remove_then_undo_of_the_add_leaves_nothing(world):
    """glue-1: 'add a couch and a lamp', 'remove the lamp', 'undo', 'undo' leaves no stray lamp."""
    p = world.player(A)
    world.ask(p, "add a couch and a lamp")
    assert world.count() >= 2
    world.ask(p, "remove the lamp")
    world.ask(p, "undo")
    world.ask(p, "undo")
    assert world.count() == 0


def test_an_admins_clear_and_undo_gives_everyone_their_things_back(world):
    """glue-8: restored props keep their original owner, so the builder can still remove them."""
    p, admin = world.player(A), world.player(B, admin=True)
    world.ask(p, "add three couches")
    owners = world.call("T.owners")
    world.ask(admin, "clear everything")
    assert world.count() == 0
    world.ask(admin, "undo")
    assert world.call("T.owners") == owners
    world.ask(p, "undo")
    assert world.count() == 0


def test_undo_of_a_clear_cannot_exceed_the_caps(world):
    """glue-2 / security-1: the undo restore honours holodeck_max_ents and holodeck_max_per_player."""
    w = World(cvars={"holodeck_max_ents": "6", "holodeck_max_per_player": "4"})
    p1, p2, p3 = w.player(A), w.player(B, x=900), w.player(C, x=-900)
    w.ask(p1, "add four lamps")
    mine = w.count(A)
    assert mine == 4
    w.ask(p1, "clear everything")
    w.ask(p2, "add three lamps")
    w.ask(p3, "add three lamps")
    w.ask(p1, "undo")
    assert w.count() <= 6
    assert w.count(A) <= 4
    for _ in range(10):
        w.ask(p1, "clear everything")
        w.ask(p1, "undo")
    assert w.count() <= 6 and w.count(A) <= 4


def test_a_player_at_their_limit_gets_a_plain_refusal_and_no_empty_undo(world):
    """glue-12: no planner run, no empty undo entry, an honest message."""
    w = World(cvars={"holodeck_max_per_player": "2"})
    p = w.player(A)
    w.ask(p, "add two lamps")
    assert w.count(A) == 2
    depth = w.rt.eval("function() return #Holodeck.undo_stacks['" + A + "'] end")()
    w.ask(p, "add a chair")
    assert w.count(A) == 2
    assert w.rt.eval("function() return #Holodeck.undo_stacks['" + A + "'] end")() == depth
    assert "limit" in w.rt.eval("function() return Holodeck.last_say or '' end")()


def test_clear_copes_with_a_huge_registry(world):
    """security-1: clear used unpack(H.order), which fails past a few thousand entries."""
    w = World(cvars={"holodeck_max_ents": "9000"})
    admin = w.player(A, admin=True)
    w.lua("""function() for i = 1, 8200 do
      Holodeck.registry[i] = { id = i, owner = 'console', kind = 'lamp', seq = i, x = 0, y = 0, z = 0, theta = 0, w = 1, d = 1, h = 1 }
      Holodeck.order[i] = i end
      Holodeck.next_id = 8201 end""")()
    w.ask(admin, "clear everything")
    assert w.count() == 0


# ------------------------------------------------------------------------------------------------ disconnects and agents
def agent_world(**cv):
    cvars = {"holodeck_bridge": "http://127.0.0.1:9", "holodeck_agent": "2", "holodeck_persona": "0", "holodeck_voice_out": "0"}
    cvars.update(cv)
    return World(cvars=cvars)


def test_an_agent_answer_for_a_player_who_left_is_dropped():
    """glue-5 / security-3: no console-privileged build at the map origin after a disconnect."""
    w = agent_world()
    p = w.player(A, x=3000, y=-2000)
    w.rt.eval("function(p) Holodeck.Handle(p, 'add three couches') end")(p)
    posts = list(w.rt.eval("function() return T.posts('/plan') end")().values())
    assert len(posts) == 1
    w.rt.eval("function(p) MOCK.disconnect(p) end")(p)
    w.post_json(posts[0], {"request": {"action": "add", "items": [{"kind": "couch", "count": 3}]}, "say": ""})
    w.advance(5)
    assert w.count() == 0


def test_the_agent_may_only_add_by_default():
    """security-3: a prompt-steered 'clear' from the agent is refused unless the server allows it."""
    w = agent_world()
    p, q = w.player(A), w.player(B, x=900)
    w.lua("function() GetConVar('holodeck_agent'):SetValue('0') end")()
    w.ask(q, "add two lamps")
    w.lua("function() GetConVar('holodeck_agent'):SetValue('2') end")()
    before = w.count()
    assert before == 2
    w.rt.eval("function(p) Holodeck.Handle(p, 'add three couches') end")(p)
    posts = list(w.rt.eval("function() return T.posts('/plan') end")().values())
    w.post_json(posts[-1], {"request": {"action": "clear"}, "say": ""})
    w.advance(3)
    assert w.count() == before
    assert "only add" in w.rt.eval("function() return Holodeck.last_say or '' end")()


def test_a_malformed_agent_answer_falls_back_to_the_plain_request():
    """security-11: wrong types in the agent's JSON never throw or run twice."""
    w = agent_world()
    p = w.player(A)
    w.rt.eval("function(p) Holodeck.Handle(p, 'add three couches') end")(p)
    posts = list(w.rt.eval("function() return T.posts('/plan') end")().values())
    w.rt.eval("function(i) MOCK.respond(i, 200, 'not json at all') end")(posts[0])
    w.advance(5)
    assert w.count(A) == 3

    w2 = agent_world()
    p2 = w2.player(B)
    w2.rt.eval("function(p) Holodeck.Handle(p, 'add two lamps') end")(p2)
    posts = list(w2.rt.eval("function() return T.posts('/plan') end")().values())
    w2.post_json(posts[0], {"say": 5, "request": {"action": "add", "items": [{"kind": 7}, "x", {"kind": "floor lamp", "count": "2"}]}})
    w2.advance(5)
    assert 1 <= w2.count(B) <= 2


def test_one_agent_question_per_player_at_a_time():
    """glue-9: a player cannot queue thirty agent calls inside the HTTP timeout."""
    w = agent_world()
    p = w.player(A)
    for _ in range(4):
        w.rt.eval("function(p) Holodeck.Handle(p, 'add three couches'); MOCK.advance(1.3) end")(p)
    posts = list(w.rt.eval("function() return T.posts('/plan') end")().values())
    assert len(posts) == 1
    assert w.count() == 0


# ------------------------------------------------------------------------------------------------ the prop allowance
def test_a_newcomer_is_served_by_taking_down_the_props_of_players_who_left():
    """glue-10 / security-9: leavers cannot lock the holodeck against everyone else."""
    w = World(cvars={"holodeck_max_ents": "6", "holodeck_orphan_minutes": "0"})
    a, b, c = w.player(A), w.player(B, x=900), w.player(C, x=-900)
    w.ask(a, "add three lamps")
    w.ask(b, "add three lamps")
    assert w.count() == 6
    w.rt.eval("function(p) MOCK.disconnect(p) end")(a)
    w.rt.eval("function(p) MOCK.disconnect(p) end")(b)
    w.ask(c, "add three lamps")
    assert w.count(C) == 3
    assert w.count() <= 6


def test_a_player_holding_more_than_a_fair_share_is_trimmed_for_a_newcomer():
    """security-9: three connected builders cannot fill the server and lock the fourth player out."""
    w = World(cvars={"holodeck_max_ents": "8", "holodeck_max_per_player": "4"})
    ps = [w.player(str(1000 + i), x=900 * i) for i in range(3)]
    for p in ps:
        w.ask(p, "add three lamps")
    d = w.player("2000", x=-900)
    w.ask(d, "add two lamps")
    assert w.count("2000") == 2
    assert w.count() <= 8


def test_orphans_are_swept_after_the_configured_time():
    """security-9: the props of a player who left go after holodeck_orphan_minutes."""
    w = World(cvars={"holodeck_orphan_minutes": "1"})
    a = w.player(A)
    w.ask(a, "add three lamps")
    w.rt.eval("function(p) MOCK.disconnect(p) end")(a)
    w.advance(30)
    assert w.count() == 3
    w.advance(120)
    assert w.count() == 0


def test_the_servers_own_spawn_rules_apply():
    """security-10: a PlayerSpawnProp hook that says no is honoured, and nothing is built."""
    w = World()
    w.lua("""function() hook.Add('PlayerSpawnProp', 'jail', function(ply, model) return false end) end""")()
    p = w.player(A)
    w.ask(p, "add three couches")
    assert w.count() == 0
    assert w.rt.eval("function() return MOCK.live_props() end")() == 0


def test_props_are_owned_by_the_player_who_asked():
    """security-10: prop-protection addons see the player as the creator."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.advance(2)
    same = w.rt.eval("""function(p) for _, e in ipairs(MOCK.ents) do if e.valid and e.creator == p then return true end end return false end""")(p)
    assert same


# ------------------------------------------------------------------------------------------------ speech: who hears what
def test_skippys_chat_goes_to_the_requester_and_those_near_not_the_whole_server():
    """glue-11 / security-5: no broadcast; a distant player sees nothing."""
    w = World()
    near, far = w.player(A), w.player(B, x=9000)
    w.ask(near, "add three couches")
    says = list(w.rt.eval("function() return T.nets('holodeck_say') end")().values())
    assert says
    for m in says:
        assert m.kind == "send"
        recipients = [m.to[i].sid for i in range(1, len(m.to) + 1)]
        assert A in recipients and B not in recipients


def test_skippys_lines_are_clean_text():
    """security-5: control characters and runaway length never reach chat."""
    w = World()
    p = w.player(A)
    w.rt.eval("function(p) Holodeck.Say(p, 'hello\\1\\2 world\\n\\n' .. string.rep('x', 1000)) end")(p)
    says = list(w.rt.eval("function() return T.nets('holodeck_say') end")().values())
    text = says[-1].writes[1]
    assert "\x01" not in text and "\n" not in text and len(text) <= 240


def test_the_admin_only_refusal_is_private_and_throttled():
    """security-4: refusing costs one private line, never a broadcast or a bridge call."""
    w = World(cvars={"holodeck_admin_only": "1", "holodeck_bridge": "http://127.0.0.1:9"})
    p = w.player(A)
    w.rt.eval("function(p) for i = 1, 100 do Holodeck.Handle(p, 'holodeck x') end end")(p)
    nets = list(w.rt.eval("function() return T.nets('holodeck_say') end")().values())
    assert len(nets) <= 1 and all(m.kind == "send" for m in nets)
    assert len(list(w.rt.eval("function() return MOCK.https end")().values())) == 0


def test_the_cooldown_is_checked_before_anything_else():
    """security-4: a second request inside the rest period is ignored outright."""
    w = World()
    p = w.player(A)
    w.rt.eval("function(p) Holodeck.Handle(p, 'add two lamps'); Holodeck.Handle(p, 'add two lamps') end")(p)
    w.advance(5)
    assert w.count(A) == 2


# ------------------------------------------------------------------------------------------------ the bridge
def bridge_world(**cv):
    cvars = {"holodeck_bridge": "http://127.0.0.1:9", "holodeck_agent": "1", "holodeck_persona": "1", "holodeck_voice_out": "1"}
    cvars.update(cv)
    return World(cvars=cvars)


def test_fixed_answers_never_cost_a_model_call():
    """glue-9: 'nothing to undo' and the limit messages skip the persona call; limit and error lines are not voiced."""
    w = bridge_world(holodeck_max_per_player="1")
    p = w.player(A)
    w.ask(p, "undo")
    assert len(list(w.rt.eval("function() return T.posts('/reply') end")().values())) == 0
    w.ask(p, "add a lamp")
    w.ask(p, "add a chair")
    sent = list(w.rt.eval("function() return MOCK.https end")().values())
    assert not any("limit" in h.body for h in sent if h.url.endswith("/speak"))


def test_bridge_calls_are_rationed_per_player():
    """glue-9: a player asking again and again gets plain text once their share of persona and voice calls is spent."""
    w = bridge_world()
    p = w.player(A)
    for i in range(12):
        w.rt.eval("function(p) Holodeck.Handle(p, 'add a lamp'); MOCK.advance(1.3) end")(p)
    replies = len(list(w.rt.eval("function() return T.posts('/reply') end")().values()))
    assert 1 <= replies <= 4


def test_the_whole_server_shares_one_bridge_budget():
    """glue-9: sixteen players cannot make the bridge work without limit."""
    w = bridge_world(holodeck_bridge_per_min="6")
    players = [w.player(str(3000 + i), x=3000 * i) for i in range(16)]
    for p in players:
        w.rt.eval("function(p) Holodeck.Handle(p, 'add a lamp') end")(p)
    w.advance(3)
    calls = len(list(w.rt.eval("function() return MOCK.https end")().values()))
    assert calls <= 4


def mp3(n):
    return "ID3" + "x" * (n - 3)


def test_audio_goes_out_paced_and_bounded():
    """glue-7 / security-11: a clip is sent a chunk at a time, never the whole thing in one tick."""
    w = bridge_world(holodeck_persona="0")
    p = w.player(A)
    w.rt.eval("function(p) Holodeck.Speak('hello there', nil, p) end")(p)
    w.rt.eval("function() MOCK.respond(1, 200, 'ID3' .. string.rep('x', 100000), {}) end")()
    assert len(list(w.rt.eval("function() return T.nets('holodeck_audio') end")().values())) <= 1
    w.advance(0.05)
    sent_early = len(list(w.rt.eval("function() return T.nets('holodeck_audio') end")().values()))
    w.advance(5)
    msgs = list(w.rt.eval("function() return T.nets('holodeck_audio') end")().values())
    assert sent_early < len(msgs) == 9
    assert all(m.bytes <= 12000 for m in msgs)


def test_audio_that_is_too_big_or_not_audio_is_refused():
    """security-11: the relay ignores a body that is not an mp3 or is larger than a sane spoken line."""
    w = bridge_world(holodeck_persona="0", holodeck_bridge_per_min="600")
    p = w.player(A)
    for body in ("ID3" + "x" * 300000, "<html>" + "x" * 5000):
        w.rt.eval("function(p) Holodeck.Speak('hello there', nil, p); MOCK.advance(10) end")(p)
        n = len(list(w.rt.eval("function() return MOCK.https end")().values()))
        w.rt.eval("function(i, b) MOCK.respond(i, 200, b, {}) end")(n, body)
    w.advance(5)
    assert len(list(w.rt.eval("function() return T.nets('holodeck_audio') end")().values())) == 0


# ------------------------------------------------------------------------------------------------ the planner's allowance
def capture_ctx(w):
    w.lua("""function()
      local run = Holodeck.planner.run
      Holodeck.planner.run = function(self, req, ctx) CAPTURED = ctx; return run(self, req, ctx) end
    end""")()


def test_players_get_a_small_planner_budget_admins_a_larger_one():
    """glue-4 / security-2: the planner's time allowance is set by the glue, with a high resolution clock."""
    w = World()
    capture_ctx(w)
    p, admin = w.player(A), w.player(B, admin=True, x=900)
    w.ask(p, "add a lamp")
    low = w.rt.eval("function() return CAPTURED.budget, CAPTURED.max_total, CAPTURED.now == SysTime end")()
    assert low == (0.05, 40, True)
    w.ask(admin, "add a lamp")
    high = w.rt.eval("function() return CAPTURED.budget, CAPTURED.max_total end")()
    assert high == (0.25, 150)


def test_the_planner_hears_about_the_other_players_and_what_was_made_last():
    """planner-14 / planner-21: other players are obstacles; '@last' means this player's own newest group."""
    w = World()
    capture_ctx(w)
    p, other = w.player(A), w.player(B, x=400)
    w.ask(p, "add two lamps")
    w.ask(p, "add a chair")
    n_players, n_last = w.rt.eval("function() return #CAPTURED.players, #CAPTURED.last end")()
    assert n_players == 1 and n_last == 2


def test_a_prop_that_was_pushed_is_planned_around_where_it_really_is():
    """glue-13: the registry is resynced with the engine before planning."""
    w = World()
    capture_ctx(w)
    p = w.player(A)
    w.ask(p, "add a couch")
    w.advance(3)
    before = w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; return e.x, e.y end")()
    w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; e.ent:SetPos(e.ent:GetPos() + Vector(300, 0, 0)) end")()
    w.ask(p, "add a lamp")
    after = w.rt.eval("function() return CAPTURED.registry[1].x, CAPTURED.registry[1].y end")()
    assert abs(after[0] - before[0] - 300) < 1.0 and abs(after[1] - before[1]) < 1.0


def test_a_knocked_over_prop_counts_as_what_it_now_covers():
    """glue-13: a tipped prop is an axis-aligned obstacle, not its upright footprint."""
    w = World()
    capture_ctx(w)
    p = w.player(A)
    w.ask(p, "add a couch")
    w.advance(3)
    w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; e.ent.ang = Angle(90, 0, 0) end")()
    w.ask(p, "add a lamp")
    tilted = w.rt.eval("function() return CAPTURED.registry[1].tilted end")()
    assert tilted


def test_clearance_sees_players_and_foreign_props_but_not_its_own_props():
    """glue-13: the clearance trace uses a solid mask with a filter."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a lamp")
    ok = w.rt.eval("""function()
      local t = MOCK.last_hull
      if not t or t.mask ~= MASK_SOLID or not t.filter then return false end
      local ours = { HolodeckId = 5, IsPlayer = function() return false end }
      local theirs = { IsPlayer = function() return false end }
      local dead = { IsPlayer = function() return true end, Alive = function() return false end, GetMoveType = function() return 2 end }
      local walking = { IsPlayer = function() return true end, Alive = function() return true end, GetMoveType = function() return 2 end }
      return not t.filter(ours) and t.filter(theirs) and not t.filter(dead) and t.filter(walking)
    end""")()
    assert ok


def test_new_props_let_players_through_until_set_free():
    """glue-13: spawned passable while they materialise; a player standing inside keeps them passable until they step out."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    ent = w.rt.eval("function() return MOCK.ents[#MOCK.ents] end")()
    assert w.rt.eval("function(e) return e.group end")(ent) in (0, 15)
    w.advance(3)
    assert w.rt.eval("function(e) return e.group end")(ent) == 0
    # a player walking into the footprint delays the release
    w2 = World()
    p2 = w2.player(A)
    w2.rt.eval("function(p) Holodeck.Handle(p, 'add a couch') end")(p2)
    w2.advance(0.2)
    e2 = w2.rt.eval("function() return MOCK.ents[#MOCK.ents] end")()
    w2.rt.eval("function(p, e) p.pos = Vector(e.pos.x, e.pos.y, 0) end")(p2, e2)
    w2.advance(0.8)
    assert w2.rt.eval("function(e) return e.group end")(e2) == 15
    w2.rt.eval("function(p) p.pos = Vector(5000, 5000, 0) end")(p2)
    w2.advance(1.5)
    assert w2.rt.eval("function(e) return e.group end")(e2) == 0


def test_catalog_mass_is_applied_to_the_spawned_prop():
    """fidelity: a variant's documented real-world mass reaches the physics object."""
    w = World()
    p = w.player(A)
    w.lua("""function()
      for _, k in pairs(Holodeck.catalog.kinds) do for _, v in ipairs(k.variants) do v.mass = 77.5 end end
    end""")()
    w.ask(p, "add a couch")
    w.advance(1)
    mass = w.rt.eval("function() local e = MOCK.ents[#MOCK.ents]; return e.phys.mass end")()
    assert mass == 77.5


# ------------------------------------------------------------------------------------------------ the voice rule
def voice_world(**cv):
    w = World(cvars=cv, load=("mentat/voice.lua",), gamemode=True)
    w.lua("""function()
      GAMEMODE.PlayerCanHearPlayersVoice = function(gm, l, t) return true, false end
      hook.Run('PostGamemodeLoaded')
    end""")()
    return w


def test_a_gag_always_wins_whatever_the_hook_order():
    """glue-14 / security-6: Mentat is the gamemode's rule, so every hook (a gag) answers first."""
    w = voice_world()
    a = w.player(A)
    b = w.player(B, x=100)
    w.lua("""function()
      hook.Add('PlayerCanHearPlayersVoice', 'ULXGag', function(l, t) if t.gagged then return false end end)
    end""")()
    w.rt.eval("function(b) b.gagged = true end")(b)
    heard, _ = w.rt.eval("function(a, b) return hook.Run('PlayerCanHearPlayersVoice', a, b) end")(a, b)
    assert heard is False
    b2 = w.player("76561190000000009", x=100)
    heard, positional = w.rt.eval("function(a, b) return hook.Run('PlayerCanHearPlayersVoice', a, b) end")(a, b2)
    assert heard is True and positional is True


def test_the_game_rule_still_holds_and_modes_work():
    """security-6: the gamemode's refusal stands; global is flat; off defers to the game."""
    w = voice_world()
    a, b = w.player(A), w.player(B, x=100)
    far = w.player(C, x=50000)
    run = w.rt.eval("function(a, b) return hook.Run('PlayerCanHearPlayersVoice', a, b) end")
    assert run(a, far)[0] is False
    w.lua("function() GetConVar('mentat_voice_mode'):SetValue('global') end")()
    assert run(a, far) == (True, False)
    w.lua("function() GetConVar('mentat_voice_mode'):SetValue('off') end")()
    assert run(a, far) == (True, False)          # the stub game rule says: hear everyone, flat
    w.lua("""function()
      GetConVar('mentat_voice_mode'):SetValue('proximity')
      GAMEMODE.MentatVoiceBase = function() return false, false end       -- a game that refuses everyone (team voice, say)
      hook.Run('PostGamemodeLoaded')
    end""")()
    assert run(a, b)[0] is False


def test_voice_is_installed_once_and_a_refresh_does_not_stack():
    """glue-14: reloading voice.lua replaces the wrapper and keeps the original rule."""
    w = voice_world()
    w.lua("""function() dofile(MOCK.ROOT .. '/mentat/voice.lua'); hook.Run('PostGamemodeLoaded') end""")()
    a, b = w.player(A), w.player(B, x=100)
    assert w.rt.eval("function(a, b) return hook.Run('PlayerCanHearPlayersVoice', a, b) end")(a, b) == (True, True)


def test_the_clients_are_told_when_not_to_fade_voices():
    """glue-6: global and off publish range 0 and the mode; the wall switch publishes a wall factor of 1."""
    w = voice_world()
    a = w.player(A)
    w.advance(1)
    assert w.rt.eval("function() return GetGlobalString('mentat_voice_mode') end")() == "proximity"
    assert w.rt.eval("function(p) return p:GetNWFloat('mentat_voice_range') end")(a) > 0
    w.lua("function() GetConVar('mentat_voice_mode'):SetValue('global') end")()
    assert w.rt.eval("function(p) return p:GetNWFloat('mentat_voice_range') end")(a) == 0
    w.lua("function() GetConVar('mentat_voice_mode'):SetValue('proximity'); GetConVar('mentat_voice_wall'):SetValue('0') end")()
    assert w.rt.eval("function() return GetGlobalFloat('mentat_voice_wallf') end")() == 1


def test_a_shout_lapses_unless_the_client_keeps_asking():
    """security-6: a lost key-up cannot leave a player shouting for ever."""
    w = voice_world()
    a = w.player(A)
    w.rt.eval("function() MOCK.rd = { bool = true } end")()
    w.rt.eval("function(p) MOCK.receivers['mentat_voice_shout'](0, p) end")(a)
    loud = w.rt.eval("function(p) return Mentat.Voice.TalkerRange(p) end")(a)
    w.advance(3)
    quiet = w.rt.eval("function(p) return Mentat.Voice.TalkerRange(p) end")(a)
    assert loud > quiet


# ------------------------------------------------------------------------------------------------ the client files
def client_world(load=("mentat/skippy_client.lua",)):
    w = World(server=False, load=())
    w.lua("""function() MOCK.me = { IsValid = function() return true end } end""")()
    for path in load:
        w.rt.execute(f'dofile(MOCK.ROOT .. "/{path}")')
    return w


def deliver(w, idx, total, data, clip=1):
    w.rt.eval("""function(id, idx, total, data)
      MOCK.deliver({ name = 'holodeck_audio', writes = { id, idx, total }, vector = Vector(0, 0, 0), data = data })
    end""")(clip, idx, total, data)


def test_a_playing_voice_is_kept_alive_until_it_ends_then_cleaned_up():
    """glue-3: the channel is held (not left to the garbage collector) and its file is removed after it ends."""
    w = client_world()
    deliver(w, 1, 2, "ID3" + "a" * 100)
    deliver(w, 2, 2, "b" * 100)
    assert w.rt.eval("function() return #MOCK.sounds end")() == 1
    w.lua("function() collectgarbage(); collectgarbage() end")()
    assert w.rt.eval("function() return MOCK.sounds[1].valid end")() is True
    assert w.rt.eval("function() return MOCK.files['mentat/skippy_1.mp3'] ~= nil end")() is True
    w.rt.eval("function() MOCK.sounds[1].state = GMOD_CHANNEL_STOPPED end")()
    w.advance(1)
    assert w.rt.eval("function() return MOCK.sounds[1].valid end")() is False
    assert w.rt.eval("function() return MOCK.files['mentat/skippy_1.mp3'] == nil end")() is True


def test_malformed_audio_messages_are_ignored():
    """security-11: a zero total, an index past the total, or a changing total never plays anything."""
    w = client_world()
    deliver(w, 1, 0, "ID3" + "a" * 100)
    deliver(w, 3, 2, "ID3" + "a" * 100)
    deliver(w, 1, 2, "ID3" + "a" * 100, clip=2)
    deliver(w, 1, 3, "b" * 100, clip=2)
    assert w.rt.eval("function() return #MOCK.sounds end")() == 0


def test_the_chat_line_names_whom_skippy_answered():
    """glue-11: people near by see who was answered; the requester sees the plain tag."""
    w = World(server=False, load=())
    w.lua("""function() MOCK.me = { IsValid = function() return true end, Nick = function() return 'Alice' end } end""")()
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/holodeck_client.lua")')
    for who in ("Alice", "Bob"):
        w.rt.eval("function(t, who) MOCK.deliver({ name = 'holodeck_say', writes = { t, who } }) end")("Three couches.", who)
    chat = list(w.rt.eval("function() return MOCK.chat end")().values())
    assert chat[0].startswith("[Skippy] ") and chat[1].startswith("[Skippy to Bob] ")


# ------------------------------------------------------------------------------------------------ what the planner asks of the engine
def test_the_engine_can_say_how_high_the_room_is():
    """planner-17: env.ceiling_z answers with the ceiling's height, or nothing outdoors (a pile is laid flat under a low ceiling)."""
    w = World()
    capture_ctx(w)
    p = w.player(A)
    w.ask(p, "add a lamp")
    assert w.rt.eval("function() return CAPTURED.env.ceiling_z(0, 0, 0) end")() is None
    w.lua("function() MOCK.ceiling = 96 end")()
    assert w.rt.eval("function() return CAPTURED.env.ceiling_z(0, 0, 0) end")() == 96


def test_a_load_rides_with_its_support_even_when_it_belongs_to_someone_else():
    """planner-11: moving a desk carries what stands on it; the carried prop need not be the mover's own."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a desk with a lamp on it")
    w.advance(3)
    ids = w.rt.eval("function() return T.ids() end")()
    assert w.count(A) >= 2, ids
    lamp = w.rt.eval("""function()
      for _, id in ipairs(Holodeck.order) do
        local e = Holodeck.registry[id]
        if e.kind:find('lamp') then e.owner = 'someone-else'; return id end
      end
    end""")()
    assert lamp is not None
    before = w.rt.eval("function(id) local e = Holodeck.registry[id]; return e.x, e.y end")(lamp)
    w.ask(p, "move the desk away from me")
    after = w.rt.eval("function(id) local e = Holodeck.registry[id]; return e.x, e.y end")(lamp)
    assert (abs(after[0] - before[0]) + abs(after[1] - before[1])) > 1.0, "the lamp stayed behind"


def test_a_search_that_ran_out_of_time_says_a_constant_sentence_without_a_model_call():
    """glue-4 / glue-9: res.timed_out is a fixed outcome: no persona call."""
    w = bridge_world()
    p = w.player(A)
    w.lua("""function()
      local run = Holodeck.planner.run
      Holodeck.planner.run = function(self, req, ctx)
        local res = run(self, req, ctx)
        res.timed_out, res.say = true, 'That is more than I can fit here.'
        return res
      end
    end""")()
    w.ask(p, "add a lamp")
    assert len(list(w.rt.eval("function() return T.posts('/reply') end")().values())) == 0


def test_the_game_keeps_saying_that_the_player_is_transmitting():
    """python-3: while the key is down ptt.txt is rewritten about twice a second, so the microphone bridge can tell a crash from a press."""
    w = client_world()
    w.lua("function() MOCK.cmds['+skippy']() end")()
    before = w.rt.eval("function() return MOCK.write_count['mentat/ptt.txt'] end")()
    w.advance(3)
    during = w.rt.eval("function() return MOCK.write_count['mentat/ptt.txt'] end")()
    assert during - before >= 5
    w.lua("function() MOCK.cmds['-skippy']() end")()
    after_release = w.rt.eval("function() return MOCK.write_count['mentat/ptt.txt'] end")()
    w.advance(3)
    assert w.rt.eval("function() return MOCK.write_count['mentat/ptt.txt'] end")() == after_release


def test_a_request_heard_long_ago_is_not_sent():
    """security-13: a stale voice_in.json (the game was busy, a file left over) is dropped, a fresh one is sent to the server."""
    w = client_world()
    w.lua("""function()
      MOCK.files['mentat/voice_in.json'] = '{"text":"Skippy add a lamp","ts":' .. (os.time() - 120) .. '}'
    end""")()
    w.advance(1)
    assert len(list(w.rt.eval("function() return T.nets('holodeck_voice') end")().values())) == 0
    w.lua("""function()
      MOCK.files['mentat/voice_in.json'] = '{"text":"Skippy add a lamp","ts":' .. os.time() .. '}'
    end""")()
    w.advance(1)
    assert len(list(w.rt.eval("function() return T.nets('holodeck_voice') end")().values())) == 1


def test_holding_the_talk_key_tells_the_microphone_bridge_without_any_bind():
    """The server may not bind keys for a player (ConCommand is blocked! (bind)), so the client watches skippy_key itself: holding
    it writes 'direct' to ptt.txt (no wake word), releasing it writes off, and it is ignored while typing or with a menu open."""
    w = client_world()
    w.lua("""function() MOCK.me.IsSpeaking = function() return false end; MOCK.me.IsTyping = function() return false end end""")()
    w.advance(0.5)
    assert w.rt.eval("function() return MOCK.files['mentat/ptt.txt']:sub(1, 1) end")() == "0"
    w.lua("function() input.keys_down[34] = true end")()
    w.advance(0.3)
    held = w.rt.eval("function() return MOCK.files['mentat/ptt.txt'] end")()
    assert held.startswith("1 ") and held.endswith(" direct"), held
    w.lua("function() MOCK.cursor_visible = true end")()                       # the spawn menu is open: the key is for typing there
    w.advance(0.3)
    assert w.rt.eval("function() return MOCK.files['mentat/ptt.txt']:sub(1, 1) end")() == "0"
    w.lua("function() MOCK.cursor_visible = false end")()
    w.advance(0.3)
    assert w.rt.eval("function() return MOCK.files['mentat/ptt.txt']:sub(1, 1) end")() == "1"
    w.lua("function() input.keys_down[34] = false end")()
    w.advance(0.3)
    assert w.rt.eval("function() return MOCK.files['mentat/ptt.txt']:sub(1, 1) end")() == "0"
