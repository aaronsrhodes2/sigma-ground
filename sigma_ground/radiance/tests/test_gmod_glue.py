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
    w.lua("""function() MOCK.me = { IsValid = function() return true end, Nick = function() return 'Alice' end, UserID = function() return 7 end } end""")()
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/holodeck_client.lua")')
    for who, uid in (("Alice", 7), ("Bob", 8)):       # the asker travels as a number (see test_the_requester_sees_the_plain_tag_whatever_their_nickname)
        w.rt.eval("function(t, who, uid) MOCK.deliver({ name = 'holodeck_say', writes = { t, who, uid } }) end")("Three couches.", who, uid)
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
    w.lua("function() MOCK.cvars.skippy_key:SetValue('x') end")()               # the key is opt-in (the launcher sets it)
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


# ================================================================================================ round 2: undo, evictions, budgets
FAKE_CLOCK = """function()
  FAKE = 0
  SysTime = function() return MOCK.now + FAKE end
  local run = Holodeck.planner.run
  CAPTURED_BUDGETS = {}
  Holodeck.planner.run = function(self, req, ctx)
    CAPTURED_BUDGETS[#CAPTURED_BUDGETS + 1] = ctx.budget
    local heavy = false
    for _, it in ipairs(req.items or {}) do if (it.count or 1) >= 10 then heavy = true end end
    FAKE = FAKE + (heavy and 0.05 or 0.001)
    return run(self, req, ctx)
  end
end"""

RECORD_SAYS = """function()
  SAYS = {}
  hook.Add('HolodeckSaid', 'rec', function(p, t) SAYS[#SAYS + 1] = { sid = p and p.sid or 'console', text = t } end)
end"""


def says_to(w, sid):
    n = w.rt.eval("function() return #SAYS end")()
    return [w.rt.eval("function(i) return SAYS[i].text end")(i) for i in range(1, n + 1)
            if w.rt.eval("function(i) return SAYS[i].sid end")(i) == sid]


def tells(w):
    return [m.writes[1] for m in w.rt.eval("function() return T.nets('holodeck_say') end")().values()]


def last_text(w):
    return w.rt.eval("function() return Holodeck.last_say or '' end")()


def overlaps(w):
    """Number of registry pairs standing on the same ground (floor-level footprints that coincide)."""
    return w.rt.eval("""function()
      local L, n = Holodeck.layout, 0
      local es = {}
      for _, id in ipairs(Holodeck.order) do es[#es + 1] = Holodeck.registry[id] end
      for i = 1, #es do for j = i + 1, #es do
        local a, b = es[i], es[j]
        local ra, rb = L.rect(a.x, a.y, a.w, a.d, a.theta), L.rect(b.x, b.y, b.w, b.d, b.theta)
        if math.abs(a.z - b.z) < 1 and L.overlap(ra, rb, 0) and L.penetration(ra, rb) > 8 then n = n + 1 end
      end end
      return n
    end""")()


BLOCK_AT = """function(x, y) MOCK.blockers[1] = { x0 = x - 30, y0 = y - 30, x1 = x + 30, y1 = y + 30, ent = { IsPlayer = function() return false end } } end"""


# ---- (b) evictions only for what is really built
def test_eviction_is_only_for_the_number_really_built():
    """glue-security 25: a player with one slot left who asks for four lamps costs the others one prop, not the whole shortfall."""
    w = World(cvars={"holodeck_max_ents": "11", "holodeck_max_per_player": "4"})
    x, v1, v2 = w.player(A), w.player(B, x=900), w.player(C, x=-900)
    for p in (v1, v2):
        w.ask(p, "add four lamps")
    w.ask(x, "add three lamps")
    assert w.count() == 11 and w.count(A) == 3
    w.ask(x, "add four lamps")
    assert w.count(A) == 4
    assert w.count() == 11, "others lost more props than were built for the requester"


def test_a_vetoed_request_takes_nothing_from_anyone():
    """glue-security 25: the spawn rules are asked before anyone's props come down."""
    w = World(cvars={"holodeck_max_ents": "8", "holodeck_max_per_player": "4"})
    v1, v2, x = w.player(B, x=900), w.player(C, x=-900), w.player(A)
    for p in (v1, v2):
        w.ask(p, "add four lamps")
    w.lua("function() hook.Add('PlayerSpawnProp', 'jail', function(ply) if ply.sid == '%s' then return false end end) end" % A)()
    w.ask(x, "add three lamps")
    assert w.count(B) == 4 and w.count(C) == 4 and w.count(A) == 0


def test_a_player_at_their_own_limit_costs_nobody_else_a_prop():
    """integration 57 / glue-security 25: the personal limit is looked at before any eviction; a request for nothing evicts nothing."""
    w = World(cvars={"holodeck_max_ents": "9", "holodeck_max_per_player": "4"})
    x, v1, v2 = w.player(A), w.player(B, x=900), w.player(C, x=-900)
    w.ask(x, "add four lamps")
    w.ask(v1, "add four lamps")
    w.ask(v2, "add a lamp")
    assert w.count() == 9
    w.ask(x, "add a chair")
    assert w.count(B) == 4 and w.count(A) == 4
    for text in ("add a unicorn", "add something"):
        w.ask(v2, text)
    assert w.count(B) == 4


# ---- (a) undo restores the right thing for the right owner
def test_an_admins_undo_gives_a_rebuilt_owner_no_more_than_their_limit():
    """glue-security 29 / critic 8: restored props count against the OWNER's limit, not the admin's."""
    w = World(cvars={"holodeck_max_per_player": "3"})
    bob, admin = w.player(A), w.player(B, admin=True, x=900)
    w.ask(bob, "add three lamps")
    w.ask(admin, "clear everything")
    w.ask(bob, "add three lamps")
    w.ask(admin, "undo")
    assert w.count(A) == 3


def test_restored_props_are_created_as_their_owners():
    """glue-security 29 / integration 42: creator and CPPI owner are the owner; unowned (not the admin) when the owner is away."""
    w = World()
    a, admin = w.player(A), w.player(B, admin=True, x=900)
    w.ask(a, "add two couches")
    w.advance(3)
    w.ask(admin, "clear everything")
    w.ask(admin, "undo")
    w.advance(3)
    live = w.rt.eval("""function() local o = {}
      for _, e in ipairs(MOCK.ents) do
        if e.valid then o[#o + 1] = (e.creator and e.creator.sid or 'none') .. '/' .. (e.cppi and e.cppi.sid or 'none') end
      end
      return o end""")()
    assert list(live.values()) == [A + "/" + A] * 2
    w.ask(admin, "clear everything")
    w.rt.eval("function(p) MOCK.disconnect(p) end")(a)
    w.ask(admin, "undo")
    w.advance(3)
    live = w.rt.eval("""function() local o = {}
      for _, e in ipairs(MOCK.ents) do
        if e.valid then o[#o + 1] = (e.creator and e.creator.sid or 'none') .. '/' .. (e.cppi and e.cppi.sid or 'none') end
      end
      return o end""")()
    assert list(live.values()) == ["none/none"] * 2


def test_undo_does_not_put_props_into_ground_someone_else_now_holds():
    """glue-security 26 / integration 52: A clears, B builds in the freed spot, A's undo restores nothing on top of it."""
    w = World()
    a, b = w.player(A), w.player(B)
    w.ask(a, "add a couch")
    w.advance(2)
    w.ask(a, "clear everything")
    w.ask(b, "add a couch")
    w.advance(2)
    w.ask(a, "undo")
    w.advance(2)
    assert w.count() == 1 and overlaps(w) == 0
    assert "something is standing there" in last_text(w)


def test_what_could_not_be_restored_stays_on_the_undo_stack():
    """integration 56 / glue-security 26: a blocked undo keeps the remainder; a later undo puts it back."""
    w = World()
    a, b = w.player(A), w.player(B)
    w.ask(a, "add a couch")
    w.advance(2)
    w.ask(a, "clear everything")
    w.ask(b, "add a couch")
    w.advance(2)
    w.ask(a, "undo")
    assert w.count(A) == 0
    w.ask(b, "clear everything")
    w.ask(a, "undo")
    w.advance(2)
    assert w.count(A) == 1


def test_a_partial_restore_under_the_server_limit_keeps_the_rest():
    """integration 56: 'N did not fit' no longer throws the N away."""
    w = World(cvars={"holodeck_max_ents": "6", "holodeck_max_per_player": "6"})
    a, b, c = w.player(A), w.player(B, x=900), w.player(C, x=-900)
    w.ask(a, "add four lamps")
    w.ask(a, "clear everything")
    w.ask(b, "add three lamps")
    w.ask(c, "add three lamps")
    w.ask(a, "undo")
    first = w.count(A)
    assert 0 < first < 4
    w.ask(b, "clear everything")
    w.ask(c, "clear everything")
    w.ask(a, "undo")
    w.advance(2)
    assert w.count(A) == 4


def test_a_blocked_spot_keeps_the_prop_from_coming_back():
    """glue-security 26: the clearance test sees other addons' props, for restores."""
    w = World()
    a = w.player(A)
    w.ask(a, "add a couch")
    w.advance(2)
    pos = w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; return e.x, e.y end")()
    w.ask(a, "clear everything")
    w.rt.eval(BLOCK_AT)(pos[0], pos[1])
    w.ask(a, "undo")
    assert w.count() == 0
    w.lua("function() MOCK.blockers = {} end")()
    w.ask(a, "undo")
    assert w.count() == 1


def test_undo_of_a_move_does_not_land_on_something_that_arrived():
    """glue-security 26: an undone move leaves the prop where it is when its old spot is taken now, and says so."""
    w = World()
    a = w.player(A)
    w.ask(a, "add a couch")
    w.advance(2)
    old = w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; return e.x, e.y end")()
    w.ask(a, "move the couch away from me")
    moved = w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; return e.x, e.y end")()
    assert abs(moved[0] - old[0]) + abs(moved[1] - old[1]) > 20
    w.rt.eval(BLOCK_AT)(old[0], old[1])
    w.ask(a, "undo")
    now = w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; return e.x, e.y end")()
    assert abs(now[0] - moved[0]) < 1 and abs(now[1] - moved[1]) < 1
    assert "could not be put back" in last_text(w)
    w.lua("function() MOCK.blockers = {} end")()
    w.ask(a, "undo")
    back = w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; return e.x, e.y end")()
    assert abs(back[0] - old[0]) < 1 and abs(back[1] - old[1]) < 1


def desk_with_foreign_lamp():
    w = World()
    p = w.player(A)
    w.ask(p, "add a desk with a lamp on it")
    w.advance(3)
    lamp = w.rt.eval("""function()
      for _, id in ipairs(Holodeck.order) do
        local e = Holodeck.registry[id]
        if e.kind:find('lamp') then e.owner = 'someone-else'; return id end
      end
    end""")()
    return w, p, lamp


def test_undo_of_a_move_puts_back_the_load_of_another_player():
    """integration 40: the load that rode along comes back with its support, whoever owns it."""
    w, p, lamp = desk_with_foreign_lamp()
    before = w.rt.eval("function(id) local e = Holodeck.registry[id]; return e.x, e.y end")(lamp)
    w.ask(p, "move the desk away from me")
    mid = w.rt.eval("function(id) local e = Holodeck.registry[id]; return e.x, e.y end")(lamp)
    assert abs(mid[0] - before[0]) + abs(mid[1] - before[1]) > 1
    w.ask(p, "undo")
    after = w.rt.eval("function(id) local e = Holodeck.registry[id]; return e.x, e.y end")(lamp)
    assert abs(after[0] - before[0]) < 1 and abs(after[1] - before[1]) < 1


def test_undo_does_not_drag_a_load_its_owner_moved_since():
    """integration 40 (verifier note): a carried prop moved by its owner after the move stays where the owner put it."""
    w, p, lamp = desk_with_foreign_lamp()
    w.ask(p, "move the desk away from me")
    w.lua("function(id) local e = Holodeck.registry[id]; e.ent:SetPos(e.ent:GetPos() + Vector(150, 150, 0)) end")(lamp)
    pos = w.rt.eval("function(id) local e = Holodeck.registry[id]; return e.ent:GetPos().x end")(lamp)
    w.ask(p, "undo")
    assert w.rt.eval("function(id) local e = Holodeck.registry[id]; return e.ent:GetPos().x end")(lamp) == pos
    assert "left where" in last_text(w)


# ---- (c) compound sentences share one budget
def test_a_compound_sentence_shares_one_planning_budget_and_one_thing_count():
    """glue-security 27: the clauses of one sentence split 0.05 s and 40 things between them."""
    w = World()
    p = w.player(A)
    w.lua(FAKE_CLOCK)()
    w.ask(p, "add thirty lamps. add thirty chairs. add two tables")
    n = w.rt.eval("function() return #CAPTURED_BUDGETS end")()
    b = [w.rt.eval("function(i) return CAPTURED_BUDGETS[i] end")(i) for i in range(1, n + 1)]
    assert b[0] == 0.05 and all(x < 0.05 for x in b[1:]), b
    w2 = World()
    q = w2.player(A)
    w2.ask(q, "add thirty lamps. add thirty chairs. add thirty tables")
    assert w2.count(A) <= 40
    assert "one go" in last_text(w2)


def test_a_long_compound_sentence_is_cut_off_and_the_player_is_told():
    """glue-security 27: at most eight requests of one sentence, and the rest is said to be dropped."""
    w = World()
    p = w.player(A)
    w.ask(p, ". ".join(["add a lamp"] * 10))
    assert w.count(A) <= 8
    assert any("first 8" in t for t in tells(w))


# ---- (d) fair shares of the planner and the bridge
def test_three_planner_hogs_do_not_lock_out_a_fourth_player():
    """integration 44 / glue-security 28: the planner allowance has a share per player; the hogs are refused alone."""
    w = World()
    hogs = [w.player(str(7000 + i), x=3000 * i) for i in range(3)]
    honest = w.player(A, x=-5000)
    w.lua(FAKE_CLOCK)()
    w.lua(RECORD_SAYS)()
    w.rt.eval("""function(h1, h2, h3, honest)
      local hogs = { h1, h2, h3 }
      for tick = 1, 40 do
        for _, h in ipairs(hogs) do Holodeck.Handle(h, 'add thirty lamps') end
        if tick % 3 == 0 then Holodeck.Handle(honest, 'add a lamp') end
        MOCK.advance(1.1)
      end
    end""")(*hogs, honest)
    mine = says_to(w, A)
    assert mine and not any("busy planning" in t or "share of planning" in t for t in mine)
    assert any("share of planning" in t for t in says_to(w, "7000"))


def test_one_player_cannot_use_up_the_servers_bridge_allowance():
    """glue-security 28: a per-player bridge share below the server's, so an honest player still gets calls."""
    w = bridge_world(holodeck_agent="0")
    hog, honest = w.player(B), w.player(A, x=5000)
    w.rt.eval("""function(hog, honest)
      local answered = 0
      local function answer()
        for i = answered + 1, #MOCK.https do
          local h = MOCK.https[i]
          if h.url:sub(-6) == '/reply' then MOCK.respond(i, 200, '{"say":"Done."}')
          else MOCK.respond(i, 200, 'ID3' .. string.rep('x', 400), {}) end
        end
        answered = #MOCK.https
      end
      for t = 1, 100 do
        Holodeck.Handle(hog, 'add a lamp'); answer()
        MOCK.advance(0.2)
        if t % 10 == 0 then Holodeck.Handle(honest, 'add a lamp'); answer() end
        MOCK.advance(1.0)
      end
    end""")(hog, honest)
    mine = {B: 0, A: 0}
    for h in w.rt.eval("function() return MOCK.https end")().values():
        who = json.loads(h.body).get("who")
        if who in mine:
            mine[who] += 1
    assert mine[B] <= 14, mine
    assert mine[A] >= 6, mine


# ---- (e) a prop is never set solid around a person
def test_a_prop_never_goes_solid_around_a_player_who_stays():
    """glue-security 32: it stays passable and the owner is told; it turns solid when the player steps out."""
    w = World()
    p = w.player(A)
    w.rt.eval("function(p) Holodeck.Handle(p, 'add a couch') end")(p)
    w.advance(0.2)
    e = w.rt.eval("function() return MOCK.ents[#MOCK.ents] end")()
    w.rt.eval("function(p, e) p.pos = Vector(e.pos.x, e.pos.y, 0) end")(p, e)
    w.advance(15)
    assert w.rt.eval("function(e) return e.group end")(e) == 15
    assert any("see-through" in t for t in tells(w))
    w.rt.eval("function(p) p.pos = Vector(5000, 5000, 0) end")(p)
    w.advance(3)
    assert w.rt.eval("function(e) return e.group end")(e) == 0


# ---- (f) the reply only counts what the engine really made
def test_props_the_engine_refused_are_reported_and_forgotten():
    """integration 60: five asked, two created: the player is told, and undo does not list ghosts."""
    w = World()
    p = w.player(A)
    w.lua("""function()
      local create, made = ents.Create, 0
      ents.Create = function(class)
        local e = create(class)
        made = made + 1
        if made > 2 then e.valid = false end
        return e
      end
    end""")()
    w.ask(p, "add five lamps")
    w.advance(3)
    assert w.count(A) == 2
    assert any("Only 2 of the 5" in t for t in tells(w))
    w.ask(p, "undo")
    assert "2 things gone" in last_text(w)


def test_a_request_is_not_promised_more_than_the_engine_has_room_for():
    """integration 60: when the engine reports few free entity slots the reply says so up front."""
    w = World()
    p = w.player(A)
    w.lua("function() ents.GetCount = function() return 7898 end end")()
    w.ask(p, "add five lamps")
    w.advance(3)
    assert w.count(A) == 2
    assert "only 2" in last_text(w)


# ------------------------------------------------------------------------------------------------ the talk key must not hijack another key (D1 / finding 31)
def _held(w):
    return w.rt.eval("function() return MOCK.files['mentat/ptt.txt'] end")()


def _quiet_client():
    w = client_world()
    w.lua("""function() MOCK.me.IsSpeaking = function() return false end; MOCK.me.IsTyping = function() return false end end""")()
    return w


def test_the_talk_key_is_off_until_the_player_or_the_launcher_chooses_one():
    """Holding a key must never start sending the microphone to the server unasked: with the default settings, holding X (the key
    the README once gave to the shout) leaves ptt.txt off, and the setting is not saved into the player's config."""
    w = _quiet_client()
    assert w.rt.eval("function() return MOCK.cvars.skippy_key:GetString() end")() == ""
    w.lua("function() input.keys_down[34] = true end")()
    w.advance(1)
    assert _held(w).startswith("0 "), _held(w)
    w.lua("function() MOCK.cvars.skippy_key:SetValue('x') end")()               # the launcher's `skippy_key x`
    w.advance(0.3)
    assert _held(w).startswith("1 ") and _held(w).endswith(" direct")


def test_a_key_that_already_has_another_job_is_never_the_talk_key():
    """The player bound X to the shout (+mentat_shout): holding it to shout at a team mate must not be transcribed and sent."""
    w = _quiet_client()
    w.lua("""function()
      input.LookupKeyBinding = function(code) if code == 34 then return '+mentat_shout' end end
      MOCK.cvars.skippy_key:SetValue('x')
      input.keys_down[34] = true
    end""")()
    w.advance(2)
    assert _held(w).startswith("0 "), _held(w)
    assert w.rt.eval("function() return T.printed('already bound') end")() is True, "the player is told once why the key does nothing"
    shown = w.rt.eval("function() local n = 0; for _, l in ipairs(MOCK.printed) do if l:find('already bound', 1, true) then n = n + 1 end end; return n end")()
    assert shown == 1, "one warning, not one per tick"
    # a key bound to nothing, or to +skippy itself, still works
    w.lua("function() input.LookupKeyBinding = function(code) return nil end end")()
    w.advance(0.3)
    assert _held(w).startswith("1 ") and _held(w).endswith(" direct")
    w.lua("function() input.LookupKeyBinding = function(code) return '+skippy' end end")()
    w.advance(0.3)
    assert _held(w).startswith("1 ")


# ================================================================================================ round 3: engine-glue classes (review F)
def kinds_standing(w):
    return list(w.rt.eval("""function() local o = {}
      for _, id in ipairs(Holodeck.order) do o[#o + 1] = Holodeck.registry[id].kind end
      return o end""")().values())


# ---- one sentence: the cap, what was cut, what "another one" repeats
def test_the_readme_sentence_runs_all_five_requests():
    """review F4/F39: 'put a lamp next to the couch. Turn the chair to face me. Remove the lamp. Undo. Clear the room.' ends with the room cleared."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.ask(p, "add a chair")
    w.ask(p, "put a lamp next to the couch. Turn the chair to face me. Remove the lamp. Undo. Clear the room.")
    w.advance(3)
    assert w.count() == 0
    assert "Cleared" in last_text(w)


def test_the_glue_follows_the_parsers_limit_and_says_what_it_left_out():
    """review F76: one limit for parser and glue (the parser's MAX_REQUESTS); clauses over it are named, not dropped in silence."""
    w = World()
    p = w.player(A)
    w.lua("""function()
      local parse_all = Holodeck.parser.parse_all
      Holodeck.parser.MAX_REQUESTS = 4
      Holodeck.parser.parse_all = function(self, text, prev) return parse_all(self, text, prev) end
    end""")()
    w.ask(p, ". ".join(["add a lamp"] * 6))
    assert w.count(A) == 4
    assert "first 4" in last_text(w)


def test_clauses_that_never_ran_do_not_become_the_thing_to_repeat():
    """review F39: 'another one' repeats the last request that really ran, not a clause the one-sentence budget cut off."""
    w = World()
    p = w.player(A)
    w.ask(p, "add thirty lamps. add thirty chairs. add thirty tables")
    assert "one go" in last_text(w)
    assert "table" not in kinds_standing(w)
    chairs = kinds_standing(w).count("chair")
    w.ask(p, "add another one")
    assert "table" not in kinds_standing(w)
    assert kinds_standing(w).count("chair") == chairs + 1


def test_a_request_over_the_read_limit_says_the_rest_was_not_read():
    """review F76/F55: the parser sees one character more than it reads, so its truncation note reaches the player."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a lamp" + " please" * 60)
    assert w.count(A) >= 1
    assert any("first 400 characters" in t for t in tells(w))


def test_nothing_is_removed_on_half_a_sentence():
    """review F76: a destructive request cut at the limit may have lost its 'except the couch': it is refused with a reason."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.ask(p, "add three chairs")
    before = w.count()
    w.ask(p, "remove everything" + " please" * 58 + " except the couch")
    assert w.count() == before
    assert "too long" in last_tell(w)
    w.ask(p, "remove everything except the couch")
    assert kinds_standing(w) == ["couch"]


def last_tell(w):
    return w.rt.eval("function() return Holodeck.last_tell or '' end")()


def test_another_one_works_after_any_request_not_only_after_an_add():
    """review F54: a turn, a move, a remove or an undo in between no longer leaves 'another one' with nothing to repeat."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.ask(p, "turn the couch around")
    w.ask(p, "add another one")
    assert kinds_standing(w).count("couch") == 2
    w2 = World()
    q = w2.player(B)
    w2.ask(q, "add three chairs")
    w2.ask(q, "undo")
    assert w2.count() == 0
    w2.ask(q, "add another one")
    assert kinds_standing(w2).count("chair") == 1


def test_undoing_an_add_gives_it_back_to_the_earlier_group():
    """review F58: after 'undo', 'it' means the couch again, not the lamp that is gone."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.ask(p, "put a lamp next to it")
    w.ask(p, "undo")
    assert "lamp" not in " ".join(kinds_standing(w))
    w.ask(p, "put a plant next to it")
    assert "nothing recent" not in last_text(w)
    assert "couch" in last_text(w)


# ---- the library changes under standing props and undo snapshots
FORGET_MODEL = """function(gone, still_on_server)
  local real = util.IsValidModel
  util.IsValidModel = function(m) return m ~= gone end
  Holodeck.BuildLibrary()
  util.IsValidModel = still_on_server and real or function(m) return m ~= gone end
end"""


def standing_couch_model(w):
    return w.rt.eval("function() return Holodeck.registry[Holodeck.order[1]].model end")()


def test_a_prop_whose_model_left_the_library_can_still_be_named_and_removed():
    """review F59: the couch stands, its model leaves the library: it is legacy (no index), still named, moved, removed, and not added twice."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.advance(2)
    model = standing_couch_model(w)
    w.lua(FORGET_MODEL)(model, True)
    e = w.rt.eval("function() local e = Holodeck.registry[Holodeck.order[1]]; return e.legacy, e.vi, e.variant and e.variant.model end")()
    assert e == (True, None, model)
    w.ask(p, "move the couch closer")
    assert "no couch" not in last_text(w).lower()
    w.ask(p, "put a lamp next to the couch")
    assert kinds_standing(w).count("couch") == 1
    w.ask(p, "remove the couch")
    assert "couch" not in kinds_standing(w)


def test_undo_never_brings_a_prop_back_under_a_stale_variant():
    """review F43/F33: a cleared prop whose model left the library comes back with ITS variant, marked legacy, never variants[old index]."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.advance(2)
    model = standing_couch_model(w)
    w.ask(p, "clear everything")
    w.lua(FORGET_MODEL)(model, True)
    w.ask(p, "undo")
    w.advance(2)
    got = w.rt.eval("""function() local e = Holodeck.registry[Holodeck.order[1]]
      return e.model, e.legacy, e.vi, e.variant and e.variant.model end""")()
    assert got == (model, True, None, model)


def test_undo_skips_props_whose_model_the_server_no_longer_has_and_says_so():
    """review F33: no engine error models respawned; the player is told how many stayed gone."""
    w = World()
    p = w.player(A)
    w.ask(p, "add two couches")
    w.advance(2)
    models = w.rt.eval("""function() local o = {}
      for _, id in ipairs(Holodeck.order) do o[#o + 1] = Holodeck.registry[id].model end
      return o end""")()
    gone = set(models.values())
    w.ask(p, "clear everything")
    w.lua("function(list) local g = {}; for _, m in pairs(list) do g[m] = true end; util.IsValidModel = function(m) return not g[m] end; Holodeck.BuildLibrary() end")(
        lua_table(w.rt, sorted(gone)))
    live_before = w.rt.eval("function() return MOCK.live_props() end")()
    w.ask(p, "undo")
    w.advance(2)
    assert w.rt.eval("function() return MOCK.live_props() end")() == live_before
    assert w.count() == 0
    assert "no longer has" in last_text(w)


# ---- floor and ceiling see what a person stands on
FROZEN_PROP = """{ IsPlayer = function() return false end, GetClass = function() return 'prop_physics' end,
  GetPhysicsObject = function() return { IsValid = function() return true end, IsMotionEnabled = function() return %s end } end }"""


def platform(w, top, motion, bottom=None):
    w.lua("function() MOCK.blockers[#MOCK.blockers + 1] = { x0 = -900, y0 = -900, x1 = 900, y1 = 900, top = %s, bottom = %s, ent = %s } end"
          % (top, bottom if bottom is not None else "nil", FROZEN_PROP % ("true" if motion else "false")))()


def test_a_requester_on_a_frozen_platform_gets_things_on_the_platform():
    """review F34: the floor probe sees a frozen prop as ground: the couch is planned onto the platform, not 300 HU below it."""
    w = World()
    p = w.player(A)
    w.rt.eval("function(p) p.pos = Vector(0, 0, 300) end")(p)
    platform(w, 300, motion=False)
    w.ask(p, "add a couch")
    z = w.rt.eval("function() return Holodeck.registry[Holodeck.order[1]].z end")()
    assert abs(z - 300) < 1.0


def test_a_loose_prop_is_not_a_floor():
    """review F34: only a FROZEN prop is ground; a crate lying about is not (the old behaviour for loose props stays)."""
    w = World()
    capture_ctx(w)
    p = w.player(A)
    w.ask(p, "add a lamp")
    platform(w, 40, motion=True)
    assert w.rt.eval("function() return CAPTURED.env.floor_z(0, 0) end")() == 0
    w.lua("function() MOCK.blockers = {} end")()
    platform(w, 40, motion=False)
    assert w.rt.eval("function() return CAPTURED.env.floor_z(0, 0) end")() == 40


def test_a_frozen_slab_overhead_is_a_ceiling():
    """review F34: ceiling_z sees the same ground; a loose prop overhead is no ceiling."""
    w = World()
    capture_ctx(w)
    p = w.player(A)
    w.ask(p, "add a lamp")
    platform(w, 200, motion=False, bottom=190)
    assert w.rt.eval("function() return CAPTURED.env.ceiling_z(0, 0, 0) end")() == 190
    w.lua("function() MOCK.blockers = {} end")()
    platform(w, 200, motion=True, bottom=190)
    assert w.rt.eval("function() return CAPTURED.env.ceiling_z(0, 0, 0) end")() is None


def test_ground_ignores_players_and_the_holodecks_own_props():
    """review F34: the ground filter is not `blocks`: a person's head and our own props are not floor."""
    w = World()
    capture_ctx(w)
    p = w.player(A)
    w.ask(p, "add a lamp")
    w.lua("""function() MOCK.blockers[1] = { x0 = -900, y0 = -900, x1 = 900, y1 = 900, top = 72, bottom = 0,
      ent = { IsPlayer = function() return true end, Alive = function() return true end, GetMoveType = function() return 2 end } } end""")()
    assert w.rt.eval("function() return CAPTURED.env.floor_z(0, 0) end")() == 0
    w.lua("""function() MOCK.blockers[1].ent = { HolodeckId = 3, IsPlayer = function() return false end } end""")()
    assert w.rt.eval("function() return CAPTURED.env.floor_z(0, 0) end")() == 0


# ---- the chat tag
def test_the_requester_sees_the_plain_tag_whatever_their_nickname():
    """review F35: who asked travels as a number, so a nickname that clean_text changes still gets '[Skippy]'."""
    for nick in ("Bob  Smith", " padded ", "tab\there", "Дмитрий Васильевич Иванов"):
        server = World()
        server.lua("function() utf8 = { force = function(s) return (s:gsub('[\\192-\\255][\\128-\\191]*$', '?')) end } end")()
        p = server.player(A)
        server.rt.eval("function(p, nick) p.name = nick end")(p, nick)
        server.rt.eval("function(p) Holodeck.Say(p, 'One floor lamp.') end")(p)
        msg = server.rt.eval("function() return T.nets('holodeck_say')[1] end")()
        writes = [msg.writes[i] for i in range(1, 4)]
        uid = server.rt.eval("function(p) return p:UserID() end")(p)
        client = World(server=False, load=())
        client.lua("function(uid, nick) MOCK.me = { IsValid = function() return true end, Nick = function() return nick end, UserID = function() return uid end } end")(uid, nick)
        client.rt.execute('dofile(MOCK.ROOT .. "/mentat/holodeck_client.lua")')
        client.rt.eval("function(a, b, c) MOCK.deliver({ name = 'holodeck_say', writes = { a, b, c } }) end")(*writes)
        other = World(server=False, load=())
        other.lua("function(uid) MOCK.me = { IsValid = function() return true end, Nick = function() return 'Eve' end, UserID = function() return uid + 1 end } end")(uid)
        other.rt.execute('dofile(MOCK.ROOT .. "/mentat/holodeck_client.lua")')
        other.rt.eval("function(a, b, c) MOCK.deliver({ name = 'holodeck_say', writes = { a, b, c } }) end")(*writes)
        mine = list(client.rt.eval("function() return MOCK.chat end")().values())[0]
        theirs = list(other.rt.eval("function() return MOCK.chat end")().values())[0]
        assert mine.startswith("[Skippy] "), (nick, mine)
        assert theirs.startswith("[Skippy to "), (nick, theirs)


# ---- the voice rule
def test_an_unknown_voice_mode_means_proximity_for_the_server_and_the_clients():
    """review F36: 'Proximity', 'PROXIMITY', 'proximity ' and typos are proximity everywhere; 'Global' and ' OFF ' mean what they say."""
    for raw in ("Proximity", "PROXIMITY", "proximity ", "prox", ""):
        w = voice_world()
        a, far = w.player(A), w.player(C, x=50000)
        w.lua("function(v) GetConVar('mentat_voice_mode'):SetValue(v) end")(raw)
        w.advance(1)
        heard = w.rt.eval("function(a, b) return hook.Run('PlayerCanHearPlayersVoice', a, b) end")(a, far)[0]
        assert heard is False, raw                                                  # the server enforces proximity
        assert w.rt.eval("function() return GetGlobalString('mentat_voice_mode') end")() == "proximity", raw
        assert w.rt.eval("function(p) return p:GetNWFloat('mentat_voice_range') end")(a) > 0, raw      # so the clients fade
    for raw, published in (("Global", "global"), ("  GLOBAL", "global"), (" OFF ", "off")):
        w = voice_world()
        a, far = w.player(A), w.player(C, x=50000)
        w.lua("function(v) GetConVar('mentat_voice_mode'):SetValue(v) end")(raw)
        w.advance(1)
        assert w.rt.eval("function(a, b) return hook.Run('PlayerCanHearPlayersVoice', a, b) end")(a, far) == (True, False), raw
        assert w.rt.eval("function() return GetGlobalString('mentat_voice_mode') end")() == published
        assert w.rt.eval("function(p) return p:GetNWFloat('mentat_voice_range') end")(a) == 0


def test_proximity_voice_is_put_back_when_a_later_addon_replaces_the_rule():
    """review F37: the existing timer notices that the gamemode's rule is no longer ours and wraps it again."""
    w = voice_world()
    a, far = w.player(A), w.player(C, x=50000)
    w.advance(1)
    run = w.rt.eval("function(a, b) return hook.Run('PlayerCanHearPlayersVoice', a, b) end")
    assert run(a, far)[0] is False
    w.lua("function() GAMEMODE.PlayerCanHearPlayersVoice = function(gm, l, t) return true, false end end")()
    assert run(a, far)[0] is True                       # for the moment the other addon's rule stands
    w.advance(1)
    w.lua("function() Mentat.Voice.ClearCache() end")()
    assert run(a, far)[0] is False
    near = w.player(B, x=100)
    assert run(a, near)[0] is True
    assert w.rt.eval("function() return T.printed('replaced the gamemode') end")()


# ---- the server's own spawn rules
def spawn_rule_world(rule=None, **kw):
    w = World(gamemode=True, **kw)
    if rule:
        w.lua("function() GAMEMODE.%s = function(gm, ply, model) return false end end" % rule)()
    return w


def test_the_gamemodes_own_prop_rule_is_a_veto():
    """review F30/F7: GM:PlayerSpawnProp and GM:PlayerSpawnObject are consulted (hook.Call with a nil gamemode never reached them)."""
    for rule in ("PlayerSpawnProp", "PlayerSpawnObject"):
        w = spawn_rule_world(rule)
        p = w.player(A)
        w.ask(p, "add three couches")
        assert w.count() == 0, rule
        assert w.rt.eval("function() return MOCK.live_props() end")() == 0
    w = spawn_rule_world()                                # no rule at all: nothing is forbidden
    w.ask(w.player(A), "add three couches")
    assert w.count() == 3


def test_a_registered_hook_answers_before_the_gamemode_as_in_the_engine():
    """review F30: a hook that says yes means the gamemode's function is not asked; a hook that says no is a veto on its own."""
    w = spawn_rule_world("PlayerSpawnProp")
    w.lua("function() hook.Add('PlayerSpawnProp', 'allow', function() return true end) end")()
    w.ask(w.player(A), "add a couch")
    assert w.count() == 1


def test_a_broken_spawn_rule_never_throws_and_is_not_a_veto():
    """review F30: a rule that errors is ignored, the request still runs."""
    import faulthandler
    w = spawn_rule_world()
    w.lua("function() GAMEMODE.PlayerSpawnProp = function() error('boom') end end")()
    was_on = faulthandler.is_enabled()
    faulthandler.disable()           # LuaJIT raises Lua errors as Windows SEH exceptions, which the fault handler would print although pcall catches them
    try:
        w.ask(w.player(A), "add a couch")
    finally:
        if was_on:
            faulthandler.enable()
    assert w.count() == 1


def test_sbox_maxprops_is_asked_once_and_never_twice():
    """review F30/F7: CheckLimit is called by the gamemode's own rule when there is one; the glue asks it itself ONLY when the
    gamemode has no PlayerSpawnProp, so the limit is never counted twice."""
    counting = """function(p) p.limit_calls = 0; p.limit_ok = %s
      p.CheckLimit = function(self, what) self.limit_calls = self.limit_calls + 1; return self.limit_ok end end"""
    w = spawn_rule_world()
    w.lua("function() GAMEMODE.PlayerSpawnProp = function(gm, ply, model) return ply:CheckLimit('props') end end")()
    p = w.player(A)
    w.lua(counting % "true")(p)
    w.ask(p, "add a couch")
    assert w.count() == 1 and w.rt.eval("function(p) return p.limit_calls end")(p) == 1
    w2 = spawn_rule_world()                                # a gamemode with no PlayerSpawnProp: the glue asks CheckLimit itself, once
    q = w2.player(A)
    w2.lua(counting % "true")(q)
    w2.ask(q, "add a couch")
    assert w2.count() == 1 and w2.rt.eval("function(p) return p.limit_calls end")(q) == 1
    w3 = spawn_rule_world()
    r = w3.player(A)
    w3.lua(counting % "false")(r)                          # sbox_maxprops reached
    w3.ask(r, "add a couch")
    assert w3.count() == 0
    w4 = spawn_rule_world()
    s = w4.player(A)
    w4.lua(counting % "false")(s)
    w4.lua("function() GAMEMODE.PlayerSpawnProp = function() return true end end")()   # a gamemode that has its own rule decides alone
    w4.ask(s, "add a couch")
    assert w4.count() == 1 and w4.rt.eval("function(p) return p.limit_calls end")(s) == 0


# ---- a prop whose builder leaves keeps its creator
def test_props_still_materialising_when_their_builder_leaves_keep_their_creator():
    """review F53: the owner is looked up while they can still be, so no prop is made ownerless although the registry says it is theirs."""
    w = World()
    p = w.player(A)
    w.rt.eval("function(p) Holodeck.Handle(p, 'add thirty lamps') end")(p)
    w.advance(0.1)
    made = w.rt.eval("function() return MOCK.live_props() end")()
    assert 0 < made < 30
    w.rt.eval("function(p) MOCK.disconnect(p) end")(p)
    w.advance(8)
    assert w.rt.eval("function() return MOCK.live_props() end")() == 30 == w.count(A)
    orphans = w.rt.eval("""function() local n = 0
      for _, e in ipairs(MOCK.ents) do if e.valid and e.class == 'prop_physics' and (not e.creator or not e.cppi) then n = n + 1 end end
      return n end""")()
    assert orphans == 0


# ---- configuration
def test_the_bridge_token_is_never_saved_to_disk():
    """review F-token: FCVAR_ARCHIVE writes a convar to server.vdf, where a revoked token would stay for ever."""
    w = World()
    flags = w.rt.eval("function() local f = MOCK.cvars.holodeck_bridge_token.flags; return bit.band(f, FCVAR_ARCHIVE), bit.band(f, FCVAR_PROTECTED), bit.band(f, FCVAR_DONTRECORD) end")()
    assert flags[0] == 0 and flags[1] != 0 and flags[2] != 0
    assert w.rt.eval("function() return bit.band(MOCK.cvars.holodeck_bridge.flags, FCVAR_ARCHIVE) end")() != 0


def test_the_bridge_example_names_the_port_the_bridge_listens_on():
    """review F116: the help text of holodeck_bridge shows the port skippy_bridge.py serves on."""
    w = World()
    help_text = w.rt.eval("function() return MOCK.cvars.holodeck_bridge.help end")()
    bridge_src = (HOLO / "skippy_bridge.py").read_text(encoding="utf-8")
    assert "8777" in help_text and "8765" not in help_text
    assert "8777" in bridge_src


# ---- parser fields the glue honours: undo counts, silence for chatter, order
def test_undo_twice_and_everything_undo_that_many_of_the_players_own_steps():
    """coordinator: req.times on an undo request runs that many undo steps (at most 30), never touching other players' work."""
    w = World()
    p, q = w.player(A), w.player(B, x=900)
    w.ask(q, "add a table")
    for k in ("couch", "chair", "lamp"):
        w.ask(p, "add a " + k)
    assert w.count(A) == 3
    w.ask(p, "undo twice")
    assert w.count(A) == 1 and "2 steps" in last_text(w)
    w.ask(p, "add a plant")
    w.ask(p, "add a bed")
    w.ask(p, "undo everything")
    assert w.count(A) == 0 and w.count(B) == 1
    w.ask(p, "undo")
    assert "nothing to undo" in last_text(w)


def test_undo_everything_after_a_clear_brings_the_room_back_then_unbuilds_it():
    """coordinator: an undo count is a count of undo steps, a clear is one step among them."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.ask(p, "add a lamp")
    w.ask(p, "clear everything")
    w.ask(p, "undo the last three")
    assert w.count(A) == 0


def test_chatter_the_parser_calls_nothing_gets_no_answer_at_all():
    """coordinator: a statement is silent: no private line, no chat line, no bridge call."""
    w = bridge_world()
    p = w.player(A)
    before = len(tells(w))
    w.rt.eval("function(p) Holodeck.Handle(p, 'the weather is nice today') end")(p)
    assert len(tells(w)) == before
    assert len(list(w.rt.eval("function() return MOCK.https end")().values())) == 0
    assert w.count() == 0


def test_requests_run_in_order_and_undo_then_remove_works_end_to_end():
    """coordinator: 'remove the couch. No wait, remove the lamp.' puts the couch back, then removes the lamp."""
    w = World()
    p = w.player(A)
    w.ask(p, "add a couch")
    w.ask(p, "add a lamp")
    w.ask(p, "remove the couch. No wait, remove the lamp.")
    w.advance(3)
    assert kinds_standing(w) == ["couch"]
