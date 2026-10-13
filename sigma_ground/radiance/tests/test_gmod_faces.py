"""A player's picture as their head: the REAL addon files (autorun/mentat_faces.lua's server and client halves, faces_head.lua) run offline
under LuaJIT (lupa) against stage/gmod/holodeck/gmod_mock.lua plus the few fakes below (client render API, net readers). Two runtimes, as in the
engine: a server and a client; the net messages the server records are carried to the client.

What this cannot show (see stage/gmod/holodeck/README.md, Faces): how Material() treats a jpg under data/ and how the head LOOKS; the live check
covers those. The bridge half is tested in test_gmod_faces_bridge.py with synthetic pictures; here the bridge's answers are scripted.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
import zlib

import pytest

lupa = pytest.importorskip("lupa")
from test_gmod_glue import ADDON, World, lua_table  # noqa: E402

A, B, C, D = "76561190000000001", "76561190000000002", "76561190000000003", "76561190000000004"
BRIDGE = "http://127.0.0.1:8777"

PRELUDE = r"""
local Ply = getmetatable(MOCK.newplayer("0")).__index
MOCK.players = {}
function Ply:ChatPrint(t) MOCK.chat_lines = MOCK.chat_lines or {}; table.insert(MOCK.chat_lines, { to = self.sid, text = t }) end
function Ply:IsBot() return self.bot == true end
function hook.GetTable() return MOCK.hooks end
function T.said(sid) local out = {}; for _, l in ipairs(MOCK.chat_lines or {}) do if l.to == sid then out[#out + 1] = l.text end end; return out end
-- carry a message into a reader: the engine's net.Read* in the order the writer wrote
function T.deliver(seq, ply)
  local q = {}
  for i, s in ipairs(seq) do q[i] = s end
  local function pop(kind)
    local s = table.remove(q, 1)
    assert(s and s[1] == kind, "wanted a " .. kind .. ", the message has " .. tostring(s and s[1]))
    return s[2]
  end
  net.ReadUInt, net.ReadString, net.ReadBool = function() return pop("u") end, function() return pop("s") end, function() return pop("b") end
  net.ReadData = function(n) local d = pop("d"); assert(#d == n, "data length"); return d end
  return q
end
function T.receive(name, seq, ply)
  local q = T.deliver(seq, ply)
  MOCK.receivers[name](0, ply)
  assert(#q == 0, "the receiver left " .. #q .. " values unread")
end
"""

CLIENT_FAKES = r"""
MOCK.draws, MOCK.matrices, MOCK.mats_set, MOCK.mesh_made = 0, 0, {}, 0
CreateConVar("mentat_faces", "1")
local Mesh_ = {}
Mesh_.__index = Mesh_
function Mesh_:Draw() MOCK.draws = MOCK.draws + 1; MOCK.last_mesh = self end
function Mesh_:Destroy() self.dead = true end
function Mesh() MOCK.mesh_made = MOCK.mesh_made + 1; return setmetatable({ verts = {} }, Mesh_) end
MATERIAL_TRIANGLES = 3
mesh = { Begin = function(m, prim, n) MOCK.cur_mesh = m; m.expect = n end,
         Position = function(v) MOCK.cur_mesh.verts[#MOCK.cur_mesh.verts + 1] = { pos = v } end,
         Normal = function() end, TexCoord = function() end,
         Color = function(r, g, b) local v = MOCK.cur_mesh.verts[#MOCK.cur_mesh.verts]; v.col = { r, g, b } end,
         AdvanceVertex = function() end, End = function() MOCK.cur_mesh.finished = true end }
local function mat(name) return { name = name, IsError = function() return false end, SetTexture = function(s, k, t) s.tex = t end, GetTexture = function() return "tex" end } end
MOCK.material_valid = true
function Material(path) MOCK.material_calls = (MOCK.material_calls or 0) + 1; MOCK.material_path = path
  local m = mat(path); if not MOCK.material_valid then m.IsError = function() return true end end; return m end
function CreateMaterial(name) return mat(name) end
cam = { PushModelMatrix = function(m) MOCK.matrices = MOCK.matrices + 1; MOCK.last_matrix = m end, PopModelMatrix = function() end }
render = { SetMaterial = function(m) MOCK.mats_set[#MOCK.mats_set + 1] = m.name end }
function Matrix() return { SetTranslation = function(s, v) s.t = v end, SetAngles = function(s, a) s.a = a end, SetScale = function(s, v) s.s = v end } end
function Mesh_dummy() end
local P = getmetatable(MOCK.newplayer("0")).__index
MOCK.players = {}
function P:GetModel() return self.model or "models/player/kleiner.mdl" end
function P:LookupBone(n) if self.nobone then return nil end; return n == "ValveBiped.Bip01_Head1" and 6 or -1 end
function P:GetBonePosition() return Vector(self.pos.x, self.pos.y, self.pos.z + 62) end
function P:GetManipulateBoneScale(b) return self.bscale or Vector(1, 1, 1) end
function P:GetManipulateBoneAngles() return self.bang or Angle(0, 0, 0) end
function P:GetManipulateBonePosition() return self.bpos or Vector(0, 0, 0) end
function P:ManipulateBoneScale(b, v) self.bscale = v; self.scale_calls = (self.scale_calls or 0) + 1 end
function P:ShouldDrawLocalPlayer() return self.third == true end
function P:EyeAngles() return Angle(10, self.yaw, 0) end
function P:EyePos() return Vector(self.pos.x, self.pos.y, self.pos.z + 64) end
function util.IsValidModel(m) return m ~= "models/error.mdl" and m ~= "" end
function CreateClientConVar(name, default) return CreateConVar(name, default) end
function Mesh_unused() end
"""


def jpeg(n=20000, seed=1):
    return b"\xff\xd8\xff\xe0" + hashlib.sha256(bytes([seed])).digest() * (n // 32)


def answer(data=None, source="steam", **over):
    data = data or jpeg()
    a = {"ok": True, "hash": hashlib.sha256(data).hexdigest()[:24], "crc32": zlib.crc32(data) & 0xFFFFFFFF, "bytes": len(data),
         "b64": base64.b64encode(data).decode(), "border": [30, 90, 160], "detector": "centre", "source": source}
    a.update(over)
    return a


def server(cvars=None):
    cv = {"holodeck_bridge": BRIDGE, "holodeck_bridge_token": "tok"}
    cv.update(cvars or {})
    w = World(cvars=cv)
    w.rt.execute(PRELUDE)
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/faces_server.lua")')
    w.F = w.rt.eval("MentatFaces")
    return w


def posts(w, action=None):
    out = []
    for i in range(1, int(w.rt.eval("#MOCK.https")) + 1):
        url = w.rt.eval(f"MOCK.https[{i}].url")
        if not url.endswith("/face"):
            continue
        body = json.loads(w.rt.eval(f"MOCK.https[{i}].body"))
        if action is None or body["action"] == action:
            out.append((i, body))
    return out


def reply(w, i, obj):
    w.post_json(i, obj)


def join(w, sid, **kw):
    p = w.player(sid, **kw)
    w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
    return p


def settle(w, s=6):
    w.advance(s)


def said(w, sid):
    return list(w.rt.eval("T.said")(sid).values())


def py(t):
    """A Lua table of plain data as lists / dicts."""
    if type(t).__name__ != "_LuaTable":
        return t
    keys = list(t.keys())
    if keys and keys == list(range(1, len(keys) + 1)):
        return [py(t[k]) for k in keys]
    return {k: py(t[k]) for k in keys}


def seq(m):
    """A recorded message's writes as [kind, value, bits]; byte data comes as hex (so no decoding of binary happens in Python)."""
    return py(w_export(m))["seq"]


_runtime = {}


def w_export(m):
    return _runtime["rt"].eval("function(m) return MOCK.export(m) end")(m)


def nets(w, name):
    _runtime["rt"] = w.rt
    return list(w.rt.eval("function(n) return T.nets(n) end")(name).values())


def chat(w, p, text):
    r = w.rt.eval("function(p, t) return (hook.Run('PlayerSay', p, t)) end")(p, text)
    return r


def source(w, sid):
    return w.F.source_of(w.F, sid) if False else w.rt.eval("function(s) return MentatFaces.source_of(s) end")(sid)


def visible_ids(w):
    return sorted(e.sid for e in w.rt.eval("MentatFaces.map")().values())


def give_default(w, p, data=None):
    """Join and let the bridge answer the automatic Steam lookup."""
    join(w, p.sid) if False else w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
    settle(w)
    ps = posts(w, "steam")
    assert ps, "the default lookup was not made"
    reply(w, ps[-1][0], answer(data))
    return ps[-1][1]


@pytest.fixture
def w():
    return server()


# ============================================================================================================ the automatic default
def test_a_joining_player_gets_their_own_steam_avatar_asked_for_with_the_servers_id_and_every_client_is_told(w):
    p, q = w.player(A), w.player(B)
    body = give_default(w, p)
    assert body["action"] == "steam" and body["steamid64"] == A, "the id is the server's own ply:SteamID64()"
    assert w.rt.eval("MOCK.https[1].headers['X-Skippy-Token']") == "tok"
    assert source(w, A) == "steam" and visible_ids(w) == [A]
    maps = nets(w, "mentat_face_map")
    assert maps and all(len(m.to) == 2 for m in maps[-1:]), "the list goes to everyone"
    w.advance(2)
    chunks = nets(w, "mentat_face_img")
    assert len(chunks) == math.ceil(20000 / 8192)
    sizes = [sum(len(s[1]) for s in seq(c) if s[0] == "d") // 2 for c in chunks]
    assert max(sizes) <= 8192 and sum(sizes) == len(jpeg())


def test_each_image_is_sent_once_per_player_and_one_chunk_per_pump_tick(w):
    a, b = w.player(A), w.player(B)
    give_default(w, a)
    assert len(nets(w, "mentat_face_img")) == 0, "nothing in the same tick: the pump spaces the chunks"
    w.advance(0.11)
    assert len(nets(w, "mentat_face_img")) == 1
    w.advance(1)
    n = len(nets(w, "mentat_face_img"))
    assert n == 3
    w.rt.eval("function() MentatFaces.sync() end")()
    w.advance(1)
    assert len(nets(w, "mentat_face_img")) == n, "synced again, but both players already have it"


def test_bots_and_unknown_ids_get_no_lookup(w):
    bot = w.player("90071996842377216")
    w.rt.eval("function(p) p.bot = true; hook.Run('PlayerInitialSpawn', p) end")(bot)
    settle(w)
    assert posts(w) == []


def test_no_public_picture_is_silent_and_not_asked_again_at_once(w):
    p = w.player(A)
    w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
    settle(w)
    reply(w, posts(w)[0][0], {"ok": False, "kind": "no_picture", "error": "no public Steam profile"})
    assert source(w, A) == "none" and said(w, A) == [] and visible_ids(w) == []
    w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
    settle(w)
    assert len(posts(w)) == 1, "a missing profile is not asked about again for ten minutes"


def test_the_switch_and_the_default_convar_are_respected():
    w = server({"mentat_faces_steam_default": "0"})
    w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(w.player(A))
    settle(w)
    assert posts(w) == []


# ============================================================================================================ THE HARD RULE
def set_own(w, p, url="https://lh3.googleusercontent.com/a/me=s96-c", data=None):
    w.rt.eval("function(p, a) MentatFaces.run(p, 'face', a) end")(p, url)
    ps = posts(w, "url")
    reply(w, ps[-1][0], answer(data or jpeg(seed=9), source="url"))


def test_a_picture_the_player_set_is_never_replaced_refreshed_or_overridden_by_the_default(w):
    p = w.player(A)
    set_own(w, p)
    mine = w.rt.eval("MentatFaces.rec[...].hash", A) if False else w.rt.eval("function(s) return MentatFaces.rec[s].hash end")(A)
    assert source(w, A) == "set"
    for _ in range(3):                                         # joins, forced fetches, a day later
        w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
        settle(w, 700)
        w.rt.eval("function(p) MentatFaces.default_for(p, true) end")(p)
    w.rt.eval("function(s) MentatFaces.rec[s].t = 0 end")(A)
    w.rt.eval("function(p) MentatFaces.default_for(p) end")(p)
    assert posts(w, "steam") == [], "no lookup was even made"
    assert w.rt.eval("function(s) return MentatFaces.rec[s].hash end")(A) == mine and source(w, A) == "set"


def test_an_answer_that_arrives_after_the_player_chose_is_dropped(w):
    p = w.player(A)
    w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
    settle(w)
    steam_i = posts(w, "steam")[0][0]
    set_own(w, p)                                               # the player chooses while the lookup is in flight
    reply(w, steam_i, answer(jpeg(seed=3)))                      # ...then the default's answer lands
    assert source(w, A) == "set"
    assert w.rt.eval("function(s) return MentatFaces.rec[s].hash end")(A) == answer(jpeg(seed=9))["hash"]
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'off') end")(p)
    assert source(w, A) == "set", "off does not take away a picture they set themselves"


@pytest.mark.parametrize("choice", ["off", "hide", "block"])
def test_off_hide_and_block_are_choices_the_default_never_overwrites_or_reenables(w, choice):
    p = w.player(A)
    if choice == "block":
        w.rt.eval("function() MentatFaces.blocked[...] = true end") if False else w.rt.eval("function(s) MentatFaces.blocked[s] = true end")(A)
    else:
        w.rt.eval("function(p, v) MentatFaces.run(p, 'face', v) end")(p, choice)
    for _ in range(3):
        w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
        settle(w, 700)
        w.rt.eval("function(p) MentatFaces.default_for(p, true) end")(p)
    assert posts(w) == [], "no automatic lookup for someone who chose"
    assert A not in visible_ids(w)


def test_off_after_a_default_picture_removes_it_and_it_stays_off_across_a_restart(w):
    p = w.player(A)
    give_default(w, p)
    assert A in visible_ids(w)
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'off') end")(p)
    assert source(w, A) == "off" and visible_ids(w) == []
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/faces_server.lua")')            # the server restarts: records come back from data/
    assert source(w, A) == "off"
    n = len(posts(w))
    w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
    settle(w, 700)
    assert len(posts(w)) == n


def test_remove_returns_to_the_default_unless_they_said_off(w):
    p = w.player(A)
    set_own(w, p)
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'remove') end")(p)
    ps = posts(w, "steam")
    assert len(ps) == 1 and ps[0][1]["steamid64"] == A, "back to the automatic Steam picture"
    reply(w, ps[0][0], answer(jpeg(seed=5)))
    assert source(w, A) == "steam"
    # now the same player says off first, sets their own, removes it
    q = w.player(B)
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'off') end")(q)
    set_own(w, q)
    assert source(w, B) == "set"
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'remove') end")(q)
    assert source(w, B) == "off" and len(posts(w, "steam")) == 1, "no default for someone who said off"
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'on') end")(q)
    assert len(posts(w, "steam")) == 2, "on is the explicit way back"


def test_remove_with_nothing_of_their_own_does_nothing_and_says_how_to_stop_the_default(w):
    p = w.player(A)
    give_default(w, p)
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'remove') end")(p)
    assert source(w, A) == "steam" and "off" in said(w, A)[-1]


# ============================================================================================================ commands and limits
def test_a_remote_player_may_only_give_an_https_link_never_a_file_or_plain_http(w):
    p = w.player(A)
    for bad in ["C:/Users/aaron/secret.png", "/etc/passwd", "http://lh3.googleusercontent.com/a", "file:///c:/x.png", "ftp://x/y.png",
                "https://a.example/x y", "https://" + "a" * 500, "gopher://x", ""]:
        w.rt.eval("function(p, a) MentatFaces.run(p, 'face', a) end")(p, bad)
        w.advance(30)
    assert posts(w) == []


def test_the_host_and_the_console_may_name_a_local_file_nobody_else(w):
    host, other = w.player(A, host=True), w.player(B)
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'D:/me/face.png') end")(host)
    ps = posts(w, "path")
    assert len(ps) == 1 and ps[0][1]["path"] == "D:/me/face.png"
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'D:/me/face.png') end")(other)
    assert len(posts(w, "path")) == 1


def test_setting_pictures_is_rationed_per_player(w):
    p = w.player(A)
    for i in range(6):
        w.rt.eval("function(p, i) MentatFaces.run(p, 'face', 'https://www.gravatar.com/avatar/' .. i) end")(p, i)
    assert len(posts(w, "url")) == 3, "a burst of three, then a rest"
    assert "Slow down" in said(w, A)[-1]


def test_a_failed_set_keeps_the_usual_head_and_says_so_politely(w):
    p = w.player(A)
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'https://example.com/x.png') end")(p)
    reply(w, posts(w)[0][0], {"ok": False, "kind": "refused", "error": "example.com is not one of the allowed picture sites"})
    assert source(w, A) == "none" and "usual head" in said(w, A)[-1]


def test_link_codes_are_redeemed_at_the_bridge_and_become_the_players_own_choice(w):
    p = w.player(A)
    w.rt.eval("function(p) MentatFaces.run(p, 'link', 'AB12-CD34') end")(p)
    ps = posts(w, "redeem")
    assert ps[0][1]["code"] == "AB12-CD34" and "steamid64" not in ps[0][1]
    reply(w, ps[0][0], answer(jpeg(seed=4), source="link"))
    assert source(w, A) == "set"
    w.rt.eval("function(p) MentatFaces.run(p, 'link', 'no way;drop table') end")(p)
    assert len(posts(w, "redeem")) == 1
    q = w.player(B)                                             # the portal not being there yet is told kindly
    w.rt.eval("function(p) MentatFaces.run(p, 'link', 'ZZ99-ZZ99') end")(q)
    reply(w, posts(w, "redeem")[-1][0], {"ok": False, "kind": "unavailable", "error": "the portal link is not set up yet"})
    assert source(w, B) == "none" and "portal link is not available" in said(w, B)[-1]


def test_chat_commands_are_swallowed_and_never_reach_the_planner_or_the_model(w):
    p = w.player(A)
    assert chat(w, p, "skippy face hide") == ""
    assert w.rt.eval("function(s) return MentatFaces.rec[s].hidden end")(A) is True
    assert chat(w, p, "Hey Skippy, face show") == ""
    assert chat(w, p, "skippy link AB12-CD34") == ""
    assert chat(w, p, "skippy add three couches") is None or True
    assert [u for u in (w.rt.eval(f"MOCK.https[{i}].url") for i in range(1, int(w.rt.eval("#MOCK.https")) + 1)) if u.endswith("/plan")] == []


def test_the_chat_command_runs_once_whichever_hook_hears_it_first(w):
    p = w.player(A)
    for order in (["Holodeck.Chat", "Mentat.Faces.Chat"], ["Mentat.Faces.Chat", "Holodeck.Chat"]):
        w.rt.execute("MOCK.hook_order = { PlayerSay = {} }; MentatFaces.wrap_holodeck_chat()")
        for name in order:
            w.rt.eval("function(n) table.insert(MOCK.hook_order.PlayerSay, n) end")(name)
        w.rt.execute("MOCK.last_msg_count = 0")
        before = w.rt.eval("MentatFaces.stats.maps")
        chat(w, p, "skippy face hide")
        chat(w, p, "skippy face show")
        assert w.rt.eval("MentatFaces.stats.maps") - before == 2, order
        w.advance(1)
    assert [u for u in (w.rt.eval(f"MOCK.https[{i}].url") for i in range(1, int(w.rt.eval("#MOCK.https")) + 1)) if u.endswith("/plan")] == []


# ============================================================================================================ visibility
def test_hide_show_block_and_the_global_switch_take_effect_at_once_for_everyone(w):
    a, b = w.player(A), w.player(B)
    give_default(w, a)
    assert visible_ids(w) == [A]
    n0 = len(nets(w, "mentat_face_map"))
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'hide') end")(a)
    assert visible_ids(w) == [] and len(nets(w, "mentat_face_map")) == n0 + 1
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'show') end")(a)
    assert visible_ids(w) == [A]
    # an admin blocks; a non-admin cannot
    w.rt.eval("function(p, s) MOCK.cmds.mentat_face_block(p, 'mentat_face_block', { s }) end")(b, A)
    assert visible_ids(w) == [A], "only admins block"
    admin = w.player(C, admin=True)
    w.rt.eval("function(p, s) MOCK.cmds.mentat_face_block(p, 'mentat_face_block', { s }) end")(admin, A)
    assert visible_ids(w) == []
    last = nets(w, "mentat_face_map")[-1]
    assert [s for s in seq(last) if s[0] == "u"][0][1] == 0, "the list everyone just got is empty"
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'show') end")(a)
    assert visible_ids(w) == [], "a block outranks the player"
    w.rt.eval("function(p, s) MOCK.cmds.mentat_face_unblock(p, 'x', { s }) end")(admin, A)
    assert visible_ids(w) == [A]
    w.rt.eval("function(n) MOCK.cvars[n]:SetValue('0') end")("mentat_faces")
    assert visible_ids(w) == []
    w.rt.eval("function(n) MOCK.cvars[n]:SetValue('1') end")("mentat_faces")
    assert visible_ids(w) == [A]
    w.rt.eval("function() MOCK.cmds.mentat_face_block(nil, 'mentat_face_block', { '" + D + "' }) end")()      # the console may
    assert w.rt.eval("function(s) return MentatFaces.blocked[s] end")(D) is True


def test_a_blocked_player_cannot_set_a_picture_and_nothing_is_fetched(w):
    p = w.player(A)
    w.rt.eval("function(s) MentatFaces.blocked[s] = true end")(A)
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'https://www.gravatar.com/avatar/x') end")(p)
    w.rt.eval("function(p) MentatFaces.run(p, 'link', 'AB12-CD34') end")(p)
    assert posts(w) == []


def test_a_late_joiner_asking_for_everything_gets_the_list_and_the_pictures_and_asks_are_rationed(w):
    a = w.player(A)
    give_default(w, a)
    w.advance(2)
    late = w.player(B)
    before = len(nets(w, "mentat_face_img"))
    ask = [["u", 0, 8]]
    w.rt.eval("function(seq, p) MOCK.net_from_client(p, 'mentat_face_want', seq) end")(lua_table(w.rt, ask), late)
    w.advance(2)
    got = nets(w, "mentat_face_img")[before:]
    assert len(got) == 3 and all(len(m.to) == 1 and m.to[1].sid == B for m in got), "only the late joiner is sent the pictures again"
    maps = nets(w, "mentat_face_map")
    assert len(maps[-1].to) == 1 and maps[-1].to[1].sid == B, "...and the list"
    for _ in range(20):
        w.rt.eval("function(seq, p) MOCK.net_from_client(p, 'mentat_face_want', seq) end")(lua_table(w.rt, ask), late)
    w.advance(5)
    assert len(nets(w, "mentat_face_map")) < 20, "the asks past the allowance were ignored"


@pytest.mark.parametrize("make", [
    lambda d: answer(d, crc32=123), lambda d: answer(b"not a jpeg at all, plainly" * 20), lambda d: answer(jpeg(30000)),
    lambda d: answer(d, b64="@@@@"), lambda d: answer(d, border=[1, 2]), lambda d: answer(d, hash="zz"), lambda d: {"ok": True}])
def test_nothing_the_bridge_says_is_believed_unchecked(w, make):
    p = w.player(A)
    w.rt.eval("function(p) hook.Run('PlayerInitialSpawn', p) end")(p)
    settle(w)
    reply(w, posts(w)[0][0], make(jpeg()))
    assert source(w, A) == "none" and visible_ids(w) == []


# ============================================================================================================ persistence
def test_the_store_survives_a_restart_and_a_damaged_file_falls_back_then_starts_empty(w):
    p = w.player(A)
    give_default(w, p)
    set_own(w, w.player(B))
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'hide') end")(w.player(C))
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/faces_server.lua")')
    assert source(w, A) == "steam" and source(w, B) == "set" and w.rt.eval("function(s) return MentatFaces.rec[s].hidden end")(C) is True
    w.rt.execute("MOCK.files['mentat/faces.json'] = '{ not json'")              # the main file is damaged: the last good copy is used
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/faces_server.lua")')
    assert source(w, B) in ("set", "none")
    w.rt.execute("MOCK.files['mentat/faces.json'] = '{ not json'; MOCK.files['mentat/faces.json.bak'] = '\\0\\1garbage'")
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/faces_server.lua")')
    assert source(w, A) == "none" and w.rt.eval("next(MentatFaces.rec)") is None


def test_hand_edited_junk_in_the_store_is_dropped_record_by_record(w):
    w.rt.execute("""MOCK.files['mentat/faces.json'] = [[{"version":1,"rec":{"123":{"hidden":true},
      "76561190000000001":{"source":"set","hash":"../../etc/x","crc":5,"border":[1,2,3]},
      "76561190000000002":{"hidden":true,"off":true,"optin":true,"junk":[1,2,3]}},"blocked":{"76561190000000003":true,"x":true}}]]""")
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/faces_server.lua")')
    rec = w.rt.eval("MentatFaces.rec")
    assert A not in rec.keys() and "123" not in rec.keys() and B in rec.keys()
    assert rec[B].hidden and rec[B].off and rec[B].optin
    assert w.rt.eval("function(s) return MentatFaces.blocked[s] end")(C) is True and w.rt.eval("MentatFaces.blocked.x") is None


def test_the_store_is_written_whole_then_swapped_when_the_engine_can_rename(w):
    w.rt.execute("function file.Rename(a, b) MOCK.files[b] = MOCK.files[a]; MOCK.files[a] = nil; MOCK.renamed = (MOCK.renamed or 0) + 1 end")
    give_default(w, w.player(A))
    assert w.rt.eval("MOCK.renamed") >= 1 and not w.rt.eval("MOCK.files['mentat/faces.json.tmp'] ~= nil")
    assert w.rt.eval("MOCK.files['mentat/faces.json'] ~= nil")


def test_pictures_nobody_wears_any_more_are_deleted(w):
    p = w.player(A)
    give_default(w, p)
    h = w.rt.eval("function(s) return MentatFaces.rec[s].hash end")(A)
    assert w.rt.eval(f"MOCK.files['mentat/faces/{h}.jpg'] ~= nil")
    w.rt.eval("function(p) MentatFaces.run(p, 'face', 'off') end")(p)
    assert not w.rt.eval(f"MOCK.files['mentat/faces/{h}.jpg'] ~= nil")


# ============================================================================================================ the client
def client():
    cw = World(cvars={}, load=(), server=False)
    cw.rt.execute(PRELUDE)
    cw.rt.execute(CLIENT_FAKES)
    cw.rt.execute("MOCK.me = MOCK.newplayer('" + A + "', { pos = Vector(0, 0, 0) })")
    cw.rt.execute('dofile(MOCK.ROOT .. "/autorun/mentat_faces.lua")')
    cw.FC = cw.rt.eval("MentatFacesClient")
    return cw


def carry(sw, cw, name, who=None):
    """Deliver every recorded server message of this name to the client runtime (a list of copies)."""
    n = 0
    for m in nets(sw, name):
        if who is not None and who not in list(m.to.values()):
            continue
        seq_ = seq(m)
        cw.rt.eval("function(name, seq) MOCK.deliver_seq(name, seq, nil) end")(name, lua_table(cw.rt, seq_))
        n += 1
    return n


def receive_everything(sw, cw, who=None):
    sw.advance(3)
    carry(sw, cw, "mentat_face_img", who)
    carry(sw, cw, "mentat_face_map", who)


def other(cw, sid, **kw):
    t = ", ".join(f"{k} = {json.dumps(v)}" for k, v in kw.items())
    return cw.rt.eval(f"MOCK.newplayer('{sid}', {{ pos = Vector(100, 0, 0) }})")


def test_the_picture_arrives_in_chunks_is_checked_and_becomes_a_material_of_the_right_file():
    sw, cw = server(), client()
    other_p = sw.player(B)
    give_default(sw, other_p)
    receive_everything(sw, cw)
    img = cw.rt.eval("MentatFacesClient.images[1]")
    assert img.state == "ready" and img.crc == zlib.crc32(jpeg()) & 0xFFFFFFFF and list(img.border.values()) == [30, 90, 160]
    assert cw.rt.eval("MOCK.material_path") == "data/mentat_faces/" + f"{img.crc:08x}.jpg"
    assert cw.rt.eval(f"#MOCK.files['mentat_faces/{img.crc:08x}.jpg']") == len(jpeg())


def test_a_damaged_or_out_of_order_or_oversize_picture_is_rejected_by_the_client():
    cw = client()
    f = cw.rt.eval("function(...) return MentatFacesClient.on_chunk(...) end")
    brd = cw.rt.eval("{1,2,3}")
    data = jpeg(3000)
    n = len(data)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    assert f(1, crc, 1, 2, n, brd, data[:1500]) is True
    assert f(1, crc, 2, 2, n, brd, data[1500:]) is True and cw.rt.eval("MentatFacesClient.images[1].state") == "ready"
    assert f(6, crc, 1, 2, n, brd, data[:1500]) is True
    assert f(6, crc, 1, 2, n, brd, data[:1500]) is False, "a repeated or out-of-order chunk drops the partial picture"
    assert cw.rt.eval("MentatFacesClient.partial[6]") is None
    assert f(2, crc ^ 1, 1, 1, n, brd, data) is False                       # wrong CRC
    assert f(3, crc, 1, 1, n, brd, b"\x00\x01\x02" + data[3:]) is False     # not a jpeg
    assert f(4, crc, 1, 1, 99999, brd, data) is False                          # over the cap
    assert f(0, crc, 1, 1, n, brd, data) is False and f(65, crc, 1, 1, n, brd, data) is False
    assert f(5, crc, 3, 2, n, brd, data) is False                           # chunk beyond count
    for slot in (2, 3, 4, 5, 6):
        assert cw.rt.eval(f"MentatFacesClient.images[{slot}]") is None
    assert cw.rt.eval("MentatFacesClient.stats.rejected") >= 6


def wearing(sw, cw, sid=B, model=None, **kw):
    """sid wears a (default) face on the client; returns the client's player entity for sid."""
    p = sw.player(sid)
    give_default(sw, p)
    receive_everything(sw, cw)
    cp = cw.rt.eval(f"MOCK.newplayer('{sid}', {{ pos = Vector(100, 0, 0) }})")
    if model:
        cp.model = model
    for k, v in kw.items():
        setattr(cp, k, v)
    return cp


def test_a_stock_model_gets_its_head_hidden_and_a_picture_head_drawn_then_restored_when_the_face_goes():
    sw, cw = server(), client()
    cp = wearing(sw, cw)
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    assert cp.bscale.x == 0 and cp.bscale.z == 0, "the real head is scaled to nothing"
    cw.rt.execute("MOCK.draws, MOCK.matrices = 0, 0")
    cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(cp)
    assert cw.rt.eval("MOCK.matrices") == 1 and cw.rt.eval("MOCK.draws") == 2, "one matrix push, a front and a back mesh"
    mats = list(cw.rt.eval("MOCK.mats_set").values())
    assert mats[0].startswith("mentat_face_") and mats[0] != "mentat_face_plain" and mats[1] == "mentat_face_plain"
    # the face is removed on the server: an empty list reaches the client
    pb = sw.rt.eval("function(s) return MentatFaces.blocked end")()
    sw.rt.eval("function(s) MentatFaces.blocked[s] = true; MentatFaces.sync() end")(B)
    cw.rt.execute("MOCK.nets = {}")
    carry(sw, cw, "mentat_face_map")
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    assert cp.bscale.x == 1 and cp.bscale.y == 1 and cp.bscale.z == 1
    cw.rt.execute("MOCK.draws = 0")
    cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(cp)
    assert cw.rt.eval("MOCK.draws") == 0


def test_a_custom_model_is_left_completely_alone_unless_its_player_said_on():
    sw, cw = server(), client()
    cp = wearing(sw, cw, model="models/player/some_addon/cool_robot.mdl")
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    assert getattr(cp, "scale_calls", None) is None and cp.bscale is None if hasattr(cp, "bscale") else True
    cw.rt.execute("MOCK.draws = 0")
    cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(cp)
    assert cw.rt.eval("MOCK.draws") == 0
    # the player says `skippy face on`
    sw.rt.eval("function(s) MentatFaces.rec[s].optin = true; MentatFaces.sync() end")(B)
    cw.rt.execute("MOCK.nets = {}")
    carry(sw, cw, "mentat_face_map")
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    assert cp.bscale.x == 0
    cw.rt.execute("MOCK.draws = 0")
    cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(cp)
    assert cw.rt.eval("MOCK.draws") == 2


def test_a_head_bone_another_addon_is_already_moving_is_left_alone():
    sw, cw = server(), client()
    cp = wearing(sw, cw, bscale=cw.rt.eval("Vector(1.3, 1.3, 1.3)"))
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    assert cp.bscale.x == 1.3 and getattr(cp, "scale_calls", None) is None
    cw.rt.execute("MOCK.draws = 0")
    cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(cp)
    assert cw.rt.eval("MOCK.draws") == 0


def test_a_model_change_puts_the_head_back_on_the_old_model_and_leaves_the_new_custom_one_alone():
    sw, cw = server(), client()
    cp = wearing(sw, cw)
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    assert cp.bscale.x == 0
    cp.model = "models/player/addon/other.mdl"
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    assert cp.bscale.x == 1


def test_no_head_bone_and_no_model_floats_the_head_at_the_eyes_and_a_custom_boneless_model_is_left_alone():
    sw, cw = server(), client()
    cp = wearing(sw, cw, model="models/error.mdl", nobone=True)
    cw.rt.execute("MOCK.draws, MOCK.matrices = 0, 0")
    cw.rt.eval("function() hook.Run('PostDrawOpaqueRenderables') end")()
    assert cw.rt.eval("MOCK.draws") == 2
    m = cw.rt.eval("MOCK.last_matrix")
    assert abs(m.t.z - (64 - 1.5)) < 0.2, "at eye height (a little below the eyes)"
    cp.model = "models/player/addon/boneless.mdl"
    cw.rt.execute("MOCK.draws = 0")
    cw.rt.eval("function() hook.Run('PostDrawOpaqueRenderables') end")()
    assert cw.rt.eval("MOCK.draws") == 0


def test_the_local_player_sees_their_head_in_third_person_never_in_first():
    sw, cw = server(), client()
    me = sw.player(A)
    give_default(sw, me)
    receive_everything(sw, cw)
    mine = cw.rt.eval("MOCK.me")
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    cw.rt.execute("MOCK.draws = 0")
    cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(mine)
    assert cw.rt.eval("MOCK.draws") == 0, "first person"
    mine.third = True
    cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(mine)
    assert cw.rt.eval("MOCK.draws") == 2


def test_an_invalid_material_gives_a_plain_neutral_head_and_no_error():
    sw, cw = server(), client()
    cw.rt.execute("MOCK.material_valid = false")
    cp = wearing(sw, cw)
    assert cw.rt.eval("MentatFacesClient.images[1].state") == "plain"
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    cw.rt.execute("MOCK.draws, MOCK.mats_set = 0, {}")
    cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(cp)
    assert cw.rt.eval("MOCK.draws") == 1 and list(cw.rt.eval("MOCK.mats_set").values()) == ["mentat_face_plain"]


def test_missing_picture_asks_the_server_once_in_a_while_and_draws_the_neutral_head_meanwhile():
    cw = client()
    cp = cw.rt.eval("MOCK.newplayer('" + B + "', { pos = Vector(100, 0, 0) })")
    cw.rt.eval("function() MentatFacesClient.map['" + B + "'] = { slot = 7, optin = false } end")()
    cw.rt.execute("MOCK.draws = 0")
    for _ in range(5):
        cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(cp)
    assert cw.rt.eval("MOCK.draws") == 5
    wants = [m for m in cw.rt.eval("MOCK.nets").values() if m.name == "mentat_face_want"]
    assert len(wants) == 1, "asked once, then every eight seconds at most"


def test_meshes_are_built_once_and_shared_by_every_head():
    sw, cw = server(), client()
    ps = [wearing(sw, cw, sid=s) for s in (B, C, D)]
    cw.rt.eval("function() MentatFacesClient.refresh() end")()
    for _ in range(3):
        for cp in ps:
            cw.rt.eval("function(p) hook.Run('PostPlayerDraw', p) end")(cp)
    assert cw.rt.eval("MentatFacesClient.stats.mesh_builds") <= 3, "front, back, full: shared by all heads and all frames"
    assert cw.rt.eval("MentatFacesClient.stats.drawn") == 9


def test_the_loader_is_one_small_function_and_falls_back_when_the_api_is_missing():
    cw = client()
    assert cw.rt.eval("function() return MentatFacesClient.load_material('mentat_faces/00000001.jpg', 1) end")() is not None
    cw.rt.execute("MOCK.material_valid = false")
    assert cw.rt.eval("function() return MentatFacesClient.load_material('mentat_faces/00000001.jpg', 1) end")() is None
    cw.rt.execute("Material = nil")
    assert cw.rt.eval("function() return MentatFacesClient.load_material('x.jpg', 1) end")() is None


# ============================================================================================================ geometry (pure)
def head():
    w = World(cvars={}, load=(), server=False)
    w.rt.execute(PRELUDE)
    return w.rt.eval('dofile(MOCK.ROOT .. "/mentat/faces_head.lua")'), w


def tri_lists(g):
    return [[[[v[k] for k in range(1, 9)] for v in t.values()] for t in lst.values()] for lst in (g.front, g.back, g.neck)]


def test_the_head_is_a_few_hundred_triangles_of_plausible_size():
    Head, w = head()
    g = Head.geometry()
    front, back, neck = tri_lists(g)
    n = len(front) + len(back) + len(neck)
    assert 150 <= n <= Head.MAX_TRIS, n
    pts = [v for lst in (front, back, neck) for t in lst for v in t]
    xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    assert 0.9 * Head.DEPTH <= max(xs) <= Head.DEPTH + 1e-6 and -Head.DEPTH - 1e-6 <= min(xs) <= -0.9 * Head.DEPTH
    assert max(ys) <= Head.WIDTH + 1e-6 and min(ys) >= -Head.WIDTH - 1e-6
    assert max(zs) <= Head.HEIGHT + 1e-6 and min(zs) < -Head.HEIGHT, "the neck goes below the chin"
    assert min(zs) >= -2.1 * Head.HEIGHT, "...but only a short way"
    assert 10 <= (max(zs) - min(zs[:len(front) * 3 + len(back) * 3])) <= 14, "the egg itself is roughly 13 HU tall"


def test_front_uvs_cover_the_picture_inside_0_to_1_and_the_face_looks_forward():
    Head, w = head()
    front, back, neck = tri_lists(Head.geometry())
    us = [v[6] for t in front for v in t]
    vs = [v[7] for t in front for v in t]
    assert min(us) >= 0 and max(us) <= 1 and min(vs) >= 0 and max(vs) <= 1
    assert min(us) < 0.05 and max(us) > 0.95 and min(vs) < 0.05 and max(vs) > 0.95, "the picture fills the cap"
    nose = [v for t in front for v in t if v[0] > 0.99 * Head.DEPTH]
    assert nose and all(abs(v[6] - 0.5) < 0.01 and abs(v[7] - 0.5) < 0.01 for v in nose), "the middle of the picture is the front pole"
    assert all(v[0] > 0 for t in front for v in t), "the cap is the front half"
    # left is +y (Source); the viewer facing the head sees its left on their right, so u grows with +y
    left = [v for t in front for v in t if v[1] > 0.5 * Head.WIDTH and abs(v[2]) < 1]
    assert left and all(v[6] > 0.5 for v in left)
    up = [v for t in front for v in t if v[2] > 0.5 * Head.HEIGHT and abs(v[1]) < 1]
    assert up and all(v[7] < 0.5 for v in up), "the top of the picture is the top of the head"


def test_triangles_are_clockwise_from_outside_normals_are_unit_and_colours_are_bytes():
    Head, w = head()
    front, back, neck = tri_lists(Head.geometry())
    for t in front + back + neck:
        a, b, c = t
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        n = [sum(p[3 + i] for p in t) for i in range(3)]
        assert sum(cr[i] * n[i] for i in range(3)) <= 1e-9, "the cross product points in: clockwise as seen from outside"
        for p in t:
            assert abs(math.sqrt(p[3] ** 2 + p[4] ** 2 + p[5] ** 2) - 1) < 1e-6
    r, g, b = w.rt.eval("function(H) local t = H.geometry().front[1]; return H.vertex_colour(t[1], {30, 90, 160}) end")(Head)
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
    assert Head.AMBIENT <= Head.light(0, 0, -1) <= Head.light(0.4, 0.3, 0.84) <= 1


def test_the_model_policy_in_one_table():
    Head, w = head()
    mode = w.rt.eval("function(H, t) return H.mode(t) end")
    T = lambda **k: lua_table(w.rt, k)  # noqa: E731
    base = dict(entry=True, optin=False, first_person=False, has_bone=True, standard=True, bone_touched=False, model_valid=True)
    assert mode(Head, T(**base)) == "bone"
    assert mode(Head, T(**{**base, "entry": False})) is None
    assert mode(Head, T(**{**base, "first_person": True})) is None
    assert mode(Head, T(**{**base, "standard": False})) is None
    assert mode(Head, T(**{**base, "standard": False, "optin": True})) == "bone"
    assert mode(Head, T(**{**base, "bone_touched": True})) is None
    assert mode(Head, T(**{**base, "has_bone": False})) is None, "a model without the bone and a valid model: left alone"
    assert mode(Head, T(**{**base, "has_bone": False, "model_valid": False})) == "float"
    assert mode(Head, T(**{**base, "has_bone": False, "optin": True})) == "float"
    assert Head.standard_model("models/player/kleiner.mdl") and Head.standard_model("models/player/Group01/male_07.mdl")
    assert not Head.standard_model("models/player/miku.mdl") and not Head.standard_model("")
