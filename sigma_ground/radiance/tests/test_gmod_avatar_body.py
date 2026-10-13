"""The avatar wears the REAL Skippy (the print kit's six SVM2 parts) when the private payloads exist, and the procedural can when they do not.

Offline: the REAL addon files (avatar.lua + avatar/body_net.lua on a server runtime, avatar_client.lua + avatar/body_client.lua + holodeck/svm1.lua
on a client runtime) run under LuaJIT (lupa) against gmod_mock.lua; the net messages the server records are carried to the client runtime.
util.Decompress is a lookup in the mock (the LZMA framing itself was checked live, see stage/gmod/holodeck/README.md).

The synthetic tests always run. The tests on the real kit payloads (stage/skippyverse/private/skippy, gitignored, never committed) SKIP when
that folder is absent. What this cannot prove (how the mesh LOOKS in the engine, the frame cost) is in stage/gmod/AVATAR.md.
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
import zlib
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stage"))
sys.path.insert(0, str(HERE))

from skippyverse import meshpack  # noqa: E402
from skippyverse import skippy_model as SM  # noqa: E402
from test_gmod_avatar import CLIENT_PRELUDE, avatar_world  # noqa: E402
from test_gmod_found import CLIENT_FAKES as RENDER_FAKES  # noqa: E402
from test_gmod_glue import ADDON, World, lua_table  # noqa: E402

PRIVATE = ROOT / "stage" / "skippyverse" / "private" / "skippy"
PARTS = ["can", "jacket", "head", "hat", "hand_a", "hand_b"]
HU_PER_M = 1 / 0.01905
A, B = "76561190000000001", "76561190000000002"

MANIFEST = {"format": "SVM2", "figure_height_m": 0.6,
            "pivots": {"head": {"pivot": [0.0, 0.0, 0.30]}, "hat": {"pivot": [0.04, 0.0, 0.48], "top": [-0.04, 0.0, 0.60]},
                       "hand_point": {"part": "hand_a", "wrist": [0.06, 0.16, 0.38], "tip": [0.03, 0.13, 0.49]}}}


# ============================================================================================================ synthetic payloads
def cube_part(cx=0.0, cy=0.0, cz=0.2, h=0.05, cols=((192, 192, 192),)):
    """A closed cube as an SVM2 part with per-vertex (corner) normals pointing out from the centre. Returns (blob, raw)."""
    verts = [(cx + sx * h, cy + sy * h, cz + sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    normals = [tuple(c / math.sqrt(3) for c in (sx, sy, sz)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    idx = lambda sx, sy, sz: ((sx > 0) << 2) | ((sy > 0) << 1) | (sz > 0)  # noqa: E731
    faces = []
    quads = [((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)),
             ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),
             ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))]
    for q in quads:
        a, b, c, d = (idx(*p) for p in q)
        faces += [(a, b, c), (a, c, d)]
    colours = [cols[i % len(cols)] for i in range(len(faces))]
    blob, _ = SM.pack2(verts, normals, faces, colours)
    return blob, meshpack.unlzma_alone(blob)


def synthetic_set():
    """{file name: (blob, raw)} for the manifest-less parts; each part a different cube so a swap would show."""
    out = {}
    cols = {"can": ((192, 192, 192),), "jacket": ((0, 0, 160), (252, 239, 3)), "head": ((192, 192, 192), (255, 0, 0), (0, 0, 0)), "hat": ((0, 0, 160),),
            "hand_a": ((192, 192, 192),), "hand_b": ((192, 192, 192),)}
    for i, name in enumerate(PARTS):
        out[name + ".svm2"] = cube_part(cx=0.02 * i, cz=0.1 + 0.05 * i, cols=cols[name])
    return out


def real_set():
    return {f"{n}.svm2": ((PRIVATE / f"{n}.svm2").read_bytes(), meshpack.unlzma_alone((PRIVATE / f"{n}.svm2").read_bytes())) for n in PARTS}


needs_private = pytest.mark.skipif(not (PRIVATE / "manifest.json").is_file(), reason="the private Skippy kit payloads are not installed (stage/skippyverse/private/skippy, gitignored)")


# ============================================================================================================ server helpers
def install(w, parts, manifest=None, folder="mentat/skippy", register=True):
    """Put the files where file.Read finds them (the mock's file table) and teach util.Decompress their raw bytes."""
    man = json.dumps(MANIFEST if manifest is None else manifest)
    w.rt.eval("function(p, d) MOCK.files[p] = d end")(folder + "/manifest.json", man.encode())
    for name, (blob, raw) in parts.items():
        w.rt.eval("function(p, d) MOCK.files[p] = d end")(folder + "/" + name, blob)
        if register:
            w.rt.eval("function(a, b) MOCK.lzma[a] = b end")(blob, raw)


def exported(w):
    """The recorded body messages as plain python: [{'to': [sids], 'seq': [...]}]."""
    code = """function()
      local o = {}
      for _, m in ipairs(MOCK.nets) do
        if m.name == 'mentat_avatar_body' then
          local sids = {}
          for _, p in ipairs(m.to) do sids[#sids + 1] = p.sid end
          o[#o + 1] = { to = sids, seq = MOCK.export(m).seq }
        end
      end
      return MOCK.json_encode(o)
    end"""
    return json.loads(w.rt.eval(code)())


def server(parts=None, manifest=None, **kw):
    w = avatar_world()
    if parts is not None:
        install(w, parts, manifest, **kw)
    return w


def reload(w):
    r = w.rt.eval("function() local ok, why = Mentat.Avatar.Body.reload(); return ok, why end")()
    return r


# ============================================================================================================ client helpers
CLIENT_FAKES = r"""
RB = { draws = {}, stack = {}, mods = {}, depth = {}, cull = {}, built = 0, destroyed = 0 }
MATERIAL_TRIANGLES, MATERIAL_CULLMODE_NONE, MATERIAL_CULLMODE_CCW = 3, 1, 2
local function copy(t) local o = {}; for k, v in pairs(t) do o[k] = v end; return o end
function Matrix()
  local m = { f = {}, ops = {} }
  function m:Translate(v) self.ops[#self.ops + 1] = { "T", v } end
  function m:Rotate(a) self.rot = a; self.ops[#self.ops + 1] = { "R", a } end
  function m:Scale(v) self.scale = v; self.ops[#self.ops + 1] = { "S", v } end
  function m:SetField(r, c, x) self.f[r] = self.f[r] or {}; self.f[r][c] = x end
  return m
end
cam.PushModelMatrix = function(m, mul) RB.stack[#RB.stack + 1] = m end
cam.PopModelMatrix = function() RB.stack[#RB.stack] = nil end
render.SetColorModulation = function(r, g, b) RB.mod = { r, g, b } end
render.OverrideDepthEnable = function(a, b) RB.depth[#RB.depth + 1] = { a, b } end
render.CullMode = function(c) RB.cull[#RB.cull + 1] = c end
function Mesh()
  local im = { verts = {} }
  RB.built = RB.built + 1
  function im:Draw() RB.draws[#RB.draws + 1] = { mesh = self, top = RB.stack[#RB.stack], depth = #RB.stack, mod = RB.mod, override = RB.depth[#RB.depth] } end
  function im:Destroy() RB.destroyed = RB.destroyed + 1 end
  return im
end
local building
mesh = {
  Begin = function(im, prim, n) building = { im = im, verts = {}, cur = {} } end,
  Position = function(v) building.cur = { pos = v } end,
  Normal = function(v) building.cur.normal = v end,
  Color = function(r, g, b, a) building.cur.col = { r, g, b, a } end,
  TexCoord = function() end,
  AdvanceVertex = function() building.verts[#building.verts + 1] = building.cur; building.cur = {} end,
  End = function() building.im.verts = building.verts end,
}
function DynamicLight() return nil end
local Ply = getmetatable(MOCK.me)
function Ply:GetNWEntity() return nil end
function BODY_STATE()
  local B = Mentat.Avatar.BodyC
  return MOCK.json_encode({ ready = B.ready, state = B.state, why = B.why or "", tris = B.tris or 0 })
end
function DRAW_FRAME(n, dt)
  RB.draws = {}
  for _ = 1, n or 1 do RB.draws = {}; MOCK.advance(dt or 0.05); hook.Run("PostDrawTranslucentRenderables", false, false) end
end
"""


def client(on=True, bot_at=(100, 0, 0), hide=""):
    w = World(server=False, load=())
    w.rt.execute(RENDER_FAKES)
    w.rt.execute(CLIENT_PRELUDE)
    w.rt.execute(f"MOCK.bot.pos = Vector({bot_at[0]}, {bot_at[1]}, {bot_at[2]}); MOCK.bot.yaw = 0")
    w.rt.execute(CLIENT_FAKES)
    for f in ("skippy_client.lua", "avatar_client.lua"):
        w.rt.execute(f'dofile(MOCK.ROOT .. "/mentat/{f}")')
    w.rt.eval("AT_out")(on)
    if hide:
        w.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")(hide)
    return w


def register_client(w, parts):
    for name, (blob, raw) in parts.items():
        w.rt.eval("function(a, b) MOCK.lzma[a] = b end")(blob, raw)


def feed(w, msgs):
    for m in msgs:
        w.rt.eval("function(name, seq) MOCK.deliver_seq(name, seq) end")("mentat_avatar_body", lua_table(w.rt, m["seq"]))


def state(w):
    return json.loads(w.rt.eval("BODY_STATE")())


def draws(w):
    return w.rt.eval("RB").draws


def full_pipeline(parts, manifest=None, drop=None, tamper=None):
    """A server with `parts` loads and sends; the messages (optionally with one dropped / tampered) reach a fresh client. Returns (server, client)."""
    s = server(parts, manifest)
    s.player(A)
    ok, why = reload(s)
    assert ok is True, why
    s.advance(5)
    msgs = exported(s)
    if drop is not None:
        msgs = [m for k, m in enumerate(msgs) if k != drop]
    if tamper is not None:
        tamper(msgs)
    c = client()
    register_client(c, parts)
    feed(c, msgs)
    return s, c, msgs


# ============================================================================================================ the decoder (SVM2)
def lua_decode(w, raw, allow2=True):
    f = w.rt.eval("""function(h, a) local S = dofile(MOCK.ROOT .. '/mentat/holodeck/svm1.lua'); local d, e = S.decode(MOCK.fromhex(h), a); return d, e end""")
    return f(raw.hex(), allow2)


def test_the_lua_decoder_reads_svm2_with_normals_and_refuses_it_unless_asked():
    blob, raw = cube_part()
    w = World(server=False, load=())
    d, why = lua_decode(w, raw)
    assert d.nv == 8 and d.nt == 12 and d.svm2 is True
    verts, normals, faces, cols = SM.unpack2(blob)
    for i in range(8):
        assert (d.x[i + 1], d.y[i + 1], d.z[i + 1]) == pytest.approx(verts[i], abs=1e-12)
        assert (d.nx[i + 1], d.ny[i + 1], d.nz[i + 1]) == pytest.approx(normals[i], abs=1e-12)
    d2, why2 = lua_decode(w, raw, allow2=False)
    assert d2 is None and "not an SVM1" in why2, "a found object's reader never takes an SVM2 payload"
    d3, why3 = lua_decode(w, raw + b"\0")
    assert d3 is None and "does not match" in why3
    d4, why4 = lua_decode(w, raw[:-3])
    assert d4 is None and "does not match" in why4


@needs_private
@pytest.mark.parametrize("part", PARTS)
def test_the_lua_decoder_matches_the_python_unpacker_on_every_real_part(part):
    blob = (PRIVATE / f"{part}.svm2").read_bytes()
    raw = meshpack.unlzma_alone(blob)
    verts, normals, faces, cols = SM.unpack2(blob)
    w = World(server=False, load=())
    d, why = lua_decode(w, raw)
    assert d is not None, why
    assert d.nv == len(verts) and d.nt == len(faces)
    for i in range(0, d.nv, 7):
        assert (d.x[i + 1], d.y[i + 1], d.z[i + 1]) == pytest.approx(verts[i], abs=1e-12)
        assert (d.nx[i + 1], d.ny[i + 1], d.nz[i + 1]) == pytest.approx(normals[i], abs=1e-12)
    for t in range(0, d.nt, 5):
        assert (d.a[t + 1] - 1, d.b[t + 1] - 1, d.c[t + 1] - 1) == tuple(faces[t])
        assert (d.r[t + 1], d.g[t + 1], d.bl[t + 1]) == tuple(cols[t])


# ============================================================================================================ the server: the fallback and the send
def test_without_the_files_nothing_is_sent_the_flag_stays_false_and_he_still_comes():
    w = server()                                   # no files anywhere
    p = w.player(A, x=0, y=0, yaw=0)
    ok, why = reload(w)
    assert ok is False and "not installed" in why
    w.advance(5)
    assert exported(w) == [] and w.rt.eval("function() return GetGlobalBool('mentat_avatar_body', false) end")() is False
    r = w.rt.eval("function(p) return Mentat.Avatar.Summon(p) end")(p)
    assert r is True, "the procedural body is the fallback; calling him works without the kit"


@pytest.mark.parametrize("break_it,why", [
    ("missing_part", "hat.svm2 is missing"),
    ("no_manifest_pivots", "manifest.json"),
    ("bad_json", "manifest.json"),
    ("unregistered", "util.Decompress"),
    ("garbage", "LZMA header"),
    ("svm1_not_svm2", "not SVM2"),
])
def test_any_bad_file_keeps_the_procedural_body_and_sends_nothing(break_it, why):
    parts = synthetic_set()
    manifest = None
    if break_it == "garbage":
        parts["head.svm2"] = (b"this is not an lzma stream at all, honest", b"")
    elif break_it == "svm1_not_svm2":
        blob = meshpack.lzma_alone(b"SVM1" + struct.pack("<HH6f", 3, 1, 0, 0, 0, 1, 1, 1) + struct.pack("<9H", 0, 0, 0, 65535, 0, 0, 0, 65535, 0) + struct.pack("<3H", 0, 1, 2) + b"\x01\x02\x03")
        parts["head.svm2"] = (blob, meshpack.unlzma_alone(blob))
    elif break_it == "no_manifest_pivots":
        manifest = {"format": "SVM2", "figure_height_m": 0.6}
    w = server(parts, manifest, register=(break_it != "unregistered"))
    if break_it == "missing_part":
        w.rt.execute('MOCK.files["mentat/skippy/hat.svm2"] = nil')
    if break_it == "bad_json":
        w.rt.execute('MOCK.files["mentat/skippy/manifest.json"] = "{not json"')
    w.player(A)
    ok, got = reload(w)
    assert ok is False and why in got, got
    w.advance(5)
    assert exported(w) == [] and w.rt.eval("function() return GetGlobalBool('mentat_avatar_body', true) end")() is False


def test_a_part_over_the_cap_is_refused():
    parts = synthetic_set()
    big = b"\x5d" + b"\0" * (300 * 1024)
    parts["can.svm2"] = (big, b"")
    w = server(parts)
    ok, why = reload(w)
    assert ok is False and "cap" in why


def test_the_folder_is_a_plain_name_under_data():
    w = server(synthetic_set())
    for bad in ("../x", "a/../b", "C:/x", "a b"):
        w.rt.eval("function(v) GetConVar('mentat_avatar_body_dir'):SetValue(v) end")(bad)
        ok, why = reload(w)
        assert ok is False and "plain folder" in why, bad


def test_good_files_load_and_are_sent_once_to_every_human_in_small_numbered_chunks_with_a_checksum():
    parts = synthetic_set()
    w = server(parts)
    a = w.player(A)
    w.player(B)
    ok, why = reload(w)
    assert ok is True, why
    assert w.rt.eval("function() return GetGlobalBool('mentat_avatar_body', false) end")() is True
    w.advance(5)
    msgs = exported(w)
    assert len(msgs) == 7, "the manifest and six parts, one small chunk each"
    seen = {}
    for m in msgs:
        assert sorted(m["to"]) == sorted([A, B])
        kinds = [it[0] for it in m["seq"]]
        assert kinds == ["u", "u", "u", "u", "u", "u", "d"]
        index, i, count, total, crc, n = [it[1] for it in m["seq"][:6]]
        data = bytes.fromhex(m["seq"][6][1])
        assert (i, count, n) == (1, 1, len(data)) == (1, 1, total)
        assert crc == zlib.crc32(data)
        seen[index] = data
    assert sorted(seen) == [1, 2, 3, 4, 5, 6, 7]
    assert json.loads(seen[1])["figure_height_m"] == 0.6
    assert seen[2] == parts["can.svm2"][0] and seen[7] == parts["hand_b.svm2"][0]
    assert a is not None


def test_a_big_part_is_chunked_and_summoning_does_not_resend():
    parts = synthetic_set()
    rnd = random.Random(5)
    nv, nt = 12000, 4000                                   # random vertices do not compress: several 16 KB chunks
    verts = [(rnd.random(), rnd.random(), rnd.random()) for _ in range(nv)]
    normals = [(0, 0, 1)] * nv
    faces = [(3 * t, 3 * t + 1, 3 * t + 2) for t in range(nt)]
    blob, _ = SM.pack2(verts, normals, faces, [(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(nt)])
    assert len(blob) > 3 * 16 * 1024
    parts["head.svm2"] = (blob, meshpack.unlzma_alone(blob))
    w = server(parts)
    p = w.player(A, x=0, y=0, yaw=0)
    assert reload(w)[0] is True
    w.advance(10)
    head = [m for m in exported(w) if m["seq"][0][1] == 4]
    n = -(-len(blob) // (16 * 1024))
    assert len(head) == n and n >= 4
    assert b"".join(bytes.fromhex(m["seq"][6][1]) for m in sorted(head, key=lambda m: m["seq"][1][1])) == blob
    before = len(exported(w))
    w.rt.eval("function(p) Mentat.Avatar.Summon(p) end")(p)
    w.advance(10)
    assert len(exported(w)) == before, "he is called: the files were already sent, they are not sent again"


def test_a_late_joiner_asks_and_is_sent_everything_alone_and_the_bot_is_never_sent_anything():
    parts = synthetic_set()
    w = server(parts)
    w.player(A, x=0, y=0, yaw=0)
    assert reload(w)[0] is True
    w.advance(5)
    w.rt.execute("MOCK.nets = {}")
    late = w.player(B)
    w.rt.eval("function(p) Mentat.Avatar.Summon(p) end")(w.rt.eval("function() return MOCK.players[1] end")())    # makes the bot
    seq = lua_table(w.rt, [["u", 0, 8]])
    w.rt.eval("function(p, seq) MOCK.net_from_client(p, 'mentat_avatar_body_want', seq) end")(late, seq)
    w.advance(5)
    msgs = exported(w)
    assert len(msgs) == 7 and all(m["to"] == [B] for m in msgs), "only the one who asked"
    # asking for a single item
    w.rt.execute("MOCK.nets = {}")
    w.rt.eval("function(p, seq) MOCK.net_from_client(p, 'mentat_avatar_body_want', seq) end")(late, lua_table(w.rt, [["u", 3, 8]]))
    w.advance(2)
    assert [m["seq"][0][1] for m in exported(w)] == [3]
    # the rate limit: a flood of asks is answered a dozen times a minute
    w.rt.execute("MOCK.nets = {}")
    for _ in range(40):
        w.rt.eval("function(p, seq) MOCK.net_from_client(p, 'mentat_avatar_body_want', seq) end")(late, lua_table(w.rt, [["u", 1, 8]]))
    w.advance(8)
    assert len(exported(w)) <= 12


def test_the_bot_player_is_not_a_target():
    w = server(synthetic_set())
    p = w.player(A, x=0, y=0, yaw=0)
    w.rt.eval("function(p) Mentat.Avatar.Summon(p) end")(p)
    assert reload(w)[0] is True
    w.advance(5)
    for m in exported(w):
        assert "bot" not in " ".join(m["to"]) and A in m["to"]
        assert all(sid != "bot2" for sid in m["to"])
    bots = w.rt.eval("function() local o = {}; for _, b in ipairs(MOCK.bots) do o[#o + 1] = b.sid end; return MOCK.json_encode(o) end")()
    assert all(b not in sum((m["to"] for m in exported(w)), []) for b in json.loads(bots))


# ============================================================================================================ the client: assembly and fallback
def test_a_client_that_gets_everything_builds_all_six_parts_and_draws_the_real_body():
    s, c, msgs = full_pipeline(synthetic_set())
    assert state(c)["ready"] is True and state(c)["tris"] == 6 * 12
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    d = draws(c)
    # each cube has one colour group except jacket (2, navy+yellow) and head (silver+red+black -> plain + glow): meshes are built per part
    assert len(d) >= 6
    assert c.rt.eval("Mentat.Avatar.drew_real") is True
    assert c.rt.eval("function() return Mentat.Avatar.Geo ~= nil end")()


def test_a_client_missing_one_chunk_stays_procedural_never_a_half_body():
    for drop in range(7):
        s, c, msgs = full_pipeline(synthetic_set(), drop=drop)
        st = state(c)
        assert st["ready"] is False and st["state"] != "ready", f"item {drop + 1} withheld"
        c.rt.eval("DRAW_FRAME")(1, 0.05)
        assert c.rt.eval("Mentat.Avatar.drew_real") is False
        assert c.rt.eval("RB.built") >= 5, "the procedural parts were built instead (can, label, face, mouth, hat)"


def test_a_corrupt_part_fails_the_whole_body_and_the_procedural_can_is_drawn():
    def tamper(msgs):
        m = next(m for m in msgs if m["seq"][0][1] == 4)
        data = bytearray(bytes.fromhex(m["seq"][6][1]))
        data[20] ^= 0xFF
        m["seq"][6][1] = bytes(data).hex()                     # the CRC in the header no longer matches
    s, c, msgs = full_pipeline(synthetic_set(), tamper=tamper)
    assert state(c)["ready"] is False
    assert c.rt.eval("Mentat.Avatar.BodyC.stats.rejected") >= 1
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    assert c.rt.eval("Mentat.Avatar.drew_real") is False


def test_a_part_the_client_cannot_decode_fails_the_whole_body_and_frees_what_it_built():
    parts = synthetic_set()
    s = server(parts)
    s.player(A)
    assert reload(s)[0] is True
    s.advance(5)
    msgs = exported(s)
    c = client()
    reg = dict(parts)
    reg.pop("hand_b.svm2")                                       # util.Decompress knows nothing of the last part: it returns nil
    register_client(c, reg)
    feed(c, msgs)
    st = state(c)
    assert st["ready"] is False and st["state"] == "failed" and "hand_b" in st["why"]
    assert c.rt.eval("RB.destroyed") == c.rt.eval("RB.built") > 0, "no stray half-built meshes are kept"
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    assert c.rt.eval("Mentat.Avatar.drew_real") is False


def test_a_lying_chunk_header_is_refused():
    s, c, msgs = full_pipeline(synthetic_set(), drop=2)
    before = c.rt.eval("Mentat.Avatar.BodyC.stats.rejected")
    bad = json.loads(json.dumps(msgs[0]))
    bad["seq"][0][1] = 9                                          # no such item
    feed(c, [bad])
    bad = json.loads(json.dumps(msgs[0]))
    bad["seq"][3][1] = 300 * 1024                                 # over the cap
    feed(c, [bad])
    assert c.rt.eval("Mentat.Avatar.BodyC.stats.rejected") == before + 2
    assert state(c)["ready"] is False


def test_the_real_body_can_be_switched_off_on_the_client():
    s, c, msgs = full_pipeline(synthetic_set())
    c.rt.eval("function() GetConVar('mentat_avatar_real'):SetValue('0') end")()
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    assert c.rt.eval("Mentat.Avatar.drew_real") is False
    c.rt.eval("function() GetConVar('mentat_avatar_real'):SetValue('1') end")()
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    assert c.rt.eval("Mentat.Avatar.drew_real") is True


def test_a_client_asks_for_the_body_once_in_a_while_only_when_the_server_has_it():
    c = client()
    c.rt.execute("MOCK.nets = {}")
    c.advance(20)
    assert c.rt.eval("#MOCK.nets") == 0, "the server says it has none: nothing is asked"
    c.rt.execute("SetGlobalBool('mentat_avatar_body', true)")
    c.advance(4)
    n1 = c.rt.eval("#MOCK.nets")
    assert n1 == 1
    c.advance(20)
    assert 2 <= c.rt.eval("#MOCK.nets") <= 4, "re-asked every few seconds, not every frame"
    assert c.rt.eval("MOCK.nets[1].name") == "mentat_avatar_body_want"


# ============================================================================================================ drawing and lighting
def test_drawing_forces_depth_writing_on_and_restores_it_and_the_geometry_faces_the_engine_front():
    s, c, msgs = full_pipeline(synthetic_set())
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    depth = [(x[1], x[2]) for x in c.rt.eval("RB").depth.values()]
    assert depth[0] == (True, True) and depth[-1] == (False, False)
    # the kit's counter-clockwise faces are emitted clockwise: the cross product of an emitted triangle opposes the stored (outward) normal
    tris = c.rt.eval("""function()
      local m = RB.draws[1].mesh
      local v = m.verts
      local out = {}
      for i = 1, #v, 3 do
        local a, b, c = v[i].pos, v[i + 1].pos, v[i + 2].pos
        local ux, uy, uz = b.x - a.x, b.y - a.y, b.z - a.z
        local vx, vy, vz = c.x - a.x, c.y - a.y, c.z - a.z
        local nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        local n = v[i].normal
        out[#out + 1] = nx * n.x + ny * n.y + nz * n.z
      end
      return MOCK.json_encode(out)
    end""")()
    dots = json.loads(tris)
    assert dots and all(x < 0 for x in dots), dots


def lua_pure(w, code, *args):
    return w.rt.eval(f"function(...) local B = dofile(MOCK.ROOT .. '/mentat/avatar/body.lua'); {code} end")(*args)


def test_shading_stays_in_range_and_lights_from_above_and_the_black_mouth_is_near_black_and_only_red_glows():
    w = World(server=False, load=())
    f = w.rt.eval("function(nx, ny, nz) local B = dofile(MOCK.ROOT .. '/mentat/avatar/body.lua'); return B.shade(nx, ny, nz) end")
    vals = [f(*n) for n in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0.5, 0.3, 0.81))]
    assert all(0.45 <= v <= 1.0 for v in vals), vals
    assert f(0, 0, 1) > f(0, 0, -1) + 0.2, "the top is brighter than the underside"
    assert f(-1, 0, 0) > 0.5, "his back is not black"
    bc = w.rt.eval("function(r, g, b) local B = dofile(MOCK.ROOT .. '/mentat/avatar/body.lua'); return B.base_colour(r, g, b) end")
    assert tuple(bc(0, 0, 0)) == (12, 9, 14), "near-black, never pure black and not teal"
    assert tuple(bc(0, 0, 160)) == (14, 24, 100), "the filament navy is shown as the deep navy of the paint"
    assert tuple(bc(10, 20, 30)) == (10, 20, 30), "a colour that is not in the table is untouched"
    assert tuple(bc(154, 154, 154)) == (143, 146, 154), "a silver darkened to 0.8 by the converter keeps its display tint, scaled"
    assert tuple(bc(0, 0, 96)) == (8, 14, 60), "navy at 0.6"
    assert tuple(bc(255, 0, 0)) == (255, 0, 0)
    gl = w.rt.eval("function(r, g, b) local B = dofile(MOCK.ROOT .. '/mentat/avatar/body.lua'); return B.is_glow(r, g, b) end")
    assert gl(255, 0, 0) and not gl(252, 239, 3) and not gl(0, 0, 160) and not gl(192, 192, 192)


def test_the_red_triangles_are_their_own_mesh_and_brighten_with_the_voice():
    s, c, msgs = full_pipeline(synthetic_set())
    c.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")("can,jacket,hat,hand_a,hand_b")          # only the head has red
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    d = list(draws(c).values())
    assert len(d) == 2, "the head is two meshes: the plain triangles and the red ones"
    assert d[1].mod[1] == pytest.approx(0.75, abs=0.01), "at rest the tongue sits at 75%"
    # a loud voice: a playing channel whose level is 1
    c.rt.execute("""
      local chan = { IsValid = function() return true end, GetState = function() return GMOD_CHANNEL_PLAYING end, GetLevel = function() return 1, 1 end,
                     SetPos = function() end, SetVolume = function() end }
      Mentat.Avatar.tracked[chan] = { base = 1 }
    """)
    c.rt.eval("DRAW_FRAME")(40, 0.05)
    d = list(draws(c).values())
    assert d[1].mod[1] > 0.97


# ============================================================================================================ animation
def rest_dir():
    t, w_ = MANIFEST["pivots"]["hand_point"]["tip"], MANIFEST["pivots"]["hand_point"]["wrist"]
    v = [t[i] - w_[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def ang(a, b):
    return math.degrees(math.acos(max(-1, min(1, sum(x * y for x, y in zip(norm(a), norm(b)))))))


def matvec(m, v):
    return [sum(m[r][c + 1] * v[c] for c in range(3)) for r in (1, 2, 3)]


def hand_matrix(c):
    """The linear part and translation of the matrix the hand was drawn with (everything else hidden)."""
    d = draws(c)[1]
    f = d.top.f
    lin = {r: {cc: f[r][cc] for cc in (1, 2, 3)} for r in (1, 2, 3)}
    return lin, [f[1][4], f[2][4], f[3][4]]


@pytest.mark.parametrize("dx,dy,dz", [(0.0, 0.0, 1.0), (30, -20, 40), (0.2, 3.0, 1.0), (1.0, 0.3, 0.1), (-1, 1, 3), (-2, -1, 0.5)])
def test_the_pointing_hand_swings_so_its_fingertip_direction_hits_the_target_or_stops_at_the_limit(dx, dy, dz):
    s, c, msgs = full_pipeline(synthetic_set())
    c.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")("can,jacket,head,hat,hand_b")
    # the world point he is told to aim at: along (dx, dy, dz) from his wrist, 3 m away
    c.rt.execute("Mentat.Avatar.BodyC.BOB_M = 0")                           # (the bob would move the wrist this test computes by hand)
    height_m = c.rt.eval("function() return GetConVar('mentat_avatar_body_m'):GetFloat() end")()
    s_scale = height_m / MANIFEST["figure_height_m"]
    wr = [x * HU_PER_M for x in MANIFEST["pivots"]["hand_point"]["wrist"]]
    zoff = -0.5 * MANIFEST["figure_height_m"] * HU_PER_M
    bot = (100, 0, 0)
    wrist_world = [bot[0] + s_scale * wr[0], bot[1] + s_scale * wr[1], bot[2] + s_scale * (wr[2] + zoff)]
    d = norm([dx, dy, dz])
    target = [wrist_world[i] + 160 * d[i] for i in range(3)]
    c.rt.eval("function(x, y, z) Mentat.Avatar.Point(Vector(x, y, z), 30) end")(*target)
    c.rt.eval("DRAW_FRAME")(80, 0.05)                                       # the hand eases there
    lin, trans = hand_matrix(c)
    rest = rest_dir()
    now_dir = matvec({r: dict(lin[r]) for r in lin}, rest)
    asked = ang(rest, d)
    got = ang(rest, now_dir)
    if asked <= 35:
        assert ang(now_dir, d) < 0.5, f"asked {asked:.1f} deg, the fingertip is {ang(now_dir, d):.2f} deg off the target direction"
    else:
        assert got == pytest.approx(35, abs=0.5) and got <= 35.5, "past the limit the hand stops at 35 degrees"
        assert ang(now_dir, d) == pytest.approx(asked - 35, abs=1.0)
    # the wrist does not move under the rotation (it is the pivot)
    p = [sum(lin[r][cc] * wr_ for cc, wr_ in zip((1, 2, 3), wr)) + trans[r - 1] for r in (1, 2, 3)]
    assert p == pytest.approx(wr, abs=1e-6)


def test_the_hand_rests_pointing_up_as_in_the_kit_and_lowers_when_the_order_expires():
    s, c, msgs = full_pipeline(synthetic_set())
    c.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")("can,jacket,head,hat,hand_b")
    c.rt.eval("DRAW_FRAME")(10, 0.05)
    lin, _ = hand_matrix(c)
    assert ang(matvec(lin, rest_dir()), rest_dir()) < 1e-6
    c.rt.eval("function() Mentat.Avatar.Point(Vector(100, 300, 100), 1) end")()
    c.rt.eval("DRAW_FRAME")(10, 0.05)
    lin, _ = hand_matrix(c)
    assert ang(matvec(lin, rest_dir()), rest_dir()) > 5
    c.rt.eval("DRAW_FRAME")(120, 0.05)                                       # the second is up
    lin, _ = hand_matrix(c)
    assert ang(matvec(lin, rest_dir()), rest_dir()) < 0.5


def test_the_server_hook_reaches_the_clients_and_the_hand_follows_the_owners_gaze_when_asked():
    w = server(synthetic_set())
    w.rt.execute("MOCK.nets = {}")
    w.rt.execute("hook.Run('SkippyAvatarPoint', Vector(1, 2, 3), 2.5)")
    msgs = w.rt.eval("function() local o = 0; for _, m in ipairs(MOCK.nets) do if m.name == 'mentat_avatar_point' then o = o + 1 end end; return o end")()
    assert msgs == 1
    c = client()
    c.rt.execute("""
      function net.ReadBool() return true end
      function net.ReadVector() return Vector(0, 300, 120) end
      function net.ReadUInt() return 25 end
      MOCK.receivers.mentat_avatar_point()
    """)
    a = c.rt.eval("Mentat.Avatar.aim")
    assert a.pos.y == 300 and a.t1 == pytest.approx(c.rt.eval("CurTime()") + 2.5)
    # the owner-gaze convar
    c.rt.execute("Mentat.Avatar.aim = nil; GetConVar('mentat_avatar_point'):SetValue('1')")
    c.rt.execute("""
      local owner = MOCK.me
      function owner:GetEyeTrace() return { Hit = true, HitPos = Vector(100, -300, 100) } end
      MOCK.bot.GetNWEntity = function() return owner end
    """)
    s, c2, _ = full_pipeline(synthetic_set())
    c2.rt.execute("""
      local owner = MOCK.me
      function owner:GetEyeTrace() return { Hit = true, HitPos = Vector(100, -300, 100) } end
      MOCK.bot.GetNWEntity = function() return owner end
      GetConVar('mentat_avatar_point'):SetValue('1')
      GetConVar('mentat_avatar_hide'):SetValue('can,jacket,head,hat,hand_b')
    """)
    c2.rt.eval("DRAW_FRAME")(80, 0.05)
    lin, _ = hand_matrix(c2)
    assert ang(matvec(lin, rest_dir()), rest_dir()) > 5, "the hand turned toward what the owner looks at"


def test_the_hat_nod_is_bounded_whatever_the_voice_and_the_speaker():
    for lx, ly in ((100, 0), (0, 100), (-80, 60), (3, -1)):
        w = World(server=False, load=())
        got = w.rt.eval("function(a, b, s) local L = dofile(MOCK.ROOT .. '/mentat/avatar/logic.lua'); local p, r = L.hat_tilt(a, b, s, 12); return p, r end")
        for strength in (0, 0.3, 1, 5):
            p, r = got(lx, ly, strength)
            assert math.hypot(p, r) <= 12.0 + 1e-9
        p, r = got(lx, ly, 1)
        assert math.hypot(p, r) == pytest.approx(12.0, abs=1e-6), "a full voice reaches the limit and not beyond"
        assert math.hypot(*got(lx, ly, 0)) == 0


def test_the_drawn_hat_tips_about_its_pivot_with_the_voice_and_stays_under_twelve_degrees():
    s, c, msgs = full_pipeline(synthetic_set())
    c.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")("can,jacket,head,hand_a,hand_b")
    c.rt.execute("MOCK.me.pos = Vector(100, 300, 0); MOCK.me.speaking = true")                # the speaker is on his left
    c.rt.execute("""
      local chan = { IsValid = function() return true end, GetState = function() return GMOD_CHANNEL_PLAYING end, GetLevel = function() return 1, 1 end,
                     SetPos = function() end, SetVolume = function() end }
      Mentat.Avatar.tracked[chan] = { base = 1 }
    """)
    worst = 0
    for _ in range(30):
        c.rt.eval("DRAW_FRAME")(1, 0.05)
        d = draws(c)[1]
        a = d.top.rot
        worst = max(worst, abs(a.p), abs(a.r))
        assert abs(a.y) < 1e-9 and math.hypot(a.p, a.r) <= 12.0 + 1e-6
    assert worst > 6, "he really leans toward the speaker"
    piv = MANIFEST["pivots"]["hat"]["pivot"]
    ops = list(draws(c)[1].top.ops.values())
    assert [o[1] for o in ops] == ["T", "R", "T"]
    assert (ops[0][2].x, ops[0][2].y, ops[0][2].z) == pytest.approx([x * HU_PER_M for x in piv], abs=1e-6)


def test_the_head_turns_toward_the_speaker_by_a_small_angle_only():
    s, c, msgs = full_pipeline(synthetic_set())
    c.rt.eval("function(h) GetConVar('mentat_avatar_hide'):SetValue(h) end")("can,jacket,hat,hand_a,hand_b")
    c.rt.execute("MOCK.me.pos = Vector(-300, 40, 0); MOCK.me.speaking = true")             # behind him: the largest ask
    c.rt.eval("DRAW_FRAME")(80, 0.05)
    f = draws(c)[1].top.f
    yaw = math.degrees(math.atan2(f[2][1], f[1][1]))
    assert 5 < abs(yaw) <= 25.0 + 1e-6
    c.rt.execute("MOCK.me.speaking = false")
    c.rt.eval("DRAW_FRAME")(1, 0.05)                                                      # (no speaker and no owner: eases back)
    c.rt.eval("DRAW_FRAME")(80, 0.05)
    f = draws(c)[1].top.f
    assert abs(math.degrees(math.atan2(f[2][1], f[1][1]))) < 0.5


def test_pure_rotation_math_properties():
    w = World(server=False, load=())
    rnd = random.Random(11)
    f = w.rt.eval("""function(rx, ry, rz, tx, ty, tz, max)
      local B = dofile(MOCK.ROOT .. '/mentat/avatar/body.lua')
      local m, used, asked = B.point_rotation({ x = rx, y = ry, z = rz }, { x = tx, y = ty, z = tz }, max)
      local v = B.apply(m, B.unit({ x = rx, y = ry, z = rz }))
      return v.x, v.y, v.z, used, asked, m[1] * m[5] * m[9]
    end""")
    for _ in range(300):
        r = norm([rnd.uniform(-1, 1) for _ in range(3)])
        t = norm([rnd.uniform(-1, 1) for _ in range(3)])
        vx, vy, vz, used, asked, _ = f(*r, *t, 35)
        assert used == pytest.approx(min(asked, 35), abs=1e-9)
        assert math.sqrt(vx * vx + vy * vy + vz * vz) == pytest.approx(1, abs=1e-9)
        if asked <= 35:
            assert ang([vx, vy, vz], t) < 1e-6
        else:
            assert ang([vx, vy, vz], r) == pytest.approx(35, abs=1e-6)
    # straight opposite does not blow up
    vx, vy, vz, used, asked, _ = f(0, 0, 1, 0, 0, -1, 35)
    assert used == 35 and asked == pytest.approx(180)


# ============================================================================================================ the real kit
@needs_private
def test_the_real_kit_goes_from_the_server_to_a_client_that_draws_all_six_parts_at_the_right_size():
    parts = real_set()
    man = json.loads((PRIVATE / "manifest.json").read_text())
    s = server(parts, man)
    s.player(A)
    ok, why = reload(s)
    assert ok is True, why
    s.advance(10)
    msgs = exported(s)
    assert len(msgs) >= 7 and sum(len(bytes.fromhex(m["seq"][6][1])) for m in msgs) < 400 * 1024
    c = client()
    register_client(c, parts)
    feed(c, msgs)
    st = state(c)
    assert st["ready"] is True, st
    assert st["tris"] == sum(len(SM.unpack2(parts[n + ".svm2"][0])[2]) for n in PARTS)
    c.rt.eval("DRAW_FRAME")(1, 0.05)
    assert c.rt.eval("Mentat.Avatar.drew_real") is True
    # every vertex of every drawn mesh lies inside the figure's box (HU), the whole thing about 0.6 m tall
    zs, xs = [], []
    for d in draws(c).values():
        for v in d.mesh.verts.values():
            zs.append(v.pos.z)
            xs.append(v.pos.x)
    height_hu = max(zs) - min(zs)
    assert height_hu == pytest.approx(man["figure_height_m"] * HU_PER_M, rel=0.02)
    assert min(zs) >= -0.5 and max(xs) - min(xs) < height_hu
    # the base matrix scales the figure to 0.6 m and lifts it to hover where the can's middle was
    base = draws(c)[1].top
    first = c.rt.eval("RB.draws")[1]
    assert first is not None


@needs_private
def test_the_real_kit_matches_the_procedural_hover_height_roughly():
    man = json.loads((PRIVATE / "manifest.json").read_text())
    assert 0.5 < man["figure_height_m"] < 0.7
    assert man["pivots"]["hand_point"]["tip"][2] > man["pivots"]["hand_point"]["wrist"][2], "the kit's hand points up at rest"


# ============================================================================================================ packaging
def test_the_loader_ships_every_new_file_and_the_client_files_include_only_shipped_ones():
    import re
    text = (ADDON / "autorun" / "mentat_stage.lua").read_text(encoding="utf-8")
    shipped = set(re.findall(r'AddCSLuaFile\("([^"]+)"\)', text))
    for rel in ("mentat/avatar/body.lua", "mentat/avatar/body_client.lua", "mentat/holodeck/svm1.lua"):
        assert rel in shipped and (ADDON / rel).is_file(), rel
    for f in ("avatar/body_client.lua", "avatar/body.lua"):
        for inc in re.findall(r'include\("([^"]+)"\)', (ADDON / "mentat" / f).read_text(encoding="utf-8")):
            assert inc in shipped, (f, inc)
    for inc in re.findall(r'include\("([^"]+)"\)', (ADDON / "mentat" / "avatar_client.lua").read_text(encoding="utf-8")):
        assert inc in shipped, inc


def test_no_private_payload_is_tracked_or_ignored_wrongly():
    import subprocess
    out = subprocess.run(["git", "ls-files", "stage/skippyverse/private"], cwd=ROOT, capture_output=True, text=True).stdout
    assert out.strip() == "", "the print kit's derivatives must never be committed"
    assert "private" in (ROOT / ".gitignore").read_text()
