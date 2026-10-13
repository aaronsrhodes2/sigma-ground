"""The real shape of a found object: the bridge's POST /mesh, the server relaying it to the clients once per model, the client decoding
it and drawing it in place of the box. Offline (no network: a guard fails any non-loopback connection).

Three runtimes meet here: the REAL bridge functions (Python), the REAL server addon (holodeck.lua + meshnet.lua + the entity) against
gmod_mock.lua, and the REAL client files (meshclient.lua + svm1.lua + the entity's cl_init.lua) against a second mock; the net messages
one records are carried to the other. util.Decompress is a lookup in the mock (LZMA itself is not reimplemented): the real engine's
reading of our framing was checked live (stage/gmod/addon/lua/mentat/tools/lzma_probe.lua, see stage/gmod/holodeck/README.md).
"""
from __future__ import annotations

import base64
import json
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
sys.path.insert(0, str(ROOT / "stage" / "gmod" / "holodeck"))
sys.path.insert(0, str(HERE))

import skippy_bridge  # noqa: E402
from skippyverse import acquire as A  # noqa: E402
from skippyverse import cards as C  # noqa: E402
from skippyverse import index as IX  # noqa: E402
from skippyverse import meshpack  # noqa: E402
from test_gmod_found import ALICE, BOB, CLIENT_FAKES, Game, Http, client_box, found_card, no_network, sized  # noqa: E402,F401
from test_gmod_glue import World, lua_table  # noqa: E402

HU = C.HU_M
M_PER_HU = 0.01905

# a model of 0.6 m (x) by 0.3 m (depth, glTF z) by 1.2 m (height, glTF y): 12 triangles, red
_V = [(x, y, z) for x in (-0.3, 0.3) for y in (0.0, 1.2) for z in (-0.15, 0.15)]
_I = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
BOX_W, BOX_D, BOX_H = 0.6, 0.3, 1.2


def box_routes(pid):
    """What Poly Haven would answer for `pid`: the file list, a glTF with an external .bin, and the .bin. (A texture is listed and must never be fetched.)"""
    pos = b"".join(struct.pack("<3f", *v) for v in _V)
    idx = b"".join(struct.pack("<H", i) for i in _I)
    blob = pos + idx
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(blob), "uri": f"{pid}.bin"}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)}, {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx)}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-0.3, 0, -0.15], "max": [0.3, 1.2, 0.15]},
                         {"bufferView": 1, "componentType": 5123, "count": len(_I), "type": "SCALAR"}],
           "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.2, 0.1, 1.0]}}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
           "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}]}
    gl = json.dumps(doc).encode()
    files = {"gltf": {"1k": {"gltf": {"url": f"https://dl.example/{pid}_1k.gltf", "size": len(gl), "include": {
        f"{pid}.bin": {"url": f"https://dl.example/{pid}.bin", "size": len(blob)},
        f"textures/{pid}_diff_1k.jpg": {"url": "https://dl.example/tex.jpg", "size": 5}}}}}}
    return {f"https://api.polyhaven.com/files/{pid}": json.dumps(files).encode(),
            f"https://dl.example/{pid}_1k.gltf": gl, f"https://dl.example/{pid}.bin": blob}


class Fetch:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, *, max_bytes):
        self.calls.append(url)
        if url not in self.routes:
            raise AssertionError(f"unexpected fetch {url}")
        return self.routes[url]


def cards():
    return [
        found_card("garden gnome", sized(BOX_W, BOX_D, BOX_H), aliases=["gnome statue"], url="https://polyhaven.com/a/garden_gnome"),
        found_card("fuel widget", sized(0.5, 0.5, 0.5), source="Gazebo Fuel", lic="CC-BY-4.0", url="https://fuel.gazebosim.org/1.0/O/models/fuel_widget"),
        found_card("nasa gizmo", sized(1.0, 1.0, 1.0, "guessed"), source="NASA 3D Resources", lic="PD-USGOV", url="https://github.com/nasa/NASA-3D-Resources"),
    ]


@pytest.fixture()
def fetch(tmp_path, monkeypatch):
    f = Fetch(box_routes("garden_gnome"))
    monkeypatch.setattr(skippy_bridge, "FOUND_FETCHER", f)
    monkeypatch.setattr(skippy_bridge, "MESH_CACHE_DIR", tmp_path / "assets")
    monkeypatch.setattr(skippy_bridge, "MESH_PACKED", {})
    return f


@pytest.fixture()
def ix(tmp_path, monkeypatch, fetch):
    IX.build_index(cards(), tmp_path / "ix", synonym_groups=[])
    index = IX.load_index(tmp_path / "ix", shared=True)
    monkeypatch.setattr(skippy_bridge, "FOUND_INDEX", index)
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", None)
    monkeypatch.setattr(skippy_bridge, "FOUND_FETCH", False)
    monkeypatch.setattr(skippy_bridge, "GATE", skippy_bridge.Gate())
    return index


TALL = "found.poly-haven.garden-gnome"


# ============================================================================================================ the payload codec
def test_lzma_alone_is_the_framing_the_engine_reads_known_size_in_the_header():
    body = b"holodeck mesh payload 0123456789 " * 20
    blob = meshpack.lzma_alone(body)
    assert blob[0] == 0x5D, "props byte: lc=3 lp=0 pb=2"
    assert struct.unpack_from("<Q", blob, 5)[0] == len(body), "util.Decompress needs the real length here; -1 (unknown) returns nothing in the real game"
    assert meshpack.unlzma_alone(blob) == body


def test_pack_with_the_lzma_codec_round_trips_and_is_smaller_than_the_raw_bytes(tmp_path):
    routes = box_routes("garden_gnome")
    d = tmp_path / "m"
    d.mkdir()
    (d / "garden_gnome_1k.gltf").write_bytes(routes["https://dl.example/garden_gnome_1k.gltf"])
    (d / "garden_gnome.bin").write_bytes(routes["https://dl.example/garden_gnome.bin"])
    blob, info = meshpack.pack(d / "garden_gnome_1k.gltf", 100, "lzma")
    assert info["codec"] == "lzma" and info["tris_out"] == 12
    verts, faces, cols = meshpack.unpack(blob)
    assert len(faces) == 12 and max(v[2] for v in verts) == pytest.approx(BOX_H, abs=1e-3)
    assert tuple(cols[0]) == (230, 123, 90), "the material's base colour, linear to display"
    with pytest.raises(ValueError):
        meshpack.pack(d / "garden_gnome_1k.gltf", 100, "gzip")


# ============================================================================================================ the bridge half
def test_resolve_offers_a_mesh_for_a_glTF_card_and_says_its_key(ix):
    r = skippy_bridge.resolve_found("garden gnome")
    assert r["place"] is True and r["mesh"] == {"key": skippy_bridge.mesh_key(TALL), "format": skippy_bridge.MESH_FORMAT}
    assert r["kind"]["variants"][0]["found"]["mesh"] == r["mesh"]["key"]
    assert len(r["mesh"]["key"]) == 16
    fuel = skippy_bridge.resolve_found("fuel widget")
    assert fuel["place"] is True and fuel["mesh"] is None and "mesh" not in fuel["kind"]["variants"][0]["found"], "a Fuel card that was never downloaded stays a box (its file is not known)"
    assert skippy_bridge.resolve_found("dragon")["mesh"] is None


# ------------------------------------------------------------------------------------------------------- NASA and Gazebo Fuel cards
NASA_PATH = "3D Models/Gizmo/Gizmo.glb"
FUEL_SHELF = ('<sdf version="1.5"><model name="Shelf"><link name="l"><visual name="a"><pose>0 0 0.5 0 0 0</pose><geometry><box><size>0.8 0.3 1.0</size></box></geometry>'
              '<material><diffuse>0.2 0.4 0.6</diffuse></material></visual><collision name="c"><geometry><box><size>5 5 5</size></box></geometry></collision></link></model></sdf>')
FUEL_MESHED = FUEL_SHELF.replace("</link>", '<visual name="m"><geometry><mesh><uri>model://Dresser/meshes/y.stl</uri></mesh></geometry></visual></link>')
FUEL_OBJ = FUEL_SHELF.replace("</link>", '<visual name="m"><geometry><mesh><uri>model://Locker/meshes/x.obj</uri></mesh></geometry></visual></link>')


OBJ_TRI = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


STL_TRI = (bytes(80) + struct.pack("<I", 1) + struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0))


def _zip(files):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


@pytest.fixture()
def real_sources(tmp_path, monkeypatch):
    """A NASA glb (a 2 x 1 x 3 box in its own units), and two Fuel models: one of primitives, one that needs an OBJ/Collada mesh."""
    from test_skippyverse_acquire import glb
    prov_nasa = {"asset": NASA_PATH}
    nasa = found_card("nasa gizmo", None, source="NASA 3D Resources", lic="PD-USGOV", url="https://github.com/nasa/NASA-3D-Resources")
    nasa["provenance"].update(prov_nasa)
    shelf = found_card("test shelf", None, source="Gazebo Fuel", lic="CC0", url="https://fuel.gazebosim.org/O/models/Shelf")
    meshed = found_card("test dresser", None, source="Gazebo Fuel", lic="CC0", url="https://fuel.gazebosim.org/O/models/Dresser")
    obj_model = found_card("test locker", None, source="Gazebo Fuel", lic="CC0", url="https://fuel.gazebosim.org/O/models/Locker")
    routes = {A.nasa_url(NASA_PATH): glb([{"mesh": 0}]),
              A.fuel_url("O", "Shelf"): _zip({"model.config": "<model/>", "model.sdf": FUEL_SHELF}),
              A.fuel_url("O", "Locker"): _zip({"model.sdf": FUEL_OBJ, "meshes/x.obj": OBJ_TRI}),
              A.fuel_url("O", "Dresser"): _zip({"model.sdf": FUEL_MESHED, "meshes/x.obj": OBJ_TRI, "meshes/y.stl": STL_TRI})}
    IX.build_index([nasa, shelf, meshed, obj_model], tmp_path / "ix2", synonym_groups=[])
    monkeypatch.setattr(skippy_bridge, "FOUND_INDEX", IX.load_index(tmp_path / "ix2", shared=True))
    f = Fetch(routes)
    monkeypatch.setattr(skippy_bridge, "FOUND_FETCHER", f)
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", None)
    monkeypatch.setattr(skippy_bridge, "FOUND_FETCH", True)
    monkeypatch.setattr(skippy_bridge, "MESH_CACHE_DIR", tmp_path / "assets")
    monkeypatch.setattr(A, "DEFAULT_CACHE", tmp_path / "assets")
    monkeypatch.setattr(skippy_bridge, "MESH_PACKED", {})
    monkeypatch.setattr(skippy_bridge, "GATE", skippy_bridge.Gate())
    return f


def test_a_nasa_card_gets_a_mesh_at_the_size_the_card_was_given(real_sources):
    r = skippy_bridge.resolve_found("nasa gizmo")
    assert r["place"] is True and r["card"]["size_m"]["basis"] == "guessed"
    assert r["mesh"] and r["kind"]["variants"][0]["found"]["mesh"] == r["mesh"]["key"]
    p = skippy_bridge.mesh_payload("found.nasa-3d-resources.nasa-gizmo", r["mesh"]["key"])
    sz = r["card"]["size_m"]
    assert p["bbox_m"] == pytest.approx([sz["w"], sz["d"], sz["h"]], abs=1e-3), "the picture is as big as the box the card stands as"


def test_a_draco_compressed_nasa_glb_stays_a_box_and_is_never_offered(real_sources, monkeypatch):
    import struct as st
    doc = {"asset": {"version": "2.0"}, "extensionsUsed": ["KHR_draco_mesh_compression"], "extensionsRequired": ["KHR_draco_mesh_compression"],
           "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "extensions": {"KHR_draco_mesh_compression": {}}}]}],
           "accessors": [{"componentType": 5126, "count": 8, "type": "VEC3", "min": [0, 0, 0], "max": [2, 1, 3]}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    real_sources.routes[A.nasa_url(NASA_PATH)] = b"glTF" + st.pack("<II", 2, 20 + len(js)) + st.pack("<II", len(js), 0x4E4F534A) + js
    r = skippy_bridge.resolve_found("nasa gizmo")
    assert r["place"] is True and r["mesh"] is None, "the box is placed (the size is in the file's accessors), but no mesh is promised for geometry we cannot read"
    with pytest.raises(ValueError, match="no mesh"):
        skippy_bridge.mesh_payload("found.nasa-3d-resources.nasa-gizmo", skippy_bridge.mesh_key("found.nasa-3d-resources.nasa-gizmo"))


def test_a_nasa_card_whose_size_is_a_verified_published_length_is_scaled_to_it(real_sources, monkeypatch):
    monkeypatch.setattr(A, "load_nasa_sizes", lambda path=None: {"Gizmo": {"longest_m": 30.0, "source_url": "https://nasa.example/fact", "verified": True}})
    r = skippy_bridge.resolve_found("nasa gizmo")
    assert r["card"]["size_m"]["basis"] == "sourced"
    p = skippy_bridge.mesh_payload("found.nasa-3d-resources.nasa-gizmo", r["mesh"]["key"])
    assert max(p["bbox_m"]) == pytest.approx(30.0, abs=0.05), "the glb's own units (3) became the published 30 m"
    assert p["bbox_m"] == pytest.approx([r["card"]["size_m"][k] for k in ("w", "d", "h")], abs=0.05)


def test_a_fuel_model_of_primitives_gets_a_mesh_and_one_with_a_collada_mesh_stays_a_box(real_sources):
    shelf = skippy_bridge.resolve_found("test shelf")
    assert shelf["place"] is True and shelf["fetched"] is True and shelf["mesh"], "sized by the one download, then offered"
    assert shelf["card"]["size_m"]["basis"] == "measured"
    p = skippy_bridge.mesh_payload("found.gazebo-fuel.test-shelf", shelf["mesh"]["key"])
    n_calls = len(real_sources.calls)
    assert p["tris"] == 12 and p["bbox_m"] == pytest.approx([0.8, 0.3, 1.0], abs=1e-3), "the collision box (5 m) is not drawn"
    verts, faces, cols = meshpack.unpack(base64.b64decode(p["b64"]))
    assert set(cols) == {(51, 102, 153)}
    assert len(real_sources.calls) == n_calls, "the /mesh used the file the /resolve had already fetched"
    locker = skippy_bridge.resolve_found("test locker")
    assert locker["place"] is True and locker["mesh"], "an OBJ mesh visual is read"
    assert skippy_bridge.mesh_payload("found.gazebo-fuel.test-locker", locker["mesh"]["key"])["tris"] == 12 + 1
    dresser = skippy_bridge.resolve_found("test dresser")
    assert dresser["place"] is True and dresser["mesh"] is None and "mesh" not in dresser["kind"]["variants"][0]["found"], "an STL mesh is not read: the box stays"
    with pytest.raises(ValueError, match="no mesh"):
        skippy_bridge.mesh_payload("found.gazebo-fuel.test-dresser", skippy_bridge.mesh_key("found.gazebo-fuel.test-dresser"))


def test_a_fuel_card_that_was_never_downloaded_is_not_offered_a_mesh(real_sources):
    ix = skippy_bridge.found_index()
    assert skippy_bridge.mesh_offer(ix.card("found.gazebo-fuel.test-shelf")) is None


def test_the_mesh_endpoint_downloads_once_packs_once_and_never_fetches_a_texture(ix, fetch):
    key = skippy_bridge.mesh_key(TALL)
    a = skippy_bridge.mesh_payload(TALL, key)
    assert sorted(fetch.calls) == ["https://api.polyhaven.com/files/garden_gnome", "https://dl.example/garden_gnome.bin", "https://dl.example/garden_gnome_1k.gltf"]
    b = skippy_bridge.mesh_payload(TALL, key)
    assert b is a and len(fetch.calls) == 3, "the second ask is answered from memory"
    blob = base64.b64decode(a["b64"])
    assert len(blob) == a["bytes"] <= skippy_bridge.MESH_MAX_BYTES and zlib.crc32(blob) == a["crc32"]
    assert a["format"] == skippy_bridge.MESH_FORMAT == "svm3-lzma" and a["tris"] == 12 and a["verts"] == 24, "a box: 12 triangles, 8 corners split into 3 (its creases keep their own normals)"
    assert a["bbox_m"] == pytest.approx([BOX_W, BOX_D, BOX_H], abs=1e-3), "x = the card's width, y = its depth, z = height"
    verts, faces, _ = meshpack.unpack(blob)
    assert len(faces) == 12 and min(v[2] for v in verts) == pytest.approx(0, abs=1e-6), "on the floor"


@pytest.mark.parametrize("card,key,why", [
    (TALL, "0" * 16, "key"),                                            # not this card's key
    ("found.gazebo-fuel.fuel-widget", None, "no mesh"),                   # a card with no mesh offer
    ("found.poly-haven.nothing-here", None, "no mesh"),                   # a card the index does not have
    ("../../etc/passwd", "0" * 16, "card"),
    (12, "0" * 16, "card"),
])
def test_the_mesh_endpoint_refuses_what_it_did_not_offer(ix, fetch, card, key, why):
    key = key or (skippy_bridge.mesh_key(card) if isinstance(card, str) else "x")
    with pytest.raises(ValueError, match=why):
        skippy_bridge.mesh_payload(card, key)
    assert fetch.calls == [], "nothing is downloaded for a refused request"


def test_a_payload_over_the_cap_is_refused_not_sent(ix, monkeypatch):
    monkeypatch.setattr(skippy_bridge, "MESH_MAX_BYTES", 30)
    with pytest.raises(ValueError, match="over the 30 byte cap"):
        skippy_bridge.mesh_payload(TALL, skippy_bridge.mesh_key(TALL))


def test_the_mesh_http_endpoint_is_guarded_rationed_and_exclusive(ix, fetch, monkeypatch):
    h = Http(monkeypatch)
    try:
        body = {"card": TALL, "key": skippy_bridge.mesh_key(TALL), "who": "alice"}
        assert h.post("/mesh", body, token="wrong")[0] == 401
        code, got = h.post("/mesh", body)
        assert code == 200 and got["ok"] and got["tris"] == 12 and got["key"] == body["key"]
        assert h.post("/mesh", {**body, "key": "f" * 16})[0] == 400
        codes = [h.post("/mesh", body)[0] for _ in range(4)]
        assert codes[-1] == 429, f"one caller may ask {skippy_bridge.PER_CALLER_PER_MINUTE['mesh']} a minute: {codes}"
        monkeypatch.setattr(skippy_bridge, "MESH_PACKED", {})
        with skippy_bridge.GATE.slot("fetch"):
            assert h.post("/mesh", {**body, "who": "bob"})[0] == 503, "one download at a time"
    finally:
        h.close()


# ============================================================================================================ a synthetic payload for chunking
def big_payload(tris=4000, seed=3):
    """A valid SVM1 payload that does not compress (random vertices), so it needs several 16 KB chunks. Returns (raw, lzma blob)."""
    rnd = random.Random(seed)
    nv = tris * 3
    body = bytearray(b"SVM1" + struct.pack("<HH6f", nv, tris, 0, 0, 0, 1, 1, 1))
    for _ in range(nv):
        body += struct.pack("<3H", *(rnd.randrange(65536) for _ in range(3)))
    for t in range(tris):
        body += struct.pack("<3H", 3 * t, 3 * t + 1, 3 * t + 2)
    for _ in range(tris):
        body += struct.pack("<3B", *(rnd.randrange(256) for _ in range(3)))
    return bytes(body), meshpack.lzma_alone(bytes(body))


# ============================================================================================================ the server half
SERVER_HELPERS = r"""
function T.mesh_msgs()
  local out = {}
  for _, m in ipairs(MOCK.nets) do
    if m.name == "holodeck_mesh" then
      local sids = {}
      for _, p in ipairs(m.to) do sids[#sids + 1] = p.sid end
      local e = MOCK.export(m)
      out[#out + 1] = { to = sids, seq = e.seq, bytes = m.bytes }
    end
  end
  return MOCK.json_encode(out)
end
function T.forget_nets() MOCK.nets = {} end
function T.visuals()
  local o = {}
  for _, e in ipairs(MOCK.ents) do
    if e.valid and e.class == "holodeck_found" then o[#o + 1] = { visual = e:GetVisual(), mesh = e:GetMeshId() } end
  end
  return MOCK.json_encode(o)
end
function T.register(blob_hex, raw_hex) MOCK.lzma[MOCK.fromhex(blob_hex)] = MOCK.fromhex(raw_hex) end
function T.mesh_state() local m = next(Holodeck.Mesh.by_key); local r = m and Holodeck.Mesh.by_key[m]; return r and r.state or "none" end
"""


class MeshGame(Game):
    def __init__(self, **cv):
        super().__init__(**cv)
        self.w.rt.execute(SERVER_HELPERS)
        self.mesh_answered = set()

    def mesh_posts(self):
        return self.posts("/mesh")

    def register(self, blob: bytes, raw: bytes):
        self.w.rt.eval("function(a, b) T.register(a, b) end")(blob.hex(), raw.hex())

    def serve_mesh(self, answer=None):
        """Answer every unanswered /mesh with the real bridge (or `answer(body)` -> (code, obj))."""
        n = 0
        for i in self.mesh_posts():
            if i in self.mesh_answered:
                continue
            self.mesh_answered.add(i)
            body = self.body(i)
            if answer:
                code, obj = answer(body)
            else:
                obj = skippy_bridge.mesh_payload(body["card"], body["key"])
                self.register(base64.b64decode(obj["b64"]), meshpack.unlzma_alone(base64.b64decode(obj["b64"])))
                code = 200
            self.w.rt.eval("function(i, code, body) MOCK.respond(i, code, body) end")(i, code, json.dumps(obj))
            n += 1
        return n

    def nets(self):
        return json.loads(self.w.call("T.mesh_msgs"))

    def visuals(self):
        return json.loads(self.w.call("T.visuals"))

    def forget_nets(self):
        self.w.call("T.forget_nets")


@pytest.fixture()
def game(ix):
    return MeshGame()


def summon(game, text="add a garden gnome", sid=ALICE):
    p = game.player(sid, admin=True)
    game.summon(p, text)
    return p


def test_a_found_object_with_a_mesh_asks_the_bridge_once_and_the_box_shows_meanwhile(game):
    summon(game)
    (box,) = game.found()
    assert box["visual"] == "box"
    posts = game.mesh_posts()
    assert len(posts) == 1
    assert game.body(posts[0]) == {"card": TALL, "key": skippy_bridge.mesh_key(TALL), "who": ALICE}
    assert game.headers(posts[0])["X-Skippy-Token"] == "tok"
    assert game.visuals() == [{"visual": "box", "mesh": 0}], "until the mesh is there it is the box (no re-plan, no second entity)"
    assert game.w.call("T.mesh_state") == "pending"


def test_three_objects_of_one_model_cost_one_download_one_send_and_all_get_the_mesh(game):
    p = summon(game, "add three garden gnomes")
    assert len(game.found()) == 3 and len(game.mesh_posts()) == 1, "once per model, not per entity"
    game.player(BOB)                                                     # a second player is connected when it arrives
    game.serve_mesh()
    game.w.advance(2)
    assert game.visuals() == [{"visual": "mesh", "mesh": 1}] * 3, "every entity of the model is told the number"
    msgs = game.nets()
    assert len(msgs) == 1, "one chunk (a small payload) sent once, not three times"
    assert sorted(msgs[0]["to"]) == sorted([ALICE, BOB]), "to everyone connected"
    assert len(game.found()) == 3 and game.count() == 3, "nothing was re-planned or re-spawned"


def test_the_chunks_are_small_numbered_and_reassemble_to_the_exact_payload(game):
    raw, blob = big_payload()
    assert len(blob) > 3 * 16 * 1024
    summon(game)
    game.register(blob, raw)
    obj = {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "svm1-lzma", "bytes": len(blob), "crc32": zlib.crc32(blob),
           "b64": base64.b64encode(blob).decode()}
    game.serve_mesh(lambda b: (200, obj))
    game.w.advance(5)
    msgs = game.nets()
    n = -(-len(blob) // (16 * 1024))
    assert len(msgs) == n and n >= 4
    parts = {}
    for k, m in enumerate(msgs, 1):
        kinds = [it[0] for it in m["seq"]]
        assert kinds == ["u", "u", "u", "u", "u", "u", "d"]
        mesh_id, i, count, total, crc, size, data = [it[1] for it in m["seq"]]
        assert (mesh_id, i, count, total, crc) == (1, k, n, len(blob), zlib.crc32(blob))
        d = bytes.fromhex(data)
        assert len(d) == size <= 16 * 1024 and m["bytes"] < 64 * 1024
        parts[i] = d
    assert b"".join(parts[i] for i in sorted(parts)) == blob


def test_chunks_go_out_one_per_tick_not_all_in_one_frame(game):
    raw, blob = big_payload()
    summon(game)
    game.register(blob, raw)
    obj = {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "svm1-lzma", "bytes": len(blob), "crc32": zlib.crc32(blob), "b64": base64.b64encode(blob).decode()}
    game.serve_mesh(lambda b: (200, obj))
    assert len(game.nets()) == 0, "nothing in the same frame as the answer"
    game.w.advance(0.15)
    assert len(game.nets()) == 1
    game.w.advance(0.1)
    assert len(game.nets()) == 2


def test_an_object_summoned_after_the_mesh_is_ready_gets_it_at_once_without_asking_again(game):
    p = summon(game)
    game.serve_mesh()
    game.w.advance(2)
    game.forget_nets()
    game.summon(p, "add another garden gnome")
    assert len(game.mesh_posts()) == 1, "the model is known"
    assert game.visuals() == [{"visual": "mesh", "mesh": 1}] * 2
    assert game.nets() == [], "and nothing is sent twice: the clients already have it"


def test_a_client_that_joins_later_asks_and_is_sent_the_mesh_alone(game):
    summon(game)
    game.serve_mesh()
    game.w.advance(2)
    game.forget_nets()
    late = game.player("76561190000000009")
    game.w.rt.eval("function(p) MOCK.deliver_seq('holodeck_mesh_want', { { 'u', 0, 8 } }, p) end")(late)
    game.w.advance(1)
    msgs = game.nets()
    assert len(msgs) == 1 and msgs[0]["to"] == ["76561190000000009"], "only the one who asked"
    game.forget_nets()
    game.w.rt.eval("function(p) MOCK.deliver_seq('holodeck_mesh_want', { { 'u', 7, 8 } }, p) end")(late)         # a number the server does not have
    game.w.advance(1)
    assert game.nets() == []


def test_asking_over_and_over_is_rationed_per_player(game):
    summon(game)
    game.serve_mesh()
    game.w.advance(2)
    p = game.player("76561190000000009")
    for _ in range(40):
        game.w.rt.eval("function(p) MOCK.deliver_seq('holodeck_mesh_want', { { 'u', 1, 8 } }, p) end")(p)
        game.w.advance(0.25)
    sent_to_p = [m for m in game.nets() if m["to"] == ["76561190000000009"]]
    assert 0 < len(sent_to_p) <= 12, f"{len(sent_to_p)} sends for 40 asks in 10 s"


@pytest.mark.parametrize("answer", [
    lambda b: (502, {"ok": False, "error": "boom"}),                                             # the bridge failed
    lambda b: (200, {"ok": False}),                                                              # said no
    lambda b: (200, {"ok": True, "key": "0" * 16, "format": "svm1-lzma", "b64": "AAAA"}),         # someone else's key
    lambda b: (200, {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "other", "b64": "AAAA"}),
    lambda b: (200, {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "svm1-lzma", "b64": "not base64 !!"}),
    lambda b: (200, {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "svm1-lzma", "b64": "A" * (400 * 1024)}),        # over the cap
    lambda b: (200, {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "svm1-lzma", "b64": base64.b64encode(b"\x5d" + b"\0" * 40).decode()}),   # not a model
])
def test_a_failed_or_hostile_answer_gives_up_and_takes_the_placeholder_down(game, answer):
    summon(game)
    game.serve_mesh(answer)
    game.w.advance(3)
    assert game.visuals() == [], "the transporter gave up: the placeholder is gone (see test_gmod_transporter.py for the effect and the sentence)"
    assert game.nets() == [], "nothing is relayed to anyone"
    assert game.w.call("T.mesh_state") == "failed"


def _answer_with(blob, raw=None, **override):
    def answer(b):
        o = {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "svm1-lzma", "bytes": len(blob), "crc32": zlib.crc32(blob),
             "b64": base64.b64encode(blob).decode()}
        o.update(override)
        return 200, o
    return answer


def test_a_checksum_or_length_that_does_not_match_is_refused(game):
    raw, blob = big_payload(200)
    summon(game)
    game.register(blob, raw)
    game.serve_mesh(_answer_with(blob, crc32=zlib.crc32(blob) ^ 1))
    game.w.advance(3)
    assert game.visuals() == [] and game.nets() == []


def test_the_length_and_the_declared_inflated_size_are_checked(game):
    raw, blob = big_payload(200)
    summon(game)
    game.register(blob, raw)
    game.serve_mesh(_answer_with(blob, bytes=len(blob) + 1))
    game.w.advance(3)
    assert game.visuals() == []
    # a bomb: the header claims 3 GB; the engine's inflate is never asked
    bomb = blob[:5] + struct.pack("<Q", 3 * 1024 ** 3) + blob[13:]
    g2 = MeshGame()
    summon(g2)
    g2.register(bomb, raw)
    g2.serve_mesh(_answer_with(bomb))
    g2.w.advance(3)
    assert g2.visuals() == [] and g2.nets() == []
    # unknown size (-1): the real engine returns nothing for it, so it is refused up front
    unknown = blob[:5] + b"\xff" * 8 + blob[13:]
    g3 = MeshGame()
    summon(g3)
    g3.serve_mesh(_answer_with(unknown))
    g3.w.advance(3)
    assert g3.visuals() == [] and g3.nets() == []


def test_a_payload_over_256_kb_is_refused_on_the_game_side_too(game):
    """A well-formed model padded past the cap: every other check passes, so only the cap can be what refuses it."""
    raw, blob = big_payload(200)
    padded = blob + bytes(256 * 1024 + 1 - len(blob))
    summon(game)
    game.register(padded, raw)
    game.serve_mesh(_answer_with(padded))
    game.w.advance(3)
    assert game.visuals() == [] and game.nets() == []
    ok = MeshGame()
    summon(ok)
    ok.register(blob, raw)
    ok.serve_mesh(_answer_with(blob))
    ok.w.advance(3)
    assert ok.visuals() == [{"visual": "mesh", "mesh": 1}], "the same model without the padding is fine"


def test_a_failed_model_is_asked_for_again_later_but_not_forever(game):
    p = summon(game)
    game.serve_mesh(lambda b: (502, {"ok": False}))
    for k in range(5):
        game.w.advance(61)
        game.summon(p, "add a garden gnome")
        game.serve_mesh(lambda b: (502, {"ok": False}))
    assert len(game.mesh_posts()) == 3, "three tries, then the box is all there is"
    # each of the three tries ended with its placeholder taken down; the objects asked for after that were never waiting for anything
    # (no request goes out), so they are plain boxes, not beams
    assert len(game.visuals()) == 3 and all(v == {"visual": "box", "mesh": 0} for v in game.visuals())


def test_with_holodeck_mesh_off_nothing_is_asked_and_resume_asks(game):
    game.w.rt.eval("function() GetConVar('holodeck_mesh'):SetValue(0) end")()
    p = summon(game)
    assert game.mesh_posts() == [] and game.visuals() == [{"visual": "box", "mesh": 0}]
    game.w.rt.eval("function() GetConVar('holodeck_mesh'):SetValue(1); return Holodeck.Mesh.resume() end")()
    assert len(game.mesh_posts()) == 1
    game.serve_mesh()
    game.w.advance(2)
    assert game.visuals() == [{"visual": "mesh", "mesh": 1}]


def test_a_card_without_a_mesh_never_asks(ix):
    g = MeshGame()
    p = g.player(ALICE, admin=True)
    g.summon(p, "add a fuel widget")
    assert len(g.found()) == 1 and g.mesh_posts() == []
    assert g.visuals() == [{"visual": "box", "mesh": 0}]


def test_the_physics_stays_the_cards_box_when_the_mesh_arrives(game):
    summon(game)
    before = game.found()[0]
    game.serve_mesh()
    game.w.advance(2)
    after = game.found()[0]
    for k in ("mins", "maxs", "mass", "x", "y", "z", "yaw"):
        assert after[k] == before[k], k
    assert after["visual"] == "mesh"


# ============================================================================================================ the client half
CLIENT_MESH_FAKES = r"""
RENDER2 = { meshes = {}, draws = 0, pushed = 0, blends = {}, mats = 0, wires = 0 }
MATERIAL_TRIANGLES = 3
function Matrix() local m = {}; function m:Translate(v) self.t = v end; function m:Rotate(a) self.a = a end; return m end
cam.PushModelMatrix = function(m) RENDER2.pushed = RENDER2.pushed + 1; RENDER2.matrix = m end
cam.PopModelMatrix = function() end
render.SetBlend = function(a) RENDER2.blends[#RENDER2.blends + 1] = a end
render.SetMaterial = function() RENDER2.mats = RENDER2.mats + 1 end
function CreateMaterial() return {} end
function Mesh()
  local im = { verts = {} }
  function im:Draw() RENDER2.draws = RENDER2.draws + 1 end
  RENDER2.meshes[#RENDER2.meshes + 1] = im
  return im
end
local building
mesh = {
  Begin = function(im, prim, n) building = { im = im, verts = {}, cur = {}, n = n } end,
  Position = function(v) building.cur = { pos = v } end,
  Normal = function(v) building.cur.normal = v end,
  Color = function(r, g, b, a) building.cur.col = { r, g, b, a } end,
  TexCoord = function() end,
  AdvanceVertex = function() building.verts[#building.verts + 1] = building.cur; building.cur = {} end,
  End = function() building.im.verts = building.verts end,
}
function T.tris()
  local out = {}
  local im = RENDER2.meshes[#RENDER2.meshes]
  for i = 1, #im.verts, 3 do
    local t = {}
    for k = 0, 2 do
      local v = im.verts[i + k]
      t[#t + 1] = { v.pos.x, v.pos.y, v.pos.z, v.normal.x, v.normal.y, v.normal.z, v.col[1], v.col[2], v.col[3] }
    end
    out[#out + 1] = t
  end
  return MOCK.json_encode(out)
end
function T.store(id) local s = Holodeck.MeshClient.store[id]; return s and s.state or "none" end
"""


def mesh_client():
    w = World(cvars={}, load=(), server=False)
    w.rt.execute(CLIENT_FAKES)
    w.rt.execute(CLIENT_MESH_FAKES)
    w.rt.execute('MOCK.load_entity("holodeck_found", "client")')          # its cl_init loads meshclient.lua (Holodeck.MeshClient)
    return w


def feed(w, msgs):
    """Carry exported messages (from game.nets()) to the client runtime's receiver, as the engine would."""
    for m in msgs:
        w.rt.eval("function(name, seq) MOCK.deliver_seq(name, seq) end")("holodeck_mesh", lua_table(w.rt, m["seq"]))


def make_msgs(blob, mesh_id=1, chunk=16 * 1024):
    n = -(-len(blob) // chunk)
    out = []
    for i in range(1, n + 1):
        piece = blob[(i - 1) * chunk:i * chunk]
        out.append({"seq": [["u", mesh_id, 8], ["u", i, 8], ["u", n, 8], ["u", len(blob), 20], ["u", zlib.crc32(blob), 32], ["u", len(piece), 16], ["d", piece.hex()]]})
    return out


def client_with_payload(raw, blob, **kw):
    w = mesh_client()
    w.rt.eval("function(a, b) MOCK.lzma[MOCK.fromhex(a)] = MOCK.fromhex(b) end")(blob.hex(), raw.hex())
    return w


def box_blob(tmp_path_factory=None):
    routes = box_routes("garden_gnome")
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "garden_gnome_1k.gltf").write_bytes(routes["https://dl.example/garden_gnome_1k.gltf"])
    (d / "garden_gnome.bin").write_bytes(routes["https://dl.example/garden_gnome.bin"])
    blob, _ = meshpack.pack(d / "garden_gnome_1k.gltf", 100, "lzma")
    return meshpack.unlzma_alone(blob), blob


def test_the_lua_decoder_matches_the_python_unpacker_on_a_real_chair():
    gltf = ROOT / "stage" / "skippyverse" / "cache" / "assets" / "found.poly-haven.dining-chair-02" / "dining_chair_02_1k.gltf"
    if not gltf.is_file():
        pytest.skip("the Poly Haven dining chair is not in the asset cache (stage/skippyverse/cache/assets, gitignored)")
    blob, _ = meshpack.pack(gltf, 1500, "lzma")
    raw = meshpack.unlzma_alone(blob)
    pv, pf, pc = meshpack.unpack(blob)
    w = mesh_client()
    dec = w.rt.eval("function(h) local S = dofile(MOCK.ROOT .. '/mentat/holodeck/svm1.lua'); return S.decode(MOCK.fromhex(h)) end")(raw.hex())
    assert dec.nv == len(pv) and dec.nt == len(pf)
    for i, v in enumerate(pv, 1):
        assert (dec.x[i], dec.y[i], dec.z[i]) == pytest.approx(v, abs=1e-12)
    for i, f in enumerate(pf, 1):
        assert (dec.a[i] - 1, dec.b[i] - 1, dec.c[i] - 1) == tuple(f)
        assert (dec.r[i], dec.g[i], dec.bl[i]) == tuple(pc[i - 1])


@pytest.mark.parametrize("mutate,why", [
    (lambda r: r[:20], "too short"),
    (lambda r: b"XXXX" + r[4:], "not an SVM1"),
    (lambda r: r + b"\0", "does not match"),
    (lambda r: r[:-1], "does not match"),
    (lambda r: r[:4] + struct.pack("<HH", 8, 0) + r[8:], "triangle count"),
    (lambda r: r[:4] + struct.pack("<HH", 2, 12) + r[8:], "vertex count"),
    (lambda r: r[:8] + struct.pack("<f", float("nan")) + r[12:], "bad bounds"),
    (lambda r: r[:8] + struct.pack("<f", float("inf")) + r[12:], "bad bounds"),
    (lambda r: r[:8] + struct.pack("<f", 5000.0) + r[12:], "bad bounds"),
])
def test_the_decoder_refuses_a_malformed_payload(mutate, why):
    raw, _ = box_blob()
    w = mesh_client()
    got = w.rt.eval("function(h) local S = dofile(MOCK.ROOT .. '/mentat/holodeck/svm1.lua'); local d, e = S.decode(MOCK.fromhex(h)); return d == nil, e end")(mutate(raw).hex())
    assert got[0] is True and why in got[1]


def test_the_decoder_refuses_a_triangle_that_names_a_vertex_that_does_not_exist_or_repeats_one():
    raw, _ = box_blob()
    nv = struct.unpack_from("<H", raw, 4)[0]
    off = 32 + 6 * nv
    for fix in (struct.pack("<3H", 0, 1, nv), struct.pack("<3H", 2, 2, 3)):
        bad = raw[:off] + fix + raw[off + 6:]
        w = mesh_client()
        ok, why = w.rt.eval("function(h) local S = dofile(MOCK.ROOT .. '/mentat/holodeck/svm1.lua'); local d, e = S.decode(MOCK.fromhex(h)); return d == nil, e end")(bad.hex())
        assert ok is True and ("does not exist" in why or "degenerate" in why)


def client_entity(w, mesh_id=1):
    me = w.rt.eval("function() MOCK.me = MOCK.newplayer('1', { pos = Vector(100, 0, 0) }); return MOCK.me end")()
    e = w.rt.eval("""function(id)
      local e = ents.Create("holodeck_found")
      e:SetFound({ mins = {-%f, -%f, 0}, maxs = {%f, %f, %f}, name = "garden gnome", source = "Poly Haven", licence = "CC0", basis = "sourced", mass = 5 })
      e:SetPos(Vector(100, 200, 0)); e:SetAngles(Angle(0, 30, 0))
      e.GetColor = function() return Color(255, 255, 255, 255) end
      e.LocalToWorld = function(s, v) return s.pos + v end
      e.SetRenderBounds = function() end
      e:SetVisual("mesh", { id = id })
      return e
    end""" % (BOX_D / 2 / M_PER_HU, BOX_W / 2 / M_PER_HU, BOX_D / 2 / M_PER_HU, BOX_W / 2 / M_PER_HU, BOX_H / M_PER_HU))(mesh_id)
    return e


def draws(w):
    r = w.rt.eval("RENDER2")
    return r.draws, r.pushed, len(w.rt.eval("RENDER").boxes), len(w.rt.eval("RENDER").wires)


def test_the_client_reassembles_decodes_and_builds_the_mesh_at_the_right_size_pose_and_orientation():
    raw, blob = box_blob()
    w = client_with_payload(raw, blob)
    e = client_entity(w)
    assert draws(w) == (0, 0, 0, 0)
    feed(w, make_msgs(blob))
    tris = json.loads(w.call("T.tris"))
    assert len(tris) == 12
    xs = [v[0] for t in tris for v in t]
    ys = [v[1] for t in tris for v in t]
    zs = [v[2] for t in tris for v in t]
    # the payload is 0.6 wide (x), 0.3 deep (y), 1.2 high; the entity's box is deep along x and wide along y, so the mesh is turned a quarter circle
    assert (max(xs) - min(xs)) * M_PER_HU == pytest.approx(BOX_D, abs=2e-3)
    assert (max(ys) - min(ys)) * M_PER_HU == pytest.approx(BOX_W, abs=2e-3)
    assert (max(zs) - min(zs)) * M_PER_HU == pytest.approx(BOX_H, abs=2e-3)
    assert min(zs) == pytest.approx(0, abs=1e-3), "the floor under the footprint"
    assert (max(xs) + min(xs)) / 2 == pytest.approx(0, abs=1e-3) and (max(ys) + min(ys)) / 2 == pytest.approx(0, abs=1e-3), "centred on the footprint"
    # every face points OUT of the solid: the normal agrees with the side of the centre the face is on
    cx, cy, cz = 0.0, 0.0, (max(zs) + min(zs)) / 2
    for t in tris:
        mx, my, mz = (sum(v[k] for v in t) / 3 for k in range(3))
        nx, ny, nz = t[0][3:6]
        assert (mx - cx) * nx + (my - cy) * ny + (mz - cz) * nz > 0, "an inside-out face"
        # and so does the WINDING the engine sees (clockwise from outside): the cross product of the engine's order points inward
        a, b, c = t
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        cr = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        assert cr[0] * nx + cr[1] * ny + cr[2] * nz < 0, "the engine's front face is clockwise from outside: the order must be flipped"
    assert all(v[6] > 0 or v[7] > 0 or v[8] > 0 for t in tris for v in t), "coloured (the material's base colour, shaded)"
    assert max(v[6] for t in tris for v in t) > max(v[8] for t in tris for v in t), "red stays the red of the model"
    assert w.call("T.store", 1) == "ready"


def test_a_ready_mesh_is_drawn_in_place_of_the_box_at_the_entitys_pose():
    raw, blob = box_blob()
    w = client_with_payload(raw, blob)
    e = client_entity(w)
    e.Draw(e)
    assert draws(w)[0] == 0 and draws(w)[2] == 1, "before it arrives: the box"
    feed(w, make_msgs(blob))
    w.rt.execute("RENDER.boxes = {}; RENDER.wires = {}")
    e.Draw(e)
    d, pushed, boxes, wires = draws(w)
    assert (d, pushed, boxes, wires) == (1, 1, 0, 0), "the mesh, with the entity's matrix, and neither the translucent box nor its edges"
    m = w.rt.eval("RENDER2.matrix")
    assert (m.t.x, m.t.y, m.t.z) == (100, 200, 0) and m.a.y == 30


def test_the_outline_convar_keeps_a_faint_box_around_the_mesh():
    raw, blob = box_blob()
    w = client_with_payload(raw, blob)
    e = client_entity(w)
    feed(w, make_msgs(blob))
    w.rt.execute("GetConVar('holodeck_found_outline'):SetValue(1); RENDER.boxes = {}; RENDER.wires = {}")
    e.Draw(e)
    r = w.rt.eval("RENDER")
    assert draws(w)[0] == 1 and len(r.boxes) == 0 and len(r.wires) == 1 and r.wires[1].col.a < 100, "faint, edges only"


def test_the_materialising_fade_applies_to_the_mesh_too():
    raw, blob = box_blob()
    w = client_with_payload(raw, blob)
    e = client_entity(w)
    feed(w, make_msgs(blob))
    e.GetColor = w.rt.eval("function() return function() return Color(255, 255, 255, 51) end end")()
    e.Draw(e)
    assert list(w.rt.eval("RENDER2.blends").values())[0] == pytest.approx(0.2) and list(w.rt.eval("RENDER2.blends").values())[-1] == 1


def test_chunks_may_arrive_out_of_order_and_twice():
    raw, blob = big_payload(1500)
    w = client_with_payload(raw, blob)
    client_entity(w)
    msgs = make_msgs(blob)
    assert len(msgs) >= 3
    feed(w, [msgs[1], msgs[1], msgs[-1]])
    assert w.call("T.store", 1) == "none", "not yet"
    feed(w, [msgs[0]] + msgs[2:-1])
    assert w.call("T.store", 1) == "ready"
    assert len(json.loads(w.call("T.tris"))) == 1500
    built = len(list(w.rt.eval("RENDER2.meshes").values()))
    feed(w, msgs)                                                       # the server answers a second ask: nothing is built again
    assert len(list(w.rt.eval("RENDER2.meshes").values())) == built


def test_a_missing_chunk_never_builds_anything_and_the_box_stays():
    raw, blob = big_payload(1500)
    w = client_with_payload(raw, blob)
    e = client_entity(w)
    msgs = make_msgs(blob)
    feed(w, msgs[:1] + msgs[2:])
    assert w.call("T.store", 1) == "none"
    e.Draw(e)
    assert draws(w)[0] == 0 and draws(w)[2] == 1


@pytest.mark.parametrize("what", ["crc", "flip_byte", "total", "short_chunk", "bad_id", "zero_count", "index_past_count", "over_cap", "oversize_chunk"])
def test_a_bad_message_is_dropped_and_the_box_stays(what):
    raw, blob = box_blob()
    w = client_with_payload(raw, blob)
    e = client_entity(w)
    (m,) = make_msgs(blob)
    s = m["seq"]
    if what == "crc":
        s[4][1] = s[4][1] ^ 1
    elif what == "flip_byte":
        d = bytearray(bytes.fromhex(s[6][1])); d[20] ^= 0xFF; s[6][1] = bytes(d).hex()
    elif what == "total":
        s[3][1] = s[3][1] + 5
    elif what == "short_chunk":
        s[5][1] = s[5][1] + 3
    elif what == "bad_id":
        s[0][1] = 200
    elif what == "zero_count":
        s[2][1] = 0
    elif what == "index_past_count":
        s[1][1] = 5
    elif what == "over_cap":
        s[3][1] = 300 * 1024
    elif what == "oversize_chunk":
        s[5][1] = 20000
    try:
        feed(w, [m])
    except Exception:
        pass                                                                                   # the mock's reader may object to a lie; the client must too
    assert w.call("T.store", 1) in ("none", "failed")
    e.Draw(e)
    assert draws(w)[0] == 0 and draws(w)[2] == 1, "still the box"


def test_a_payload_that_decodes_to_nonsense_fails_to_the_box():
    raw, blob = box_blob()
    bad_raw = raw[:4] + struct.pack("<HH", 8, 12) + raw[8:-3]                               # counts that do not match the length
    bad_blob = meshpack.lzma_alone(bad_raw)
    w = client_with_payload(bad_raw, bad_blob)
    e = client_entity(w)
    feed(w, make_msgs(bad_blob))
    assert w.call("T.store", 1) == "failed"
    e.Draw(e)
    assert draws(w)[0] == 0 and draws(w)[2] == 1


def test_an_engine_that_cannot_inflate_it_fails_to_the_box():
    raw, blob = box_blob()
    w = mesh_client()                                                                       # util.Decompress knows nothing: it returns nil
    e = client_entity(w)
    feed(w, make_msgs(blob))
    assert w.call("T.store", 1) == "failed"
    e.Draw(e)
    assert draws(w)[0] == 0 and draws(w)[2] == 1


def test_a_client_that_lacks_the_mesh_asks_the_server_once_in_a_while_and_a_new_client_asks_for_everything():
    raw, blob = box_blob()
    w = mesh_client()
    e = client_entity(w)
    for _ in range(5):
        e.Draw(e)
    asked = w.rt.eval("function() local n = 0; for _, m in ipairs(MOCK.nets) do if m.name == 'holodeck_mesh_want' then n = n + 1 end end return n end")()
    assert asked == 1, "once, not every frame"
    w.rt.execute("MOCK.hooks.InitPostEntity.holodeck_mesh_hello()")
    wants = [m for m in w.rt.eval("MOCK.nets").values() if m.name == "holodeck_mesh_want"]
    assert [m.writes[1] for m in wants] == [1, 0], "the entity's own mesh, then everything on joining"


def test_a_server_to_client_round_trip_ends_with_the_mesh_on_screen(game):
    """The server runtime's own messages, carried to a client runtime: from the bridge's answer to a mesh draw call."""
    summon(game)
    game.serve_mesh()
    game.w.advance(2)
    obj = skippy_bridge.mesh_payload(TALL, skippy_bridge.mesh_key(TALL))
    blob = base64.b64decode(obj["b64"])
    w = client_with_payload(meshpack.unlzma_alone(blob), blob)
    e = client_entity(w, 1)
    feed(w, game.nets())
    assert w.call("T.store", 1) == "ready"
    e.Draw(e)
    assert draws(w)[0] == 1


# ============================================================================================================ the look pass: SVM3 (colour regions, normals, material cues), lit drawing
LIT_FAKES = r"""
local V = getmetatable(Vector(0, 0, 0)).__index
function V:GetNormalized() local v = Vector(self.x, self.y, self.z); v:Normalize(); return v end
function V:Angle() return Angle(0, 0, 0) end
function EyePos() return Vector(0, 0, 0) end
MATERIAL_LIGHT_DIRECTIONAL = 2
function Matrix() local m = {}; function m:Translate(v) self.t = v end; function m:Rotate(a) self.a = a end; function m:Scale(v) self.s = v end; return m end
LIGHT = { suppress_on = 0, suppress_off = 0, ambient = 0, lights = 0, nlights = 0, resets = 0, mats = {}, order = {}, depth = 0 }
render.SuppressEngineLighting = function(b) if b then LIGHT.suppress_on = LIGHT.suppress_on + 1 else LIGHT.suppress_off = LIGHT.suppress_off + 1 end end
render.SetModelLighting = function() LIGHT.ambient = LIGHT.ambient + 1 end
render.SetLocalModelLights = function(t) if t then LIGHT.lights = LIGHT.lights + 1; LIGHT.nlights = #t else LIGHT.resets = LIGHT.resets + 1 end end
render.ComputeLighting = function() return Vector(0.5, 0.5, 0.5) end
render.OverrideDepthEnable = function() LIGHT.depth = LIGHT.depth + 1 end
function CreateMaterial(name, shader, p)
  local m = { name = name, shader = shader, p = p }
  function m:SetFloat() end
  LIGHT.mats[#LIGHT.mats + 1] = m
  return m
end
render.SetMaterial = function(m) RENDER2.mats = RENDER2.mats + 1; LIGHT.order[#LIGHT.order + 1] = m and m.shader or "?" end
util.TraceLine = function(t) return { Hit = true, HitPos = Vector(t.start.x, t.start.y, 0), HitNormal = Vector(0, 0, 1) } end
MASK_SOLID_BRUSHONLY = 1
"""


def mesh_client_lit(lit=True):
    w = World(cvars={}, load=(), server=False)
    w.rt.execute(CLIENT_FAKES)
    w.rt.execute(CLIENT_MESH_FAKES)
    if lit:
        w.rt.execute(LIT_FAKES)
    w.rt.execute('MOCK.load_entity("holodeck_found", "client")')
    return w


def client_with_payload_lit(raw, blob, lit=True):
    w = mesh_client_lit(lit)
    w.rt.eval("function(a, b) MOCK.lzma[MOCK.fromhex(a)] = MOCK.fromhex(b) end")(blob.hex(), raw.hex())
    return w


def svm3_blob():
    """A box with a red-metal half and a blue-matte half, as an SVM3 payload: (raw, lzma blob)."""
    import io
    import tempfile
    from PIL import Image
    from test_skippyverse_surface import BLUE, RED, png, textured_gltf
    d = Path(tempfile.mkdtemp())
    p = textured_gltf(d, png([RED, BLUE]))
    mr = Image.new("RGB", (8, 8), (0, 255, 0))                         # matte: rough 1, metal 0
    for x in range(4):
        for y in range(8):
            mr.putpixel((x, y), (0, 40, 255))                          # the left half: polished metal
    b = io.BytesIO()
    mr.save(b, "PNG")
    doc = json.loads(p.read_text())
    doc["images"].append({"uri": "textures/mr.png"})
    doc["textures"].append({"source": 1})
    doc["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 1}
    (d / "textures" / "mr.png").write_bytes(b.getvalue())
    p.write_text(json.dumps(doc))
    blob, _ = meshpack.pack(p, 100, "lzma", fmt="svm3")
    return meshpack.unlzma_alone(blob), blob


DECODE3 = "function(h) local S = dofile(MOCK.ROOT .. '/mentat/holodeck/svm1.lua'); return S.decode3(MOCK.fromhex(h)) end"
DECODE3_ERR = "function(h) local S = dofile(MOCK.ROOT .. '/mentat/holodeck/svm1.lua'); local d, e = S.decode3(MOCK.fromhex(h)); return d == nil, e end"


def test_the_lua_svm3_decoder_matches_the_python_unpacker():
    raw, blob = svm3_blob()
    d = meshpack.unpack3(blob)
    w = mesh_client()
    dec = w.rt.eval(DECODE3)(raw.hex())
    assert dec.svm3 is True and dec.nv == len(d["verts"]) and dec.nt == len(d["faces"])
    for i, v in enumerate(d["verts"], 1):
        assert (dec.x[i], dec.y[i], dec.z[i]) == pytest.approx(v, abs=1e-12)
    for i, n in enumerate(d["normals"], 1):
        assert (dec.nx[i], dec.ny[i], dec.nz[i]) == pytest.approx(n, abs=1e-9)
    for i, e in enumerate(d["palette"], 1):
        assert [dec.pal[i][k] for k in range(1, 8)] == list(e)
    for i, k in enumerate(d["face_pal"], 1):
        assert dec.pi[i] == k + 1 and (dec.r[i], dec.g[i], dec.bl[i]) == tuple(d["palette"][k][:3]), "a flat-colour consumer still sees the palette colour"
    found = "function(h) local S = dofile(MOCK.ROOT .. '/mentat/holodeck/svm1.lua'); return S.decode_found(MOCK.fromhex(h)).svm3 end"
    assert w.rt.eval(found)(raw.hex()) is True
    raw1, _ = box_blob()
    assert w.rt.eval(found)(raw1.hex()) is None, "SVM1 still decodes, as before"


@pytest.mark.parametrize("mutate,why", [
    (lambda r: r[:20], "too short"),
    (lambda r: r + b"\0", "does not match"),
    (lambda r: r[:32] + b"\0" + r[33:], "palette size"),
    (lambda r: r[:4] + struct.pack("<HH", 2, 12) + r[8:], "vertex count"),
    (lambda r: r[:-1], "does not match"),
])
def test_a_malformed_svm3_is_refused(mutate, why):
    raw, _ = svm3_blob()
    w = mesh_client()
    ok, e = w.rt.eval(DECODE3_ERR)(mutate(raw).hex())
    assert ok is True and why in e


def test_an_svm3_palette_index_past_the_palette_is_refused():
    raw, _ = svm3_blob()
    npal = raw[32]
    nv, nt = struct.unpack_from("<HH", raw, 4)
    off = 33 + 7 * npal + 6 * nv + 6 * nt
    bad = raw[:off] + bytes([npal]) + raw[off + 1:]
    w = mesh_client()
    ok, e = w.rt.eval(DECODE3_ERR)(bad.hex())
    assert ok is True and "palette entry" in e


def client_entity_lit(w, mesh_id=1):
    e = client_entity(w, mesh_id)
    up = w.rt.eval("Vector(0, 0, 1)")
    e.GetUp = lambda s: up
    e.EntIndex = lambda s: 7
    return e


def test_the_unlit_fallback_builds_one_mesh_per_colour_region_with_the_shade_baked_in():
    raw, blob = svm3_blob()
    w = client_with_payload_lit(raw, blob, lit=False)
    e = client_entity_lit(w)
    feed(w, make_msgs(blob))
    assert w.call("T.store", 1) == "ready"
    n_pal = len(meshpack.unpack3(blob)["palette"])
    assert len(list(w.rt.eval("Holodeck.MeshClient.store[1].parts").values())) == n_pal >= 2
    assert w.rt.eval("Holodeck.MeshClient.store[1].lit") is False
    e.Draw(e)
    assert w.rt.eval("RENDER2.draws") == n_pal, "one draw per region, the flat material, no engine lighting calls"


def test_a_found_object_is_drawn_lit_with_its_own_materials_and_a_contact_shadow():
    raw, blob = svm3_blob()
    w = client_with_payload_lit(raw, blob)
    e = client_entity_lit(w)
    feed(w, make_msgs(blob))
    assert w.call("T.store", 1) == "ready" and w.rt.eval("Holodeck.MeshClient.store[1].lit") is True
    n_pal = len(meshpack.unpack3(blob)["palette"])
    e.Draw(e)
    L = w.rt.eval("LIGHT")
    assert (L.suppress_on, L.suppress_off, L.lights, L.resets, L.nlights) == (1, 1, 1, 1, 3), "key + rim + fill, set once and put back once"
    assert L.ambient == 6, "the map's ambient on six sides"
    assert w.rt.eval("RENDER2.draws") == n_pal + 1, "every colour region once, and the shadow blob"
    mats = [m for m in list(L.mats.values()) if m.shader == "VertexLitGeneric"]
    assert len(mats) == n_pal and all(m.p["$phong"] == 1 and m.p["$basetexture"] == "color/white" for m in mats)
    boosts = sorted(m.p["$phongboost"] for m in mats)
    assert boosts[-1] > 2 * boosts[0], "the polished metal half glints far more than the matte half"
    assert any(m.p["$phongtint"] for m in mats if m.p["$phongboost"] == boosts[-1]), "a metal tints its highlight"
    col = w.rt.eval("RENDER2.meshes[1].verts[1].col")
    assert (col[1], col[2], col[3]) == (255, 255, 255), "lit meshes carry no colour: the material has it"


def test_the_materialise_fade_uses_translucent_variants_of_the_lit_materials():
    raw, blob = svm3_blob()
    w = client_with_payload_lit(raw, blob)
    e = client_entity_lit(w)
    feed(w, make_msgs(blob))
    e.Draw(e)
    solid = [m for m in list(w.rt.eval("LIGHT.mats").values()) if m.shader == "VertexLitGeneric"]
    assert solid and not any(m.p["$translucent"] for m in solid)
    e.GetColor = w.rt.eval("function() return function() return Color(255, 255, 255, 100) end end")()
    e.Draw(e)
    fade = [m for m in list(w.rt.eval("LIGHT.mats").values()) if m.shader == "VertexLitGeneric" and m.p["$translucent"]]
    assert len(fade) == len(solid) and list(w.rt.eval("RENDER2.blends").values())[-1] == 1


def test_the_lit_switch_draws_the_old_flat_way_when_off():
    raw, blob = svm3_blob()
    w = client_with_payload_lit(raw, blob)
    w.rt.execute("GetConVar('holodeck_found_lit'):SetValue(0)")
    e = client_entity_lit(w)
    feed(w, make_msgs(blob))
    assert w.rt.eval("Holodeck.MeshClient.store[1].lit") is False
    e.Draw(e)
    assert w.rt.eval("LIGHT.suppress_on") == 0


def test_an_svm1_object_from_an_older_bridge_still_draws_as_before():
    raw, blob = box_blob()
    w = client_with_payload_lit(raw, blob)
    e = client_entity_lit(w)
    feed(w, make_msgs(blob))
    e.Draw(e)
    assert w.rt.eval("RENDER2.draws") == 1 and w.rt.eval("LIGHT.suppress_on") == 0
