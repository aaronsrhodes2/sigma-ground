"""The two object caches (README "Object caches"): the packed-payload cache on the PC (stage/skippyverse/packcache.py, wired into the bridge's
POST /mesh) and the client cache between sessions (meshcache.lua + the offer / have handshake in meshnet.lua and meshclient.lua), plus the
prewarm command. Offline: no network, the real bridge functions against a fake fetch, the real Lua files against the mock engine in strict-file mode.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
import zlib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stage"))
sys.path.insert(0, str(ROOT / "stage" / "gmod" / "holodeck"))
sys.path.insert(0, str(HERE))

from skippyverse import meshpack, packcache  # noqa: E402

# ============================================================================================================ the PC cache, on its own


class Tick:
    """A clock that moves forward on every look, so least-recently-used order is exact and the tests never sleep."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def cache(tmp_path, cap=10_000_000, clock=None):
    return packcache.PackCache(tmp_path / "packed", cap_bytes=cap, clock=clock or Tick())


def put(c, name, payload=b"x" * 100, version="1", fp="fp1", facts=None):
    k = packcache.make_key(name, "svm3-lzma", 3000, version)
    assert c.put(k, payload, facts or {"tris": 12}, fmt="svm3-lzma", tris=3000, pack_version=version, card_id=name, fp=fp)
    return k


def test_the_key_is_stable_and_names_the_card_the_format_the_target_and_the_pack_version():
    a = packcache.make_key("found.x", "svm3-lzma", 3000, "1")
    assert a == packcache.make_key("found.x", "svm3-lzma", 3000, "1") and re.fullmatch(r"[0-9a-f]{32}", a)
    others = {packcache.make_key("found.y", "svm3-lzma", 3000, "1"), packcache.make_key("found.x", "svm1-lzma", 3000, "1"),
              packcache.make_key("found.x", "svm3-lzma", 1500, "1"), packcache.make_key("found.x", "svm3-lzma", 3000, "2")}
    assert a not in others and len(others) == 4


def test_a_new_pack_version_is_a_new_key_and_an_old_entry_is_never_served(tmp_path):
    c = cache(tmp_path)
    k = put(c, "found.x", version="1")
    assert c.get(k, None, "1") is not None
    k2 = packcache.make_key("found.x", "svm3-lzma", 3000, "2")
    assert k2 != k and c.get(k2, None, "2") is None, "the new version asks under another key"
    assert c.get(k, None, "2") is None and not c.path(k).exists(), "and the same key read with another version is a deleted stale entry, not a hit"
    assert c.stats["stale"] == 1


def test_the_bridges_key_follows_the_packers_version(monkeypatch):
    import skippy_bridge
    a = skippy_bridge.packcache_key("found.x")
    monkeypatch.setattr(meshpack, "PACK_VERSION", "999")
    assert skippy_bridge.packcache_key("found.x") != a


def test_a_hit_returns_the_identical_bytes_and_the_facts(tmp_path):
    c = cache(tmp_path)
    payload = os.urandom(5000)
    k = put(c, "found.x", payload, facts={"tris": 12, "bbox_m": [1, 2, 3]})
    e = c.get(k, "fp1", "1")
    assert e.payload == payload and e.facts == {"tris": 12, "bbox_m": [1, 2, 3]}
    assert c.get(k, "fp1", "1").payload == payload


def test_a_changed_source_fingerprint_drops_the_entry(tmp_path):
    c = cache(tmp_path)
    k = put(c, "found.x", fp="old")
    assert c.get(k, "new", "1") is None and not c.path(k).exists()
    k = put(c, "found.x", fp="old")
    assert c.get(k, None, "1") is not None, "no fingerprint to compare (the download was pruned): trusted"


def test_a_write_is_atomic_and_leaves_no_temp_file(tmp_path, monkeypatch):
    c = cache(tmp_path)
    k = put(c, "found.x")
    assert [p.name for p in c.dir.iterdir()] == [k + ".svmc"]
    real = os.replace

    def boom(src, dst):
        raise OSError("disk went away")
    monkeypatch.setattr(os, "replace", boom)
    k2 = packcache.make_key("found.y", "svm3-lzma", 3000, "1")
    with pytest.raises(OSError):
        c.put(k2, b"y" * 50, {"tris": 1}, fmt="svm3-lzma", tris=3000, pack_version="1")
    monkeypatch.setattr(os, "replace", real)
    assert not c.path(k2).exists(), "a failed write never leaves a half entry under the real name"
    assert c.get(k, None, "1") is not None, "and the old entries are untouched"
    # the temp file the crash left is swept by a later write once it is old
    leftovers = list(c.dir.glob(".tmp-*"))
    assert len(leftovers) == 1
    os.utime(leftovers[0], (1, 1))
    put(c, "found.z")
    assert not list(c.dir.glob(".tmp-*"))


@pytest.mark.parametrize("damage", ["truncate", "flip", "garbage", "empty", "wrong_key"])
def test_a_damaged_entry_is_dropped_and_rebuilt_never_served(tmp_path, damage):
    c = cache(tmp_path)
    k = put(c, "found.x", os.urandom(2000))
    p = c.path(k)
    raw = p.read_bytes()
    if damage == "truncate":
        p.write_bytes(raw[:-10])
    elif damage == "flip":
        p.write_bytes(raw[:-5] + bytes([raw[-5] ^ 0xFF]) + raw[-4:])
    elif damage == "garbage":
        p.write_bytes(b"not an entry at all")
    elif damage == "empty":
        p.write_bytes(b"")
    else:
        other = packcache.make_key("found.other", "svm3-lzma", 3000, "1")
        p.rename(c.path(other))
        k = other
    assert c.get(k, None, "1") is None
    assert not c.path(k).exists() and c.stats["corrupt"] == 1
    built = []
    e, hit = c.get_or_build(k, lambda: (built.append(1) or b"fresh payload", {"tris": 1}), fmt="svm3-lzma", tris=3000, pack_version="1", card_id="found.x")
    assert not hit and e.payload == b"fresh payload" and len(built) == 1
    assert c.get(k, None, "1").payload == b"fresh payload", "the rebuilt entry is good"


def test_the_least_recently_used_entry_goes_first_when_the_cap_is_reached(tmp_path):
    c = cache(tmp_path, cap=3 * 1200)
    a = put(c, "found.a", b"a" * 1000)
    b = put(c, "found.b", b"b" * 1000)
    assert c.get(a, None, "1") is not None, "a is used again: b is now the oldest"
    d = put(c, "found.d", b"d" * 1000)                      # 3 entries of about 1.3 KB do not fit in 3.6 KB together with a 4th: someone goes
    names = {p.stem for p in c.dir.glob("*.svmc")}
    assert d in names, "the entry just written is never the one evicted"
    assert a in names and b not in names or (c.total_bytes() <= c.cap and b not in names), names
    put(c, "found.e", b"e" * 1000)
    assert c.total_bytes() <= c.cap
    assert a not in {p.stem for p in c.dir.glob("*.svmc")} or d in {p.stem for p in c.dir.glob("*.svmc")}
    assert c.stats["evicted"] >= 2


def test_an_entry_bigger_than_the_cap_is_not_stored(tmp_path):
    c = cache(tmp_path, cap=1000)
    assert c.put("ab" * 16, b"x" * 5000, {}, fmt="f", tris=1, pack_version="1") is False and not list(c.dir.glob("*.svmc"))


def test_two_simultaneous_first_requests_build_once(tmp_path):
    c = cache(tmp_path, clock=time.time)
    k = packcache.make_key("found.x", "svm3-lzma", 3000, "1")
    built, got = [], []
    start = threading.Barrier(8)

    def build():
        built.append(1)
        time.sleep(0.15)
        return b"the one payload", {"tris": 7}

    def worker():
        start.wait()
        got.append(c.get_or_build(k, build, fmt="svm3-lzma", tris=3000, pack_version="1", card_id="found.x"))
    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(built) == 1, "packed once"
    assert all(e.payload == b"the one payload" for e, _ in got) and sum(1 for _, hit in got if not hit) == 1
    assert c.stats["builds"] == 1


def test_the_fingerprint_follows_the_files_content(tmp_path):
    d = tmp_path / "asset"
    d.mkdir()
    (d / "a.gltf").write_bytes(b"one")
    (d / "acquired.json").write_text("{}")
    f1 = packcache.fingerprint(d)
    (d / "acquired.json").write_text('{"x": 1}')
    assert packcache.fingerprint(d) == f1, "the marker is not an input of the packer"
    (d / "a.gltf").write_bytes(b"two")
    assert packcache.fingerprint(d) != f1 and packcache.fingerprint(tmp_path / "nope") is None
    assert packcache.fingerprint(d, b"x") != packcache.fingerprint(d)


# ============================================================================================================ wired into POST /mesh
import skippy_bridge  # noqa: E402
from test_gmod_found_mesh import TALL, ix, fetch  # noqa: E402,F401  (fixtures)


@pytest.fixture()
def spy(monkeypatch):
    """Count the real packer's calls and record every stage the beam is told about."""
    calls = {"pack": 0, "stages": []}
    real = meshpack.pack

    def counting(*a, **k):
        calls["pack"] += 1
        time.sleep(0.05)
        return real(*a, **k)
    monkeypatch.setattr(meshpack, "pack", counting)
    real_set = skippy_bridge.MeshProgress.set

    def rec(self, stage, error=None):
        calls["stages"].append(stage)
        return real_set(self, stage, error)
    monkeypatch.setattr(skippy_bridge.MeshProgress, "set", rec)
    return calls


def forget_memory(monkeypatch):
    monkeypatch.setattr(skippy_bridge, "MESH_PACKED", {})
    skippy_bridge.MESH_PROGRESS.clear()


def test_the_second_request_for_a_model_is_a_file_read_no_download_no_measure_no_pack(ix, fetch, spy, monkeypatch):
    key = skippy_bridge.mesh_key(TALL)
    a = skippy_bridge.mesh_payload(TALL, key)
    assert spy["pack"] == 1 and len(fetch.calls) == 3 and "cached" not in a
    assert spy["stages"] == ["fetching", "packing", "done"], "a cold request reports what it did"
    spy["stages"].clear()
    forget_memory(monkeypatch)                                    # a new bridge process / another session: only the disk is left
    fetch.calls.clear()
    fetch.routes.clear()                                          # and the network would fail the test if it were touched
    b = skippy_bridge.mesh_payload(TALL, key)
    assert spy["pack"] == 1 and fetch.calls == [], "no download and no pack"
    assert b["b64"] == a["b64"] and b["crc32"] == a["crc32"] and b["bytes"] == a["bytes"] and (b["tris"], b["verts"], b["bbox_m"]) == (a["tris"], a["verts"], a["bbox_m"])
    assert b["cached"] is True


def test_a_hit_still_tells_the_beam_the_truth_it_was_cached(ix, fetch, spy, monkeypatch):
    key = skippy_bridge.mesh_key(TALL)
    skippy_bridge.mesh_payload(TALL, key)
    spy["stages"].clear()
    forget_memory(monkeypatch)
    skippy_bridge.mesh_payload(TALL, key)
    assert spy["stages"] == ["cached", "done"], "the stage string says cached, then done: never a pretend download"
    snap = skippy_bridge.mesh_progress(key)
    assert snap["stage"] == "done" and snap["cached"] is True and snap["bytes"] == 0, "the poll sees done, flagged cached, with no bytes downloaded"


def test_changing_the_downloaded_files_invalidates_the_cached_payload(ix, fetch, spy, monkeypatch, tmp_path):
    key = skippy_bridge.mesh_key(TALL)
    skippy_bridge.mesh_payload(TALL, key)
    assert spy["pack"] == 1
    gl = next((tmp_path / "assets").rglob("*.gltf"))
    gl.write_bytes(gl.read_bytes() + b"\n")                       # same model, different file: the fingerprint moves
    forget_memory(monkeypatch)
    skippy_bridge.mesh_payload(TALL, key)
    assert spy["pack"] == 2, "packed again"


def test_a_new_packer_version_repacks(ix, fetch, spy, monkeypatch):
    key = skippy_bridge.mesh_key(TALL)
    skippy_bridge.mesh_payload(TALL, key)
    forget_memory(monkeypatch)
    monkeypatch.setattr(meshpack, "PACK_VERSION", "next")
    skippy_bridge.mesh_payload(TALL, key)
    assert spy["pack"] == 2


def test_a_corrupt_entry_on_disk_is_repacked_not_served(ix, fetch, spy, monkeypatch):
    key = skippy_bridge.mesh_key(TALL)
    a = skippy_bridge.mesh_payload(TALL, key)
    pc = skippy_bridge.pack_cache()
    p = pc.path(skippy_bridge.packcache_key(TALL))
    raw = p.read_bytes()
    p.write_bytes(raw[:-3] + b"\x00\x00\x00")
    forget_memory(monkeypatch)
    b = skippy_bridge.mesh_payload(TALL, key)
    assert spy["pack"] == 2 and b["b64"] == a["b64"] and "cached" not in b


def test_two_simultaneous_first_requests_for_one_model_download_and_pack_once(ix, fetch, spy, monkeypatch):
    key = skippy_bridge.mesh_key(TALL)
    outs, errs = [], []
    start = threading.Barrier(2)

    def go():
        start.wait()
        try:
            outs.append(skippy_bridge.mesh_payload(TALL, key))
        except Exception as e:                                     # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(outs) == 2
    assert spy["pack"] == 1 and len(fetch.calls) == 3, "one download, one pack"
    assert outs[0]["b64"] == outs[1]["b64"]
    assert sorted(bool(o.get("cached")) for o in outs) == [False, True], "the second read the first one's entry"


def test_the_cache_can_be_switched_off(ix, fetch, spy, monkeypatch):
    monkeypatch.setattr(skippy_bridge, "PACK_CACHE_CAP", 0)
    key = skippy_bridge.mesh_key(TALL)
    skippy_bridge.mesh_payload(TALL, key)
    forget_memory(monkeypatch)
    skippy_bridge.mesh_payload(TALL, key)
    assert spy["pack"] == 2


def test_a_scratch_download_directory_never_reads_the_real_cache(ix, fetch, tmp_path):
    pc = skippy_bridge.pack_cache()
    assert tmp_path in pc.dir.parents and pc.dir != packcache.DEFAULT_DIR


# ============================================================================================================ prewarm
import prewarm  # noqa: E402


def test_prewarm_prints_the_plan_and_does_nothing_without_go(ix, fetch, monkeypatch, capsys):
    monkeypatch.setattr(prewarm, "load_list", lambda path=None: [{"noun": "garden gnome", "card": TALL, "approx_mb": 1.5}])
    assert prewarm.main([]) == 0
    out = capsys.readouterr().out
    assert "garden gnome" in out and "to fetch" in out and "1.5 MB" in out and "Re-run with --go" in out
    assert fetch.calls == [], "nothing was downloaded"
    assert not (skippy_bridge.pack_cache().dir.exists() and list(skippy_bridge.pack_cache().dir.glob("*.svmc")))


def test_prewarm_go_packs_what_is_missing_and_skips_what_is_cached(ix, fetch, monkeypatch, capsys):
    items = [{"noun": "garden gnome", "card": TALL, "approx_mb": 1.5}]
    monkeypatch.setattr(prewarm, "load_list", lambda path=None: items)
    assert prewarm.main(["--go"]) == 0
    assert len(fetch.calls) == 3
    capsys.readouterr()
    assert prewarm.plan(items, skippy_bridge)["rows"][0]["status"] == "cached"
    fetch.calls.clear()
    assert prewarm.main(["--go"]) == 0
    assert fetch.calls == [] and "already" in capsys.readouterr().out
    skippy_bridge.MESH_PACKED.clear()
    assert skippy_bridge.mesh_payload(TALL, skippy_bridge.mesh_key(TALL))["cached"] is True, "a player's request now hits"


def test_prewarm_refuses_a_plan_over_the_cap(ix, fetch, monkeypatch, capsys):
    monkeypatch.setattr(prewarm, "load_list", lambda path=None: [{"noun": "garden gnome", "card": TALL, "approx_mb": 200.0}])
    assert prewarm.main(["--go"]) == 2 and fetch.calls == []


def test_the_tracked_prewarm_list_is_small_official_and_in_the_index():
    items = prewarm.load_list()
    assert 5 <= len(items) <= 30 and sum(float(i["approx_mb"]) for i in items) < prewarm.TOTAL_CAP_MB
    assert all(re.fullmatch(r"found\.(poly-haven|gazebo-fuel|nasa-3d-resources)\.[a-z0-9._-]+", i["card"]) for i in items), "official public sources only"
    assert len({i["card"] for i in items}) == len(items)


# ============================================================================================================ the client cache and the handshake (Lua)
from test_gmod_found_mesh import (MeshGame, box_blob, client_entity, client_with_payload, draws, feed, make_msgs, mesh_client, summon)  # noqa: E402
from test_gmod_found import ALICE, BOB  # noqa: E402
from test_gmod_glue import lua_table  # noqa: E402

CAROL = "76561190000000003"
SHA = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731

SERVER_MSGS = r"""
function T.msgs(name)
  local out = {}
  for _, m in ipairs(MOCK.nets) do
    if m.name == name then
      local sids = {}
      for _, p in ipairs(m.to or {}) do sids[#sids + 1] = p.sid end
      local e = MOCK.export(m)
      out[#out + 1] = { to = sids, seq = e.seq, bytes = m.bytes, kind = m.kind }
    end
  end
  return MOCK.json_encode(out)
end
function T.mesh_stage() local m = next(Holodeck.Mesh.by_key); return m and Holodeck.Mesh.by_key[m].fx_stage or "none" end
"""


def msgs(w, name):
    return json.loads(w.call("T.msgs", name))


def to_server(game, ply, name, seq):
    game.w.rt.eval("function(p, n, s) MOCK.deliver_seq(n, s, p) end")(ply, name, lua_table(game.w.rt, seq))


def hello(game, ply):
    to_server(game, ply, "holodeck_mesh_hello", [])


def answer(game, ply, mesh_id, crc, have):
    to_server(game, ply, "holodeck_mesh_have", [["u", mesh_id, 8], ["u", crc, 32], ["b", have]])


@pytest.fixture()
def cgame(ix):
    g = MeshGame()
    g.w.rt.execute(SERVER_MSGS)
    return g


def ready(cgame, sids=(BOB,)):
    """Alice summons; Bob (and others) are in the game; the model becomes ready. Returns (alice, {sid: player}, payload blob)."""
    alice = summon(cgame)
    others = {s: cgame.player(s) for s in sids}
    return alice, others


def serve_ready(game):
    game.serve_mesh()
    obj = skippy_bridge.mesh_payload(TALL, skippy_bridge.mesh_key(TALL))
    return base64.b64decode(obj["b64"])


def chunk_count(game, sid=None):
    ms = msgs(game.w, "holodeck_mesh")
    return len([m for m in ms if sid is None or sid in m["to"]])


def test_a_client_that_said_hello_is_offered_the_model_not_streamed_to(cgame):
    alice, others = ready(cgame)
    hello(cgame, others[BOB])
    blob = serve_ready(cgame)
    cgame.w.advance(0.5)
    offers = msgs(cgame.w, "holodeck_mesh_offer")
    assert [m["to"] for m in offers] == [[BOB]], "only the client that can answer is offered"
    assert [it[0] for it in offers[0]["seq"]] == ["u", "u", "u", "s"]
    mesh_id, crc, total, sha = [it[1] for it in offers[0]["seq"]]
    assert (mesh_id, crc, total, sha) == (1, zlib.crc32(blob), len(blob), SHA(blob))
    assert chunk_count(cgame, BOB) == 0, "nothing streamed to Bob until he says he wants it"
    assert chunk_count(cgame, ALICE) > 0 and chunk_count(cgame, ALICE) == chunk_count(cgame), "Alice never said hello: she is streamed to as before (an old client)"


def test_have_means_nothing_is_ever_sent_to_that_client(cgame):
    alice, others = ready(cgame)
    hello(cgame, others[BOB])
    blob = serve_ready(cgame)
    cgame.w.advance(0.5)
    answer(cgame, others[BOB], 1, zlib.crc32(blob), True)
    cgame.w.advance(10)
    assert chunk_count(cgame, BOB) == 0
    assert cgame.w.rt.eval("Holodeck.Mesh.cache_hits") == 1 and cgame.w.rt.eval("Holodeck.Mesh.cache_wants") == 0


def test_want_streams_the_model_to_that_client_alone(cgame):
    alice, others = ready(cgame, (BOB, CAROL))
    hello(cgame, others[BOB])
    hello(cgame, others[CAROL])
    blob = serve_ready(cgame)
    cgame.w.advance(0.5)
    answer(cgame, others[BOB], 1, zlib.crc32(blob), False)
    answer(cgame, others[CAROL], 1, zlib.crc32(blob), True)
    cgame.w.advance(10)
    assert chunk_count(cgame, BOB) == -(-len(blob) // (16 * 1024)) and chunk_count(cgame, CAROL) == 0


def test_an_offer_nobody_answers_falls_back_to_streaming(cgame):
    alice, others = ready(cgame)
    hello(cgame, others[BOB])
    serve_ready(cgame)
    cgame.w.advance(1)
    assert chunk_count(cgame, BOB) == 0
    cgame.w.advance(3)                                            # past OFFER_WAIT: a client with a broken cache path still gets its object
    assert chunk_count(cgame, BOB) > 0


def test_an_answer_with_the_wrong_checksum_or_no_offer_changes_nothing(cgame):
    alice, others = ready(cgame)
    hello(cgame, others[BOB])
    blob = serve_ready(cgame)
    cgame.w.advance(0.5)
    answer(cgame, others[BOB], 1, zlib.crc32(blob) ^ 1, True)      # not the offered bytes
    answer(cgame, others[BOB], 9, 5, False)                        # a model the server does not have
    answer(cgame, others[CAROL] if CAROL in others else cgame.player(CAROL), 1, zlib.crc32(blob), False)    # never offered
    cgame.w.advance(1)
    assert chunk_count(cgame, BOB) == 0 and chunk_count(cgame, CAROL) == 0, "an unsolicited answer cannot make the server send"
    answer(cgame, others[BOB], 1, zlib.crc32(blob), False)
    answer(cgame, others[BOB], 1, zlib.crc32(blob), False)         # answered twice: one stream
    cgame.w.advance(10)
    assert chunk_count(cgame, BOB) == -(-len(blob) // (16 * 1024))


def test_a_late_joiner_who_says_hello_is_offered_everything_that_is_ready(cgame):
    alice, _ = ready(cgame, ())
    blob = serve_ready(cgame)
    cgame.w.advance(2)
    cgame.w.rt.execute("MOCK.nets = {}")
    late = cgame.player(BOB)
    hello(cgame, late)
    cgame.w.advance(0.5)
    assert chunk_count(cgame) == 0
    offers = msgs(cgame.w, "holodeck_mesh_offer")
    assert len(offers) == 1 and offers[0]["to"] == [BOB]
    answer(cgame, late, 1, zlib.crc32(blob), False)
    cgame.w.advance(2)
    assert chunk_count(cgame, BOB) > 0


def test_the_objects_finish_when_every_offer_was_answered_have(cgame):
    alice, others = ready(cgame, ())
    hello(cgame, alice)                                             # the only player is cache-capable
    blob = serve_ready(cgame)
    cgame.w.advance(0.5)
    assert cgame.w.call("T.mesh_stage") == "sending", "waiting for the answer is reported as sending"
    answer(cgame, alice, 1, zlib.crc32(blob), True)
    cgame.w.advance(0.1)
    assert cgame.w.call("T.mesh_stage") == "decoding"
    cgame.w.advance(3)
    assert cgame.visuals() == [{"visual": "mesh", "mesh": 1}]


# ---------------------------------------------------------------------------------------------- the client's files
def files_of(w):
    return json.loads(w.rt.eval("function() local o = {} for k, v in pairs(MOCK.files) do o[k] = MOCK.tohex(v) end return MOCK.json_encode(o) end")())


def put_files(w, files):
    w.rt.eval("function(h) for k, v in pairs(h) do MOCK.files[k] = MOCK.fromhex(v) end end")(lua_table(w.rt, files))


def strict_client(files=None):
    w = mesh_client()
    w.rt.execute("MOCK.strict_files = true")
    if files:
        put_files(w, files)
    return w


def register(w, raw, blob):
    w.rt.eval("function(a, b) MOCK.lzma[MOCK.fromhex(a)] = MOCK.fromhex(b) end")(blob.hex(), raw.hex())


def outbox(w, name):
    return [m for m in json.loads(w.rt.eval("function() local o = {} for _, m in ipairs(MOCK.nets) do if m.kind == 'server' then local e = MOCK.export(m); o[#o + 1] = e end end return MOCK.json_encode(o) end")()) if m["name"] == name]


def offer_msg(blob, mesh_id=1, sha=None, crc=None, total=None):
    return [["u", mesh_id, 8], ["u", zlib.crc32(blob) if crc is None else crc, 32], ["u", len(blob) if total is None else total, 20], ["s", sha or SHA(blob)]]


def deliver(w, name, seq):
    w.rt.eval("function(n, s) MOCK.deliver_seq(n, s) end")(name, lua_table(w.rt, seq))


def test_a_received_payload_is_kept_under_its_content_hash_with_an_allowed_extension_and_an_index_without_steam_ids():
    raw, blob = box_blob()
    w = strict_client()
    register(w, raw, blob)
    client_entity(w)
    feed(w, make_msgs(blob))
    assert w.call("T.store", 1) == "ready"
    files = files_of(w)
    dat = [k for k in files if k.endswith(".dat")]
    assert dat == [f"mentat/meshcache/{SHA(blob)[:32]}.dat"] and bytes.fromhex(files[dat[0]]) == blob, "stored whole, binary-safe, content-hash name"
    index = json.loads(bytes.fromhex(files["mentat/meshcache/index.json"]))
    assert index["v"] == 1 and index["entries"][0]["sha"] == SHA(blob) and index["entries"][0]["bytes"] == len(blob)
    assert not re.search(r"\d{17}", json.dumps(index)), "no SteamID anywhere in the index"
    for k in files:
        assert k.rsplit(".", 1)[-1] in ("dat", "json", "txt"), f"{k}: the engine would silently refuse any other extension"
    assert all(k.startswith("mentat/meshcache/") for k in files)


def test_a_returning_client_answers_have_and_downloads_nothing():
    raw, blob = box_blob()
    w1 = strict_client()
    register(w1, raw, blob)
    client_entity(w1)
    feed(w1, make_msgs(blob))
    w2 = strict_client(files_of(w1))                               # a new session on the same machine: the data folder is all that is left
    register(w2, raw, blob)
    client_entity(w2)
    deliver(w2, "holodeck_mesh_offer", offer_msg(blob))
    have = outbox(w2, "holodeck_mesh_have")
    assert len(have) == 1 and have[0]["seq"] == [["u", 1, 8], ["u", zlib.crc32(blob), 32], ["b", True]]
    assert w2.call("T.store", 1) == "ready", "the shape is built from the file"
    assert w2.rt.eval("Holodeck.MeshClient.stats.chunks") == 0, "no payload byte crossed the net"
    assert w2.rt.eval("Holodeck.MeshClient.stats.cache_hits") == 1
    e = w2.rt.eval("ents.FindByClass('holodeck_found')[1]")
    e.Draw(e)
    assert draws(w2)[0] == 1


def test_a_client_without_the_model_answers_want_then_keeps_what_arrives():
    raw, blob = box_blob()
    w = strict_client()
    register(w, raw, blob)
    client_entity(w)
    deliver(w, "holodeck_mesh_offer", offer_msg(blob))
    have = outbox(w, "holodeck_mesh_have")
    assert have[0]["seq"][2] == ["b", False] and w.call("T.store", 1) == "none"
    feed(w, make_msgs(blob))
    assert w.call("T.store", 1) == "ready" and any(k.endswith(".dat") for k in files_of(w))


@pytest.mark.parametrize("damage", ["flip", "truncate", "empty", "wrong_name"])
def test_a_corrupt_or_mismatched_cache_entry_is_dropped_and_re_requested(damage):
    raw, blob = box_blob()
    w1 = strict_client()
    register(w1, raw, blob)
    client_entity(w1)
    feed(w1, make_msgs(blob))
    files = files_of(w1)
    key = f"mentat/meshcache/{SHA(blob)[:32]}.dat"
    data = bytearray(bytes.fromhex(files[key]))
    if damage == "flip":
        data[len(data) // 2] ^= 0x55
    elif damage == "truncate":
        data = data[:-7]
    elif damage == "empty":
        data = bytearray()
    files[key] = bytes(data).hex()
    w2 = strict_client(files)
    register(w2, raw, blob)
    client_entity(w2)
    if damage == "wrong_name":
        deliver(w2, "holodeck_mesh_offer", offer_msg(blob, sha="ab" * 32))      # the server announces other bytes than the file holds
    else:
        deliver(w2, "holodeck_mesh_offer", offer_msg(blob))
    have = outbox(w2, "holodeck_mesh_have")
    assert have[0]["seq"][2] == ["b", False], "the client says it wants it"
    assert w2.call("T.store", 1) == "none"
    if damage != "wrong_name":
        left = files_of(w2)
        assert key not in left, "the damaged file is deleted"
        assert json.loads(bytes.fromhex(left["mentat/meshcache/index.json"]))["entries"] == []
    feed(w2, make_msgs(blob))                                      # the stream that follows is accepted and re-cached
    assert w2.call("T.store", 1) == "ready" and key in files_of(w2)


def test_an_offer_that_is_malformed_is_ignored():
    raw, blob = box_blob()
    w = strict_client()
    client_entity(w)
    deliver(w, "holodeck_mesh_offer", offer_msg(blob, sha="zz" * 32))
    deliver(w, "holodeck_mesh_offer", offer_msg(blob, sha="ab" * 5))
    deliver(w, "holodeck_mesh_offer", offer_msg(blob, mesh_id=200))
    assert outbox(w, "holodeck_mesh_have") == [] and w.rt.eval("Holodeck.MeshClient.stats.rejected") == 3


def test_the_client_cache_evicts_the_least_recently_used_at_its_cap():
    w = strict_client()
    cache_ = w.rt.eval("Holodeck.MeshClient.Cache")
    blobs = [bytes([i]) * 1000 for i in range(1, 5)]
    shas = [SHA(b) for b in blobs]
    w.rt.execute("Holodeck.MeshClient.Cache.CAP_BYTES = 3500")
    for b in blobs[:3]:
        assert cache_.store(SHA(b), b.decode("latin-1"), zlib.crc32(b))
    assert cache_.lookup(shas[0], 1000, zlib.crc32(blobs[0])) is not None, "0 is used again: 1 is the oldest"
    assert cache_.store(shas[3], blobs[3].decode("latin-1"), zlib.crc32(blobs[3]))
    files = [k for k in files_of(w) if k.endswith(".dat")]
    assert len(files) == 3 and cache_.total_bytes() <= 3500
    assert f"mentat/meshcache/{shas[1][:32]}.dat" not in files and f"mentat/meshcache/{shas[0][:32]}.dat" in files and f"mentat/meshcache/{shas[3][:32]}.dat" in files
    index = json.loads(bytes.fromhex(files_of(w)["mentat/meshcache/index.json"]))
    assert {e["sha"] for e in index["entries"]} == {shas[0], shas[2], shas[3]}, "the index agrees with the folder"
    assert not cache_.store(SHA(b"big"), "x" * (300 * 1024), 1), "over the payload cap: never stored"


def test_the_client_cache_has_a_documented_default_cap_inside_the_download_budget():
    w = strict_client()
    assert w.rt.eval("Holodeck.MeshClient.Cache.CAP_BYTES") == 120 * 1024 * 1024 < 200 * 1024 * 1024


def test_a_client_with_the_cache_off_keeps_nothing_and_always_asks_for_everything():
    raw, blob = box_blob()
    w = strict_client()
    w.rt.execute("GetConVar('holodeck_mesh_cache'):SetValue(0)")
    register(w, raw, blob)
    client_entity(w)
    feed(w, make_msgs(blob))
    assert w.call("T.store", 1) == "ready" and not any(k.endswith(".dat") for k in files_of(w))
    w.rt.execute("MOCK.hooks.InitPostEntity.holodeck_mesh_hello()")
    assert [m["name"] for m in outbox(w, "holodeck_mesh_hello")] == [] and [m["seq"] for m in outbox(w, "holodeck_mesh_want")] == [[["u", 0, 8]]]


def test_a_joining_client_says_hello_and_an_old_server_gets_the_old_ask():
    w = strict_client()
    w.rt.execute("MOCK.hooks.InitPostEntity.holodeck_mesh_hello()")
    assert len(outbox(w, "holodeck_mesh_hello")) == 1 and outbox(w, "holodeck_mesh_want") == [], "with a cache the server offers instead of streaming"
    old = strict_client()
    old.rt.execute("local real = net.Start; net.Start = function(n) if n == 'holodeck_mesh_hello' then return false end return real(n) end")      # a server that never heard of it
    old.rt.execute("MOCK.hooks.InitPostEntity.holodeck_mesh_hello()")
    assert outbox(old, "holodeck_mesh_hello") == [] and [m["seq"] for m in outbox(old, "holodeck_mesh_want")] == [[["u", 0, 8]]]


def test_a_needed_mesh_waits_for_the_offer_then_asks_and_not_at_all_while_an_offer_was_answered():
    raw, blob = box_blob()
    w = strict_client()
    e = client_entity(w)
    for _ in range(3):
        e.Draw(e)
    assert outbox(w, "holodeck_mesh_want") == [], "the server's offer may be on its way: no download request yet"
    w.rt.execute("MOCK.advance(2)")
    e.Draw(e)
    assert len(outbox(w, "holodeck_mesh_want")) == 1, "no offer came (an older server): it asks, once"
    w2 = strict_client()
    e2 = client_entity(w2)
    deliver(w2, "holodeck_mesh_offer", offer_msg(blob))
    w2.rt.execute("MOCK.advance(5)")
    e2.Draw(e2)
    assert outbox(w2, "holodeck_mesh_want") == [], "answered want: the stream is coming, it is not asked for twice"
    w2.rt.execute("MOCK.advance(10)")
    e2.Draw(e2)
    assert len(outbox(w2, "holodeck_mesh_want")) == 1, "but a stream that never came is asked for after a while"


def test_the_client_writes_only_allowed_extensions_in_a_directory_it_made():
    raw, blob = box_blob()
    w = strict_client()
    register(w, raw, blob)
    client_entity(w)
    feed(w, make_msgs(blob))
    wrote = w.rt.eval("MOCK.write_count")
    assert wrote is not None and len(list(wrote.keys())) >= 2
    for path in wrote.keys():
        assert path.rsplit(".", 1)[-1] in ("dat", "json", "txt")
    assert w.rt.eval("MOCK.dirs['mentat/meshcache']") is True


# ---------------------------------------------------------------------------------------------- the whole trip: server and two client sessions
def test_end_to_end_a_second_session_costs_two_small_messages_and_no_payload(cgame):
    # session 1: the client has nothing; it says hello, is offered the model, wants it, receives it, keeps it
    alice, _ = ready(cgame, ())
    hello(cgame, alice)
    blob = serve_ready(cgame)
    raw = meshpack.unlzma_alone(blob)
    cgame.w.advance(0.5)
    c1 = strict_client()
    register(c1, raw, blob)
    client_entity(c1)
    offer = msgs(cgame.w, "holodeck_mesh_offer")
    assert len(offer) == 1
    deliver(c1, "holodeck_mesh_offer", offer[0]["seq"])
    have = outbox(c1, "holodeck_mesh_have")[0]
    answer(cgame, alice, *[it[1] for it in have["seq"]])
    cgame.w.advance(10)
    chunks = msgs(cgame.w, "holodeck_mesh")
    assert chunks, "session 1 downloaded the payload"
    for m in chunks:
        c1.rt.eval("function(n, s) MOCK.deliver_seq(n, s) end")("holodeck_mesh", lua_table(c1.rt, m["seq"]))
    assert c1.call("T.store", 1) == "ready"
    first_bytes = sum(m["bytes"] for m in chunks)
    # session 2: a fresh server and a fresh client runtime that only share the data folder
    g2 = MeshGame()
    g2.w.rt.execute(SERVER_MSGS)
    a2, _ = ready(g2, ())
    hello(g2, a2)
    serve_ready(g2)
    g2.w.advance(0.5)
    c2 = strict_client(files_of(c1))
    register(c2, raw, blob)
    client_entity(c2)
    offer2 = msgs(g2.w, "holodeck_mesh_offer")
    deliver(c2, "holodeck_mesh_offer", offer2[0]["seq"])
    have2 = outbox(c2, "holodeck_mesh_have")[0]
    answer(g2, a2, *[it[1] for it in have2["seq"]])
    g2.w.advance(10)
    assert msgs(g2.w, "holodeck_mesh") == [], "the server streamed nothing"
    handshake = len(offer2) + len(outbox(c2, "holodeck_mesh_have"))
    assert handshake == 2 and first_bytes > 0
    assert c2.call("T.store", 1) == "ready" and c2.rt.eval("Holodeck.MeshClient.stats.chunks") == 0
    assert g2.visuals() == [{"visual": "mesh", "mesh": 1}]
