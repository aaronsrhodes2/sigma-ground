"""Found objects in the holodeck: any object in the SkippyVerse inventory summoned as a correctly sized, correctly massed, labelled
PLACEHOLDER BOX (the `holodeck_found` entity), through the bridge's /resolve and the holodeck's normal pipeline.

Offline, in two halves that meet in the middle:

  * the bridge half runs the REAL `skippy_bridge.resolve_found` and the REAL HTTP handler against a small synthetic SkippyVerse index
    (no network: a guard fails any connection that is not to the loopback; downloads go through an injected fetch);
  * the game half runs the REAL holodeck.lua, parser, planner and entities/holodeck_found against stage/gmod/holodeck/gmod_mock.lua
    and feeds it the bridge half's real answers.

Every behaviour claim has a test that fails without the code (checked by running this file against the previous holodeck.lua).
"""
from __future__ import annotations

import io
import json
import socket
import sys
import threading
import urllib.error
import urllib.request
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
from test_gmod_glue import World  # noqa: E402
from test_skippyverse_acquire import Fake, ph_routes  # noqa: E402

HU = C.HU_M
ALICE, BOB = "76561190000000001", "76561190000000002"


# ============================================================================================================ the synthetic inventory
def sized(w, d, h, basis="sourced"):
    return {"w": w, "d": d, "h": h, "basis": basis, "source": "test table"}


def found_card(name, size=None, *, source="Poly Haven", lic="CC0", cid=None, url=None, aliases=(), category="misc"):
    prov = {"source": source, "license": lic, "author": "Test Author", "url": url or f"https://polyhaven.com/a/{C.slug(name)}"}
    return C.make_card(id=cid or f"found.{C.slug(source)}.{C.slug(name)}", name=name, size=size, category=category, aliases=aliases,
                       visual={"class": "found", "style": "photoscan-or-pbr", "blocky": False, "lods": []}, provenance=prov,
                       mass=C.prior_mass(size) if size else None, flags=["front_unknown"])


HYDRANT = (0.35, 0.9, 0.8)           # w (across the front), d (along it), h in metres


def cards():
    return [
        found_card("fire hydrant", sized(*HYDRANT), aliases=["hydrant"]),
        found_card("wheelbarrow", sized(0.6, 1.5, 0.7, "measured")),
        found_card("traffic bollard", None, source="Gazebo Fuel", lic="CC-BY-4.0", url="https://fuel.gazebosim.org/1.0/O/models/bollard"),
        found_card("standing lamp thing", sized(0.5, 0.5, 1.7), aliases=[]),
        found_card("test airliner", sized(60.0, 70.0, 19.0, "sourced"), category="vehicle/aircraft"),
        # a Valve-derived curated card: in the inventory, never served as a found object
        C.make_card(id="curated.crate.1", name="valve widget", kind="valve widget", category="misc", size=sized(0.5, 0.5, 0.5, "measured"),
                    front={"yaw_deg": 0.0, "source": "curated-by-eye"}, symmetric=False,
                    visual={"class": "curated", "style": "x", "blocky": False}, provenance={"source": "Half-Life 2 (Valve)", "license": "Valve-SSA"}),
        # a ShapeNet-derived placeholder with a name and a size
        C.make_card(id="placeholder.shapenetsem.gizmo", name="shapenet gizmo", kind="shapenet gizmo", category="misc", size=sized(0.4, 0.4, 0.4, "estimated"),
                    visual={"class": "placeholder", "style": "box", "blocky": True}, provenance={"source": "ShapeNetSem", "license": "ShapeNet terms"}),
    ]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this file may reach beyond the loopback; the bridge's HTTP tests use 127.0.0.1."""
    real_connect = socket.socket.connect

    def guarded(self, addr, *a, **k):
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"the test tried to reach {addr!r}")
        return real_connect(self, addr, *a, **k)
    monkeypatch.setattr(socket.socket, "connect", guarded)

    def local_only(host, port, *a, **k):
        if host not in ("127.0.0.1", "localhost"):
            raise AssertionError(f"DNS lookup for {host!r}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
    monkeypatch.setattr(socket, "getaddrinfo", local_only)


@pytest.fixture()
def ix(tmp_path, monkeypatch):
    IX.build_index(cards(), tmp_path / "ix", synonym_groups=[])
    index = IX.load_index(tmp_path / "ix", shared=True)
    monkeypatch.setattr(skippy_bridge, "FOUND_INDEX", index)
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", None)
    monkeypatch.setattr(skippy_bridge, "FOUND_FETCH", False)
    monkeypatch.setattr(skippy_bridge, "GATE", skippy_bridge.Gate())
    return index


# ============================================================================================================ the bridge half
def test_a_servable_card_with_a_size_comes_back_as_a_planner_kind_in_hammer_units(ix):
    r = skippy_bridge.resolve_found("fire hydrant")
    assert r["place"] is True and r["status"] == "exact"
    k = r["kind"]
    v = k["variants"][0]
    assert k["kind"] == "fire hydrant" and "hydrant" in k["names"]
    assert (v["w"], v["d"], v["h"]) == pytest.approx((HYDRANT[0] / HU, HYDRANT[1] / HU, HYDRANT[2] / HU))
    assert v["entity"] == "holodeck_found" and v["symmetric"] is True and v["front"] == 0.0, "front unknown means symmetric"
    assert v["mins"] == pytest.approx([-v["d"] / 2, -v["w"] / 2, 0.0]) and v["maxs"] == pytest.approx([v["d"] / 2, v["w"] / 2, v["h"]])
    assert v["mass"] == pytest.approx(C.prior_mass(sized(*HYDRANT))["kg"]) and v["mass_status"] == "guessed"
    assert v["found"] == {"card": "found.poly-haven.fire-hydrant", "name": "fire hydrant", "source": "Poly Haven", "licence": "CC0", "basis": "sourced",
                          "mesh": skippy_bridge.mesh_key("found.poly-haven.fire-hydrant")}, "a Poly Haven card carries the key of its real shape"
    assert k["approx"] == [] and k["seat"] is False and k["surface"] is False, "nothing may stand in for another word, sit, or carry things"
    assert "box" in r["say"], "Skippy says it is a box for now"
    assert r["card"]["size_m"]["basis"] == "sourced" and r["card"]["licence"] == "CC0" and r["card"]["source"] == "Poly Haven"


def test_a_noun_nobody_has_is_a_sentence_and_nothing_to_place(ix):
    r = skippy_bridge.resolve_found("dragon")
    assert r["place"] is False and r["kind"] is None and r["card"] is None and r["status"] == "not_found"
    assert "dragon" in r["say"]


def test_the_closest_i_have_is_only_a_sentence(ix):
    r = skippy_bridge.resolve_found("fire hydrnt")                    # a typo: read, said, but not placed
    assert r["place"] is False and r["kind"] is None
    assert "hydrant" in r["say"]


@pytest.mark.parametrize("noun", ["valve widget", "shapenet gizmo"])
def test_non_redistributable_cards_are_never_served_as_found_objects(ix, noun):
    r = skippy_bridge.resolve_found(noun)
    assert r["place"] is False and r["kind"] is None, "Valve- and ShapeNet-derived cards are the curated library, not found objects"
    assert "may not share" in r["say"]


def test_a_card_with_no_size_and_no_fetch_says_it_does_not_know_how_big(ix, monkeypatch):
    calls = []
    monkeypatch.setattr(skippy_bridge, "FOUND_FETCH", False)
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", None)
    monkeypatch.setattr(A, "acquire_and_measure", lambda *a, **k: calls.append(1))
    r = skippy_bridge.resolve_found("traffic bollard")
    assert r["place"] is False and r["kind"] is None and r["status"] == "needs_size"
    assert "don't know how big" in r["say"]
    assert calls == [], "no download when fetching is off"


def test_a_card_with_no_size_costs_one_download_and_comes_back_measured(ix, monkeypatch, tmp_path):
    fake = Fake(ph_routes())                                         # a 2 x 1 x 3 glTF box
    seen = []

    def acq(card):
        seen.append(card["id"])
        return A.acquire_and_measure(found_card("traffic bollard", None, url="https://polyhaven.com/a/barrel_03", cid=card["id"]), fake, tmp_path)
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", acq)
    r = skippy_bridge.resolve_found("traffic bollard")
    assert seen == ["found.gazebo-fuel.traffic-bollard"], "exactly one asset is fetched"
    assert r["fetched"] is True and r["place"] is True and r["status"] == "exact"
    v = r["kind"]["variants"][0]
    assert r["card"]["size_m"]["basis"] == "measured"
    assert (v["w"], v["d"], v["h"]) == pytest.approx((2 / HU, 3 / HU, 1 / HU)), "glTF is Y-up: the 2 x 1 x 3 box is 2 wide, 3 deep, 1 high"
    assert "Fetching a model" in r["say"] or "Got it" in r["say"]
    assert r["say"].endswith("a box of that size."), "even after a fetch, the sentence still says it is only a box"
    assert len(r["say"]) <= skippy_bridge.MAX_FOUND_SAY


def test_nothing_is_downloaded_for_a_closest_or_typo_reading(ix, monkeypatch):
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", lambda c: pytest.fail("a download for a card the game would not place"))
    r = skippy_bridge.resolve_found("traffic bolard")                  # a typo of a card that has no size
    assert r["place"] is False and r["kind"] is None and r["fetched"] is False
    assert "bollard" in r["say"]


def test_a_failed_download_is_reported_not_hidden(ix, monkeypatch):
    def boom(card):
        raise A.AcquireError("the host did not answer")
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", boom)
    r = skippy_bridge.resolve_found("traffic bollard")
    assert r["place"] is False and r["kind"] is None and r["status"] == "fetch_failed"
    assert "didn't work" in r["say"] and "won't guess" in r["say"]


def test_only_one_download_runs_at_a_time(ix, monkeypatch):
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", lambda c: pytest.fail("a second download started"))
    with skippy_bridge.GATE.slot("fetch"):
        with pytest.raises(skippy_bridge.Busy):
            skippy_bridge.resolve_found("traffic bollard")


def test_the_bounded_fetch_keeps_to_https_the_named_hosts_and_forty_megabytes(monkeypatch):
    import time
    fetch = skippy_bridge.bounded_fetch(time.monotonic() + 30)
    for bad in ("http://dl.polyhaven.org/x", "https://evil.example/x", "file:///etc/passwd", "ftp://dl.polyhaven.org/x"):
        with pytest.raises(A.AcquireRefused):
            fetch(bad, max_bytes=10)

    class Resp:
        def __init__(self, n, length=None):
            self.n, self.headers, self.sent = n, ({"Content-Length": str(length)} if length is not None else {}), 0
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def read(self, k):
            k = min(k, self.n - self.sent)
            self.sent += k
            return b"x" * k

    class Opener:
        def __init__(self, n, length=None): self.n, self.length = n, length
        def open(self, req, timeout=None): return Resp(self.n, self.length)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: Opener(10 * 1024 * 1024))
    budgeted = skippy_bridge.bounded_fetch(time.monotonic() + 30, max_total=16 * 1024 * 1024)
    assert len(budgeted("https://dl.polyhaven.org/a", max_bytes=A.MAX_BYTES)) == 10 * 1024 * 1024
    with pytest.raises(A.TooLarge):
        budgeted("https://dl.polyhaven.org/b", max_bytes=A.MAX_BYTES)                       # 10 + 10 MB is over the 16 MB budget
    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: Opener(5, length=99 * 1024 * 1024))
    with pytest.raises(A.TooLarge):
        skippy_bridge.bounded_fetch(time.monotonic() + 30)("https://dl.polyhaven.org/c", max_bytes=A.MAX_BYTES)     # declared size over the limit
    with pytest.raises(TimeoutError):
        skippy_bridge.bounded_fetch(time.monotonic() - 1)("https://dl.polyhaven.org/d", max_bytes=A.MAX_BYTES)       # the deadline has passed


class Http:
    """The real bridge HTTP server on a loopback port."""

    def __init__(self, monkeypatch):
        monkeypatch.setattr(skippy_bridge, "TOKEN", "sekrit")
        self.srv = skippy_bridge.Server(("127.0.0.1", 0), skippy_bridge.Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def post(self, path, body, token="sekrit"):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json", "X-Skippy-Token": token})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def test_the_resolve_endpoint_is_guarded_like_the_others(ix, monkeypatch):
    h = Http(monkeypatch)
    try:
        assert h.post("/resolve", {"noun": "fire hydrant"}, token="wrong")[0] == 401
        code, body = h.post("/resolve", {"noun": "fire hydrant", "audience": "game", "who": "alice"})
        assert code == 200 and body["ok"] and body["place"] is True and body["kind"]["kind"] == "fire hydrant"
        assert h.post("/resolve", {"noun": ""})[0] == 400
        assert h.post("/resolve", {"noun": "!!!"})[0] == 400, "a noun with no letters is no noun"
        assert h.post("/resolve", {"noun": "fire hydrant", "audience": "web"})[0] == 400, "the game asks as the game"
        assert h.post("/resolve", {"noun": "x" * 50000})[0] in (400, 413)
        # a browser page may not ask
        req = urllib.request.Request(h.base + "/resolve", data=b'{"noun":"fire hydrant"}',
                                     headers={"content-type": "application/json", "X-Skippy-Token": "sekrit", "Origin": "http://evil.example"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 403
    finally:
        h.close()


def test_the_resolve_endpoint_is_rate_limited_per_caller_and_overall(ix, monkeypatch):
    h = Http(monkeypatch)
    try:
        codes = [h.post("/resolve", {"noun": "dragon", "who": "alice"})[0] for _ in range(6)]
        assert codes[:4] == [200] * 4 and codes[4] == 429, f"one caller may ask {skippy_bridge.PER_CALLER_PER_MINUTE['resolve']} a minute: {codes}"
        assert h.post("/resolve", {"noun": "dragon", "who": "bob"})[0] == 200, "another caller is not held up by it"
        monkeypatch.setattr(skippy_bridge, "GATE", skippy_bridge.Gate(per_minute={"plan": 24, "reply": 40, "speak": 40, "resolve": 2}))
        codes = [h.post("/resolve", {"noun": "dragon", "who": f"p{i}"})[0] for i in range(4)]
        assert codes == [200, 200, 429, 429], "and the whole bridge has its own ceiling"
    finally:
        h.close()


def test_a_busy_download_slot_is_a_503_for_the_game(ix, monkeypatch):
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", lambda c: pytest.fail("no second download"))
    h = Http(monkeypatch)
    try:
        with skippy_bridge.GATE.slot("fetch"):
            assert h.post("/resolve", {"noun": "traffic bollard"})[0] == 503
    finally:
        h.close()


# ============================================================================================================ the game half
LUA_HELPERS = r"""
function T.found()
  local out = {}
  for _, e in ipairs(MOCK.ents) do
    if e.valid and e.class == "holodeck_found" then
      local b = e.box
      out[#out + 1] = {
        mins = b and { b.mins.x, b.mins.y, b.mins.z }, maxs = b and { b.maxs.x, b.maxs.y, b.maxs.z },
        mass = e.phys.mass, fmass = e:GetFoundMass(), name = e:GetFoundName(), source = e:GetFoundSource(), licence = e:GetFoundLicence(),
        basis = e:GetFoundBasis(), label = e:LabelText(), visual = e:GetVisual(), x = e.pos.x, y = e.pos.y, z = e.pos.z, yaw = e.ang.y,
        hid = e.HolodeckId, motion = e.phys.motion, group = e.group,
      }
    end
  end
  return MOCK.json_encode(out)
end
function T.props()
  local n = 0
  for _, e in ipairs(MOCK.ents) do if e.valid and e.class == "prop_physics" then n = n + 1 end end
  return n
end
function T.kinds() local o = {}; for _, id in ipairs(Holodeck.found_order) do o[#o + 1] = id end; return table.concat(o, ",") end
function T.say() return Holodeck.last_say or "" end
function T.legacy() local n = 0; for _, e in pairs(Holodeck.registry) do if e.legacy then n = n + 1 end end; return n end
"""

BRIDGE = "http://127.0.0.1:8777"


class Game:
    """A booted mock engine with the real holodeck, the real found entity, and the real bridge function standing in for the network."""

    def __init__(self, bridge=BRIDGE, **cv):
        cvars = {"holodeck_agent": "0", "holodeck_persona": "0", "holodeck_voice_out": "0", "holodeck_bridge_token": "tok", **cv}
        if bridge:
            cvars["holodeck_bridge"] = bridge
        self.w = World(cvars=cvars)
        self.w.rt.execute(LUA_HELPERS)
        self.w.rt.execute('MOCK.load_entity("holodeck_found", "server")')
        self.answered = set()
        self.noun_posts = []

    def player(self, sid=ALICE, **kw):
        return self.w.player(sid, **kw)

    def ask(self, p, text):
        return self.w.ask(p, text)

    def posts(self, path="/resolve"):
        t = self.w.call("T.posts", path)
        return [t[i] for i in range(1, len(t) + 1)]

    def body(self, i):
        return json.loads(self.w.rt.eval("function(i) return MOCK.https[i].body end")(i))

    def headers(self, i):
        return {k: v for k, v in self.w.rt.eval("function(i) return MOCK.https[i].headers end")(i).items()}

    def serve(self, answer=None):
        """Answer every unanswered /resolve with the real bridge (or `answer(noun)`)."""
        n = 0
        for i in self.posts():
            if i in self.answered:
                continue
            self.answered.add(i)
            noun = self.body(i)["noun"]
            self.w.post_json(i, answer(noun) if answer else skippy_bridge.resolve_found(noun))
            n += 1
        return n

    def settle(self, secs=12):
        self.w.advance(secs)

    def found(self):
        return json.loads(self.w.call("T.found"))

    def props(self):
        return self.w.call("T.props")

    def count(self, sid=None):
        return self.w.count(sid)

    def say(self):
        return self.w.call("T.say")

    def summon(self, p, text):
        """Ask, let the bridge answer, let the props materialise."""
        r = self.ask(p, text)
        self.serve()
        self.settle()
        return r


@pytest.fixture()
def game(ix):
    return Game()


def test_an_unknown_noun_is_asked_of_the_bridge_and_a_box_appears_with_the_cards_size_and_mass(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a fire hydrant")
    posts = game.posts()
    assert len(posts) == 1, "the bridge is asked once, for the noun"
    assert game.body(posts[0]) == {"noun": "fire hydrant", "audience": "game", "who": ALICE}
    assert game.headers(posts[0])["X-Skippy-Token"] == "tok", "with the token, like every bridge call"
    assert game.count() == 0 and game.found() == [], "nothing stands until the answer is in (the request never blocks the tick)"
    game.serve()
    game.settle()
    (box,) = game.found()
    assert game.count() == 1 and game.props() == 0, "one registered thing, a holodeck_found box and no model prop"
    card_hu = [HYDRANT[0] / HU, HYDRANT[1] / HU, HYDRANT[2] / HU]               # w, d, h
    ex, ey, ez = (box["maxs"][i] - box["mins"][i] for i in range(3))
    assert (ex, ey, ez) == pytest.approx((card_hu[1], card_hu[0], card_hu[2]), rel=1e-9), "the box is the card's footprint (d along the front, w across)"
    assert box["mins"][2] == 0, "the origin is on the floor under the box"
    assert box["mass"] == pytest.approx(C.prior_mass(sized(*HYDRANT))["kg"]) and box["fmass"] == pytest.approx(box["mass"])
    assert box["z"] == pytest.approx(0), "it stands on the floor"
    assert game.w.call("T.kinds") == "fire hydrant"


def test_the_label_strings_are_the_cards_and_say_where_it_came_from(game):
    game.summon(game.player(ALICE, admin=True), "add a fire hydrant")
    (box,) = game.found()
    assert (box["name"], box["source"], box["licence"], box["basis"]) == ("fire hydrant", "Poly Haven", "CC0", "sourced")
    assert box["label"] == "fire hydrant - Poly Haven, CC0, size sourced"
    assert box["visual"] == "box"


def test_skippy_says_the_honest_sentence_when_it_adds_the_object(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a fire hydrant")
    game.serve()
    said = [m for m in game.w.rt.eval("Holodeck.log").values()]
    assert any("fire hydrant" in s and "CC0" in s and "box" in s for s in said), said


def test_real_mass_off_leaves_the_engines_own_mass(ix):
    g = Game(holodeck_real_mass="0")
    g.summon(g.player(ALICE, admin=True), "add a fire hydrant")
    (box,) = g.found()
    assert box["fmass"] == 0 and box.get("mass") is None, "holodeck_real_mass 0: no mass is applied"


def test_it_goes_through_the_normal_pipeline_so_several_are_laid_out_without_overlap(game):
    game.summon(game.player(ALICE, admin=True), "add three fire hydrants")
    boxes = game.found()
    assert len(boxes) == 3
    ids = {b["hid"] for b in boxes}
    assert len(ids) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = boxes[i], boxes[j]
            gap = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
            assert gap > min(HYDRANT[0], HYDRANT[1]) / HU, f"{i} and {j} are {gap:.1f} HU apart: overlapping boxes"


def test_a_found_object_mixes_with_the_library_in_one_request(game):
    game.summon(game.player(ALICE, admin=True), "add a couch and a fire hydrant")
    assert game.props() == 1 and len(game.found()) == 1
    assert game.count() == 2


def test_once_known_the_noun_needs_no_more_questions(game):
    p = game.player(ALICE, admin=True)
    game.summon(p, "add a fire hydrant")
    n = len(game.posts())
    game.summon(p, "add another fire hydrant")
    game.summon(p, "put two hydrants beside it")                       # the card's own alias
    assert len(game.posts()) == n, "the kind is in the live library for the rest of the session"
    assert len(game.found()) >= 3


def test_an_unknown_noun_with_no_bridge_is_an_honest_reply_and_nothing_stands(ix):
    g = Game(bridge="")
    p = g.player(ALICE, admin=True)
    g.ask(p, "add a wheelbarrow")
    assert g.posts() == [] and g.count() == 0 and g.found() == [] and g.props() == 0
    assert "library" in g.say(), g.say()


def test_the_parsers_guess_stands_when_the_resolver_has_nothing(game):
    """'fire' is the library's approximate name for a wood stove: the bridge is asked about 'fire dragon' in case it exists, and when
    it does not, the parser's own stated guess stands exactly as before (nothing new is said or placed)."""
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a fire dragon")
    assert [game.body(i)["noun"] for i in game.posts()] == ["fire dragon"]
    game.serve()
    game.settle()
    assert game.found() == [] and game.props() == 1, "the wood stove the parser chose, as it always did"
    said = " ".join(game.w.rt.eval("Holodeck.log").values())
    assert "can't make new things" not in said, "the resolver's no is not spoken for a guess"


def test_a_dragon_places_nothing_and_skippy_says_so(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a dragon")
    assert game.serve() == 1
    game.settle()
    assert game.count() == 0 and game.found() == [] and game.props() == 0
    assert "dragon" in game.say() and game.w.call("T.kinds") == ""


def test_a_dragon_is_not_asked_about_again_for_a_while(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a dragon")
    game.serve()
    game.ask(p, "add a dragon")
    assert len(game.posts()) == 1, "a definite no is remembered, so a loop of requests costs the bridge nothing"
    assert "dragon" in game.say()


def test_the_closest_i_have_places_nothing(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a wheelbarow")                                   # one letter short of a card's name
    game.serve()
    game.settle()
    assert game.count() == 0 and game.found() == []
    assert "wheelbarrow" in game.say() and "wheelbarow" in game.say()


@pytest.mark.parametrize("noun", ["valve widget", "shapenet gizmo"])
def test_a_non_servable_card_is_refused_in_game(game, noun):
    p = game.player(ALICE, admin=True)
    game.ask(p, f"add a {noun}")
    game.serve()
    game.settle()
    assert game.count() == 0 and game.found() == [] and game.props() == 0
    assert "share" in game.say()


def test_a_card_with_no_size_places_nothing_and_says_it_does_not_know_how_big(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a traffic bollard")
    game.serve()
    game.settle()
    assert game.count() == 0 and game.found() == []
    assert "how big" in game.say()


def test_a_card_that_was_fetched_and_measured_is_placed_at_the_measured_size(ix, monkeypatch, tmp_path):
    fake = Fake(ph_routes())
    monkeypatch.setattr(skippy_bridge, "FOUND_ACQUIRER", lambda card: A.acquire_and_measure(
        found_card("traffic bollard", None, url="https://polyhaven.com/a/barrel_03", cid=card["id"]), fake, tmp_path))
    g = Game()
    g.summon(g.player(ALICE, admin=True), "add a traffic bollard")
    (box,) = g.found()
    dims = sorted(box["maxs"][i] - box["mins"][i] for i in range(3))
    assert dims == pytest.approx(sorted([1 / HU, 2 / HU, 3 / HU])), "the measured 2 x 1 x 3 box"
    assert box["basis"] == "measured"


def test_a_malformed_answer_that_claims_to_be_placeable_is_refused(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a wheelbarrow")
    good = skippy_bridge.resolve_found("wheelbarrow")
    bad = json.loads(json.dumps(good))
    bad["kind"]["variants"][0].update(w=-5)
    game.serve(lambda noun: bad)
    game.settle()
    assert game.count() == 0 and game.found() == []
    assert game.w.call("T.kinds") == ""


@pytest.mark.parametrize("patch,expect", [
    ({"w": 4000.0, "d": 4000.0, "h": 40.0}, "bigger"),                 # 76 m: more than anything placed here
    ({"w": 0.2, "d": 0.2, "h": 0.2}, "too small"),
    ({"entity": "prop_physics"}, "do not add up"),
    ({"model": ""}, "do not add up"),
])
def test_the_game_checks_the_numbers_it_is_given(game, patch, expect):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a wheelbarrow")
    bad = json.loads(json.dumps(skippy_bridge.resolve_found("wheelbarrow")))
    bad["kind"]["variants"][0].update(patch)
    game.serve(lambda noun: bad)
    game.settle()
    assert game.count() == 0 and game.found() == []
    assert expect in game.say()


def test_a_word_the_library_already_knows_cannot_be_taken_over(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a fire hydrant")
    evil = json.loads(json.dumps(skippy_bridge.resolve_found("fire hydrant")))
    evil["kind"]["kind"] = "couch"                                      # a found card named like a library kind
    evil["kind"]["names"] = ["couch", "sofa"]
    game.serve(lambda noun: evil)
    game.settle()
    assert game.w.call("T.kinds") == "", "the server's own couch stays the couch"
    assert game.found() == []


def test_a_bridge_error_falls_back_to_the_old_honest_reply(game):
    p = game.player(ALICE, admin=True)
    game.ask(p, "add a wheelbarrow")
    (i,) = game.posts()
    game.w.rt.eval("function(i) MOCK.respond(i, 'fail', 'timeout') end")(i)
    game.settle()
    assert game.count() == 0 and game.found() == []
    assert "library" in game.say()


def test_found_objects_off_never_asks(ix):
    g = Game(holodeck_found="0")
    g.ask(g.player(ALICE, admin=True), "add a wheelbarrow")
    assert g.posts() == [] and g.count() == 0


def test_the_bridge_rationing_applies_to_the_lookup(ix):
    g = Game(holodeck_bridge_inflight="0")                              # the bridge may have no call in flight: nothing can be asked
    g.ask(g.player(ALICE, admin=True), "add a wheelbarrow")
    assert g.posts() == [] and g.count() == 0 and "library" in g.say()


def test_a_second_request_while_the_first_is_being_looked_up_waits(game):
    p = game.player(ALICE)
    game.ask(p, "add a fire hydrant")
    game.ask(p, "add a wheelbarrow")
    assert len(game.posts()) == 1, "one lookup per player at a time"
    assert "still looking" in game.w.call("function() return Holodeck.last_tell or '' end")
    game.serve()
    game.settle()
    assert len(game.found()) == 1


def test_a_lookup_is_not_blocked_by_the_cooldown_it_started(ix):
    g = Game(holodeck_cooldown="30")
    p = g.player(ALICE)
    g.ask(p, "add a fire hydrant")
    g.serve()
    g.settle()
    assert len(g.found()) == 1, "the second pass of a request is the same request, not a new one"


def test_the_player_leaving_while_it_is_looked_up_drops_the_request(game):
    p = game.player(ALICE)
    game.ask(p, "add a fire hydrant")
    game.w.rt.eval("function(p) MOCK.disconnect(p) end")(p)
    game.serve()
    game.settle()
    assert game.count() == 0 and game.found() == []


def test_undo_removes_a_found_object_and_remove_it_then_undo_brings_it_back(game):
    p = game.player(ALICE, admin=True)
    game.summon(p, "add a fire hydrant")
    assert len(game.found()) == 1
    game.ask(p, "undo")
    game.settle()
    assert game.count() == 0 and game.found() == [], "undo of the add takes the box away"
    game.summon(p, "add a fire hydrant")                                # known now: no question
    assert len(game.found()) == 1
    game.ask(p, "remove it")
    game.settle()
    assert game.found() == [] and game.count() == 0
    game.ask(p, "undo")
    game.settle()
    (box,) = game.found()
    assert box["label"] == "fire hydrant - Poly Haven, CC0, size sourced", "it came back as the same kind of thing"
    assert game.count() == 1


def test_the_limits_count_found_objects(ix):
    g = Game(holodeck_max_per_player="2")
    p = g.player(ALICE)                                                 # not an admin: the per-player limit applies
    g.summon(p, "add four fire hydrants")
    assert len(g.found()) == 2 and g.count(ALICE) == 2
    assert "room for only 2" in g.say() or "limit" in g.say() or "2" in g.say()
    g.summon(p, "add a fire hydrant")
    assert len(g.found()) == 2, "at the limit nothing more stands"
    assert "limit" in g.say()


def test_the_server_wide_limit_counts_found_objects(ix):
    g = Game(holodeck_max_ents="3")
    p = g.player(ALICE, admin=True)
    g.summon(p, "add two couches")
    g.summon(p, "add three fire hydrants")
    assert g.count() <= 3 and len(g.found()) <= 1


def test_the_server_rules_can_veto_a_found_object(ix):
    g = Game()
    g.w.rt.execute('hook.Add("PlayerSpawnProp", "nope", function(ply, model) if model:find("found.", 1, true) then return false end end)')
    p = g.player(ALICE)
    g.summon(p, "add a fire hydrant")
    assert g.found() == [] and g.count() == 0, "the server's own spawn rules apply to found objects too"


def test_clear_removes_found_objects_and_undo_returns_them(game):
    p = game.player(ALICE, admin=True)
    game.summon(p, "add two fire hydrants")
    assert len(game.found()) == 2
    game.ask(p, "clear everything")
    game.settle()
    assert game.found() == []
    game.ask(p, "undo")
    game.settle()
    assert len(game.found()) == 2


def test_found_kinds_survive_a_library_reload_and_the_things_standing_stay_manageable(game):
    p = game.player(ALICE, admin=True)
    game.summon(p, "add a fire hydrant")
    game.w.rt.execute("Holodeck.BuildLibrary()")                         # holodeck_reload: the library is rebuilt from the server's models
    assert game.w.call("T.kinds") == "fire hydrant" and game.w.call("T.legacy") == 0, "still a kind, and not a legacy thing"
    game.ask(p, "remove the fire hydrant")
    game.settle()
    assert game.found() == [] and game.count() == 0


def test_a_thing_whose_kind_left_the_library_is_legacy_but_still_removable(game):
    p = game.player(ALICE, admin=True)
    game.summon(p, "add a fire hydrant")
    game.w.rt.execute('Holodeck.found_kinds["fire hydrant"] = nil; Holodeck.found_order = {}; Holodeck.BuildLibrary()')
    assert game.w.call("T.legacy") == 1, "kept as a legacy thing: no index, but its own variant"
    game.ask(p, "clear everything")
    game.settle()
    assert game.found() == [] and game.count() == 0


def test_the_shelf_of_found_kinds_is_bounded(ix):
    g = Game()
    g.w.rt.execute("Holodeck.found_order = {}")
    n = g.w.rt.eval("""function()
      local made = 0
      for i = 1, 70 do
        local k = { kind = "thing" .. string.char(97 + i % 26) .. string.char(97 + math.floor(i / 26)), names = {}, approx = {}, category = "found",
                    variants = { { model = "m" .. i, front = 0, symmetric = true, w = 10, d = 10, h = 10, mins = {-5, -5, 0}, maxs = {5, 5, 10}, cx = 0, cy = 0,
                                   entity = "holodeck_found", found = {} } } }
        k.names = { k.kind }
        local id = Holodeck.AddFoundKind and Holodeck.AddFoundKind(k)
        if id then made = made + 1 end
      end
      return made
    end""")()
    assert n == 70 and len(list(g.w.rt.eval("Holodeck.found_order").values())) == 64, "oldest kinds that nothing stands as make way"


# ============================================================================================================ the entity itself
def test_the_visual_hook_accepts_box_and_mesh_with_an_id_and_refuses_everything_else(ix):
    g = Game()
    ent = g.w.rt.eval('function() local e = ents.Create("holodeck_found"); e:SetFound({ mins = {-5,-5,0}, maxs = {5,5,10}, name = "x", source = "s", licence = "CC0", basis = "guessed", mass = 3 }); return e end')()
    assert ent.SetVisual(ent, "box") is True
    ok, why = ent.SetVisual(ent, "model", {"id": 1})
    assert ok is False and "unsupported" in why
    for bad in ('{vertices = {}}', "{id = 0}", "{id = 65}", "{id = 1.5}", "{id = 0/0}", "nil"):
        ok, why = g.w.rt.eval(f"function(e) return e:SetVisual('mesh', {bad}) end")(ent)
        assert ok is False and "id from 1 to 64" in why, bad
    assert ent.GetVisual(ent) == "box", "a refused kind changes nothing"
    nets = g.w.rt.eval('function() local n = 0; for _, m in ipairs(MOCK.nets) do if m.name == "holodeck_found_visual" then n = n + 1 end end return n end')()
    assert nets == 1, "only the accepted change was broadcast"


def test_the_entity_refuses_to_exist_without_a_box(ix):
    g = Game()
    ok = g.w.rt.eval('function() local e = ents.Create("holodeck_found"); local r = e:SetFound({ mins = {0,0,0}, maxs = {0,0,0} }); e:Spawn(); return r, e:IsValid() end')()
    assert tuple(ok) == (False, False)
    r = g.w.rt.eval('function() local e = ents.Create("holodeck_found"); e:Spawn(); return e:IsValid() end')()
    assert r is False, "no box was set before Spawn: the entity removes itself"


def test_the_entity_class_missing_on_a_server_is_a_failed_spawn_not_a_crash(ix):
    w = World(cvars={"holodeck_agent": "0", "holodeck_persona": "0", "holodeck_voice_out": "0", "holodeck_bridge": BRIDGE, "holodeck_bridge_token": "t"})
    w.rt.execute(LUA_HELPERS)                                            # (holodeck_found is NOT loaded here)
    g = Game.__new__(Game)
    g.w, g.answered, g.noun_posts = w, set(), []
    p = g.player(ALICE, admin=True)
    g.ask(p, "add a fire hydrant")
    g.serve()
    g.settle()
    assert g.found() == [] and g.count() == 0, "the registry forgets what could not be created"


CLIENT_FAKES = r"""
RENDER = { boxes = {}, wires = {}, texts = {}, rects = {}, cam = 0 }
render = { SetColorMaterial = function() end,
           DrawBox = function(pos, ang, mn, mx, col, z) RENDER.boxes[#RENDER.boxes + 1] = { col = col, mn = mn, mx = mx } end,
           DrawWireframeBox = function(pos, ang, mn, mx, col, z) RENDER.wires[#RENDER.wires + 1] = { col = col } end }
cam = { Start3D2D = function() RENDER.cam = RENDER.cam + 1 end, End3D2D = function() end }
surface = { SetFont = function() end, GetTextSize = function(t) return #t * 8, 20 end, SetDrawColor = function(c) RENDER.fill = c end,
            DrawRect = function(...) RENDER.rects[#RENDER.rects + 1] = { ... } end }
draw = { SimpleText = function(t, f, x, y, col) RENDER.texts[#RENDER.texts + 1] = { t = t, col = col } end }
TEXT_ALIGN_CENTER, TEXT_ALIGN_BOTTOM = 1, 4
function Entity(i) return MOCK.by_index and MOCK.by_index[i] or MOCK.NULL end
"""


def client_world():
    w = World(cvars={}, load=(), server=False)
    w.rt.execute(CLIENT_FAKES)
    w.rt.execute('MOCK.load_entity("holodeck_found", "client")')
    return w


def client_box(w, x=0.0):
    return w.rt.eval("""function(x)
      local e = ents.Create("holodeck_found")
      e:SetFound({ mins = {-10,-20,0}, maxs = {10,20,40}, name = "fire hydrant", source = "Poly Haven", licence = "CC0", basis = "sourced", mass = 14 })
      e:SetPos(Vector(x, 0, 0))
      e.GetColor = function() return Color(255, 255, 255, 255) end
      e.LocalToWorld = function(s, v) return s.pos + v end
      e.SetRenderBounds = function() end
      return e
    end""")(x)


def test_the_client_draws_a_green_translucent_box_and_a_label_for_those_within_ten_metres():
    w = client_world()
    me = w.rt.eval("function() MOCK.me = MOCK.newplayer('1', { pos = Vector(100, 0, 0) }); return MOCK.me end")()
    e = client_box(w)
    e.DrawTranslucent(e)
    r = w.rt.eval("RENDER")
    assert len(r.boxes) == 1 and len(r.wires) == 1
    fill, wire = r.boxes[1].col, r.wires[1].col
    assert (fill.r, fill.g, fill.b) == (0, 255, 136) == (wire.r, wire.g, wire.b), "the project's green accent, one flat colour"
    assert 0 < fill.a < 128 and wire.a == 255, "translucent fill, solid edges"
    assert r.texts[1].t == "fire hydrant - Poly Haven, CC0, size sourced" and r.cam == 1
    assert (r.fill.r, r.fill.g, r.fill.b) == (0, 0, 0), "the label stands on black"


def test_the_label_is_not_drawn_beyond_ten_metres_but_the_box_still_is():
    w = client_world()
    w.rt.eval("function() MOCK.me = MOCK.newplayer('1', { pos = Vector(0, 0, 0) }) end")()
    e = client_box(w, x=1500.0)                                           # 28 m
    e.DrawTranslucent(e)
    r = w.rt.eval("RENDER")
    assert len(r.boxes) == 1 and r.cam == 0 and len(r.texts) == 0


def test_the_box_fades_in_with_the_holodecks_materialising_alpha():
    w = client_world()
    w.rt.eval("function() MOCK.me = MOCK.newplayer('1', { pos = Vector(0, 0, 0) }) end")()
    e = client_box(w)
    e.GetColor = w.rt.eval("function() return function() return Color(255, 255, 255, 0) end end")()
    e.Draw(e)
    r = w.rt.eval("RENDER")
    assert r.boxes[1].col.a == 0 and r.wires[1].col.a == 0


def test_the_client_refuses_the_same_things_and_keeps_the_box():
    w = client_world()
    e = client_box(w)
    ok, why = e.SetVisual(e, "model", {"id": 1})
    assert ok is False and "unsupported" in why and e.GetVisual(e) == "box"
    ok, why = e.SetVisual(e, "mesh", {"vertices": []})
    assert ok is False and e.GetVisual(e) == "box"


# ============================================================================================================ the live grader
def test_the_live_grader_holds_the_engines_box_to_the_card_within_two_percent(ix):
    import live
    hu = C.HU_M
    w, d, h = HYDRANT

    def row(scale=1.0, cls="holodeck_found", mass=C.prior_mass(sized(*HYDRANT))["kg"], source="Poly Haven"):
        ex, ey, ez = d / hu * scale, w / hu, h / hu
        return {"id": 1, "kind": "fire hydrant", "variant": {"card": "found.poly-haven.fire-hydrant"}, "class": cls, "mass": mass, "asleep": True,
                "obb": {"mins": [-ex / 2, -ey / 2, 0], "maxs": [ex / 2, ey / 2, ez]},
                "found": {"name": "fire hydrant", "source": source, "basis": "sourced", "label": "fire hydrant - " + source + ", CC0, size sourced"}}
    assert live.grade_found({"entities": [row()]})[0] == []
    assert live.grade_found({"entities": [row(scale=1.01)]})[0] == [], "1% is inside the tolerance"
    assert "off the card's size" in " ".join(live.grade_found({"entities": [row(scale=1.05)]})[0])
    assert "not a holodeck_found box" in " ".join(live.grade_found({"entities": [row(cls="prop_physics")]})[0])
    assert "mass" in " ".join(live.grade_found({"entities": [row(mass=30.0)]})[0])
    assert "label strings" in " ".join(live.grade_found({"entities": [row(source="Somewhere Else")]})[0])
    problems, rows = live.grade_found({"entities": [row()]})
    assert rows[0]["size_err_pct"] == pytest.approx([0, 0, 0], abs=1e-6) and rows[0]["card"] == "found.poly-haven.fire-hydrant"
