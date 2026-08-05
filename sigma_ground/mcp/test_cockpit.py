"""Tests for cockpit.py — the tier-4 composed entry point (engine E2).

Covers the E2 gates at the SOCKET level plus the moved-route parity:
  * GATE 1 (socket form): two real websocket clients on one world name see
    frames from the SAME authoritative world — an impulse applied through
    client A is visible in the frames client B receives.
  * Save over the wire round-trips through WorldHost.load to the exact
    wire hash (the socket form of GATE 2).
  * /chat and /apply answer through the cockpit with envelopes byte-equal
    to the front door's own (the moved-verbatim identity).
  * The OLD /live protocol still answers (inherited from serve untouched).
  * serve.py still boots standalone as the pure-static dev server.

Port doctrine (project memory: stale servers shadow :8765): every server
here binds port 0 and reads the real port back; every socket is closed in
a finally; teardown order is clients → shutdown_worlds() → httpd.shutdown.

The websocket client reuses serve._WebSocket over a hand-rolled HTTP
upgrade — viable because our server's recv_json zero-masks when the mask
bit is absent, so unmasked client frames are tolerated by our own server.
Every receive is deadline-bounded; never a bare blocking recv.

NO happy-path /apply render test here (it would write real render bundles
via _render_push's sweep — slow and polluting; front_door's own green
tests cover that path). Disclosed.
"""
import base64
import http.client
import json
import socket
import threading
import time
import uuid

import pytest

from sigma_ground.engine.host import WorldHost
from sigma_ground.mcp import cockpit
from sigma_ground.radiance.web import library, serve

ITEMS = ["1700s table", "plate", "Doughnut"]
SIM = {"builder": "push_off", "items": ITEMS,
       "pushed": "plate", "pushed_index": 1}
IMPULSE = {"kind": "impulse", "target": "plate",
           "vector": [1.0, 0.0, 0.0],
           "variables": {"push_speed_m_s": 2.2}}


@pytest.fixture(autouse=True)
def _library_in_tmp(tmp_path, monkeypatch):
    """/chat's record_turn must never touch the real store — the documented
    seam: repoint the module-level _LIB_DIR at a tmp dir."""
    monkeypatch.setattr(library, "_LIB_DIR", str(tmp_path / "library"))


@pytest.fixture()
def port():
    httpd = cockpit.make_server(0)                  # ephemeral — NEVER 8765
    p = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield p
    finally:
        # client sockets are already closed by each test's own finally;
        # then worlds, then the server (the documented teardown order)
        cockpit.shutdown_worlds()
        httpd.shutdown()
        httpd.server_close()


def _wname():
    return "w-" + uuid.uuid4().hex[:12]


class _Client:
    """A minimal /world (or /live) websocket client over serve._WebSocket."""

    def __init__(self, port, path="/world"):
        self.sock = socket.create_connection(("127.0.0.1", port),
                                             timeout=10.0)
        key = base64.b64encode(b"0123456789abcdef").decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\n"
               f"Sec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket handshake failed")
            buf += chunk
        status, rest = buf.split(b"\r\n", 1)
        assert b"101" in status, status
        self.ws = serve._WebSocket(self.sock)
        self.ws._buf = rest.split(b"\r\n\r\n", 1)[1]   # early frames, if any
        self.frames = []            # every {"type":"frame"} seen, in order

    def send(self, obj):
        self.ws.send_json(obj)

    def recv(self, timeout=0.2):
        msg = self.ws.recv_json(timeout=timeout)
        if msg and msg.get("type") == "frame":
            self.frames.append(msg)
        return msg

    def expect(self, mtype, timeout=10.0):
        """Bounded wait for one message of the given type (frames seen on
        the way are still captured into self.frames)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.recv(timeout=0.2)
            if msg and msg.get("type") == mtype:
                return msg
        raise AssertionError(f"timed out waiting for {mtype!r}")

    def collect_frames(self, n, timeout=15.0):
        deadline = time.monotonic() + timeout
        while len(self.frames) < n and time.monotonic() < deadline:
            self.recv(timeout=0.2)
        assert len(self.frames) >= n, \
            f"only {len(self.frames)} frames within {timeout}s"
        return self.frames

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def _post(port, path, obj, timeout=60.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(obj),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, json.loads(body)
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# § 1. GATE 1 (socket form) — two clients, ONE authoritative world
# ═══════════════════════════════════════════════════════════════════════════

def test_gate1_two_sockets_observe_one_world(port):
    w = _wname()
    a = _Client(port)
    b = None
    try:
        a.send({"type": "attach", "world": w, "items": ITEMS})
        ready_a = a.expect("ready", timeout=120.0)  # first attach pays build_stack
        assert ready_a["created"] is True
        assert ready_a["labels"] == ITEMS

        b = _Client(port)
        b.send({"type": "attach", "world": w})
        ready_b = b.expect("ready", timeout=10.0)
        assert ready_b["created"] is False          # JOINED, not rebuilt
        assert ready_b["labels"] == ready_a["labels"]

        a.collect_frames(6)
        b.collect_frames(6)
        x0 = b.frames[-1]["bodies"][1]["pos"][0]

        # the impulse goes in through A ...
        a.send({"type": "apply", "id": 1, "target": "plate",
                "force": dict(IMPULSE)})
        applied = a.expect("applied", timeout=10.0)
        assert applied["id"] == 1
        assert applied["target"] == "plate"
        assert isinstance(applied["step_index"], int)

        # ... and becomes visible in the frames B receives: ONE world,
        # not one sim per socket (which is what the old /live does)
        deadline = time.monotonic() + 15.0
        moved = False
        while time.monotonic() < deadline and not moved:
            b.recv(timeout=0.2)
            if b.frames and b.frames[-1]["bodies"][1]["pos"][0] > x0 + 1e-6:
                moved = True
        assert moved, "A's impulse never showed up in B's frames"

        # soak up A's queued frames for the comparison window
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            a.recv(timeout=0.1)

        # frames matched by step must be IDENTICAL across A and B. The
        # hello frame (frames[0]) is per-client — never naive-zip streams.
        map_a = {f["step"]: f for f in a.frames[1:]}
        map_b = {f["step"]: f for f in b.frames[1:]}
        common = sorted(set(map_a) & set(map_b))
        assert len(common) >= 3, (len(map_a), len(map_b))
        for step in common:
            assert map_a[step] == map_b[step]
        # sim time is monotonic per client (hello included)
        for cl in (a, b):
            ts = [f["t"] for f in cl.frames]
            assert ts == sorted(ts)
    finally:
        a.close()
        if b is not None:
            b.close()


# ═══════════════════════════════════════════════════════════════════════════
# § 2. persistence across disconnects + join refusals
# ═══════════════════════════════════════════════════════════════════════════

def test_world_persists_across_disconnect_and_refuses_mismatch(port):
    w = _wname()
    c1 = _Client(port)
    try:
        c1.send({"type": "attach", "world": w, "items": ITEMS})
        r1 = c1.expect("ready", timeout=120.0)
        assert r1["created"] is True
        step0 = r1["step"]
    finally:
        c1.close()

    time.sleep(0.3)                     # the world keeps running unobserved

    c2 = _Client(port)
    try:
        c2.send({"type": "attach", "world": w})
        r2 = c2.expect("ready", timeout=10.0)
        assert r2["created"] is False
        assert r2["step"] > step0       # it LIVED across the disconnect

        c3 = _Client(port)
        try:
            c3.send({"type": "attach", "world": w,
                     "items": ["plate", "Doughnut"]})
            ref = c3.expect("refusal", timeout=10.0)
            assert "refused" in ref["text"] or "exists" in ref["text"]
        finally:
            c3.close()
    finally:
        c2.close()


def test_attach_with_load_onto_existing_world_refused(port):
    """A load naming an EXISTING world is refused — restoring a save can
    neither replace the live world nor be silently ignored (the join would
    hand the client a different state than it asked to restore)."""
    w = _wname()
    c1 = _Client(port)
    try:
        c1.send({"type": "attach", "world": w, "items": ITEMS})
        c1.expect("ready", timeout=120.0)
        c1.send({"type": "save", "id": 1})
        data = c1.expect("save", timeout=10.0)["data"]

        c2 = _Client(port)
        try:
            c2.send({"type": "attach", "world": w,
                     "load": json.loads(json.dumps(data))})
            ref = c2.expect("refusal", timeout=10.0)
            assert "already exists" in ref["text"]
            assert "load" in ref["text"]
        finally:
            c2.close()

        # the live world is untouched by the refused load: still joinable
        c3 = _Client(port)
        try:
            c3.send({"type": "attach", "world": w})
            r3 = c3.expect("ready", timeout=10.0)
            assert r3["created"] is False
        finally:
            c3.close()
    finally:
        c1.close()


def test_attach_unknown_world_without_items_refused(port):
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": "nope-" + uuid.uuid4().hex[:8]})
        ref = c.expect("refusal", timeout=10.0)
        assert "no world named" in ref["text"]
    finally:
        c.close()


# ═══════════════════════════════════════════════════════════════════════════
# § 3. GATE 2 (socket form) — save over the wire round-trips exactly
# ═══════════════════════════════════════════════════════════════════════════

def test_save_over_socket_round_trips_to_wire_hash(port):
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "apply", "id": 1, "target": "plate",
                "force": dict(IMPULSE)})
        c.expect("applied", timeout=10.0)
        c.send({"type": "pause"})
        c.expect("paused", timeout=10.0)
        c.send({"type": "hash", "id": 7})
        hv = c.expect("hash", timeout=10.0)
        assert hv["id"] == 7
        c.send({"type": "save", "id": 8})
        sv = c.expect("save", timeout=10.0)
        assert sv["id"] == 8
        data = sv["data"]
    finally:
        c.close()
    assert data["state_hash"] == hv["value"]        # paused: same boundary
    assert [e for e in data["command_log"]]         # the impulse is in the log
    # the wire save restores to the EXACT wire hash (real rebuild + replay;
    # load() also self-verifies against the embedded checksum)
    loaded = WorldHost.load(json.loads(json.dumps(data)))
    assert loaded.state_hash() == hv["value"]


# ═══════════════════════════════════════════════════════════════════════════
# § 4. refusal correlation over the wire
# ═══════════════════════════════════════════════════════════════════════════

def test_refusals_forwarded_honestly_with_ids(port):
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "apply", "id": 3, "target": "plate",
                "force": {"kind": "source", "target": "plate",
                          "power_w": 50.0, "duration": 1.0}})
        ref = c.expect("refusal", timeout=10.0)
        assert ref["id"] == 3
        assert "thermal" in ref["text"]
        c.send({"type": "no-such-verb"})
        ref2 = c.expect("refusal", timeout=10.0)
        assert "unknown message type" in ref2["text"]
    finally:
        c.close()


# ═══════════════════════════════════════════════════════════════════════════
# § 5. moved-route envelope parity (/chat, /apply)
# ═══════════════════════════════════════════════════════════════════════════

def test_chat_envelope_parity(port):
    front_door = pytest.importorskip("sigma_ground.mcp.front_door")
    text = "what is the speed of light"
    expected = dict(front_door.dispatch(text, use_llm=False,
                                        session=front_door.Session(),
                                        mode=None))
    expected["session_id"] = "PARITY"
    status, resp = _post(port, "/chat", {"text": text, "use_llm": False,
                                         "session_id": "PARITY"})
    assert status == 200
    assert json.dumps(resp, sort_keys=True) == \
        json.dumps(expected, sort_keys=True)

    # the empty-text clarify envelope (skips record_turn) — byte-exact too
    expected2 = dict(front_door.dispatch("", use_llm=False,
                                         session=front_door.Session(),
                                         mode=None))
    expected2["session_id"] = "PARITY2"
    status2, resp2 = _post(port, "/chat", {"text": "", "use_llm": False,
                                           "session_id": "PARITY2"})
    assert status2 == 200
    assert json.dumps(resp2, sort_keys=True) == \
        json.dumps(expected2, sort_keys=True)


def test_apply_envelope_parity_on_deterministic_refusals(port):
    pytest.importorskip("sigma_ground.mcp.front_door")
    reqs = [
        # unknown kind
        {"sim": SIM, "force": {"kind": "yeet", "target": "plate"}},
        # in the closed set, solver not landed
        {"sim": SIM, "force": {"kind": "force", "target": "plate",
                               "vector": [1.0, 0.0, 0.0], "duration": 1.0}},
        # missing sim provenance
        {"sim": {}, "force": {"kind": "impulse", "target": "plate",
                              "variables": {"push_speed_m_s": 2.0}}},
        # target not in the scene
        {"sim": SIM, "force": {"kind": "impulse", "target": "spoon",
                               "variables": {"push_speed_m_s": 2.0}}},
        # the base is never a force target
        {"sim": SIM, "force": {"kind": "impulse", "target": "1700s table",
                               "variables": {"push_speed_m_s": 2.0}}},
        # push_speed outside (0, 20]
        {"sim": SIM, "force": {"kind": "impulse", "target": "plate",
                               "variables": {"push_speed_m_s": 25.0}}},
    ]
    for req in reqs:
        direct = cockpit._apply_force(json.loads(json.dumps(req)))
        status, via = _post(port, "/apply", req)
        assert status == 200
        assert json.dumps(via, sort_keys=True) == \
            json.dumps(direct, sort_keys=True), req


# ═══════════════════════════════════════════════════════════════════════════
# § 6. the OLD /live still answers through the cockpit; serve stays static
# ═══════════════════════════════════════════════════════════════════════════

def test_old_live_protocol_still_answers_through_cockpit(port):
    c = _Client(port, path="/live")
    try:
        c.send({"type": "start",
                "sim": {"items": ITEMS, "pushed_index": 1}})
        c.expect("building", timeout=30.0)
        r = c.expect("ready", timeout=120.0)
        assert r["labels"] == ITEMS
        fr = c.expect("frame", timeout=10.0)
        assert "bodies" in fr and len(fr["bodies"]) == 3
    finally:
        c.close()


def test_serve_still_boots_standalone_pure_static():
    import functools
    import http.server
    import os
    web_dir = os.path.dirname(os.path.abspath(serve.__file__))
    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(serve._Handler,
                                            directory=web_dir))
    p = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", p, timeout=10)
        conn.request("GET", "/")
        r = conn.getresponse()
        assert r.status == 200
        r.read()
        conn.close()
        # the front-door routes are honestly GONE from the static server
        conn = http.client.HTTPConnection("127.0.0.1", p, timeout=10)
        conn.request("POST", "/chat", body="{}",
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        assert r.status == 404
        r.read()
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


# ═══════════════════════════════════════════════════════════════════════════
# § 7. the intent channel (E3) — game verbs over the real /world socket
# ═══════════════════════════════════════════════════════════════════════════

def test_intent_world_verb_applies_over_socket(port):
    """A shove intent reduces SERVER-SIDE (engine.intents) and answers with
    the EXISTING applied envelope — same ticket path as raw apply."""
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "intent", "id": 11, "verb": "shove",
                "params": {"speed_m_s": 2.2}, "target": "plate"})
        ap = c.expect("applied", timeout=10.0)
        assert ap["id"] == 11
        assert ap["target"] == "plate"
        assert ap["force"]["kind"] == "impulse"
        assert ap["force"]["variables"]["push_speed_m_s"] == 2.2
        assert isinstance(ap["step_index"], int)
    finally:
        c.close()


def test_intent_engine_ops_and_camera_refusal(port):
    """pace/pause/resume intents run the EXISTING protocol ops (+id echoed
    additively); cam_* intents are refused honestly — the server has no
    camera. The connection and host loop survive all of it."""
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "intent", "id": 1, "verb": "pause"})
        m = c.expect("paused", timeout=10.0)
        assert m.get("id") == 1
        c.send({"type": "intent", "id": 2, "verb": "pace",
                "params": {"factor": 2.0}})
        m = c.expect("pace", timeout=10.0)
        assert m["factor"] == 2.0 and m.get("id") == 2
        c.send({"type": "intent", "id": 3, "verb": "resume"})
        m = c.expect("resumed", timeout=10.0)
        assert m.get("id") == 3
        # EVERY client-local verb (E3's cam_orbit + R2's cam_move/cam_mode/
        # follow) is refused via the ONE generic server_side=False arm —
        # pins that new camera verbs need ZERO cockpit source changes.
        cam_verbs = [
            ("cam_orbit", {"daz": 0.1}),
            ("cam_move", {"dz": 1.0}),
            ("cam_mode", {}),
            ("follow", {}),
        ]
        for k, (verb, params) in enumerate(cam_verbs):
            c.send({"type": "intent", "id": 4 + k, "verb": verb,
                    "params": params})
            ref = c.expect("refusal", timeout=10.0)
            assert ref["id"] == 4 + k, (verb, ref)
            assert "client-local" in ref["text"], (verb, ref)
        # liveness: the connection + host loop still answer
        c.send({"type": "hash", "id": 99})
        assert c.expect("hash", timeout=10.0)["id"] == 99
    finally:
        c.close()


def test_malformed_intents_never_kill_connection_or_host(port):
    """Every malformed intent gets a refusal envelope carrying its id; the
    NaN case crosses the wire as a REAL bare-NaN JSON payload (Python's
    json emits and parses it). Then a valid shove still applies — the
    liveness proof."""
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        bads = [
            {"type": "intent", "id": 1},                        # no verb
            {"type": "intent", "id": 2, "verb": "yeet"},
            {"type": "intent", "id": 3, "verb": "shove",
             "params": "fast", "target": "plate"},
            {"type": "intent", "id": 4, "verb": "shove",
             "params": {"speed_m_s": 25.0}, "target": "plate"},
            {"type": "intent", "id": 5, "verb": "shove",
             "params": {"speed_m_s": float("nan")}, "target": "plate"},
            {"type": "intent", "id": 6, "verb": "shove",
             "target": None},
            {"type": "intent", "id": 7, "verb": "tilt",
             "target": "plate"},
        ]
        for msg in bads:
            c.send(msg)
            ref = c.expect("refusal", timeout=10.0)
            assert ref["id"] == msg["id"], (msg, ref)
            assert ref["text"], msg
            if msg["id"] == 2:                      # names the closed set
                assert "shove" in ref["text"] and "cam_orbit" in ref["text"]
        c.send({"type": "intent", "id": 99, "verb": "shove",
                "target": "plate"})
        ap = c.expect("applied", timeout=10.0)      # liveness
        assert ap["id"] == 99
    finally:
        c.close()


def test_oversized_json_integer_intent_refuses_instead_of_killing_the_socket(port):
    """ADVERSARIAL REGRESSION. json.dumps writes a Python big int as a bare
    JSON integer literal, so ``{"speed_m_s": 10**400}`` crosses the wire as
    400 digits and reaches intents.reduce as an arbitrary-precision int.
    float() on one raises OverflowError — not an IntentRefusal — so it used
    to escape cockpit's single IntentRefusal arm and tear the /world
    connection down mid-frame instead of answering with a refusal."""
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        for i, msg in enumerate((
                {"verb": "shove", "target": "plate",
                 "params": {"speed_m_s": 10 ** 400}},
                {"verb": "press", "target": "plate",
                 "params": {"force_n": -(10 ** 400)}},
                {"verb": "pace", "params": {"factor": 10 ** 400}},
                {"verb": "tilt", "params": {"angle_deg": 10 ** 400,
                                            "duration_s": 2.0}})):
            c.send(dict(msg, type="intent", id=700 + i))
            ref = c.expect("refusal", timeout=10.0)
            assert ref["id"] == 700 + i, (msg, ref)
            assert "finite number" in ref["text"], ref
        # the SAME connection and the SAME world still work afterwards
        c.send({"type": "intent", "id": 799, "verb": "shove",
                "target": "plate"})
        assert c.expect("applied", timeout=10.0)["id"] == 799
    finally:
        c.close()


def test_unhandled_engine_op_refuses_loudly_and_changes_nothing(port,
                                                                monkeypatch):
    """A registry verb this server has no handler for must REFUSE, never
    silently do something else. The intent arm used to end in a bare
    ``else`` commented "resume — the only op left", so a seventh
    server-side engine verb would have been answered {"type":"resumed"} —
    a wrong action reported as success, and one that restarts a paused
    world. The fake verb is the E4-seam stand-in."""
    from sigma_ground.engine import intents
    monkeypatch.setitem(
        intents.ENGINE_VERBS, "teleport",
        intents.EngineVerb("teleport",
                           (intents.Param("x", default=0.0, lo=-1e3, hi=1e3),),
                           "an unhandled server-side engine verb",
                           continuous=False, op="teleport", server_side=True))
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "intent", "id": 1, "verb": "pause"})
        c.expect("paused", timeout=10.0)
        c.send({"type": "save", "id": 2})
        before = c.expect("save", timeout=10.0)["data"]
        c.send({"type": "intent", "id": 3, "verb": "teleport",
                "params": {"x": 500.0}})
        ref = c.expect("refusal", timeout=10.0)
        assert ref["id"] == 3, ref
        assert "teleport" in ref["text"] and "no handler" in ref["text"], ref
        c.send({"type": "save", "id": 4})
        after = c.expect("save", timeout=10.0)["data"]
        # still paused (not silently resumed) and physics untouched
        assert after["step_count"] == before["step_count"], (before, after)
        assert after["state_hash"] == before["state_hash"]
        assert after["command_log"] == before["command_log"]
    finally:
        c.close()


def test_intent_driven_save_round_trips_to_wire_hash(port):
    """An uncapped press flood (the 60 Hz cap is CLIENT politeness — the
    server takes the flood raw) cannot corrupt physics: every window is
    logged honestly, and the wire save replays to its embedded hash."""
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "intent", "id": 1, "verb": "shove",
                "target": "plate"})
        c.expect("applied", timeout=10.0)
        n = 0
        for _burst in range(4):                     # 48 presses, bursts of 12
            for _ in range(12):
                n += 1
                c.send({"type": "intent", "id": 100 + n, "verb": "press",
                        "params": {"force_n": 5.0}, "target": "plate"})
            for _ in range(12):
                c.expect("applied", timeout=10.0)
        c.send({"type": "pause"})
        c.expect("paused", timeout=10.0)
        c.send({"type": "save", "id": 9})
        data = c.expect("save", timeout=10.0)["data"]
    finally:
        c.close()
    assert len(data["command_log"]) == 49           # 1 shove + 48 presses
    loaded = WorldHost.load(json.loads(json.dumps(data)))
    assert loaded.state_hash() == data["state_hash"]


# ═══════════════════════════════════════════════════════════════════════════
# § E3-refutation hardening (R4): valid NON-OBJECT JSON frames must refuse,
#   never AttributeError the connection dead
# ═══════════════════════════════════════════════════════════════════════════

def test_non_object_json_frames_refused_not_fatal(port):
    name = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": name,
                "items": ["1700s table", "plate", "Doughnut"]})
        c.expect("ready", timeout=120.0)   # first attach pays build_stack
        # valid JSON, not an object — used to be AttributeError → dead socket
        c.send([1, 2, 3])
        r1 = c.expect("refusal")
        assert "JSON object" in r1["text"]
        c.send(42)
        r2 = c.expect("refusal")
        assert "JSON object" in r2["text"]
        # the connection genuinely survived: a real op still answers
        c.send({"type": "pause"})
        c.expect("paused")
    finally:
        c.close()


def test_non_object_json_attach_refused_not_fatal(port):
    c = _Client(port)
    try:
        c.send(42)                      # pre-attach: same hole, same guard
        r = c.expect("refusal")
        assert "attach" in r["text"]
    finally:
        c.close()


# ═══════════════════════════════════════════════════════════════════════════
# § 8. E4 — audio events over the wire + the pace-scaled press window
# ═══════════════════════════════════════════════════════════════════════════

def _collect_events(c, timeout=40.0, want=1):
    """Frames ride wall-clock stepping — generous deadline (CI-flake
    doctrine); returns every wire event seen."""
    out = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(out) < want:
        c.recv(timeout=0.2)
        for f in c.frames:
            for ev in f.get("events", []):
                if ev not in out:
                    out.append(ev)
    return out


def test_ready_audio_block_and_impact_events_over_socket(port):
    """attach → ready carries the audio disclosure; a hard shove sends the
    plate off the table; impact events arrive on the frame stream with
    server-computed physics: energy == 0.5*m*v² (m from the disclosed
    profile — server-side mass) and ring frequency matching an INDEPENDENT
    acoustics call on the profile's OWN mapped_key/size (never a
    hard-coded material)."""
    from sigma_ground.field.interface import acoustics
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        ready = c.expect("ready", timeout=120.0)
        audio = ready.get("audio")
        assert audio, "ready carries no audio block"
        profiles = audio["profiles"]
        assert set(profiles) == set(ITEMS)
        assert profiles["plate"]["synth"] == "ring"
        assert profiles["Doughnut"]["synth"] == "thud"      # honest gap
        assert profiles["Doughnut"]["frequency_hz"] is None
        assert "choices" in audio and "eta_acoustic" in audio["choices"]
        # the hard shove: plate leaves the table, falls, lands
        c.send({"type": "intent", "id": 1, "verb": "shove",
                "params": {"speed_m_s": 8.0}, "target": "plate"})
        c.expect("applied", timeout=10.0)
        events = _collect_events(c, timeout=40.0, want=1)
        assert events, "no impact events arrived on the stream"
        for ev in events:
            prof = profiles[ev["body"]]
            # energy: the server's Deduction, exact same op order
            assert ev["energy_j"] == \
                0.5 * prof["mass_kg"] * ev["impact_speed_m_s"] ** 2
            assert ev["acoustic_energy_j"] < ev["energy_j"]
            if ev["synth"]["kind"] == "ring":
                assert ev["synth"]["frequency_hz"] == \
                    acoustics.ring_frequency(prof["mapped_key"],
                                             prof["size_m"])
                assert ev["synth"]["q"] == prof["q"]
            else:
                assert ev["synth"]["frequency_hz"] is None
        # every frame that carried events is strict JSON
        for f in c.frames:
            json.dumps(f, allow_nan=False)
    finally:
        c.close()


def test_refused_intent_produces_no_events(port):
    """A refusal must leave the world untouched: zero events across a
    settled observation window after the refusal."""
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "intent", "id": 5, "verb": "yeet"})
        ref = c.expect("refusal", timeout=10.0)
        assert ref["id"] == 5
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            c.recv(timeout=0.2)
        assert len(c.frames) >= 2, "no observation window"
        for f in c.frames:
            assert "events" not in f, f
    finally:
        c.close()


def test_save_after_events_replays_hash_equal_and_log_event_free(port):
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "intent", "id": 1, "verb": "shove",
                "params": {"speed_m_s": 8.0}, "target": "plate"})
        c.expect("applied", timeout=10.0)
        assert _collect_events(c, timeout=40.0, want=1), "no events"
        c.send({"type": "pause"})
        c.expect("paused", timeout=10.0)
        c.send({"type": "hash", "id": 7})
        hv = c.expect("hash", timeout=10.0)
        c.send({"type": "save", "id": 8})
        data = c.expect("save", timeout=10.0)["data"]
    finally:
        c.close()
    # events never entered the log or the save
    assert "events" not in json.dumps(data["command_log"])
    loaded = WorldHost.load(json.loads(json.dumps(data)))
    assert loaded.state_hash() == hv["value"] == data["state_hash"]


def test_press_duration_scales_by_wire_pace(port):
    """{"type":"pace",factor:0.5} then a press intent → the applied
    ticket's force.duration == PRESS_TICK_S*0.5 EXACTLY (the E3 deferral
    fixed at reduce time; the log records what actually applied)."""
    from sigma_ground.engine.intents import PRESS_TICK_S
    w = _wname()
    c = _Client(port)
    try:
        c.send({"type": "attach", "world": w, "items": ITEMS})
        c.expect("ready", timeout=120.0)
        c.send({"type": "pace", "factor": 0.5})
        m = c.expect("pace", timeout=10.0)
        assert m["factor"] == 0.5
        c.send({"type": "intent", "id": 21, "verb": "press",
                "params": {"force_n": 5.0}, "target": "plate"})
        ap = c.expect("applied", timeout=10.0)
        assert ap["id"] == 21
        assert ap["force"]["duration"] == PRESS_TICK_S * 0.5
        # and the save's log carries the SAME scaled duration
        c.send({"type": "save", "id": 22})
        data = c.expect("save", timeout=10.0)["data"]
        press = [f for (_k, _t, f) in data["command_log"]
                 if f["kind"] == "force"]
        assert press and press[-1]["duration"] == PRESS_TICK_S * 0.5
    finally:
        c.close()
