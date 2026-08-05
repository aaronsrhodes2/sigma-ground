"""The cockpit — the tier-4 composed entry point (engine phase E2).

    python -m sigma_ground.mcp.cockpit [port]   →   http://127.0.0.1:8765/

Serves EVERYTHING radiance.web.serve serves (static viewer, /library,
/conversation, /probe, the OLD /live god-hand websocket — all inherited
untouched) PLUS the routes that need the Mentat front door: POST /chat and
POST /apply, moved here verbatim from serve.py. This retires the
grandfathered tier-3→4 layering exception: serve.py is pure static again,
and the composition of "web server + front door" happens at tier 4, where
composition belongs (tests/test_layering.py prescribed exactly this fix).

NEW in E2: GET /world — the generalized live protocol. Unlike the old
/live (one private StackSim per websocket, dead with the socket), /world
attaches to a NAMED world backed by ONE authoritative engine.host.WorldHost
that persists in the process across disconnects; later attaches join as
additional observers of the same world. Wire protocol (deliberately NOT the
old /live vocabulary — that endpoint still exists and still speaks it):

  C→S {"type":"attach","world":str, "items":[...]?, "pushed_index":int?,
       "load":{save-dict}?}
  C→S {"type":"apply","id"?,"target":str|null,"force":{AppliedForce wire
       dict}}     # target null ONLY for tilt (force dict targets "world")
  C→S {"type":"intent","id"?,"verb":str,"params":{...}?,"target":str|null?}
       # E3: game verbs (engine.intents' closed set). World verbs reduce
       # SERVER-SIDE to AppliedForce and answer with the EXISTING
       # "applied"/"refusal" ticket envelopes; pace/pause/resume answer
       # with their existing replies (+"id" echoed additively when the
       # intent carried one); cam_* verbs are refused — the server has no
       # camera (camera state is client-local, input.js's engine channel).
       # Raw "apply" stays byte-untouched: the wire-level escape hatch.
  C→S {"type":"pace","factor":float} | {"type":"pause"} | {"type":"resume"}
  C→S {"type":"save","id"?} | {"type":"hash","id"?} | {"type":"detach"}
  S→C {"type":"ready",...} → one hello frame → broadcast frames
       # E4: ready carries an ADDITIVE "audio" block (per-body synthesis
       # profiles + every tuning Choice with its reason — engine/
       # audio_events.audio_disclosure). Broadcast frames gain ADDITIVE
       # keys when audio events occurred since the last emit:
       #   "events": [wire event dicts]   (impact/bounce; server-computed
       #             energy/frequency/Q — the client is a dumb synth)
       #   "events_dropped": int          (flood-capped excess, counted)
       #   "events_error": str            (one-shot tap breaker, disclosed)
       # Quiet frames are byte-identical to R2's. Events never enter the
       # command log or any save. The cockpit relays frames VERBATIM.
  S→C {"type":"applied","id","step_index","target","force"}
  S→C {"type":"refusal","id"?,"text"} | {"type":"pace","factor"}
  S→C {"type":"paused"} | {"type":"resumed"}
  S→C {"type":"save","id","data":{...}} | {"type":"hash","id","value":str}

SEAM RULE: the host loop never touches a socket; a handler thread never
touches physics — host.submit/pause/resume/set_pace/request_* are the only
crossings. E2 simplification (disclosed): the world registry has no idle
reaping — worlds persist for the process lifetime; the registry lock is
held across build_stack on first attach (seconds), which also kills the
double-first-attach race.
"""
import base64
import functools
import hashlib
import http.server
import json
import os
import queue
import sys
import threading
import urllib.parse
import uuid

from ..dynamics.mechanisms.applied import KINDS
from ..engine import intents
from ..engine.host import WorldHost
from ..engine.region import STACK_LIVE_KINDS
from ..radiance.web import library, serve

# One front-door Session per browser tab (the client posts a stable session_id).
# In-process and ephemeral — a restart clears them. This threads multi-turn state
# so a bare "yes" renders the simulation the previous turn set up.
_SESSIONS = {}

# The world registry: name -> the ONE authoritative WorldHost. Worlds persist
# across disconnects for the process lifetime (no idle reaping in E2).
_REGISTRY: dict = {}
_REG_LOCK = threading.Lock()


def _apply_force(req: dict) -> dict:
    """Run one AppliedForce against a bundle's sim provenance.

    Returns a /chat-shaped envelope so the viewer handles both identically.
    Validation is the taxonomy's own (__post_init__) — the closed set is
    enforced in exactly one place, and an out-of-set kind or a kind whose
    solver hasn't landed refuses with the same honesty as the chat route.
    """
    from sigma_ground.dynamics.mechanisms.applied import KINDS
    from sigma_ground.mcp.front_door import _render_push, _IMPLEMENTED_KINDS
    sim = req.get("sim") or {}
    # The drawer posts a force REQUEST (kind + target + variables), not the
    # finished AppliedForce — the viewer can't know J = m*v0 because matter
    # (the mass) lives server-side. The recorder constructs the real,
    # disclosed AppliedForce from the actual mass; here we validate the
    # request against the same closed set.
    force = req.get("force") or {}
    kind = force.get("kind")
    if kind not in KINDS:
        return {"intent": "clarify", "error": True, "can_render": False,
                "text": (f"❓ unknown artificial-force kind {kind!r} — the "
                         f"set is CLOSED: {sorted(KINDS)}. A new kind is a "
                         f"Golden Rule 7 doctrine amendment, not a "
                         f"parameter.")}
    if kind not in _IMPLEMENTED_KINDS:
        return {"intent": "clarify", "can_render": False,
                "text": (f"❓ '{kind}' is a defined member of the closed "
                         f"artificial-force set {sorted(KINDS)} (Golden Rule "
                         f"7), but its solver hasn't landed yet — today only "
                         f"'impulse' runs.")}
    if sim.get("builder") != "push_off" or not sim.get("items"):
        return {"intent": "clarify", "can_render": False,
                "text": "❓ this scene carries no sim provenance to re-run — "
                        "apply works on simulations built by the push route."}
    items = list(sim["items"])
    target = force.get("target")
    if target not in items:
        return {"intent": "clarify", "can_render": False,
                "text": f"❓ '{target}' isn't in this scene "
                        f"({', '.join(items)})."}
    idx = items.index(target)
    if idx == 0:
        return {"intent": "clarify", "can_render": False,
                "text": f"❓ the {items[0]} is the base everything stands on — "
                        f"pushing it is a tipping simulation this route "
                        f"honestly doesn't model yet."}
    v = (force.get("variables") or {}).get("push_speed_m_s")
    if not isinstance(v, (int, float)) or not 0 < v <= 20:
        return {"intent": "clarify", "can_render": False,
                "text": "❓ an impulse needs variables.push_speed_m_s in "
                        "(0, 20] — the one during-sim knob."}
    return _render_push(items, idx, use_llm=False, push_speed=float(v))


def _warm():
    """Pre-import the Mentat front door so the first /chat isn't a cold-import wait."""
    try:
        import sigma_ground.mcp.front_door  # noqa: F401
    except Exception:
        pass


def shutdown_worlds():
    """Stop every registered world's loop and clear the registry.

    Used by main()'s finally AND test teardown."""
    with _REG_LOCK:
        hosts = list(_REGISTRY.values())
        _REGISTRY.clear()
        for host in hosts:
            host.stop()


def _get_or_create_world(name, items, pushed_index, load):
    """(host, created, refusal_text) — the registry's single entry point.

    Held under _REG_LOCK across build_stack (seconds on first attach) —
    disclosed E2 simplification that kills the double-first-attach race.
    Mismatched items on join are REFUSED, and so is a ``load`` naming an
    EXISTING world: silently handing a client a different world (or state)
    than it asked for would be a disclosure failure. Join with no items and
    no load always succeeds.
    """
    with _REG_LOCK:
        host = _REGISTRY.get(name)
        if host is not None:
            if load is not None:
                return None, False, (
                    f"world {name!r} already exists — refusing the load: "
                    f"restoring a save can neither replace the live world "
                    f"nor be silently ignored (that would hand you a "
                    f"different state than you asked for). Attach with no "
                    f"load to join the live world, or restore the save "
                    f"under a new world name.")
            if items is not None and list(items) != list(host.scenario["items"]):
                return None, False, (
                    f"world {name!r} already exists with items "
                    f"{host.scenario['items']} — attach with no items to "
                    f"join it, or pick a new world name (mismatched items "
                    f"are refused, never silently swapped)")
            return host, False, None
        try:
            if load is not None:
                host = WorldHost.load(load)
            elif items:
                host = WorldHost.from_items(list(items),
                                            pushed_index=int(pushed_index))
            else:
                return None, False, (
                    f"no world named {name!r} — the first attach must carry "
                    f"items (to build one) or load (to restore one)")
        except Exception as e:
            return None, False, f"could not create world {name!r}: {e}"
        host.start()
        _REGISTRY[name] = host
        return host, True, None


def _world_session(ws):
    """One observer's connection to a named persistent world.

    The handler thread relays: broadcast frames from the host's observer
    queue go out; commands go in through host.submit (tickets answered
    per-socket). Physics is never touched here — SEAM RULE in the module
    docstring. The hello frame is PER-CLIENT, outside the broadcast
    sequence (a paused world still renders on attach); frame streams are
    compared by "step", never by naive position.
    """
    attach = ws.recv_json(timeout=30.0)
    # isinstance FIRST: recv_json returns whatever json.loads produced, and
    # VALID non-object JSON (a bare 42, [1,2,3]) has no .get — before this
    # guard it raised AttributeError and killed the connection
    # (adversarially proven, E3's refutation pass, finding R4).
    if not isinstance(attach, dict) or attach.get("type") != "attach" \
            or not attach.get("world"):
        ws.send_json({"type": "refusal",
                      "text": "expected an attach message naming a world: "
                              '{"type":"attach","world":...,"items":[...]?}'})
        return
    name = str(attach["world"])
    host, created, err = _get_or_create_world(
        name, attach.get("items"), attach.get("pushed_index", 1),
        attach.get("load"))
    if err:
        ws.send_json({"type": "refusal", "text": err})
        return
    ready = {"type": "ready", "world": name, "created": created,
             "labels": list(host.region.sim.names),
             "live_kinds": sorted(STACK_LIVE_KINDS),
             "all_kinds": sorted(KINDS),
             "dt": host.dt, "step": host.step_count,
             "paused": host.paused, "pace": host.pace}
    if getattr(host, "_audio_profiles", None) is not None:
        # E4 additive: the server-side synthesis disclosure (profiles +
        # Choices). A host without profiles simply omits the block.
        from ..engine import audio_events
        ready["audio"] = audio_events.audio_disclosure(host._audio_profiles)
    ws.send_json(ready)
    ws.send_json(host.snapshot())               # the hello / catch-up frame
    # E3: the game-verb reduction needs the mass table (an impulse's vector
    # discloses J = m·v — the client can't know it; matter lives
    # server-side). Construction-fixed floats, the same read class as the
    # ready message's sim.names read; the SEAM RULE (handler threads never
    # MUTATE physics) is intact. If an E5 backend ever mutates mass mid-run
    # this must move behind a host accessor.
    sim = host.region.sim
    masses = {sim.names[i]: float(sim.mass[i]) for i in range(sim.n)}
    q = host.subscribe()
    pending = []
    try:
        while True:
            # 1. the broadcast stream
            while True:
                try:
                    ws.send_json(q.get_nowait())
                except queue.Empty:
                    break
            # 2. per-socket addressed replies for resolved tickets
            for cmd in [c for c in pending if c.done.is_set()]:
                pending.remove(cmd)
                r = cmd.result
                if r.get("ok"):
                    out = {"type": "applied", "id": cmd.id}
                    out.update((k, v) for k, v in r.items() if k != "ok")
                    ws.send_json(out)
                else:
                    ws.send_json({"type": "refusal", "id": cmd.id,
                                  "text": r.get("refusal")})
            # 3. client messages
            msg = ws.recv_json(timeout=0.02)
            if msg is None:
                continue        # timeout (or a close frame; a real close
                #                 surfaces as ConnectionError on the next
                #                 recv/send and unwinds through finally)
            if not isinstance(msg, dict):
                # valid non-object JSON (42, [1,2,3]) has no .get —
                # refuse honestly instead of AttributeError killing the
                # connection (E3 refutation finding R4)
                ws.send_json({"type": "refusal",
                              "text": f"messages must be JSON objects, got "
                                      f"{type(msg).__name__}"})
                continue
            mtype = msg.get("type")
            if mtype == "apply":
                pending.append(host.submit(msg.get("target"),
                                           msg.get("force") or {},
                                           id=msg.get("id")))
            elif mtype == "intent":
                # intents.reduce owns ALL message validation, so this one
                # IntentRefusal arm keeps the connection and the host loop
                # unkillable by any malformed intent.
                try:
                    # pace=host.pace (the _ctl-locked property, same
                    # handler-thread read class as ready's host.paused):
                    # E4's press-window fix — tick_scaled verbs get
                    # duration = PRESS_TICK_S * pace. Disclosed: a pace
                    # change racing between reduce and boundary-apply
                    # scales by the reduce-time value; the log records
                    # what actually applied.
                    red = intents.reduce(msg.get("verb"), msg.get("params"),
                                         target=msg.get("target"),
                                         masses=masses, pace=host.pace)
                except intents.IntentRefusal as e:
                    ws.send_json({"type": "refusal", "id": msg.get("id"),
                                  "text": str(e)})
                else:
                    if red["channel"] == "world":
                        pending.append(host.submit(red["target"],
                                                   red["force"],
                                                   id=msg.get("id")))
                    elif not red["server_side"]:
                        ws.send_json({
                            "type": "refusal", "id": msg.get("id"),
                            "text": (f"{red['verb']!r} is a camera verb — "
                                     f"the server has no camera; camera "
                                     f"state is client-local (route it "
                                     f"through input.js's engine channel). "
                                     f"Nothing was applied.")})
                    elif red["op"] == "pace":
                        out = {"type": "pace",
                               "factor": host.set_pace(
                                   red["args"]["factor"])}
                        if msg.get("id") is not None:
                            out["id"] = msg["id"]
                        ws.send_json(out)
                    elif red["op"] == "pause":
                        host.pause()
                        out = {"type": "paused"}
                        if msg.get("id") is not None:
                            out["id"] = msg["id"]
                        ws.send_json(out)
                    elif red["op"] == "resume":
                        host.resume()
                        out = {"type": "resumed"}
                        if msg.get("id") is not None:
                            out["id"] = msg["id"]
                        ws.send_json(out)
                    else:
                        # An op this server has no handler for. Previously
                        # this arm was a bare else commented "resume — the
                        # only op left": a registry gaining a seventh
                        # server-side engine verb would have been silently
                        # RESUMED (a wrong action reported as success), the
                        # exact silent no-op the closed-set doctrine
                        # forbids. Mismatches refuse LOUDLY.
                        ws.send_json({
                            "type": "refusal", "id": msg.get("id"),
                            "text": (f"{red['verb']!r} reduced to engine op "
                                     f"{red['op']!r}, which this server has "
                                     f"no handler for — refusing rather than "
                                     f"guessing which command you meant. "
                                     f"Nothing was applied and nothing "
                                     f"changed.")})
            elif mtype == "pace":
                ws.send_json({"type": "pace",
                              "factor": host.set_pace(msg.get("factor", 1.0))})
            elif mtype == "pause":
                host.pause()
                ws.send_json({"type": "paused"})
            elif mtype == "resume":
                host.resume()
                ws.send_json({"type": "resumed"})
            elif mtype == "save":
                data = host.request_save().wait(5.0)["save"]
                ws.send_json({"type": "save", "id": msg.get("id"),
                              "data": data})
            elif mtype == "hash":
                value = host.request_hash().wait(5.0)["hash"]
                ws.send_json({"type": "hash", "id": msg.get("id"),
                              "value": value})
            elif mtype == "detach":
                return          # the world PERSISTS
            else:
                ws.send_json({"type": "refusal", "id": msg.get("id"),
                              "text": f"unknown message type {mtype!r} — "
                                      f"apply / intent / pace / pause / "
                                      f"resume / save / hash / detach"})
    finally:
        host.unsubscribe(q)


class _CockpitHandler(serve._Handler):
    """serve's handler + the front-door routes + the /world live protocol.

    Everything not handled here is inherited untouched: static files, the
    OLD /live loop, /library, /conversation (GET/POST/DELETE), /probe.
    """

    # ── GET /world : the generalized live websocket ──
    def do_GET(self):
        if urllib.parse.urlparse(self.path).path == "/world":
            ws = self._ws_upgrade()
            if ws is not None:
                try:
                    _world_session(ws)
                except (ConnectionError, BrokenPipeError, OSError):
                    pass                        # client left — fine
                self.close_connection = True
            return
        super().do_GET()

    def _ws_upgrade(self):
        # The RFC 6455 handshake — duplicated from serve._Handler._live
        # (serve.py stays byte-stable this phase, so no shared-helper
        # refactor; disclosed copy).
        key = self.headers.get("Sec-WebSocket-Key")
        if (self.headers.get("Upgrade", "").lower() != "websocket"
                or not key):
            self.send_error(400, "expected a websocket upgrade")
            return None
        accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
        ).digest()).decode()
        self.protocol_version = "HTTP/1.1"     # RFC 6455 needs a 1.1 status line
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.wfile.flush()
        return serve._WebSocket(self.connection)

    def do_POST(self):
        path = self.path.split("?")[0]
        # ── POST /apply : an artificial force from the closed taxonomy ──
        # The viewer's apply drawer posts {sim: <bundle.sim provenance>,
        # force: <AppliedForce dict>}. The force is validated through the
        # closed set (Golden Rule 7 — only artificial forces are adjustable
        # during a simulation); kinds without a landed solver get an honest
        # refusal JSON, never a fake run.
        if path == "/apply":
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                self._send_json(400, {"error": f"bad request: {e}"})
                return
            self._send_json(200, _apply_force(req))
            return
        if path != "/chat":
            super().do_POST()                   # /probe, /conversation, 404
            return
        # ── POST /chat : the Mentat front door (text → ASK / SIMULATE / RENDER) ──
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send_json(400, {"error": f"bad request: {e}"})
            return
        text = (req.get("text") or "").strip()
        sid = req.get("session_id") or uuid.uuid4().hex
        use_llm = bool(req.get("use_llm", True))
        mode = req.get("mode")                       # None/auto | ask | simulate | render
        # Lazy import: the cockpit still boots if Mentat/ollama aren't ready;
        # the first chat request surfaces any problem instead of crashing launch.
        from sigma_ground.mcp.front_door import dispatch, Session
        sess = _SESSIONS.setdefault(sid, Session())
        try:
            env = dispatch(text, use_llm=use_llm, session=sess, mode=mode)
        except Exception as e:
            # ollama down / model missing → retry the deterministic path and flag it,
            # so ASK/SIM and catalogued RENDER still work, just without the qwen residual.
            try:
                env = dict(dispatch(text, use_llm=False, session=sess, mode=mode))
                env["degraded"], env["degraded_reason"] = True, str(e)[:200]
            except Exception as e2:
                self._send_json(200, {"intent": "error", "error": True,
                                      "text": f"Mentat backend error: {e2}",
                                      "session_id": sid})
                return
        env = dict(env)
        env["session_id"] = sid
        # Phase 0: append this exchange to the durable store so the thread
        # survives a reload. Persistence must never break the reply.
        if text:
            try:
                library.record_turn(sid, user_text=text, mode=mode, envelope=env)
            except Exception:
                pass
        self._send_json(200, env)


def make_server(port: int) -> http.server.ThreadingHTTPServer:
    """Bind the cockpit on 127.0.0.1:port. Port 0 is legal and IS the test
    seam (stale-:8765 doctrine: never assume the port is free — bind
    ephemeral and read the real port back from server_address)."""
    web_dir = os.path.dirname(os.path.abspath(serve.__file__))
    handler = functools.partial(_CockpitHandler, directory=web_dir)
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(port: int = 8765):
    httpd = make_server(port)
    threading.Thread(target=_warm, daemon=True).start()
    real_port = httpd.server_address[1]
    print(f"Sigma-ground cockpit  ->  http://127.0.0.1:{real_port}/   "
          f"(/chat, /apply, /world live)")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
        shutdown_worlds()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
