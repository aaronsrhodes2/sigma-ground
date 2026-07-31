"""Serve the Radiance web viewer — standard library only, zero dependencies.

    python -m sigma_ground.radiance.web.serve   →   http://127.0.0.1:8765/

Sends no-cache headers so a browser reload always picks up fresh viewer.js /
data (essential while iterating).
"""
import functools
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.parse
import uuid

# Make `sigma_ground` importable no matter where the server is launched from, so
# the /chat endpoint can reach the Mentat front door. web_dir is
# .../sigma_ground/radiance/web → the repo root is three levels up.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sigma_ground.radiance.web import library    # durable conversation store (Phase 0)

# One front-door Session per browser tab (the client posts a stable session_id).
# In-process and ephemeral — a restart clears them. This threads multi-turn state
# so a bare "yes" renders the simulation the previous turn set up.
_SESSIONS = {}


class _Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):     # quiet, but show 404s
        if "404" in (fmt % args):
            super().log_message(fmt, *args)

    # ── POST /chat : the Mentat front door (text → ASK / SIMULATE / RENDER) ──
    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()                          # also sends the no-cache headers
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionError):
            pass                                    # client gave up mid-dispatch — fine

    # ── GET /library, /conversation : the durable history (Phase 0) ──
    def do_GET(self):
        raw = urllib.parse.urlparse(self.path)
        if raw.path == "/library":
            self._send_json(200, library.load_index())
            return
        if raw.path == "/conversation":
            cid = (urllib.parse.parse_qs(raw.query).get("id") or [""])[0]
            conv = library.load_conversation(cid)
            if conv is None:
                self._send_json(404, {"error": "no such conversation", "id": cid})
            else:
                self._send_json(200, conv)
            return
        super().do_GET()                            # everything else: static files

    # ── DELETE /conversation : forget a thread (its render bundles stay) ──
    def do_DELETE(self):
        raw = urllib.parse.urlparse(self.path)
        if raw.path == "/conversation":
            cid = (urllib.parse.parse_qs(raw.query).get("id") or [""])[0]
            library.delete_conversation(cid)
            self._send_json(200, {"ok": True, "id": cid})
            return
        self.send_error(404, "only /conversation accepts DELETE")

    def do_POST(self):
        # ── POST /probe : the browser gauntlet's verification sink ──
        if self.path.split("?")[0] == "/probe":
            try:
                n = int(self.headers.get("Content-Length", 0))
                rep = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                rep = {}
            path = os.path.join(_ROOT, "misc", "BROWSER_REPORT.json")
            try:
                cur = json.load(open(path, encoding="utf-8"))
            except Exception:
                cur = {}
            if rep.get("slug"):
                cur[rep["slug"]] = rep
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cur, f, indent=1)
            self._send_json(200, {"ok": True, "count": len(cur)})
            return
        # ── POST /conversation : create (the rail's "+ New") or rename ──
        if self.path.split("?")[0] == "/conversation":
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                self._send_json(400, {"error": f"bad request: {e}"})
                return
            cid = body.get("id") or uuid.uuid4().hex
            self._send_json(200, library.upsert_conversation(cid, title=body.get("title")))
            return
        # ── POST /apply : an artificial force from the closed taxonomy ──
        # The viewer's apply drawer posts {sim: <bundle.sim provenance>,
        # force: <AppliedForce dict>}. The force is validated through the
        # closed set (Golden Rule 7 — only artificial forces are adjustable
        # during a simulation); kinds without a landed solver get an honest
        # refusal JSON, never a fake run.
        if self.path.split("?")[0] == "/apply":
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                self._send_json(400, {"error": f"bad request: {e}"})
                return
            self._send_json(200, _apply_force(req))
            return
        if self.path.split("?")[0] != "/chat":
            self.send_error(404, "only /chat accepts POST")
            return
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
        # Lazy import: the static server still boots if Mentat/ollama aren't ready;
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


def main(port: int = 8765):
    web_dir = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(_Handler, directory=web_dir)
    # Threaded: a slow dispatch (ollama latency, or the cold front-door import) must
    # NOT block the browser from fetching the page, chat.js, or the freshly-rendered
    # data JSON. ThreadingHTTPServer also sets allow_reuse_address + daemon threads.
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=_warm, daemon=True).start()
    print(f"Radiance viewer  ->  http://127.0.0.1:{port}/   (no-cache, /chat live)")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
