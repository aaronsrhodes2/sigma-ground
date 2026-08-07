"""Serve the Radiance web viewer — standard library only, zero dependencies.

    python -m sigma_ground.radiance.web.serve   →   http://127.0.0.1:8765/

Sends no-cache headers so a browser reload always picks up fresh viewer.js /
data (essential while iterating).
"""
import base64
import functools
import hashlib
import http.server
import json
import os
import re
import select
import socketserver
import struct
import sys
import threading
import time
import urllib.parse
import uuid

# Make `sigma_ground` importable no matter where the server is launched from.
# web_dir is .../sigma_ground/radiance/web → the repo root is three levels up.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sigma_ground.radiance import png_io         # stdlib PNG codec (R0 capture sink)
from sigma_ground.radiance.web import library    # durable conversation store (Phase 0)


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
        if raw.path == "/live":
            self._live()                        # websocket — never returns here
            return
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
        # ── POST /perf : the dev-environment performance spigot (Captain,
        # 2026-08-06 — "add a performance monitoring spigot so you can read
        # that real-time"). viewer.js POSTs a compact sample roughly every
        # 500ms (fps + draw()'s own real, uncapped JS-side wall-clock cost —
        # NOT the vsync-gated presentation rate fps alone would give);
        # this sink keeps a short rolling history (last 60 samples, ~30s at
        # the default cadence) in misc/PERF_REPORT.json — a plain file, no
        # HTTP GET route needed, readable directly. Dev-environment only:
        # never shipped with the game, mirrors /probe's pattern exactly.
        if self.path.split("?")[0] == "/perf":
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 8192:                        # a perf sample is tiny; refuse the rest
                    self._send_json(400, {"error": "payload too large"})
                    return
                rep = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                rep = {}
            path = os.path.join(_ROOT, "misc", "PERF_REPORT.json")
            try:
                hist = json.load(open(path, encoding="utf-8"))
                if not isinstance(hist, list):
                    hist = []
            except Exception:
                hist = []
            hist.append(rep)
            hist = hist[-60:]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=1)
            self._send_json(200, {"ok": True, "count": len(hist)})
            return
        # ── POST /capture : the R0 pixel-regression sink (viewer.js capture
        # mode → tools/render_regression.py). Mirrors /probe's defensive
        # parse: exact Content-Length, 400 on malformed/oversized, id
        # sanitized to the slug charset, and the pixel payload NEVER logged
        # or forwarded into BROWSER_REPORT.json — it lands only as
        # misc/render_captures/<id>.png + <id>.json (metadata + sha256). ──
        if self.path.split("?")[0] == "/capture":
            try:
                n = int(self.headers.get("Content-Length", 0))
                # 64 MB cap: 640x360 RGBA base64 is ~1.2 MB; the cap leaves
                # headroom for a future high-res baseline without letting a
                # runaway client exhaust memory.
                if n <= 0 or n > 64 * 1024 * 1024:
                    self._send_json(400, {"error": f"bad capture size {n}"})
                    return
                rep = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._send_json(400, {"error": "malformed capture body"})
                return
            cid = str(rep.get("id") or "")
            if not re.fullmatch(r"[a-z0-9_]+", cid):    # slug-sanitization precedent
                self._send_json(400, {"error": "bad capture id"})
                return
            out_dir = os.path.join(_ROOT, "misc", "render_captures")
            os.makedirs(out_dir, exist_ok=True)
            if rep.get("error"):
                # the viewer refused honestly (e.g. no float render targets) —
                # record the refusal so the runner reports infrastructure, not
                # a silent timeout
                meta = {k: rep.get(k) for k in
                        ("id", "slug", "error", "renderer", "ua")}
                with open(os.path.join(out_dir, cid + ".json"), "w",
                          encoding="utf-8") as f:
                    json.dump(meta, f, indent=1)
                self._send_json(200, {"ok": True, "error_recorded": True})
                return
            try:
                w, h = int(rep["w"]), int(rep["h"])
                raw = base64.b64decode(rep.pop("rgba_b64"))
                if w <= 0 or h <= 0 or len(raw) != w * h * 4:
                    raise ValueError(f"rgba is {len(raw)} bytes, "
                                     f"expected {w}x{h}x4")
            except Exception as e:
                self._send_json(400, {"error": f"bad capture payload: {e}"})
                return
            # CANONICALIZE server-side: GL readPixels rows are bottom-up →
            # flip top-down and strip alpha. The canonical top-down RGB
            # stream is THE artifact (its sha256 is what the manifest pins);
            # the PNG is a filter-0 wrapper around exactly these bytes, so
            # every committed baseline is self-verifying.
            rows = []
            for row in range(h - 1, -1, -1):
                line = bytearray(raw[row * w * 4:(row + 1) * w * 4])
                del line[3::4]                       # strip alpha
                rows.append(bytes(line))
            rgb = b"".join(rows)
            png_io.write_rgb_png(os.path.join(out_dir, cid + ".png"), w, h, rgb)
            meta = dict(rep)
            meta["sha256_rgb"] = hashlib.sha256(rgb).hexdigest()
            with open(os.path.join(out_dir, cid + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
            self._send_json(200, {"ok": True, "bytes": len(rgb)})
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
        self.send_error(404, "POST here is /probe, /capture or /conversation; "
                             "/chat and /apply live on the cockpit "
                             "(python -m sigma_ground.mcp.cockpit)")


    # ── GET /live : the god-hand websocket (Interactive Simulation Phase I/II)
    def _live(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if (self.headers.get("Upgrade", "").lower() != "websocket"
                or not key):
            self.send_error(400, "expected a websocket upgrade")
            return
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
        try:
            _live_session(_WebSocket(self.connection))
        except (ConnectionError, BrokenPipeError, OSError):
            pass                                # client left — fine
        self.close_connection = True


class _WebSocket:
    """Minimal RFC 6455 server endpoint over an accepted socket — stdlib
    only, matching this server's zero-dependency doctrine. Text frames
    both ways; client frames are masked per the RFC; ping is ponged."""

    def __init__(self, sock):
        self.sock = sock
        self._buf = b""

    def send_json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        n = len(data)
        if n < 126:
            head = struct.pack("!BB", 0x81, n)
        elif n < 1 << 16:
            head = struct.pack("!BBH", 0x81, 126, n)
        else:
            head = struct.pack("!BBQ", 0x81, 127, n)
        self.sock.sendall(head + data)

    def _read_exact(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("socket closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv_json(self, timeout=0.0):
        """One JSON message, or None if nothing arrives within timeout /
        the client closed. Ping→pong and close are handled here."""
        while True:
            if not self._buf:
                r, _w, _x = select.select([self.sock], [], [], timeout)
                if not r:
                    return None
            b0, b1 = self._read_exact(2)
            opcode, masked, ln = b0 & 0x0F, b1 & 0x80, b1 & 0x7F
            if ln == 126:
                ln = struct.unpack("!H", self._read_exact(2))[0]
            elif ln == 127:
                ln = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else b"\x00" * 4
            payload = bytes(c ^ mask[i % 4]
                            for i, c in enumerate(self._read_exact(ln)))
            if opcode == 0x8:                   # close
                return None
            if opcode == 0x9:                   # ping → pong
                self.sock.sendall(struct.pack("!BB", 0x8A, len(payload))
                                  + payload)
                continue
            if opcode in (0x1, 0x2):
                try:
                    return json.loads(payload.decode("utf-8"))
                except Exception:
                    return {}
            # continuation/pong: ignore, keep reading within this call
            timeout = 0.0


def _live_session(ws):
    """The god-hand loop: step the REAL solver in real time, stream poses,
    accept artificial forces mid-run — the closed taxonomy only (Golden
    Rule 7). The sim runs forward indefinitely; 'settled' means cheap
    steps, never a stop. Physics is push_off.StackSim — the exact object
    the recorder bakes frames through; there is no second integrator.

    Live kinds: impulse, force, tilt, hold. SOURCE (heat) and DRIVE
    (joints) have no state in this planar stack model yet — refused
    honestly with the reason, never faked.
    """
    from sigma_ground.dynamics.mechanisms.applied import KINDS
    from sigma_ground.deckard.scene import build_stack
    from sigma_ground.radiance.push_off import StackSim, make_lookups

    LIVE_KINDS = {"impulse", "force", "tilt", "hold"}
    SIM_DT = 1.0 / 960.0
    SEND_DT = 1.0 / 30.0

    start = ws.recv_json(timeout=30.0)
    if not start or start.get("type") != "start":
        ws.send_json({"type": "refusal",
                      "text": "expected a start message with sim provenance"})
        return
    sim_prov = start.get("sim") or {}
    items = sim_prov.get("items")
    if not items:
        ws.send_json({"type": "refusal",
                      "text": "no sim provenance — go live from a scene "
                              "built by the push route"})
        return
    pushed_index = int(sim_prov.get("pushed_index", 1))
    ws.send_json({"type": "building",
                  "text": f"re-researching {', '.join(items)}…"})
    built = build_stack(list(items), allow_llm=False)
    mu, e = make_lookups(pushed_index)
    sim = StackSim(built, mu_lookup=mu, e_lookup=e)
    names = sim.names
    ws.send_json({"type": "ready", "labels": names,
                  "live_kinds": sorted(LIVE_KINDS),
                  "all_kinds": sorted(KINDS)})

    def handle(msg):
        if msg.get("type") == "pace":
            return max(0.01, min(10.0, float(msg.get("factor", 1.0))))
        if msg.get("type") != "apply":
            return None
        f = msg.get("force") or {}
        kind, target = f.get("kind"), f.get("target")
        v = f.get("variables") or {}
        if kind not in KINDS:
            ws.send_json({"type": "refusal",
                          "text": f"unknown kind {kind!r} — the set is "
                                  f"CLOSED: {sorted(KINDS)} (Golden Rule 7)"})
            return None
        if kind not in LIVE_KINDS:
            ws.send_json({"type": "refusal",
                          "text": f"'{kind}' is in the closed set, but this "
                                  f"stack model has no "
                                  f"{'thermal' if kind == 'source' else 'joint'}"
                                  f" state yet — live kinds today: "
                                  f"{sorted(LIVE_KINDS)}"})
            return None
        if kind == "tilt":
            idx, target = None, "the scene"     # tilt acts on the whole scene
        else:
            if target not in names or names.index(target) == 0:
                ws.send_json({"type": "refusal",
                              "text": f"target must be one of "
                                      f"{names[1:]} (the {names[0]} is "
                                      f"the base)"})
                return None
            idx = names.index(target)
        if kind == "impulse":
            sim.apply_impulse(idx, float(v.get("push_speed_m_s", 1.0)))
        elif kind == "force":
            sim.apply_force(idx, float(v.get("force_n", 1.0)),
                            float(v.get("duration_s", 1.0)))
        elif kind == "tilt":
            sim.apply_tilt(float(v.get("angle_deg", 10.0)),
                           float(v.get("duration_s", 2.0)))
        elif kind == "hold":
            sim.hold(idx, bool(v.get("on", True)))
        ws.send_json({"type": "applied", "kind": kind, "target": target,
                      "variables": v})
        return None

    pace = 1.0
    wall0, sim0 = time.monotonic(), sim.t
    next_send = 0.0
    while True:
        msg = ws.recv_json(timeout=0.0)
        if msg is None and not ws._buf:
            pass                                 # nothing waiting
        elif msg is None:
            return                               # closed
        else:
            newpace = handle(msg)
            if newpace is not None:
                pace = newpace
                wall0, sim0 = time.monotonic(), sim.t
        target_t = sim0 + (time.monotonic() - wall0) * pace
        steps = 0
        while sim.t < target_t and steps < 4000:  # bounded catch-up
            sim.step(SIM_DT)
            steps += 1
        if sim.t >= next_send:
            fr = sim.frame()
            fr["type"] = "frame"
            fr["support"] = [s if isinstance(s, str) or s is None
                             else names[s] for s in sim.support]
            ws.send_json(fr)
            next_send = sim.t + SEND_DT
        time.sleep(0.002)


def main(port: int = 8765):
    web_dir = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(_Handler, directory=web_dir)
    # Threaded: a slow dispatch (ollama latency, or the cold front-door import) must
    # NOT block the browser from fetching the page, chat.js, or the freshly-rendered
    # data JSON. ThreadingHTTPServer also sets allow_reuse_address + daemon threads.
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Radiance viewer  ->  http://127.0.0.1:{port}/   (no-cache)")
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
