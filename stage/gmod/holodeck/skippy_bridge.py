"""Skippy's bridge: the small local service that lets the holodeck in Garry's Mod use an agent and a voice.

    python stage/gmod/holodeck/skippy_bridge.py [--port 8777] [--model qwen3:8b] [--tts sapi|auto|elevenlabs]

It runs on the machine that hosts the game server (or the game, for single player) and offers FIXED actions only, to localhost,
behind an optional shared token (PLATINUM §19: nothing here runs a command, edits a file or reaches beyond the three jobs):

    GET  /health
    POST /plan   {"text": "make it cozy", "nearby": ["couch", "lamp"]}
                 -> {"ok": true, "request": {action, items, arrangement, target, scene}, "say": "...", "notes": [...]}
         The agent (a local Ollama model) turns a vague wish into a holodeck request, using ONLY the kinds in the library. Every
         field is validated against the catalog: an invented kind, arrangement or count never reaches the game.
    POST /reply  {"user": "...", "did": "three couches in a U shape", "notes": []}
                 -> {"ok": true, "say": "<one or two short in-character sentences>"}
    POST /speak  {"text": "..."}  -> audio/mpeg bytes (header X-Skippy-Voice says which voice made it)

Voices: `sapi` (Windows speech, free and local; the default), `elevenlabs` (Skippy's own voice through SkippyTel's /tts/stream,
which costs API credit per character), `auto` (ElevenLabs when SkippyTel has it configured, else SAPI).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOG = json.loads((HERE / "data" / "catalog.json").read_text(encoding="utf-8"))
KINDS: dict = CATALOG["kinds"]
RECIPES: dict = CATALOG["recipes"]

ACTIONS = ["add", "remove", "clear", "undo", "move", "rotate", "rearrange", "none"]
ARRANGEMENTS = ["unspecified", "row", "circle", "around", "facing_each_other", "facing_me", "u_shape", "l_shape", "grid",
                "against_wall", "beside", "in_front_of", "behind", "scattered", "stacked", "corner", "on"]
WORLD_TARGETS = ["", "me", "here", "wall", "corner", "last"]

OLLAMA = os.environ.get("SKIPPY_OLLAMA", "http://127.0.0.1:11434")
MODEL = os.environ.get("SKIPPY_MODEL", "qwen3:8b")
SKIPPYTEL = os.environ.get("SKIPPY_TEL", "http://127.0.0.1:3003")
PERSONA_PATH = Path(os.environ.get("SKIPPY_PERSONA", "D:/Aaron/development/SkippyTel/services/flask-api/app/skippy_persona.md"))
TTS_MODE = os.environ.get("SKIPPY_TTS", "sapi")
SAPI_VOICE = os.environ.get("SKIPPY_SAPI_VOICE", "Microsoft Zira Desktop")
MAX_SPEAK_CHARS = 400


# --------------------------------------------------------------------------------------------- the agent's instructions
def kind_lines() -> str:
    out = []
    for kid, k in KINDS.items():
        extra = [n for n in k["names"] if n != kid][:4]
        out.append(f"- {kid}" + (f" (also: {', '.join(extra)})" if extra else "") + f" [{k['category']}]")
    return "\n".join(out)


def system_prompt() -> str:
    scenes = ", ".join(RECIPES)
    return f"""You are the planner inside a holodeck in a video game. A player says what they want in plain words; you answer with ONE JSON
object that the game turns into furniture and props. You never ask questions: you always choose the most sensible reading.

The game can build ONLY these kinds of things (use the first name exactly as written):
{kind_lines()}

Whole rooms the game knows how to furnish (put the name in "scene"): {scenes}.

JSON fields:
  "action": one of {", ".join(ACTIONS)}.  Use "add" to summon things, "remove" to take things away, "clear" to wipe everything,
            "undo" for the last change, "move"/"rotate"/"rearrange" to change what already stands, "none" when nothing should happen.
  "items": a list of {{"kind": <a kind above>, "count": <1-40>, "adjectives": [colour or size words the player used]}}.
  "arrangement": one of {", ".join(ARRANGEMENTS)} ("unspecified" lets the game choose).
  "target": "" or "me" or "wall" or "corner" or a kind above (what the arrangement is relative to, e.g. chairs "around" a table).
  "scene": "" or one of the room names above.
  "say": ONE short sentence telling the player what you are doing (or why you cannot).

Rules: never invent a kind that is not listed; pick the nearest listed kind instead and say so. A vague wish ("make it cozy", "a place to relax",
"something to eat at", "an office for a detective") becomes a sensible set of listed kinds or a scene. Counts the player gives are exact.
Output JSON only."""


FEW_SHOT = [
    ("make it cozy", {"action": "add", "items": [{"kind": "armchair", "count": 2, "adjectives": []}, {"kind": "floor lamp", "count": 1, "adjectives": []},
                                                 {"kind": "plant", "count": 1, "adjectives": []}, {"kind": "coffee table", "count": 1, "adjectives": []}],
                      "arrangement": "unspecified", "target": "", "scene": "", "say": "Two armchairs, a lamp, a plant and a low table, nice and snug."}),
    ("somewhere to eat", {"action": "add", "items": [], "arrangement": "unspecified", "target": "", "scene": "dining room",
                          "say": "A dining room, coming up."}),
    ("a dragon", {"action": "none", "items": [], "arrangement": "unspecified", "target": "", "scene": "",
                  "say": "I have no dragon in my library, but I can furnish a lair with crates and barrels if you like."}),
]


def response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ACTIONS},
            "items": {"type": "array", "items": {"type": "object", "properties": {
                "kind": {"type": "string"}, "count": {"type": "integer"}, "adjectives": {"type": "array", "items": {"type": "string"}}},
                "required": ["kind", "count"]}},
            "arrangement": {"type": "string", "enum": ARRANGEMENTS},
            "target": {"type": "string"},
            "scene": {"type": "string"},
            "say": {"type": "string"},
        },
        "required": ["action", "say"],
    }


# --------------------------------------------------------------------------------------------- validation: the agent proposes, the catalog decides
def resolve_kind(name: str) -> str | None:
    n = str(name or "").strip().lower()
    if n in KINDS:
        return n
    for kid, k in KINDS.items():
        if n in [x.lower() for x in k["names"]] or n in [x.lower() for x in k.get("approx", [])]:
            return kid
    if n.endswith("s") and n[:-1] in KINDS:
        return n[:-1]
    return None


def validate_request(obj: dict) -> tuple[dict, list[str]]:
    """Make whatever the agent returned a request the game can run, dropping what it invented (and saying so)."""
    notes: list[str] = []
    action = obj.get("action") if obj.get("action") in ACTIONS else "none"
    items = []
    for it in obj.get("items") or []:
        if not isinstance(it, dict):
            continue
        kid = resolve_kind(it.get("kind"))
        if not kid:
            notes.append(f"I do not have a {it.get('kind')} in my library")
            continue
        try:
            n = int(it.get("count", 1))
        except (TypeError, ValueError):
            n = 1
        adjs = [str(a).lower() for a in (it.get("adjectives") or []) if isinstance(a, (str, int))][:4]
        items.append({"kind": kid, "count": max(1, min(40, n)), "adjectives": adjs, "word": kid})
    arrangement = obj.get("arrangement") if obj.get("arrangement") in ARRANGEMENTS else "unspecified"
    target = str(obj.get("target") or "").strip().lower()
    if target and target not in WORLD_TARGETS:
        target = resolve_kind(target) or ""
    scene = str(obj.get("scene") or "").strip().lower()
    if scene not in RECIPES:
        scene = ""
    if scene:
        # a room already brings its own furniture: do not double it with items the agent listed from the same room
        in_scene = {kind for _, kind, _ in RECIPES[scene]}
        items = [i for i in items if i["kind"] not in in_scene]
    if action == "add" and not items and not scene:
        action = "none"
    req = {"action": action, "items": items, "arrangement": arrangement, "target": target, "scene": scene}
    return req, notes


# --------------------------------------------------------------------------------------------- the model
def ollama_chat(messages: list[dict], schema: dict | None, model: str = MODEL, timeout: float = 40.0, temperature: float = 0.2) -> str:
    body = {"model": model, "messages": messages, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": 600}}
    if schema:
        body["format"] = schema
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["message"]["content"]


chat = ollama_chat          # replaced in tests


def plan(text: str, nearby: list[str] | None = None) -> dict:
    messages = [{"role": "system", "content": system_prompt()}]
    for q, a in FEW_SHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": json.dumps(a)})
    ctx = f" (already standing nearby: {', '.join(nearby)})" if nearby else ""
    messages.append({"role": "user", "content": text + ctx})
    raw = chat(messages, response_schema())
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group(0)) if m else {"action": "none", "say": "I could not make sense of that."}
    req, notes = validate_request(obj)
    return {"ok": True, "request": req, "say": str(obj.get("say") or "")[:300], "notes": notes}


def persona_text() -> str:
    try:
        return PERSONA_PATH.read_text(encoding="utf-8")[:2600]
    except OSError:
        return "You are Skippy the Magnificent, an ancient and sarcastic AI who is secretly loyal to the player. Speak in one or two short sentences."


def reply(user: str, did: str, notes: list[str] | None = None) -> str:
    facts = did + ((" Also: " + "; ".join(notes)) if notes else "")
    messages = [
        {"role": "system", "content": persona_text() + "\n\nYou are speaking out loud inside a video game holodeck you control. Say, in character, in ONE or TWO short "
                                                       "sentences (under 35 words, no markdown, no lists), what you just did for the player. Stay truthful: say only what the FACTS say."},
        {"role": "user", "content": f"The player said: {user}\nFACTS: {facts}"},
    ]
    text = chat(messages, None, temperature=0.8).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"\*[^*]{1,40}\*", "", text)               # stage directions like *sigh* are not for speaking
    return re.sub(r"\s+", " ", text.replace("*", "")).strip()[:300] or did


# --------------------------------------------------------------------------------------------- the voice
_tts_cache: dict[str, tuple[bytes, str]] = {}


def _sapi_wav(text: str, out: Path) -> None:
    txt = out.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"try {{ $s.SelectVoice('{SAPI_VOICE}') }} catch {{}};"
        f"$s.SetOutputToWaveFile('{out}');"
        f"$s.Speak([IO.File]::ReadAllText('{txt}', [Text.Encoding]::UTF8));$s.Dispose()"
    )
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], check=True, capture_output=True, timeout=60)
    txt.unlink(missing_ok=True)


def to_mp3(wav: Path, mp3: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-ar", "22050", "-b:a", "40k", str(mp3)], check=True, timeout=60)


def speak_sapi(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        wav, mp3 = Path(d) / "s.wav", Path(d) / "s.mp3"
        _sapi_wav(text, wav)
        to_mp3(wav, mp3)
        return mp3.read_bytes()


def speak_elevenlabs(text: str) -> bytes:
    url = f"{SKIPPYTEL}/tts/stream?text=" + urllib.parse.quote(text)
    with urllib.request.urlopen(url, timeout=25) as r:
        data = r.read()
    if len(data) < 1000:
        raise RuntimeError("SkippyTel returned no audio")
    return data


def speak(text: str, mode: str | None = None) -> tuple[bytes, str]:
    text = re.sub(r"\s+", " ", text).strip()[:MAX_SPEAK_CHARS]
    mode = mode or TTS_MODE
    key = hashlib.sha1(f"{mode}|{SAPI_VOICE}|{text}".encode()).hexdigest()
    if key in _tts_cache:
        return _tts_cache[key]
    voice = "sapi"
    data = b""
    if mode in ("elevenlabs", "auto"):
        try:
            data, voice = speak_elevenlabs(text), "elevenlabs"
        except Exception as e:
            if mode == "elevenlabs":
                raise
            print(f"[bridge] ElevenLabs unavailable ({e}); using SAPI", file=sys.stderr)
    if not data:
        data, voice = speak_sapi(text), "sapi"
    _tts_cache[key] = (data, voice)
    if len(_tts_cache) > 200:
        _tts_cache.pop(next(iter(_tts_cache)))
    return data, voice


# --------------------------------------------------------------------------------------------- the server
TOKEN = os.environ.get("SKIPPY_BRIDGE_TOKEN", "")


class Handler(BaseHTTPRequestHandler):
    server_version = "SkippyBridge/1"

    def log_message(self, fmt, *args):                 # quiet
        if os.environ.get("SKIPPY_BRIDGE_VERBOSE"):
            sys.stderr.write("[bridge] " + fmt % args + "\n")

    def _authorised(self) -> bool:
        return not TOKEN or self.headers.get("X-Skippy-Token", "") == TOKEN

    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
        try:
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, TimeoutError):
            pass                                          # the game gave up waiting; nothing to tell it

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ok": True, "model": MODEL, "tts": TTS_MODE, "kinds": len(KINDS)})
        self._json(404, {"ok": False, "error": "no such path"})

    def do_POST(self):
        if not self._authorised():
            return self._json(401, {"ok": False, "error": "bad token"})
        try:
            n = int(self.headers.get("content-length", "0"))
            if n > 20000:
                return self._json(413, {"ok": False, "error": "too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"ok": False, "error": "bad json"})
        try:
            if self.path == "/plan":
                text = str(body.get("text", ""))[:400]
                if not text.strip():
                    return self._json(400, {"ok": False, "error": "text required"})
                return self._json(200, plan(text, [str(x) for x in (body.get("nearby") or [])][:20]))
            if self.path == "/reply":
                return self._json(200, {"ok": True, "say": reply(str(body.get("user", ""))[:300], str(body.get("did", ""))[:400],
                                                                   [str(x) for x in (body.get("notes") or [])][:5])})
            if self.path == "/speak":
                text = str(body.get("text", ""))
                if not text.strip():
                    return self._json(400, {"ok": False, "error": "text required"})
                data, voice = speak(text)
                return self._send(200, data, "audio/mpeg", {"X-Skippy-Voice": voice})
        except Exception as e:                                                  # the game must always get an answer
            return self._json(502, {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        self._json(404, {"ok": False, "error": "no such path"})


class Server(ThreadingHTTPServer):
    allow_reuse_address = False        # on Windows, reuse would let two programs share a port; fail loudly instead


def warm() -> None:
    """Load the model now, so the first request is not the slow one."""
    def go():
        try:
            chat([{"role": "user", "content": "ok"}], None)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


def serve(host: str = "127.0.0.1", port: int = 8777) -> ThreadingHTTPServer:
    srv = Server((host, port), Handler)
    if chat is ollama_chat:
        warm()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main(argv=None) -> int:
    global MODEL, TTS_MODE, TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tts", default=TTS_MODE, choices=["sapi", "auto", "elevenlabs"])
    ap.add_argument("--token", default=TOKEN)
    a = ap.parse_args(argv)
    MODEL, TTS_MODE, TOKEN = a.model, a.tts, a.token
    srv = Server((a.host, a.port), Handler)
    warm()
    print(f"Skippy bridge on http://{a.host}:{a.port}  model={MODEL}  voice={TTS_MODE}  token={'set' if TOKEN else 'none'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
