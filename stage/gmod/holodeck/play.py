"""One command to play with the holodeck by voice: Skippy's bridge, your microphone and the game, started in order and wired together.

    python stage/gmod/holodeck/play.py [--map gm_construct] [--key x] [--device "Microphone (...)"] [--text-only]
    python stage/gmod/holodeck/play.py --selftest-voice ["add two couches"] [--stop-after]      # prove the whole voice path, hands off
    bash stage/gmod/skippy_play.sh ...                                                           # the same, from bash

What it does, in order (and says what it found):
  1. checks what you need: SkippyTel's Whisper (port 3003), Ollama, ffmpeg and a microphone, your Tailscale address;
  2. starts Skippy's bridge on your Tailscale address (a game that hosts its own world cannot reach localhost: the engine's
     -allowlocalhttp works on dedicated servers only), behind a fresh token. The token is short-lived: it is made for this run and
     dies with the bridge when this program ends. This program writes it nowhere except the tool file below, which is deleted the
     moment the game has consumed it (and again on exit and on a crash). The game's own convar holodeck_bridge_token is archived by
     the game (server.vdf) and keeps the dead token; it opens nothing once the bridge is gone;
  3. starts the game as a small listen server (so voice chat works) with the Mentat addon, which is already linked into its folder;
  4. once you are in, a tool (mentat/tools/skippy_play.lua, armed through data/mentat/tool.txt) points the game at the bridge, turns on Skippy's persona and voice and
     binds a key (default X) that talks to Skippy alone;
  5. starts the microphone bridge (skippy_ears.py): the microphone is read into memory only while you hold the key.
Then: hold X and say "add three couches" (no wake word), or hold V and say "Skippy, add three couches". Ctrl+C stops the
bridge and the microphone; the game keeps running until you close it (or `--stop-after`).

--selftest-voice needs no human: it speaks a request with Windows speech, hands the recording to the SAME code the microphone bridge
uses (the real local Whisper in SkippyTel recognises it), lets the game pick the request up, and checks THREE things: the recognised
text contains the request's words, the game's answer is a build (not a refusal, a "nothing to add" or an error) naming what was asked
for, and Skippy's spoken answer played in the game (with --tts elevenlabs, in his own voice). Only the physical microphone and the
key press are not exercised.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import run as exp_run  # noqa: E402  (ctl(), DATA)
import skippy_bridge  # noqa: E402
import skippy_ears  # noqa: E402

SKIPPY_TEL = os.environ.get("SKIPPY_TEL", "http://127.0.0.1:3003")
OLLAMA = os.environ.get("SKIPPY_OLLAMA", "http://127.0.0.1:11434")
DATA: Path = exp_run.DATA
CONSOLE_LOG = DATA.parent.parent / "console.log"


def disarm() -> None:
    """Remove the armed tool file (it holds the bridge URL and the token). Safe to call at any time."""
    try:
        (DATA / "tool.txt").unlink(missing_ok=True)
    except OSError:
        pass


def say(msg: str) -> None:
    print(f"[play] {msg}", flush=True)


def fetch_json(url: str, timeout: float = 4.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))             # loopback: never through a proxy
    with opener.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def tailscale_ip() -> str:
    for exe in ("tailscale", "C:/Program Files/Tailscale/tailscale.exe"):
        try:
            out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=10).stdout.split()
            if out:
                return out[0]
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def preflight(text_only: bool, host: str) -> dict:
    """What is there. Returns {ok, problems, mic, device, alt, host}; problems stop us, notes do not."""
    found: dict = {"problems": [], "notes": [], "mic": False, "device": "", "alt": "", "host": host}
    try:
        h = fetch_json(SKIPPY_TEL + "/health")
        found["whisper"] = bool(h)
        say(f"SkippyTel answers on {SKIPPY_TEL} (speech recognition and Skippy's voice)")
    except Exception as e:  # noqa: BLE001
        found["whisper"] = False
        found["notes"].append(f"SkippyTel does not answer at {SKIPPY_TEL} ({e}): your voice cannot be recognised until it is up")
    try:
        tags = fetch_json(OLLAMA + "/api/tags")
        names = [m.get("name", "") for m in tags.get("models", [])]
        if any(n.split(":")[0] == skippy_bridge.MODEL.split(":")[0] for n in names):
            say(f"Ollama has {skippy_bridge.MODEL} (Skippy's own words and vague wishes)")
        else:
            found["notes"].append(f"Ollama does not have {skippy_bridge.MODEL}: Skippy answers in plain sentences, and the parser alone reads your words")
    except Exception as e:  # noqa: BLE001
        found["notes"].append(f"Ollama does not answer at {OLLAMA} ({e}): Skippy answers in plain sentences")
    if not host:
        found["problems"].append("no Tailscale address (the game cannot reach the bridge on localhost). Start Tailscale, or set SKIPPY_BRIDGE_HOST")
    else:
        say(f"the bridge will listen on {host} (your Tailscale address)")
    if exp_run.ctl("status").returncode == 0:
        found["problems"].append("Garry's Mod is already running (one instance only): close it, or run bash stage/gmod/gmod_ctl.sh stop")
    if not text_only:
        if not shutil.which("ffmpeg"):
            found["notes"].append("ffmpeg is not on the PATH: no microphone bridge (text chat still works)")
        else:
            try:
                devices = skippy_ears.dshow_audio_devices()
            except Exception as e:  # noqa: BLE001
                devices = []
                found["notes"].append(f"ffmpeg could not list microphones ({e})")
            if devices:
                found["mic"], (found["device"], found["alt"]) = True, devices[0]
                say(f"microphone: {ascii(found['device'])}" + ("" if len(devices) == 1 else f" (first of {len(devices)}; --device chooses another)"))
            else:
                found["notes"].append("no microphone found by ffmpeg (run skippy_ears.py --list): text chat still works")
    return found


def wait_for(path: Path, seconds: float, alive=None) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if path.exists():
            return True
        if alive is not None and not alive():
            return False
        time.sleep(0.5)
    return False


def game_running() -> bool:
    return exp_run.ctl("status").returncode == 0


def log_tail_since(offset: int) -> str:
    try:
        with open(CONSOLE_LOG, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_size() -> int:
    try:
        return CONSOLE_LOG.stat().st_size
    except OSError:
        return 0


def speak_to_ears(phrase: str, api: str, direct: bool) -> str | None:
    """Say `phrase` with Windows speech, give the recording to the microphone bridge's own recogniser, return the text it handed on."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        wav, wav16 = Path(d) / "a.wav", Path(d) / "b.wav"
        skippy_bridge._sapi_wav(phrase, wav)
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-ar", str(skippy_ears.RATE), str(wav16)], check=True)
        return skippy_ears.handle_speech(wav16.read_bytes(), api, DATA, direct)


NUMBER_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
                "9": "nine", "10": "ten", "11": "eleven", "12": "twelve"}
FILLER = {"skippy", "hey", "please", "a", "an", "the", "some", "and", "me", "can", "you", "could"}
VERBS = ("add", "put", "make", "build", "place", "spawn", "give", "bring", "set", "get")
# what the game says when it did NOT build (holodeck.lua limit_text, plan.lua refusals and notes, parse.lua)
NOT_BUILT = re.compile(r"nothing in that to add|so I added nothing|I don't have|limit of|as many props|Clear some first|could not|no room|"
                       r"not allowed|None of that|nothing to undo|nothing recent|more than I can fit|sounded like a no|too soon|cooldown",
                       re.I)


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9']+", (text or "").lower()):
        w = NUMBER_WORDS.get(w, w)
        if w not in FILLER:
            out.append(w)
    return out


def _stem(w: str) -> str:
    return w[:-2] if w.endswith("es") and len(w) > 4 else (w[:-1] if w.endswith("s") and len(w) > 3 else w)


def heard_request(request: str, got: str | None) -> bool:
    """Did the recogniser hear the request? Every content word of the request (numbers as words, plurals ignored, no wake word) must be there."""
    if not got:
        return False
    have = {_stem(w) for w in _words(got)}
    need = [_stem(w) for w in _words(request)]
    return bool(need) and all(w in have for w in need)


def selftest_verdict(request: str, got: str | None, console: str, played: bool, count_before: int | None = None,
                     count_after: int | None = None, tts: str = "auto") -> tuple[bool, str]:
    """Judge the self-test from what was observed. Returns (passed, why). Pure: the tests drive it with canned console text."""
    if not heard_request(request, got):
        return False, f"the recogniser heard {got!r}, which does not contain the request {request!r}"
    answers = re.findall(r"\[holodeck\] \(for [^)]*\) (.*\S)", console)
    if not answers:
        return False, "the game never answered the request"
    if count_before is not None and count_after is not None:
        built = count_after > count_before
        proof = f"{count_after - count_before} new props"
    else:
        number_words = set(NUMBER_WORDS.values())
        nouns = [_stem(w) for w in _words(request) if not w.isdigit() and w not in number_words and w not in VERBS]
        last = answers[-1]
        built = not NOT_BUILT.search(last) and (not nouns or any(n in last.lower() for n in nouns))
        proof = f"the answer {last!r}"
    if not built:
        return False, f"nothing was built ({proof})"
    if not played:
        return False, "Skippy's voice did not play in the game (see the bridge lines above)"
    voices = re.findall(r"Skippy's voice is now: (\S+)", console)
    if tts == "elevenlabs" and (not voices or voices[-1] != "elevenlabs"):
        return False, f"--tts elevenlabs was asked for but the voice that spoke was {voices[-1] if voices else 'not reported'}"
    return True, f"built ({proof}) and the voice played" + (f" ({voices[-1]})" if voices else "")


def selftest_voice(request: str, api: str, tts: str = "auto") -> bool:
    """Drive the real voice path once and report each link."""
    (DATA / "skippy_played.txt").unlink(missing_ok=True)
    start = log_size()
    say(f'speaking "{request}" to the microphone bridge\'s recogniser (the real local Whisper)')
    try:
        got = speak_to_ears("Skippy, " + request, api, False)
    except skippy_ears.DeliveryError as e:
        say(f"FAIL: recognised, but could not hand it to the game: {e}")
        return False
    if got is None:
        say("FAIL: the recogniser did not hand anything on (no wake word heard, or Whisper did not answer)")
        return False
    say(f"the recogniser heard and handed on: {got!r}")
    ok_answered = ok_played = False
    seen = ""
    t0 = time.time()
    while time.time() - t0 < 90:
        time.sleep(1)
        seen = log_tail_since(start)
        ok_answered = bool(re.search(r"\[holodeck\] \(for [^)]*\) .*\S", seen))
        ok_played = (DATA / "skippy_played.txt").exists()
        if ok_answered and ok_played:
            break
    spoken = re.findall(r"\[holodeck\] \(for [^)]*\) (.*)", seen)
    voice = re.findall(r"Skippy's voice is now: (\S+)", seen)
    if voice:
        say(f"the voice that spoke: {voice[-1]}" + ("  (Skippy's own)" if voice[-1] == "elevenlabs" else "  (Windows speech)"))
    else:
        say("the voice that spoke: not reported")
    say(f"the game picked it up and answered: {spoken[-1] if spoken else '(nothing)'}")
    played = (DATA / "skippy_played.txt").read_text(encoding="utf-8").strip() if ok_played else ""
    if played:
        say(f"played clip: {played} (id, seconds long)")
    passed, why = selftest_verdict(request, got, seen, ok_played, tts=tts)
    say(("verdict: " if passed else "FAIL: ") + why)
    return passed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="gm_construct")
    ap.add_argument("--key", default="x", help="the key bound to talk to Skippy alone (default x)")
    ap.add_argument("--device", default="", help="DirectShow microphone name (default: the first one)")
    ap.add_argument("--text-only", action="store_true", help="no microphone bridge (and the game does not capture the mouse: look with the arrow keys)")
    ap.add_argument("--mouse", action="store_true", help="with --text-only: let the game own the mouse as usual")
    ap.add_argument("--tts", default="auto", choices=["sapi", "auto", "elevenlabs"],
                    help="Skippy's voice: auto = his own ElevenLabs voice through SkippyTel (costs a little credit per word), else Windows speech; sapi = free Windows speech only")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--players", type=int, default=4, help="max players of the little listen server (voice chat needs more than 1)")
    ap.add_argument("--selftest-voice", nargs="?", const="add two couches", default=None, metavar="REQUEST")
    ap.add_argument("--stop-after", action="store_true", help="close the game when done (for --selftest-voice)")
    a = ap.parse_args(argv)

    skippy_bridge.TTS_MODE = a.tts
    host = os.environ.get("SKIPPY_BRIDGE_HOST") or tailscale_ip()
    found = preflight(a.text_only or a.selftest_voice is not None, host)
    for n in found["notes"]:
        say("note: " + n)
    if found["problems"]:
        for p in found["problems"]:
            say("cannot start: " + p)
        return 1
    if a.selftest_voice is not None:
        if not found.get("whisper"):
            say("cannot run the voice self-test without SkippyTel's Whisper")
            return 1
        if not shutil.which("ffmpeg"):
            say("cannot run the voice self-test without ffmpeg")
            return 1

    token = secrets.token_hex(12)
    skippy_bridge.TOKEN = token
    srv = None
    port = a.port
    for attempt in range(10):
        try:
            srv = skippy_bridge.serve(host, port)
            break
        except OSError:
            port += 1
    if srv is None:
        say("cannot start: no free port for the bridge")
        return 1
    bridge_url = f"http://{host}:{port}"
    voice_note = "Skippy's own voice through SkippyTel, Windows speech if that is unavailable" if a.tts == "auto" else a.tts
    say(f"bridge up at {bridge_url} (voice: {voice_note})")

    ears = None
    result = 0
    try:
        # arm the tool that connects the running game to the bridge (it runs five seconds after you spawn)
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "tool_skippy_play.json").unlink(missing_ok=True)
        (DATA / "skippy_played.txt").unlink(missing_ok=True)
        (DATA / "voice_in.json").unlink(missing_ok=True)
        arm = {"script": "mentat/tools/skippy_play.lua", "id": time.strftime("%H%M%S"),
               "args": {"bridge": bridge_url, "token": token, "key": a.key, "debug": a.selftest_voice is not None}}
        (DATA / "tool.txt").write_text(json.dumps(arm), encoding="utf-8")
        os.environ["GMOD_EXTRA_ARGS"] = f"+maxplayers {max(2, a.players)}"
        # a person plays this session, so the engine owns the mouse; the hands-off self-test must not take it (gmod_ctl.sh, gmod_behind.ps1)
        # GMOD_MOUSE: 0 = hands-off test (no mouse, window kept behind), 1 = the engine owns the mouse, 2 = played but the mouse stays yours
        os.environ["GMOD_MOUSE"] = "0" if a.selftest_voice is not None else ("2" if (a.text_only and not a.mouse) else "1")
        say(f"starting Garry's Mod on {a.map} (a window will open; you will spawn in a few seconds)")
        started = exp_run.ctl("start", a.map)
        if started.returncode != 0:
            say("cannot start the game: " + (started.stdout or "").strip()[:300])
            return 1
        answered = wait_for(DATA / "tool_skippy_play.json", 180, alive=game_running)
        disarm()                                                          # the game has consumed the tool: the token leaves the disk now
        if not answered:
            say("the game never reached the point where it connects to the bridge (is the Mentat addon linked into its addons folder?)")
            return 1
        try:
            tool = json.loads((DATA / "tool_skippy_play.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            tool = {}
        if tool.get("error"):
            say("the game could not be connected to the bridge: " + str(tool["error"]))
            return 1
        say(f"the game is connected to the bridge; hold {a.key.upper()} and talk to Skippy")

        if found["mic"] and not a.text_only and a.selftest_voice is None:
            cmd = [sys.executable, str(HERE / "skippy_ears.py"), "--data", str(DATA), "--api", SKIPPY_TEL]
            if a.device or found["device"]:
                cmd += ["--device", a.device or found["device"]]
            ears = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            threading.Thread(target=lambda: [print(line.rstrip(), flush=True) for line in ears.stdout], daemon=True).start()
            say("your microphone bridge is running (it records only while the key is held)")

        if a.selftest_voice is not None:
            time.sleep(6)                                                 # let the tool finish and the player settle in
            ok = selftest_voice(a.selftest_voice, SKIPPY_TEL, a.tts)
            say("PASS: the spoken request was recognised as asked, the game built what was asked, and Skippy answered aloud" if ok else "FAIL")
            result = 0 if ok else 1
            if a.stop_after:
                return result
        say("running. Ctrl+C stops the bridge and the microphone; close the game window when you are done.")
        while game_running():
            if ears is not None and ears.poll() is not None:
                say(f"the microphone bridge stopped (exit {ears.returncode}); text chat still works")
                ears = None
            time.sleep(2)
        say("the game closed")
    except KeyboardInterrupt:
        say("stopping")
    finally:
        if ears is not None and ears.poll() is None:
            ears.terminate()
        srv.shutdown()
        srv.server_close()
        disarm()
        if a.stop_after and game_running():
            exp_run.ctl("stop")
    return result


if __name__ == "__main__":
    sys.exit(main())
