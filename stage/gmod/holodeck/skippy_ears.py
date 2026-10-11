"""Skippy's ears: the player's own microphone bridge. Optional; text chat ("Skippy, add three couches") needs none of this.

    python stage/gmod/holodeck/skippy_ears.py [--data DIR] [--device "Microphone (...)"] [--api http://localhost:3003]
    python stage/gmod/holodeck/skippy_ears.py --list                               # the microphones ffmpeg can see
    python stage/gmod/holodeck/skippy_ears.py --wav speech.wav [--direct]          # one file instead of the microphone
    python stage/gmod/holodeck/skippy_ears.py --selftest                           # speak a request to myself and follow it through

How it works, and what it does NOT do:
  * The game (skippy_client.lua) writes `data/mentat/ptt.txt` while YOU are transmitting on voice chat ("1 <time>"), and "0 <time>"
    when you stop. `+skippy` (a key you bind) writes "1 <time> direct": talk to Skippy alone, no wake word. The file is only
    believed while it is fresh: a "1" left behind by a crashed game is ignored, and one written before this program started is
    ignored until the game writes it again.
  * The microphone is read continuously into a short ring buffer in MEMORY only (about a second and a half while you are not
    transmitting, so the first word of a sentence is not lost to the key press). Nothing is written to disk and nothing is sent
    anywhere while you are not transmitting.
  * When you stop transmitting, the sentence you just said (and only that) goes to the LOCAL Whisper in SkippyTel
    (`POST /translate/audio`, never through an HTTP proxy; audio never leaves your machine). The recording exists only in memory.
    Recognition happens on a worker thread, so the next sentence is heard even while the last one is still being recognised.
  * If the text starts with the wake word "Skippy" (or you used +skippy), the rest is written to `data/mentat/voice_in.json`, which
    the game picks up and sends to the server like a chat request (a request the game has not picked up within 15 s is withdrawn).
    Otherwise it is dropped on the floor: what you say to other players is neither kept nor sent.
Needs ffmpeg on the PATH for the microphone (Windows DirectShow). No Python packages beyond the standard library.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import io
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATE = 16000                       # samples per second the microphone is read at (mono, 16 bit)
PREROLL_S = 0.9                    # kept in memory before the transmit key goes down
TAIL_S = 0.35                      # kept after it comes up
MIN_S, MAX_S = 0.5, 25.0
IDLE_KEEP_S = PREROLL_S + 0.5      # all the microphone audio kept while nobody is transmitting
PTT_HEARTBEAT_STALE_S = 3.0        # a game that keeps rewriting ptt.txt while transmitting: older than this means it is gone
PTT_SINGLE_STALE_S = MAX_S + 5.0   # a game that writes it once per change: a press this old has been cut off already
PTT_UNREADABLE_S = 1.0             # a file caught half-written keeps the previous state for this long
VOICE_IN_MAX_AGE_S = 15.0          # a request the game has not picked up by then is withdrawn
WORKER_QUEUE = 4                   # sentences waiting for recognition; the next one is refused with a message
MIC_START_S = 5.0                  # how long a microphone gets to deliver its first samples
MIC_RESTARTS = 3                   # times ffmpeg is started again after the microphone stops, before giving up

WAKE_WORDS = ["skippy", "skipy", "skippie", "skipi", "skippi", "skipee", "skipe", "skippe", "skipper", "skypie", "skippee"]
LEAD_WORDS = {"hey", "ok", "okay", "yo", "hi", "hello", "oi", "so"}
NOT_WAKE = {"skipping", "skimpy", "skippers", "skipped"}


def find_gmod_data() -> Path:
    """The game's data/mentat folder: GMOD_DATA if set, else the first Steam library that has Garry's Mod, else the owner's."""
    env = os.environ.get("GMOD_DATA")
    if env:
        return Path(env) / "mentat"
    libs = [Path("D:/Steam"), Path("C:/Program Files (x86)/Steam"), Path("C:/Program Files/Steam")]
    for steam in list(libs):
        try:
            text = (steam / "steamapps" / "libraryfolders.vdf").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        libs += [Path(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"([^"]+)"', text)]
    for lib in libs:
        data = lib / "steamapps" / "common" / "GarrysMod" / "garrysmod" / "data"
        if data.is_dir():
            return data / "mentat"
    return Path("D:/Steam/steamapps/common/GarrysMod/garrysmod/data") / "mentat"


DEFAULT_DATA = find_gmod_data()

# the speech never goes through an HTTP proxy: Whisper is on this machine (or on the tailnet), by contract
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# --------------------------------------------------------------------------------------------- the wake word
_OPENERS = "\"'([,.-" + "".join(chr(c) for c in (0x201C, 0x2018, 0x2013, 0x2014, 0x2026))
_LEAD = re.compile(r"[\s" + re.escape(_OPENERS) + r"]*([A-Za-z']+)")
_TIDY = re.compile(r"^[\s,:;.!?\-" + "".join(chr(c) for c in (0x2013, 0x2014, 0x2026)) + r"]+")


def _is_wake(word: str) -> bool:
    w = re.sub(r"'s$", "", word.lower())
    if w in WAKE_WORDS:
        return True
    # speech recognition spells it other ways: a close spelling of the same sound counts, but only a word that starts like it
    # ("skimpy", "slippy" and "snippy" are other words)
    return 5 <= len(w) <= 7 and w[:2] in ("sk", "sc") and w not in NOT_WAKE and difflib.SequenceMatcher(None, w, "skippy").ratio() >= 0.8


def strip_wake(text: str) -> tuple[bool, str]:
    """(True, what follows the wake word) when the sentence begins with 'Skippy' (speech recognition spells it several ways)."""
    raw = text.strip()
    m = _LEAD.match(raw)
    if m and m.group(1).lower() in LEAD_WORDS:
        m = _LEAD.match(raw, m.end())
    if not m:
        return False, raw
    if _is_wake(m.group(1)):
        return True, _TIDY.sub("", raw[m.end():])
    m2 = _LEAD.match(raw, m.end())                                   # "skip e": only whitespace may separate the two halves
    if m2 and not raw[m.end():m2.start(1)].strip() and (m.group(1) + m2.group(1)).lower() in WAKE_WORDS:
        return True, _TIDY.sub("", raw[m2.end():])
    return False, raw


# --------------------------------------------------------------------------------------------- the local Whisper
UPLOAD_LIMIT = 100_000             # SkippyTel refuses a body over 128 KB; 16 kHz 16-bit audio is 32 KB a second, so past ~3 s it must be compressed


def compress_for_upload(wav_bytes: bytes) -> tuple[bytes, str, str]:
    """(bytes, file name, content type) to send: the WAV itself when it is small, else AAC made in memory (pipes, no file), at a
    quarter of the size, which the Whisper service reads like the phone's AAC. A sentence of up to MAX_S seconds then fits."""
    if len(wav_bytes) <= UPLOAD_LIMIT:
        return wav_bytes, "speech.wav", "audio/wav"
    done = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(RATE),
                           "-c:a", "aac", "-b:a", "32k", "-f", "adts", "pipe:1"], input=wav_bytes, capture_output=True, timeout=30)
    if done.returncode != 0 or not done.stdout:
        raise OSError("could not compress the recording: " + done.stderr.decode("utf-8", "replace")[:200])
    return done.stdout, "speech.aac", "audio/aac"


def transcribe(wav_bytes: bytes, api: str, timeout: float = 60.0) -> str:
    data, name, ctype = compress_for_upload(wav_bytes)
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(api.rstrip("/") + "/translate/audio", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with _DIRECT.open(req, timeout=timeout) as r:
        return str(json.loads(r.read()).get("english_text", "")).strip()


def pcm_to_wav(pcm: bytes) -> bytes:
    """A WAV file for raw 16-bit mono samples, built in memory: the speech is never written to disk."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


class DeliveryError(Exception):
    """The sentence was recognised but could not be handed to the game."""


def deliver(data_dir: Path, text: str, heard: str, direct: bool, retries: int = 20, delay: float = 0.05, sleep=time.sleep) -> None:
    """Hand a request to the game: write the file atomically so the game never reads half of it. On Windows the game may have the
    old file open for a moment (a reader without delete-sharing makes the replace fail), so the replace is retried."""
    dest, tmp = data_dir / "voice_in.json", data_dir / "voice_in.tmp"
    err: Exception | None = None
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"text": text, "heard": heard, "direct": direct, "ts": time.time()}), encoding="utf-8")
        for _ in range(max(1, retries)):
            try:
                os.replace(tmp, dest)
                return
            except PermissionError as e:
                err = e
                sleep(delay)
    except OSError as e:
        err = e
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    raise DeliveryError(f"{type(err).__name__}: {err}")


def expire_voice_in(data_dir: Path, max_age: float = VOICE_IN_MAX_AGE_S, now: float | None = None) -> bool:
    """Withdraw a request the game never picked up (it may not be running), so it is not sent to some later server. True if removed."""
    path = data_dir / "voice_in.json"
    try:
        age = (time.time() if now is None else now) - path.stat().st_mtime
        if age > max_age:
            path.unlink()
            return True
    except OSError:
        pass
    return False


def handle_speech(wav_bytes: bytes, api: str, data_dir: Path, direct: bool, log=print) -> str | None:
    """Recognise one sentence and, if it is for Skippy, hand it to the game. Returns the request text, or None."""
    heard = transcribe(wav_bytes, api)
    if not heard:
        return None
    woke, after = strip_wake(heard)
    ok, rest = (True, after if woke else heard) if direct else (woke, after)
    if not ok or not rest:
        log(f"[ears] not for me ({len(heard)} characters, discarded)")
        return None
    log(f"[ears] request: {rest!r}")
    deliver(data_dir, rest, heard, direct)
    return rest


# --------------------------------------------------------------------------------------------- the microphone
def parse_dshow_audio(listing: str) -> list[tuple[str, str]]:
    """(friendly name, alternative name) of every audio device in ffmpeg's DirectShow listing. The alternative name is ASCII."""
    out: list[tuple[str, str]] = []
    for line in listing.splitlines():
        m = re.search(r'"([^"]+)" \(audio\)', line)
        if m:
            out.append((m.group(1), ""))
            continue
        m = re.search(r'Alternative name "([^"]+)"', line)
        if m and out and not out[-1][1]:
            out[-1] = (out[-1][0], m.group(1))
    return out


def dshow_audio_devices() -> list[tuple[str, str]]:
    # ffmpeg writes device names as UTF-8 whatever the console code page is
    res = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], stdin=subprocess.DEVNULL,
                         capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    return parse_dshow_audio(res.stderr or "")


class Microphone:
    """ffmpeg reading the microphone into a ring buffer of raw samples (memory only), and watching that ffmpeg stays alive."""

    def __init__(self, device: str, cmd: list[str] | None = None):
        self.proc = subprocess.Popen(
            cmd or ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "dshow", "-i", f"audio={device}", "-ac", "1",
                    "-ar", str(RATE), "-f", "s16le", "pipe:1"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.lock = threading.Lock()
        self.chunks: collections.deque = collections.deque()      # (sample_index_of_first_sample, bytes)
        self.total = 0                                            # samples read so far
        self.idle_samples = int(RATE * IDLE_KEEP_S)               # what is kept while nobody is transmitting
        self.max_samples = int(RATE * (MAX_S + PREROLL_S + TAIL_S + 2))     # the most that is ever kept, even for a held key
        self.hold_from: int | None = None                         # keep everything from this sample on (a sentence is being recorded)
        self.err: collections.deque = collections.deque(maxlen=20)
        self.dead: str | None = None                              # why the microphone stopped, once it has
        self.last_data = time.monotonic()
        self._err_thread = threading.Thread(target=self._drain_err, daemon=True)
        self._err_thread.start()
        threading.Thread(target=self._pump, daemon=True).start()

    def _drain_err(self):
        for line in iter(self.proc.stderr.readline, b""):
            self.err.append(line.decode("utf-8", "replace").rstrip())

    def _pump(self):
        while True:
            data = self.proc.stdout.read(3200)                     # 0.1 s
            if not data:
                rc = self.proc.wait()
                self._err_thread.join(1.0)
                self.dead = f"ffmpeg exited (code {rc}): " + (" | ".join(self.err) or "no message")
                return
            data = data[:len(data) // 2 * 2]
            with self.lock:
                self.chunks.append((self.total, data))
                self.total += len(data) // 2
                floor = self.total - self.idle_samples
                if self.hold_from is not None:
                    floor = max(min(floor, self.hold_from), self.total - self.max_samples)
                while self.chunks and self.chunks[0][0] + len(self.chunks[0][1]) // 2 <= floor:
                    self.chunks.popleft()
            self.last_data = time.monotonic()

    def mark(self) -> int:
        with self.lock:
            return self.total

    def hold(self, start: int) -> None:
        """A sentence starts at sample `start`: keep the audio from there until release()."""
        with self.lock:
            self.hold_from = start

    def release(self) -> None:
        with self.lock:
            self.hold_from = None

    def slice(self, start: int, end: int) -> bytes:
        with self.lock:
            parts = []
            for idx, data in self.chunks:
                n = len(data) // 2
                if idx + n <= start or idx >= end:
                    continue
                a, b = max(start - idx, 0), min(end - idx, n)
                parts.append(data[a * 2:b * 2])
            return b"".join(parts)

    def health(self, grace: float = 3.0) -> str | None:
        """None while samples keep arriving, else why not (ffmpeg exited, or the device went silent for `grace` seconds)."""
        if self.dead:
            return self.dead
        if time.monotonic() - self.last_data > grace:
            return f"no samples for {grace:.0f} s (microphone unplugged, or grabbed by another program?) " + " | ".join(self.err)
        return None

    def wait_ready(self, timeout: float = 3.0) -> str | None:
        """None once samples are flowing, else the reason they are not."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.dead:
                return self.dead
            if self.total > 0:
                return None
            time.sleep(0.05)
        return self.dead or f"no samples within {timeout:.0f} s " + " | ".join(self.err)

    def close(self):
        try:
            self.proc.kill()
        except OSError:
            pass


# --------------------------------------------------------------------------------------------- the transmit key
class PttWatcher:
    """Reads the game's ptt.txt and says whether the player is transmitting NOW. The file holds only the last state, so its age
    matters: a "1" the game never took back (it crashed, or was closed mid-sentence) must not make the microphone record forever.

      * a "1" written before this program started is ignored until the game writes the file again;
      * a game that keeps rewriting the file while the key is down (a heartbeat) is trusted for 3 s after its last write;
      * a game that writes only when the state changes is trusted for MAX_S + 5 s, which is longer than any one sentence is kept;
      * a file caught half-written (empty, or locked by the game) keeps the previous state for a second.
    Returns (on, direct, token); token changes whenever the file was written again."""

    def __init__(self, path: Path, clock=time.time, log=print):
        self.path, self.clock, self.log = path, clock, log
        self.base: int | None = None            # the file's write time when this program started
        self.prev: tuple[bool, bool, int | None] = (False, False, None)
        self.beats: set[int] = set()            # distinct write times seen during the current press
        self.unreadable_since: float | None = None
        self.warned: set[str] = set()

    def _warn(self, key: str, msg: str) -> None:
        if key not in self.warned:
            self.warned.add(key)
            self.log(msg)

    def poll(self) -> tuple[bool, bool, int | None]:
        try:
            st = self.path.stat()
            parts = self.path.read_text(encoding="utf-8").split()
        except FileNotFoundError:
            if self.base is None:
                self.base = -1                                              # no file at startup: whatever appears later is new
            self.beats.clear()
            self.unreadable_since = None
            self.prev = (False, False, None)
            return self.prev
        except OSError:
            parts, st = [], None
        if not parts or st is None:                                      # unreadable or caught half-written
            now = self.clock()
            if self.unreadable_since is None:
                self.unreadable_since = now
            if now - self.unreadable_since > PTT_UNREADABLE_S:
                self.beats.clear()
                self.prev = (False, False, None)
            return self.prev
        self.unreadable_since = None
        on, direct, mtime = parts[0] == "1", "direct" in parts, st.st_mtime_ns
        if self.base is None:
            self.base = mtime
        if not on:
            self.beats.clear()
            self.warned.discard("stale")
            self.prev = (False, False, mtime)
            return self.prev
        if mtime == self.base:
            self._warn("left over", "[ears] ptt.txt says you are transmitting but it was written before I started; waiting for the game to write it again")
            self.prev = (False, False, mtime)
            return self.prev
        self.beats.add(mtime)
        age = max(0.0, self.clock() - st.st_mtime)
        limit = PTT_HEARTBEAT_STALE_S if len(self.beats) >= 2 else PTT_SINGLE_STALE_S
        if age > limit:
            self._warn("stale", f"[ears] ptt.txt has said 'transmitting' for {age:.0f} s without a new write (the game closed?); treating it as off")
            self.prev = (False, False, mtime)
            return self.prev
        self.prev = (True, direct, mtime)
        return self.prev


def read_ptt(path: Path) -> tuple[bool, bool]:
    """(transmitting, direct) according to ptt.txt right now, with the same age rules as PttWatcher but no history (a one-off look)."""
    w = PttWatcher(path, log=lambda *_: None)
    w.base = -1                                                       # nothing counts as "written before I started"
    on, direct, _ = w.poll()
    return on, direct


class Recorder:
    """Turns the transmit state into sentences without ever blocking: it only remembers where in the microphone's sample counter a
    sentence starts and ends, and hands the slice to `submit` the moment the tail has arrived. A new press during the tail or
    during recognition is therefore seen at once."""

    def __init__(self, mic, submit, log=print):
        self.mic, self.submit, self.log = mic, submit, log
        self.state = "idle"                      # idle | rec | tail
        self.start = self.end = 0
        self.floor = 0                           # audio before this belongs to a sentence already taken
        self.direct = False
        self.spent: int | None = None            # the press (file write) whose MAX_S has run out; it may not start another sentence

    def step(self, on: bool, direct: bool = False, token: int | None = None) -> None:
        if not on:
            self.spent = None
        elif token is not None and token == self.spent:
            on = False                                                   # the same press, already cut at MAX_S
        mark = self.mic.mark()
        if self.state == "idle" and on:
            self.start = max(self.floor, mark - int(PREROLL_S * RATE), 0)
            self.direct = direct
            self.mic.hold(self.start)
            self.state = "rec"
        elif self.state == "rec":
            cut = mark - self.start > MAX_S * RATE
            if not on or cut:
                if on and cut:
                    self.spent = token
                self.end = mark + int(TAIL_S * RATE)
                self.state = "tail"
        if self.state == "tail" and mark >= self.end:
            pcm = self.mic.slice(self.start, self.end)
            self.mic.release()
            self.floor = self.end
            self.state = "idle"
            if len(pcm) / 2 / RATE >= MIN_S:
                self.submit(pcm, self.direct)


class Worker:
    """Recognises sentences one at a time on its own thread, so the key is watched while Whisper works (its first answer after
    a start can take 10 to 20 s while the model loads)."""

    def __init__(self, api: str, data_dir: Path, log=print, handler=None):
        self.api, self.data_dir, self.log = api, data_dir, log
        self.handler = handler or handle_speech
        self.jobs: queue.Queue = queue.Queue(maxsize=WORKER_QUEUE)
        self.busy = False
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, pcm: bytes, direct: bool) -> None:
        try:
            self.jobs.put_nowait((pcm, direct))
            self.log(f"[ears] heard {len(pcm) / 2 / RATE:.1f} s, recognising")
        except queue.Full:
            self.log("[ears] still busy with earlier sentences; dropped this one")

    def idle(self) -> bool:
        return self.jobs.empty() and not self.busy

    def _work(self) -> None:
        while True:
            pcm, direct = self.jobs.get()
            self.busy = True
            try:
                self.handler(pcm_to_wav(pcm), self.api, self.data_dir, direct, log=self.log)
            except DeliveryError as e:
                self.log(f"[ears] understood you, but could not hand it to the game: {e}")
            except Exception as e:
                self.log(f"[ears] could not recognise that: {e}")
            finally:
                self.busy = False


def open_microphone(device: str, alt: str = "", log=print) -> Microphone | None:
    """Start ffmpeg on the device and make sure samples really arrive. Tries the ASCII alternative name first, then the friendly one."""
    for name in [n for n in (alt, device) if n]:
        try:
            mic = Microphone(name)
        except OSError as e:
            log(f"[ears] cannot start ffmpeg: {e} (is it on the PATH?)")
            return None
        why = mic.wait_ready(MIC_START_S)
        if why is None:
            return mic
        mic.close()
        log(f"[ears] the microphone {ascii(name)} did not start: {why}")
    return None


def listen(data_dir: Path, device: str, api: str, alt: str = "", log=print, stop: threading.Event | None = None) -> int:
    """Listen until Ctrl+C (or `stop` is set). Returns 0, or 2 when there is no microphone or it stops working."""
    mic = open_microphone(device, alt, log)
    if mic is None:
        log("[ears] no microphone; run with --list to see the names ffmpeg knows, then pass one with --device")
        return 2
    log(f"[ears] listening to {ascii(device)}; the game's transmit key ({data_dir / 'ptt.txt'}) decides what is kept. Ctrl+C to stop.")
    ptt_file = data_dir / "ptt.txt"
    watcher = PttWatcher(ptt_file, log=log)
    worker = Worker(api, data_dir, log)
    rec = Recorder(mic, worker.submit, log)
    expire_voice_in(data_dir)
    started = last_check = time.monotonic()
    warned_missing = False
    code, restarts = 0, 0
    try:
        while stop is None or not stop.is_set():
            on, direct, token = watcher.poll()
            rec.step(on, direct, token)
            now = time.monotonic()
            if now - last_check >= 1.0:
                last_check = now
                why = mic.health()
                if why:
                    log(f"[ears] the microphone stopped: {why}")
                    mic.close()
                    restarts += 1
                    mic = open_microphone(device, alt, log) if restarts <= MIC_RESTARTS else None
                    if mic is None:
                        log("[ears] giving up on the microphone")
                        code = 2
                        break
                    log("[ears] the microphone is back")
                    rec = Recorder(mic, worker.submit, log)                  # a sentence cut off by the stop is not continued
                expire_voice_in(data_dir)
                if not warned_missing and now - started > 10 and not ptt_file.exists():
                    warned_missing = True
                    log(f"[ears] {ptt_file} has not appeared yet: is the game running with the Mentat addon, and is --data the game's data/mentat folder?")
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        if mic is not None:
            mic.close()
    return code


# --------------------------------------------------------------------------------------------- self test
def selftest(data_dir: Path, api: str) -> int:
    sys.path.insert(0, str(HERE))
    import skippy_bridge
    cases = [("Skippy, add three couches around a coffee table.", True),
             ("hey skippy put a lamp next to the couch", True),
             ("I was skipping stones with my brother yesterday", False)]
    failures = 0
    for phrase, expect in cases:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            wav = Path(d) / "a.wav"
            wav16 = Path(d) / "b.wav"
            skippy_bridge._sapi_wav(phrase, wav)
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-ar", str(RATE), str(wav16)], check=True)
            (data_dir / "voice_in.json").unlink(missing_ok=True)
            got = handle_speech(wav16.read_bytes(), api, data_dir, False)
            ok = (got is not None) == expect
            failures += 0 if ok else 1
            print(("PASS " if ok else "FAIL ") + f"{phrase!r} -> {got!r}")
    (data_dir / "voice_in.json").unlink(missing_ok=True)
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="the game's data/mentat folder")
    ap.add_argument("--device", default="", help="DirectShow microphone name (default: the first one)")
    ap.add_argument("--list", action="store_true", help="print the microphones ffmpeg can see and exit")
    ap.add_argument("--api", default=os.environ.get("SKIPPY_TEL", "http://127.0.0.1:3003"))
    ap.add_argument("--wav", default="", help="recognise this recording instead of listening")
    ap.add_argument("--direct", action="store_true", help="with --wav: treat it as the +skippy key (no wake word)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    data = Path(a.data)
    host = urllib.parse.urlsplit(a.api).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[ears] note: speech is sent to {host}, not to this machine")
    if a.selftest:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            return selftest(Path(d), a.api)
    if a.wav:
        try:
            return 0 if handle_speech(Path(a.wav).read_bytes(), a.api, data, a.direct) else 1
        except DeliveryError as e:
            print(f"[ears] understood you, but could not hand it to the game: {e}")
            return 1
    devices: list[tuple[str, str]] = []
    if a.list or not a.device:
        try:
            devices = dshow_audio_devices()
        except (OSError, subprocess.SubprocessError) as e:
            print(f"cannot run ffmpeg: {e}")
            return 1
    if a.list:
        for friendly, alt in devices:
            print(f"{ascii(friendly)}" + (f"   (also: {alt})" if alt else ""))
        return 0
    device, alt = (a.device, "") if a.device else (devices[0] if devices else ("", ""))
    if not device:
        print("no microphone found (run with --list; ffmpeg -list_devices true -f dshow -i dummy)")
        return 1
    if not data.is_dir():
        print(f"[ears] warning: {data} does not exist (yet). Pass the game's data/mentat folder with --data, or set GMOD_DATA.")
    return listen(data, device, a.api, alt)


if __name__ == "__main__":
    sys.exit(main())
