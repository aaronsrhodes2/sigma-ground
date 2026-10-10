"""Skippy's ears: the player's own microphone bridge. Optional; text chat ("Skippy, add three couches") needs none of this.

    python stage/gmod/holodeck/skippy_ears.py [--data DIR] [--device "Microphone (...)"] [--api http://localhost:3003]
    python stage/gmod/holodeck/skippy_ears.py --wav speech.wav [--direct]          # one file instead of the microphone
    python stage/gmod/holodeck/skippy_ears.py --selftest                           # speak a request to myself and follow it through

How it works, and what it does NOT do:
  * The game (skippy_client.lua) writes `data/mentat/ptt.txt` while YOU are transmitting on voice chat ("1 <time>"), and "0 <time>"
    when you stop. `+skippy` (a key you bind) writes "1 <time> direct": talk to Skippy alone, no wake word.
  * The microphone is read continuously into a short ring buffer in MEMORY only (the last second or so, so the first word of a
    sentence is not lost to the key press). Nothing is written to disk and nothing is sent anywhere while you are not transmitting.
  * When you stop transmitting, the sentence you just said (and only that) goes to the LOCAL Whisper in SkippyTel
    (`POST /translate/audio`; audio never leaves your machine). The recording is deleted straight away.
  * If the text starts with the wake word "Skippy" (or you used +skippy), the rest is written to `data/mentat/voice_in.json`, which
    the game picks up and sends to the server like a chat request. Otherwise it is dropped on the floor: what you say to other
    players is neither kept nor sent.
Needs ffmpeg on the PATH for the microphone (Windows DirectShow). No Python packages beyond the standard library.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path(os.environ.get("GMOD_DATA", "D:/Steam/steamapps/common/GarrysMod/garrysmod/data")) / "mentat"
RATE = 16000                       # samples per second the microphone is read at (mono, 16 bit)
PREROLL_S = 0.9                    # kept in memory before the transmit key goes down
TAIL_S = 0.35                      # kept after it comes up
MIN_S, MAX_S = 0.5, 25.0

WAKE_WORDS = ["skippy", "skipy", "skippie", "skipi", "skippi", "skipee", "skipe", "skippe"]
LEAD_WORDS = {"hey", "ok", "okay", "yo", "hi", "hello", "oi", "so"}


# --------------------------------------------------------------------------------------------- the wake word
def strip_wake(text: str) -> tuple[bool, str]:
    """(True, what follows the wake word) when the sentence begins with 'Skippy' (speech recognition spells it several ways)."""
    raw = text.strip()
    tokens = re.findall(r"\S+", raw)
    i = 0
    if tokens and re.sub(r"[^a-z]", "", tokens[0].lower()) in LEAD_WORDS:
        i = 1

    def close(word: str) -> bool:
        w = re.sub(r"[^a-z']", "", word.lower())
        return w in WAKE_WORDS or (len(w) >= 5 and difflib.SequenceMatcher(None, w, "skippy").ratio() >= 0.8 and w != "skipping")

    def rest_after(n_tokens: int) -> str:
        consumed = 0
        pos = 0
        for m in re.finditer(r"\S+", raw):
            consumed += 1
            pos = m.end()
            if consumed == n_tokens:
                break
        return re.sub(r"^[\s,:;.!\-]+", "", raw[pos:])

    if i < len(tokens) and close(tokens[i]):
        return True, rest_after(i + 1)
    if i + 1 < len(tokens) and close(tokens[i] + tokens[i + 1]):          # "skip e"
        return True, rest_after(i + 2)
    return False, raw


# --------------------------------------------------------------------------------------------- the local Whisper
def transcribe(wav_bytes: bytes, api: str, timeout: float = 60.0) -> str:
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="speech.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(api.rstrip("/") + "/translate/audio", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return str(json.loads(r.read()).get("english_text", "")).strip()


def pcm_to_wav(pcm: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def deliver(data_dir: Path, text: str, heard: str, direct: bool) -> None:
    """Hand a request to the game: write the file atomically so the game never reads half of it."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = data_dir / "voice_in.tmp"
    tmp.write_text(json.dumps({"text": text, "heard": heard, "direct": direct, "ts": time.time()}), encoding="utf-8")
    os.replace(tmp, data_dir / "voice_in.json")


def handle_speech(wav_bytes: bytes, api: str, data_dir: Path, direct: bool, log=print) -> str | None:
    """Recognise one sentence and, if it is for Skippy, hand it to the game. Returns the request text, or None."""
    heard = transcribe(wav_bytes, api)
    if not heard:
        return None
    if direct:
        ok, rest = True, strip_wake(heard)[1] if strip_wake(heard)[0] else heard
    else:
        ok, rest = strip_wake(heard)
    if not ok or not rest:
        log(f"[ears] not for me ({len(heard)} characters, discarded)")
        return None
    log(f"[ears] request: {rest!r}")
    deliver(data_dir, rest, heard, direct)
    return rest


# --------------------------------------------------------------------------------------------- the microphone
def dshow_audio_devices() -> list[str]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                         capture_output=True, text=True, timeout=30).stderr
    return re.findall(r'"([^"]+)" \(audio\)', out)


class Microphone:
    """ffmpeg reading the microphone into a ring buffer of raw samples (memory only)."""

    def __init__(self, device: str):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "dshow", "-i", f"audio={device}", "-ac", "1", "-ar", str(RATE),
             "-f", "s16le", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.lock = threading.Lock()
        self.chunks: collections.deque = collections.deque()      # (sample_index_of_first_sample, bytes)
        self.total = 0                                            # samples read so far
        self.max_samples = int(RATE * (MAX_S + PREROLL_S + 2))
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            data = self.proc.stdout.read(3200)                     # 0.1 s
            if not data:
                return
            with self.lock:
                self.chunks.append((self.total, data))
                self.total += len(data) // 2
                while self.total - self.chunks[0][0] > self.max_samples:
                    self.chunks.popleft()

    def mark(self) -> int:
        with self.lock:
            return self.total

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

    def close(self):
        try:
            self.proc.stdin and self.proc.stdin.close()
        except Exception:
            pass
        self.proc.kill()


def read_ptt(path: Path) -> tuple[bool, bool]:
    try:
        parts = path.read_text(encoding="utf-8").split()
    except OSError:
        return False, False
    return (bool(parts) and parts[0] == "1"), ("direct" in parts)


def listen(data_dir: Path, device: str, api: str) -> None:
    mic = Microphone(device)
    print(f"[ears] listening to {device!r}; the game's transmit key ({data_dir / 'ptt.txt'}) decides what is kept. Ctrl+C to stop.")
    ptt_file = data_dir / "ptt.txt"
    active, start_mark, direct = False, 0, False
    try:
        while True:
            on, is_direct = read_ptt(ptt_file)
            if on and not active:
                active, direct = True, is_direct
                start_mark = max(0, mic.mark() - int(PREROLL_S * RATE))
            elif active and (not on or mic.mark() - start_mark > MAX_S * RATE):
                time.sleep(TAIL_S)
                end = mic.mark()
                active = False
                pcm = mic.slice(start_mark, end)
                if len(pcm) / 2 / RATE >= MIN_S:
                    try:
                        handle_speech(pcm_to_wav(pcm), api, data_dir, direct)
                    except Exception as e:
                        print(f"[ears] could not recognise that: {e}")
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()


# --------------------------------------------------------------------------------------------- self test
def selftest(data_dir: Path, api: str) -> int:
    sys.path.insert(0, str(HERE))
    import skippy_bridge
    cases = [("Skippy, add three couches around a coffee table.", True),
             ("hey skippy put a lamp next to the couch", True),
             ("I was skipping stones with my brother yesterday", False)]
    failures = 0
    for phrase, expect in cases:
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "a.wav"
            wav16 = Path(d) / "b.wav"
            skippy_bridge._sapi_wav(phrase, wav)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-ar", str(RATE), str(wav16)], check=True)
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
    ap.add_argument("--api", default=os.environ.get("SKIPPY_TEL", "http://127.0.0.1:3003"))
    ap.add_argument("--wav", default="", help="recognise this recording instead of listening")
    ap.add_argument("--direct", action="store_true", help="with --wav: treat it as the +skippy key (no wake word)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    data = Path(a.data)
    if a.selftest:
        return selftest(Path(tempfile.mkdtemp()), a.api)
    if a.wav:
        return 0 if handle_speech(Path(a.wav).read_bytes(), a.api, data, a.direct) else 1
    device = a.device or (dshow_audio_devices() or [""])[0]
    if not device:
        print("no microphone found (ffmpeg -list_devices true -f dshow -i dummy)")
        return 1
    listen(data, device, a.api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
