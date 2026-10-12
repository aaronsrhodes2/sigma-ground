"""Skippy's audio router: decode once, feed N output devices in sync. The HOST-SIDE player that the house listener
(SkippyTel/services/satellite brain) and admin Skippy call; it does not hear, think or catalogue.

    python stage/skippy_home/audio_router.py serve            # long-running service + token-guarded localhost JSON API
    python stage/skippy_home/audio_router.py devices          # list Windows output endpoints
    python stage/skippy_home/audio_router.py chime <name>     # quiet identifying chime on a config output or a device substring
    python stage/skippy_home/audio_router.py play <source> [--to everywhere] [--book]
    python stage/skippy_home/audio_router.py say "play thriller everywhere"   (needs a running service)

Design (see README.md):
  * one decoder thread (PyAV, 48 kHz stereo float) pushes the same blocks into one Output ring per target device;
  * each Output has a delay (the slowest link gets none, faster links wait the difference) and a near-silent lead-in so a
    sleeping Bluetooth/HDMI link is awake before the first note;
  * clock drift between devices is measured as the audible-position difference between outputs and corrected by a gentle
    resample (at most +-1000 ppm) in the faster output's callback;
  * an output that disappears is dropped from the feed, the music continues on the rest, and it is re-joined (aligned to
    where the others are) when it comes back.
"""
from __future__ import annotations

import argparse
import collections
import hmac
import json
import math
import os
import random
import re
import secrets
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_config as acfg  # noqa: E402

RATE = acfg.RATE
BLOCK = 1024
AUDIO_EXT = {".mp3", ".flac", ".m4a", ".m4b", ".ogg", ".oga", ".opus", ".wav", ".aac", ".wma"}
BOOK_EXT = {".m4b"}
MAX_DRIFT_PPM = 1000.0
DRIFT_DEADBAND = 0.002       # seconds of audible-position error ignored
HARD_RESYNC_S = 0.060       # beyond this the gentle resampling would take minutes: jump instead
KP, KI = 0.07, 0.0025       # drift controller: ratio change per second of audible error, and its integral gain
NOISE_AMP = 1.0e-4           # about -80 dBFS: inaudible, but not digital silence
STATE_PATH = acfg.PRIVATE / "audio_state.json"


# ---------------------------------------------------------------------------------------------------------------- tracks
@dataclass
class Track:
    path: str
    title: str = ""
    artist: str = ""
    duration: float | None = None
    book: str | None = None            # book id (folder or m4b path) when this is part of an audiobook
    book_index: int = 0
    chapters: list = field(default_factory=list)   # [(start_s, end_s, title)]
    is_stream: bool = False

    def label(self) -> str:
        return f"{self.artist} - {self.title}" if self.artist and self.title else (self.title or Path(self.path).stem)


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_tags(path: str) -> Track:
    t = Track(path=path, title=Path(path).stem)
    try:
        import mutagen
        f = mutagen.File(path, easy=True)
        if f is not None:
            if f.info is not None and getattr(f.info, "length", None):
                t.duration = float(f.info.length)
            if f.tags:
                t.title = (f.tags.get("title") or [t.title])[0]
                t.artist = (f.tags.get("artist") or f.tags.get("albumartist") or [""])[0]
    except Exception:
        pass
    return t


def read_chapters(path: str) -> list:
    try:
        import av
        with av.open(path) as c:
            out = []
            for ch in getattr(c, "chapters", None) or []:
                tb = ch.get("time_base", 1)
                s, e = float(ch["start"] * tb), float(ch["end"] * tb)
                out.append((s, e, (ch.get("metadata") or {}).get("title", f"Chapter {len(out) + 1}")))
            return out
    except Exception:
        return []


def resolve_source(source: str, kind: str | None = None) -> list[Track]:
    """A file, a folder (natural order), an .m3u, or an http(s) stream URL -> tracks.
    kind='book' (or an .m4b) marks tracks with a book id so the position is remembered."""
    if re.match(r"^https?://", source):
        return [Track(path=source, title=source, is_stream=True)]
    p = Path(source)
    if p.is_dir():
        files = sorted((f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT), key=lambda f: _natural_key(str(f.relative_to(p))))
        tracks = [read_tags(str(f)) for f in files]
        if kind == "book":
            for i, t in enumerate(tracks):
                t.book, t.book_index = str(p), i
        return tracks
    if not p.exists():
        raise FileNotFoundError(source)
    if p.suffix.lower() in (".m3u", ".m3u8"):
        out = []
        for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                q = Path(line) if Path(line).is_absolute() else p.parent / line
                if q.exists():
                    out.append(read_tags(str(q)))
        return out
    t = read_tags(str(p))
    if p.suffix.lower() in BOOK_EXT or kind == "book":
        t.book, t.book_index = str(p), 0
        t.chapters = read_chapters(str(p))
    return [t]


def decode_blocks(path: str, start_s: float = 0.0, end_s: float | None = None, rate: int = RATE, block: int = BLOCK):
    """Yield float32 (n, 2) blocks at `rate`, from start_s to end_s (or the end)."""
    import av
    c = av.open(path, timeout=10.0) if path.startswith("http") else av.open(path)
    try:
        s = c.streams.audio[0]
        rs = av.AudioResampler(format="flt", layout="stereo", rate=rate)
        if start_s > 0.05 and not path.startswith("http"):
            c.seek(int(start_s * 1_000_000))
        skip = None
        limit = None if end_s is None else int(max(0.0, end_s - start_s) * rate)
        produced = 0
        carry = np.zeros((0, 2), np.float32)

        def frames():
            for fr in c.decode(s):
                yield from rs.resample(fr)
            yield from rs.resample(None)

        for fr in frames():
            a = fr.to_ndarray().reshape(-1, 2).astype(np.float32, copy=False)
            if skip is None:
                ft = fr.time if fr.time is not None else start_s
                skip = max(0, int(round((start_s - ft) * rate))) if start_s > 0.05 else 0
            if skip:
                d = min(skip, len(a)); a = a[d:]; skip -= d
                if not len(a):
                    continue
            carry = np.concatenate([carry, a]) if len(carry) else a
            while len(carry) >= block:
                out, carry = carry[:block], carry[block:]
                if limit is not None and produced + len(out) >= limit:
                    yield out[: limit - produced]
                    return
                produced += len(out)
                yield out
        if len(carry):
            if limit is not None:
                carry = carry[: max(0, limit - produced)]
            if len(carry):
                yield carry
    finally:
        c.close()


# ---------------------------------------------------------------------------------------------------------------- output
class Output:
    """One target device's ring buffer plus the sample-pulling callback body. Pure data/numpy: no audio device needed."""

    def __init__(self, name: str, cfg: dict, rate: int = RATE, clock=time.monotonic):
        self.name, self.cfg, self.rate, self.clock = name, cfg, rate, clock
        self.last_t = None             # when the last callback ran (to compare outputs at a common instant)
        self.lat_s = float(cfg["latency_ms"]) / 1000.0
        self.volume = float(cfg["volume"])
        self.gain_now = self.volume ** 2
        self.chunks: collections.deque = collections.deque()
        self.buffered = 0
        self.consumed = 0.0            # output time consumed, in source frames (fractional because of drift correction)
        self.phase = 0.0
        self.base = 0.0                # global timeline frame that corresponds to consumed == 0
        self.paused = False
        self.hold = False              # stream just opened: output zeros without consuming until every stream is running
        self.off_count = 0
        self.eof = False
        self.state = "closed"          # closed | open | lost
        self.underruns = 0
        self.ratio = 1.0
        self.err_ema = 0.0
        self.trim = 0.0
        self.backend = None
        self.last_try = 0.0
        self.lock = threading.Lock()
        self._rng = np.random.default_rng(1)

    def push(self, a: np.ndarray):
        with self.lock:
            self.chunks.append(a); self.buffered += len(a)

    def push_noise(self, n: int):
        if n > 0:
            self.push(self._rng.uniform(-NOISE_AMP, NOISE_AMP, (n, 2)).astype(np.float32))

    def flush(self):
        with self.lock:
            self.chunks.clear(); self.buffered = 0; self.consumed = 0.0; self.phase = 0.0
            self.eof = False; self.ratio = 1.0; self.err_ema = 0.0; self.trim = 0.0

    def fill(self) -> int:
        return self.buffered

    def audible(self, at: float | None = None) -> float:
        """Global-timeline frame being heard now (what has been handed over, minus the link latency). With `at` (a clock time) the
        position is extrapolated from the last callback, so two outputs whose callbacks fall at different moments compare fairly."""
        ahead = 0.0
        if at is not None and self.last_t is not None and not self.paused:
            ahead = max(0.0, at - self.last_t) * self.rate * self.ratio
        return self.base + self.consumed + ahead - self.lat_s * self.rate

    def pull(self, n: int) -> np.ndarray:
        """Return n frames. Called from the audio callback (or by tests/simulations)."""
        with self.lock:
            self.last_t = self.clock()
            if self.paused or self.hold:
                return np.zeros((n, 2), np.float32)
            ratio = self.ratio
            total = self.phase + n * ratio
            need = int(math.floor(total)) + 2
            take, got = [], 0
            while got < need and self.chunks:
                c = self.chunks.popleft(); take.append(c); got += len(c)
            buf = np.concatenate(take) if take else np.zeros((0, 2), np.float32)
            real = len(buf)
            if real < need:
                if real and not self.eof:
                    self.underruns += 1
                buf = np.concatenate([buf, np.zeros((need - real, 2), np.float32)])
            if ratio == 1.0 and self.phase == 0.0:
                out, used = buf[:n], n
            else:
                pos = self.phase + np.arange(n) * ratio
                i0 = pos.astype(np.int64); fr = (pos - i0).astype(np.float32)[:, None]
                out = buf[i0] * (1 - fr) + buf[i0 + 1] * fr
                used = int(math.floor(total)); self.phase = total - used
            if real > used:
                self.chunks.appendleft(buf[used:real])
            self.buffered = sum(len(c) for c in self.chunks)
            self.consumed += n * ratio if real >= need else min(n * ratio, real)   # an underrun does not advance the stream position
            g0, g1 = self.gain_now, self.volume ** 2
            self.gain_now = g1
            ramp = np.linspace(g0, g1, n, dtype=np.float32)[:, None] if g0 != g1 else g1
            return (out * ramp).astype(np.float32, copy=False)

    def set_volume(self, v: float):
        self.volume = float(min(1.0, max(0.0, v)))

    def jump(self, frames: int):
        """Hard re-sync: frames > 0 moves this output EARLIER in the stream by dropping that much of its queued audio (we were behind);
        frames < 0 makes it wait by putting silence at its head (we were ahead). Used when streams start staggered or one stalls."""
        with self.lock:
            if frames < 0:
                self.chunks.appendleft(np.zeros((-frames, 2), np.float32)); self.buffered += -frames
                self.base += frames
                return
            left = frames
            while left > 0 and self.chunks:
                c = self.chunks.popleft()
                if len(c) > left:
                    self.chunks.appendleft(c[left:]); self.buffered -= left; left = 0
                else:
                    self.buffered -= len(c); left -= len(c)
            self.consumed += frames - left


def drift_ratio(err_s: float, trim: float = 0.0) -> float:
    """Resample ratio for an output whose audible position is `err_s` seconds AHEAD of the reference (positive -> consume slower).
    P term with a deadband, plus the integrated `trim`; clamped to +-MAX_DRIFT_PPM: an inaudible pitch change that converges in a minute."""
    p = 0.0 if abs(err_s) <= DRIFT_DEADBAND else math.copysign(abs(err_s) - DRIFT_DEADBAND, err_s) * KP
    adj = max(-MAX_DRIFT_PPM * 1e-6, min(MAX_DRIFT_PPM * 1e-6, p + trim))
    return 1.0 - adj


# ---------------------------------------------------------------------------------------------------------------- backends
class FakeBackend:
    """No device. Tests pull from the Output themselves; .kill() simulates the device vanishing."""
    instances: list = []
    fail_names: dict = {}

    def __init__(self, output: Output):
        self.output, self.running, self.dead = output, False, False
        FakeBackend.instances.append(self)

    def start(self):
        if FakeBackend.fail_names.get(self.output.name):
            raise OSError("device not present")
        self.running = True

    def stop(self):
        self.running = False

    def alive(self) -> bool:
        return self.running and not self.dead

    def kill(self):
        self.dead = True


def find_device(name_sub: str, hostapi_sub: str = "WASAPI", want_input: bool = False) -> int:
    import sounddevice as sd
    key = "max_input_channels" if want_input else "max_output_channels"
    apis = sd.query_hostapis()
    best = None
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0 and name_sub.lower() in d["name"].lower():
            if hostapi_sub.lower() in apis[d["hostapi"]]["name"].lower():
                return i
            best = best if best is not None else i
    if best is None:
        raise OSError(f"no {'input' if want_input else 'output'} device matching {name_sub!r}")
    return best


class SoundDeviceBackend:
    def __init__(self, output: Output):
        self.output, self.stream, self.error = output, None, None

    def start(self):
        import sounddevice as sd
        o = self.output
        idx = find_device(o.cfg["device"], o.cfg.get("hostapi", "WASAPI"))
        extra = None
        if "wasapi" in sd.query_hostapis(sd.query_devices(idx)["hostapi"])["name"].lower():
            extra = sd.WasapiSettings(auto_convert=True)       # the device may run at 44.1k while we decode at 48k

        def cb(outdata, frames, tinfo, status):
            if status:
                self.error = str(status)
            outdata[:] = o.pull(frames)

        self.stream = sd.OutputStream(device=idx, samplerate=o.rate, channels=2, dtype="float32", blocksize=BLOCK,
                                      latency="high", callback=cb, extra_settings=extra)
        self.stream.start()

    def stop(self):
        try:
            if self.stream:
                self.stream.stop(); self.stream.close()
        except Exception:
            pass
        self.stream = None

    def alive(self) -> bool:
        try:
            return bool(self.stream is not None and self.stream.active)
        except Exception:
            return False


# ---------------------------------------------------------------------------------------------------------------- router
class Router:
    def __init__(self, cfg: dict | None = None, backend_factory=SoundDeviceBackend, threads: bool = True,
                 state_path: Path | str | None = None, clock=time.monotonic, sync_start: bool | None = None):
        if cfg is None:
            cfg, self.notes = acfg.load()
        else:
            self.notes = []
        self.cfg, self.factory, self.clock = cfg, backend_factory, clock
        # Opened streams start pulling at slightly different moments (a Bluetooth stream takes ~0.2 s longer): hold them all silent until
        # every one is running, then release together. Off for fake backends, whose tests pull by hand.
        self.sync_start = (backend_factory is SoundDeviceBackend) if sync_start is None else sync_start
        self.state_path = Path(state_path) if state_path else STATE_PATH
        self.lock = threading.RLock()
        self.outputs = {n: Output(n, c, clock=clock) for n, c in cfg["outputs"].items()}
        self.active: list[str] = []             # outputs currently fed
        self.wanted: list[str] = []             # outputs we should be feeding (active + lost ones we are reconnecting)
        self.queue: list[Track] = []
        self.index = 0
        self.playing = False
        self.paused = False
        self.eof_fed = False
        self.gen = 0                            # bumped on every flush/seek so a stale decode is discarded
        self.it = None
        self.global_pos = 0                     # frames pushed on the global timeline since the last flush
        self.bounds: list = []                  # (global_start_frame, queue_index, start_offset_s)
        self.events: collections.deque = collections.deque(maxlen=100)
        self.idle_since: float | None = None
        st = self._load_state()
        self.books = st.get("books", {})
        self.last_book: str | None = st.get("last_book")
        for n, v in (st.get("volumes") or {}).items():
            if n in self.outputs:
                self.outputs[n].set_volume(min(float(v), float(cfg.get("max_gain_volume", 0.9))))
                self.outputs[n].gain_now = self.outputs[n].volume ** 2
        self.last_book_save = 0.0
        self._last_drift = 0.0
        self._stop = threading.Event()
        self.threads = []
        if threads:
            for fn in (self._decoder_loop, self._monitor_loop):
                t = threading.Thread(target=fn, daemon=True); t.start(); self.threads.append(t)

    # ---- events / state
    def event(self, msg: str):
        self.events.append({"t": round(time.time(), 1), "msg": msg})

    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"books": self.books, "last_book": self.last_book,
                                       "volumes": {n: o.volume for n, o in self.outputs.items()}}, indent=1), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError as e:
            self.event(f"could not save state: {e}")

    def remember_book(self, finished: bool = False):
        with self.lock:
            t = self._cur()
            if not t or not t.book:
                return
            idx, pos = self._pos_info()
            t = self.queue[idx] if 0 <= idx < len(self.queue) else t
            self.books[t.book] = {"index": 0 if finished else t.book_index, "pos_s": 0.0 if finished else round(pos, 2),
                                  "finished": finished, "title": t.title, "updated": round(time.time())}
            self.last_book = t.book
            self._save_state()

    def last_book_path(self) -> str | None:
        b = self.books.get(self.last_book or "")
        return self.last_book if b and not b.get("finished") else None

    # ---- helpers
    def _cur(self) -> Track | None:
        return self.queue[self.index] if 0 <= self.index < len(self.queue) else None

    def expand(self, targets) -> list[str]:
        if not targets:
            return list(self.wanted) or acfg.target_names(self.cfg, self.cfg["default_target"])
        if isinstance(targets, str):
            targets = [targets]
        names: list[str] = []
        for t in targets:
            for n in acfg.target_names(self.cfg, t):
                if n not in names:
                    names.append(n)
        return names

    def delays(self) -> dict[str, float]:
        return acfg.effective_delays(self.cfg["outputs"], list(dict.fromkeys(self.wanted)))

    def _ref(self) -> Output | None:
        return self.outputs[self.active[0]] if self.active else None

    def _open(self, name: str, hold: bool = False) -> bool:
        o = self.outputs[name]
        if o.backend is None or o.state != "open":
            o.backend = self.factory(o)
            o.last_t, o.hold = None, hold and self.sync_start
            try:
                o.backend.start()
            except Exception as e:
                o.backend, o.state, o.last_try = None, "lost", self.clock()
                self.event(f"output {name} unavailable: {e}")
                return False
        o.state = "open"
        return True

    def _start_fresh(self):
        """All wanted outputs: flush, prefix (lead-in + delay), same global-timeline origin."""
        self.active = [n for n in self.active if n in self.wanted]
        prefixes = acfg.start_prefix_ms(self.cfg["outputs"], list(dict.fromkeys(self.wanted)))
        for n in self.wanted:
            o = self.outputs[n]
            if n not in self.active:
                if not self._open(n, hold=True):
                    continue
                self.active.append(n)
            o.flush()
            o.lat_s = float(o.cfg["latency_ms"]) / 1000
            pre = int(prefixes[n] / 1000 * RATE)
            o.push_noise(pre)               # lead-in always: the link may have slept even if the stream stayed open
            o.base = -float(pre)
            o.paused = False
        self.global_pos = 0
        self.eof_fed = False

    def _join(self, name: str):
        """Bring one output (new, or back from loss) into line with where the reference output is."""
        if not self._open(name):
            return
        o = self.outputs[name]
        o.flush()
        o.lat_s = float(o.cfg["latency_ms"]) / 1000
        ref = self._ref()
        if ref is None:
            self.active.append(name); self._start_fresh(); return
        base_i = ref.audible() + o.lat_s * RATE          # makes audible_i == audible_ref at this instant (consumed_i == 0)
        lead = int(o.cfg["lead_in_ms"] / 1000 * RATE)
        with ref.lock:
            pending = np.concatenate(list(ref.chunks)) if ref.chunks else np.zeros((0, 2), np.float32)
            g0 = ref.base + ref.consumed                  # global frame at the head of the reference ring
        s = max(lead, int(math.ceil(g0 - base_i)))        # the newcomer may only start from data the reference still holds
        o.push_noise(s)
        start_g = base_i + s
        if start_g <= self.global_pos and len(pending):
            o.push(pending[max(0, int(round(start_g - g0))):].copy())      # share the reference's not-yet-played frames
        elif start_g < self.global_pos:
            o.push_noise(int(self.global_pos - start_g))
        o.base = float(base_i)
        o.paused = self.paused
        self.active.append(name)
        self.event(f"output {name} joined in step with {ref.name}, {s / RATE * 1000:.0f} ms of lead-in")

    # ---- transport
    def play(self, source=None, targets=None, kind: str | None = None, shuffle: bool = False, append: bool = False,
             tracks: list[Track] | None = None, resume: bool = True) -> dict:
        with self.lock:
            tr = tracks if tracks is not None else resolve_source(source, kind)
            if not tr:
                raise ValueError(f"nothing playable in {source!r}")
            if shuffle:
                tr = list(tr); random.shuffle(tr)
            if append and self.queue and self.playing:
                self.queue.extend(tr)
                self.eof_fed = False
                if self.it is None:
                    self.it = iter(())
                return self.now()
            self.remember_book()
            names = self.expand(targets)
            self.queue, self.index = list(tr), 0
            start = 0.0
            first = tr[0]
            if first.book and resume and first.book in self.books and not self.books[first.book].get("finished"):
                b = self.books[first.book]
                self.index = min(int(b.get("index", 0)), len(tr) - 1)
                start = float(b.get("pos_s", 0.0))
                if start > 5:
                    start -= 2.0                      # a little recap
            self._set_wanted(names)
            self.playing, self.paused = True, False
            self._begin_track(start)
            self.idle_since = None
            return self.now()

    def _set_wanted(self, names: list[str]):
        for n in list(self.active):
            if n not in names:
                self._drop(n)
        self.wanted = list(names)

    def _drop(self, name: str):
        o = self.outputs[name]
        if name in self.active:
            self.active.remove(name)
        if o.backend:
            o.backend.stop()
        o.backend, o.state = None, "closed"
        o.flush()

    def _begin_track(self, start_s: float):
        t = self._cur()
        self.gen += 1
        self.it = decode_blocks(t.path, start_s) if t else iter(())
        self.bounds = [(0, self.index, start_s)]
        self.paused = False
        self._start_fresh()

    def set_targets(self, targets) -> dict:
        """Move/extend the playing music to other outputs without restarting it."""
        with self.lock:
            names = self.expand(targets)
            if not self.playing:
                self.wanted = names
                return self.now()
            for n in list(self.active):
                if n not in names:
                    self._drop(n)
            self.wanted = names
            for n in names:
                if n not in self.active:
                    self._join(n)
            return self.now()

    def pause(self):
        with self.lock:
            if self.playing:
                self.paused = True
                for o in self.outputs.values():
                    o.paused = True
                self.remember_book()
            return self.now()

    def resume(self):
        with self.lock:
            self.paused = False
            for o in self.outputs.values():
                o.paused = False
            return self.now()

    def stop(self, targets=None):
        """No targets: stop everything. With targets: silence just those outputs, the rest keep playing."""
        with self.lock:
            if targets:
                names = self.expand(targets)
                if self.wanted and not set(self.wanted) <= set(names):
                    for n in names:
                        self._drop(n)
                    self.wanted = [n for n in self.wanted if n not in names]
                    return self.now()
            self.remember_book()
            self.playing, self.paused, self.it = False, False, None
            self.gen += 1
            for n in list(self.outputs):
                self._drop(n); self.outputs[n].paused = False
            self.active, self.wanted, self.queue, self.index, self.bounds = [], [], [], 0, []
            self.idle_since = None
            return self.now()

    def seek(self, seconds: float, relative: bool = False) -> dict:
        with self.lock:
            t = self._cur()
            if not t or t.is_stream:
                raise ValueError("nothing seekable is playing")
            idx, cur = self._pos_info()
            self.index = idx
            pos = max(0.0, (cur + seconds) if relative else seconds)
            if t.duration and pos >= t.duration:
                return self.skip()
            was_paused = self.paused
            self._begin_track(pos)
            if was_paused:
                self.pause()
            self.remember_book()
            return self.now()

    def skip(self) -> dict:
        with self.lock:
            idx, p = self._pos_info()
            self.index = idx
            t = self._cur()
            if t and t.chapters:
                nxt = [c for c in t.chapters if c[0] > p + 0.5]
                if nxt:
                    return self.seek(nxt[0][0])
            if self.index + 1 < len(self.queue):
                self.index += 1
                self._begin_track(0.0)
                self.remember_book()
            else:
                self.stop()
            return self.now()

    next = skip

    def prev(self) -> dict:
        with self.lock:
            idx, p = self._pos_info()
            self.index = idx
            t = self._cur()
            if t and t.chapters:
                before = [c for c in t.chapters if c[0] < p - 3.0]
                if before:
                    return self.seek(before[-1][0])
            if p > 3.0 or self.index == 0:
                return self.seek(0.0)
            self.index -= 1
            self._begin_track(0.0)
            return self.now()

    def set_volume(self, targets, level: float | None = None, delta: float | None = None, mute: bool | None = None) -> dict:
        with self.lock:
            cap = float(self.cfg.get("max_gain_volume", 0.9))
            names = self.expand(targets) if targets else (self.wanted or acfg.target_names(self.cfg, self.cfg["default_target"]))
            for n in names:
                o = self.outputs[n]
                if mute is not None:
                    o.set_volume(0.0 if mute else max(o.volume, 0.05)); continue
                v = level if level is not None else o.volume + (delta or 0.0)
                o.set_volume(min(cap, max(0.0, v)))
            self._save_state()
            return self.now()

    def add_to_queue(self, source, kind=None) -> dict:
        with self.lock:
            return self.play(source, kind=kind, append=True) if self.playing else self.play(source, kind=kind)

    # ---- timeline
    def _pos_info(self) -> tuple[int, float]:
        """(queue index, seconds into that track) of what is audible now."""
        with self.lock:
            ref = self._ref()
            if ref is None or not self.bounds:
                return self.index, 0.0
            g = max(0.0, ref.audible())
            cur = self.bounds[0]
            for b in self.bounds:
                if b[0] <= g:
                    cur = b
            return cur[1], cur[2] + (g - cur[0]) / RATE

    def position(self) -> float:
        return self._pos_info()[1]

    def now(self) -> dict:
        with self.lock:
            idx, pos = self._pos_info() if self.playing else (self.index, 0.0)
            t = self.queue[idx] if 0 <= idx < len(self.queue) else None
            chap = None
            if t and t.chapters:
                for c in t.chapters:
                    if c[0] <= pos < c[1]:
                        chap = c[2]
            return {"playing": self.playing and not self.paused, "paused": self.paused, "track": t.label() if t else None,
                    "title": t.title if t else None, "artist": t.artist if t else None, "chapter": chap,
                    "book": bool(t and t.book), "position_s": round(pos, 1), "duration_s": t.duration if t else None,
                    "queue_len": len(self.queue), "queue_index": idx,
                    "targets": list(self.wanted), "active": list(self.active),
                    "lost": [n for n in self.wanted if n not in self.active],
                    "volumes": {n: round(o.volume, 2) for n, o in self.outputs.items()},
                    "delays_ms": {n: round(d) for n, d in self.delays().items()},
                    "events": list(self.events)[-5:]}

    def queue_list(self) -> list[str]:
        return [t.label() for t in self.queue]

    # ---- decoder (one step is synchronous and testable)
    def release_holds(self):
        held = [self.outputs[n] for n in self.active if self.outputs[n].hold]
        if held and all(self.outputs[n].last_t is not None for n in self.active):
            for o in held:
                o.hold = False

    def decode_step(self) -> bool:
        with self.lock:
            self.release_holds()
            if not self.playing or self.paused or not self.active or self.eof_fed or self.it is None:
                return False
            if max(self.outputs[n].fill() for n in self.active) > int(self.cfg["buffer_s"] * RATE):
                return False
            gen, it = self.gen, self.it
        try:
            blk, err = next(it), None
        except StopIteration:
            blk, err = None, None
        except Exception as e:                                   # a corrupt file must not stop the book
            blk, err = None, e
        with self.lock:
            if gen != self.gen:
                return True
            if err is not None:
                self.event(f"decode error in {self._cur().label() if self._cur() else '?'}: {err}")
            if blk is None:
                if self.index + 1 < len(self.queue):
                    if self._cur() and self._cur().book:
                        self.remember_book()
                    self.index += 1
                    t = self._cur()
                    self.it = decode_blocks(t.path, 0.0)
                    self.bounds.append((self.global_pos, self.index, 0.0))
                else:
                    self.eof_fed = True
                    for n in self.active:
                        self.outputs[n].eof = True
                    self.it = None
                return True
            for n in self.active:
                self.outputs[n].push(blk)
            self.global_pos += len(blk)
            ref = self._ref()
            while len(self.bounds) > 2 and ref is not None and self.bounds[1][0] <= ref.audible():
                self.bounds.pop(0)
            return True

    def drift_step(self, dt: float | None = None):
        """Compare each output's audible position with the reference and set its resample ratio (PI controller)."""
        with self.lock:
            now = self.clock()
            if dt is None:
                dt = min(2.0, max(0.05, now - self._last_drift)) if self._last_drift else 0.5
            self._last_drift = now
            ref = self._ref()
            if ref is None or self.paused or self.eof_fed:
                return                                         # (at the very end the rings run dry and positions stop meaning anything)
            alpha = 1.0 - math.exp(-dt / 2.0)
            for n in self.active:
                o = self.outputs[n]
                if o is ref:
                    o.ratio = 1.0
                    continue
                if o.last_t is None or ref.last_t is None:
                    continue                                   # a stream that has not started pulling yet has nothing to compare
                err = (o.audible(now) - ref.audible(now)) / RATE
                o.off_count = o.off_count + 1 if abs(err) > HARD_RESYNC_S else 0
                if o.off_count >= 2:
                    o.off_count = 0
                    o.jump(-int(round(err * RATE)))
                    o.err_ema = 0.0
                    self.event(f"output {n} was {err * 1000:+.0f} ms off: re-aligned")
                    continue
                o.err_ema += alpha * (err - o.err_ema)
                if abs(o.err_ema) > DRIFT_DEADBAND:
                    o.trim = max(-MAX_DRIFT_PPM * 1e-6, min(MAX_DRIFT_PPM * 1e-6, o.trim + KI * o.err_ema * dt))
                o.ratio = drift_ratio(o.err_ema, o.trim)

    def monitor_step(self):
        now = self.clock()
        with self.lock:
            for n in list(self.active):
                o = self.outputs[n]
                if o.backend is None or not o.backend.alive():
                    self.active.remove(n)
                    if o.backend:
                        o.backend.stop()
                    o.backend, o.state, o.last_try = None, "lost", now
                    self.event(f"output {n} lost: continuing on {self.active or 'nothing'}; will retry")
            if self.playing:
                for n in self.wanted:
                    o = self.outputs[n]
                    if n not in self.active and now - o.last_try >= float(self.cfg["reconnect_s"]):
                        o.last_try = now
                        before = len(self.active)
                        self._join(n)
                        if len(self.active) > before:
                            self.event(f"output {n} reconnected")
            self.drift_step()
            if self.playing and self.eof_fed and all(self.outputs[n].fill() == 0 for n in self.active):
                self.remember_book(finished=bool(self._cur() and self._cur().book))
                self.playing = False
                self.idle_since = now
            if not self.playing and self.idle_since is not None and now - self.idle_since > float(self.cfg["idle_close_s"]):
                for n in list(self.active):
                    self._drop(n)
                self.idle_since = None
            if self.playing and not self.paused and now - self.last_book_save > 5:
                self.last_book_save = now
                self.remember_book()

    def _decoder_loop(self):
        while not self._stop.is_set():
            try:
                if not self.decode_step():
                    time.sleep(0.01)
            except Exception as e:
                self.event(f"decoder error: {e}"); time.sleep(0.2)

    def _monitor_loop(self):
        while not self._stop.is_set():
            try:
                self.monitor_step()
            except Exception as e:
                self.event(f"monitor error: {e}")
            time.sleep(0.5)

    def close(self):
        self._stop.set()
        self.stop()


# ---------------------------------------------------------------------------------------------------------------- HTTP API
MAX_BODY = 65536


def load_token() -> str:
    t = os.environ.get("SKIPPY_AUDIO_TOKEN", "").strip()
    if t:
        return t
    p = acfg.PRIVATE / "audio_token.txt"
    if p.exists():
        return p.read_text().strip()
    p.parent.mkdir(parents=True, exist_ok=True)
    t = secrets.token_urlsafe(24)
    p.write_text(t)
    return t


def make_handler(router: Router, token: str, library=None):
    import tools

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def _guard(self) -> bool:
            if any(h.lower().startswith("sec-fetch") for h in self.headers.keys()) or "Origin" in self.headers:
                self._json(403, {"ok": False, "error": "web pages may not call this"}); return False
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower()
            if host not in ("127.0.0.1", "localhost", "[::1]", "skippy-pc") and not host.endswith(".local"):
                self._json(403, {"ok": False, "error": "bad host"}); return False
            sup = self.headers.get("X-Skippy-Token", "").encode("latin-1", "replace")
            if not hmac.compare_digest(sup, token.encode()):
                self._json(401, {"ok": False, "error": "bad token"}); return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            if self.path == "/health": return self._json(200, {"ok": True})
            if self.path == "/now": return self._json(200, {"ok": True, **router.now()})
            if self.path == "/queue": return self._json(200, {"ok": True, "queue": router.queue_list()})
            if self.path == "/devices": return self._json(200, {"ok": True, "devices": list_devices()})
            if self.path == "/tools": return self._json(200, {"ok": True, "tools": tools.TOOLS})
            self._json(404, {"ok": False, "error": "no such path"})

        def do_POST(self):
            if not self._guard():
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > MAX_BODY or "json" not in self.headers.get("Content-Type", ""):
                    return self._json(400, {"ok": False, "error": "JSON body required"})
                b = json.loads(self.rfile.read(n) or b"{}")
                p = self.path
                if p == "/play":
                    r = router.play(b["source"], b.get("targets"), b.get("kind"), bool(b.get("shuffle")), bool(b.get("append")))
                elif p == "/pause": r = router.pause()
                elif p == "/resume": r = router.resume()
                elif p == "/stop": r = router.stop(b.get("targets"))
                elif p == "/skip": r = router.skip()
                elif p == "/prev": r = router.prev()
                elif p == "/seek": r = router.seek(float(b["seconds"]), bool(b.get("relative")))
                elif p == "/targets": r = router.set_targets(b["targets"])
                elif p == "/volume": r = router.set_volume(b.get("targets"), b.get("level"), b.get("delta"), b.get("mute"))
                elif p == "/queue": r = router.add_to_queue(b["source"], b.get("kind"))
                elif p == "/say":
                    import intents
                    return self._json(200, {"ok": True, **intents.run(str(b["text"])[:300], router, library)})
                elif p == "/tool":
                    return self._json(200, {"ok": True, "result": tools.run_tool(str(b["name"]), b.get("input") or {}, router, library)})
                else:
                    return self._json(404, {"ok": False, "error": "no such path"})
                self._json(200, {"ok": True, **r})
            except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
                self._json(400, {"ok": False, "error": str(e)})
            except Exception as e:
                self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    return H


def serve(router: Router, host: str, port: int, token: str, library=None):
    srv = ThreadingHTTPServer((host, port), make_handler(router, token, library))
    srv.daemon_threads = True
    return srv


# ---------------------------------------------------------------------------------------------------------------- devices / CLI
def list_devices() -> list[dict]:
    import sounddevice as sd
    apis = sd.query_hostapis()
    return [{"index": i, "name": d["name"], "hostapi": apis[d["hostapi"]]["name"], "rate": d["default_samplerate"]}
            for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]


def chime(cfg: dict, which: str, amp: float = 0.08) -> str:
    """Two soft notes on one output, to identify it. `which` is a config output name or a device-name substring."""
    import sounddevice as sd
    if which in cfg["outputs"]:
        dev, api = cfg["outputs"][which]["device"], cfg["outputs"][which].get("hostapi", "WASAPI")
    else:
        dev, api = which, "WASAPI"
    idx = find_device(dev, api)
    amp = min(float(amp), 0.15)
    t = np.arange(int(0.35 * RATE)) / RATE
    env = np.minimum(1, t / 0.02) * np.exp(-t * 6)
    lead = np.random.default_rng(2).uniform(-NOISE_AMP, NOISE_AMP, int(0.5 * RATE))
    mono = np.concatenate([lead, np.sin(2 * np.pi * 660 * t) * env * amp, np.sin(2 * np.pi * 880 * t) * env * amp,
                           np.zeros(int(0.4 * RATE))]).astype(np.float32)
    data = np.stack([mono, mono], axis=1)
    ha = sd.query_hostapis(sd.query_devices(idx)["hostapi"])["name"].lower()
    extra = sd.WasapiSettings(auto_convert=True) if "wasapi" in ha else None
    sd.play(data, RATE, device=idx, extra_settings=extra, blocking=True)
    return f"chime played on device {idx}: {sd.query_devices(idx)['name']}"


def _call(cfg, path, body=None):
    req = urllib.request.Request(f"http://{cfg['api']['host']}:{cfg['api']['port']}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"X-Skippy-Token": load_token(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--no-library", action="store_true")
    sub.add_parser("devices")
    c = sub.add_parser("chime"); c.add_argument("name"); c.add_argument("--amp", type=float, default=0.08)
    p = sub.add_parser("play"); p.add_argument("source"); p.add_argument("--to", default=None); p.add_argument("--book", action="store_true")
    sy = sub.add_parser("say"); sy.add_argument("text")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    cfg, notes = acfg.load()
    for n in notes:
        print("note:", n, file=sys.stderr)
    if a.cmd == "devices":
        cur = {o["device"]: n for n, o in cfg["outputs"].items()}
        for d in list_devices():
            tag = next((f"  <- config '{n}'" for dev, n in cur.items() if dev.lower() in d["name"].lower() and "wasapi" in d["hostapi"].lower()), "")
            print(f"{d['index']:3d}  {d['hostapi']:<22} {d['name']}{tag}")
    elif a.cmd == "chime":
        print(chime(cfg, a.name, a.amp))
    elif a.cmd == "serve":
        import audio_library
        lib = None if a.no_library else audio_library.Library.load_default()
        r = Router(cfg)
        srv = serve(r, cfg["api"]["host"], int(cfg["api"]["port"]), load_token(), lib)
        print(f"audio router on http://{cfg['api']['host']}:{cfg['api']['port']}  token in {acfg.PRIVATE / 'audio_token.txt'} (or SKIPPY_AUDIO_TOKEN)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            r.close()
    elif a.cmd == "play":
        print(json.dumps(_call(cfg, "/play", {"source": a.source, "targets": a.to, "kind": "book" if a.book else None})))
    elif a.cmd == "say":
        print(_call(cfg, "/say", {"text": a.text}).get("say"))
    elif a.cmd == "status":
        print(json.dumps(_call(cfg, "/now"), indent=1))


if __name__ == "__main__":
    main()
