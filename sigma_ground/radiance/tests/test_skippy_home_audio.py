"""Offline tests for stage/skippy_home (the audio router): no audio device, a fake output backend, real decoding of generated wav files."""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

HOME = Path(__file__).resolve().parents[3] / "stage" / "skippy_home"
sys.path.insert(0, str(HOME))
pytest.importorskip("av")

import audio_config as acfg  # noqa: E402
import audio_library as alib  # noqa: E402
import audio_router as ar  # noqa: E402
import calibrate  # noqa: E402
import intents  # noqa: E402
import tools  # noqa: E402

RATE = ar.RATE


def wav(path: Path, seconds: float, freq: float = 440.0, amp: float = 0.5):
    t = np.arange(int(seconds * RATE)) / RATE
    x = (np.sin(2 * np.pi * freq * t) * amp * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(np.repeat(x[:, None], 2, axis=1).tobytes())
    return str(path)


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def make(tmp_path):
    ar.FakeBackend.instances.clear(); ar.FakeBackend.fail_names.clear()

    def _make(over=None):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(over or {}))
        cfg, notes = acfg.load(p)
        clk = Clock()
        r = ar.Router(cfg, backend_factory=ar.FakeBackend, threads=False, state_path=tmp_path / "state.json", clock=clk)
        r.test_clock = clk
        return r
    return _make


def pump(r, n=200):
    for _ in range(n):
        if not r.decode_step():
            break


def drain(o, frames):
    out = []
    while frames > 0:
        k = min(1024, frames); out.append(o.pull(k)); frames -= k
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------ config
def test_default_config_maps_here_to_hdmi_and_upstairs_to_enclave(tmp_path):
    cfg, notes = acfg.load(tmp_path / "missing.json")
    assert "EPSON PJ" in cfg["outputs"]["here"]["device"]
    assert "Enclave" in cfg["outputs"]["upstairs"]["device"]
    assert acfg.target_names(cfg, "everywhere") == ["here", "upstairs"]
    assert any("no private config" in n for n in notes)


def test_config_rejects_bad_things(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"groups": {"all": ["nope"]}}))
    with pytest.raises(ValueError, match="unknown output"):
        acfg.load(p)
    p.write_text(json.dumps({"outputs": {"x": {"kind": "wired"}}}))
    with pytest.raises(ValueError, match="device"):
        acfg.load(p)
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        acfg.load(p)
    p.write_text(json.dumps({"default_target": "attic"}))
    with pytest.raises(ValueError, match="default_target"):
        acfg.load(p)


def test_unmeasured_latency_is_flagged_not_silent(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"outputs": {"a": {"device": "A", "kind": "bluetooth"}}, "groups": {}, "default_target": "a"}))
    cfg, notes = acfg.load(p)
    assert cfg["outputs"]["a"]["latency_ms"] == 200.0 and any("assuming 200" in n for n in notes)


def test_new_named_output_is_just_config(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"outputs": {"kitchen": {"device": "Pi Kitchen", "latency_ms": 90}}, "groups": {"everywhere": ["kitchen"]}, "default_target": "kitchen"}))
    cfg, _ = acfg.load(p)
    assert acfg.target_names(cfg, "kitchen") == ["kitchen"]


# ------------------------------------------------------------------------------------------------ sync / delay logic
def test_faster_output_is_delayed_by_the_difference():
    outs = {"a": {"latency_ms": 60, "lead_in_ms": 300}, "b": {"latency_ms": 200, "lead_in_ms": 400}}
    assert acfg.effective_delays(outs, ["a", "b"]) == {"a": 140.0, "b": 0.0}
    assert acfg.effective_delays(outs, ["a"]) == {"a": 0.0}           # alone, nothing to wait for
    outs["a"]["delay_ms"] = 10
    assert acfg.effective_delays(outs, ["a", "b"])["a"] == 10.0       # explicit override wins


def test_music_starts_audibly_together_despite_different_latency_and_lead_in(make):
    r = make()
    r.play(tracks=[ar.Track(wav(Path(r.state_path).parent / "a.wav", 3.0))], targets="everywhere")
    pump(r)
    h, u = r.outputs["here"], r.outputs["upstairs"]
    # first loud sample position in each ring, plus link latency = when it is heard
    heard = {}
    for o in (h, u):
        a = drain(o, int(1.5 * RATE))
        first = int(np.argmax(np.abs(a[:, 0]) > 0.01))
        assert first > 0
        heard[o.name] = first / RATE + o.lat_s
    assert abs(heard["here"] - heard["upstairs"]) < 0.002
    # the faster link (here, 60 ms) waited; both leads were respected
    assert h.base != u.base and -h.base / RATE >= 0.3 and -u.base / RATE >= 0.4


def test_lead_in_is_inaudible_noise_not_silence_and_does_not_clip_the_start(make):
    r = make()
    wavp = wav(Path(r.state_path).parent / "a.wav", 2.0)
    r.play(tracks=[ar.Track(wavp)], targets="upstairs"); pump(r)
    o = r.outputs["upstairs"]
    a = drain(o, int(0.35 * RATE))
    assert 0 < np.max(np.abs(a)) <= 2e-4                         # about -80 dBFS and not digital zero
    rest = drain(o, int(1.0 * RATE))
    assert np.max(np.abs(rest)) > 0.07                           # (0.5 * volume^2) then the music, un-clipped, begins


# ------------------------------------------------------------------------------------------------ drift
def simulate(r, seconds, clocks, correct=True, block=1024):
    """Advance each output's device clock separately (clocks[name] = relative speed) and return the audible-position error (s)."""
    t = {n: 0.0 for n in clocks}
    nxt_drift = 0.0
    now = 0.0
    errs = []
    while now < seconds:
        n = min(t, key=t.get)
        now = t[n]
        pump(r, 4)
        r.test_clock.t = now
        r.outputs[n].pull(block)
        t[n] += block / (RATE * clocks[n])
        if now >= nxt_drift:
            r.test_clock.t = now
            if correct:
                r.drift_step(0.1)
            nxt_drift += 0.1
            ref, o = r.outputs["here"], r.outputs["upstairs"]
            errs.append((o.audible(now) - ref.audible(now)) / RATE)
    return errs


def long_track(r, seconds):
    return ar.Track(wav(Path(r.state_path).parent / "long.wav", seconds))


def test_synthetic_clock_drift_diverges_without_correction(make):
    r = make({"buffer_s": 1.0})
    r.play(tracks=[long_track(r, 130)], targets="everywhere")
    errs = simulate(r, 100, {"here": 1.0, "upstairs": 1.0005}, correct=False)
    assert abs(errs[-1]) > 0.03        # 500 ppm for 100 s = 50 ms


def test_drift_correction_holds_two_clocks_together(make):
    r = make({"buffer_s": 1.0})
    r.play(tracks=[long_track(r, 130)], targets="everywhere")
    errs = simulate(r, 100, {"here": 1.0, "upstairs": 1.0005}, correct=True)
    tail = errs[len(errs) // 2:]
    assert max(abs(e) for e in tail) < 0.012
    r2 = make()
    assert ar.drift_ratio(0.0) == 1.0 and ar.drift_ratio(0.5) == pytest.approx(1 - 1e-3) and ar.drift_ratio(-0.5) == pytest.approx(1 + 1e-3)


def test_drift_in_the_other_direction_and_no_audible_glitches(make):
    r = make()
    r.play(tracks=[long_track(r, 130)], targets="everywhere")
    errs = simulate(r, 100, {"here": 1.0003, "upstairs": 1.0}, correct=True)
    assert max(abs(e) for e in errs[len(errs) // 2:]) < 0.012
    assert r.outputs["here"].underruns == 0 and r.outputs["upstairs"].underruns == 0


# ------------------------------------------------------------------------------------------------ device loss and recovery
def test_staggered_stream_start_is_realigned_at_once(make):
    r = make()
    r.play(tracks=[long_track(r, 30)], targets="everywhere"); pump(r)
    clk = r.test_clock
    h, u = r.outputs["here"], r.outputs["upstairs"]
    t = clk.t
    for k in range(18):                       # 'here' starts pulling 0.38 s before 'upstairs' does
        clk.t = t + k * 1024 / RATE; h.pull(1024)
    clk.t = t + 18 * 1024 / RATE
    u.pull(1024); h.pull(1024)
    bad = abs(u.audible(clk.t) - h.audible(clk.t)) / RATE
    assert bad > 0.25                         # the problem is real
    r.drift_step(0.5)
    assert abs(u.audible(clk.t) - h.audible(clk.t)) / RATE > 0.25      # one sighting is not enough (Bluetooth callbacks are bursty)
    r.drift_step(0.5)
    assert abs(u.audible(clk.t) - h.audible(clk.t)) / RATE < 0.005
    assert any("re-aligned" in e["msg"] for e in r.events)


def test_output_loss_keeps_playing_on_the_other_then_rejoins_in_step(make):
    r = make({"reconnect_s": 5})
    r.play(tracks=[long_track(r, 60)], targets="everywhere")
    pump(r); drain(r.outputs["here"], 2 * RATE); drain(r.outputs["upstairs"], 2 * RATE)
    next(b for b in ar.FakeBackend.instances if b.output.name == "upstairs").kill()
    r.monitor_step()
    assert r.active == ["here"] and r.playing
    assert any("upstairs lost" in e["msg"] for e in r.events)
    ar.FakeBackend.fail_names["upstairs"] = True
    for _ in range(5):                                           # music continues on 'here'
        pump(r); drain(r.outputs["here"], 1024)
    r.test_clock.t += 6; r.monitor_step()                        # retry while still gone: reported, not fatal
    assert r.active == ["here"] and any("unavailable" in e["msg"] for e in r.events)
    ar.FakeBackend.fail_names.clear()
    r.test_clock.t += 6
    pump(r); r.monitor_step()
    assert r.active == ["here", "upstairs"]
    assert any("reconnected" in e["msg"] for e in r.events)
    h, u = r.outputs["here"], r.outputs["upstairs"]
    assert abs(u.audible() - h.audible()) / RATE < 0.003          # re-entered aligned with where 'here' is
    assert np.max(np.abs(drain(u, 2 * RATE))) > 0.07               # and the music does arrive after the lead-in


def test_unavailable_at_start_plays_on_the_rest_and_reports(make):
    r = make()
    ar.FakeBackend.fail_names["upstairs"] = True
    st = r.play(tracks=[long_track(r, 10)], targets="everywhere")
    assert st["active"] == ["here"] and st["lost"] == ["upstairs"] and r.playing


def test_all_outputs_lost_does_not_crash_or_lose_the_queue(make):
    r = make()
    r.play(tracks=[long_track(r, 10)], targets="everywhere"); pump(r)
    for b in ar.FakeBackend.instances:
        b.kill()
    r.monitor_step()
    assert r.active == [] and r.decode_step() is False and len(r.queue) == 1


# ------------------------------------------------------------------------------------------------ transport, targets, volume
def test_stop_one_target_keeps_the_other(make):
    r = make()
    r.play(tracks=[long_track(r, 10)], targets="everywhere"); pump(r)
    r.stop(["upstairs"])
    assert r.active == ["here"] and r.playing
    r.stop()
    assert not r.playing and r.active == [] and all(o.state == "closed" for o in r.outputs.values())


def test_set_targets_adds_an_output_in_step(make):
    r = make()
    r.play(tracks=[long_track(r, 20)], targets="here"); pump(r); drain(r.outputs["here"], RATE)
    r.set_targets("everywhere")
    assert set(r.active) == {"here", "upstairs"}
    assert abs(r.outputs["upstairs"].audible() - r.outputs["here"].audible()) / RATE < 0.003


def test_volume_per_target_cap_and_persistence(make, tmp_path):
    r = make()
    r.set_volume("upstairs", level=0.2)
    r.set_volume("here", level=0.99)
    v = r.now()["volumes"]
    assert v["upstairs"] == 0.2 and v["here"] == 0.9                 # capped at max_gain_volume
    r.set_volume("everywhere", delta=-0.1)
    assert r.now()["volumes"]["upstairs"] == pytest.approx(0.1)
    r2 = ar.Router(r.cfg, backend_factory=ar.FakeBackend, threads=False, state_path=r.state_path)
    assert r2.outputs["upstairs"].volume == pytest.approx(0.1)       # remembered across restarts


def test_pause_outputs_silence_and_resume_continues(make):
    r = make()
    r.play(tracks=[long_track(r, 10)], targets="here"); pump(r); drain(r.outputs["here"], int(0.8 * RATE))
    pos = r.now()["position_s"]
    r.pause()
    assert np.max(np.abs(r.outputs["here"].pull(1024))) == 0.0
    r.resume()
    assert r.now()["paused"] is False and r.now()["position_s"] == pos


def test_queue_advances_gaplessly_and_reports_the_audible_track(make):
    r = make()
    d = Path(r.state_path).parent
    r.play(tracks=[ar.Track(wav(d / "t1.wav", 1.0), title="one"), ar.Track(wav(d / "t2.wav", 1.0), title="two")], targets="here")
    for _ in range(300):
        pump(r, 10); r.outputs["here"].pull(1024)
        if r.eof_fed:
            break
    assert r.eof_fed
    assert r.now()["queue_len"] == 2


# ------------------------------------------------------------------------------------------------ audiobooks
def book_dir(tmp_path):
    d = tmp_path / "Some Book"; d.mkdir()
    for i in (1, 2, 10):
        wav(d / f"Chapter {i:02d}.wav", 20.0, freq=300 + i)
    return d


def test_audiobook_folder_is_natural_ordered_and_resumes_where_left_off(make, tmp_path):
    d = book_dir(tmp_path)
    r = make()
    tr = ar.resolve_source(str(d), "book")
    assert [Path(t.path).name for t in tr] == ["Chapter 01.wav", "Chapter 02.wav", "Chapter 10.wav"]
    r.play(str(d), "here", kind="book")
    for _ in range(40):
        pump(r, 20); r.outputs["here"].pull(1024)
    for _ in range(int(30 * RATE / 1024)):                         # listen through chapter 1 into chapter 2
        pump(r, 5); r.outputs["here"].pull(1024)
    idx, pos = r._pos_info()
    assert idx == 1 and pos > 5
    r.pause()
    saved = json.loads(Path(r.state_path).read_text())["books"][str(d)]
    assert saved["index"] == 1 and saved["pos_s"] == pytest.approx(pos, abs=0.05)
    r2 = ar.Router(r.cfg, backend_factory=ar.FakeBackend, threads=False, state_path=r.state_path)    # a restart
    assert r2.last_book_path() == str(d)
    st = r2.play(str(d), "upstairs", kind="book")
    assert st["queue_index"] == 1 and st["position_s"] == pytest.approx(saved["pos_s"] - 2.0, abs=0.1)   # a two second recap


def test_finishing_a_book_marks_it_finished_so_it_starts_over(make, tmp_path):
    f = wav(tmp_path / "short.wav", 1.0)
    r = make()
    r.play(f, "here", kind="book")
    for _ in range(100):
        pump(r, 5); r.outputs["here"].pull(1024); r.monitor_step()
    assert r.books[f]["finished"] is True and r.last_book_path() is None


def test_chapter_skip_and_back_use_chapters(make, tmp_path):
    f = wav(tmp_path / "b.wav", 60.0)
    t = ar.Track(f, title="Book", book=f, chapters=[(0, 20, "One"), (20, 40, "Two"), (40, 60, "Three")])
    r = make()
    r.play(tracks=[t], targets="here")
    assert r.now()["chapter"] == "One"
    r.skip()
    assert r.bounds[0][2] == 20 and r.now()["chapter"] == "Two"
    r.skip()
    assert r.bounds[0][2] == 40
    r.prev()
    assert r.bounds[0][2] == 20


def test_seek_relative_and_absolute_and_decoder_starts_there(make, tmp_path):
    f = wav(tmp_path / "s.wav", 30.0)
    r = make()
    r.play(tracks=[ar.Track(f, duration=30.0)], targets="here")
    r.seek(12.0)
    assert r.bounds[0][2] == 12.0
    r.seek(5.0, relative=True)
    assert r.bounds[0][2] == pytest.approx(17.0, abs=0.05)
    blocks = list(ar.decode_blocks(f, 28.0))
    assert abs(sum(len(b) for b in blocks) / RATE - 2.0) < 0.05


def test_m3u_and_missing_file(tmp_path):
    a = wav(tmp_path / "a.wav", 0.5); wav(tmp_path / "b.wav", 0.5)
    (tmp_path / "l.m3u").write_text("#EXTM3U\n#EXTINF:1,x\na.wav\nb.wav\ngone.wav\n")
    assert len(ar.resolve_source(str(tmp_path / "l.m3u"))) == 2
    with pytest.raises(FileNotFoundError):
        ar.resolve_source(str(tmp_path / "nope.mp3"))


def test_a_corrupt_file_is_skipped_not_fatal(make, tmp_path):
    bad = tmp_path / "bad.mp3"; bad.write_bytes(b"not audio at all" * 100)
    good = wav(tmp_path / "good.wav", 1.0)
    r = make()
    r.play(tracks=[ar.Track(str(bad)), ar.Track(good)], targets="here")
    for _ in range(60):
        pump(r, 5); r.outputs["here"].pull(1024)
    assert any("decode error" in e["msg"] for e in r.events) and r.index == 1


# ------------------------------------------------------------------------------------------------ intents
CORPUS = [
    ("play thriller everywhere", "play", "thriller", ["everywhere"]),
    ("Play Thriller on everywhere please", "play", "thriller", ["everywhere"]),
    ("play some pink floyd upstairs only", "play", "pink floyd", ["upstairs"]),
    ("play abbey road by the beatles in here", "play", "abbey road by the beatles", ["here"]),
    ("hey skippy could you play abbey road on both speakers", "play", "abbey road", ["everywhere"]),
    ("play here comes the sun", "play", "here comes the sun", []),           # 'here' inside a title is not a room
    ("put on some jazz downstairs", "play", "jazz", ["here"]),
    ("play the dark side of the moon throughout the house", "play", "dark side of the moon throughout the house", []),
    ("plya thriller everwhere", "play", "thriller", ["everywhere"]),         # typos
    ("paly radiohead upstaris", "play", "radiohead", ["upstairs"]),
    ("play music", "play", "", []),
    ("play my audiobook", "play", "", []),
    ("read me dune upstairs", "play", "dune", ["upstairs"]),
    ("pause", "pause", "", []),
    ("pasue the music", "pause", "", []),
    ("hold on pause that upstairs", "pause", "", ["upstairs"]),
    ("resume", "resume", "", []),
    ("resume my audiobook", "resume", "", []),
    ("continue the book", "resume", "", []),
    ("keep going", "resume", "", []),
    ("skip", "skip", "", []),
    ("next song", "skip", "", []),
    ("skip this one", "skip", "", []),
    ("previous track", "prev", "", []),
    ("go back", "prev", "", []),
    ("stop everything", "stop", "", []),
    ("stop", "stop", "", []),
    ("stop the music everywhere", "stop", "", []),
    ("stop upstairs", "stop", "", ["upstairs"]),
    ("shut up", "stop", "", []),
    ("volume up here", "volume", "", ["here"]),
    ("louder", "volume", "", []),
    ("quieter upstairs", "volume", "", ["upstairs"]),
    ("turn it down a bit upstairs", "volume", "", ["upstairs"]),
    ("turn down the volume upstaris", "volume", "", ["upstairs"]),
    ("set the volume to 40 percent upstairs", "volume", "", ["upstairs"]),
    ("volume 30", "volume", "", []),
    ("mute everywhere", "volume", "", ["everywhere"]),
    ("it is too loud in here", "volume", "", ["here"]),
    ("skip forward 30 seconds", "seek", "", []),
    ("go back a minute", "seek", "", []),
    ("rewind two minutes", "seek", "", []),
    ("move it to everywhere", "move", "", ["everywhere"]),
    ("send this upstairs", "move", "", ["upstairs"]),
    ("what's playing", "now", "", []),
    ("what song is this", "now", "", []),
    ("yes", "confirm", "", []),
    ("no thanks", "decline", "", []),
    ("the second one", "pick", "", []),
    ("what is the weather", "now", "", []),   # placeholder replaced below
]


@pytest.mark.parametrize("phrase,action,thing,targets", [c for c in CORPUS if "weather" not in c[0]])
def test_intent_corpus(phrase, action, thing, targets):
    it = intents.parse(phrase)
    assert it.action == action, it
    if action == "play":
        assert it.thing == thing, it
    if action not in ("confirm", "decline", "pick", "now"):
        assert it.targets == targets, it


def test_intent_details():
    assert intents.parse("play thriller upstairs only").only is True
    assert intents.parse("quieter upstairs").delta < 0 and intents.parse("louder").delta > 0
    assert intents.parse("a lot quieter").delta == pytest.approx(-0.2)
    assert intents.parse("set the volume to 40 percent").level == pytest.approx(0.4)
    assert intents.parse("skip forward 30 seconds").seconds == 30
    assert intents.parse("go back a minute").seconds == -60
    assert intents.parse("resume my audiobook").kind == "book" and intents.parse("play my audiobook").kind == "book"
    assert intents.parse("shuffle the beatles").shuffle is True
    assert intents.parse("mute").mute is True and intents.parse("unmute").mute is False
    assert intents.parse("what is the weather like").action in ("unknown", "now")
    assert intents.parse("").action == "unknown"


def test_custom_alias_from_config():
    it = intents.parse("play jazz in the kitchen", {"kitchen": ["the kitchen"]})
    assert it.targets == ["kitchen"] and it.thing == "jazz"


# ------------------------------------------------------------------------------------------------ library resolution (honesty)
ROWS = [
    ["Michael Jackson/Thriller (1982)/01 Wanna Be Startin' Somethin'.mp3", "Wanna Be Startin' Somethin'", "Michael Jackson", "Thriller (1982)", 1, 1, "Pop", 360, 1, 1982, 0],
    ["Michael Jackson/Thriller (1982)/02 Thriller.mp3", "Thriller", "Michael Jackson", "Thriller (1982)", 2, 1, "Pop", 357, 1, 1982, 0],
    ["Pink Floyd/The Wall (1979)/01 Another Brick in the Wall.mp3", "Another Brick in the Wall", "Pink Floyd", "The Wall (1979)", 1, 1, "Rock", 190, 1, 1979, 0],
    ["Pink Floyd/The Wall (1979)/02 Mother.mp3", "Mother", "Pink Floyd", "The Wall (1979)", 2, 1, "Rock", 190, 1, 1979, 0],
    ["Pink Floyd/Animals (1977)/01 Dogs.mp3", "Dogs", "Pink Floyd", "Animals (1977)", 1, 1, "Rock", 1000, 1, 1977, 0],
    ["Pink Floyd/Animals (1977)/02 Pigs.mp3", "Pigs", "Pink Floyd", "Animals (1977)", 2, 1, "Rock", 1000, 1, 1977, 0],
    ["Beatles/Abbey Road (1969)/01 Come Together.mp3", "Come Together", "Beatles", "Abbey Road (1969)", 1, 1, "Rock", 259, 1, 1969, 0],
    ["Beatles/Abbey Road (1969)/02 Something.mp3", "Something", "Beatles", "Abbey Road (1969)", 2, 1, "Rock", 183, 1, 1969, 0],
    ["Various/Hits (2000)/01 Hurt.mp3", "Hurt", "Nine Inch Nails", "Hits (2000)", 1, 1, "Rock", 200, 1, 2000, 0],
    ["Johnny Cash/American IV (2002)/01 Hurt.mp3", "Hurt", "Johnny Cash", "American IV (2002)", 1, 1, "Country", 218, 1, 2002, 0],
]


@pytest.fixture
def lib():
    return alib.Library(ROWS, books=[{"title": "Dune", "author": "Frank Herbert", "path": "C:/books/dune.m4b"}], root="M:/music")


def test_resolver_exact_prefers_album_over_same_named_track_and_never_substitutes(lib):
    r = lib.resolve("thriller")
    assert r.playable and r.entity.kind == "album" and len(r.entity.sources) == 2
    r = lib.resolve("Abbey Road")
    assert r.playable and r.entity.kind == "album"


def test_resolver_ambiguous_asks_instead_of_picking(lib):
    r = lib.resolve("hurt")
    assert r.status == "ambiguous" and not r.playable and "Johnny Cash" in r.say and "Nine Inch Nails" in r.say
    r = lib.resolve("hurt by johnny cash")
    assert r.playable and r.entity.artist == "Johnny Cash"


def test_resolver_closest_is_an_offer_not_a_play(lib):
    r = lib.resolve("thriler")
    assert r.status == "closest" and not r.playable and "closest I have is" in r.say and "Thriller" in r.say
    r = lib.resolve("quantum jazz odyssey")
    assert r.status == "not_found" and not r.playable


def test_resolver_artist_and_book_and_kind_filter(lib):
    assert lib.resolve("pink floyd").entity.kind == "artist" and len(lib.resolve("pink floyd").entity.sources) == 4
    assert lib.resolve("dune").entity.book is True
    assert lib.resolve("dune", "music").status in ("not_found", "closest")
    assert lib.resolve("the wall").entity.kind == "album"


class FakeRouterOut:
    pass


def test_execute_flow_offers_then_confirms_and_picks(make, lib, tmp_path):
    r = make()
    r.pending = None
    # build real small files at the library paths so the router can open them
    root = tmp_path / "m";
    rows = []
    for row in ROWS:
        p = root / row[0]; p.parent.mkdir(parents=True, exist_ok=True); wav(p.with_suffix(".wav"), 0.3)
        rows.append([row[0].replace(".mp3", ".wav")] + row[1:])
    L = alib.Library(rows, root=root)
    say = intents.run("play thriler everywhere", r, L)["say"]
    assert "closest I have is" in say and not r.playing                    # offered, nothing played
    say = intents.run("yes", r, L)["say"]
    assert say.startswith("Playing Thriller") and r.playing and set(r.wanted) == {"here", "upstairs"}
    say = intents.run("play hurt upstairs", r, L)["say"]
    assert "more than one" in say
    say = intents.run("the second one", r, L)["say"]
    assert say.startswith("Playing Hurt") and r.wanted == ["upstairs"]
    assert intents.run("quieter upstairs", r, L)["say"].startswith("Volume")
    assert intents.run("pause", r, L)["say"] == "Paused."
    assert intents.run("stop everything", r, L)["say"] == "Stopped everything." and not r.playing
    assert "audiobook" in intents.run("resume my audiobook", r, L)["say"]      # none started: says so honestly


# ------------------------------------------------------------------------------------------------ tools shape
def test_tools_have_the_satellite_brain_shape_and_run(make, tmp_path):
    for t in tools.TOOLS:
        assert set(t) == {"name", "description", "input_schema"} and t["input_schema"]["type"] == "object"
        assert t["description"] and isinstance(t["input_schema"]["properties"], dict)
    names = {t["name"] for t in tools.TOOLS}
    assert {"play_music", "pause_audio", "resume_audio", "skip_audio", "set_audio_volume", "stop_audio", "move_audio", "play_audiobook"} <= names
    assert tools.TOOLS[0]["input_schema"]["properties"]["where"]["enum"] == ["here", "upstairs", "everywhere"]
    r = make(); r.pending = None
    assert tools.run_tool("set_audio_volume", {"where": "upstairs", "percent": 25}, r).startswith("Volume 25")
    assert tools.run_tool("nonsense", {}, r) == "I have no such audio tool."
    assert tools.run_tool("pause_audio", "garbage", r) == "Paused."


# ------------------------------------------------------------------------------------------------ http api
def test_http_api_is_token_guarded_and_rejects_web_pages(make):
    r = make(); r.pending = None
    srv = ar.serve(r, "127.0.0.1", 0, "s3cret", None)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(path, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body is not None else None, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert call("/now")[0] == 401
        assert call("/now", headers={"X-Skippy-Token": "wrong"})[0] == 401
        code, j = call("/now", headers={"X-Skippy-Token": "s3cret"})
        assert code == 200 and j["ok"] and j["playing"] is False
        assert call("/now", headers={"X-Skippy-Token": "s3cret", "Origin": "http://evil.example"})[0] == 403
        assert call("/volume", {"targets": "upstairs", "level": 0.3}, {"X-Skippy-Token": "s3cret"})[1]["volumes"]["upstairs"] == 0.3
        assert call("/play", {"source": "/no/such/file.mp3"}, {"X-Skippy-Token": "s3cret"})[0] == 400
        code, j = call("/tool", {"name": "now_playing", "input": {}}, {"X-Skippy-Token": "s3cret"})
        assert code == 200 and "Nothing is playing" in j["result"]
        assert len(call("/tools", headers={"X-Skippy-Token": "s3cret"})[1]["tools"]) == len(tools.TOOLS)
    finally:
        srv.shutdown()


# ------------------------------------------------------------------------------------------------ calibration maths
def test_calibrate_finds_known_delay_in_noise():
    rate = 48000
    ch = calibrate.make_chirp(rate)
    rng = np.random.default_rng(3)
    mic = rng.normal(0, 0.01, rate * 3).astype(np.float32)
    at = int(1.2345 * rate)
    mic[at:at + len(ch)] += ch * 0.3
    idx, snr = calibrate.estimate_arrival(mic, ch)
    assert abs(idx - at) <= 1 and snr > calibrate.MIN_SNR


def test_calibrate_reports_low_confidence_for_noise_only():
    rate = 48000
    ch = calibrate.make_chirp(rate)
    mic = np.random.default_rng(4).normal(0, 0.01, rate * 3).astype(np.float32)
    assert calibrate.estimate_arrival(mic, ch)[1] < calibrate.MIN_SNR
    assert calibrate.make_chirp(rate, amp=5.0).max() <= 0.15          # never louder than the quiet-test ceiling


def test_calibrate_manual_set_writes_config(tmp_path, monkeypatch):
    p = tmp_path / "c.json"; p.write_text(json.dumps({"outputs": {"upstairs": {"device": "X"}}}))
    calibrate.write_latency(p, "upstairs", 231.46)
    assert json.loads(p.read_text())["outputs"]["upstairs"]["latency_ms"] == 231.5


def test_sync_start_holds_every_stream_silent_until_all_are_running(tmp_path):
    ar.FakeBackend.instances.clear()
    p = tmp_path / "c.json"; p.write_text("{}")
    cfg, _ = acfg.load(p)
    clk = Clock()
    r = ar.Router(cfg, backend_factory=ar.FakeBackend, threads=False, state_path=tmp_path / "s.json", clock=clk, sync_start=True)
    r.play(tracks=[ar.Track(wav(tmp_path / "a.wav", 5.0))], targets="everywhere")
    h, u = r.outputs["here"], r.outputs["upstairs"]
    assert h.hold and u.hold
    pump(r)
    for _ in range(10):                                           # 'here' is running, 'upstairs' has not called back yet
        h.pull(1024); pump(r)
    assert h.consumed == 0 and h.hold                             # nothing consumed, so nothing to be out of step with
    u.pull(1024); pump(r)                                         # now every stream is running
    assert not h.hold and not u.hold
    h.pull(1024); u.pull(1024)
    assert h.consumed == 1024 and u.consumed == 1024
