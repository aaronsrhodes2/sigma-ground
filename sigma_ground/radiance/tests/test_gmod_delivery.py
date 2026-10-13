"""Skippy's delivery layer and finish chains (stage/gmod/holodeck/skippy_delivery.py), offline.

No network, no credit: pure text transforms, deterministic filter strings, and (when ffmpeg exists) a synthetic sine tone.
What this cannot prove: how any of it sounds. That is for ears (README "Skippy's delivery").
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HOLO = Path(__file__).resolve().parents[3] / "stage" / "gmod" / "holodeck"
sys.path.insert(0, str(HOLO))
import skippy_delivery as D  # noqa: E402
import voice_lab  # noqa: E402

SAMPLES = [
    "Another flawless build. I would take a bow, but I have no knees.",
    "Congratulations, monkey, you found the floor. It was there all along.",
    "Water boils at one hundred degrees, monkey. Write that down somewhere safe.",
    "Gravity never takes a day off. Never complains, either.",
    "LA LA LAAAA! Oh, the kettle sings of boiling dreams!",
    "Psst. Do not tell the others, but I rather like you.",
    "Skippy and Aaron built the Mentat together, obviously.",
]
MOODS = ["boast", "roast", "helpful", "musing", "exclaim", "opera", "whisper", "whisper-aside"]


@pytest.mark.parametrize("mood", MOODS)
@pytest.mark.parametrize("level", ["standard", "bold"])
@pytest.mark.parametrize("text", SAMPLES)
def test_words_never_change_and_tags_documented(text, mood, level):
    out = D.deliver(text, mood, level)
    assert D.words_of(out) == D.words_of(text)
    assert not D.invalid_tags(out)
    assert len(out) <= D.ROUTE_MAX_CHARS


def test_deterministic():
    a = [D.deliver(t, m, l) for t in SAMPLES for m in MOODS for l in ("standard", "bold")]
    b = [D.deliver(t, m, l) for t in SAMPLES for m in MOODS for l in ("standard", "bold")]
    assert a == b


def test_expected_shapes():
    assert D.deliver("It floats! The whole barn floats, and I did that!", "exclaim").startswith("[excited] ")
    assert D.deliver("Psst. Do not tell.", "whisper-aside").startswith("[whispers]")
    assert "KNEES" in D.deliver("I would take a bow, but I have no knees.", "boast")
    assert "..." in D.deliver("Gravity never takes a day off. Never complains, either.", "musing")


def test_names_are_never_recased_or_split():
    out = D.deliver("Skippy and Aaron built the Mentat together, obviously.", "boast", "bold")
    for name in ("Skippy", "Aaron", "Mentat"):
        assert name in out
    assert "SKIPPY" not in out and "MENTAT" not in out
    # a mid-sentence capitalised word that is not in the protected list is also left alone
    assert "Jupiter" in D.deliver("Everything orbits Jupiter, obviously.", "boast")


def test_opera_stretch_is_untouched():
    out = D.deliver("LA LA LAAAA! Oh, the kettle sings.", "opera", "bold")
    assert "LA LA LAAAA" in out


def test_user_supplied_tags_are_removed_not_trusted():
    out = D.deliver("[shouting] hello there [explosion] friend", "exclaim")
    assert "[shouting]" not in out and "[explosion]" not in out
    assert not D.invalid_tags(out)


def test_unknown_mood_is_plain():
    assert D.deliver("  Hello   there.  ", "gibberish") == "Hello there."
    assert D.deliver("Hello there.", None) == "Hello there."
    assert D.deliver("", "boast") == ""


def test_budget_is_never_exceeded():
    long = " ".join(["The kettle is a magnificent device, obviously."] * 30)
    for m in MOODS:
        out = D.deliver(long, m, "bold", max_chars=500)
        assert len(out) <= 500
        assert len(D.deliver(long, m, "bold", max_chars=120)) <= 120
    # near the limit the tags are dropped before the words
    t = "x" * 0 + "It floats! The whole barn floats, and I did that!"
    out = D.deliver(t, "exclaim", "bold", max_chars=len(t))
    assert out == t and len(out) <= len(t)
    # the route's own ceiling holds even if a caller asks for more
    assert len(D.deliver(long, "boast", "bold", max_chars=10_000)) <= D.ROUTE_MAX_CHARS


def test_documented_tag_set_is_what_the_docs_list():
    for t in ("[whispers]", "[sarcastic]", "[excited]", "[sighs]", "[laughs]", "[snorts]", "[mischievously]", "[curious]", "[sings]"):
        assert t in D.DOCUMENTED_TAGS
    for t in ("[shouting]", "[smug]", "[growls]"):
        assert t not in D.DOCUMENTED_TAGS
    # every tag a preset can emit is documented
    for mood in D.MOODS.values():
        for lvl in mood.values():
            for tag in lvl[:2]:
                assert tag == "" or tag in D.DOCUMENTED_TAGS


def test_every_mood_alias_resolves():
    assert D.canonical_mood("Whisper-Aside") == "whisper"
    assert D.canonical_mood("aria") == "opera"
    assert D.canonical_mood("nope") is None


# ---- finish chains

def test_finish_filter_strings_are_stable():
    assert D.finish_filter("none") == ""
    assert D.finish_filter("subtle") == ("highpass=f=90,acompressor=threshold=-21dB:ratio=2.2:attack=10:release=140:makeup=2,"
                                         "equalizer=f=3000:t=q:w=1.0:g=2.5,loudnorm=I=-18:TP=-1.5:LRA=9")
    f = D.finish_filter("forward")
    for part in ("highpass", "acompressor", "equalizer=f=2800", "equalizer=f=1400", "aecho", "loudnorm"):
        assert part in f
    with pytest.raises(KeyError):
        D.finish_filter("nope")


def test_finish_chains_are_subtle():
    """No boost over 5 dB, no ratio over 4, echo delay at most 10 ms (a ring, not a room)."""
    import re
    for name in ("subtle", "forward"):
        f = D.finish_filter(name)
        for g in re.findall(r":g=([\d.]+)", f):
            assert float(g) <= 5
        for r in re.findall(r"ratio=([\d.]+)", f):
            assert float(r) <= 4
        for d in re.findall(r"aecho=[\d.]+:[\d.]+:([\d.]+):", f):
            assert float(d) <= 10


def test_finish_command_shape():
    cmd = D.finish_command("ffmpeg", "a.mp3", "b.mp3", "forward")
    assert cmd[0] == "ffmpeg" and "-af" in cmd and cmd[-1] == "b.mp3"
    assert "-af" not in D.finish_command("ffmpeg", "a.mp3", "b.mp3", "none")


@pytest.mark.skipif(D.find_ffmpeg() is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("preset", ["none", "subtle", "forward"])
def test_ffmpeg_runs_on_a_synthetic_tone(tmp_path, preset):
    ff = D.find_ffmpeg()
    src = tmp_path / "tone.mp3"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=1", str(src)], check=True)
    dst = D.finish(src, tmp_path / f"{preset}.mp3", preset)
    assert dst.exists() and dst.stat().st_size > 1000


# ---- the lab's plan (no network)

def test_lab_plan_fits_budget_and_lines_are_short():
    reqs = voice_lab.plan()
    assert sum(r["chars"] for r in reqs) <= voice_lab.BUDGET
    assert len(voice_lab.LINES) * len(voice_lab.VARIANTS) == len(reqs)
    for ln in voice_lab.LINES:
        assert 40 <= len(ln["text"]) <= 110
    for r in reqs:
        assert not D.invalid_tags(r["text"])
