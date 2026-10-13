"""Skippy's delivery: what to send to the voice (text with audio tags, caps, pauses) and how to finish the audio that comes back.

Two independent parts, both deterministic and offline:

1. deliver(text, mood, level, max_chars)  -> the TTS text. Moods: boast, roast, helpful, musing, exclaim, opera, whisper (alias
   whisper-aside). Only audio tags that ElevenLabs documents for Eleven v3 are ever added (DOCUMENTED_TAGS, checked 2026-10-13 against
   elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices). The words are never changed: only case (emphasis),
   punctuation (ellipses, dashes, sentence beats) and tags are touched; the result is verified word-for-word against the input and
   falls back to a plainer form if anything differs. Names are never capitalised or split. The result never exceeds max_chars
   (the route's limit, 500): if it does not fit, tags and tricks are dropped in order until it does.

2. finish_filter(preset) / finish(src, dst, preset) -> an ffmpeg filter chain for the 'forward, slightly tinny beer-can' colour:
   a high-pass, a gentle compressor, a presence boost near 3 kHz, a narrow resonance near 1.4 kHz and a very short echo, then
   loudness normalisation. Presets: none, subtle, forward. Every chain is gentle on purpose; the speech must stay natural.

Style only: this module tunes delivery and character. It never selects, trains on, or imitates a person's voice.

How the bridge would use it (not wired yet; see README "Skippy's delivery"):
    skippy_bridge.speak_elevenlabs(text) sends whatever text it is given, so the call site decides:
    in skippy_quips.py, Engine.quip, `clip, voice = self.synth(text)` becomes `self.synth(delivery.deliver(text, mood))`,
    and the clip cache key must stay the plain `text` (the cache is by spoken words, not by tags).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROUTE_MAX_CHARS = 500          # SkippyTel /tts/health max_chars

# The documented set (Eleven v3 best practices, 2026-10-13). Experimental ones are marked in the docs as less consistent per voice.
VOICE_TAGS = ("[laughs]", "[laughs harder]", "[starts laughing]", "[wheezing]", "[whispers]", "[sighs]", "[exhales]",
              "[sarcastic]", "[curious]", "[excited]", "[crying]", "[snorts]", "[mischievously]")
SOUND_TAGS = ("[gunshot]", "[applause]", "[clapping]", "[explosion]", "[swallows]", "[gulps]")
SPECIAL_TAGS = ("[sings]", "[woo]", "[fart]")            # experimental; [strong X accent] also exists and is not used here
DOCUMENTED_TAGS = frozenset(VOICE_TAGS + SOUND_TAGS + SPECIAL_TAGS)

TAG_RE = re.compile(r"\[[^\]]*\]")
WORD_RE = re.compile(r"[A-Za-z0-9']+")

# names and words that must never be recased or split
PROTECTED = frozenset(w.lower() for w in (
    "Skippy", "Aaron", "Captain", "Mentat", "SkippyTel", "SkippyVerse", "Sigma", "Garry", "Newton", "Einstein", "Joe", "Bilby"))

# words worth leaning on, in priority order
EMPHASIS_WORDS = ("never", "obviously", "genius", "magnificent", "brilliant", "perfect", "incredible", "absolutely", "completely",
                  "entirely", "nothing", "everything", "nobody", "wrong", "simply", "actually", "again", "still", "only", "best",
                  "greatest", "easy", "hard", "always", "not", "really", "very", "monkey")

LEVELS = ("standard", "bold")

# mood -> level -> (leading tag, trailing tag, max emphasis words, pace style). pace: 'tight' keeps sentences snappy, 'beat' puts
# an ellipsis between sentences, 'turn' also puts a beat at the turn of the joke (before the last comma clause) and after a short opener.
MOODS = {
    "boast":    {"standard": ("[mischievously]", "", 1, "tight"), "bold": ("[mischievously]", "", 2, "turn")},
    "roast":    {"standard": ("[sarcastic]", "", 1, "tight"),     "bold": ("[sarcastic]", "[snorts]", 2, "turn")},
    "helpful":  {"standard": ("[exhales]", "", 1, "beat"),        "bold": ("[sighs]", "", 2, "turn")},
    "musing":   {"standard": ("[curious]", "", 1, "beat"),        "bold": ("[sighs]", "", 1, "turn")},
    "exclaim":  {"standard": ("[excited]", "", 1, "tight"),       "bold": ("[excited]", "[laughs]", 2, "tight")},
    "opera":    {"standard": ("[sings]", "", 0, "tight"),         "bold": ("[sings]", "", 0, "beat")},
    "whisper":  {"standard": ("[whispers]", "", 0, "beat"),       "bold": ("[whispers]", "", 0, "turn")},
}
ALIASES = {"whisper-aside": "whisper", "whisper_aside": "whisper", "aside": "whisper", "aria": "opera"}


def canonical_mood(mood: str | None) -> str | None:
    m = (mood or "").strip().lower()
    m = ALIASES.get(m, m)
    return m if m in MOODS else None


# ----------------------------------------------------------------------------------------------------------------- text helpers

def strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", text)).strip()


def words_of(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(TAG_RE.sub(" ", text))]


def tags_in(text: str) -> list[str]:
    return TAG_RE.findall(text)


def invalid_tags(text: str) -> list[str]:
    """Tags in `text` that are not in the documented set."""
    return [t for t in tags_in(text) if t not in DOCUMENTED_TAGS]


def clean_input(text: str) -> str:
    """Whitespace squeezed; tags the writer put in are removed (the layer is the only thing that adds tags)."""
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", str(text or ""))).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _is_name(word: str, sentence_start: bool) -> bool:
    bare = word.strip(".,!?;:'\"").rstrip("'s").lower()
    if bare in PROTECTED or word.strip(".,!?;:'\"").lower() in PROTECTED:
        return True
    return word[:1].isupper() and not sentence_start


def emphasise(text: str, n: int) -> str:
    """Capitalise up to n emphasis words (first occurrence each); names and already-shouted words are skipped. Only case changes."""
    if n <= 0:
        return text
    tokens = text.split(" ")
    start_flags = []
    prev_end = True
    for t in tokens:
        start_flags.append(prev_end)
        prev_end = bool(re.search(r"[.!?]$", t))
    chosen: set[int] = set()
    for target in EMPHASIS_WORDS:
        if len(chosen) >= n:
            break
        for i, t in enumerate(tokens):
            core = t.strip(".,!?;:'\"-")
            if core.lower() == target and not core.isupper() and not _is_name(t, start_flags[i]):
                chosen.add(i)
                break
    if not chosen:                                          # fallback: lean on the last long plain word of the last sentence
        for i in range(len(tokens) - 1, -1, -1):
            core = tokens[i].strip(".,!?;:'\"-")
            if len(core) >= 5 and core.isalpha() and not core.isupper() and not _is_name(tokens[i], start_flags[i]):
                chosen.add(i)
                break
    out = []
    for i, t in enumerate(tokens):
        if i in chosen:
            m = re.match(r"^(\W*)([A-Za-z']+)(.*)$", t)
            t = (m.group(1) + m.group(2).upper() + m.group(3)) if m else t
        out.append(t)
    return " ".join(out)


def pace(text: str, style: str) -> str:
    """Punctuation only. tight: sentences as they are. beat: an ellipsis between sentences. turn: beat, plus a pause after a short
    opener and at the turn of the joke (the last comma of a long sentence becomes an ellipsis)."""
    sents = split_sentences(text)
    if not sents:
        return text
    if style in ("beat", "turn"):
        fixed = []
        for k, s in enumerate(sents):
            if style == "turn":
                s = re.sub(r"^(\w+),\s+", r"\1... ", s, count=1) if len(s.split(",")[0].split()) <= 2 and "," in s else s
                if len(s.split()) >= 8 and s.count(",") >= 1 and "..." not in s:
                    head, _, tail = s.rpartition(",")
                    s = head + "..." + tail
            if k < len(sents) - 1 and s[-1:] in ".!?" and not s.endswith("..."):
                s = s[:-1] + "..." if s[-1] == "." else s + "..."
            fixed.append(s)
        sents = fixed
    return " ".join(sents).replace("...  ", "... ").replace("....", "...")


def _compose(text: str, lead: str, trail: str, emph: int, style: str) -> str:
    body = pace(emphasise(text, emph), style) if style else emphasise(text, emph)
    return " ".join(p for p in (lead, body, trail) if p)


def _cut_to_sentences(text: str, limit: int) -> str:
    out = ""
    for s in split_sentences(text):
        cand = (out + " " + s).strip()
        if len(cand) > limit:
            break
        out = cand
    return out or text[:limit].rsplit(" ", 1)[0].rstrip(",;:-") or text[:limit]


def deliver(text: str, mood: str | None = None, level: str = "standard", max_chars: int = ROUTE_MAX_CHARS) -> str:
    """The text to send to the TTS. Never longer than max_chars, never different words, only documented tags.
    Unknown or missing mood -> plain cleaned text. level 'plain' returns the cleaned text untouched."""
    plain = clean_input(text)
    max_chars = max(1, min(int(max_chars), ROUTE_MAX_CHARS))
    m = canonical_mood(mood)
    if not plain or m is None or level == "plain":
        return plain if len(plain) <= max_chars else _cut_to_sentences(plain, max_chars)
    if level not in LEVELS:
        level = "standard"
    lead, trail, emph, style = MOODS[m][level]
    if m == "opera":
        emph = 0                                           # the stretched capitals are the performance; do not add more
    # candidates from richest to plainest; first that fits and keeps the words wins
    cands = [
        _compose(plain, lead, trail, emph, style),
        _compose(plain, lead, "", emph, style),
        _compose(plain, lead, "", emph, "tight"),
        _compose(plain, lead, "", 0, "tight"),
        plain,
    ]
    want = words_of(plain)
    for c in cands:
        if len(c) <= max_chars and words_of(c) == want and not invalid_tags(c):
            return c
    return _cut_to_sentences(plain, max_chars)


# ----------------------------------------------------------------------------------------------------------------- finishing

FINISH_PRESETS = {
    "none": [],
    # a hint of body control and presence; hardly colour
    "subtle": [
        "highpass=f=90",
        "acompressor=threshold=-21dB:ratio=2.2:attack=10:release=140:makeup=2",
        "equalizer=f=3000:t=q:w=1.0:g=2.5",
        "loudnorm=I=-18:TP=-1.5:LRA=9",
    ],
    # forward and a little tinny: thinner low end, presence at 3 kHz, a narrow ring near 1.4 kHz, a 6 ms metallic echo
    "forward": [
        "highpass=f=140",
        "acompressor=threshold=-22dB:ratio=3.2:attack=6:release=110:makeup=3",
        "equalizer=f=2800:t=q:w=1.1:g=4",
        "equalizer=f=1400:t=q:w=5:g=2.5",
        "aecho=0.85:0.9:6:0.22",
        "loudnorm=I=-17:TP=-1.5:LRA=8",
    ],
}


def finish_filter(preset: str) -> str:
    """The -af string for a preset ('' for none). Unknown preset raises KeyError."""
    return ",".join(FINISH_PRESETS[preset])


FFMPEG_FALLBACK = Path("C:/Users/aaron/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
                       "ffmpeg-8.1-full_build/bin/ffmpeg.exe")


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    return str(FFMPEG_FALLBACK) if FFMPEG_FALLBACK.exists() else None


def finish_command(ffmpeg: str, src, dst, preset: str) -> list[str]:
    """The ffmpeg argv (deterministic). 'none' re-encodes without a filter so every variant is the same format."""
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src)]
    af = finish_filter(preset)
    if af:
        cmd += ["-af", af]
    cmd += ["-codec:a", "libmp3lame", "-q:a", "2", str(dst)]
    return cmd


def finish(src, dst, preset: str) -> Path:
    import subprocess
    ff = find_ffmpeg()
    if ff is None:
        raise RuntimeError("ffmpeg not found")
    subprocess.run(finish_command(ff, src, dst, preset), check=True, capture_output=True)
    return Path(dst)
