"""Phrases to router calls, deterministic, no LLM: 'play thriller everywhere', 'quieter upstairs', 'resume my audiobook', 'stop everything'.

`parse(text, aliases)` is pure (text -> Intent). `execute(intent, router, library)` makes the router calls and returns a short spoken
sentence. `run(text, router, library)` does both. Typo tolerance: the first word is matched to the verb table and a trailing room word to
the target aliases with difflib (so 'plya', 'pasue', 'upstaris', 'everwhere' work); the thing to play is never "corrected" here, it goes to
the library resolver, which says 'the closest I have is X' instead of substituting.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

DEFAULT_ALIASES = {
    "here": ["downstairs", "pc", "desk", "this room", "hdmi", "computer", "in here"],
    "upstairs": ["up stairs", "enclave", "bluetooth", "the enclave"],
    "everywhere": ["every where", "all rooms", "whole house", "both", "everything", "all speakers", "all of them", "both speakers",
                   "all the speakers", "the whole house", "every room", "all"],
}
FILLER_LEAD = re.compile(r"^(?:(?:hey|ok|okay)\s+)?(?:skippy\s+)?(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
                         r"(?:i\s+(?:want|wanna|would like|d like|need)\s+(?:you\s+)?to\s+)?(?:i\s+(?:want|wanna)\s+)?")
PREPS = {"on", "in", "to", "through", "over", "at", "via", "from", "into", "around", "throughout", "across", "the", "just", "only", "and", "also"}

VERBS = {
    "play": ["play", "start", "put on", "listen to", "hear", "queue", "read me", "read", "shuffle"],
    "pause": ["pause", "hold", "freeze"],
    "resume": ["resume", "continue", "unpause", "carry on", "keep going", "keep reading", "go on"],
    "skip": ["skip", "next", "forward"],
    "prev": ["previous", "back", "rewind", "restart", "again", "last"],
    "stop": ["stop", "halt", "quit", "silence", "end", "cancel", "off", "kill"],
    "louder": ["louder", "raise", "increase", "up"],
    "quieter": ["quieter", "softer", "lower", "decrease", "down", "quiet"],
    "volume": ["volume", "set"],
    "mute": ["mute", "unmute"],
    "move": ["move", "send", "switch", "bring", "transfer", "route", "also"],
    "now": ["what", "whats", "who", "which"],
}
VERB_LOOKUP = {w: a for a, ws in VERBS.items() for w in ws if " " not in w}
NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
           "hundred": 100, "a": 1, "an": 1, "half": 0.5}


@dataclass
class Intent:
    action: str                       # play pause resume skip prev stop volume seek move now confirm decline pick unknown
    thing: str = ""
    kind: str | None = None           # 'book' | 'music' | None
    targets: list = field(default_factory=list)   # target names as in config ('here', 'upstairs', 'everywhere')
    only: bool = False
    level: float | None = None
    delta: float | None = None
    mute: bool | None = None
    seconds: float | None = None
    shuffle: bool = False
    pick: int | None = None
    text: str = ""


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b(what|who|where|that|it|there|here|he|she)'s\b", lambda m: m.group(1) + "s", t)
    t = re.sub(r"[^a-z0-9%' ]+", " ", t).replace("'", "")
    return re.sub(r"\s+", " ", t).strip()


def _fuzzy(word: str, choices, cutoff: float = 0.74):
    if word in choices:
        return word
    if len(word) < 4:
        return None
    m = difflib.get_close_matches(word, list(choices), n=1, cutoff=cutoff)
    return m[0] if m else None


def _alias_table(aliases: dict | None) -> dict[str, str]:
    table: dict[str, str] = {}
    for tgt, al in {**DEFAULT_ALIASES, **(aliases or {})}.items():
        table[tgt] = tgt
        for a in al:
            table[a] = tgt
    return table


def _strip_target(words: list[str], table: dict[str, str], allow_here_anywhere: bool) -> tuple[list[str], list[str], bool]:
    """Take a trailing (or leading, after the verb) room phrase off `words`. -> (rest, targets, only)."""
    only = False
    targets: list[str] = []
    w = list(words)
    if w and w[-1] == "only":
        only = True; w.pop()
    for n in (3, 2, 1):
        if len(w) >= n:
            cand = " ".join(w[-n:])
            hit = table.get(cand)
            if hit is None and n == 1 and len(cand) >= 5 and cand not in table:
                f = _fuzzy(cand, table, 0.8)
                hit = table.get(f) if f else None
            if hit is None and n == 2 and len(cand) >= 8:
                f = _fuzzy(cand, [k for k in table if " " in k], 0.85)
                hit = table.get(f) if f else None
            if hit and (hit != "here" or allow_here_anywhere or n == 1 or True):
                targets.append(hit); w = w[:-n]
                while w and w[-1] in PREPS:
                    if w[-1] == "only": only = True
                    w.pop()
                break
    if not targets and len(w) >= 2 and w[0] == "only":
        hit = table.get(w[1]) or table.get(_fuzzy(w[1], table, 0.8) or "")
        if hit:
            targets, only, w = [hit], True, w[2:]
    return w, targets, only


def _number(words: list[str]) -> float | None:
    for i, w in enumerate(words):
        if re.fullmatch(r"\d+(\.\d+)?", w):
            return float(w)
        if w in NUMBERS and w not in ("a", "an"):
            v = NUMBERS[w]
            if i + 1 < len(words) and words[i + 1] in NUMBERS and v >= 20 and NUMBERS[words[i + 1]] < 10:
                v += NUMBERS[words[i + 1]]
            return float(v)
    return None


def parse(text: str, aliases: dict | None = None) -> Intent:
    raw = text
    t = normalize(text)
    t = FILLER_LEAD.sub("", t).strip()
    t = re.sub(r"(?:\s+(?:please|thanks|thank you|now|for me))+$", "", t).strip()
    table = _alias_table(aliases)
    if not t:
        return Intent("unknown", text=raw)
    if t in ("yes", "yeah", "yep", "yes please", "sure", "ok", "okay", "do it", "play that", "go ahead", "please do", "that one", "thats the one"):
        return Intent("confirm", text=raw)
    if t in ("no", "nope", "never mind", "nevermind", "no thanks", "cancel that", "forget it"):
        return Intent("decline", text=raw)
    ordinals = {"first": 0, "1st": 0, "one": 0, "second": 1, "2nd": 1, "two": 1, "third": 2, "3rd": 2, "three": 2, "last": -1}
    m = re.fullmatch(r"(?:the |number )?(first|second|third|last|1st|2nd|3rd|one|two|three)(?: one)?", t)
    if m:
        return Intent("pick", pick=ordinals[m.group(1)], text=raw)
    if re.match(r"^(what|whats|who|which)\b.*\b(playing|song|track|this|listening|reading|book|sings|singing|on)\b", t) or t in ("whats on", "what is this"):
        return Intent("now", text=raw)

    words = t.split()
    # --- seek: "skip forward 30 seconds", "go back a minute", "rewind two minutes"
    mseek = re.search(r"\b(\d+|a|an|half|one|two|three|four|five|ten|fifteen|twenty|thirty|forty|fifty|sixty)(?: and a half)?\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours)\b", t)
    first = _fuzzy(words[0], VERB_LOOKUP) or words[0]
    if mseek and (first in ("skip", "forward", "rewind", "back", "go", "jump", "fast", "ff", "move") or "back" in words or "forward" in words or "ahead" in words):
        n = _number([mseek.group(1)]) or 1.0
        unit = mseek.group(2)
        sec = n * (3600 if unit.startswith("hour") else 60 if unit.startswith("min") else 1)
        back = first in ("rewind", "back") or "back" in words or "backwards" in words or "backward" in words
        w2, tg, only = _strip_target(words, table, True)
        return Intent("seek", seconds=-sec if back else sec, targets=tg, text=raw)

    verb = VERB_LOOKUP.get(first) or (VERB_LOOKUP.get(_fuzzy(first, VERB_LOOKUP) or ""))
    rest = words[1:] if verb else words
    # two-word openers
    two = " ".join(words[:2])
    if two in ("put on", "turn on", "listen to", "read me", "keep going", "keep reading", "carry on", "go on", "fast forward"):
        verb = {"put on": "play", "turn on": "play", "listen to": "play", "read me": "play", "keep going": "resume", "keep reading": "resume",
                "carry on": "resume", "go on": "resume", "fast forward": "skip"}[two]
        rest = words[2:]
    if two in ("turn up", "turn it up", "volume up", "louder please"):
        verb, rest = "louder", words[2:]
    if two in ("turn down", "volume down", "turn it down"):
        verb, rest = "quieter", words[2:]
    if two in ("shut up", "be quiet"):
        verb, rest = "stop", words[2:]
    if two in ("go back", "play previous", "previous track", "last track"):
        verb, rest = "prev", words[2:]
    if two == "play next" or two == "next track" or two == "next song":
        verb, rest = "skip", words[2:]
    if verb == "move" and rest and rest[0] in ("the", "this", "it", "music", "audio", "that"):
        pass
    if verb == "now":
        return Intent("now", text=raw)

    if verb in ("louder", "quieter", "volume", "mute") or any(w in words for w in ("louder", "quieter", "softer", "mute", "unmute", "volume")) \
            or re.search(r"\b(too (loud|quiet|soft)|turn (it )?(up|down))\b", t):
        rest_w, tg, only = _strip_target(words if not verb else rest, table, True)
        joined = " ".join(words)
        if "unmute" in words:
            return Intent("volume", mute=False, targets=tg, text=raw)
        if "mute" in words:
            return Intent("volume", mute=True, targets=tg, text=raw)
        mlev = re.search(r"(?:to|at)?\s*(\d{1,3})\s*(?:%|percent|per cent)?", " ".join(rest_w))
        if re.search(r"\bvolume\b", joined) and mlev and not re.search(r"\b(up|down)\b", joined) or re.search(r"\bset\b.*\b\d", joined):
            n = _number(rest_w)
            if n is not None:
                return Intent("volume", level=min(1.0, max(0.0, n / 100.0)), targets=tg, text=raw)
        big = 0.2 if re.search(r"\b(lot|much|way|loads)\b", joined) else 0.05 if re.search(r"\b(bit|little|touch|tad|slightly)\b", joined) else 0.1
        down = verb == "quieter" or bool(re.search(r"\b(quieter|softer|lower|down|decrease|too loud|quiet)\b", joined))
        if verb == "louder" or re.search(r"\b(louder|up|raise|increase|too (quiet|soft))\b", joined) or True:
            return Intent("volume", delta=-big if down else big, targets=tg, text=raw)

    if verb == "play" or verb is None and first not in ("pause", "stop"):
        # shuffle / audiobook / thing / target
        shuffle = first == "shuffle" or "shuffle" in rest
        rest = [w for w in rest if w != "shuffle"]
        body, tg, only = _strip_target(rest, table, True)
        kind = None
        if "audiobook" in body or "audiobooks" in body or first == "read" or two == "read me" or (body[:1] == ["book"]) or "book" in body[-2:]:
            kind = "book"
        while body and body[0] in ("my", "the", "some", "a", "an", "me", "to", "audiobook", "book", "audio", "music", "for", "of", "from", "something", "anything", "by"):
            if body[0] in ("something", "anything") and len(body) == 1:
                break
            body = body[1:]
        thing = " ".join(body).strip()
        if verb is None and not (tg or thing):
            return Intent("unknown", text=raw)
        if verb is None:
            return Intent("unknown", text=raw)
        if thing in ("", "something", "anything", "music", "audio", "stuff", "tunes", "songs"):
            thing = ""
        if thing in ("it", "that", "this") and verb == "play":
            thing = ""
        return Intent("play", thing=thing, kind=kind, targets=tg, only=only, shuffle=shuffle, text=raw)

    body, tg, only = _strip_target(rest, table, True)
    if verb == "pause":
        return Intent("pause", targets=tg, text=raw)
    if verb == "resume":
        return Intent("resume", kind="book" if re.search(r"book|reading|story", t) else None, targets=tg, text=raw)
    if verb == "skip":
        return Intent("skip", targets=tg, text=raw)
    if verb == "prev":
        return Intent("prev", targets=tg, text=raw)
    if verb == "stop":
        return Intent("stop", targets=[] if (not tg or tg == ["everywhere"]) else tg, only=only, text=raw)
    if verb == "move":
        return Intent("move", targets=tg, only=only, text=raw)
    return Intent("unknown", text=raw)


# ---------------------------------------------------------------------------------------------------------------- execution
def _where(router, names: list[str]) -> str:
    if not names:
        return ""
    s = set(names)
    allout = set(router.outputs)
    if s >= allout and len(s) > 1:
        return " everywhere"
    return " " + " and ".join(names)


def _tracks_for(entity):
    from audio_router import Track
    if entity.kind == "track":
        return [Track(path=entity.sources[0], title=entity.name, artist=entity.artist)]
    return [Track(path=p, title=_stem(p), artist=entity.artist if entity.kind == "album" else "") for p in entity.sources]


def _stem(p: str) -> str:
    return re.sub(r"^\d+[-_. ]+", "", p.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0])


def _play_entity(router, entity, targets, shuffle=False) -> str:
    kind = "book" if entity.book else None
    if entity.book:
        router.play(entity.sources[0], targets or None, kind="book")
    else:
        router.play(tracks=_tracks_for(entity), targets=targets or None, shuffle=shuffle or entity.kind == "artist")
    st = router.now()
    where = _where(router, st["targets"])
    lost = f" ({', '.join(st['lost'])} is not answering yet; I will keep trying)" if st["lost"] else ""
    return f"Playing {entity.say()}{where}.{lost}"


def execute(it: Intent, router, library=None) -> str:
    try:
        return _execute(it, router, library)
    except FileNotFoundError as e:
        return f"I cannot find that file: {e}"
    except (ValueError, KeyError) as e:
        return f"That did not work: {e}"


def _execute(it: Intent, router, library) -> str:
    a = it.action
    tg = it.targets
    pend = getattr(router, "pending", None)
    if a == "confirm":
        if pend and pend.get("entity"):
            router.pending = None
            return _play_entity(router, pend["entity"], pend["targets"], pend.get("shuffle", False))
        return "Confirm what? I was not offering anything."
    if a == "decline":
        router.pending = None
        return "Fine. Nothing changes."
    if a == "pick":
        if pend and pend.get("options"):
            opts = pend["options"]
            e = opts[it.pick] if -len(opts) <= it.pick < len(opts) else None
            if e:
                router.pending = None
                return _play_entity(router, e, pend["targets"])
        return "Pick from what? I did not ask."
    if a == "play":
        if not it.thing:
            if it.kind == "book":
                bp = router.last_book_path()
                if bp:
                    router.play(bp, tg or None, kind="book")
                    return f"Resuming your audiobook{_where(router, router.now()['targets'])}."
                return "I do not know of an audiobook you have started. Tell me which one."
            if router.paused:
                router.resume()
                return "Resuming."
            return "What would you like to hear?"
        if library is None:
            return "I have no music library loaded."
        r = library.resolve(it.thing, it.kind)
        if r.playable:
            return _play_entity(router, r.entity, tg, it.shuffle)
        router.pending = {"entity": r.entity if r.status == "closest" else None,
                          "options": r.options if r.status == "ambiguous" else [], "targets": tg, "shuffle": it.shuffle}
        return r.say
    if a == "pause":
        router.pause(); return "Paused."
    if a == "resume":
        if router.playing and router.paused:
            router.resume(); return "Resuming."
        if it.kind == "book" or not router.queue:
            bp = router.last_book_path()
            if bp:
                router.play(bp, tg or None, kind="book")
                return "Back to your audiobook, where you left off."
            return "There is nothing paused, and no audiobook I know of to resume."
        router.resume(); return "Resuming."
    if a == "skip":
        if not router.queue: return "Nothing is playing."
        router.skip(); n = router.now(); return f"Next: {n['track']}." if n["track"] else "That was the last one."
    if a == "prev":
        if not router.queue: return "Nothing is playing."
        router.prev(); return "Going back."
    if a == "seek":
        if not router.queue: return "Nothing is playing."
        router.seek(it.seconds, relative=True)
        return f"{'Back' if it.seconds < 0 else 'Forward'} {_dur(abs(it.seconds))}."
    if a == "stop":
        if tg:
            router.stop(tg)
            return f"Stopped {' and '.join(tg)}." + (" Still playing elsewhere." if router.playing else "")
        router.stop(); return "Stopped everything."
    if a == "volume":
        router.set_volume(tg or None, level=it.level, delta=it.delta, mute=it.mute)
        names = tg or router.wanted or [router.cfg["default_target"]]
        vols = router.now()["volumes"]
        shown = router.expand(names)
        v = max(vols[n] for n in shown)
        if it.mute is not None:
            return "Muted." if it.mute else "Unmuted."
        return f"Volume {round(v * 100)} percent{(' ' + ' and '.join(tg)) if tg else ''}."
    if a == "move":
        if not tg: return "Move it where?"
        if not router.playing: return "Nothing is playing to move."
        router.set_targets(tg)
        n = router.now()
        return f"Now playing{_where(router, n['targets'])}." + (f" {', '.join(n['lost'])} is not answering." if n["lost"] else "")
    if a == "now":
        n = router.now()
        if not n["track"]: return "Nothing is playing."
        return f"{'Paused on' if n['paused'] else 'Playing'} {n['track']}" + (f", {n['chapter']}" if n["chapter"] else "") + f",{_where(router, n['targets'])}."
    return "I did not understand that one."


def _dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 60} minute{'s' if s // 60 != 1 else ''}" if s >= 60 and s % 60 == 0 else f"{s} seconds"


def run(text: str, router, library=None) -> dict:
    it = parse(text, router.cfg.get("aliases"))
    say = execute(it, router, library) if it.action != "unknown" else "I did not catch that as an audio command."
    return {"say": say, "intent": it.action}
