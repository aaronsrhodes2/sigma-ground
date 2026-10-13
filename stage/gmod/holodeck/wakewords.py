"""wakewords.py -- the Python twin of addon/lua/mentat/holodeck/wakewords.lua (the names Skippy answers to, one list per player).

The microphone bridge (skippy_ears.py) runs on the player's own PC and decides which recognised sentences are sent on to the game, so it
must wake on the same names the server does. The two files are held to the same table of sentences
(sigma_ground/radiance/tests/wake_vectors.json) and to the same accent folding. Pure functions, no I/O except load_names().

    strip(text, names, default_strip=None) -> (woke, rest, name)
    normalise_name(raw)                   -> name | None
    load_names(path)                      -> the list in the client's wake_names.json, or the default ["skippy"]
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

DEFAULT = "skippy"
MAX_NAMES, MIN_LEN, MAX_LEN, MAX_WORDS = 8, 2, 24, 3
HEAD_BYTES, HEAD_TOKENS, MAX_SPAN = 160, 14, 4
FILE_MAX_BYTES = 4096                       # the file the game writes is a few hundred bytes; anything bigger is not ours
SKIPPY_VARIANTS = ["skippy", "skipy", "skippie", "skipie", "skipi", "skippi", "skipee", "skippee", "skippey", "skipey", "skeepy",
                   "skeepi", "skeepie"]


def _set(s: str) -> frozenset:
    return frozenset(s.split())


RESERVED = _set("""
add put remove delete clear undo make build place create spawn give get bring move turn rotate show set erase destroy summon please can could
would will should shall may might must do does did is are was were am be been have has had let lets want need like try just stop start
come go here there away dismiss vanish leave appear follow hide help look see watch tell say ask answer answering respond responding listen
call calling name names forget reset only also stay sit stand open close pick take drop throw push pull lift hold keep cut paint colour color
yes no ok okay hey hi hello yo oi so well um uh er erm hmm oh ah now then and but or if when what who why how where which whose
the a an this that these those my your our its his her their it them they he she we you me us him i mine yours
one two three four five six seven eight nine ten eleven twelve twenty thirty hundred another again more some any every all both each
everything something anything nothing everyone nobody other next last first big small tiny huge little long short tall
left right front back up down over under near far inside outside
couch sofa chair table lamp bed desk door light lights tv television shelf bookshelf stool bench rug plant fridge oven sink toilet bath mirror
pairs skipping skipped skimpy skippers""")
LEAD = _set("""hey hi hello hiya howdy yo oi ok okay so um umm uh uhh er erm hmm hm well alright all right there please excuse me oh ah now
then and but yes yeah sure""")
ORDERS = _set("""add put remove delete clear undo make build place create spawn give get bring move turn rotate show set erase destroy
summon please can could would will""")

_EXTRA = {"ß": "ss", "æ": "ae", "Æ": "ae", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "œ": "oe", "Œ": "oe",
          "þ": "th", "Þ": "th", "ð": "d", "Ð": "d", "ı": "i"}
_QUOTES = {"‘": "'", "’": "'", "‛": "'", "ʼ": "'", "′": "'", "´": "'", "“": '"', "”": '"',
           "«": '"', "»": '"', "–": "-", "—": "-", " ": " ", "…": "."}


def fold_char(ch: str) -> str:
    """One character to its plain form: ASCII lower-cased, a typographic quote or dash as its ASCII twin, an accented Latin letter
    without its accent, anything else as chr(1) (never a letter)."""
    if len(ch) == 1 and ord(ch) < 128:
        return "'" if ch == "`" else ch.lower()
    if ch in _QUOTES:
        return _QUOTES[ch]
    if 0xC0 <= ord(ch) < 0x180 and ord(ch) not in (0xD7, 0xF7):
        if ch in _EXTRA:
            return _EXTRA[ch]
        a = "".join(c for c in unicodedata.normalize("NFKD", ch) if ord(c) < 128).lower()
        if a and a.isalpha():
            return a
    return "\x01"


def fold(s: str) -> str:
    return "".join(fold_char(c) for c in str(s or ""))


def osa(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            v = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, d[i - 2][j - 2] + 1)
            d[i][j] = v
    return d[la][lb]


def _compact(name: str) -> str:
    return re.sub(r"[\s\-']", "", name)


def normalise_name(raw: str) -> str | None:
    """A name as typed or said -> its stored form, or None when it is not a usable name (same rules as the Lua side)."""
    s = fold(raw)
    s = re.sub(r"^[\s'\"\-]+", "", s)
    s = re.sub(r"[\s'\"\-.,!?;:]+$", "", s)
    if "\x01" in s or re.search(r"[^a-z0-9' \-]", s):
        return None
    s = re.sub(r"\s+", " ", s)
    if not re.search(r"[a-z0-9]", s) or not (MIN_LEN <= len(s) <= MAX_LEN) or len(s.split()) > MAX_WORDS:
        return None
    return s


def _allowed(n: int) -> int:
    return min(3, n // 5)


def _tokens(raw: str, start: int) -> list[dict]:
    toks: list[dict] = []
    cur = None
    # positions are character indexes; the head is cut at HEAD_BYTES *bytes* in Lua, here at as many characters (ASCII-equal)
    end = min(len(raw), start + HEAD_BYTES + 1)
    for i in range(start, end):
        piece = fold_char(raw[i])
        if re.fullmatch(r"[a-z0-9']+", piece):
            if cur is None:
                cur = {"text": "", "s": i}
                toks.append(cur)
            cur["text"] += piece
            cur["e"] = i
        else:
            cur = None
    out: list[dict] = []
    for t in toks:
        w = t["text"].strip("'")
        m = re.fullmatch(r"(.*)'s", w)
        if m and m.group(1):
            t["poss"], w = True, m.group(1)
        if w and len(out) < HEAD_TOKENS:
            t["word"], t["compact"] = w, w.replace("'", "")
            out.append(t)
    return out


def _match_at(toks, i, compact, exact_only, is_noun):
    acc = ""
    for k in range(1, min(MAX_SPAN, len(toks) - i) + 1):
        acc += toks[i + k - 1]["compact"]
        if acc == compact:
            return k, True
        if len(acc) > len(compact) + 3:
            break
    d = _allowed(len(compact))
    if exact_only or d == 0:
        return None
    acc, best_k, best_d = "", None, None
    for k in range(1, min(MAX_SPAN, len(toks) - i) + 1):
        acc += toks[i + k - 1]["compact"]
        if acc[:1] == compact[:1] and abs(len(acc) - len(compact)) <= d and acc not in RESERVED and not (is_noun and is_noun(acc)):
            dist = osa(acc, compact)
            if dist <= d and (best_d is None or dist < best_d):
                best_k, best_d = k, dist
    return (best_k, False) if best_k else None


def _tidy(rest: str) -> str:
    return re.sub(r"^[\s\W_]+", "", rest)


def strip(text: str, names=None, default_strip=None, is_noun=None) -> tuple[bool, str, str | None]:
    """Does the sentence START with one of `names`? (True, the request without the name, the name) or (False, text, None)."""
    raw = str(text or "")
    m0 = re.search(r"\S", raw)
    start = m0.start() if m0 else 0
    names = list(names or [DEFAULT])
    entries, has_default = [], False
    for nm in names:
        if nm == DEFAULT:
            has_default = True
            if default_strip is None:
                entries += [(DEFAULT, _compact(v), True) for v in SKIPPY_VARIANTS]
        else:
            entries.append((nm, _compact(nm), False))
    if has_default and default_strip is not None:
        ok, rest = default_strip(raw)
        if ok:
            return True, rest, DEFAULT
    if not entries:
        return False, raw, None
    toks = _tokens(raw, start)
    for lead in range(4):
        if lead > 0 and not (len(toks) >= lead and toks[lead - 1]["word"] in LEAD):
            break
        i = lead
        if i >= len(toks):
            break
        best = None
        for name, compact, exact_only in entries:
            got = _match_at(toks, i, compact, exact_only, is_noun)
            if got:
                k, exact = got
                if best is None or (exact and not best[1]) or (exact == best[1] and k > best[0]):
                    best = (k, exact, name)
        if best:
            last = toks[i + best[0] - 1]
            after = raw[last["e"] + 1:]
            if last.get("poss"):
                m = re.match(r"[^A-Za-z0-9]*([A-Za-z0-9]+)", after)
                if m and m.group(1).lower() in ORDERS:
                    return True, _tidy(after), best[2]
                return False, raw, None
            return True, _tidy(after), best[2]
    return False, raw, None


def parse_names(obj) -> list[str]:
    """The names in a decoded wake_names.json, validated; the default when nothing usable is there."""
    out: list[str] = []
    names = obj.get("names") if isinstance(obj, dict) else None
    if isinstance(names, list):
        for n in names[:MAX_NAMES]:
            if isinstance(n, str):
                nn = normalise_name(n)
                if nn and nn not in out:
                    out.append(nn)
    return out or [DEFAULT]


def load_names(path: Path) -> list[str]:
    """Read the client's names file; missing, huge or corrupt -> the default list (the bridge keeps working)."""
    try:
        if path.stat().st_size > FILE_MAX_BYTES:
            return [DEFAULT]
        return parse_names(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return [DEFAULT]
