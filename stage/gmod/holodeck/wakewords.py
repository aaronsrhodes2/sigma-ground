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


# Letters outside Latin (hex ranges identical to LETTERS in wakewords.lua). A name may be written in any of these scripts; it is
# matched EXACTLY (after case-folding and NFC), never fuzzily. Latin letters are folded as above.
_LETTERS = [(0x386, 0x386), (0x388, 0x38A), (0x38C, 0x38C), (0x38E, 0x3A1), (0x3A3, 0x3CE),            # Greek
            (0x400, 0x481), (0x48A, 0x52F),                                                            # Cyrillic
            (0x531, 0x556), (0x561, 0x586),                                                            # Armenian
            (0x5D0, 0x5EA),                                                                            # Hebrew
            (0x620, 0x63F), (0x641, 0x65F), (0x66E, 0x66F), (0x670, 0x6D3), (0x6FA, 0x6FC),            # Arabic
            (0x900, 0x939), (0x93A, 0x94D), (0x958, 0x961),                                            # Devanagari
            (0x985, 0x9B9), (0x9BC, 0x9CC), (0xB85, 0xBB9), (0xBBE, 0xBCD),                            # Bengali, Tamil
            (0xE01, 0xE3A), (0xE40, 0xE4E),                                                            # Thai
            (0x10D0, 0x10FA), (0x10FD, 0x10FF), (0x1100, 0x11FF), (0x1200, 0x135A),                    # Georgian, Hangul jamo, Ethiopic
            (0x3041, 0x3096), (0x309D, 0x309F), (0x30A1, 0x30FA), (0x30FC, 0x30FF), (0x3105, 0x312F),  # kana, Bopomofo
            (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0x20000, 0x2FA1D), (0xAC00, 0xD7A3)]                  # Han, Hangul syllables


def _is_letter(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _LETTERS)


def _lower_cp(cp: int) -> int:
    """Lower case of a Greek / Cyrillic / Armenian letter by explicit rules (the same table as Lua); everything else unchanged."""
    if 0x410 <= cp <= 0x42F:
        return cp + 32
    if 0x400 <= cp <= 0x40F:
        return cp + 80
    if 0x391 <= cp <= 0x3AB and cp != 0x3A2:
        return cp + 32
    if cp == 0x3C2:
        return 0x3C3
    if cp == 0x386:
        return 0x3AC
    if 0x388 <= cp <= 0x38A:
        return cp + 37
    if cp == 0x38C:
        return 0x3CC
    if cp in (0x38E, 0x38F):
        return cp + 63
    if 0x531 <= cp <= 0x556:
        return cp + 48
    if 0x460 <= cp <= 0x481 or 0x48A <= cp <= 0x4BF or 0x4D0 <= cp <= 0x52F:
        return cp + 1 if cp % 2 == 0 else cp
    if cp == 0x4C0:
        return 0x4CF
    if 0x4C1 <= cp <= 0x4CE:
        return cp + 1 if cp % 2 == 1 else cp
    return cp


_COMPOSE: dict = {(0x438, 0x306): 0x439, (0x435, 0x308): 0x451, (0x456, 0x308): 0x457, (0x443, 0x306): 0x45E,
                  (0x438, 0x300): 0x45D, (0x435, 0x300): 0x450, (0x433, 0x301): 0x453, (0x43A, 0x301): 0x45C,
                  (0x3B9, 0x308): 0x3CA, (0x3C5, 0x308): 0x3CB, (0x3046, 0x3099): 0x3094, (0x30A6, 0x3099): 0x30F4}
for _b, _t in zip((0x3B1, 0x3B5, 0x3B7, 0x3B9, 0x3BF, 0x3C5, 0x3C9), (0x3AC, 0x3AD, 0x3AE, 0x3AF, 0x3CC, 0x3CD, 0x3CE)):
    _COMPOSE[(_b, 0x301)] = _t
for _h in (0x304B, 0x304D, 0x304F, 0x3051, 0x3053, 0x3055, 0x3057, 0x3059, 0x305B, 0x305D, 0x305F, 0x3061, 0x3064, 0x3066, 0x3068,
           0x306F, 0x3072, 0x3075, 0x3078, 0x307B):
    for _off in (0, 0x60):
        _COMPOSE[(_h + _off, 0x3099)] = _h + _off + 1
for _h in (0x306F, 0x3072, 0x3075, 0x3078, 0x307B):
    for _off in (0, 0x60):
        _COMPOSE[(_h + _off, 0x309A)] = _h + _off + 2


def _compose(prev: int, cp: int):
    got = _COMPOSE.get((prev, cp))
    if got:
        return got
    if 0x1100 <= prev <= 0x1112 and 0x1161 <= cp <= 0x1175:                       # Hangul L+V
        return 0xAC00 + ((prev - 0x1100) * 21 + (cp - 0x1161)) * 28
    if 0xAC00 <= prev <= 0xD7A3 and (prev - 0xAC00) % 28 == 0 and 0x11A8 <= cp <= 0x11C2:   # LV+T
        return prev + (cp - 0x11A7)
    return None


def _utf8_len(cp: int) -> int:
    return 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4


def _chars(raw: str, start: int, limit_bytes: int) -> list[dict]:
    """The characters of raw from index `start`, a character being in when it begins within `limit_bytes` bytes of `start` (the
    Lua side counts bytes), each in plain form: {piece, cp, s, e} with s, e the first and last index in raw. A lone surrogate (what
    invalid UTF-8 becomes) is chr(1), like the Lua side's bad bytes."""
    out: list[dict] = []
    offset = 0
    for i in range(start, len(raw)):
        if offset > limit_bytes:
            break
        ch = raw[i]
        cp = ord(ch)
        offset += _utf8_len(cp) if not 0xD800 <= cp <= 0xDFFF else 1
        piece, lcp = None, None
        if 0xD800 <= cp <= 0xDFFF:
            piece = "\x01"
        elif cp < 128:
            piece = "'" if ch == "`" else ch.lower()
        elif ch in _QUOTES:
            piece = _QUOTES[ch]
        elif 0xC0 <= cp < 0x180 and cp not in (0xD7, 0xF7):
            if ch in _EXTRA:
                piece = _EXTRA[ch]
            else:
                a = "".join(c for c in unicodedata.normalize("NFKD", ch) if ord(c) < 128).lower()
                piece = a if a and a.isalpha() else "\x01"
        else:
            lo = _lower_cp(cp)
            prev = out[-1] if out else None
            if 0x300 <= cp <= 0x36F or cp in (0x3099, 0x309A):                    # a combining mark
                c = _compose(prev["cp"], cp) if prev and prev["cp"] else None
                if c:
                    prev.update(cp=c, piece=chr(c), e=i)
                    continue
                if prev and cp <= 0x36F and re.fullmatch(r"[a-z]+", prev["piece"]):
                    prev["e"] = i                                                   # an accent on a plain Latin letter is dropped
                    continue
                piece = "\x01"
            elif _is_letter(lo):
                c = _compose(prev["cp"], lo) if prev and prev["cp"] else None
                if c:                                                               # Hangul jamo join
                    prev.update(cp=c, piece=chr(c), e=i)
                    continue
                piece, lcp = chr(lo), lo
            else:
                piece = "\x01"
        out.append({"piece": piece, "cp": lcp, "s": i, "e": i})
    return out


def fold(s: str) -> str:
    return "".join(c["piece"] for c in _chars(str(s or ""), 0, 1 << 60))


def valid_text(s: str) -> bool:
    """True when the string could be UTF-8 (no lone surrogates: what bad bytes decode to)."""
    try:
        s.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


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
    if not valid_text(str(raw or "")):
        return None
    s = fold(raw)
    s = re.sub(r"^[\s'\"\-]+", "", s)
    s = re.sub(r"[\s'\"\-.,!?;:]+$", "", s)
    if "\x01" in s or re.search(r"[^a-z0-9' \-\u0080-\U0010ffff]", s):
        return None
    s = re.sub(r"\s+", " ", s)
    if not re.search(r"[a-z0-9\u0080-\U0010ffff]", s) or not (MIN_LEN <= len(s) <= MAX_LEN) or len(s.split()) > MAX_WORDS:
        return None
    return s


def _allowed(n: int) -> int:
    return min(3, n // 5)


def _tokens(raw: str, start: int) -> list[dict]:
    toks: list[dict] = []
    cur = None
    for c in _chars(raw, start, HEAD_BYTES):
        if re.fullmatch(r"[a-z0-9'\u0080-\U0010ffff]+", c["piece"]):
            if cur is None:
                cur = {"text": "", "s": c["s"]}
                toks.append(cur)
            cur["text"] += c["piece"]
            cur["e"] = c["e"]
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
        if acc[:1] == compact[:1] and abs(len(acc) - len(compact)) <= d and acc.isascii() and acc not in RESERVED and not (is_noun and is_noun(acc)):
            dist = osa(acc, compact)
            if dist <= d and (best_d is None or dist < best_d):
                best_k, best_d = k, dist
    return (best_k, False) if best_k else None


def _tidy(rest: str) -> str:
    """Drop what separates the name from the request: every leading character that is not a letter or digit."""
    i = 0
    while i < len(rest):
        c = _chars(rest, i, 0)
        if c and re.fullmatch(r"[a-z0-9\u0080-\U0010ffff]+", c[0]["piece"]):
            break
        i = (c[0]["e"] if c else i) + 1
    return rest[i:]


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
            c = _compact(nm)
            entries.append((nm, c, not c.isascii()))                  # a non-Latin name is matched exactly, never fuzzily
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
