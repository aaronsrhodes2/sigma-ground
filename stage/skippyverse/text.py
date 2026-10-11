"""Text normalisation shared by the index and the resolver: lower-casing, simple singularisation, edit distance, and a small
request-phrase parser that pulls the object phrase (and a count) out of 'Skippy, add three couches'.

Everything here is deterministic string handling; the same `norm` is applied when the index is built and when a query arrives, which
is what makes plurals work."""
from __future__ import annotations

import re

_IRREGULAR = {"men": "man", "women": "woman", "children": "child", "feet": "foot", "teeth": "tooth", "mice": "mouse",
              "shelves": "shelf", "knives": "knife", "wolves": "wolf", "leaves": "leaf", "loaves": "loaf", "halves": "half",
              "lives": "life", "wives": "wife", "thieves": "thief", "shoes": "shoe", "sofas": "sofa", "bus": "bus", "glasses": "glass",
              "lenses": "lens", "boxes": "box", "buses": "bus", "dishes": "dish", "benches": "bench", "wardrobes": "wardrobe",
              "couches": "couch", "radios": "radio", "pianos": "piano", "tomatoes": "tomato", "potatoes": "potato", "heroes": "hero",
              "stoves": "stove", "toilets": "toilet", "microwaves": "microwave", "tables": "table", "vases": "vase", "plates": "plate"}
_KEEP = {"bus", "gas", "glass", "lens", "series", "species", "news", "grass", "mattress", "boss", "cross", "dress", "tennis", "chess",
         "canvas", "atlas", "bias", "iris", "pancreas", "plus", "its", "this", "is", "us", "as", "has", "was", "his", "yes", "oasis",
         "sunglass", "tvs"}


def singular(w: str) -> str:
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w in _KEEP or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("ss") or w.endswith("us") or w.endswith("is"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def tokens(s: str) -> list[str]:
    s = (s or "").lower().replace("'", "").replace("’", "")
    return re.findall(r"[a-z0-9]+", s)


def norm(s: str) -> str:
    """lower-case, punctuation to spaces, each word singularised: 'Bookshelves' -> 'bookshelf', 'Coffee Tables' -> 'coffee table'."""
    return " ".join(singular(t) for t in tokens(s))


def damerau(a: str, b: str) -> int:
    """Optimal-string-alignment edit distance (insert, delete, substitute, adjacent swap)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


# --- request parsing ----------------------------------------------------------------------------------------------------------
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "dozen": 12, "couple": 2, "pair": 2}
SIZE_UP = {"big", "large", "huge", "giant", "bigger", "larger", "biggest", "largest", "enormous", "massive"}
SIZE_DOWN = {"small", "little", "tiny", "mini", "smaller", "smallest", "compact"}
COLOURS = {"red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "grey", "gray", "brown", "gold", "silver"}
# modifiers the resolver may drop, because they change the look or size of the thing, not what it is
MODIFIERS = SIZE_UP | SIZE_DOWN | COLOURS | {"old", "new", "nice", "cheap", "fancy", "plain", "shiny", "dirty", "clean", "tall", "short",
                                              "long", "wide", "narrow", "wooden", "leather", "metal", "plastic", "simple", "normal",
                                              "regular", "standard", "ordinary", "random", "whole", "real", "single"}
_FILLER = re.compile(
    r"^(?:hey|ok|okay|so|um|uh|well|right|now|yo)\b[ ,:]*|^skippy\b[ ,:]*|^(?:please|pls)\b[ ,]*|"
    r"^(?:can|could|would|will|may)\s+(?:you|i|we)(?:\s+please)?\b[ ,]*|^(?:id|we?d|i|we)\s+(?:would\s+)?(?:like|want|need|wanna|love)(?:\s+to\s+(?:have|get|see))?\b[ ,]*|"
    r"^(?:lets|let\s+us)\b[ ,]*|^(?:i\s+(?:wanna|gotta)|gimme|lemme)\b[ ,]*|"
    r"^(?:add|make|give|spawn|create|place|put|get|show|bring|build|summon|conjure|fetch|find|grab|drop|throw|set\s*up|setup|have|see|use|need|want)\b[ ,]*|"
    r"^(?:me|us|in|here|there|over\s+there|out)\b[ ,]*", re.I)
_DET = re.compile(r"^(?:a|an|the|some|another|any|few|several|more|that|this|those|these|my|your|one\s+more)\b[ ,]*", re.I)
_CUT = re.compile(r"\b(?:with|next\s+to|beside|near|nearby|in|on|at|by|for|that|which|where|facing|against|and|over|under|into|from|"
                  r"please|here|there|now|around|behind|between|inside|outside|to\s+me|to\s+us|right\s+now|thanks|thank\s+you)\b.*$", re.I)


def parse_query(q: str) -> dict:
    """'Skippy, add three red couches next to the wall' -> {'phrase': 'red couch', 'count': 3, 'modifiers': ['red'], 'size_pref': None,
    'raw_phrase': 'red couches'}. Never invents: if nothing is left, phrase is ''."""
    s = (q or "").lower().replace("'", "").replace("’", "")
    s = re.sub(r"[?!.]+$", "", s.strip())
    s = re.sub(r"\s+", " ", s)
    for _ in range(8):                                              # strip leading filler and verbs repeatedly
        n = _FILLER.sub("", s, count=1).strip()
        if n == s:
            break
        s = n
    pre_count = s
    count = None
    m = re.match(r"^(\d+)\s*(?:x\s*)?", s)
    if m:
        count, s = int(m.group(1)), s[m.end():].strip()
    else:
        m = re.match(r"^(?:a\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|couple|pair)\b(?:\s+of)?\s*", s)
        if m:
            count, s = NUMBER_WORDS[m.group(1)], s[m.end():].strip()
    s = _DET.sub("", s, count=1).strip()
    s = _DET.sub("", s, count=1).strip()
    s = _CUT.sub("", s).strip(" ,;:")
    raw_phrase = s
    toks = s.split()
    mods = []
    while toks and toks[0] in MODIFIERS and len(toks) > 1:
        mods.append(toks.pop(0))
    size_pref = "up" if any(m in SIZE_UP for m in mods) else "down" if any(m in SIZE_DOWN for m in mods) else None
    # the phrase with the number left in, for names that begin with one ("two seater", "2 seater")
    with_number = norm(_CUT.sub("", _DET.sub("", pre_count, count=1).strip()).strip(" ,;:")) if count is not None else None
    return {"phrase": norm(" ".join(toks)), "raw_phrase": raw_phrase, "count": count, "modifiers": mods, "size_pref": size_pref,
            "with_number": with_number}
