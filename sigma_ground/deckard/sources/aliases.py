"""Name aliases from the ShapeNet/WordNet taxonomy — widen the sources' reach.

``expand(name)`` returns the OTHER lemmas of any WordNet group the (normalized)
name belongs to ("cellphone" -> {"cellular telephone", "mobile phone", ...}), so
a source keyed under one lemma still answers a query phrased with another. The
table (``inventory/data/shape_aliases.json``) ships vocabulary only — distilled
offline by ``tools/distill_shapenet_taxonomy.py``, no model data.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

_JSON = (pathlib.Path(__file__).resolve().parents[2]
         / "inventory" / "data" / "shape_aliases.json")
_ARTICLES = re.compile(r"^(a|an|the)\s+", re.I)


def _norm(s: str) -> str:
    return _ARTICLES.sub("", (s or "").strip().lower())


@functools.lru_cache(maxsize=1)
def _by_lemma() -> dict:
    try:
        groups = json.loads(_JSON.read_text(encoding="utf-8")).get("groups") or []
    except Exception:
        return {}
    out: dict[str, set] = {}
    for g in groups:
        gs = {str(w).lower() for w in g}
        for w in gs:
            out.setdefault(w, set()).update(gs - {w})
    return out

def expand(name: str) -> set:
    """Alternative names for ``name`` (possibly empty); never includes itself."""
    q = _norm(name)
    found = set(_by_lemma().get(q) or ())
    return found - {q}


# Words that open a prepositional / clausal tail: "a chair WITH arms", "a lamp FOR reading". The head
# of the phrase is the last word BEFORE the tail.
_TAIL_WORDS = {"of", "with", "for", "in", "on", "from", "without", "containing", "that", "which",
               "who", "using", "holding"}
# "a pair of scissors", "a set of chairs": the quantity word is not the head, the noun after 'of' is.
_QUANTITY_WORDS = {"pair", "set", "couple", "bunch", "stack", "pile", "row", "group", "collection",
                   "dozen", "lot", "bundle", "series"}
# A head that names no particular kind of thing: "a tv SET", "a chair UNIT", "a lamp MODEL".
# A query whose head is one of these is judged by the word before it.
GENERIC_HEADS = {"set", "unit", "piece", "model", "thing", "object", "item", "type", "kind", "style",
                 "design", "version", "variant", "example"}


def _phrase_words(name: str) -> list:
    """Words of the head phrase: the text before any comma / parenthesis, split on non-alphanumerics
    (so "The Coffee-Table" is coffee table), minus the prepositional tail, and with "a pair of X"
    reduced to X."""
    head_part = re.split(r"[,;(]", (name or "").lower(), maxsplit=1)[0]
    head_part = re.sub(r"\b([a-z]+)-(in|on|out)\b", r"\1\2", head_part)   # "walk-in closet": one word, not a tail
    ws = [w for w in re.split(r"[^a-z0-9]+", head_part) if w]
    for i in range(1, len(ws)):
        if ws[i] in _TAIL_WORDS:
            if ws[i] == "of" and ws[i - 1] in _QUANTITY_WORDS and i + 1 < len(ws):
                return _phrase_words(" ".join(ws[i + 1:]))        # "a pair of red scissors" -> red scissors
            return ws[:i]
    return ws


def head_noun(name: str) -> str:
    """The head of a noun phrase: its LAST word ("desk LAMP" is a lamp, "coffee
    TABLE" a table) — English compounds put the head last and the modifiers
    before it — taken before any prepositional tail ("a chair WITH arms" is a chair,
    not arms) and past a quantity word ("a pair of SCISSORS"). "" for an empty name."""
    ws = _phrase_words(name)
    return ws[-1] if ws else ""


def holds_head(words: set, head: str) -> bool:
    """True if a key's ``words`` contain the query's ``head`` noun (singular or
    plural: "two desk lamps"). The tie-break between equally specific matches:
    when "desk" and "lamp" both fit "a desk lamp", the one with the head wins."""
    return bool(head) and (head in words or (head.endswith("s") and head[:-1] in words)
                           or (head.endswith("es") and head[:-2] in words))


def names_head(words: set, name: str) -> bool:
    """True if a key's ``words`` are about the same KIND of thing as ``name``: they hold its head
    noun, or its head noun is generic ("a tv SET": the kind is the word before it). A key that
    matches only a MODIFIER ("tv" in "a tv stand", "plant" in "a plant cell", "shelf" in "a shelf
    bracket") is another kind of thing and does not."""
    ws = _phrase_words(name)
    if not ws:
        return False
    head = ws[-1]
    if head in GENERIC_HEADS and len(ws) > 1:
        head = ws[-2]
    return holds_head(words, head)


__all__ = ["expand", "head_noun", "holds_head", "names_head", "GENERIC_HEADS"]
