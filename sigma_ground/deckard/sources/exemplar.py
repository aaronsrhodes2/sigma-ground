"""Representative real-object part layouts, distilled from PartNet.

For each category, ``tools/distill_partnet.py exemplar`` picked representative
real model(s) and recorded each one's leaf parts as primitives at their REAL
positions (normalized to the model's bbox). ``exemplar_of(name)`` returns one
such layout, so Deckard assembles "a chair" from how a real chair is actually
built — legs where they really are — rather than from a hand-written template.

When a category ships MORE THAN ONE exemplar (a variant POOL — distinct real
models with the same category name + distinct ``source_anno_id``), the selector
can pick among them:
  * ``exclude`` — a set of anno_ids already used, so "a *different* chair" grabs
    a fresh model rather than the one we already solved;
  * ``hint`` — prefers a model tagged with a subtype/material ("office", "wood").
If exclusion would empty the pool, the best remaining is returned (reuse) and
the caller can detect it by the returned anno_id. The SHAPE is always a real
model's layout — never fabricated to satisfy a request.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re

_DIR = (pathlib.Path(__file__).resolve().parents[2]
        / "inventory" / "data" / "exemplars")


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w}


@functools.lru_cache(maxsize=1)
def _table() -> list:
    """Every shipped exemplar as (category_names, parts, source, license,
    anno_id, tags). Multiple files may share a category → a variant pool."""
    out = []
    if not _DIR.is_dir():
        return out
    for p in sorted(_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        parts = d.get("parts") or []
        if not parts:
            continue
        names = {(d.get("category") or p.stem).lower()} | {
            str(a).strip().lower() for a in (d.get("aliases") or [])}
        tags = {str(t).strip().lower() for t in (d.get("tags") or [])}
        out.append((names, parts, d.get("_source", ""), d.get("_license", ""),
                    str(d.get("source_anno_id", "")), tags, (d.get("category") or p.stem).lower()))
    return out


def _tag_match(tags: set, hint: str) -> bool:
    """True if any word of the hint appears in any of the model's tags."""
    hw = _words(hint or "")
    return any(any(w in t for w in hw) for t in tags)


# ── Which names may stand for an exemplar category ────────────────────────────────────────────
# A category name or alias identifies the query only when it is the HEAD of the phrase (``aliases.
# names_head``): "a plant" and "a potted plant" are a plant, "a plant stand", "a plant cell" and "a tv
# stand" are not — the alias is a modifier there and the thing is a stand, a cell, a stand. Aliases that
# are also the head of well-known OTHER things ("power PLANT", "tea POT", "water CLOSET", "heart MONITOR",
# "bottle CAP", "tea BAG") are POLYSEMOUS: they identify the category only when every other word of the
# query is neutral (an article, a number, a size, a colour, a material) or one of the extra words listed
# here for that alias. A closed list, on purpose: an unlisted modifier fails safe to "not identified".
_NEUTRAL = frozenset("""
a an the one two three four five six seven eight nine ten some my your that this
small big large tiny little huge tall short medium mini giant long wide narrow thin thick old new
red green blue yellow white black brown grey gray orange pink purple silver gold golden beige tan
wooden wood metal metallic plastic glass leather ceramic clay stone fabric cloth
modern classic simple plain cheap nice pretty decorative ornamental antique vintage rustic
""".split())
_POLYSEMOUS = {
    "plant": frozenset("potted house indoor outdoor leafy flowering living live artificial fake tropical "
                       "desk office table floor garden flower pot fern ficus cactus succulent herb palm "
                       "monstera".split()),
    "houseplant": frozenset(),
    "pot": frozenset("flower plant garden planter terracotta potted".split()),
    "screen": frozenset("computer monitor lcd led flat flatscreen pc desktop display tv hd widescreen "
                        "curved gaming".split()),
    "display": frozenset("computer monitor lcd led flat pc desktop screen hd gaming".split()),
    "monitor": frozenset("computer pc lcd led flat desktop screen display gaming curved widescreen hd crt".split()),
    "cap": frozenset("baseball knit wool woolly winter hunting trucker peaked school ski swim".split()),
    "closet": frozenset("clothes clothing linen broom storage bedroom hall coat wall walk built in walkin builtin "
                        "wardrobe".split()),
    "bin": frozenset("trash garbage waste rubbish recycling litter kitchen office dust paper wastepaper "
                     "pedal".split()),
    "bag": frozenset("hand shoulder shopping tote travel school messenger duffel gym beach grocery carry "
                     "evening clutch".split()),
    "tap": frozenset("water kitchen bathroom sink basin".split()),
}


def _name_fits(nm: str, name: str, qwords: set) -> bool:
    """True if the category name / alias ``nm`` (whose words are all in the query) really names what
    ``name`` asks for: it is the phrase's head, and an ambiguous alias has only neutral company."""
    from . import aliases
    w = _words(nm)
    if not aliases.names_head(w, name):
        return False
    extra = _POLYSEMOUS.get(nm.strip().lower())
    if extra is None:
        return True
    rest = qwords - w - {x + "s" for x in w}
    rest = {r for r in rest if not (r.endswith("s") and r[:-1] in w)}
    return all(r in _NEUTRAL or r in extra or r in aliases.GENERIC_HEADS or r.isdigit() for r in rest)


def _resolve(name: str):
    """(entries, matched_phrase_words, category_names): the variant pool of the best-matching category
    for ``name``, the words of the longest category name / alias that matched, and the matched
    category's own (primary) name. ([], set(), None) when nothing fits."""
    from . import aliases
    qw = _words(name)
    scored = []
    for entry in _table():
        best = (0, set())
        for nm in entry[0]:
            w = _words(nm)
            if w and w <= qw and len(w) > best[0] and _name_fits(nm, name, qw):
                best = (len(w), w)
        if best[0] > 0:
            scored.append((best, entry))
    if not scored:
        return [], set(), None
    top = max(b[0] for b, _ in scored)
    pool = [entry for b, entry in scored if b[0] == top]
    matched = next(b[1] for b, e in scored if b[0] == top)
    return pool, matched, pool[0][6]


def _category_pool(name: str):
    """The variant pool: all exemplars of the best-matching category for name.
    Best = the longest whole-word category/alias match among the names that are the phrase's HEAD
    (see ``_name_fits``): "a desk LAMP" pools the lamps, not the tables that 'desk' aliases, and
    "a tv stand" pools nothing — 'tv' is a modifier there. Deterministic, offline."""
    return _resolve(name)[0]


def anchor_words(name: str) -> set:
    """The words ShapeNetSem keys must share one of for ``name``'s size and materials to describe the
    SAME object the exemplar layout was chosen for: the category's own name, plus those words of the
    matched phrase that are themselves single-word names of the category ("a potted PLANT" keys
    'plant' and 'vase'; "a WARDROBE" keys 'wardrobe' and 'cabinet'). The other word of a compound
    adds nothing: "frame" in "bed frame" and "oven" in "microwave oven" are exactly the words that
    would pick another object's size. Empty set when no exemplar category fits."""
    pool, matched, cat = _resolve(name)
    if not pool:
        return set()
    singles = {nm for nm in pool[0][0] if len(_words(nm)) == 1}
    return set(_words(cat)) | {w for w in matched if w in singles}


@functools.lru_cache(maxsize=1)
def _standard_dims() -> dict:
    try:
        d = json.loads((_DIR.parent / "object_dims.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    return d.get("categories") or {}


def standard_dims_of(category: str):
    """(dims_m, source, license) — the standard overall W x D x H of an exemplar category that
    ShapeNetSem has no key for (door, dishwasher), or None."""
    e = _standard_dims().get((category or "").strip().lower())
    return (list(e["dims_m"]), e["source"], e.get("license", "")) if e else None


def category_of(name: str):
    """The primary category name ('cabinet' for 'a wardrobe') the exemplar pool of ``name`` belongs
    to, or None."""
    return _resolve(name)[2]


def exemplar_of(name: str, *, exclude=None, hint=None, _aliased: bool = False):
    """(parts, source, license, anno_id) for the object's category, or None.

    ``exclude`` (anno_ids) skips already-used models so a "different" request
    gets a fresh one; ``hint`` prefers a tagged variant. Matched by whole-word
    category containment (longest category wins); WordNet aliases retried once.
    """
    exclude = {str(x) for x in (exclude or ())}
    pool = _category_pool(name)
    if not pool:
        if not _aliased:
            from . import aliases
            for alt in sorted(aliases.expand(name)):
                aw = _words(alt)
                for names, parts, src, lic, anno, tags, _cat in _table():
                    if any(_words(nm) == aw for nm in names):
                        return (parts, src, lic, anno)
        return None

    def rank(entry):
        _names, _parts, _src, _lic, anno, tags, _cat = entry
        return (0 if (hint and _tag_match(tags, hint)) else 1,   # hinted variant first
                1 if anno in exclude else 0)                     # then a fresh model

    _names, parts, src, lic, anno, _tags, _cat = min(pool, key=rank)
    return (parts, src, lic, anno)


__all__ = ["exemplar_of", "anchor_words", "category_of", "standard_dims_of"]
