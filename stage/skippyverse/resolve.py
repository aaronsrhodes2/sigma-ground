"""The lookup ladder (misc/SKIPPYVERSE_OBJECTS_DESIGN.md section 5) as far as it can run offline.

  rung 0   known to the page   - a curated starter kind matched by name or alias
  rung 1   cached / indexed    - any other card in the index matched by name or alias
  rung 1b  mid-band            - a synonym, an `approx` name, a one-or-two-letter typo, or a margin-checked partial text match:
                                 Skippy says "the closest I have is X" or "I read that as X"; NEVER a silent substitution
  rung 2   fetch               - a found card with no size, or no local asset yet: `plan[0]` is {action: 'fetch', ...}. Executed ONLY when
                                 the caller passes `acquirer` (acquire.acquire_and_measure or a stand-in); otherwise nothing is fetched
  rung 3/4  STUBS              - generate / build from parts: `plan` says what WOULD happen; nothing is executed
  rung 5   honest refusal      - unknown, too big for the room, size not trustworthy, size unknown

There is no cosine cutoff anywhere: acceptance is by exact/alias evidence, then by a coverage + score MARGIN between the best and
the second-best kind, and ambiguity is reported as ambiguity (a short list of kinds), not resolved by guessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skippyverse import cards as C  # noqa: E402
from skippyverse.index import Index, size_sanity  # noqa: E402
from skippyverse.taxonomy import guess_category  # noqa: E402
from skippyverse.text import damerau, norm, parse_query  # noqa: E402

CLASS_RANK = {"curated": 0, "found": 1, "generated": 2, "deckard": 3, "placeholder": 4}
OUTDOOR = {"w": 400.0, "d": 400.0, "h": 1000.0}           # the apron world of design section 7
STOP = {"of", "the", "a", "an", "to", "for", "and", "or", "my", "your", "thing", "stuff", "something", "anything", "item", "object"}
PLAN_NEEDS_APPROVAL = "Aaron's approval (downloads, API keys, accounts)"
DECKARD_TEMPLATES = {"vehicle/road", "vehicle/aircraft"}   # design section 9: vehicles first
MARGIN = 0.25                                             # relative score margin between best and second-best kind
MAX_SHORTLIST = 7


def a_an(name: str) -> str:
    vowel = name[:1] in "aeiou" and not name.startswith(("uni", "use", "eu", "one", "ubiq"))
    return ("an " if vowel else "a ") + name


def _have(phrase: str) -> str:
    """'a dragon' for a plain noun phrase; a quoted phrase when it is not one (so we never say 'a something to sit')."""
    ws = phrase.split()
    return a_an(phrase) if len(ws) <= 3 and not any(w in STOP for w in ws) else f"anything called '{phrase}'"


def _fmt_m(m: float) -> str:
    return f"{m:.1f} m" if m < 10 else f"{m:.0f} m"


# --------------------------------------------------------------------------------------------------------------------- planning
def plan_rungs(phrase: str) -> list[dict]:
    """What rungs 2, 3 and 4 WOULD do for this phrase. Pure description; nothing is fetched, generated or built."""
    plan = [
        {"rung": 2, "name": "found", "executed": False,
         "would": f"search Poly Haven, Kenney, Gazebo Fuel (CC0 and CC-BY-4.0 only) and NASA 3D for '{phrase}', fetch the best match "
                  "on the PC, normalise scale and orientation to a sourced size, optimise, hash, publish, then resolve again",
         "needs": ["PC on", "--allow-network", PLAN_NEEDS_APPROVAL]},
        {"rung": 3, "name": "generated", "executed": False,
         "would": f"draw a concept picture of '{phrase}', run TRELLIS.2 on the PC, run the quality gate, take the size from a table",
         "needs": ["generation worker (design increment 6)", "download of trellis.cpp and weights (Aaron's approval)"]},
    ]
    if guess_category(phrase) in DECKARD_TEMPLATES:
        plan.append({"rung": 4, "name": "deckard", "executed": False,
                     "would": f"build a blocky '{phrase}' from parts with a sourced size and weight",
                     "needs": ["Deckard v2 (design increment 7)"]})
    return plan


FETCH_SOURCES = {"NASA 3D Resources": "NASA", "Gazebo Fuel": "Gazebo Fuel", "Poly Haven": "Poly Haven"}      # mirrors acquire.SOURCE_LABEL


def fetchable(card: dict) -> bool:
    """A found, servable card from a source we can download from, with no local copy yet."""
    return (card["visual"]["class"] == "found" and bool(card["redistributable"])
            and (card.get("provenance") or {}).get("source") in FETCH_SOURCES and not card["visual"].get("local_asset"))


def fetch_plan(card: dict, phrase: str) -> list[dict]:
    """plan_rungs with rung 2 made concrete for this card: {action: 'fetch', card_id, source}. Still nothing is executed."""
    plan = plan_rungs(phrase)
    if fetchable(card):
        label = FETCH_SOURCES[card["provenance"]["source"]]
        plan[0] = {**plan[0], "action": "fetch", "card_id": card["id"], "source": label,
                   "would": f"download this one model from {label} ({card['provenance'].get('url')}), measure it, then resolve again"}
    return plan[:1]


def _acquire(acquirer, card: dict):
    """-> (result or None, error text or None). Never raises: a failed fetch is reported, not hidden."""
    try:
        return acquirer(card), None
    except Exception as e:                                  # AcquireRefused / AcquireError / anything the stand-in raises
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------------------------------------------------- choosing
def _tier(card: dict, audience: str) -> tuple:
    servable = card["redistributable"] or audience != "web"
    return (0 if servable else 1, CLASS_RANK.get(card["visual"]["class"], 9))


def _volume(c: dict) -> float:
    s = c["size_m"]
    return s["w"] * s["d"] * s["h"]


def shortlist(cards_: list[dict], n: int = MAX_SHORTLIST) -> list[dict]:
    """Up to n variants of one kind spanning its size range (smallest .. largest), evenly by rank."""
    cs = sorted((c for c in cards_ if c.get("size_m")), key=_volume)
    if len(cs) <= n:
        return cs
    idx = sorted({round(i * (len(cs) - 1) / (n - 1)) for i in range(n)})
    return [cs[i] for i in idx]


def _fits(card: dict, room: dict | None) -> bool:
    if not room:
        return True
    s = card["size_m"]
    a, b = sorted((s["w"], s["d"]), reverse=True)
    ra, rb = sorted((room["w"], room["d"]), reverse=True)
    return a <= ra and b <= rb and (room.get("h") is None or s["h"] <= room["h"])


def _group(cards_: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for c in cards_:
        g.setdefault(norm(c["kind"]), []).append(c)
    return g


def _prefer_curated(kinds: dict[str, list[dict]]) -> dict[str, list[dict]]:
    cur = {k: v for k, v in kinds.items() if any(c["visual"]["class"] == "curated" for c in v)}
    return cur or kinds


def _kind_rank(cards_: list[dict], audience: str) -> tuple:
    return min(_tier(c, audience) for c in cards_)


# ---------------------------------------------------------------------------------------------------------------------- matching
def _match(index: Index, phrase: str):
    """-> (how, {kind: [cards]}, detail). how in exact|alias|approx|synonym|category|typo|text|none."""
    hits = index.by_term(phrase)
    if not hits and " " in phrase:                         # 'wine glass' vs the glued 'wineglass': a space is not a typo
        hits = index.by_term(phrase.replace(" ", ""))
    named = [c for c, role in hits if role in ("name", "alias")]
    if named:
        kinds = _prefer_curated(_group(named))
        how = "exact" if any(phrase == norm(c["kind"]) or phrase == norm(c["name"]) for cs in kinds.values() for c in cs) else "alias"
        return how, kinds, {}
    approx = [c for c, role in hits if role == "approx"]
    if approx:
        return "approx", _prefer_curated(_group(approx)), {}
    # WordNet-style synonym groups from shape_aliases.json
    syn_cards = []
    via = None
    for s in sorted(index.synonyms(phrase)):
        found = [c for c, role in index.by_term(s) if role in ("name", "alias")]
        if found:
            syn_cards += found
            via = via or s
    if syn_cards:
        return "synonym", _prefer_curated(_group(syn_cards)), {"via": via}
    # a bare category word is a question, not an object
    cats = {}
    for cat in index.categories():
        for part in {cat, cat.replace("/", " "), cat.rsplit("/", 1)[-1]}:
            cats.setdefault(norm(part), cat)
    if phrase in cats and cats[phrase] != "misc":
        return "category", _group(index.category_cards(cats[phrase])), {"category": cats[phrase]}
    # one or two typos in the whole phrase
    if len(phrase) >= 4:
        thr = 1 if len(phrase) <= 7 else 2
        vocab = index.vocabulary()
        best, bd = [], thr + 1
        for term, ids in vocab.items():
            if abs(len(term) - len(phrase)) > thr or term.count(" ") != phrase.count(" "):
                continue
            d = damerau(phrase, term)
            if 0 < d <= thr:
                if d < bd:
                    best, bd = [(term, ids)], d
                elif d == bd:
                    best.append((term, ids))
        if best:
            cs = [index.card(i) for _, ids in best for i in ids]
            return "typo", _prefer_curated(_group(cs)), {"read_as": best[0][0], "distance": bd}
    # partial text match, accepted only with the head noun present and a clear margin
    toks = [t for t in phrase.split() if t not in STOP]
    if toks:
        head, want = toks[-1], set(toks)
        scored: dict[str, tuple[float, float, list]] = {}
        for card, score in index.search(" ".join(toks), 40, any_token=True):
            nt = index.name_tokens(card)
            if head not in nt:
                continue
            cov = len(want & nt) / len(want)
            k = norm(card["kind"])
            cur = scored.get(k)
            if cur is None or (cov, score) > (cur[0], cur[1]):
                scored[k] = (cov, score, cur[2] if cur else [])
            scored[k][2].append(card)
        if scored:
            order = sorted(scored.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
            kinds = {k: v[2] for k, v in order}
            (k1, (c1, s1, _)), rest = order[0], order[1:]
            if c1 >= 0.5 and (not rest or c1 > rest[0][1][0] or (s1 - rest[0][1][1]) / max(s1, 1e-9) >= MARGIN):
                return "text", {k1: kinds[k1]}, {"coverage": c1, "alternatives": [k for k, _ in rest[:2]]}
            if c1 >= 0.5:
                return "ambiguous", dict(list(kinds.items())[:3]), {}
    return "none", {}, {}


def _neighbours(index: Index, phrase: str, exclude: set[str]) -> list[str]:
    """Other kinds the text search finds for the phrase (for a top-3 list only; never used to accept a match)."""
    toks = [t for t in phrase.split() if t not in STOP]
    seen, out = set(exclude), []
    for card, _ in index.search(" ".join(toks), 12, any_token=True) if toks else []:
        k = norm(card["kind"])
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# --------------------------------------------------------------------------------------------------------------------- the ladder
def resolve(query: str, index: Index, *, room: dict | None = None, audience: str = "local", online: bool = False, acquirer=None) -> dict:
    """Resolve a natural request to a card, or to an honest refusal.

    room      None (no check), {"w","d","h"} in metres, or {"outdoor": True} for the open apron.
    audience  "web" = the card must be servable to friends' browsers (a non-redistributable card is still reported, with its
              size, but as a box: `servable` is False and the card is the web-safe copy); "local" = Garry's Mod side.
    online    True only changes the wording and the plan for unknown things ('would fetch'); nothing is ever fetched by it.
    acquirer  None (default, offline): nothing is fetched. A callable card -> {"card": updated_card, "pose": ..., ...} (normally
              acquire.acquire_and_measure) makes rung 2 real for a found card with no size or no local copy: the result carries
              `fetched`, `pose`, `announce` and the sentence includes the honesty clause with the measured size and its basis.
    """
    pq = parse_query(query)
    phrase = pq["phrase"]
    res = {"query": query, "phrase": phrase, "raw_phrase": pq["raw_phrase"], "count": pq["count"], "modifiers": pq["modifiers"],
           "status": "not_found", "rung": "5", "card": None, "kind": None, "shortlist": [], "options": [], "candidates": [],
           "servable": False, "refusal": None, "plan": None, "match": None, "sentence": "", "fetched": False, "pose": None, "announce": None}
    if room and room.get("outdoor"):
        room = OUTDOOR
    if not phrase:
        res.update(status="not_found", refusal="empty", sentence="I didn't catch what you want. Name a thing, like a couch or a lamp.")
        return res

    how, kinds, detail = _match(index, phrase)
    if pq["with_number"] and pq["with_number"] != phrase:               # "two seater" is a name, not a count of "seater"
        h2, k2, d2 = _match(index, pq["with_number"])
        if h2 in ("exact", "alias"):
            how, kinds, detail, phrase = h2, k2, d2, pq["with_number"]
            res.update(phrase=phrase, count=None)
    res["match"] = {"how": how, **detail}
    shown = pq["raw_phrase"] or phrase

    if how == "none":
        plan = plan_rungs(phrase)
        res.update(status="would_fetch" if online else "not_found", refusal="unknown", plan=plan)
        if online:
            res["rung"] = "2"
            res["sentence"] = (f"I don't have {_have(phrase)} on the shelf yet. I would look for a model in the free sources "
                               "(not done in this offline build).")
        else:
            res["sentence"] = f"I don't have {_have(phrase)}, and I can't make new things right now."
        return res

    if how == "ambiguous":
        opts = list(kinds)
        res.update(status="ambiguous", options=opts, candidates=opts,
                   sentence=f"I'm not sure which you mean by {shown}: " + _join(opts) + ". Which one?")
        return res

    if how == "category":
        order = sorted(kinds, key=lambda k: (_kind_rank(kinds[k], audience), k))
        res.update(status="category", options=order[:12], candidates=order[:3], match={"how": how, **detail})
        res["sentence"] = f"{shown.capitalize()} is a whole category, so I won't pick one for you. I have: " + _join(order[:8]) + "."
        return res

    # one chosen kind (several kinds only when two curated kinds share a name: report that as ambiguity)
    if len(kinds) > 1:
        opts = sorted(kinds, key=lambda k: (_kind_rank(kinds[k], audience), k))
        res.update(status="ambiguous", options=opts, candidates=opts[:3],
                   sentence=f"{shown.capitalize()} could be " + _join(opts[:4]) + ". Which one?")
        return res
    kind, kcards = next(iter(kinds.items()))
    res["kind"] = kind
    res["candidates"] = [kind, *_neighbours(index, phrase, {kind})][:3]
    if how == "text" and detail.get("alternatives"):
        res["options"] = [kind, *detail["alternatives"]]

    # pick the tier the audience can use, then the variants whose size can be trusted
    best = min(_tier(c, audience) for c in kcards)
    pool = [c for c in kcards if _tier(c, audience) == best]
    sized = [c for c in pool if c.get("size_m")]
    trusted = [c for c in sized if index.size_ok(c["id"])]
    rung = "0" if any(c["visual"]["class"] == "curated" for c in pool) else "1"
    if how in ("approx", "synonym", "typo", "text"):
        rung = "1b"
    res["rung"] = rung
    lead = _lead(how, shown, kind, detail)

    announce = None
    if not sized:
        target = next((c for c in pool if fetchable(c)), None)
        if acquirer is not None and target is not None:
            label = FETCH_SOURCES[target["provenance"]["source"]]
            announce = f"I don't have {a_an(kind)} on the shelf yet. Fetching a model from {label}..."
            got, err = _acquire(acquirer, target)
            if got is None:
                res.update(status="fetch_failed", refusal="fetch_failed", card=_out(target, audience), plan=fetch_plan(target, phrase),
                           announce=announce, sentence=f"{announce} It didn't work ({err}), so I still don't know its size and won't guess.")
                return res
            new_card = got["card"]
            res.update(fetched=True, pose=got.get("pose"), announce=announce, rung="2")
            pool = [new_card]
            sized = [new_card]
            trusted = [new_card] if new_card["id"] not in size_sanity([new_card]) else []
            lead = "Got it. " + _lead(how, shown, kind, detail)
        else:
            res.update(status="needs_size", refusal="no_size", card=_out(pool[0], audience),
                       plan=fetch_plan(target, phrase) if target else plan_rungs(phrase)[:1],
                       sentence=f"{lead} I know the name but not its size, so I won't guess where to put it.")
            return res
    if not trusted:
        c = sized[0]
        res.update(status="untrusted_size", refusal="size_suspect", card=_out(c, audience), plan=plan_rungs(phrase)[:1],
                   sentence=(f"{announce} " if announce else "") + f"{lead} But my size for it ({_fmt_m(max(c['size_m']['w'], c['size_m']['d'], c['size_m']['h']))} on its "
                            "longest side) looks wrong, so I won't place it by that.")
        return res

    fits = [c for c in trusted if _fits(c, room)]
    if not fits:
        small = min(trusted, key=_volume)
        big = max(small["size_m"]["w"], small["size_m"]["d"], small["size_m"]["h"])
        where = "the open ground I have" if room is OUTDOOR else "this room"
        res.update(status="too_big", refusal="too_big", card=_out(small, audience), shortlist=[_out(c, audience) for c in shortlist(trusted)],
                   sentence=(f"{announce} " if announce else "") + f"{lead} That's bigger than {where} ({_fmt_m(big)})." + ("" if room is OUTDOOR else " Try it outdoors."))
        return res

    cs = sorted(fits, key=_volume)
    pref = pq["size_pref"]
    rep = cs[-1] if pref == "up" else cs[0] if pref == "down" else cs[len(cs) // 2]
    if acquirer is not None and not res["fetched"] and fetchable(rep):
        label = FETCH_SOURCES[rep["provenance"]["source"]]
        announce = f"I don't have {a_an(kind)} on the shelf yet. Fetching a model from {label}..."
        got, err = _acquire(acquirer, rep)
        if got is None:
            res["plan"] = fetch_plan(rep, phrase)
            announce += f" It didn't work ({err}), so this stays a box at the size I already had."
        else:
            rep = got["card"]
            res.update(fetched=True, pose=got.get("pose"), rung="2")
            lead = "Got it. " + lead
            if rep["id"] in size_sanity([rep]) or not _fits(rep, room):
                big = max(rep["size_m"][k] for k in ("w", "d", "h"))
                bad = rep["id"] in size_sanity([rep])
                res.update(status="untrusted_size" if bad else "too_big", refusal="size_suspect" if bad else "too_big",
                           card=_out(rep, audience), announce=announce,
                           sentence=f"{announce} {lead} " + (f"But its measured size ({_fmt_m(big)} on its longest side) looks wrong, so I won't place it by that."
                                                            if bad else f"That's bigger than this room ({_fmt_m(big)}). Try it outdoors."))
                return res
    res["announce"] = announce
    servable = bool(rep["redistributable"])
    sentence = (f"{announce} " if announce else "") + f"{lead} {rep['say']}"
    if audience == "web" and not servable:
        sentence += " I can't share a model of it, so it would be a box."
    res.update(status={"exact": "exact", "alias": "alias", "approx": "closest", "synonym": "synonym", "typo": "closest",
                       "text": "closest"}[how],
               card=_out(rep, audience), shortlist=[_out(c, audience) for c in shortlist(fits)], servable=servable, sentence=sentence)
    return res


def _out(card: dict, audience: str) -> dict:
    return C.strip_for_web(card) if audience == "web" else card


def _join(names: list[str]) -> str:
    names = [a_an(n) for n in names]
    return ", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 else (names[0] if names else "")


def _lead(how: str, shown: str, kind: str, detail: dict) -> str:
    """The first sentence, generated from the way the match was made."""
    if how == "exact":
        return f"I have {a_an(kind)}."
    if how == "alias":
        return f"I have {a_an(shown)}." if norm(shown) == norm(kind) else f"I have {a_an(shown)} (filed as {a_an(kind)})."
    if how == "approx":
        return f"I don't have {a_an(shown)} by that name. The closest I have is {a_an(kind)}. Using that."
    if how == "synonym":
        return f"I took '{shown}' to mean {kind} (a listed synonym)."
    if how == "typo":
        return f"I read '{shown}' as '{detail.get('read_as', kind)}'. If that's wrong, say so."
    if how == "text":
        return f"I don't have {a_an(shown)} by that name. The closest I have is {a_an(kind)}. Using that."
    return ""
