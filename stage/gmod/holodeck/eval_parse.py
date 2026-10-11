"""Run the Lua parser (LuaJIT via lupa) over the independent utterance corpora and report where it disagrees.

    python stage/gmod/holodeck/eval_parse.py [--show 40]                    (needs lupa: pip install lupa)
    python stage/gmod/holodeck/eval_parse.py --corpus utterance_corpus_heldout.json
The audited corpus (data/utterance_corpus.json) was written and audited by independent agents that never saw the parser.
The held-out corpus (data/utterance_corpus_heldout.json) is built from the hand-judged requests of an independent review: the
stressed, messy and compound sentences the first corpus does not cover. Its cases may also carry
    "sequence": [ {...}, {...} ]       the requests parse_all() must return, in order (compound sentences)
and, on any expected request, these optional checks: "exclude", "facing", "facing_away", "shape", "seats", "every", "rows", "cols",
"unknown_verb", "truncated", "notes_contain" (a substring of some note), "unknown" (the unknown words). An item may give "count": null
(any count) or "count_min"; "arrangement" and "target" may be left out (not checked)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LUA = HERE.parent / "addon" / "lua" / "mentat" / "holodeck"
DATA = HERE / "data"


def runtime():
    try:
        from lupa import luajit21 as mod
    except ImportError:
        from lupa import lua51 as mod
    rt = mod.LuaRuntime(unpack_returned_tuples=True)
    rt.execute("jit.off()")
    return rt, mod


def to_py(mod, t):
    if mod.lua_type(t) != "table":
        return t
    keys = list(t.keys())
    if keys and all(isinstance(k, (int, float)) and float(k).is_integer() for k in keys) and sorted(int(k) for k in keys) == list(range(1, len(keys) + 1)):
        return [to_py(mod, t[i]) for i in range(1, len(keys) + 1)]
    return {k: to_py(mod, t[k]) for k in keys}


def make_parser():
    """A function text -> the FIRST request of the words (a dict). `.parse_all(text)` -> every request (a list of dicts)."""
    rt, mod = runtime()
    load = rt.eval("function(p) return dofile(p) end")
    catalog = load((LUA / "catalog.lua").as_posix())
    Parse = load((LUA / "parse.lua").as_posix())
    parser = Parse.new(catalog)

    def lua_value(obj):
        """A Python dict / list (a request the parser made earlier) as a Lua table."""
        if isinstance(obj, dict):
            t = rt.table()
            for k, v in obj.items():
                t[k] = lua_value(v)
            return t
        if isinstance(obj, (list, tuple)):
            t = rt.table()
            for i, v in enumerate(obj, 1):
                t[i] = lua_value(v)
            return t
        return obj

    def parse(text, prev=None):
        return to_py(mod, parser.parse(parser, text, lua_value(prev) if prev else None))

    def parse_all(text, prev=None):
        out = to_py(mod, parser.parse_all(parser, text, lua_value(prev) if prev else None))
        return out if isinstance(out, list) else [out]

    parse.parse_all = parse_all
    parse.parser = parser
    parse.Parse = Parse
    parse.mod = mod
    return parse


# ---------------------------------------------------------------------------------------------- comparing
def _items_of(got: dict) -> list[dict]:
    items = got.get("items")
    return items if isinstance(items, list) else []


def _match_items(exp_items: list[dict], got_items: list[dict]) -> str | None:
    """None when the items agree (kinds as a multiset; a count of null / count_min is open), else a description."""
    pseudo = [i for i in got_items if str(i["kind"]).startswith("@")]
    exp_has_pseudo = any(str(i["noun"]).startswith("@") for i in exp_items)
    if not exp_items and pseudo and all(i["kind"] in ("@all", "@last") for i in pseudo):
        # "rearrange everything" in the first corpus predates the "@all" / "@last" items: no named items is the same thing
        got_items = [i for i in got_items if not str(i["kind"]).startswith("@")]
    remaining = list(got_items)
    for e in exp_items:
        hit = None
        for g in remaining:
            if g["kind"] != e["noun"]:
                continue
            ec = e.get("count", None)
            if ec is None and "count" in e and e.get("count_min") is None:
                hit = g
                break
            if e.get("count_min") is not None:
                if g["count"] >= e["count_min"]:
                    hit = g
                    break
                continue
            if g["count"] == ec:
                hit = g
                break
        if hit is None:
            have = sorted((g["kind"], g["count"]) for g in got_items)
            want = sorted((i["noun"], i.get("count")) for i in exp_items)
            return f"items {have} != {want}"
        remaining.remove(hit)
    if remaining:
        have = sorted((g["kind"], g["count"]) for g in got_items)
        want = sorted((i["noun"], i.get("count")) for i in exp_items)
        return f"items {have} != {want}"
    return None


def compare(exp: dict, got: dict) -> list[str]:
    bad = []
    if got["action"] != exp["action"]:
        bad.append(f"action {got['action']} != {exp['action']}")
    if "items" in exp:
        problem = _match_items(exp["items"], _items_of(got))
        if problem:
            bad.append(problem)
    if exp["action"] in ("add", "move", "rearrange", "rotate"):
        if "arrangement" in exp and got["arrangement"] != exp["arrangement"] and not (got.get("facing") and exp["arrangement"] == "facing_me"):
            bad.append(f"arrangement {got['arrangement']} != {exp['arrangement']}")
        gt = got.get("target", "") or ""
        if "target" in exp and gt != exp["target"] and not (got.get("facing") == exp["target"]):
            bad.append(f"target {gt!r} != {exp['target']!r}")
    if "scene" in exp and (got.get("scene") or "") != exp["scene"]:
        bad.append(f"scene {got.get('scene')!r} != {exp['scene']!r}")
    for key in ("facing", "facing_away", "shape", "seats", "every", "rows", "cols", "unknown_verb", "truncated"):
        if key in exp and got.get(key) != exp[key]:
            bad.append(f"{key} {got.get(key)!r} != {exp[key]!r}")
    if "exclude" in exp:
        ge = sorted(got.get("exclude") or []) if isinstance(got.get("exclude"), list) else []
        if ge != sorted(exp["exclude"]):
            bad.append(f"exclude {ge} != {sorted(exp['exclude'])}")
    if "unknown" in exp:
        gu = sorted(got.get("unknown") or []) if isinstance(got.get("unknown"), list) else []
        if gu != sorted(exp["unknown"]):
            bad.append(f"unknown {gu} != {sorted(exp['unknown'])}")
    if "notes_contain" in exp:
        notes = got.get("notes") if isinstance(got.get("notes"), list) else []
        if not any(exp["notes_contain"] in n for n in notes):
            bad.append(f"no note containing {exp['notes_contain']!r} in {notes}")
    return bad


def compare_case(case: dict, parse) -> list[str]:
    """The problems of one corpus case: a single request (the first one) or a `sequence` (parse_all)."""
    if "sequence" in case:
        got = parse.parse_all(case["utterance"])
        want = case["sequence"]
        if len(got) != len(want):
            return [f"{len(got)} requests != {len(want)}: " + " | ".join(f"{g['action']}:{[i['kind'] for i in _items_of(g)]}" for g in got)]
        bad = []
        for k, (e, g) in enumerate(zip(want, got), 1):
            bad += [f"[{k}] {b}" for b in compare(e, g)]
        return bad
    return compare(case, parse(case["utterance"]))


def load_cases(name: str = "utterance_corpus.json") -> list[dict]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))["cases"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--slice", default="")
    ap.add_argument("--corpus", default="utterance_corpus.json", help="a file in data/ (utterance_corpus_heldout.json)")
    ap.add_argument("--max-fail", type=int, default=0, help="exit non-zero when more cases than this fail")
    a = ap.parse_args(argv)
    cases = load_cases(a.corpus)
    parse = make_parser()
    fails, by_slice, known = [], Counter(), []
    for c in cases:
        if a.slice and c["slice"] != a.slice:
            continue
        by_slice[c["slice"]] += 1
        try:
            bad = compare_case(c, parse)
        except Exception as e:  # a crash is a failure with its reason
            bad = [f"CRASH {e}"]
        if bad and c.get("known_miss"):
            known.append((c, bad))
            by_slice[c["slice"]] -= 1
        elif bad:
            fails.append((c, bad))
    total = sum(by_slice.values())
    print(f"{total - len(fails)}/{total} utterances parse to the audited meaning "
          f"({len(known)} more are known misses: words the library cannot know)")
    for slice_name, n in by_slice.items():
        f = sum(1 for c, _ in fails if c["slice"] == slice_name)
        print(f"   {slice_name:26s} {n - f}/{n}")
    for c, bad in fails[: a.show]:
        print(f"\n[{c['slice']}] {c['utterance']!r}")
        for b in bad:
            print("     ", b)
    return 1 if len(fails) > a.max_fail else 0


if __name__ == "__main__":
    sys.exit(main())
