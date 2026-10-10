"""Run the Lua parser (LuaJIT via lupa) over the independent utterance corpus and report where it disagrees.

    python stage/gmod/holodeck/eval_parse.py [--show 40]       (needs lupa: pip install lupa)
The corpus (data/utterance_corpus.json) was written and audited by independent agents that never saw the parser."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LUA = HERE.parent / "addon" / "lua" / "mentat" / "holodeck"


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
    rt, mod = runtime()
    load = rt.eval("function(p) return dofile(p) end")
    catalog = load((LUA / "catalog.lua").as_posix())
    Parse = load((LUA / "parse.lua").as_posix())
    parser = Parse.new(catalog)
    def parse(text):
        return to_py(mod, parser.parse(parser, text))
    return parse


def compare(exp: dict, got: dict) -> list[str]:
    bad = []
    if got["action"] != exp["action"]:
        bad.append(f"action {got['action']} != {exp['action']}")
    ge = sorted((i["kind"], i["count"]) for i in got.get("items", []) if isinstance(got.get("items"), list))
    ee = sorted((i["noun"], i["count"]) for i in exp["items"])
    if ge != ee:
        bad.append(f"items {ge} != {ee}")
    if exp["action"] in ("add", "move", "rearrange", "rotate"):
        if got["arrangement"] != exp["arrangement"] and not (got.get("facing") and exp["arrangement"] == "facing_me"):
            bad.append(f"arrangement {got['arrangement']} != {exp['arrangement']}")
        gt = got.get("target", "") or ""
        if gt != exp["target"] and not (got.get("facing") == exp["target"]):
            bad.append(f"target {gt!r} != {exp['target']!r}")
    if (got.get("scene") or "") != exp["scene"]:
        bad.append(f"scene {got.get('scene')!r} != {exp['scene']!r}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--slice", default="")
    ap.add_argument("--max-fail", type=int, default=0, help="exit non-zero when more cases than this fail")
    a = ap.parse_args(argv)
    cases = json.loads((HERE / "data" / "utterance_corpus.json").read_text(encoding="utf-8"))["cases"]
    parse = make_parser()
    fails, by_slice, known = [], Counter(), []
    for c in cases:
        if a.slice and c["slice"] != a.slice:
            continue
        by_slice[c["slice"]] += 1
        try:
            got = parse(c["utterance"])
            bad = compare(c, got)
        except Exception as e:  # a crash is a failure with its reason
            got, bad = {}, [f"CRASH {e}"]
        if bad and c.get("known_miss"):
            known.append((c, got, bad))
            by_slice[c["slice"]] -= 1
        elif bad:
            fails.append((c, got, bad))
    total = sum(by_slice.values())
    print(f"{total - len(fails)}/{total} utterances parse to the audited meaning "
          f"({len(known)} more are known misses: words the library cannot know)")
    for slice_name, n in by_slice.items():
        f = sum(1 for c, _, _ in fails if c["slice"] == slice_name)
        print(f"   {slice_name:26s} {n - f}/{n}")
    for c, got, bad in fails[: a.show]:
        print(f"\n[{c['slice']}] {c['utterance']!r}")
        for b in bad:
            print("     ", b)
    return 1 if len(fails) > a.max_fail else 0


if __name__ == "__main__":
    sys.exit(main())
