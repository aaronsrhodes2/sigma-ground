"""Scores Skippy's solid turns against the labelled corpus (data/turn_corpus.json).

    python stage/gmod/holodeck/turn_eval.py                 offline: the deterministic layers (addressee scoring + route) and the fallback replies. No network, no key.
    python stage/gmod/holodeck/turn_eval.py --show 20       also list the first 20 disagreements
    python stage/gmod/holodeck/turn_eval.py --model         COUNT the live model calls a sample would make and stop
    python stage/gmod/holodeck/turn_eval.py --model --go    run the sample against the live model route (frontier if a key is reachable, else the local model)
                                                            [--sample 12]  at most this many talk turns, evenly spread; refuses a worst case above --max-calls (default 40)

Needs lupa (the Lua 5.1 runtime the game's files run under): the project's GMod test environment has it (pip install lupa).
The rules scored are the REAL ones: mentat/holodeck/turnlogic.lua with the real parser, wake matcher, and his own command matchers.

Reported: addressee precision and recall (a "yes" prediction vs the label), the FALSE-WAKE rate (he answers a line that was not for him, or was only
maybe for him: the worst failure), the MISSED-WAKE rate (a line for him that gets no answer), route accuracy (overall and for the addressed ones),
how "maybe" lines were handled, and per-category figures. The corpus is synthetic (build_turn_corpus.py); read the numbers as a regression gauge for
these scenarios, not as the error rate of a real afternoon.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LUA = HERE.parent / "addon" / "lua"
CORPUS = HERE / "data" / "turn_corpus.json"
sys.path.insert(0, str(HERE))


def runtime():
    try:
        from lupa import luajit21 as mod
    except ImportError:
        from lupa import lua51 as mod
    rt = mod.LuaRuntime(unpack_returned_tuples=True)
    rt.execute("jit.off()")
    return rt, mod


def lua_value(rt, obj):
    if isinstance(obj, dict):
        t = rt.table()
        for k, v in obj.items():
            if v is not None:
                t[k] = lua_value(rt, v)
        return t
    if isinstance(obj, (list, tuple)):
        t = rt.table()
        for i, v in enumerate(obj, 1):
            t[i] = lua_value(rt, v)
        return t
    return obj


class Turns:
    """The real Lua rules, loaded once. decide(turn) -> dict."""

    def __init__(self):
        self.rt, self.mod = runtime()
        root = LUA.as_posix()
        self.rt.execute(f"""
          function include(p) return dofile('{root}/' .. p) end
          Wake = include('mentat/holodeck/wakewords.lua')
          Catalog = include('mentat/holodeck/catalog.lua')
          Parse = include('mentat/holodeck/parse.lua')
          Parser = Parse.new(Catalog)
          Logic = include('mentat/holodeck/turnlogic.lua')
          AvLogic = include('mentat/avatar/logic.lua')
          DirLogic = include('mentat/avatar/director_logic.lua')
          WaterIntent = include('mentat/water/intent.lua')
          WaterC = include('mentat/water/constants.lua')
          WorldIntent = include('mentat/world/intent.lua')
          Prev = Parser:parse('add a lamp')
          function Decide(text, ctx)
            local names = {{}}                       -- the stored form of each name, as the game keeps it (wakenames.lua normalises on the way in)
            for i, n in ipairs(ctx.names) do names[i] = Wake.normalise_name(n) or n end
            local deps = {{
              Wake = Wake, parser = Parser, prev = Prev, intents = {{ function(t) return WaterIntent.parse(t, WaterC) end, function(t) return WorldIntent.parse(t) end }},
              wake_intent = Wake.intent, av_command = AvLogic.match_command, dir_command = DirLogic.match_command,
              strip = function(t) return Wake.strip(t, names, {{ default_strip = function(x) return Parse.strip_wake(x) end,
                                                              is_noun = function(w) return Parser.noun[w] ~= nil end }}) end,
            }}
            return Logic.decide(text, ctx, deps)
          end
        """)
        self._decide = self.rt.eval("Decide")

    def decide(self, turn: dict) -> dict:
        c = dict(turn["ctx"])
        c["src"] = turn.get("src", "chat")
        out = self._decide(turn["text"], lua_value(self.rt, c))
        return {"addressed": out["addressed"], "score": out["score"], "route": out["route"], "final": out["final"],
                "reasons": [out["reasons"][i] for i in range(1, len(out["reasons"]) + 1)], "promoted": bool(out["promoted"]), "rest": out["rest"]}


def load_corpus(path: Path = CORPUS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["turns"]


def score(turns: list[dict], decide) -> dict:
    """Run every turn through `decide` and compute the figures. Returns {rows, metrics}."""
    rows = []
    for t in turns:
        got = decide(t)
        rows.append({"turn": t, "got": got})
    exp_yes = [r for r in rows if r["turn"]["addressee"] == "yes" and r["turn"]["route"] != "ignore"]
    exp_notfor = [r for r in rows if r["turn"]["addressee"] in ("no", "maybe")]
    answered = lambda r: r["got"]["final"] != "ignore"          # noqa: E731
    tp = sum(1 for r in rows if r["turn"]["addressee"] == "yes" and r["got"]["addressed"] == "yes")
    fp = sum(1 for r in rows if r["turn"]["addressee"] != "yes" and r["got"]["addressed"] == "yes")
    fn = sum(1 for r in rows if r["turn"]["addressee"] == "yes" and r["got"]["addressed"] != "yes")
    false_wake = [r for r in exp_notfor if answered(r)]
    missed = [r for r in exp_yes if not answered(r)]
    route_ok = sum(1 for r in rows if r["got"]["final"] == r["turn"]["route"])
    addr_rows = [r for r in rows if r["turn"]["route"] != "ignore"]
    route_only = sum(1 for r in addr_rows if r["got"]["route"] == r["turn"]["route"] and answered(r))
    maybes = [r for r in rows if r["turn"]["addressee"] == "maybe"]
    by_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "ok": 0, "false_wake": 0, "missed": 0})
    for r in rows:
        c = by_cat[r["turn"]["cat"]]
        c["n"] += 1
        c["ok"] += r["got"]["final"] == r["turn"]["route"]
        c["false_wake"] += r in false_wake
        c["missed"] += r in missed
    m = {
        "turns": len(rows),
        "addressee_precision": tp / (tp + fp) if tp + fp else 1.0,
        "addressee_recall": tp / (tp + fn) if tp + fn else 1.0,
        "false_wake_rate": len(false_wake) / len(exp_notfor) if exp_notfor else 0.0,
        "false_wake_n": len(false_wake), "not_for_him_n": len(exp_notfor),
        "missed_wake_rate": len(missed) / len(exp_yes) if exp_yes else 0.0,
        "missed_wake_n": len(missed), "for_him_n": len(exp_yes),
        "route_accuracy": route_ok / len(rows) if rows else 1.0,
        "route_accuracy_addressed": route_only / len(addr_rows) if addr_rows else 1.0,
        "maybe_n": len(maybes), "maybe_kept_maybe": sum(1 for r in maybes if r["got"]["addressed"] == "maybe"),
        "maybe_answered": sum(1 for r in maybes if answered(r)),
        "by_category": dict(by_cat),
    }
    return {"rows": rows, "metrics": m, "false_wake": false_wake, "missed": missed}


def reply_check(turns: list[dict]) -> dict:
    """The offline floor of the talk route: for every talk turn, the fallback line must satisfy the rubric (non-empty, <= 25 words, no markdown or link)
    and, for questions with keywords, say something helpful."""
    import skippy_turns as ST
    import skippy_quips as Q
    n = bad = unhelpful = 0
    problems = []
    for t in turns:
        if t["route"] != "talk":
            continue
        for level in ("gentle", "savage"):
            line, cat = ST.fallback_line(t["text"], level)
            n += 1
            probs = ST.rubric_problems(line, None)
            ok, why, _ = Q.validate_line(line, None, level)
            if not ok:
                probs.append("validator: %s" % why)
            if probs:
                bad += 1
                problems.append((t["id"], level, line, probs))
        kw = (t.get("rubric") or {}).get("kw")
        if kw:
            line, _ = ST.fallback_line(t["text"], "gentle")
            if ST.rubric_problems(line, {"kw": kw}) != []:
                unhelpful += 1
    return {"talk_replies": n, "rubric_failures": bad, "keyworded_turns_not_helpful_by_fallback": unhelpful, "problems": problems[:10]}


def pct(x: float) -> str:
    return "%5.1f%%" % (100 * x)


def print_report(res: dict, show: int) -> None:
    m = res["metrics"]
    print(f"turns: {m['turns']}")
    print(f"addressee precision {pct(m['addressee_precision'])}   recall {pct(m['addressee_recall'])}")
    print(f"FALSE-WAKE rate {pct(m['false_wake_rate'])}  ({m['false_wake_n']} of {m['not_for_him_n']} lines that were not for him got an answer)")
    print(f"MISSED-WAKE rate {pct(m['missed_wake_rate'])}  ({m['missed_wake_n']} of {m['for_him_n']} lines for him got no answer)")
    print(f"route accuracy {pct(m['route_accuracy'])} overall, {pct(m['route_accuracy_addressed'])} for lines that should be answered")
    print(f"ambiguous lines: {m['maybe_n']}; recognised as maybe {m['maybe_kept_maybe']}; answered anyway {m['maybe_answered']}")
    print("per category (n, route/answer ok, false wakes, missed):")
    for cat, c in sorted(m["by_category"].items()):
        print(f"  {cat:18s} {c['n']:3d}  {c['ok']:3d}  {c['false_wake']:2d}  {c['missed']:2d}")
    bad = [r for r in res["rows"] if r["got"]["final"] != r["turn"]["route"]]
    if show and bad:
        print(f"\nfirst {min(show, len(bad))} of {len(bad)} disagreements:")
        for r in bad[:show]:
            t, g = r["turn"], r["got"]
            print(f"  {t['id']} [{t['cat']}] {t['text'][:60]!r}  want {t['addressee']}/{t['route']}  got {g['addressed']}({g['score']})/{g['final']}  {'; '.join(g['reasons'][:3])}")


# ============================================================================================================ the live model sample
def sample_talk(turns: list[dict], n: int) -> list[dict]:
    talk = [t for t in turns if t["route"] == "talk" and t["addressee"] == "yes"]
    if n >= len(talk):
        return talk
    step = len(talk) / n
    return [talk[int(i * step)] for i in range(n)]


def run_model(turns: list[dict], n: int, max_calls: int, go: bool) -> int:
    import skippy_turns as ST
    sample = sample_talk(turns, n)
    secret, src = ST.find_key()
    print(f"live sample: {len(sample)} talk turns; up to {len(sample) * ST.MAX_ATTEMPTS} model calls in the worst case (one per turn when the first reply is valid).")
    print(f"key present: {'yes' if secret else 'no'} ({src}); route: {'frontier ' + ST.frontier_model() if secret else 'the local Ollama model qwen3:8b'}")
    worst = len(sample) * ST.MAX_ATTEMPTS
    if worst > max_calls:
        print(f"refusing: the worst case of {worst} calls is more than --max-calls {max_calls}")
        return 2
    if not go:
        print("nothing was called. Add --go to run it.")
        return 0

    import skippy_bridge as B
    eng = ST.make_engine(B)
    eng.ledger = ST.Ledger(None)          # the sample does not touch the day's real counters
    eng.cache = ST.TurnCache()
    problems = 0
    for t in sample:
        text = Turns_strip(t)
        r = eng.turn({"text": text, "roast": "gentle", "facts": ["gravity here is 1.00 of Earth's"], "sit": "g=norm"})
        probs = ST.rubric_problems(r["say"], t.get("rubric"))
        problems += bool(probs)
        print(f"  {t['id']} {r['source']:8s} tries {r['tries']} {'OK ' if not probs else 'BAD'} {text[:40]!r} -> {r['say']!r} {probs or ''}")
    st = eng.ledger.view(0)
    print(f"done: {len(sample)} turns, {problems} rubric failures; calls {st['calls']}, tokens in {st['in_tokens']} out {st['out_tokens']}, local {st['local']}, fallbacks {st['fallbacks']}")
    return 0


def Turns_strip(t: dict) -> str:
    """The sentence a model would get: the wake name taken off the front."""
    rest = t["text"]
    first, _, tail = rest.partition(" ")
    for n in t["ctx"]["names"]:
        if first.lower().strip(",") == n.lower().split(" ")[0]:
            return tail or rest
    if first.lower() in ("hey", "ok") and tail:
        return Turns_strip({**t, "text": tail})
    return rest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--json", help="write the metrics to this file")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--max-calls", type=int, default=40, help="refuse when the worst case (3 attempts per turn) would exceed this")
    a = ap.parse_args(argv)
    turns = load_corpus(Path(a.corpus))
    if a.model:
        return run_model(turns, a.sample, a.max_calls, a.go)
    T = Turns()
    res = score(turns, T.decide)
    print_report(res, a.show)
    stress = [t for t in turns if t["cat"].startswith("stress_")]
    if stress:
        sm = score(stress, T.decide)["metrics"]
        print(f"\nstress set alone ({len(stress)} turns written in one sitting and run once): false-wake {pct(sm['false_wake_rate'])} ({sm['false_wake_n']}/{sm['not_for_him_n']}), "
              f"missed-wake {pct(sm['missed_wake_rate'])} ({sm['missed_wake_n']}/{sm['for_him_n']}), route accuracy {pct(sm['route_accuracy'])}")
    rc = reply_check(turns)
    print(f"\noffline reply floor (the fallback line for every talk turn, gentle and savage): {rc['talk_replies']} lines, {rc['rubric_failures']} rubric failures, "
          f"{rc['keyworded_turns_not_helpful_by_fallback']} keyworded questions where the fallback alone is not helpful (the model must be)")
    for p in rc["problems"]:
        print("   ", p)
    if a.json:
        Path(a.json).write_text(json.dumps({"metrics": res["metrics"], "reply_floor": {k: v for k, v in rc.items() if k != "problems"}}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
