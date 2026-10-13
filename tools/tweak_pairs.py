"""Pair each tweak a profile switches off with the training data that replaces it.

A tweak may be off in a run only if the model was TRAINED to do its job. This reads a
training set (train.jsonl, v2 plan format) and, for every tweak the profile turns off, lists
the tools its replacement has to teach and how many examples teach each. A tool with fewer
than --min examples is a gap; any gap makes the exit status 1, and the pipeline then refuses
to run the model with that tweak off ("no double-ignores": a rule is never both removed and
untaught).

    python tools/tweak_pairs.py --data local-cache/router_dataset_v3 --profile trained
    python tools/tweak_pairs.py --data local-cache/router_dataset_v3 --profile model-only --min 20 --json out.json

Also checked: no training target uses a made-up parameter name (PARAM_ALIASES keys), because
a model trained on a Qwen-ism would be handed the very error the alias tweak used to hide.
Needs no Mentat server: tool and parameter names are read from the signatures in server.py.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sigma_ground.mcp.benchmark import router_plan as RP  # noqa: E402
from sigma_ground.mcp.benchmark import tweaks as T  # noqa: E402
from sigma_ground.mcp.benchmark.param_aliases import PARAM_ALIASES  # noqa: E402


def read_targets(path: Path) -> tuple[collections.Counter, collections.Counter, int]:
    """Examples per tool (a tool counts once per example), per (tool, parameter), and the total."""
    per_tool: collections.Counter = collections.Counter()
    per_param: collections.Counter = collections.Counter()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        try:
            plan = json.loads(json.loads(line)["messages"][-1]["content"])
        except (ValueError, KeyError, IndexError):
            continue
        canon = RP.canonical(plan)
        if canon is None:
            continue
        if canon["kind"] == "decline":
            per_tool["<decline>"] += 1
            continue
        tools = set()
        for s in canon["steps"]:
            tools.add(s["tool"])
            for k in s["args"]:
                per_param[(s["tool"], k)] += 1
        for t in tools:
            per_tool[t] += 1
    return per_tool, per_param, n


def needs(name: str, catalogue: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    """requirement label -> the tools (or tool.parameter) whose examples satisfy it."""
    tw = T.REGISTRY[name]
    if name.startswith("cascade."):
        tools = set(tw.covers) | T.emitted_tools(name.split(".", 1)[1], set(catalogue))
        return {t: (t,) for t in sorted(tools)}
    if name == "param_aliases":
        wanted: dict[str, tuple[str, ...]] = {}
        for canonical in sorted(set(PARAM_ALIASES.values())):
            owners = sorted(t for t, ps in catalogue.items() if canonical in ps)
            if owners:
                wanted[f"param {canonical}"] = tuple(f"{t}.{canonical}" for t in owners)
        return wanted
    return {t: (t,) for t in tw.covers}


def check(data: Path, profile: str, minimum: int, extra_off: list[str]) -> dict:
    T.configure(profile, off=extra_off)
    off = T.snapshot()["off"]
    catalogue = {name: set(params) for name, params in T.server_catalogue().items()}
    per_tool, per_param, total = read_targets(data / "train.jsonl")
    report: dict = {"data": str(data), "profile": profile, "examples": total, "min": minimum,
                    "tweaks": {}, "gaps": 0}
    for name in off:
        rows = []
        for label, keys in needs(name, catalogue).items():
            if name == "param_aliases":
                n = sum(per_param[tuple(k.split(".", 1))] for k in keys)
            else:
                n = sum(per_tool[k] for k in keys)
            blocked = label in T.GUARD_BLOCKED and n < minimum
            rows.append({"needs": label, "examples": n, "gap": n < minimum and not blocked, "blocked": blocked})
        gaps = [r for r in rows if r["gap"]]
        report["tweaks"][name] = {"needs": len(rows), "gaps": gaps,
                                  "blocked": [r["needs"] for r in rows if r["blocked"]]}
        report["gaps"] += len(gaps)
    unsafe = sorted(f"{t}.{k}" for (t, k), n in per_param.items()
                    if k in PARAM_ALIASES and k not in catalogue.get(t, set()))
    report["made_up_parameters_in_training_targets"] = unsafe
    report["gaps"] += len(unsafe)
    return report


def render(r: dict) -> str:
    out = [f"tweak pairing: profile {r['profile']}, {r['examples']} training examples, min {r['min']} per need",
           "", "| tweak switched off | needs | gaps | worst gaps | guard-blocked |", "|---|---|---|---|---|"]
    for name, t in r["tweaks"].items():
        worst = ", ".join(f"{g['needs']} ({g['examples']})"
                          for g in sorted(t["gaps"], key=lambda g: g["examples"])[:5]) or "-"
        out.append(f"| {name} | {t['needs']} | {len(t['gaps'])} | {worst} | {', '.join(t['blocked']) or '-'} |")
    if r["made_up_parameters_in_training_targets"]:
        out.append(f"\nMade-up parameter names in training targets: {r['made_up_parameters_in_training_targets']}")
    out.append(f"\n{'PAIRED' if not r['gaps'] else 'NOT PAIRED'}: {r['gaps']} gap(s)  (guard-blocked tools are not "
               "gaps: the benchmark guard forbids teaching them, so their classifier keeps answering)")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="dataset directory holding train.jsonl")
    ap.add_argument("--profile", default="trained", choices=list(T.PROFILES))
    ap.add_argument("--off", default="", help="more tweaks to switch off on top of the profile")
    ap.add_argument("--min", type=int, default=20, help="examples a tool needs (default 20)")
    ap.add_argument("--json", type=Path, help="also write the report here")
    a = ap.parse_args(argv)
    report = check(a.data, a.profile, a.min, [x for x in a.off.split(",") if x])
    print(render(report))
    if a.json:
        a.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if report["gaps"] else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
