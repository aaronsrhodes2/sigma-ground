"""Which cascade groups can the trained model replace? Group by group, from two runs.

The legacy run (all tweaks on) lets each deterministic classifier answer its questions
before the model is asked. The model-only run (`--tweaks model-only`) switches the cascade
off, so the same questions reach the model alone. For every cascade group this compares the
two on the questions that group answered in the legacy run:

    python tools/tweak_ablation.py legacy=<run.json> model_only=<run.json> [--min-n 3] [--json out.json]

A group is RETIRE when the model alone gets at least as many right as the classifier did
(and there are at least --min-n questions to judge from); otherwise KEEP, and the questions
the model missed are listed. A group whose classifier does arithmetic itself (result_override:
optics, classical_intro, modern, astro) is marked "computes in harness": Platinum rule 21 says
Mentat does the physics, so for those a KEEP is a training gap to close, not a settled answer.

Both runs are scored with the benchmark's own scorer, unchanged. The runs must have been made
with the same model, prompt and cascade settings except for the tweaks; the tweak profile each
record carries is checked and printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_router_runs import load  # noqa: E402
from sigma_ground.mcp.benchmark import tweaks as T  # noqa: E402

COMPUTES_IN_HARNESS = {"astro", "optics", "classical_intro", "modern"}
GROUP_OF_KEY = {
    "frontier_router_hit": "frontier", "new_tools_router_hit": "new_tools",
    "refusal_classifier_hit": "refusal", "clarification_hit": "clarification",
    "conversion_classifier_hit": "conversion", "gr_classifier_hit": "gr",
    "cosmology_classifier_hit": "cosmology", "math_classifier_hit": "math",
    "thermo_classifier_hit": "thermo", "modern_classifier_hit": "modern",
    "classical_intro_hit": "classical_intro", "optics_classifier_hit": "optics",
    "astro_classifier_hit": "astro", "semantic_router_hit": "semantic",
}


def group_of(rec: dict) -> str | None:
    for key, group in GROUP_OF_KEY.items():
        if rec.get(key):
            return group
    return None


def profile_of(run: dict) -> str:
    snaps = {json.dumps(r.get("tweaks", {"profile": "legacy", "off": [], "on": []}), sort_keys=True)
             for r in run["recs"].values()}
    if len(snaps) != 1:
        return f"MIXED ({len(snaps)} different tweak settings in one run)"
    return T.label(json.loads(next(iter(snaps))))


def compare(legacy: dict, model_only: dict, min_n: int) -> dict:
    groups: dict[str, dict] = {}
    for qid, rec in legacy["recs"].items():
        g = group_of(rec)
        if g is None or qid not in model_only["rows"]:
            continue
        row = groups.setdefault(g, {"n": 0, "cascade_correct": 0, "model_correct": 0, "model_missed": [],
                                    "model_errors": 0})
        row["n"] += 1
        row["cascade_correct"] += bool(legacy["rows"][qid].correct)
        ok = bool(model_only["rows"][qid].correct)
        row["model_correct"] += ok
        if not ok:
            row["model_missed"].append(qid)
        row["model_errors"] += "router_error" in model_only["recs"][qid]
    for g, row in groups.items():
        row["computes_in_harness"] = g in COMPUTES_IN_HARNESS
        row["verdict"] = ("RETIRE" if row["n"] >= min_n and row["model_correct"] >= row["cascade_correct"]
                          else "KEEP" if row["n"] >= min_n else "TOO FEW")
    return groups


def render(groups: dict, lp: str, mp: str) -> str:
    out = [f"cascade vs the model alone ({lp} run vs {mp} run)", "",
           "| group | questions | cascade right | model alone right | verdict | notes |", "|---|---|---|---|---|---|"]
    for g, r in sorted(groups.items()):
        notes = []
        if r["computes_in_harness"]:
            notes.append("computes in harness")
        if r["model_errors"]:
            notes.append(f"{r['model_errors']} router errors")
        if r["model_missed"]:
            notes.append("model missed " + ", ".join(r["model_missed"]))
        out.append(f"| cascade.{g} | {r['n']} | {r['cascade_correct']} | {r['model_correct']} | "
                   f"{r['verdict']} | {'; '.join(notes) or '-'} |")
    tot = [sum(r[k] for r in groups.values()) for k in ("n", "cascade_correct", "model_correct")]
    out.append(f"| all cascade questions | {tot[0]} | {tot[1]} | {tot[2]} | | |")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs=2, metavar="NAME=RUN.json", help="legacy=... model_only=...")
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    named = dict(x.split("=", 1) for x in a.runs)
    if set(named) != {"legacy", "model_only"}:
        ap.error("give legacy=... and model_only=...")
    legacy, model_only = load(Path(named["legacy"])), load(Path(named["model_only"]))
    lp, mp = profile_of(legacy), profile_of(model_only)
    groups = compare(legacy, model_only, a.min_n)
    print(render(groups, lp, mp))
    if a.json:
        a.json.write_text(json.dumps({"legacy_profile": lp, "model_only_profile": mp, "groups": groups}, indent=2),
                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
