"""Fold the improvement-distillation days into permanent tests (misc/DISTILLATION_BRANCH_PLAN.md, the Captain 2026-10-31).

    python tools/distill_fold.py            # rebuild sigma_ground/mcp/benchmark/distill_pinned.json from the day files
    python tools/distill_fold.py --check    # rebuild in memory; fail if the stored file differs (the test runs this)

Every day the loop asks a new set and scores it against OUR oracle. A row is worth keeping for good when it teaches something the three standing suites (the 150
physics, the 329 maths, the 18 adversarial) do not already teach. Two kinds are pinned:

  new_combination  Mentat answered it right (the oracle agrees) and the row's TOOL COMBINATION has not been seen: the sorted tuple of tool names in its plan
                   (`circular_orbit_velocity` then `momentum` is one combination; `curl` alone is another). "Seen" is the standing suites plus every earlier
                   pinned row, days taken in order, so the first row to show a combination is the one that is pinned.
  fixed_miss       Mentat missed it, and misc/distill_fixes.json says that cause for that tool is fixed: the row is the regression test for the fix.

Not pinned, listed under `open_misses` so nobody forgets them: a miss whose fix is not recorded as fixed. Not pinned and not listed: rows with no oracle (the wild
questions are for a person to read), rows not yet asked, and rows whose fix is recorded as `set wording` (the oracle, not Mentat, was at fault).

A pinned row carries the question as asked, the plan, the oracle's answer and its kind. It is run two ways: `tests/test_distill_pinned.py` runs the plan through the
tools in process (the tool's behaviour, no router, no network), and the live suite `pinned` (benchmark/live_ask.py) asks the deployed Mentat the question itself.
For a physics row the oracle is the tool at the spec's arguments, so the test pins the tool against its past self; for a maths row it is router_math's independent
sympy check, so it pins the answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "sigma_ground" / "mcp" / "benchmark"
DAYS = BENCH / "results" / "distill"
OUT = BENCH / "distill_pinned.json"
FIXES = REPO / "misc" / "distill_fixes.json"
MATH_SUITE = REPO / "local-cache" / "router_math_suite.json"


def signature(expected: dict) -> tuple[str, ...]:
    """The tool combination of a row: every tool its plan runs, sorted (a single-tool row is a one-tuple)."""
    plan = expected.get("plan")
    tools = [s["tool"] for s in plan] if plan else [expected.get("tool")]
    return tuple(sorted(t for t in tools if t))


def standing_signatures() -> set[tuple[str, ...]]:
    """What the three standing suites already teach."""
    seen: set[tuple[str, ...]] = set()
    for q in json.loads((BENCH / "questions.json").read_text(encoding="utf-8")):
        tools = [q.get("primary_tool_expected"), *(q.get("secondary_tools_expected") or [])]
        sig = tuple(sorted(t for t in tools if t))
        if sig:
            seen.add(sig)
    if MATH_SUITE.is_file():                                      # local-cache: absent on a fresh clone, and then every maths tool counts as new
        for it in json.loads(MATH_SUITE.read_text(encoding="utf-8"))["items"]:
            seen.add(signature(it.get("expected") or {}))
    return seen


def day_files() -> list[tuple[str, dict, dict]]:
    """[(date, scored day, set file)] in date order, for the days that have both."""
    out = []
    for f in sorted(DAYS.glob("distill-????-??-??.json")):
        day, setf = json.loads(f.read_text(encoding="utf-8")), f.with_name(f.stem + ".set.json")
        if setf.is_file():
            out.append((f.stem.removeprefix("distill-"), day, json.loads(setf.read_text(encoding="utf-8"))))
    return out


def fixes() -> dict:
    return json.loads(FIXES.read_text(encoding="utf-8")) if FIXES.is_file() else {}


def build() -> dict:
    seen = standing_signatures()
    fx = fixes()
    pinned, open_misses = [], []
    for date, day, setfile in day_files():
        items = {it["id"]: it for it in setfile["items"]}
        for row in day["rows"]:
            it = items.get(row["id"])
            exp = (it or {}).get("expected") or {}
            if not it or exp.get("result") is None or exp.get("oracle") in (None, "none") or row.get("cause") in ("unasked", "unverified"):
                continue
            sig = signature(exp)
            cause = row.get("cause")
            reason = None
            if cause is None and row.get("agrees"):
                if sig not in seen:
                    reason = "new_combination"
            else:
                status = (fx.get(f"{cause}|{exp.get('tool')}") or {}).get("status")
                if status == "fixed":
                    reason = "fixed_miss"
                elif status != "set wording":
                    open_misses.append({"id": row["id"], "day": date, "question": row["question"], "cause": cause, "tool": exp.get("tool"), "got": row.get("mentat_value")})
            if reason:
                seen.add(sig)
                pinned.append({"id": row["id"], "day": date, "reason": reason, "question": row["question"], "domain": row.get("domain"), "signature": list(sig),
                               "expected": {k: exp[k] for k in ("tool", "args", "plan", "result", "units", "oracle") if k in exp}})
    return {"what": "Rows of the improvement-distillation days kept for good (tools/distill_fold.py): a new tool combination Mentat answered right, or the regression test of a recorded fix.",
            "rules": "see the docstring of tools/distill_fold.py", "pinned": pinned, "open_misses": open_misses}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    doc = build()
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if a.check:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print("distill_pinned.json is missing or differs from a rebuild (run tools/distill_fold.py)", file=sys.stderr)
            return 1
        print("distill_pinned.json matches a rebuild")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    by = {}
    for p in doc["pinned"]:
        by[p["reason"]] = by.get(p["reason"], 0) + 1
    print(f"wrote {OUT.relative_to(REPO)}: {len(doc['pinned'])} pinned ({', '.join(f'{k} {v}' for k, v in by.items())}); {len(doc['open_misses'])} open misses")
    for m in doc["open_misses"]:
        print("  open:", m["day"], m["cause"], m["tool"], "|", m["question"][:80])
    return 0


if __name__ == "__main__":
    sys.exit(main())
