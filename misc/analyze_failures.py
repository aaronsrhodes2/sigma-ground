"""One-shot failure analysis for the sigma-ground main run.

Reads results/sigma_ground_run.json + sigma_ground_scored.json and the
questions corpus, classifies each record into a failure mode, prints
the breakdown.

Not part of the daily_job -- this is a deeper ad-hoc analysis for when
you want to understand WHY things failed, not just WHAT failed.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    here = Path(__file__).parent
    results = here.parent / "sigma_ground" / "mcp" / "benchmark" / "results"
    with (results / "sigma_ground_run.json").open(encoding="utf-8") as f:
        sg = json.load(f)
    with (results / "sigma_ground_scored.json").open(encoding="utf-8") as f:
        scored = json.load(f)["rows"]
    q_path = here.parent / "sigma_ground" / "mcp" / "benchmark" / "questions.json"
    with q_path.open(encoding="utf-8") as f:
        qs = json.load(f)

    correct_by_id = {r["id"]: r["correct"] for r in scored}
    q_by_id = {q["id"]: q for q in qs}
    sg_by_id = {r["id"]: r for r in sg}

    buckets: dict[str, list[dict]] = {
        "CORRECT": [],
        "FITTED_DUE_TO_INCOMPETENCE": [],
        "WRONG_TOOL_USED": [],
        "NO_TOOL_CALLED": [],
        "EXCEEDED_MAX_TURNS": [],
        "WRONG_VALUE_FROM_CORRECT_TOOL": [],
        "ERROR": [],
    }

    for qid, q in q_by_id.items():
        r = sg_by_id.get(qid)
        if not r:
            continue
        val = r.get("extracted_value")
        ans = r.get("answer_text", "") or ""
        tcs = r.get("tool_calls", []) or []
        tool_names = [tc.get("name") for tc in tcs]
        expected = q.get("primary_tool_expected")
        correct = correct_by_id.get(qid, False)
        rec = {
            "id": qid, "domain": q["domain"],
            "q": q["question"][:90],
            "expected": expected,
            "tools_used": list(dict.fromkeys(tool_names))[:5],
        }
        if correct:
            buckets["CORRECT"].append(rec)
            continue
        if ans.startswith("<ERROR"):
            buckets["ERROR"].append(rec)
            continue
        if ans == "<exceeded max turns>":
            buckets["EXCEEDED_MAX_TURNS"].append(rec)
            continue
        if isinstance(val, str) and "Fitted due to incompetence" in val:
            buckets["FITTED_DUE_TO_INCOMPETENCE"].append(rec)
            continue
        if not tcs:
            buckets["NO_TOOL_CALLED"].append(rec)
            continue
        if expected and expected not in tool_names:
            buckets["WRONG_TOOL_USED"].append(rec)
            continue
        buckets["WRONG_VALUE_FROM_CORRECT_TOOL"].append(rec)

    print("=== FAILURE MODE BREAKDOWN (sigma-ground main, 150 questions) ===")
    print()
    for cat, items in buckets.items():
        pct = 100.0 * len(items) / 150
        print(f"  {cat:35s} {len(items):3d}  ({pct:5.1f}%)")
    print()

    print("=== TOP 15 WRONG-TOOL CHOICES (what Qwen kept calling wrongly) ===")
    wrong_choices = Counter()
    for r in buckets["WRONG_TOOL_USED"]:
        for t in r["tools_used"][:2]:
            wrong_choices[t] += 1
    for tool, n in wrong_choices.most_common(15):
        print(f"  {tool:40s} {n} times")
    print()

    print("=== TOOLS QWEN SHOULD HAVE CALLED BUT DIDN'T ===")
    missed_calls = Counter()
    for r in buckets["WRONG_TOOL_USED"]:
        if r["expected"]:
            missed_calls[r["expected"]] += 1
    for tool, n in missed_calls.most_common(15):
        print(f"  {tool:40s} {n} questions wanted this")
    print()

    print("=== DOMAIN-LEVEL FAILURE TYPE ===")
    domain_buckets: dict[str, Counter] = defaultdict(Counter)
    for cat, items in buckets.items():
        for r in items:
            domain_buckets[r["domain"]][cat] += 1
    print(f"{'domain':32s} {'correct':>7s}  {'wrong_tool':>10s}  "
          f"{'no_tool':>7s}  {'max_t':>5s}  {'fitted':>6s}  {'tot':>3s}")
    for dom in sorted(domain_buckets):
        counts = domain_buckets[dom]
        total = sum(counts.values())
        print(f"{dom:32s} {counts['CORRECT']:>7d}  "
              f"{counts['WRONG_TOOL_USED']:>10d}  "
              f"{counts['NO_TOOL_CALLED']:>7d}  "
              f"{counts['EXCEEDED_MAX_TURNS']:>5d}  "
              f"{counts['FITTED_DUE_TO_INCOMPETENCE']:>6d}  "
              f"{total:>3d}")
    print()

    print("=== SAMPLE WRONG-TOOL FAILURES (top 12) ===")
    for r in buckets["WRONG_TOOL_USED"][:12]:
        print(f"  {r['id']} ({r['domain']})")
        print(f"    Q: {r['q']}")
        print(f"    expected: {r['expected']}")
        print(f"    qwen tried: {r['tools_used']}")
    print()

    if buckets["NO_TOOL_CALLED"]:
        print("=== NO-TOOL-CALLED FAILURES (Qwen reasoned in prose) ===")
        for r in buckets["NO_TOOL_CALLED"][:8]:
            print(f"  {r['id']} ({r['domain']}): {r['q']}")
            print(f"    expected: {r['expected']}")
        print()


if __name__ == "__main__":
    main()
