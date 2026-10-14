"""Fill the wa-trickle queue (SkippyTel services/wa-trickle) with questions, and import the Wolfram answers we already have.

    python tools/wa_trickle_load.py math                  # the held-out math suite (329 questions; wild ones first, then round-robin over tools)
    python tools/wa_trickle_load.py historic              # the 150-question benchmark + the adversarial set, WITH their frozen Wolfram answers
    python tools/wa_trickle_load.py status                # what the service reports
    python tools/wa_trickle_load.py --url http://localhost:3010 math --dry-run

The service spends a small daily budget of Wolfram|Alpha's free tier (misc/CONTENDER_PLAN.md, the Captain's decision of 2026-10-13: no paid
Wolfram use). Priority decides the order; a question already answered (historic or by the API) is never asked again.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "sigma_ground" / "mcp" / "benchmark"
MATH_SUITE_CANDIDATES = [REPO / "local-cache" / "router_math_suite.json",
                         REPO / ".claude" / "worktrees" / "router-v2" / "local-cache" / "router_math_suite.json"]


def post(url: str, path: str, body: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def math_items(path: Path) -> list[dict]:
    """The held-out math suite as queue items. Order: the 25 hand-written questions first (the most informative), then the templated
    ones round-robin over the tools, so the first answers already cover every kind of question."""
    suite = json.loads(path.read_text(encoding="utf-8"))["items"]
    wild = [i for i in suite if i["kind"] == "wild"]
    by_recipe = defaultdict(list)
    for i in suite:
        if i["kind"] != "wild":
            by_recipe[i["recipe"]].append(i)
    ordered = list(wild)
    depth = 0
    while any(by_recipe.values()):
        for recipe in sorted(by_recipe):
            if by_recipe[recipe]:
                ordered.append(by_recipe[recipe].pop(0))
        depth += 1
    out = []
    for rank, i in enumerate(ordered):
        out.append({"id": i["id"], "question": i["question"], "priority": 10000 - rank,
                    "expected": {"result": i["expected"].get("result"), "tool": i["expected"]["tool"], "args": i["expected"].get("args"),
                                 "kind": i["kind"], "recipe": i["recipe"], "spec": i["expected"].get("spec")}})
    return out


def historic_items() -> dict[str, list[dict]]:
    """suite -> items with the Wolfram answer we already hold (never re-asked)."""
    suites = {}
    questions = json.loads((BENCH / "questions.json").read_text(encoding="utf-8"))
    questions = questions["questions"] if isinstance(questions, dict) else questions
    qmap = {q["id"]: q for q in questions}
    runs = json.loads((BENCH / "results" / "wolfram_run.json").read_text(encoding="utf-8"))
    items = []
    for r in runs:
        q = qmap.get(r["id"])
        if not q:
            continue
        items.append({"id": r["id"], "question": q["question"], "priority": 0,
                      "expected": {"value": q.get("expected_value"), "units": q.get("expected_units"), "tolerance_rel": q.get("tolerance_rel"),
                                   "domain": q.get("domain"), "tool": q.get("primary_tool_expected")},
                      "wolfram": {k: r.get(k) for k in ("answer_text", "primary_result_pod", "extracted_value", "extracted_units",
                                                        "elapsed_s", "wolfram_variant_used", "wolfram_variant_text")}})
    suites["physics150"] = items
    adv = BENCH / "results" / "adversarial_wolfram_run.json"
    aq = BENCH / "adversarial_questions.json"
    if adv.exists() and aq.exists():
        aqs = json.loads(aq.read_text(encoding="utf-8"))
        aqs = aqs["questions"] if isinstance(aqs, dict) else aqs
        amap = {q["id"]: q for q in aqs}
        a_items = []
        for r in json.loads(adv.read_text(encoding="utf-8")):
            q = amap.get(r["id"])
            if q:
                a_items.append({"id": r["id"], "question": q["question"], "priority": 0,
                                "expected": {k: q.get(k) for k in ("expected_behavior", "domain", "expected_value") if k in q},
                                "wolfram": {k: r.get(k) for k in ("answer_text", "primary_result_pod", "extracted_value", "elapsed_s")}})
        suites["adversarial18"] = a_items
    return suites


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["math", "historic", "status"])
    ap.add_argument("--url", default="http://localhost:3010")
    ap.add_argument("--suite-file", type=Path, help="math suite JSON (default: local-cache/router_math_suite.json)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.what == "status":
        print(json.dumps(get(a.url, "/status"), indent=1))
        return 0
    if a.what == "math":
        path = a.suite_file or next((p for p in MATH_SUITE_CANDIDATES if p.exists()), None)
        if path is None:
            print("no router_math_suite.json found (pass --suite-file)", file=sys.stderr)
            return 2
        items = math_items(path)
        print(f"math suite {path}: {len(items)} questions ({sum(1 for i in items if i['expected']['kind'] == 'wild')} hand-written first)")
        if not a.dry_run:
            print(post(a.url, "/enqueue", {"suite": "math329", "items": items}))
        return 0
    for suite, items in historic_items().items():
        print(f"{suite}: {len(items)} questions with a frozen Wolfram answer")
        if not a.dry_run:
            print(post(a.url, "/import_answers", {"suite": suite, "items": items}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
