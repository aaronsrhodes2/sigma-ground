"""Daily benchmark job — incremental Wolfram + Gemini queries + failure tracking.

Run once daily (Windows Task Scheduler or cron):
  python -m sigma_ground.mcp.benchmark.daily_job

What it does, in order:

  1. WOLFRAM ALPHA -- identify unanswered questions, reformulate the
     phrasing (Wolfram parses terse queries best), retry up to the
     daily free-tier quota (~90/day).
  2. GEMINI -- identify questions still without a valid Gemini answer,
     re-query at low pace.
  3. SCORE -- re-score all three systems against ground truth, write
     {system}_scored.json files.
  4. CATALOG FAILURES -- enumerate each failed question per system:
       - sigma-ground "Fitted due to incompetence" -> library gap
       - sigma-ground wrong-tool / wrong-extraction -> discoverability gap
       - Wolfram success=false -> phrasing limit
       - Gemini wrong answer -> bare-LLM hallucination
     into IMPROVEMENT_PLAN.md (appended, never overwritten).
  5. PRINT a one-screen summary.

Idempotent: re-running same day just re-checks any new failures. Won't
exceed daily API caps (Wolfram resume already skips, Gemini paces at
0.5s/query).

Scheduling on Windows (run daily at 6 AM):

  schtasks /create /tn "sigma-ground benchmark daily" /tr ^
    "python -m sigma_ground.mcp.benchmark.daily_job" /sc daily /st 06:00

(Or use Task Scheduler GUI: New Task -> Trigger "daily 6:00 AM" ->
Action "python -m sigma_ground.mcp.benchmark.daily_job".)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


# ============================================================
# WOLFRAM REFORMULATION
# ============================================================
# Wolfram's NLP works best on terse, math-flavored queries:
#   GOOD: "speed of light in m/s"
#   GOOD: "free fall time from 10 m"
#   GOOD: "1 light year in meters"
#   BAD:  "If I drop a copper ball from 10 m at sea level, how many seconds before it hits the ground?"
#
# The strategy: try the verbatim question first (some succeed), then
# try a sequence of progressively-more-stripped reformulations. First
# one that returns numpods > 0 wins. We attribute the response to the
# reformulation that succeeded so we can learn which patterns work.

_CONVERSATIONAL_PREFIXES = [
    r"^if i\s+",
    r"^if you\s+",
    r"^how many\s+",
    r"^how much\s+",
    r"^how fast\s+",
    r"^how far\s+",
    r"^how long\s+",
    r"^what(?:'s| is)\s+(?:the\s+)?",
    r"^what(?:'s| is)\s+",
    r"^where(?:'s| is)\s+",
    r"^when\s+",
    r"^calculate\s+",
    r"^find\s+(?:the\s+)?",
    r"^compute\s+(?:the\s+)?",
    r"^determine\s+(?:the\s+)?",
    r"^suppose\s+",
    r"^assume\s+",
    r"^a\s+",
    r"^an\s+",
    r"^the\s+",
]

_CONVERSATIONAL_SCAFFOLDING = [
    (r"at sea level\b", ""),
    (r"\bdoes (it|he|she|the \w+)\b", ""),
    (r"\bwould (it|he|she|the \w+)\b", ""),
    (r"\bwill (it|he|she|the \w+)\b", ""),
    (r"\b(would|could|should)\b", ""),
    (r"\bbefore (it|he|she|the \w+) (hits|reaches|arrives|gets to)\b",
     "to ground"),
    (r"\bplease\b", ""),
    (r"\bapproximately\b", ""),
    (r"\babout\b", ""),
    (r"\bin (a|an) \w+ container\b", ""),
    (r"\bassuming \w+ \w+\b", ""),
    (r"\?$", ""),  # trailing question mark
]


def reformulate_for_wolfram(question: str) -> list[str]:
    """Return a list of phrasings to try, terse first to verbose last.

    Wolfram parses terse queries better, so try minimal forms first.
    """
    variants: list[str] = []
    s = question.strip()

    # Variant 1: strip leading conversational prefix
    stripped = s
    for pat in _CONVERSATIONAL_PREFIXES:
        new = re.sub(pat, "", stripped, count=1, flags=re.IGNORECASE)
        if new != stripped:
            stripped = new
            break
    if stripped != s:
        variants.append(stripped.rstrip(".? "))

    # Variant 2: strip ALL conversational scaffolding inline
    scaffold_free = s
    for pat, repl in _CONVERSATIONAL_SCAFFOLDING:
        scaffold_free = re.sub(pat, repl, scaffold_free, flags=re.IGNORECASE)
    for pat in _CONVERSATIONAL_PREFIXES:
        scaffold_free = re.sub(pat, "", scaffold_free, count=1, flags=re.IGNORECASE)
    scaffold_free = re.sub(r"\s+", " ", scaffold_free).strip(".? ")
    if scaffold_free and scaffold_free != stripped:
        variants.append(scaffold_free)

    # Variant 3: original
    variants.append(s)

    # Dedupe while preserving order
    seen = set()
    out = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ============================================================
# RUN-RECORD HELPERS
# ============================================================

def _load_run(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _record_is_good(rec: dict) -> bool:
    """A record is 'good' if it has a non-None extracted_value AND no error/timeout."""
    ans = rec.get("answer_text", "") or ""
    if ans.startswith("<ERROR") or ans == "<exceeded max turns>":
        return False
    val = rec.get("extracted_value")
    if val is None:
        return False
    # Reject "Fitted due to incompetence" strings dumped into the value.
    if isinstance(val, str) and "[SOURCE:" in val:
        return False
    return True


def _ids_needing_query(run_records: list[dict], all_qids: list[str]) -> list[str]:
    """Return ids in `all_qids` that aren't successfully answered in `run_records`."""
    good = {r["id"] for r in run_records if _record_is_good(r)}
    return [qid for qid in all_qids if qid not in good]


# ============================================================
# WOLFRAM STEP
# ============================================================

def run_wolfram_step(questions: list[dict], existing_records: list[dict],
                       app_id: str, pace_per_day: int, pause_s: float,
                       output_path: Path) -> dict:
    """Query Wolfram for unanswered questions, with reformulation."""
    from sigma_ground.mcp.benchmark.run_wolfram import _query_wolfram_raw, _extract_value
    all_qids = [q["id"] for q in questions]
    by_id = {q["id"]: q for q in questions}
    needed = _ids_needing_query(existing_records, all_qids)
    print(f"  [Wolfram] {len(needed)} unanswered; will attempt up to "
          f"{pace_per_day} today")
    by_record = {r["id"]: r for r in existing_records}
    queries_today = 0
    new_ok = 0
    new_attempted = 0
    for qid in needed:
        if queries_today >= pace_per_day:
            print(f"  [Wolfram] daily cap {pace_per_day} reached; stopping")
            break
        q = by_id[qid]
        variants = reformulate_for_wolfram(q["question"])
        success = False
        for variant_idx, variant in enumerate(variants):
            try:
                rows = _query_wolfram_raw(app_id, variant, timeout_s=30)
            except Exception as e:
                rows = []
                err = str(e)[:200]
            else:
                err = None
            queries_today += 1
            if rows:
                # extract primary result pod
                primary = ""
                for title, pid, text in rows:
                    if (pid and "Result" in pid) or title == "Result":
                        primary = text
                        break
                joined = "\n".join(f"[{t}] {x}" for t, _, x in rows)
                primary = primary or joined
                val, units = _extract_value(primary)
                rec = {
                    "id": qid,
                    "system": "wolfram",
                    "answer_text": joined,
                    "primary_result_pod": primary,
                    "extracted_value": val,
                    "extracted_units": units,
                    "elapsed_s": 0.0,
                    "wolfram_variant_used": variant_idx,
                    "wolfram_variant_text": variant,
                }
                by_record[qid] = rec
                success = val is not None
                if success:
                    new_ok += 1
                    print(f"    {qid}: OK via variant {variant_idx}: {variant[:60]!r}")
                break
            if queries_today >= pace_per_day:
                break
            time.sleep(pause_s)
        new_attempted += 1
        if not success:
            rec = by_record.get(qid, {})
            rec.update({
                "id": qid,
                "system": "wolfram",
                "answer_text": "<no parseable result after reformulation>",
                "extracted_value": None,
                "extracted_units": "",
                "elapsed_s": 0.0,
                "wolfram_variants_tried": len(variants),
            })
            by_record[qid] = rec
        time.sleep(pause_s)
        # save after each question (resumable)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(list(by_record.values()), f, indent=2, default=str)
    return {"new_attempted": new_attempted, "new_ok": new_ok,
              "queries_consumed": queries_today,
              "remaining": max(0, len(needed) - new_attempted)}


# ============================================================
# GEMINI STEP
# ============================================================

def run_gemini_step(questions: list[dict], existing_records: list[dict],
                      api_key: str, model: str, pause_s: float,
                      output_path: Path) -> dict:
    """Gemini removed from this build — a no-op so the daily job still runs.

    Deckard's researcher and this benchmark are now fully local (qwen via
    ollama); the Gemini comparison baseline is no longer maintained. Kept as a
    stub (rather than deleted) so the runner's call sites stay intact.
    """
    return {"error": "gemini removed from this build", "new_attempted": 0, "new_ok": 0}


# ============================================================
# FAILURE CATALOG / PLAN FILE
# ============================================================

_PLAN_HEADER = """# sigma-ground benchmark — Improvement Plan

This file accumulates findings from `daily_job.py`. Each daily section
catalogs the failures that day, so when you sit down to work on the
benchmark you have a queue of concrete improvements to tackle.

Categories:
  - **LIBRARY GAP** — sigma-ground answered "Fitted due to incompetence".
    Build the missing tool.
  - **DISCOVERABILITY GAP** — sigma-ground has the tool but Qwen used the
    wrong one. Add keywords or pattern hints.
  - **WOLFRAM PHRASING** — Wolfram couldn't parse a question even after
    reformulation. Add a manual `wolfram_phrasing` field to the question.
  - **GEMINI HALLUCINATION** — Gemini gave a confidently-wrong answer.
    Worth keeping as evidence of bare-LLM failure modes.

---

"""


def catalog_failures(questions: list[dict], ground_truth: dict,
                       sg_records: list[dict], wf_records: list[dict],
                       gm_records: list[dict], plan_path: Path) -> dict:
    """Walk every question, classify each system's failure, write the plan."""
    from sigma_ground.mcp.benchmark.score import _score_one

    qmap = {q["id"]: q for q in questions}
    sg_by_id = {r["id"]: r for r in sg_records}
    wf_by_id = {r["id"]: r for r in wf_records}
    gm_by_id = {r["id"]: r for r in gm_records}

    library_gaps: list[dict] = []
    discoverability: list[dict] = []
    wrong_value_right_tool: list[dict] = []
    silent_non_response: list[dict] = []
    wolfram_phrasing: list[dict] = []
    gemini_halluc: list[dict] = []

    # Dict keys the 13 fast-path regex classifiers in run_sigma_ground.py
    # stash their tool choice under -- they bypass tool_calls entirely
    # (elapsed_s~0.1s, turns=0) but DO record which tool answered. Without
    # folding these in, every classifier-path answer looks like "no tool
    # called" and gets mislabeled a discoverability gap even when the real
    # defect is a wrong value or a ground-truth bug. refusal/conversion hits
    # aren't a tool NAME (they're a type string / True), so they're excluded.
    _CLASSIFIER_HIT_KEYS = (
        "frontier_router_hit", "new_tools_router_hit", "gr_classifier_hit",
        "cosmology_classifier_hit", "math_classifier_hit",
        "thermo_classifier_hit", "modern_classifier_hit",
        "optics_classifier_hit", "astro_classifier_hit",
        "classical_intro_hit",
    )

    for qid, q in qmap.items():
        truth = ground_truth.get(qid)
        if truth is None:
            continue
        sg = sg_by_id.get(qid, {})
        wf = wf_by_id.get(qid, {})
        gm = gm_by_id.get(qid, {})

        # sigma-ground analysis
        sg_val = sg.get("extracted_value")
        sg_ans = (sg.get("answer_text", "") or "")
        if isinstance(sg_val, str) and "[SOURCE: Fitted due to incompetence" in sg_val:
            library_gaps.append({
                "id": qid, "domain": q.get("domain"),
                "question": q["question"],
                "expected": truth.get("expected_value"),
                "expected_tool": q.get("primary_tool_expected"),
                "sigma_ground_said": sg_val[:200],
            })
        elif sg_val is not None:
            correct, _, _ = _score_one(sg, q, truth)
            if not correct:
                tcs = sg.get("tool_calls", []) or []
                expected_tool = q.get("primary_tool_expected")
                tool_names = [tc.get("name") for tc in tcs]
                for k in _CLASSIFIER_HIT_KEYS:
                    v = sg.get(k)
                    if v:
                        tool_names.append(v)
                if expected_tool and expected_tool not in tool_names:
                    discoverability.append({
                        "id": qid, "domain": q.get("domain"),
                        "question": q["question"],
                        "expected_tool": expected_tool,
                        "tools_qwen_tried": list(dict.fromkeys(tool_names))[:5],
                    })
                elif expected_tool:
                    # right tool, wrong final value -- e.g. a unit/reference-
                    # point bug, or a tool-call loop that never converges
                    wrong_value_right_tool.append({
                        "id": qid, "domain": q.get("domain"),
                        "question": q["question"],
                        "expected": truth.get("expected_value"),
                        "sigma_ground_extracted": sg_val,
                        "tool_used": expected_tool,
                    })
        elif sg_val is None:
            # zero tool calls, no classifier hit, no final text -- Qwen
            # refused to engage even after the harness's nudges. Invisible
            # to every other bucket until now (falls through the `elif
            # sg_val is not None` above by construction).
            silent_non_response.append({
                "id": qid, "domain": q.get("domain"),
                "question": q["question"],
                "expected_tool": q.get("primary_tool_expected"),
                "turns": sg.get("turns"),
                "nudges_sent": sg.get("nudges_sent"),
                "answer_text": sg_ans[:120],
            })

        # Wolfram analysis -- a question with NO Wolfram record at all (never
        # attempted, e.g. cut off by the daily pace cap) is `wf == {}`, same
        # symptom as one that was attempted and returned no parseable value:
        # both mean "no Wolfram answer we can score." The prior truthy-
        # answer_text requirement only caught the second case, hiding every
        # never-attempted question from the phrasing backlog.
        if wf.get("extracted_value") is None:
            wolfram_phrasing.append({
                "id": qid, "domain": q.get("domain"),
                "question": q["question"],
                "wolfram_variants_tried": wf.get("wolfram_variants_tried", 0),
                "attempted": bool(wf),
            })

        # Gemini analysis: only count CONFIDENT wrong (had a value, was wrong)
        if gm.get("extracted_value") is not None:
            correct, _, _ = _score_one(gm, q, truth)
            if not correct:
                gemini_halluc.append({
                    "id": qid, "domain": q.get("domain"),
                    "question": q["question"],
                    "expected": truth.get("expected_value"),
                    "gemini_gave": gm.get("extracted_value"),
                })

    # Compose plan entry
    today = dt.datetime.now().strftime("%Y-%m-%d")
    section: list[str] = [f"## {today}", ""]
    section.append(f"- LIBRARY GAP count:        {len(library_gaps)}")
    section.append(f"- DISCOVERABILITY GAP count: {len(discoverability)}")
    section.append(f"- WRONG VALUE (right tool) count: {len(wrong_value_right_tool)}")
    section.append(f"- SILENT NON-RESPONSE count: {len(silent_non_response)}")
    section.append(f"- WOLFRAM PHRASING count:    {len(wolfram_phrasing)}")
    section.append(f"- GEMINI HALLUCINATION count:{len(gemini_halluc)}")
    section.append("")

    if library_gaps:
        section.append("### LIBRARY GAP — build the missing tool")
        for g in library_gaps[:20]:
            section.append(f"- `{g['id']}` ({g['domain']}): {g['question'][:90]}")
            section.append(f"    expected tool: `{g.get('expected_tool')}`; expected value: {g.get('expected')}")
        if len(library_gaps) > 20:
            section.append(f"- ... +{len(library_gaps) - 20} more")
        section.append("")

    if discoverability:
        section.append("### DISCOVERABILITY GAP — keywords or pattern hint needed")
        for g in discoverability[:20]:
            section.append(f"- `{g['id']}` ({g['domain']}): {g['question'][:90]}")
            section.append(f"    expected: `{g['expected_tool']}` ; Qwen tried: {g['tools_qwen_tried']}")
        if len(discoverability) > 20:
            section.append(f"- ... +{len(discoverability) - 20} more")
        section.append("")

    if wrong_value_right_tool:
        section.append("### WRONG VALUE (right tool called) — units/loop/formula bug")
        for g in wrong_value_right_tool[:20]:
            section.append(f"- `{g['id']}` ({g['domain']}): {g['question'][:90]}")
            section.append(f"    tool: `{g['tool_used']}` ; expected {g['expected']}, "
                              f"got {g['sigma_ground_extracted']}")
        if len(wrong_value_right_tool) > 20:
            section.append(f"- ... +{len(wrong_value_right_tool) - 20} more")
        section.append("")

    if silent_non_response:
        section.append("### SILENT NON-RESPONSE — Qwen made no tool call, no answer text")
        for g in silent_non_response[:20]:
            section.append(f"- `{g['id']}` ({g['domain']}): {g['question'][:90]}")
            section.append(f"    expected tool: `{g['expected_tool']}` ; "
                              f"turns={g['turns']} nudges={g['nudges_sent']} "
                              f"said: {g['answer_text']!r}")
        if len(silent_non_response) > 20:
            section.append(f"- ... +{len(silent_non_response) - 20} more")
        section.append("")

    if wolfram_phrasing:
        section.append("### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question")
        for g in wolfram_phrasing[:10]:
            section.append(f"- `{g['id']}` ({g['domain']}): {g['question'][:90]}")
        if len(wolfram_phrasing) > 10:
            section.append(f"- ... +{len(wolfram_phrasing) - 10} more")
        section.append("")

    if gemini_halluc:
        section.append("### GEMINI HALLUCINATION (Gemini confident, wrong)")
        for g in gemini_halluc[:10]:
            section.append(f"- `{g['id']}` ({g['domain']}): expected {g['expected']}, "
                              f"gemini said {g['gemini_gave']}")
        if len(gemini_halluc) > 10:
            section.append(f"- ... +{len(gemini_halluc) - 10} more")
        section.append("")

    section.append("---")
    section.append("")

    if not plan_path.exists():
        plan_path.write_text(_PLAN_HEADER, encoding="utf-8")
    with plan_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(section) + "\n")

    return {
        "library_gaps": len(library_gaps),
        "discoverability": len(discoverability),
        "wrong_value_right_tool": len(wrong_value_right_tool),
        "silent_non_response": len(silent_non_response),
        "wolfram_phrasing": len(wolfram_phrasing),
        "gemini_halluc": len(gemini_halluc),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="main",
                        choices=["main", "adversarial"],
                        help="Which corpus to run.")
    parser.add_argument("--wolfram-pace", type=int, default=85,
                        help="Max Wolfram queries/day (free tier ~100/day).")
    parser.add_argument("--wolfram-pause", type=float, default=2.0)
    parser.add_argument("--gemini-model", default="gemini-2.5-flash",
                        help="Gemini model. Flash works on free tier "
                              "(250 req/day) which is plenty for the 150-Q "
                              "corpus. Use 'gemini-2.5-pro' only if billing "
                              "is on pay-as-you-go (not prepayment-depleted).")
    parser.add_argument("--gemini-pause", type=float, default=15.0,
                        help="Seconds between Gemini calls. Default 15s "
                              "respects free-tier flash rate limit (5 req/min) "
                              "with safety margin. On paid pay-as-you-go you "
                              "can drop to 0.5.")
    parser.add_argument("--skip-wolfram", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true")
    args = parser.parse_args()

    # Auto-load env vars from the dev-root .env
    from sigma_ground.mcp.benchmark import load_env_from_dev_root
    load_env_from_dev_root(verbose=True)

    here = Path(__file__).parent
    if args.corpus == "main":
        q_path = here / "questions.json"
        gt_path = here / "ground_truth.json"
        prefix = ""
    else:
        q_path = here / "adversarial_questions.json"
        gt_path = here / "adversarial_ground_truth.json"
        prefix = "adversarial_"
    results_dir = here / "results"
    results_dir.mkdir(exist_ok=True)
    plan_path = Path(__file__).parents[3] / "misc" / "IMPROVEMENT_PLAN.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    with q_path.open(encoding="utf-8") as f:
        questions = json.load(f)
    with gt_path.open(encoding="utf-8") as f:
        ground_truth = json.load(f)

    sg_path = results_dir / f"{prefix}sigma_ground_run.json"
    wf_path = results_dir / f"{prefix}wolfram_run.json"
    gm_path = results_dir / f"{prefix}gemini_run.json"

    print(f"\n=== Daily job: corpus={args.corpus} ({len(questions)} questions) ===")
    print(f"  plan file: {plan_path}")

    # Step 1: Wolfram
    if args.skip_wolfram:
        print("  [Wolfram] skipped")
    else:
        app_id = os.environ.get("WOLFRAM_ALPHA_APP_ID")
        if not app_id:
            print("  [Wolfram] WOLFRAM_ALPHA_APP_ID not set; skipping")
        else:
            existing = _load_run(wf_path)
            stats = run_wolfram_step(questions, existing, app_id,
                                       args.wolfram_pace, args.wolfram_pause,
                                       wf_path)
            print(f"  [Wolfram] {stats}")

    # Step 2: Gemini (prefer free-tier key over paid key -- the paid
    # key on Aaron's project is in prepayment mode with zero credits).
    if args.skip_gemini:
        print("  [Gemini] skipped")
    else:
        api_key = (os.environ.get("GEMINI_FREE_API_KEY")
                     or os.environ.get("GEMINI_API_KEY"))
        if not api_key:
            print("  [Gemini] no GEMINI_FREE_API_KEY or GEMINI_API_KEY; skipping")
        else:
            key_source = ("FREE" if os.environ.get("GEMINI_FREE_API_KEY")
                            else "paid")
            print(f"  [Gemini] using {key_source}-tier key")
            existing = _load_run(gm_path)
            stats = run_gemini_step(questions, existing, api_key,
                                      args.gemini_model, args.gemini_pause,
                                      gm_path)
            print(f"  [Gemini] {stats}")

    # Step 3: Re-score (write _scored.json files)
    from sigma_ground.mcp.benchmark.score import score_run, summarize
    print("  [Score]")
    for label, path in [("sigma_ground", sg_path), ("wolfram", wf_path),
                            ("gemini", gm_path)]:
        if not path.exists():
            continue
        rows = score_run(path, q_path, gt_path)
        summary = summarize(rows)
        scored_path = results_dir / f"{prefix}{label}_scored.json"
        with scored_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": [r.to_dict() for r in rows]},
                       f, indent=2, default=str)
        n_correct = summary["overall_correct"]
        n_total = summary["overall_total"]
        pct = summary["overall_accuracy_pct"]
        print(f"    {label:13s}: {n_correct:3d}/{n_total:3d} = {pct:5.1f}%")

    # Step 4-5: Catalog failures + append to plan
    sg_records = _load_run(sg_path)
    wf_records = _load_run(wf_path)
    gm_records = _load_run(gm_path)
    stats = catalog_failures(questions, ground_truth, sg_records, wf_records,
                                  gm_records, plan_path)
    print(f"  [Catalog] {stats}")
    print(f"  [Plan]    appended to {plan_path}")

    # Step 6: Print target-progress dashboard (from PROJECT_GOAL.md).
    _print_target_dashboard(results_dir, prefix)
    return 0


def _print_target_dashboard(results_dir: Path, prefix: str) -> None:
    """Print the current state of every PROJECT_GOAL.md target.

    Each row marks PASS / FAIL so the daily log surfaces 'we hit the
    target' the same morning it happens. The pytest xfail tests in
    test_targets.py are the CI-enforced version; this is the eyeball
    version that runs every day.
    """
    print()
    print("  +--------------------------------------------------------------+")
    print("  | TARGET DASHBOARD (PROJECT_GOAL.md)                           |")
    print("  +--------------------------------------------------------------+")

    def _scored(name: str) -> dict | None:
        p = results_dir / f"{prefix}{name}_scored.json"
        if not p.exists():
            return None
        with p.open(encoding="utf-8") as f:
            return json.load(f)

    def _run(name: str) -> list[dict] | None:
        p = results_dir / f"{prefix}{name}_run.json"
        if not p.exists():
            return None
        with p.open(encoding="utf-8") as f:
            return json.load(f)

    sg_scored = _scored("sigma_ground")
    wf_scored = _scored("wolfram")
    sg_run = _run("sigma_ground")
    adv_scored = _scored("adversarial_sigma_ground") if not prefix else None

    def _row(label: str, ok: bool, detail: str) -> None:
        marker = "PASS" if ok else "...."
        bullet = "X" if ok else " "
        print(f"  | [{bullet}] {marker} {label:<28s} {detail:<25s} |")

    # Target 1: overall >= Wolfram overall
    if sg_scored and wf_scored:
        sg_pct = sg_scored["summary"]["overall_accuracy_pct"]
        wf_pct = wf_scored["summary"]["overall_accuracy_pct"]
        _row("sg overall >= WA overall", sg_pct >= wf_pct,
              f"sg={sg_pct:.1f}% WA={wf_pct:.1f}%")
    # Target 2: match WA on subset WA got right (>=98%)
    if sg_scored and wf_scored:
        wf_ok_ids = {r["id"] for r in wf_scored["rows"] if r.get("correct")}
        sg_by_id = {r["id"]: r for r in sg_scored["rows"]}
        if wf_ok_ids:
            overlap = sum(1 for q in wf_ok_ids
                            if sg_by_id.get(q, {}).get("correct"))
            rate = overlap / len(wf_ok_ids)
            _row("sg matches WA-correct >=98%", rate >= 0.98,
                  f"{overlap}/{len(wf_ok_ids)} = {rate:.0%}")
        else:
            _row("sg matches WA-correct >=98%", False, "WA got 0 right")
    # Target 3: overall >= 80%
    if sg_scored:
        sg_pct = sg_scored["summary"]["overall_accuracy_pct"]
        _row("sg overall >= 80%", sg_pct >= 80.0, f"{sg_pct:.1f}%")
    # Target 4: every domain >= 80%
    if sg_scored:
        by_dom = sg_scored["summary"].get("by_domain", {})
        below = [d for d, s in by_dom.items() if s["pct"] < 80]
        _row("all 14 domains >= 80%", not below,
              f"{len(below)} below" if below else "all pass")
    # Target 5: median latency <= 30s
    if sg_run:
        times = sorted(r["elapsed_s"] for r in sg_run
                          if isinstance(r.get("elapsed_s"), (int, float)))
        if times:
            med = times[len(times) // 2]
            _row("median latency <= 30s", med <= 30.0, f"{med:.1f}s")
    # Target 6: library-gap <= 5%
    if sg_run:
        fitted = sum(1 for r in sg_run
                       if isinstance(r.get("extracted_value"), str)
                       and "Fitted due to incompetence" in (r.get("extracted_value") or ""))
        rate = fitted / max(len(sg_run), 1)
        _row("library-gap rate <= 5%", rate <= 0.05, f"{rate:.1%} ({fitted}/{len(sg_run)})")
    # Target 7 (adversarial): refusal rate >= 50%
    if adv_scored:
        refusal = [r for r in adv_scored["rows"]
                       if r.get("domain") in ("adversarial_false_premise",
                                                "adversarial_nonsense")]
        if refusal:
            ok = sum(1 for r in refusal if r.get("correct"))
            rate = ok / len(refusal)
            _row("adversarial refusal >= 50%", rate >= 0.5,
                  f"{ok}/{len(refusal)} = {rate:.0%}")
    # Target 8: conversation mode -- placeholder
    _row("conversation mode built", False, "roadmap")

    print("  +--------------------------------------------------------------+")


if __name__ == "__main__":
    sys.exit(main())
