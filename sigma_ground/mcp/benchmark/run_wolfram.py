"""Run the 150-question benchmark against Wolfram Alpha free-tier API.

Requires:
    pip install wolframalpha
    # Free APP ID from https://developer.wolframalpha.com/
    export WOLFRAM_ALPHA_APP_ID=<your APP_ID>

Free tier limits: 2000 calls/month, ~100/day. 150 questions in 2 days.
Runner is RESUMABLE -- it skips ids already in the output file.

Usage:
    python -m sigma_ground.mcp.benchmark.run_wolfram \
        --output sigma_ground/mcp/benchmark/results/wolfram_run.json \
        --pace-per-day 90 --pause-s 2

Strategy: ask the question verbatim. Wolfram's NLP usually figures it out.
Extract the "Result" pod's plaintext as the primary answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


_VALUE_RE = re.compile(
    r"([\-+]?[0-9]+(?:\.[0-9]+)?(?:\s*[\xd7x*]\s*10\^([\-+]?[0-9]+))?"
    r"|([\-+]?[0-9]+(?:\.[0-9]+)?[eE][\-+]?[0-9]+))"
)


def _extract_value(text: str) -> tuple[Any, str]:
    """Best-effort numeric extraction from Wolfram result text."""
    if not text:
        return None, ""
    # Try scientific notation first (e.g. "2.954 x 10^3 meters")
    m_sci = re.search(
        r"([\-+]?[0-9]+(?:\.[0-9]+)?)\s*[\xd7x*]\s*10\^([\-+]?[0-9]+)\s*([^\d\n]*)",
        text)
    if m_sci:
        try:
            mantissa = float(m_sci.group(1))
            exp = int(m_sci.group(2))
            val = mantissa * (10 ** exp)
            units = (m_sci.group(3) or "").strip()
            return val, units
        except ValueError:
            pass
    # Try standard scientific notation (e.g. "2.954e3")
    m_e = re.search(
        r"([\-+]?[0-9]+(?:\.[0-9]+)?[eE][\-+]?[0-9]+)\s*([^\d\n]*)", text)
    if m_e:
        try:
            val = float(m_e.group(1))
            units = (m_e.group(2) or "").strip()
            return val, units
        except ValueError:
            pass
    # Plain decimal at start of line or after a colon
    m_plain = re.search(r"[:=]\s*([\-+]?[0-9]+(?:\.[0-9]+)?)\s*([^\d\n]*)",
                         text)
    if m_plain:
        try:
            return float(m_plain.group(1)), (m_plain.group(2) or "").strip()
        except ValueError:
            pass
    # Fallback: first number in the string
    m = re.search(r"([\-+]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        try:
            return float(m.group(1)), ""
        except ValueError:
            pass
    return None, ""


def run_question(client, question: str, timeout_s: float = 30.0) -> dict:
    """Query Wolfram Alpha, return record dict."""
    t0 = time.time()
    try:
        res = client.query(question)
        # Collect text from all pods
        all_text = []
        result_pod_text = ""
        for pod in res.pods:
            for sub in pod.subpods:
                text = (sub.plaintext or "").strip()
                if not text:
                    continue
                all_text.append(f"[{pod.title}] {text}")
                # Wolfram tags primary result pods
                if (pod.id and "Result" in pod.id) or pod.title == "Result":
                    if not result_pod_text:
                        result_pod_text = text
        joined = "\n".join(all_text)
        primary = result_pod_text or joined
    except Exception as e:
        return {
            "answer_text": f"<ERROR: {e}>",
            "extracted_value": None,
            "extracted_units": "",
            "elapsed_s": time.time() - t0,
            "raw_pods": [],
        }
    elapsed = time.time() - t0
    val, units = _extract_value(primary)
    return {
        "answer_text": joined,
        "primary_result_pod": primary,
        "extracted_value": val,
        "extracted_units": units,
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "wolfram_run.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--pace-per-day", type=int, default=90,
                        help="Max queries per day (free tier ~100/day)")
    parser.add_argument("--pause-s", type=float, default=2.0,
                        help="Pause between queries to be polite")
    args = parser.parse_args()

    app_id = os.environ.get("WOLFRAM_ALPHA_APP_ID")
    if not app_id:
        print("ERROR: set WOLFRAM_ALPHA_APP_ID env var", file=sys.stderr)
        return 1

    try:
        import wolframalpha
    except ImportError:
        print("ERROR: pip install wolframalpha", file=sys.stderr)
        return 1

    client = wolframalpha.Client(app_id)

    here = Path(__file__).parent
    with (here / "questions.json").open(encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for rec in json.load(f):
                existing[rec["id"]] = rec

    out = list(existing.values())
    queries_today = 0
    for i, q in enumerate(questions):
        if q["id"] in existing:
            print(f"[{i+1}/{len(questions)}] {q['id']}: skipped (resume)")
            continue
        if queries_today >= args.pace_per_day:
            print(f"\nReached daily limit of {args.pace_per_day}. "
                  f"Run again tomorrow to resume.")
            break
        print(f"[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        result = run_question(client, q["question"])
        rec = {"id": q["id"], "system": "wolfram", **result}
        out.append(rec)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        queries_today += 1
        time.sleep(args.pause_s)
    print(f"\nWrote {args.output}. Queried {queries_today} this session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
