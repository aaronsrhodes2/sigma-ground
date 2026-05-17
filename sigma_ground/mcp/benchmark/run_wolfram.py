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


def _query_wolfram_raw(app_id: str, question: str, timeout_s: float = 30.0
                          ) -> list[tuple[str, str, str]]:
    """Hit the Wolfram Alpha v2 query API and return [(pod_title, pod_id, plaintext)].

    We bypass the `wolframalpha` Python package (5.1.3) because its
    Content-Type assertion is too strict: it expects exactly
    `text/xml;charset=utf-8` but the server now returns
    `text/xml; charset=utf-8` (with a space). Raw urllib + xmltodict
    works fine.
    """
    import urllib.parse
    import urllib.request
    import xmltodict
    url = "https://api.wolframalpha.com/v2/query?" + urllib.parse.urlencode({
        "input":  question,
        "appid":  app_id,
        "output": "XML",
    })
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = xmltodict.parse(body)
    qr = parsed.get("queryresult", {}) or {}
    if qr.get("@success") != "true":
        return []
    pods = qr.get("pod", []) or []
    if isinstance(pods, dict):
        pods = [pods]
    rows: list[tuple[str, str, str]] = []
    for pod in pods:
        title = pod.get("@title", "") or ""
        pid = pod.get("@id", "") or ""
        subpods = pod.get("subpod", []) or []
        if isinstance(subpods, dict):
            subpods = [subpods]
        for sub in subpods:
            plaintext = (sub.get("plaintext") or "").strip()
            if plaintext:
                rows.append((title, pid, plaintext))
    return rows


def run_question(client_app_id: str, question: str,
                   timeout_s: float = 30.0) -> dict:
    """Query Wolfram Alpha, return record dict."""
    t0 = time.time()
    try:
        rows = _query_wolfram_raw(client_app_id, question, timeout_s)
        all_text = [f"[{title}] {text}" for title, _, text in rows]
        result_pod_text = ""
        for title, pid, text in rows:
            if (pid and "Result" in pid) or title == "Result":
                if not result_pod_text:
                    result_pod_text = text
                    break
        joined = "\n".join(all_text)
        primary = result_pod_text or joined
    except Exception as e:
        return {
            "answer_text": f"<ERROR: {type(e).__name__}: {e}>",
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
    parser.add_argument("--questions", type=Path,
                        default=Path(__file__).parent / "questions.json",
                        help="Path to questions corpus JSON.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--pace-per-day", type=int, default=90,
                        help="Max queries per day (free tier ~100/day)")
    parser.add_argument("--pause-s", type=float, default=2.0,
                        help="Pause between queries to be polite")
    args = parser.parse_args()

    # Auto-load API keys from the dev-root .env (or any ancestor)
    from sigma_ground.mcp.benchmark import load_env_from_dev_root
    load_env_from_dev_root(verbose=True)

    app_id = os.environ.get("WOLFRAM_ALPHA_APP_ID")
    if not app_id:
        print("ERROR: set WOLFRAM_ALPHA_APP_ID env var (or add to dev-root .env)",
              file=sys.stderr)
        return 1

    try:
        import xmltodict  # noqa: F401
    except ImportError:
        print("ERROR: pip install xmltodict (or wolframalpha which "
              "pulls it in)", file=sys.stderr)
        return 1

    with args.questions.open(encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for rec in json.load(f):
                # Skip errored records so they get retried
                ans = rec.get("answer_text", "") or ""
                if ans.startswith("<ERROR"):
                    continue
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
        result = run_question(app_id, q["question"])
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
