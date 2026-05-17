"""Run the 150-question benchmark against Gemini via the Google API.

Requires:
    pip install google-generativeai
    export GEMINI_API_KEY=<your key from https://aistudio.google.com/>

Usage:
    python -m sigma_ground.mcp.benchmark.run_gemini \
        --model gemini-2.5-pro \
        --output sigma_ground/mcp/benchmark/results/gemini_run.json

This is the SIMPLEST runner: no tool calling, no MCP server, just
ask each question and capture the answer. Gemini is the "bare LLM
baseline" against which both Wolfram Alpha and sigma-ground+Qwen
are compared.
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


_SYSTEM_PROMPT = """\
You are a physics assistant. Answer the user's physics question in
plain English. Give a clear, exact numerical answer in SI units (or the
units the user implies). Begin your final answer with the phrase
"ANSWER:" followed by the numeric value and units on one line, e.g.:

    ANSWER: 1.43 s

After the ANSWER line, you may briefly explain reasoning if useful.

If a question is conceptual (no numeric answer), write:
    ANSWER: <one-word or short phrase>
"""


_VALUE_RE = re.compile(
    r"ANSWER:\s*([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)"
    r"(?:\s*([^\n]*))?",
    re.IGNORECASE,
)


def _extract_value(answer: str) -> tuple[Any, str]:
    """Pull numeric value + units from the ANSWER: line."""
    m = _VALUE_RE.search(answer)
    if not m:
        # Try string-only answer: ANSWER: <word>
        m2 = re.search(r"ANSWER:\s*([^\n]+)", answer, re.IGNORECASE)
        if m2:
            tail = m2.group(1).strip()
            return tail, ""
        return None, ""
    val_str = m.group(1)
    units = (m.group(2) or "").strip()
    try:
        val = float(val_str)
    except ValueError:
        val = val_str
    return val, units


def _parse_retry_delay(err_str: str, default: float = 30.0) -> float:
    """Pull the 'retry in Xs' hint from a Gemini 429 error string."""
    m = re.search(r"retry in ([\d.]+)s", err_str)
    if m:
        try:
            return float(m.group(1)) + 1.0  # +1s safety margin
        except ValueError:
            pass
    return default


def run_question(model, question: str, model_name: str = "",
                   timeout_s: float = 60.0,
                   max_429_retries: int = 8) -> dict:
    """Ask Gemini one question, return record dict.

    Retries on 429 (free-tier rate limit) up to max_429_retries times,
    parsing the suggested retry-delay from the error if present.
    """
    t0 = time.time()
    attempt = 0
    while True:
        try:
            response = model.generate_content(question)
            text = response.text
            usage = getattr(response, "usage_metadata", None)
            tokens_in = getattr(usage, "prompt_token_count", None) if usage else None
            tokens_out = getattr(usage, "candidates_token_count", None) if usage else None
            break
        except Exception as e:
            err_str = str(e)
            is_429 = "429" in err_str or "quota" in err_str.lower() \
                       or "rate" in err_str.lower()
            if is_429 and attempt < max_429_retries:
                delay = _parse_retry_delay(err_str, default=15.0 + attempt * 5)
                print(f"    [429 retry {attempt+1}/{max_429_retries}] "
                      f"sleeping {delay:.1f}s...", flush=True)
                time.sleep(delay)
                attempt += 1
                continue
            return {
                "answer_text": f"<ERROR: {e}>",
                "extracted_value": None,
                "extracted_units": "",
                "elapsed_s": time.time() - t0,
                "tokens_in": None, "tokens_out": None, "cost_usd": None,
            }
    elapsed = time.time() - t0
    val, units = _extract_value(text)
    return {
        "answer_text": text,
        "extracted_value": val,
        "extracted_units": units,
        "elapsed_s": elapsed,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": _estimate_cost(tokens_in, tokens_out, model_name),
    }


def _estimate_cost(tokens_in: int | None, tokens_out: int | None,
                     model: str = "gemini-2.5-flash") -> float | None:
    """Best-effort cost estimate using published Gemini pricing.

    Published rates as of 2026-05:
      gemini-2.5-pro        $1.25/M in, $10.00/M out (<=200k context)
      gemini-2.5-flash      $0.30/M in,  $2.50/M out
      gemini-2.5-flash-lite $0.10/M in,  $0.40/M out
    """
    if tokens_in is None or tokens_out is None:
        return None
    rates = {
        "gemini-2.5-pro":        (1.25e-6, 10.00e-6),
        "gemini-2.5-flash":      (0.30e-6,  2.50e-6),
        "gemini-2.5-flash-lite": (0.10e-6,  0.40e-6),
    }
    rate_in, rate_out = rates.get(model, rates["gemini-2.5-flash"])
    return tokens_in * rate_in + tokens_out * rate_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model. 'flash' is the default (free tier-"
                              "friendly). 'pro' needs paid billing; 'flash-lite' "
                              "is cheapest.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "gemini_run.json")
    parser.add_argument("--questions", type=Path,
                        default=Path(__file__).parent / "questions.json",
                        help="Path to questions corpus JSON. Defaults to the "
                              "main 150-question physics corpus. Pass "
                              "adversarial_questions.json for the trick set.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N questions (for testing).")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip questions already in the output file.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--pause-s", type=float, default=13.0,
                        help="Sleep between questions. Default 13s respects "
                              "the free-tier rate limit (5 req/min). On a "
                              "paid plan you can drop this to 0.2.")
    args = parser.parse_args()

    # Auto-load API keys from the dev-root .env (or any ancestor)
    from sigma_ground.mcp.benchmark import load_env_from_dev_root
    load_env_from_dev_root(verbose=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY env var (or add to dev-root .env)",
              file=sys.stderr)
        return 1

    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: pip install google-generativeai", file=sys.stderr)
        return 1

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model,
                                     system_instruction=_SYSTEM_PROMPT)

    with args.questions.open(encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for rec in json.load(f):
                # Skip records that errored or used a different model -- re-run them
                ans = rec.get("answer_text", "") or ""
                if ans.startswith("<ERROR"):
                    continue
                if rec.get("model") and rec["model"] != args.model:
                    continue
                existing[rec["id"]] = rec

    out: list[dict] = list(existing.values())
    for i, q in enumerate(questions):
        if q["id"] in existing:
            print(f"[{i+1}/{len(questions)}] {q['id']}: skipped (resume)")
            continue
        print(f"[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        result = run_question(model, q["question"], model_name=args.model)
        rec = {"id": q["id"], "system": "gemini",
                "model": args.model, **result}
        out.append(rec)
        # Save after each question (resumable)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        time.sleep(args.pause_s)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
