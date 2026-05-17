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


def run_question(model, question: str, timeout_s: float = 60.0) -> dict:
    """Ask Gemini one question, return record dict."""
    t0 = time.time()
    try:
        response = model.generate_content(question)
        text = response.text
        # token counts (best-effort; API may not include for all models)
        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", None) if usage else None
        tokens_out = getattr(usage, "candidates_token_count", None) if usage else None
    except Exception as e:
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
        "cost_usd": _estimate_cost(tokens_in, tokens_out),
    }


def _estimate_cost(tokens_in: int | None, tokens_out: int | None,
                     model: str = "gemini-2.5-pro") -> float | None:
    """Best-effort cost estimate using Gemini 2.5 Pro published rates."""
    if tokens_in is None or tokens_out is None:
        return None
    # 2.5 Pro: $1.25/M input, $10/M output (<=200K context)
    rate_in = 1.25e-6
    rate_out = 10.0e-6
    return tokens_in * rate_in + tokens_out * rate_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "gemini_run.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N questions (for testing).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions already in the output file.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY env var", file=sys.stderr)
        return 1

    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: pip install google-generativeai", file=sys.stderr)
        return 1

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model,
                                     system_instruction=_SYSTEM_PROMPT)

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

    out: list[dict] = list(existing.values())
    for i, q in enumerate(questions):
        if q["id"] in existing:
            print(f"[{i+1}/{len(questions)}] {q['id']}: skipped (resume)")
            continue
        print(f"[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        result = run_question(model, q["question"])
        rec = {"id": q["id"], "system": "gemini",
                "model": args.model, **result}
        out.append(rec)
        # Save after each question (resumable)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        # Light rate-limit guard
        time.sleep(0.2)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
