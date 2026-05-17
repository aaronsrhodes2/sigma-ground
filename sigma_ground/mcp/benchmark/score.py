"""Compare benchmark runner output against ground truth.

Each runner saves a JSON list of records:
  [
    {
      "id": "mech_intro_001",
      "system": "sigma_ground" | "wolfram" | "gemini",
      "answer_text": "<system's prose response>",
      "extracted_value": <float or None>,
      "extracted_units": "<str or None>",
      "tool_calls": [...],         (sigma_ground only)
      "elapsed_s": <float>,
      "tokens_in": <int|None>,
      "tokens_out": <int|None>,
      "cost_usd": <float|None>,
    },
    ...
  ]

score.py compares `extracted_value` against `ground_truth_value` within
`tolerance_rel`. Strings are compared lowercased. Lists (e.g. quadratic
solutions) are compared as sets.

For answers where the extracted_value is None (the LLM didn't give a
parseable number), we mark the score as "no_value_extracted" but
preserve the answer_text for manual review.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScoreRow:
    """One (system, question) scoring outcome."""
    id: str
    system: str
    domain: str
    correct: bool             # within tolerance OR exact string/list match
    extracted_value: Any      # what we parsed from the system's answer
    expected_value: Any
    rel_error: float | None   # |extracted - expected| / |expected|, or None
    tolerance_rel: float
    answer_text: str
    elapsed_s: float | None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_to_float(x: Any) -> float | None:
    if x is None: return None
    if isinstance(x, bool): return None  # bool is int subclass; reject
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # Try numeric parse
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _values_match(extracted: Any, expected: Any,
                   tolerance_rel: float) -> tuple[bool, float | None, str]:
    """Compare extracted and expected. Returns (correct, rel_error, notes).

    Numeric: relative-error tolerance.
    Strings: case-insensitive exact match.
    Lists/sets: set-equality at numeric tolerance.
    Inf: matches inf.
    """
    # Both None
    if expected is None and extracted is None:
        return True, 0.0, "both None"
    if extracted is None:
        return False, None, "no value extracted"

    # Inf handling
    import math
    if isinstance(expected, float) and math.isinf(expected):
        e_float = _coerce_to_float(extracted)
        if e_float is not None and math.isinf(e_float):
            return True, 0.0, "both inf"
        return False, None, "expected inf, extracted finite"

    # List/set comparison (e.g. roots of a quadratic)
    if isinstance(expected, list):
        if not isinstance(extracted, list):
            return False, None, "expected list, got scalar"
        if len(extracted) != len(expected):
            return False, None, f"len mismatch ({len(extracted)} vs {len(expected)})"
        # Sort numerically when possible
        try:
            e_sorted = sorted(_coerce_to_float(x) for x in extracted)
            x_sorted = sorted(_coerce_to_float(x) for x in expected)
            for a, b in zip(e_sorted, x_sorted):
                if a is None or b is None:
                    return False, None, "non-numeric list item"
                if abs(a - b) > tolerance_rel * max(abs(b), 1e-30):
                    return False, None, "list element mismatch"
            return True, 0.0, "list match"
        except (TypeError, ValueError):
            return False, None, "list comparison failed"

    # String comparison (case-insensitive, lstrip)
    if isinstance(expected, str):
        if not isinstance(extracted, str):
            return False, None, "expected string, got non-string"
        if expected.strip().lower() == extracted.strip().lower():
            return True, 0.0, "string match"
        return False, None, f"string mismatch"

    # Numeric comparison
    e_float = _coerce_to_float(expected)
    x_float = _coerce_to_float(extracted)
    if e_float is None or x_float is None:
        return False, None, "couldn't coerce to float"
    if abs(e_float) < 1e-30 and abs(x_float) < 1e-30:
        return True, 0.0, "both ~0"
    if abs(e_float) < 1e-30:
        # Expected is 0; check absolute deviation
        return abs(x_float) < tolerance_rel, None, "absolute tolerance (expected ~0)"
    rel_err = abs(x_float - e_float) / abs(e_float)
    return rel_err <= tolerance_rel, rel_err, f"rel_err {rel_err:.3e}"


def score_run(run_path: Path, corpus_path: Path,
                ground_truth_path: Path) -> list[ScoreRow]:
    """Score one runner's output against ground truth."""
    with run_path.open(encoding="utf-8") as f:
        run = json.load(f)
    with corpus_path.open(encoding="utf-8") as f:
        corpus = {q["id"]: q for q in json.load(f)}
    with ground_truth_path.open(encoding="utf-8") as f:
        gt = json.load(f)
    out: list[ScoreRow] = []
    for record in run:
        qid = record["id"]
        q = corpus.get(qid)
        truth = gt.get(qid)
        if q is None or truth is None:
            continue
        correct, rel_err, notes = _values_match(
            record.get("extracted_value"),
            truth["expected_value"],
            truth["tolerance_rel"],
        )
        out.append(ScoreRow(
            id=qid,
            system=record.get("system", "unknown"),
            domain=q["domain"],
            correct=correct,
            extracted_value=record.get("extracted_value"),
            expected_value=truth["expected_value"],
            rel_error=rel_err,
            tolerance_rel=truth["tolerance_rel"],
            answer_text=record.get("answer_text", ""),
            elapsed_s=record.get("elapsed_s"),
            notes=notes,
        ))
    return out


def summarize(rows: list[ScoreRow]) -> dict[str, Any]:
    """Aggregate accuracy by domain + overall."""
    by_domain: dict[str, dict[str, int]] = {}
    for r in rows:
        d = by_domain.setdefault(r.domain, {"correct": 0, "total": 0})
        d["total"] += 1
        if r.correct:
            d["correct"] += 1
    overall_correct = sum(r.correct for r in rows)
    overall_total = len(rows)
    return {
        "overall_accuracy_pct": (100.0 * overall_correct / overall_total
                                   if overall_total else 0.0),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "by_domain": {
            d: {
                "correct": s["correct"],
                "total":   s["total"],
                "pct":     100.0 * s["correct"] / s["total"] if s["total"] else 0.0,
            }
            for d, s in by_domain.items()
        },
    }


def main() -> None:
    """CLI: score all available runner outputs in results/."""
    here = Path(__file__).parent
    corpus_path = here / "questions.json"
    ground_truth_path = here / "ground_truth.json"
    results_dir = here / "results"
    if not results_dir.exists():
        print(f"No results/ directory. Run a runner first.")
        return
    for run_file in sorted(results_dir.glob("*_run.json")):
        rows = score_run(run_file, corpus_path, ground_truth_path)
        summary = summarize(rows)
        print(f"\n=== {run_file.name} ===")
        print(f"Overall: {summary['overall_correct']}/{summary['overall_total']} "
              f"= {summary['overall_accuracy_pct']:.1f}%")
        print("By domain:")
        for d, s in sorted(summary["by_domain"].items()):
            print(f"  {d:<35s} {s['correct']:3d}/{s['total']:3d} ({s['pct']:5.1f}%)")
        # Save detailed scoring
        score_out = results_dir / run_file.name.replace("_run.json", "_scored.json")
        with score_out.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary,
                        "rows": [r.to_dict() for r in rows]},
                       f, indent=2, default=str)


if __name__ == "__main__":
    main()
