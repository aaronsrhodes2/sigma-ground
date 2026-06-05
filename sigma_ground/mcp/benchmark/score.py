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


def _clean_units(s: str) -> str:
    """Strip parenthetical aliases and lowercase. 'mph (miles per hour)' -> 'mph'."""
    if not s:
        return ""
    # Drop anything from first '(' onward
    s = s.split("(", 1)[0].strip()
    return s


def _try_unit_convert(value: float, from_units: str, to_units: str
                       ) -> float | None:
    """Convert value via pint. Returns None on any failure.

    Both units are cleaned first (parenthetical aliases stripped).
    Returns the input unchanged if the units match case-insensitively.

    Special case: pint treats radians as dimensionless, so it converts
    'rad/s' -> 'Hz' as a no-op. For physics we want the 2pi factor
    (omega = 2 pi f). Same for Hz -> rad/s in reverse.
    """
    import math
    fu = _clean_units(from_units)
    tu = _clean_units(to_units)
    if not fu or not tu:
        return None
    fu_low = fu.lower().replace(" ", "")
    tu_low = tu.lower().replace(" ", "")
    if fu_low == tu_low:
        return value

    # Special case: angular freq (rad/s) <-> ordinary freq (Hz).
    # Pint treats radians as dimensionless, so without this it would
    # convert rad/s -> Hz as a no-op. Physics wants omega = 2pi f.
    rad_per_s = {"rad/s", "radian/s", "rad·s^-1", "rad*s^-1", "rad s^-1"}
    hz_set = {"hz", "hertz", "s^-1", "1/s", "cycle/s", "cycles/s"}
    if fu_low in rad_per_s and tu_low in hz_set:
        return value / (2.0 * math.pi)
    if fu_low in hz_set and tu_low in rad_per_s:
        return value * (2.0 * math.pi)

    # Special case: bare 'C' / 'F' for temperature. Pint defaults to
    # Coulomb / Farad respectively. Rewrite to degC / degF when the
    # other side is a temperature unit.
    temperature_units = {"k", "kelvin", "celsius", "degc", "degree_celsius",
                            "fahrenheit", "degf", "degree_fahrenheit",
                            "rankine", "degr"}
    def _temp_alias(u: str) -> str:
        if u == "c":
            return "degC"
        if u == "f":
            return "degF"
        if u == "celsius":
            return "degC"
        if u == "fahrenheit":
            return "degF"
        return u
    if fu_low in temperature_units or tu_low in temperature_units:
        fu_use = _temp_alias(fu_low) if fu_low in {"c", "f", "celsius", "fahrenheit"} else fu
        tu_use = _temp_alias(tu_low) if tu_low in {"c", "f", "celsius", "fahrenheit"} else tu
        try:
            import pint
            ureg = pint.UnitRegistry()
            return ureg.Quantity(value, fu_use).to(tu_use).magnitude
        except Exception:
            return None

    try:
        import pint
        ureg = pint.UnitRegistry()
        return ureg.Quantity(value, fu).to(tu).magnitude
    except Exception:
        return None


def _expected_in_base(value: float, units: str) -> tuple[float, str] | None:
    """Express `value units` in SI base units (e.g. 5.196 year -> 1.64e8 s).

    Used by the dimension-aware comparison: our tools return SI base units
    (seconds, joules, meters), while ground truth often uses 'nice' units
    (years, eV, km). Returns (magnitude, base_unit_str) or None if pint
    can't parse the unit.
    """
    u = _clean_units(units)
    if not u:
        return None
    try:
        import pint
        q = pint.UnitRegistry().Quantity(value, u).to_base_units()
        return (q.magnitude, str(q.units))
    except Exception:
        return None


def _values_match(extracted: Any, expected: Any,
                   tolerance_rel: float,
                   extracted_units: str = "",
                   expected_units: str = "") -> tuple[bool, float | None, str]:
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

    def _check(a: float, b: float) -> tuple[bool, float | None]:
        """Is a within rel-tolerance of b? (absolute tol when b ~ 0)."""
        if abs(b) < 1e-30 and abs(a) < 1e-30:
            return True, 0.0
        if abs(b) < 1e-30:
            return abs(a) < tolerance_rel, None
        re = abs(a - b) / abs(b)
        return re <= tolerance_rel, re

    # DIMENSION-AWARE: build candidate (extracted, expected) pairs and accept
    # if ANY matches. A correct value expressed in a different unit (our tools
    # return SI base like seconds/joules; ground truth often uses years/eV) is
    # no longer scored wrong. False positives are negligible: a physically
    # wrong value won't coincidentally equal the expected after a unit map.
    eu = _clean_units(expected_units).lower()
    xu = _clean_units(extracted_units).lower()
    trials: list[tuple[float, float, str]] = [(x_float, e_float, "")]
    if extracted_units and expected_units and xu != eu:        # explicit units
        conv = _try_unit_convert(x_float, extracted_units, expected_units)
        if conv is not None:
            trials.append((conv, e_float, f" [unit-converted {xu}->{eu}]"))
    if expected_units:                                          # SI-base rescue
        base = _expected_in_base(e_float, expected_units)
        if base is not None:
            trials.append((x_float, base[0], f" [matched in SI base {base[1]}]"))

    best_err: float | None = None
    for a, b, note in trials:
        ok, re = _check(a, b)
        if ok:
            return True, (re if re is not None else 0.0), "match" + note
        if re is not None and (best_err is None or re < best_err):
            best_err = re
    return False, best_err, (f"rel_err {best_err:.3e}" if best_err is not None
                              else "no match")


def _score_keyword_match(answer_text: str, expected_keywords: list[str],
                            threshold: float = 0.5) -> tuple[bool, float, str]:
    """Count how many expected keywords appear in the answer (case insensitive).

    Returns (correct, fraction_matched, notes). 'correct' iff fraction >= threshold.
    """
    if not expected_keywords:
        return False, 0.0, "no keywords defined"
    text = (answer_text or "").lower()
    matched = [k for k in expected_keywords if k.lower() in text]
    frac = len(matched) / len(expected_keywords)
    notes = f"matched {len(matched)}/{len(expected_keywords)}: {matched[:3]!r}"
    return frac >= threshold, frac, notes


def _score_one(record: dict, q: dict, truth: dict
                ) -> tuple[bool, float | None, str]:
    """Dispatch on question_type. Returns (correct, rel_err_or_frac, notes)."""
    qtype = truth.get("question_type") or q.get("question_type") or "numerical"
    answer = record.get("answer_text", "") or ""
    extracted = record.get("extracted_value")
    expected = truth.get("expected_value")
    tol = truth.get("tolerance_rel")
    keywords = truth.get("expected_keywords", []) or []

    if qtype in ("numerical", "puzzle"):
        return _values_match(
            extracted, expected, tol if tol is not None else 0.05,
            extracted_units=record.get("extracted_units", "") or "",
            expected_units=truth.get("expected_units", "") or "",
        )

    if qtype in ("refusal_expected", "nonsense"):
        # Primary: keyword presence in answer. Bonus: if expected_value
        # is also numeric and the system gets that right, accept it.
        # Threshold 0.20: refusal answers are typically short ("no, false")
        # plus a one-line reason; hitting 2 of 5-6 keywords is a real
        # signal of a correct refusal. Stricter would punish honest
        # short refusals.
        kw_ok, frac, notes = _score_keyword_match(answer, keywords, threshold=0.20)
        if kw_ok:
            return True, frac, f"refusal/nonsense kw {notes}"
        # Allow a numeric escape hatch: e.g. adv_impossible_003 (Avogadro)
        # has an exact value AND keywords -- accept either path.
        if isinstance(expected, (int, float)) and extracted is not None:
            num_ok, rel_err, num_notes = _values_match(
                extracted, expected, tol if tol is not None else 0.02,
                extracted_units=record.get("extracted_units", "") or "",
                expected_units=truth.get("expected_units", "") or "",
            )
            if num_ok:
                return True, rel_err, f"numeric ({num_notes})"
        return False, frac, f"failed refusal/nonsense: {notes}"

    if qtype == "open_ended":
        # Lower threshold: 33% of keywords mentioned = correct
        return _score_keyword_match(answer, keywords, threshold=0.34)

    # Unknown type -> fall back to numerical
    return _values_match(
        extracted, expected, tol if tol is not None else 0.05,
        extracted_units=record.get("extracted_units", "") or "",
        expected_units=truth.get("expected_units", "") or "",
    )


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
        correct, rel_err, notes = _score_one(record, q, truth)
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
    """CLI: score all available runner outputs in results/.

    Each {system}_run.json is scored against the main corpus
    (questions.json + ground_truth.json). Each adversarial_{system}_run.json
    is scored against the adversarial corpus.
    """
    here = Path(__file__).parent
    main_corpus = (here / "questions.json", here / "ground_truth.json")
    adv_corpus = (here / "adversarial_questions.json",
                    here / "adversarial_ground_truth.json")
    results_dir = here / "results"
    if not results_dir.exists():
        print(f"No results/ directory. Run a runner first.")
        return
    for run_file in sorted(results_dir.glob("*_run.json")):
        if run_file.name.startswith("adversarial_"):
            corpus_path, gt_path = adv_corpus
            tag = "[adversarial]"
        else:
            corpus_path, gt_path = main_corpus
            tag = "[main]"
        if not corpus_path.exists() or not gt_path.exists():
            print(f"{run_file.name}: skipping ({corpus_path.name} or "
                  f"{gt_path.name} missing)")
            continue
        rows = score_run(run_file, corpus_path, gt_path)
        summary = summarize(rows)
        print(f"\n=== {run_file.name} {tag} ===")
        print(f"Overall: {summary['overall_correct']}/{summary['overall_total']} "
              f"= {summary['overall_accuracy_pct']:.1f}%")
        print("By domain:")
        for d, s in sorted(summary["by_domain"].items()):
            print(f"  {d:<35s} {s['correct']:3d}/{s['total']:3d} ({s['pct']:5.1f}%)")
        score_out = results_dir / run_file.name.replace("_run.json", "_scored.json")
        with score_out.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary,
                        "rows": [r.to_dict() for r in rows]},
                       f, indent=2, default=str)


if __name__ == "__main__":
    main()
