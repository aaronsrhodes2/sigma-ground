"""Tests for tools/baselines_report.py: stored baseline rows are re-scored, hand-read verdicts are joined, errors are not counted as wrong."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
_spec = importlib.util.spec_from_file_location("baselines_report", REPO / "tools" / "baselines_report.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["baselines_report"] = B
_spec.loader.exec_module(B)


@pytest.mark.parametrize("stored,answer", [
    ("boxed(-39)", "-39"),
    ("boxed(((z)/(sqrt(z^2 + 1))))", "((z)/(sqrt(z^2 + 1)))"),
    ("boxed(y = Cexp(6x))", "Cexp(6x)"),                       # the same tidying as a fresh extraction: 'y = ' goes
    ("2*x", "2*x"),
    (None, None),
])
def test_old_rows_lose_the_boxed_wrapper(stored, answer):
    assert B.legacy_unwrap(stored) == answer


def _row(qid, expected, extracted, **kw):
    base = {"id": qid, "recipe": "r", "expected": expected, "extracted": extracted, "answer_ok": False, "error": None, "seconds": 10.0, "tokens": 100}
    return {**base, **kw}


def test_an_old_boxed_row_is_right_after_the_unwrap():
    right, ans = B.rescored(_row("a", "-39", "boxed(-39)"))
    assert right and ans == "-39"


def test_a_row_with_text_is_extracted_again_with_the_current_extractor():
    right, ans = B.rescored(_row("a", "3", "wrong old value", text="so the result is\nANSWER: \\boxed{3}"))
    assert right and ans == "3"


def test_summary_joins_verdicts_and_keeps_errors_out_of_the_misses():
    rows = [_row("a", "3", "3", answer_ok=True),                                # right
            _row("b", "x**2", "x^3"),                                           # wrong, no verdict yet
            _row("c", ["exp(2)"], "e^2"),                                       # a scalar for the one-element list: strictly a miss, read as equivalent
            _row("d", "1", None, error="ReadTimeout: timed out"),               # infrastructure, not a miss
            _row("e", "x", "x - x^3/6"),                                        # a convention
            _row("f", "7", None)]                                               # wrong, hand-read as no_answer
    adjud = {"M|c": {"verdict": "equivalent", "note": ""}, "M|e": {"verdict": "convention", "note": ""}, "M|f": {"verdict": "no_answer", "note": ""}}
    s = B.summarize_model("M", {"rows": rows}, adjud)
    assert (s["n"], s["errors"], s["strict"]) == (6, 1, 1)
    assert (s["equivalent"], s["lenient"], s["convention"], s["lenient_plus_convention"]) == (1, 2, 1, 3)
    assert s["other_misses"] == {"UNREAD": 1, "no_answer": 1}                  # 'b' has no verdict yet and is reported as such, never as right


def test_the_paired_sign_test_uses_only_answered_questions():
    rows = [_row("a", "3", "3", answer_ok=True), _row("b", "x", "x^2"), _row("c", "5", None, error="ReadTimeout")]
    s = B.summarize_model("M", {"rows": rows}, {})
    p = B.paired(s, {"a": True, "b": True, "c": True}, lambda r: r["right"])
    assert (p["n"], p["both"], p["only_router"], p["neither"]) == (2, 1, 1, 0)
