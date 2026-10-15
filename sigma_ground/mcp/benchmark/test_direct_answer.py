"""Tests for tools/eval_direct_answer.py: the final answer of a free-text model is extracted and normalised before the shared scorer sees it."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
_spec = importlib.util.spec_from_file_location("eval_direct_answer", REPO / "tools" / "eval_direct_answer.py")
D = importlib.util.module_from_spec(_spec)
sys.modules["eval_direct_answer"] = D
_spec.loader.exec_module(D)


@pytest.mark.parametrize("latex,plain", [
    (r"\frac{1}{2}", "((1)/(2))"), (r"\frac{x^{2}}{3}", "((x^(2))/(3))"), (r"\sqrt{2}", "sqrt(2)"), (r"2\sqrt{x+1}", "2sqrt(x+1)"),
    (r"\dfrac{\pi}{4}", "((pi)/(4))"), (r"e^{-x^{2}}", "e^(-x^(2))"), (r"x \cdot y", "x * y"), (r"\infty", "oo"),
    (r"\left(x+1\right)^{2}", "(x+1)^(2)"), (r"\sin(x)\cos(x)", "sin(x)cos(x)"), (r"\frac{\sqrt{3}}{2}", "((sqrt(3))/(2))"),
])
def test_latex_becomes_plain_math(latex, plain):
    assert D.latex_to_plain(latex) == plain


@pytest.mark.parametrize("text,answer", [
    ("Let me compute.\nThe derivative is 2x.\nANSWER: 2*x", "2*x"),
    ("so\nANSWER: x^3/3 + C", "x^3/3"),
    ("**Answer:** 12", "12"),
    ("blah blah \\boxed{\\frac{1}{2}}", "((1)/(2))"),
    ("first \\boxed{3} then finally \\boxed{x^{2}+1}", "x^(2)+1"),
    ("<think>maybe ANSWER: 99</think>\nANSWER: 7", "7"),
    ("ANSWER: x = 5", "5"),
    ("ANSWER: [1, 2]", "[1, 2]"),
    ("ANSWER: f(x) = 3x^2", "3x^2"),
    ("ANSWER: 0.5.", "0.5"),
    (r"ANSWER: \boxed{-39}", "-39"),                                          # Phi-4-mini wraps its ANSWER line in \boxed
    (r"ANSWER: \boxed{\frac{z}{\sqrt{z^2+1}}}", "((z)/(sqrt(z^2+1)))"),
    (r"ANSWER: \boxed{y^2 + 2z}", "y^2 + 2z"),
])
def test_the_last_answer_is_extracted_and_cleaned(text, answer):
    assert D.extract_answer(text) == answer


@pytest.mark.parametrize("text", ["", "I do not know.", "<think>ANSWER: 3</think>", None])
def test_no_final_answer_is_none_not_a_guess(text):
    assert D.extract_answer(text) is None


def test_extracted_answers_score_with_the_shared_scorer():
    from eval_math_suite import same_answer
    score = lambda expected, text: same_answer(expected, D.canonical(D.extract_answer(text)))
    assert score("exp(x)*cos(x) + exp(x)*sin(x)", "ANSWER: e^x (cos x + sin x)")            # a student's writing, read by Mentat's parser
    assert score("x**3/3", "ANSWER: x^3/3 + C") and score("sqrt(pi)/2", r"\boxed{\frac{\sqrt{\pi}}{2}}")
    assert score("3", "ANSWER: 3.0") and score("0.5", r"\boxed{\frac{1}{2}}")
    assert score(["-2", "2"], "ANSWER: [2, -2]") and score("[[1, 2], [3, 4]]", "ANSWER: [[1,2],[3,4]]")
    assert not score("3", "ANSWER: 4") and not score("x**2", "ANSWER: x^3")


def test_the_stratified_sample_spreads_over_recipes_and_is_reproducible():
    items = [{"recipe": f"r{i % 5}", "id": k} for i in range(5) for k in range(i * 20, i * 20 + 20)]
    a, b = D.pick(items, 10, seed=1), D.pick(items, 10, seed=1)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert {x["recipe"] for x in a} == {"r0", "r1", "r2", "r3", "r4"} and len(a) == 10
    assert [x["id"] for x in D.pick(items, 10, seed=2)] != [x["id"] for x in a]


def _tiny_suite(tmp_path):
    import json
    items = [{"id": f"q{i}", "recipe": "r", "kind": "templated-holdout", "question": f"what is {i}+{i}?", "expected": {"result": str(2 * i), "tool": "t", "args": {}}}
             for i in range(1, 4)]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"items": items}), encoding="utf-8")
    return path


def test_retry_errors_keeps_answered_rows_and_asks_only_the_failed_ones(tmp_path, monkeypatch):
    import json
    asked = []

    def fake_ask(http, url, model, question, openai, num_predict, think, temperature=0.0, seed=7, num_ctx=16384):
        asked.append(question)
        return {"text": "ANSWER: 4", "tokens": 5, "seconds": 1.0}

    monkeypatch.setattr(D, "ask", fake_ask)
    base = {"recipe": "r", "kind": "templated-holdout", "tool_expected": "t", "tool_ok": False, "args_ok": False, "no_answer": False, "tokens": 5, "seconds": 1.0}
    rows = [dict(base, id="q1", question="what is 1+1?", expected="2", extracted="2", answer_ok=True, error=None),
            dict(base, id="q2", question="what is 2+2?", expected="4", extracted=None, answer_ok=False, no_answer=True, tokens=None, seconds=None,
                 error="ReadTimeout: timed out"),
            dict(base, id="q3", question="what is 3+3?", expected="6", extracted="9", answer_ok=False, error=None)]
    first, out = tmp_path / "first.json", tmp_path / "out.json"
    first.write_text(json.dumps({"stratified": 0, "seed": 7, "rows": rows}), encoding="utf-8")
    assert D.main(["--model", "m", "--suite", str(_tiny_suite(tmp_path)), "--retry-errors", str(first), "--out", str(out)]) == 0
    assert asked == ["what is 2+2?"]                                          # only the errored row went back to the model
    got = {r["id"]: r for r in json.loads(out.read_text(encoding="utf-8"))["rows"]}
    assert got["q1"]["answer_ok"] and "retried" not in got["q1"]              # kept as it was
    assert got["q3"]["extracted"] == "9" and not got["q3"]["answer_ok"]       # a wrong answer is still an answer: not asked again
    assert got["q2"]["answer_ok"] and got["q2"]["retried"] and got["q2"]["error"] is None


def test_retry_errors_refuses_a_different_sample(tmp_path):
    import json
    first = tmp_path / "first.json"
    first.write_text(json.dumps({"stratified": 100, "seed": 7, "rows": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        D.main(["--model", "m", "--suite", str(_tiny_suite(tmp_path)), "--stratified", "50", "--retry-errors", str(first)])
