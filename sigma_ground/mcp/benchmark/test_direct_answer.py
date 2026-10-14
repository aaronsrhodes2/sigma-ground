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
