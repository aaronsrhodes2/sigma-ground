"""Tests for tools/wa_trickle_score.py: Wolfram's pods -> an answer the router's scorer can read, and the bookkeeping around it.

The pod shapes are the real ones from the first 61 frozen replies (2026-10-14); the pod ids in them are not trustworthy ('Input' can hold the
Derivative pod), so selection goes by title.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
_spec = importlib.util.spec_from_file_location("wa_trickle_score", REPO / "tools" / "wa_trickle_score.py")
W = importlib.util.module_from_spec(_spec)
sys.modules["wa_trickle_score"] = W
_spec.loader.exec_module(W)


def pod(title, *plaintext, pid=None):
    return {"id": pid or title, "title": title, "plaintext": list(plaintext)}


# ── which pod is the answer ────────────────────────────────────────────────────────────────────────

def test_the_input_id_can_carry_the_answer_pod():
    pods = {"pods": [pod("Derivative", "d/dx(sin(x) cos(x)) = cos^2(x) - sin^2(x)", pid="Input"), pod("Alternate forms", "cos(2 x)")]}
    assert W.answer_pod(pods)["title"] == "Derivative"


def test_input_echo_pods_are_skipped():
    pods = {"pods": [pod("Input interpretation", "solve x^2 = 49 for x"), pod("Result", "x = ±7"), pod("Sum of roots", "0")]}
    assert W.answer_pod(pods)["title"] == "Result"


def test_a_result_pod_beats_a_restatement_pod_before_it():
    pods = {"pods": [pod("Input interpretation", "grad(x^2 y)"), pod("Del operator form", "del (x^2 y)"),
                     pod("Result in 2D Cartesian coordinates", "grad(x^2 y) = (2 x y, x^2)\n(x: first Cartesian coordinate | y: second Cartesian coordinate)")]}
    assert W.answer_pod(pods)["title"].startswith("Result in 2D")


def test_empty_pods_are_not_answers():
    assert W.answer_pod({"pods": []}) is None
    assert W.answer_pod({"pods": [pod("Plot", )]}) is None
    assert W.answer_pod(None) is None


# ── from a pod to the value ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("plaintext,answer", [
    (["d/dx(sin(x) cos(x)) = cos^2(x) - sin^2(x)"], "cos^2(x) - sin^2(x)"),
    (["integral_0^∞ e^(-x) dx = 1"], "1"),
    (["integral_1^3 x^2 dx = 26/3≈8.66667"], "26/3"),                                    # the decimal tail is dropped
    (["integral_0^π (cos^2(x) - sin^2(x))^2/π dx = 1/2 = 0.5"], "1/2"),
    (["integral(6 x^2 + 4 x) dx = 2 x^3 + 2 x^2 + constant"], "2 x^3 + 2 x^2"),            # "+ constant" is not part of the result
    (["sum_(i=1)^100 i = 5050"], "5050"),                                                  # the '=' inside the brackets is not the split
    (["lim_(x->0) sin(3 x)/x = 3"], "3"),
    (["1 - x^2/2 + x^4/24 + O(x^5)\n(Taylor series)"], "1 - x^2/2 + x^4/24"),              # remainder and the note line are dropped
    (["(x + 4) (x + 3)"], "(x + 4) (x + 3)"),
    (["y(x) = c_2 sin(2 x) + c_1 cos(2 x)"], "c_2 sin(2 x) + c_1 cos(2 x)"),               # a function on the left: one value, not a solution set
    (["x + 3", "x^2/(x - 3) - 9/(x - 3)"], "x + 3"),                                       # a second subpod that is not 'name = value' is an alternate form
    (["13"], "13"),
])
def test_scalar_answers(plaintext, answer):
    assert W.pod_answer(pod("Result", *plaintext)) == answer


@pytest.mark.parametrize("plaintext,answer", [
    (["x = ±7"], "[+7, -7]"),
    (["x = -4", "x = 2"], "[-4, 2]"),
    (["λ_1 = 3", "λ_2 = 1"], "[3, 1]"),
    (["x = 2 and y = 1"], "[2, 1]"),
    (["x = e^2"], "[e^2]"),                                                                # a single solution is still the solution set {e^2}
    (["grad(x^2 y) = (2 x y, x^2)\n(x: first Cartesian coordinate | y: second Cartesian coordinate)"], "(2 x y, x^2)"),
])
def test_solution_sets_and_vectors(plaintext, answer):
    assert W.pod_answer(pod("Result", *plaintext)) == answer


@pytest.mark.parametrize("text,matrix", [
    ("(-5 | 2\n3 | -1)", "[[-5, 2], [3, -1]]"),
    ("(sin(x) | 1\n2 | cos(x))", "[[sin(x), 1], [2, cos(x)]]"),                          # a cell that ends in ')' keeps it
    ("(3 | 4\n10 | 12)", "[[3, 4], [10, 12]]"),
])
def test_matrices(text, matrix):
    assert W.matrix_text(text) == matrix
    assert W.pod_answer(pod("Result", text)) == matrix


def test_a_plain_line_is_not_a_matrix():
    assert W.matrix_text("a | b") is None


# ── reading it ─────────────────────────────────────────────────────────────────────────────────────

def test_canonical_spelling_goes_through_mentats_parser():
    assert W.to_canonical("e^x (sin(x) + cos(x))") == "(sin(x) + cos(x))*exp(x)"
    assert W.to_canonical("[+7, -7]") == "[7, -7]"
    assert W.to_canonical("(2 x y, x^2)") == "[2*x*y, x**2]"


def test_text_that_is_not_a_result_is_unparsed_not_wrong():
    assert W.to_canonical("(x - 1) (x^2 + x + 1) is continuous on R") is None


# ── the bookkeeping ────────────────────────────────────────────────────────────────────────────────

def _q(qid, result, wolfram, kind="templated-holdout", question="q"):
    return {"id": qid, "suite": "math329", "question": question, "expected": {"result": result, "tool": "t", "kind": kind},
            "wolfram": wolfram, "mentat": {"result": {"value": None}}}


def _ok(*pods):
    return {"status": "ok", "pods": {"pods": list(pods)}}


def test_outcomes_and_counts():
    export = [
        _q("a", "x + 3", _ok(pod("Input interpretation", "simplify | ..."), pod("Result", "x + 3"))),                         # right
        _q("b", "x + 4", _ok(pod("Result", "x + 3"))),                                                                       # wrong
        _q("c", "x + 3", {"status": "no_result", "pods": {"pods": []}}),                                                      # no result
        _q("d", None, _ok(pod("Result", "5"))),                                                                              # no oracle -> unscored
        _q("e", "x", _ok(pod("Result", "x is continuous on R"))),                                                             # unparsed
    ]
    res = W.score_rows(export)
    assert {r["id"]: r["auto"] for r in res["rows"]} == {"a": "right", "b": "wrong", "c": "no_result", "d": "unscored", "e": "unparsed"}
    s = res["summary"]
    assert (s["asked"], s["answered"], s["no_result"], s["scorable"], s["unscored_no_oracle"]) == (5, 4, 1, 4, 1)
    assert s["wolfram_strict"] == 1


def test_only_equivalent_verdicts_raise_the_lenient_count():
    export = [_q("a", "x + 4", _ok(pod("Result", "x + 3"))), _q("b", "x + 4", _ok(pod("Result", "x + 3"))), _q("c", "x + 4", _ok(pod("Result", "x + 3")))]
    adjud = {"a": {"verdict": "equivalent", "note": ""}, "b": {"verdict": "misread", "note": ""}, "c": {"verdict": "convention", "note": ""}}
    s = W.score_rows(export, adjud=adjud)["summary"]
    assert (s["wolfram_strict"], s["wolfram_lenient"]) == (0, 1)


def test_router_pairing_by_id():
    export = [_q("a", "1", _ok(pod("Result", "1"))), _q("b", "1", _ok(pod("Result", "2"))), _q("c", "1", {"status": "no_result", "pods": {"pods": []}})]
    router = {"rows": [{"id": "a", "answer_ok": True, "tool_ok": True}, {"id": "b", "answer_ok": True, "tool_ok": True}, {"id": "c", "answer_ok": False, "tool_ok": True}]}
    p = W.score_rows(export, router=router)["summary"]["paired_strict"]
    assert (p["both"], p["only_wolfram"], p["only_router"], p["neither"]) == (1, 0, 1, 1)
    assert p["router_total"] == 2 and p["wolfram_total"] == 1
    assert W.score_rows(export, router=router)["summary"]["paired_strict_answered_only"]["n"] == 2        # 'c' got no pods and drops out


def test_wording_groups():
    export = [_q("a", "1", _ok(pod("Result", "1")), kind="wild"), _q("b", "1", _ok(pod("Result", "1")), question="expand (x+1)**2"),
              _q("c", "1", _ok(pod("Result", "1")), question="det [[1,2],[3,4]]"), _q("d", "1", _ok(pod("Result", "1")), question="expand (x+1)^2")]
    groups = {r["id"]: r["wording"] for r in W.score_rows(export)["rows"]}
    assert groups == {"a": "wild", "b": "templated, Python syntax", "c": "templated, Python syntax", "d": "templated, plain"}


def test_exact_sign_test():
    assert W.mcnemar_exact(0, 0) == 1.0
    assert W.mcnemar_exact(1, 5) == pytest.approx(14 / 64)
    assert W.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert W.mcnemar_exact(5, 5) == 1.0
