"""Offline tests for the router math generator and the held-out math evaluator (no Mentat server, no GPU).

What they protect:
  * the renderers never produce an undefined function name (sympy would "compute" it) and rename variables
    as whole words (the x in exp(x) is not the x in "x squared");
  * every generator yields well-formed problems in both the training and the held-out wording, and the two
    wordings are different pools;
  * the independent oracles accept a right answer and reject a wrong one;
  * the evaluator's notion of "same argument" and "same answer" is about meaning, not spelling.
"""
from __future__ import annotations

import importlib.util
import random
import re
import sys
from pathlib import Path

import pytest

from sigma_ground.mcp.benchmark import router_math as M

REPO = Path(__file__).resolve().parents[3]


def _load_eval():
    spec = importlib.util.spec_from_file_location("eval_math_suite", REPO / "tools" / "eval_math_suite.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_math_suite"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── renderers ─────────────────────────────────────────────────────────────

def test_rename_is_whole_word():
    assert M.rename("exp(x) + x**2", "x", "t") == "exp(t) + t**2"
    assert M.rename("the next x", "x", "y") == "the next y"
    assert M.rename("", "x", "t") == ""


def test_unknown_functions_flags_only_undefined_names():
    assert M.unknown_functions("sin(x) + exp(-x**2) + sqrt(x)") == set()
    assert M.unknown_functions("etp(x) + sin(x)") == {"etp"}
    assert M.unknown_functions("f(x) * g(y)") == {"f", "g"}


@pytest.mark.parametrize("name", sorted(M.GENERATORS))
def test_generators_make_well_formed_problems(name):
    gen, _weight = M.GENERATORS[name]
    for holdout in (False, True):
        for seed in range(25):
            p = gen(random.Random(seed), holdout)
            # p_sum picks finite or infinite itself, so both sum names share one generator
            assert p.recipe == name or {p.recipe, name} <= {"math_sum_finite", "math_sum_infinite"}
            assert p.tool and p.question.strip(), (name, seed)
            assert M.unknown_functions(" ".join(str(v) for v in p.args.values())) == set(), (name, seed, p.args)
            for k, v in p.args.items():
                assert isinstance(v, (str, int, float)) and not isinstance(v, bool), (name, k, v)
                assert v != "", (name, k)
            assert not re.search(r"\{[a-z_]+\}", p.question), ("unfilled template slot", p.question)       # L{...} is Laplace notation, not a slot


@pytest.mark.parametrize("name", ["math_diff", "math_int_definite", "math_limit", "math_series", "math_det", "math_solve"])
def test_train_and_holdout_use_different_wordings(name):
    gen, _ = M.GENERATORS[name]
    def pool(holdout):
        shapes = set()
        for seed in range(80):
            q = gen(random.Random(seed), holdout).question
            shapes.add(" ".join(w for w in q.lower().split() if w.isalpha()))
        return shapes
    train, held = pool(False), pool(True)
    assert train and held
    assert len(held - train) > len(held) // 2, f"{name}: the held-out wordings overlap the training ones too much"


# ── oracles ───────────────────────────────────────────────────────────────

def test_oracle_derivative_accepts_right_rejects_wrong():
    spec = {"kind": "diff", "var": "x", "expr": "x**3 + sin(x)", "order": 1}
    assert M.check(spec, "3*x**2 + cos(x)") is True
    assert M.check(spec, "3*x**2 - cos(x)") is False


def test_oracle_antiderivative_checks_by_differentiating():
    spec = {"kind": "int", "var": "x", "expr": "2*x + cos(x)"}
    assert M.check(spec, "x**2 + sin(x)") is True
    assert M.check(spec, "x**2 + sin(x) + 7") is True          # a constant is free
    assert M.check(spec, "x**2 - sin(x)") is False


# ── the evaluator ─────────────────────────────────────────────────────────

E = _load_eval()


def test_same_arg_is_about_meaning():
    assert E.same_arg("expression", "x^2 + 1", "x**2 + 1")
    assert E.same_arg("upper", "pi / 2", "pi/2")
    assert E.same_arg("order", 2, 2.0)
    assert E.same_arg("matrix", "[[1, 2], [3, 4]]", "[[1,2],[3,4]]")
    assert not E.same_arg("expression", "x^2 + 1", "x**2 + 2")
    assert not E.same_arg("matrix", "[[1, 2], [3, 4]]", "[[1, 2], [4, 3]]")
    assert not E.same_arg("order", 2, "three")


def test_same_answer_numbers_lists_and_expressions():
    assert E.same_answer("3.0", 3.0000001)
    assert not E.same_answer("3.0", 3.1)
    assert E.same_answer("[1, 2]", "[2, 1]")                    # a set of roots
    assert not E.same_answer("[1, 2]", "[1, 3]")
    assert E.same_answer("x**2 + 1", "1 + x**2")
    assert not E.same_answer("x**2 + 1", "x**2 + 2")
    assert not E.same_answer("3.0", None)


def test_summarize_counts_by_kind_and_tool():
    rows = [
        {"kind": "holdout", "tool_expected": "differentiate_expr", "tool_ok": True, "args_ok": True, "answer_ok": True, "error": None},
        {"kind": "holdout", "tool_expected": "differentiate_expr", "tool_ok": False, "args_ok": False, "answer_ok": False, "error": "RouterError: x"},
        {"kind": "wild", "tool_expected": "integrate_expr", "tool_ok": True, "args_ok": False, "answer_ok": False, "error": None},
    ]
    s = E.summarize({"model": "m", "rows": rows})
    assert (s["n"], s["tool"], s["args"], s["answer"]) == (3, "2/3", "1/3", "1/3")
    assert s["by_kind"]["holdout"]["answer"] == "1/2"
    assert s["by_tool"]["integrate_expr"]["args"] == "0/1"
    assert s["errors"] == {"RouterError": 1}
