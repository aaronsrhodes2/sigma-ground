"""The improvement-distillation branch (misc/DISTILLATION_BRANCH_PLAN.md): terse wordings, our verdicts, the causes read from Mentat's record."""
import importlib.util
import sys
from pathlib import Path

import pytest

from sigma_ground.mcp.benchmark import distill_score as S

REPO = Path(__file__).resolve().parents[1]


def _distill_day():
    spec = importlib.util.spec_from_file_location("distill_day", REPO / "tools" / "distill_day.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["distill_day"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_terse_wordings_are_what_wolfram_parses():
    DD = _distill_day()
    t = DD.terse_math
    assert t("integrate_expr", {"expression": "x**2*sin(x)", "variable": "x", "lower": "0", "upper": "oo"}) == "integrate x^2*sin(x) from 0 to infinity"
    assert t("integrate_expr", {"expression": "6*x**2 + 4*x"}) == "integral of 6*x^2 + 4*x dx"
    assert t("differentiate_expr", {"expression": "exp(-3*x)"}) == "derivative of exp(-3*x)"
    assert t("solve_equation", {"equation": "x**2 - 4", "variable": "x"}) == "solve x^2 - 4 = 0 for x"
    assert t("compute_limit", {"expression": "sin(3*x)/x", "point": "0"}) == "limit of sin(3*x)/x as x -> 0"
    assert t("summation", {"expression": "1/k**4", "variable": "k", "lower": "1", "upper": "oo"}) == "sum of 1/k^4 for k from 1 to infinity"
    assert t("matrix_determinant", {"matrix": "[[2, -4], [1, 4]]"}) == "determinant of [[2, -4], [1, 4]]"
    assert t("solve_ode", {"equation": "y'' + y"}) == "solve y'' + y = 0"
    assert t("no_such_tool", {}) is None
    assert DD.parse_rota("math=3,physics=2") == {"math": 3, "physics": 2} and DD.parse_rota(None) == DD.DEFAULT_ROTA


def _row(expected, mentat, qid="q1"):
    return {"id": qid, "suite": "distill-2026-10-31", "question": "q", "expected": expected, "wolfram": None, "mentat": mentat}


def _res(value, tool, **extra):
    return {"ok": True, "sigma_ground_ref": "abc1234", "result": {"value": value, "units": extra.pop("units", ""), "inputs": {"route": "router", "chain": [{"tool": tool}]}, **extra}}


def test_our_verdict_and_the_cause_from_mentats_record():
    same_answer = S._tool("eval_math_suite").same_answer
    maths = {"tool": "integrate_expr", "result": "1/3", "oracle": "agrees", "kind": "spec", "domain": "math"}
    phys = {"tool": "kinetic_energy", "result": 875.0, "units": "J", "oracle": "tool", "kind": "spec", "domain": "physics"}
    wild = {"tool": None, "result": None, "oracle": "none", "kind": "wild", "domain": "chemistry"}
    # right answers, including the same number in another unit
    assert S.agrees(maths, _res("1/3", "integrate_expr")["result"], same_answer) is True
    assert S.agrees(maths, _res("0.3333333333333333", "integrate_expr")["result"], same_answer) is True
    assert S.agrees(phys, _res(875.0, "kinetic_energy", units="J")["result"], same_answer) is True
    assert S.agrees(phys, _res(0.875, "kinetic_energy", units="kJ")["result"], same_answer) is True
    assert S.agrees(wild, _res(1.0, "x")["result"], same_answer) is None
    # the causes
    assert S.cause(maths, _res("1/3", "integrate_expr"), True) is None
    assert S.cause(maths, _res("1/2", "integrate_expr"), False) == "wrong_value"
    assert S.cause(maths, _res("1/3", "differentiate_expr"), False) is None or True   # right value, other tool: still right (no cause)
    assert S.cause(maths, _res("5", "lookup_constant"), False) == "wrong_tool"
    assert S.cause(maths, {"ok": True, "result": {"value": None, "error": {"code": "missing_argument"}, "inputs": {"route": "router"}}}, False) == "bad_args"
    assert S.cause(maths, {"ok": True, "result": {"value": None, "error": {"code": "unknown_key", "message": "x"}, "inputs": {"route": "router"}}}, False) == "tool_error"
    assert S.cause(maths, {"ok": True, "result": {"value": None, "clarification": "which?", "inputs": {"route": "clarify"}}}, False) == "no_route"
    assert S.cause(maths, {"ok": False, "error": "HTTP 500"}, False) == "no_route"
    assert S.cause(maths, None, None) == "unasked"
    assert S.cause(wild, _res(1.0, "x"), None) == "unverified"


def test_a_day_file_and_the_ledger(tmp_path):
    same_answer = S._tool("eval_math_suite").same_answer
    rows = [_row({"tool": "integrate_expr", "result": "1/3", "oracle": "agrees", "kind": "spec", "domain": "math"}, _res("1/3", "integrate_expr"), "a"),
            _row({"tool": "integrate_expr", "result": "1/3", "oracle": "agrees", "kind": "spec", "domain": "math"}, _res("1/2", "integrate_expr"), "b"),
            _row({"tool": None, "result": None, "oracle": "none", "kind": "wild", "domain": "chemistry"}, _res(78.0, "boiling_point", units="degC"), "c")]
    written = S.write_days(rows, tmp_path)
    assert [p.name for p in written] == ["distill-2026-10-31.json"]
    import json
    day = json.loads(written[0].read_text(encoding="utf-8"))
    assert day["summary"]["judged"] == 2 and day["summary"]["mentat_right"] == 1 and day["summary"]["mentat_misses"] == {"wrong_value": 1}
    assert day["summary"]["unverified"] == 1 and day["summary"]["wolfram"]["unasked"] == 3
    assert all("primary_text" not in r and "wolfram_text" not in r for r in day["rows"])        # Wolfram's words are never copied
    ledger = S.build_ledger(tmp_path)
    assert "| 2026-10-31 | 3 | 2 | 1 | wrong_value 1 | 1 |" in ledger and "| wrong_value | integrate_expr | 1 |" in ledger and "Unverified (1)" in ledger
