"""router_results: every non-numeric result shape the census found becomes an answer, and a
rejected call is reported by its cause.

The inputs are the result texts the real server returned for the benchmark's math questions
(tools/tool_census.py shapes) and the pydantic rejection texts of the failures in the
cascade-off runs, so these are not invented shapes.
"""
import json

import pytest

from sigma_ground.mcp.benchmark import router_results as RR


def result(value, units="", source="sympy"):
    return json.dumps({"value": value, "units": units, "source": source})


@pytest.mark.parametrize("value,expected", [
    (["-2", "2"], [-2.0, 2.0]),                       # solve_equation: x**2 - 4
    (["1", "2", "3"], [1.0, 2.0, 3.0]),               # the cubic
    ("cos(x)", "cos(x)"),                             # differentiate_expr: stays symbolic
    ("1", 1.0),                                       # simplify_expr: sin^2 + cos^2
    ("pi", 3.141592653589793),                        # integrate_expr over the real line
    ("1/3", 0.3333333333333333),                      # a definite integral
    ("transition", "transition"),                     # mond_regime_classifier: a label
    ("[[1, 2], [3, 4]]", "[[1, 2], [3, 4]]"),         # a matrix stays intact
    (["3", "1"], [3.0, 1.0]),                         # eigenvalues
    ("x**2/2 + y", "x**2/2 + y"),
])
def test_a_string_or_list_result_becomes_the_answer_the_cascade_would_give(value, expected):
    got = RR.present(result(value))
    assert got is not None and got[0] == pytest.approx(expected) if not isinstance(expected, str) else got[0] == expected


def test_the_conversion_is_the_cascades_own_not_a_copy():
    from sigma_ground.mcp.benchmark.math_classifier import _try_evaluate
    for v in (["-2", "2"], "cos(x)", "pi", "[[1, 2], [3, 4]]"):
        assert RR.present(result(v))[0] == _try_evaluate(v)


@pytest.mark.parametrize("text", [
    result(1.5),                                       # numbers are the harness's own path
    result({"mass_kg": 1.0}),                          # dicts go through the question-keyword picker
    result(None),
    result(True),
    result(""),
    result([]),
    json.dumps({"procedure": "black_hole_profile", "steps": []}),     # no `value` key
    "Error executing tool x: boom",
    "not json",
])
def test_everything_else_is_left_to_the_existing_readers(text):
    assert RR.present(text) is None


def test_units_come_through():
    assert RR.present(result("2", "m"))[1] == "m"


@pytest.mark.parametrize("text,expected", [
    ("Error executing tool critical_angle_for_tir: 1 validation error for critical_angle_for_tirArguments\n"
     "n_rare\n  Field required [type=missing, input_value={'n_dense': 1.333}, input_type=dict]",
     "critical_angle_for_tir: missing required parameter n_rare"),
    ("Error executing tool gravitational_redshift: 2 validation errors for gravitational_redshiftArguments\n"
     "radius_m\n  Field required [type=missing]\nmass_kg\n  Field required [type=missing]",
     "gravitational_redshift: missing required parameter radius_m, mass_kg"),
    ("Error executing tool integrate_expr: 1 validation error for integrate_exprArguments\n"
     "lower\n  Input should be a valid string [type=string_type, input_value=0.0, input_type=float]",
     "integrate_expr: wrong type for lower (expected a valid string)"),
    (json.dumps({"value": None, "units": "", "source": "invalid input", "notes": "Provide exactly two"}),
     "energy_power_time: invalid input (Provide exactly two)"),
])
def test_a_rejected_call_is_named_by_its_cause(text, expected):
    tool = expected.split(":")[0]
    assert RR.tool_error(tool, text) == expected


def test_a_good_result_is_not_an_error():
    assert RR.tool_error("t", result(1.0)) is None
    assert RR.tool_error("t", "not json") is None


# ── the harness switch: structured_results ──────────────────────────────────────────────

import asyncio
from types import SimpleNamespace

from sigma_ground.mcp.benchmark import run_sigma_ground as H
from sigma_ground.mcp.benchmark import tweaks as T


@pytest.fixture
def tweaks_reset():
    T.reset()
    yield
    T.reset()


class _Http:
    def __init__(self, plan):
        self.plan = plan

    async def post(self, url, json=None, **kw):
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"message": {"content": self.plan}})


class _Session:
    def __init__(self, text):
        self.text, self.called = text, []

    async def call_tool(self, name, args):
        self.called.append((name, dict(args)))
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def route(tool, args, tool_result, real_params):
    tools = [{"type": "function", "function": {"name": tool, "description": "", "parameters": {}}}]
    session = _Session(tool_result)
    plan = json.dumps({"tool": tool, "tool_args": args})
    rec = asyncio.run(H._route_once(session, _Http(plan), "http://x", "m", "q?", "sys", tools,
                                    {tool: set(real_params)}, 0.0, "free", "v3"))
    return rec


def test_without_the_switch_a_list_result_is_still_no_value_as_in_every_earlier_run(tweaks_reset):
    rec = route("solve_equation", {"equation": "x**2 - 4", "variable": "x"}, result(["-2", "2"]), {"equation", "variable"})
    assert rec["router_error"] == "solve_equation returned no value" and rec["extracted_value"] is None


def test_with_the_switch_a_list_result_is_the_answer(tweaks_reset):
    T.configure(on=["structured_results"])
    rec = route("solve_equation", {"equation": "x**2 - 4", "variable": "x"}, result(["-2", "2"]), {"equation", "variable"})
    assert "router_error" not in rec and rec["extracted_value"] == [-2.0, 2.0]
    rec = route("differentiate_expr", {"expression": "sin(x)", "variable": "x"}, result("cos(x)"), {"expression", "variable"})
    assert rec["extracted_value"] == "cos(x)" and rec["finalized_from_tool"] == "differentiate_expr"


def test_with_the_switch_a_rejected_call_says_why(tweaks_reset):
    T.configure(on=["structured_results"])
    rejected = ("Error executing tool critical_angle_for_tir: 1 validation error for critical_angle_for_tirArguments"
                + chr(10) + "n_rare" + chr(10) + "  Field required [type=missing]")
    rec = route("critical_angle_for_tir", {"n_dense": 1.333}, rejected, {"n_dense", "n_rare"})
    assert rec["router_error"] == "critical_angle_for_tir: missing required parameter n_rare"


def test_the_switch_is_off_in_every_profile_so_earlier_comparisons_stand(tweaks_reset):
    for profile in T.PROFILES:
        T.configure(profile)
        assert not T.on("structured_results")
    T.configure("model-only", on=["structured_results"])
    assert T.snapshot()["on"] == ["strict_params", "structured_results"]



def test_tool_error_reads_the_contracts_error_shape():
    import json
    from sigma_ground.mcp.benchmark import router_results as RR
    missing = json.dumps({"value": None, "error": {"code": "missing_argument", "param": "mass_kg", "message": "f needs 'mass_kg'"}})
    assert RR.tool_error("f", missing) == "f: missing required parameter mass_kg (f needs 'mass_kg')"
    dim = json.dumps({"value": None, "error": {"code": "wrong_dimension", "param": "mass_kg", "message": "'54 km/h' is a length / time"}})
    assert RR.tool_error("f", dim).startswith("f: wrong dimension for mass_kg")
    declined = json.dumps({"value": None, "error": {"code": "unknown_key", "message": "'steel' not in lookup table"}})
    assert RR.tool_error("f", declined) == "f: unknown_key ('steel' not in lookup table)"
    assert RR.tool_error("f", json.dumps({"value": 3.0, "error": None})) is None
