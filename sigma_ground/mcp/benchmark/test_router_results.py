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
