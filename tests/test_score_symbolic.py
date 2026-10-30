"""The physics scorer reads a symbolic answer that is a number, and a list of roots (misc/STATIC_INTENTS_PLAN.md group B).

math_001 ([-2, 2]), math_002 (1/3), math_004 ([1, 2, 3]) and math_005 (pi) were answered right on the live /ask and scored as
misses: `_coerce_to_float` read only float("..."), and the replay runner dropped list values. The tool contract now also puts the
decimal beside a symbolic value (`value_float`), for the reader, not the scorer.
"""
import pytest

from sigma_ground.mcp.benchmark.score import _coerce_to_float, _values_match
from sigma_ground.mcp.contract import _symbolic_number


def test_a_symbolic_number_is_read_as_its_decimal():
    assert _coerce_to_float("1/3") == pytest.approx(1 / 3)
    assert _coerce_to_float("pi") == pytest.approx(3.141592653589793)
    assert _coerce_to_float("-1 + E") == pytest.approx(1.718281828459045)
    assert _coerce_to_float("sqrt(2)/2") == pytest.approx(0.7071067811865476)
    assert _coerce_to_float("26/3") == pytest.approx(26 / 3)


def test_an_expression_with_a_free_symbol_is_not_a_number():
    assert _coerce_to_float("x**2") is None
    assert _coerce_to_float("2*x**3 + 2*x**2") is None
    assert _coerce_to_float("Eq(y(x), C1*exp(-4*x))") is None
    assert _coerce_to_float("not a number at all") is None
    assert _coerce_to_float(True) is None


def test_the_four_physics_rows_now_match_their_ground_truth():
    assert _values_match("1/3", 0.3333333333333333, 0.001)[0]
    assert _values_match("pi", 3.14159265358979, 0.001)[0]
    assert _values_match(["-2", "2"], [-2, 2], 0.001)[0]
    assert _values_match(["1", "2", "3"], [1, 2, 3], 0.001)[0]
    # and a wrong symbolic answer is still wrong
    assert not _values_match("1/2", 0.3333333333333333, 0.001)[0]
    assert not _values_match(["-2", "3"], [-2, 2], 0.001)[0]


def test_the_contract_adds_the_decimal_beside_a_symbolic_value():
    assert _symbolic_number("1/3") == pytest.approx(1 / 3)
    assert _symbolic_number("pi") == pytest.approx(3.141592653589793)
    assert _symbolic_number("x**2") is None and _symbolic_number("") is None and _symbolic_number("oo") is None
    from sigma_ground.mcp.contract import _finish
    r = _finish({"value": "1/3", "units": "", "source": "sympy.integrate"}, {})
    assert r["value"] == "1/3" and r["value_float"] == pytest.approx(1 / 3)
    r = _finish({"value": "2*x**3 + 2*x**2", "units": "", "source": "sympy.integrate"}, {})
    assert "value_float" not in r
    r = _finish({"value": 42.0, "units": "m", "source": "s"}, {})
    assert "value_float" not in r
