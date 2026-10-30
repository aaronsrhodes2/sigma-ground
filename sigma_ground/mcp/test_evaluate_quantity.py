"""evaluate_quantity: quantity arithmetic with units, units cancelling, and the guard against expressions that never return."""
from __future__ import annotations

import time

import pytest

from sigma_ground.mcp.tools import units as t_units


def ev(expression, to_units=None):
    return t_units.evaluate_quantity(expression, to_units)


@pytest.mark.parametrize("expression, value, units", [
    ("80 year * 70 / minute", 2945376000.0, ""),                    # 70 beats per minute for 80 years (the count word is left out)
    ("8848.86 m / (18 cm)", 49160.3333, ""),                        # how many 18 cm lengths
    ("13.787e9 year / nanosecond", 4.350846312e26, ""),             # nanoseconds in the age of the universe
    ("100 km / (60 km/h + 40 km/h)", 1.0, "hour"),                  # two trains closing
    ("3 km/h * 2 h", 6.0, "kilometer"),
    ("2 m * 3 m", 6.0, "meter ** 2"),
    ("12 / 4", 3.0, ""),                                            # no unit at all
])
def test_the_arithmetic_reduces_to_one_unit_per_dimension(expression, value, units):
    r = ev(expression)
    assert r.value == pytest.approx(value, rel=1e-6)
    assert r.units == units
    assert r.provenance_tag == "DERIVED"
    assert r.interpreted_as and expression in r.interpreted_as


def test_to_units_names_the_unit_asked_for():
    r = ev("3 km/h * 2 h", "m")
    assert r.value == pytest.approx(6000.0)
    assert r.units == "meter"


def test_a_count_word_is_not_a_unit_and_the_note_says_so():
    r = ev("80 year * 70 beats / minute")
    assert r.value is None
    assert "count word" in r.notes and "leave it out" in r.notes


def test_unlike_units_do_not_add():
    r = ev("3 m + 2 s")
    assert r.value is None and "dimension" in r.source


def test_asking_for_a_unit_of_another_dimension_is_an_error_not_a_number():
    r = ev("3 m", "second")
    assert r.value is None and "dimension" in r.source


@pytest.mark.parametrize("expression", [
    "9**9**9",                              # a tower never returns
    "2^3^4",
    "10**100000",                           # an exponent past the cap
    "9 ** x",                               # an exponent that is not a plain number
    "__import__('os')",                     # not an expression
    "3 m; 4 m",
    "1 " + "+ 1 " * 100,                    # longer than the cap
    "2**3 * 2**3 * 2**3 * 2**3",            # more than three powers
])
def test_expressions_that_could_hang_or_are_not_arithmetic_are_refused_at_once(expression):
    started = time.monotonic()
    r = ev(expression)
    assert time.monotonic() - started < 5
    assert r.value is None
    assert r.notes.startswith("refused:")


def test_a_small_power_is_fine():
    assert ev("(3 m)^2").value == pytest.approx(9.0)
    assert ev("2**10").value == pytest.approx(1024.0)


def test_division_by_zero_is_a_miss_not_a_crash():
    r = ev("1 m / 0")
    assert r.value is None


def test_the_tool_is_in_the_manifest():
    from sigma_ground.mcp import manifest

    names = {t["name"] for t in manifest.get_manifest().value["primary"]}
    assert "evaluate_quantity" in names
