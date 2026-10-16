"""Tests for mcp/unit_display.py: an answer comes back in the measurement system the question used, with the SI value in parentheses.

The Captain's rule (2026-10-15): "Return the answer to match the measurement system used, but put the scientific/metric standard in (parentheses)
like '1 yard (0.914m)'." `value` stays SI; only the extra `display` field changes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from sigma_ground.mcp import unit_display as U
from sigma_ground.mcp.provenance import ToolResult

pytest.importorskip("pint")
REPO = Path(__file__).resolve().parents[2]


# ── the rule ───────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,si,user,expected", [
    (0.9144, "m", ["yard"], "1 yard (0.9144 m)"),                                   # the Captain's own example
    (3 * 0.9144, "m", ["yard"], "3 yards (2.743 m)"),                                # plural
    (0.3048, "m", ["foot"], "1 foot (0.3048 m)"),
    (1.5240, "m", ["foot"], "5 feet (1.524 m)"),
    (26.8224, "m/s", ["mile_per_hour"], "60 mph (26.82 m/s)"),
    (0.05, "m", ["centimeter"], "5 cm (0.05 m)"),                                    # metric but not SI: shown in the unit used
    (10.0, "m/s", ["kilometer / hour"], "36 km/h (10 m/s)"),
    (294.261, "K", ["degree_Fahrenheit"], "70 °F (294.3 K)"),
    (0.45359237, "kg", ["pound"], "1 pound (0.4536 kg)"),
    (7200.0, "s", ["hour"], "2 h (7200 s)"),
])
def test_the_answer_is_in_the_unit_the_caller_used(value, si, user, expected):
    assert U.display_answer(value, si, user) == expected


@pytest.mark.parametrize("value,si,user,expected", [
    (14.0, "m/s", ["foot"], "31.32 mph (14 m/s)"),                                    # feet in, a speed out: the customary unit of a speed
    (101325.0, "Pa", ["foot"], "14.7 psi (101325 Pa)"),
    (100.0, "J", ["pound"], "73.76 ft·lbf (100 J)"),
    (1000.0, "W", ["foot"], "1.341 hp (1000 W)"),
    (8.0, "m/s^2", ["foot"], "26.25 ft/s² (8 m/s^2)"),
    (2.0, "m^2", ["foot"], "21.53 ft² (2 m^2)"),
    (4.0, "kg", ["mile"], "8.818 pounds (4 kg)"),
    (1609.344, "m", ["mile_per_hour"], "1 mile (1609 m)"),
])
def test_another_kind_of_quantity_gets_the_customary_unit_when_the_question_was_customary(value, si, user, expected):
    assert U.display_answer(value, si, user) == expected


def test_the_unit_with_the_most_natural_number_is_chosen():
    assert U.display_answer(0.0254, "m", ["pound"]) == "1 inch (0.0254 m)"
    assert U.display_answer(1609.344, "m", ["pound"]) == "1 mile (1609 m)"
    assert U.display_answer(0.9144, "m", ["pound"]).startswith("1 yard")
    assert U.display_answer(0.003, "m", ["pound"]) == "0.1181 inches (0.003 m)"             # nothing reaches 1: the smallest unit


@pytest.mark.parametrize("value,si,user", [
    (5.0, "m", ["meter"]),                  # SI in, nothing to add
    (5.0, "m", []),                         # no units written at all
    (5.0, "m", [""]),
    (3.0, "s", ["foot", "second"]),         # the answer's own kind is already SI
    (3.0, "s", ["foot"]),                   # time is not converted
    (3.0, "", ["foot"]),                    # dimensionless
    (3.0, "kg", ["centimeter"]),            # a metric question that never used this kind of quantity: stay SI
    ("5 m", "m", ["foot"]),                 # not a number
    (float("nan"), "m", ["foot"]),
    (float("inf"), "m", ["foot"]),
    (True, "m", ["foot"]),
    (50.0, "m", ["no_such_unit"]),
])
def test_nothing_is_added_when_there_is_nothing_to_say(value, si, user):
    assert U.display_answer(value, si, user) is None


def test_the_number_formatting():
    assert U._num(0) == "0" and U._num(1.0) == "1" and U._num(0.9144) == "0.9144"
    assert U._num(101325.0) == "101325" and U._num(1609.344) == "1609"
    assert U._num(1.5e9) == "1.5×10^9" and U._num(0.0004) == "0.0004" and U._num(0.00004) == "4×10^-5"
    assert U._num(-16.404) == "-16.4"


def test_unit_classification():
    reg = U._registry()
    cls = lambda t: U.classify(reg, reg.Unit(t))
    assert [cls(t) for t in ("yard", "foot", "mph", "degF", "psi", "lbf", "pound", "gallon", "knot")] == ["customary"] * 9
    assert [cls(t) for t in ("centimeter", "gram", "km/h", "degC", "hour", "liter", "bar")] == ["metric"] * 7
    assert [cls(t) for t in ("meter", "kilogram", "newton", "m/s", "joule", "kelvin")] == ["si"] * 6


# ── in the tool results ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def census():
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("mentat_contract_census", REPO / "tools" / "mentat_contract_census.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mentat_contract_census"] = mod
    spec.loader.exec_module(mod)
    return mod, mod.build_server()


def _call(census, name, args):
    mod, server = census
    r = mod.call(server, name, args)
    assert r["status"] == "ok", r
    return r["result"]


def test_a_tool_answers_in_the_callers_units_and_keeps_value_in_si(census):
    r = _call(census, "free_fall_velocity", {"height_m": "30 feet"})
    plain = _call(census, "free_fall_velocity", {"height_m": 9.144})
    assert r["value"] == pytest.approx(plain["value"], rel=1e-12) and r["units"] == "m/s"        # the number a machine reads is untouched
    assert r["display"] == "29.96 mph (13.39 m/s)"
    assert "display" not in plain or plain["display"] is None                                 # an SI question gets an SI-only answer


def test_mixed_units_in_one_call(census):
    r = _call(census, "kinetic_energy", {"mass_kg": "145 g", "velocity_m_s": "90 mph"})
    assert r["display"] == "86.56 ft·lbf (117.4 J)" and r["value"] == pytest.approx(117.3588362496)


def test_a_unit_that_matches_the_answers_own_kind_is_used_even_in_a_metric_question(census):
    r = _call(census, "free_fall_time", {"height_m": "10 yards"})
    assert r["units"] == "s" and not r.get("display")                                         # a time is not converted


def test_toolresult_shows_the_display_first_when_formatted_for_the_model():
    r = ToolResult(value=13.39, units="m/s", source="x", display="29.96 mph (13.39 m/s)")
    assert r.format_for_llm().split(" | ")[1] == "answer: 29.96 mph (13.39 m/s)"
    assert "answer:" not in ToolResult(value=1.0, units="m").format_for_llm()
    assert ToolResult(value=1.0).to_dict()["display"] is None
