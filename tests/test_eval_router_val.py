"""tools/eval_router_val.py: how a predicted plan is compared with the gold plan.

The evaluator is the yardstick for the router fine-tune, so its comparison gets
positive and negative controls: identical plans pass, any change that matters
fails, and the two spellings of a plan (`args` / `tool_args`) are the same plan.
"""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "eval_router_val", Path(__file__).resolve().parents[1] / "tools" / "eval_router_val.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)

ONE = {"tool": "kinetic_energy", "tool_args": {"mass_kg": 1000.0, "velocity_m_s": 3074.0}}
TWO = {"steps": [{"tool": "circular_orbit_velocity",
                  "tool_args": {"central_mass_kg": 5.972e24, "radius_m": 42164000.0}},
                 {"tool": "kinetic_energy", "tool_args": {"mass_kg": 1000.0, "velocity_m_s": "$1"}}]}
DECLINE = {"decline": "not a question about a physical system"}
CLARIFY = {"tool": "request_clarification",
           "tool_args": {"variable": "mass_kg", "question": "What is the mass?"}}


@pytest.mark.parametrize("plan", [ONE, TWO, DECLINE, CLARIFY])
def test_a_plan_is_exact_against_itself(plan):
    assert E.compare(copy.deepcopy(plan), plan) == (True, True, "")


def test_the_two_spellings_are_the_same_plan():
    v1 = {"tool": "kinetic_energy", "args": {"mass_kg": 1000.0, "velocity_m_s": 3074.0}}
    assert E.compare(v1, ONE) == (True, True, "")
    assert E.compare(ONE, v1) == (True, True, "")


def test_numbers_within_a_millionth_pass_and_a_hundred_thousandth_does_not():
    near = copy.deepcopy(ONE)
    near["tool_args"]["velocity_m_s"] *= 1 + 5e-7
    assert E.compare(near, ONE)[1] is True
    far = copy.deepcopy(ONE)
    far["tool_args"]["velocity_m_s"] *= 1 + 1e-5
    tool_ok, exact, why = E.compare(far, ONE)
    assert tool_ok and not exact and "velocity_m_s" in why


def test_a_string_where_a_number_belongs_and_a_bool_are_not_numbers():
    s = copy.deepcopy(ONE)
    s["tool_args"]["mass_kg"] = "1000.0"
    assert E.compare(s, ONE)[1] is False
    gold = {"tool": "t", "tool_args": {"flag": True}}
    assert E.compare({"tool": "t", "tool_args": {"flag": 1}}, gold)[1] is False
    assert E.compare({"tool": "t", "tool_args": {"n": 2}}, {"tool": "t", "tool_args": {"n": 2.0}})[1]


def test_wrong_and_missing_keys_are_reported():
    extra = copy.deepcopy(ONE)
    extra["tool_args"]["g_m_s2"] = 9.8
    tool_ok, exact, why = E.compare(extra, ONE)
    assert tool_ok and not exact and why.startswith("arg keys") and "g_m_s2" in why
    missing = {"tool": "kinetic_energy", "tool_args": {"mass_kg": 1000.0}}
    assert "velocity_m_s" in E.compare(missing, ONE)[2]


def test_steps_are_compared_in_order_with_their_references():
    assert E.compare(TWO, ONE)[:2] == (False, False)             # one call is not two
    assert E.compare(ONE, TWO)[:2] == (False, False)
    swapped = {"steps": list(reversed(TWO["steps"]))}
    assert E.compare(swapped, TWO)[:2] == (False, False)
    ref = copy.deepcopy(TWO)
    ref["steps"][1]["tool_args"]["velocity_m_s"] = 3074.0         # a literal instead of "$1"
    tool_ok, exact, why = E.compare(ref, TWO)
    assert tool_ok and not exact and why.startswith("step 2 values")
    wrong_arg = copy.deepcopy(TWO)
    wrong_arg["steps"][0]["tool_args"]["radius_m"] = 1.0
    assert E.compare(wrong_arg, TWO)[2].startswith("step 1 values")


def test_decline_clarification_and_garbage():
    assert E.compare(DECLINE, ONE)[:2] == (False, False)
    assert E.compare(ONE, DECLINE)[:2] == (False, False)
    assert E.compare({"decline": "other reason"}, DECLINE) == (True, False, "decline reason differs")
    assert E.compare(CLARIFY, ONE)[:2] == (False, False)
    other = copy.deepcopy(CLARIFY)
    other["tool_args"]["variable"] = "velocity_m_s"
    assert E.compare(other, CLARIFY)[:2] == (True, False)
    for garbage in (None, {}, {"tool": 3}, {"steps": []}):
        assert E.compare(garbage, ONE) == (False, False, "unparseable or no tool")


def test_signature_names_the_calls_in_order():
    assert E.signature(TWO) == ("circular_orbit_velocity", "kinetic_energy")
    assert E.signature(ONE) == ("kinetic_energy",) and E.signature(DECLINE) == ("decline",)
    assert E.signature(None) is None


def test_a_clarification_is_judged_by_the_variable_not_the_wording():
    reworded = {"tool": "request_clarification",
                "tool_args": {"variable": "mass_kg", "question": "How heavy is it?"}}
    assert E.compare(reworded, CLARIFY) == (True, True, "")
    other = {"tool": "request_clarification", "tool_args": {"variable": "velocity_m_s", "question": "What is the mass?"}}
    tool_ok, exact, why = E.compare(other, CLARIFY)
    assert tool_ok and not exact and "variable" in why
