"""The v2 plan format: reading both spellings, references, the schema's key order."""
import json
import re

import pytest

from sigma_ground.mcp.benchmark import router_plan as P


def test_both_spellings_read_the_same():
    v1 = {"tool": "kinetic_energy", "args": {"mass_kg": 2.0, "velocity_m_s": 3.0}}
    v2 = {"tool": "kinetic_energy", "tool_args": {"mass_kg": 2.0, "velocity_m_s": 3.0}}
    assert P.canonical(v1) == P.canonical(v2)
    assert P.canonical(v2) == {"kind": "steps", "steps": [
        {"tool": "kinetic_energy", "args": {"mass_kg": 2.0, "velocity_m_s": 3.0}}]}


def test_tool_args_wins_if_both_are_present():
    c = P.canonical({"tool": "t", "args": {"a": 1}, "tool_args": {"a": 2}})
    assert c["steps"][0]["args"] == {"a": 2}


def test_decline_clarification_and_garbage():
    assert P.canonical({"decline": "no"}) == {"kind": "decline", "reason": "no"}
    clar = P.canonical({"tool": "request_clarification",
                        "tool_args": {"variable": "x", "question": "q"}})
    assert clar["steps"][0]["tool"] == "request_clarification"
    for bad in (None, [], "x", {}, {"tool": 3}, {"steps": []}, {"steps": "x"},
                {"steps": [{"tool": "a"}] * 4}, {"steps": [{"nope": 1}]}):
        assert P.canonical(bad) is None, bad


def test_make_plan_and_v1_conversion_round_trip():
    one = [{"tool": "a", "args": {"x": 1}}]
    two = one + [{"tool": "b", "args": {"y": "$1"}}]
    assert P.make_plan(one) == {"tool": "a", "tool_args": {"x": 1}}
    assert P.make_plan(two) == {"steps": [{"tool": "a", "tool_args": {"x": 1}},
                                          {"tool": "b", "tool_args": {"y": "$1"}}]}
    assert P.canonical(P.make_plan(two))["steps"] == two
    assert P.to_v2({"tool": "a", "args": {"x": 1}}) == {"tool": "a", "tool_args": {"x": 1}}
    assert P.to_v2({"decline": "r"}) == {"decline": "r"}
    assert P.to_v2({"tool": "request_clarification", "args": {"variable": "v", "question": "q"}}) \
        == {"tool": "request_clarification", "tool_args": {"variable": "v", "question": "q"}}


def test_references_resolve_to_earlier_steps_only():
    assert P.refs_in({"a": "$2", "b": 5, "c": "x$1", "d": "$1"}) == [2, 1]
    assert P.resolve_args({"v": "$1", "m": 4.0, "name": "mars"}, [12.5]) == \
        {"v": 12.5, "m": 4.0, "name": "mars"}
    with pytest.raises(P.PlanError):
        P.resolve_args({"v": "$2"}, [1.0])           # has not run
    with pytest.raises(P.PlanError):
        P.resolve_args({"v": "$1"}, ["text"])         # not a number
    with pytest.raises(P.PlanError):
        P.resolve_args({"v": "$1"}, [True])           # a bool is not a number here
    P.check_refs([{"tool": "a", "args": {}}, {"tool": "b", "args": {"v": "$1"}}])
    for bad in ([{"tool": "a", "args": {"v": "$1"}}],                       # refers to itself
                [{"tool": "a", "args": {"v": "$2"}}, {"tool": "b", "args": {}}]):  # forward
        with pytest.raises(P.PlanError):
            P.check_refs(bad)


def test_the_schema_keeps_the_training_order_and_the_tool_enum():
    """The v1 failure: an alphabetical schema put `args` before `tool`. Here every
    object's properties are already alphabetical, and `tool` < `tool_args`."""
    names = ["b_tool", "a_tool", "a_tool", "request_clarification"]
    fmt = P.router_format(names)
    assert P.keys_in_training_order(fmt)
    call, steps, decline = fmt["anyOf"]
    assert list(call["properties"]) == ["tool", "tool_args"]
    assert call["properties"]["tool"]["enum"] == ["a_tool", "b_tool", "request_clarification"]
    assert [list(a["properties"])[0] for a in fmt["anyOf"]] == ["tool", "steps", "decline"]
    assert steps["properties"]["steps"]["items"] == call
    assert (steps["properties"]["steps"]["minItems"], steps["properties"]["steps"]["maxItems"]) == (2, 3)
    assert len(P.router_format(names, steps=False)["anyOf"]) == 2
    # the v1 schema, for contrast, is out of order
    v1 = {"anyOf": [{"type": "object", "properties": {"tool": {}, "args": {}}}]}
    assert not P.keys_in_training_order(v1)


def test_the_prompt_describes_every_shape_the_schema_allows():
    for needle in ('"tool_args"', '"steps"', '"$1"', "request_clarification", '"decline"'):
        assert needle in P.ROUTER_SYSTEM_PROMPT_V2
    assert '"args"' not in P.ROUTER_SYSTEM_PROMPT_V2.replace('"tool_args"', "")


def test_exponents_are_written_without_a_leading_zero_and_mean_the_same():
    plan = {"tool": "t", "tool_args": {"a": 2.42e-06, "b": 6.53e-09, "c": 1.9e+30, "d": 5.5e-10,
                                       "e": 1e-05, "f": 0.0728, "g": 12, "h": -3.2e-07}}
    text = P.dumps_plan(plan)
    assert text == ('{"tool": "t", "tool_args": {"a": 2.42e-6, "b": 6.53e-9, "c": 1.9e+30, '
                    '"d": 5.5e-10, "e": 1e-5, "f": 0.0728, "g": 12, "h": -3.2e-7}}')
    import json
    assert json.loads(text) == plan                                     # same numbers exactly
    assert not re.search(r"[eE][-+]?0\d", text)


def test_strings_that_look_like_numbers_are_left_alone():
    plan = {"tool": "simplify_expr", "tool_args": {"expression": "x*1e-07 + 2.5e-06*y"}}
    assert P.dumps_plan(plan).count("1e-07") == 1 and "2.5e-06" in P.dumps_plan(plan)
    assert json.loads(P.dumps_plan(plan)) == plan


def test_a_zero_exponent_stays_a_zero():
    assert json.loads(P.dumps_plan({"x": 1.5e+00})) == {"x": 1.5}
    assert P.dumps_plan({"x": 1e-00}) == '{"x": 1.0}'
